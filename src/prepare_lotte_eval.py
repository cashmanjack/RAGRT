"""
Build the LoTTE evaluation set for the unified (pooled dev) index: every domain,
search and forum queries, with answers mapped into unified pid space.

The unified collection was assembled from the five per-domain dev collections, but
the order and offsets were never recorded (build_domain_predicates.py hard-coded
boundaries that only match for science). This script finds each domain's offset by
matching passage text, verifies every passage, and refuses to guess.

Inputs (official LoTTE release, https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz):
  <lotte_root>/<domain>/dev/collection.tsv
  <lotte_root>/<domain>/dev/questions.{search,forum}.tsv
  <lotte_root>/<domain>/dev/qas.{search,forum}.jsonl      {"qid", "query", "answer_pids"}

Outputs in --outdir:
  questions.tsv   qid<TAB>text            qid = "<domain>.<search|forum>.<local qid>"
  qrels.tsv       qid<TAB>0<TAB>pid<TAB>1 pid in unified space (one line per answer)
  queries.json    {qid: {"domain", "qtype"}}
  domains.json    {domain: {"offset", "count"}} plus "num_passages"

Usage:
  python3 prepare_lotte_eval.py --lotte_root /local/scratch/a/cashman3/lotte \
      --unified_collection /local/scratch/a/cashman3/lotte/unified/collection.tsv \
      --outdir /local/scratch/a/cashman3/lotte/eval_pooled_dev
"""
import os, sys, json, argparse, hashlib
from collections import Counter
import numpy as np

DOMAINS = ["writing", "recreation", "science", "technology", "lifestyle"]
QTYPES = ["search", "forum"]


def text_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def load_collection_hashes(path, check_pids=True):
    """uint64 hash of each passage text, in file order. Optionally asserts pid == line index."""
    hashes = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            pid, _, text = line.rstrip("\n").partition("\t")
            if check_pids and pid.strip() != str(i):
                raise ValueError(f"{path}: line {i} has pid {pid!r}; expected pids 0..n-1 in order")
            hashes.append(text_hash(text))
    return np.asarray(hashes, dtype=np.uint64)


def find_offset(unified, domain_hashes):
    """
    Offset o with unified[o:o+n] == domain_hashes, or None. Unique by construction:
    if several offsets match, the collections share an identical run of passages and
    we refuse to choose.
    """
    n = len(domain_hashes)
    starts = np.flatnonzero(unified == domain_hashes[0])
    matches = [int(o) for o in starts
               if o + n <= len(unified) and np.array_equal(unified[o:o + n], domain_hashes)]
    if len(matches) > 1:
        raise ValueError(f"domain collection matches at several offsets {matches[:5]}")
    return matches[0] if matches else None


def check_layout(offsets, counts, total):
    """Domains must tile [0, total) exactly, without gaps or overlaps."""
    spans = sorted((offsets[d], offsets[d] + counts[d], d) for d in offsets)
    pos = 0
    for a, b, d in spans:
        if a != pos:
            raise ValueError(f"gap or overlap before {d}: expected offset {pos:,}, found {a:,}")
        pos = b
    if pos != total:
        raise ValueError(f"domains cover {pos:,} passages but the unified collection has {total:,}")


def load_questions(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, text = line.rstrip("\n").partition("\t")
            out[qid.strip()] = text
    return out


def load_qas(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["qid"])] = [int(p) for p in r["answer_pids"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotte_root", required=True)
    ap.add_argument("--unified_collection", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Hashing unified collection {args.unified_collection} ...", flush=True)
    unified = load_collection_hashes(args.unified_collection, check_pids=True)
    total = len(unified)
    print(f"  {total:,} passages")

    offsets, counts = {}, {}
    for d in DOMAINS:
        path = os.path.join(args.lotte_root, d, args.split, "collection.tsv")
        if not os.path.exists(path):
            sys.exit(f"FATAL: {path} missing. Download and extract lotte.tar.gz into --lotte_root.")
        h = load_collection_hashes(path)
        o = find_offset(unified, h)
        if o is None:
            sys.exit(f"FATAL: {d} ({len(h):,} passages) is not a contiguous block of the unified "
                     f"collection. Refusing to map pids by guesswork.")
        offsets[d], counts[d] = o, len(h)
        print(f"  {d:<11} offset {o:>9,}  count {len(h):>9,}")
    check_layout(offsets, counts, total)

    questions, qrels, meta = {}, [], {}
    per_set = Counter()
    for d in DOMAINS:
        for t in QTYPES:
            qpath = os.path.join(args.lotte_root, d, args.split, f"questions.{t}.tsv")
            apath = os.path.join(args.lotte_root, d, args.split, f"qas.{t}.jsonl")
            if not (os.path.exists(qpath) and os.path.exists(apath)):
                sys.exit(f"FATAL: missing {qpath} or {apath}")
            qs, qas = load_questions(qpath), load_qas(apath)
            for local_qid, text in qs.items():
                answers = qas.get(local_qid, [])
                if not answers:
                    continue                       # unanswerable queries cannot be scored
                if max(answers) >= counts[d] or min(answers) < 0:
                    sys.exit(f"FATAL: {d}.{t} qid {local_qid} answer pid outside the domain collection")
                qid = f"{d}.{t}.{local_qid}"
                questions[qid] = text
                meta[qid] = {"domain": d, "qtype": t}
                qrels.extend((qid, offsets[d] + p) for p in answers)
                per_set[(d, t)] += 1

    with open(os.path.join(args.outdir, "questions.tsv"), "w", encoding="utf-8") as f:
        for qid, text in questions.items():
            f.write(f"{qid}\t{text}\n")
    with open(os.path.join(args.outdir, "qrels.tsv"), "w") as f:
        for qid, pid in qrels:
            f.write(f"{qid}\t0\t{pid}\t1\n")
    json.dump(meta, open(os.path.join(args.outdir, "queries.json"), "w"))
    json.dump({"num_passages": total,
               "domains": {d: {"offset": offsets[d], "count": counts[d]} for d in DOMAINS}},
              open(os.path.join(args.outdir, "domains.json"), "w"), indent=1)

    print(f"\n{len(questions):,} queries with answers, {len(qrels):,} answer pairs")
    for (d, t), n in sorted(per_set.items()):
        print(f"  {d:<11} {t:<6} {n:>6,}")
    print(f"Wrote questions.tsv, qrels.tsv, queries.json, domains.json to {args.outdir}")


if __name__ == "__main__":
    main()
