"""
Build passage_id -> domain_id map for LoTTE.
Reads the per-domain collection.tsv files and emits:
  - lotte_domain_map.json   : {global_pid: domain_id}
  - lotte_domain_stats.json : per-domain counts and offsets
"""
import os
import json
import sys

DOMAINS = ["writing", "recreation", "science", "technology", "lifestyle"]
DOMAIN_TO_ID = {d: i for i, d in enumerate(DOMAINS)}   # 0..4

LOTTE_ROOT = os.environ.get("LOTTE_ROOT", "../data/lotte")
OUT_DIR    = os.environ.get("OUT_DIR", ".")

def build_map():
    pid_to_domain = {}
    stats = {}
    offset = 0

    for domain in DOMAINS:
        coll_path = os.path.join(LOTTE_ROOT, domain, "dev", "collection.tsv")
        if not os.path.exists(coll_path):
            print(f"ERROR: missing {coll_path}", file=sys.stderr)
            sys.exit(1)

        count = 0
        with open(coll_path) as f:
            for line in f:
                local_pid_str, _ = line.split("\t", 1)
                local_pid = int(local_pid_str)
                global_pid = offset + local_pid
                pid_to_domain[global_pid] = DOMAIN_TO_ID[domain]
                count += 1

        stats[domain] = {
            "domain_id": DOMAIN_TO_ID[domain],
            "offset":    offset,
            "count":     count,
            "end_pid":   offset + count - 1,
        }
        print(f"{domain:>12}  offset={offset:>8}  count={count:>8}  "
              f"pids=[{offset}, {offset + count - 1}]")
        offset += count

    return pid_to_domain, stats


if __name__ == "__main__":
    print(f"LOTTE_ROOT = {os.path.abspath(LOTTE_ROOT)}")
    pid_to_domain, stats = build_map()

    out_map   = os.path.join(OUT_DIR, "lotte_domain_map.json")
    out_stats = os.path.join(OUT_DIR, "lotte_domain_stats.json")
    with open(out_map,   "w") as f: json.dump(pid_to_domain, f)   # str keys for JSON
    with open(out_stats, "w") as f: json.dump(stats, f, indent=2)

    print(f"\nTotal passages: {len(pid_to_domain)}")
    print(f"Wrote {out_map}")
    print(f"Wrote {out_stats}")
    print("\nSample entries:")
    for pid in [0, 1000, stats["technology"]["offset"] + 5]:
        print(f"  pid={pid} -> domain_id={pid_to_domain[pid]} "
              f"({DOMAINS[pid_to_domain[pid]]})")
