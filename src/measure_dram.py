"""
DRAM Data Movement Measurement - LIVE ONLY. No hardcoded values.
Uses nsys GPU metrics with torch.cuda.profiler markers.

Usage:
  python3 measure_dram.py  # Runs nsys for all three engines
  
Output: dram_results.csv with live-measured MB/query
"""
import os, sys, time, subprocess, csv, sqlite3, glob
from collections import defaultdict

BASE_DIR = "/home/min/a/cashman3/RTRAG/src"
PEAK_BW_GBs = 960.0  # RTX 6000 Ada
NUM_QUERIES = 20

sys.path.insert(0, BASE_DIR)


def run_nsys(engine):
    """Run dram_profile_worker.py under nsys for one engine."""
    worker = os.path.join(BASE_DIR, "dram_profile_worker.py")
    if not os.path.exists(worker):
        print(f"FATAL: {worker} not found. Copy it to Triton first.", flush=True)
        sys.exit(1)
    
    out_prefix = f"/tmp/dram_{engine}"
    # Clean old files
    for f in glob.glob(f"{out_prefix}*"):
        try:
            os.remove(f)
        except:
            pass
    
    cmd = [
        "nsys", "profile",
        "--gpu-metrics-device=all",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "-o", out_prefix,
        "--force-overwrite=true",
        sys.executable, worker, engine,
    ]
    print(f"\n{'='*70}", flush=True)
    print(f"Profiling {engine} with nsys ({NUM_QUERIES} queries)...", flush=True)
    print(f"{'='*70}", flush=True)
    
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=1200)
    
    rep_path = f"{out_prefix}.nsys-rep"
    if not os.path.exists(rep_path):
        print(f"FATAL: nsys failed for {engine}", flush=True)
        print(f"STDERR: {result.stderr[-1000:]}", flush=True)
        sys.exit(1)
    
    print(f"  Report: {rep_path}", flush=True)
    return rep_path






def parse_dram_bytes(rep_path):
    print(f"  Parsing {rep_path}...", flush=True)
    cmd = ["nsys", "stats", "--report=gpumetrics", rep_path,
           "--force-export=true", "--force-overwrite=true"]
    subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=300)
    
    sqlite_path = rep_path.replace('.nsys-rep', '.sqlite')
    if not os.path.exists(sqlite_path):
        print(f"FATAL: SQLite not generated for {rep_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    
    # Query metric table by name on Ada GPU
    cur.execute("""
        SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS 
        WHERE metricName LIKE '%dram__bytes_read%' 
           OR metricName LIKE '%dram__bytes_write%'
           OR metricName LIKE '%DRAM%Read%' 
           OR metricName LIKE '%DRAM%Write%'
           OR metricName LIKE '%dram__throughput%'
    """)
    rows = cur.fetchall()
    print(f"  Matched DRAM metrics: {rows}", flush=True)
    
    dram_ids = [r[0] for r in rows]

    dram_ids = list(set(dram_ids))
    print(f"  Using unique DRAM metric IDs: {dram_ids}", flush=True)

    if not dram_ids:
        # Fallback to any metric containing DRAM
        cur.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS WHERE metricName LIKE '%dram%'")
        dram_ids = [r[0] for r in cur.fetchall()]

    if not dram_ids:
        print("FATAL: No DRAM metric IDs found")
        sys.exit(1)

    placeholders = ','.join('?' * len(dram_ids))
    cur.execute(f"""
        SELECT timestamp, value FROM GPU_METRICS 
        WHERE metricId IN ({placeholders})
        ORDER BY timestamp
    """, dram_ids)
    metric_rows = cur.fetchall()
    conn.close()

    ts_values = defaultdict(float)
    for ts, val in metric_rows:
        ts_values[ts] += float(val or 0)
    
    total_bytes = 0.0
    sorted_ts = sorted(ts_values.keys())
    for i in range(1, len(sorted_ts)):
        dt_sec = (sorted_ts[i] - sorted_ts[i-1]) / 1e9
        if 0 < dt_sec < 10:
            bw_pct = ts_values[sorted_ts[i]]
            total_bytes += (bw_pct / 100.0) * (PEAK_BW_GBs * 1e9) * dt_sec
    
    mb_per_query = (total_bytes / 1e6) / NUM_QUERIES
    print(f"  Total: {total_bytes/1e6:.1f} MB, Per query: {mb_per_query:.1f} MB", flush=True)
    return mb_per_query




def main():
    print("=" * 70)
    print("DRAM MEASUREMENT - LIVE NSYS PROFILING")
    print("=" * 70)
    
    results = {}
    for engine in ['plaid', 'plaid_tms', 'ragrt']:
        rep_path = run_nsys(engine)
        mb_per_query = parse_dram_bytes(rep_path)
        results[engine] = mb_per_query
        print(f"\n  {engine}: {mb_per_query:.1f} MB/query", flush=True)
    
    # Save
    with open('dram_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['engine', 'mb_per_query'])
        for k, v in results.items():
            w.writerow([k, f"{v:.1f}"])
    
    print("\n" + "=" * 70)
    print("RESULTS (live-measured)")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v:.1f} MB/query")
    
    if 'plaid' in results and 'ragrt' in results:
        print(f"\n  RAGRT vs PLAID: {results['plaid']/results['ragrt']:.2f}x less data")
    if 'plaid_tms' in results and 'ragrt' in results:
        ratio = results['plaid_tms']/results['ragrt']
        print(f"  RAGRT vs PLAID+TMS: {ratio:.2f}x less data")


if __name__ == '__main__':
    main()
