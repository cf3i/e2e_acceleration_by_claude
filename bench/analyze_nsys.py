"""Summarize `nsys stats` CSVs: top kernels, NVTX ranges, CUDA API calls.

    python bench/analyze_nsys.py results/<stage>/profile
"""
import collections
import csv
import glob
import sqlite3
import sys
from pathlib import Path


def read(pdir, report):
    files = glob.glob(str(Path(pdir) / f"*_{report}.csv"))
    if not files:
        return []
    with open(files[0]) as f:
        return list(csv.DictReader(f))


def show(rows, title, name_key, n=20):
    if not rows:
        print(f"\n## {title}: (no data)")
        return
    ck = next(k for k in ("Instances", "Num Calls", "Count") if k in rows[0])  # column name differs per report
    total = sum(float(r["Total Time (ns)"]) for r in rows)
    count = sum(int(r[ck]) for r in rows)
    print(f"\n## {title}: total {total / 1e9:.2f}s over {count} instances")
    print(f"{'time%':>6} {'total_s':>9} {'count':>9} {'avg_us':>9}  name")
    for r in rows[:n]:
        print(f"{float(r['Time (%)']):6.1f} {float(r['Total Time (ns)']) / 1e9:9.3f} {int(r[ck]):9d} "
              f"{float(r['Avg (ns)']) / 1e3:9.1f}  {r[name_key][:110]}")


def category(name: str) -> str:
    n = name.lower()
    if "nvjet" in n or "gemm" in n or "splitkreduce" in n:
        return "GEMM"
    if "fmha" in n or "flash" in n or "attention" in n:
        return "attention"
    if any(k in n for k in ("chunk", "recurrent", "layer_norm", "norm_gate", "conv1d", "l2norm", "solve_tril", "wy_")):
        return "fla/conv1d"
    if "triton" in n:
        return "triton (inductor)"
    if "reduce" in n:
        return "torch reduce"
    return "torch elementwise/copy/other"


def graph_steps(pdir):
    """Per-replay breakdown of kernels launched from CUDA graphs (needs nsys --cuda-graph-trace=node)."""
    db = Path(pdir) / "nsys.sqlite"
    if not db.exists():
        return
    c = sqlite3.connect(db)
    rows = c.execute("""select s.value, count(*), sum(k.end - k.start) from CUPTI_ACTIVITY_KIND_KERNEL k
                        join StringIds s on s.id = k.shortName where k.graphNodeId is not null
                        group by s.value order by 3 desc""").fetchall()
    steps = c.execute("""select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
                         where s.value like 'cudaGraphLaunch%'""").fetchone()[0]
    if not rows or not steps:
        return
    tot, n = sum(r[2] for r in rows), sum(r[1] for r in rows)
    print(f"\n## CUDA-graph kernels: {steps} replays, {n / steps:.0f} kernels/replay, "
          f"{tot / steps / 1e6:.2f} ms GPU time/replay")
    agg = collections.defaultdict(lambda: [0, 0])
    for name, k, t in rows:
        a = agg[category(name)]
        a[0] += t
        a[1] += k
    for cat, (t, k) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"  {cat:30s} {100 * t / tot:5.1f}%  {t / steps / 1e6:6.2f} ms/replay  {k / steps:6.0f} kernels/replay  "
              f"avg {t / k / 1e3:6.1f} us")
    print(f"{'time%':>6} {'us/replay':>10} {'n/replay':>9} {'avg_us':>8}  name")
    for name, k, t in rows[:15]:
        print(f"{100 * t / tot:6.1f} {t / steps / 1e3:10.1f} {k / steps:9.0f} {t / k / 1e3:8.1f}  {name[:90]}")


def main():
    pdir = sys.argv[1]
    show(read(pdir, "nvtx_sum"), "NVTX ranges (L1 sections)", "Range")
    show(read(pdir, "cuda_gpu_kern_sum"), "GPU kernels", "Name", n=25)
    show(read(pdir, "cuda_api_sum"), "CUDA API (host side)", "Name", n=12)
    show(read(pdir, "cuda_gpu_mem_time_sum"), "GPU memcpy/memset", "Operation", n=8)
    graph_steps(pdir)


if __name__ == "__main__":
    main()
