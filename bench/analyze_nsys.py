"""Summarize `nsys stats` CSVs: top kernels, NVTX ranges, CUDA API calls.

    python bench/analyze_nsys.py results/<stage>/profile
"""
import csv
import glob
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


def main():
    pdir = sys.argv[1]
    show(read(pdir, "nvtx_sum"), "NVTX ranges (L1 sections)", "Range")
    show(read(pdir, "cuda_gpu_kern_sum"), "GPU kernels", "Name", n=25)
    show(read(pdir, "cuda_api_sum"), "CUDA API (host side)", "Name", n=12)
    show(read(pdir, "cuda_gpu_mem_time_sum"), "GPU memcpy/memset", "Operation", n=8)


if __name__ == "__main__":
    main()
