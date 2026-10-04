"""Collect every results/<stage>/ into results/summary.csv and print a comparison table.

    python bench/report.py              # all stages
    python bench/report.py <stage>      # also print that stage's Amdahl breakdown
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BASE = "stage00_baseline"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def micro_at(micro, ctx, key):
    if not micro:
        return None
    for r in micro["rows"]:
        if r["ctx"] == ctx:
            return r[key]
    return None


def stage_row(d: Path) -> dict:
    m, micro = load(d / "metrics.json"), load(d / "micro.json")
    prof = load(d / "profile" / "torch_profile_summary.json")
    phase = "p1" if d.name.startswith("p1_") else "p2" if d.name.startswith("p2_") else "-"
    return {
        "stage": d.name,
        "phase": phase,
        "makespan_s": m and round(m["makespan_s"], 1),
        "accuracy": m and round(m["accuracy"], 4),
        "gen_tok_per_s_e2e": m and round(m["gen_tok_per_s_e2e"], 1),
        "e2e_decode_ms_per_tok": m and round(m["decode_ms_per_token"], 2),
        "micro_decode_ms_warm_1024": micro and round(micro_at(micro, 1024, "decode_ms_per_tok_warm"), 2),
        "micro_decode_ms_cold_1024": micro and round(micro_at(micro, 1024, "decode_ms_per_tok_cold"), 2),
        "micro_prefill_ms_warm_1024": micro and round(micro_at(micro, 1024, "prefill_ms_warm"), 1),
        "roofline_frac_1024": micro and round(micro_at(micro, 1024, "decode_bw_roofline_frac"), 3),
        "gpu_busy_ratio": prof and round(prof["gpu_busy_ratio"], 3),
        "kernels_per_token": prof and round(prof["kernels_per_generated_token"], 1),
    }


def main():
    dirs = sorted(d for d in RES.iterdir() if d.is_dir() and not d.name.startswith("_"))
    rows = [stage_row(d) for d in dirs]
    base = next((r for r in rows if r["stage"] == BASE), None)
    for r in rows:
        r["e2e_speedup"] = (round(base["makespan_s"] / r["makespan_s"], 2)
                            if base and base["makespan_s"] and r["makespan_s"] else None)
        r["micro_decode_speedup"] = (round(base["micro_decode_ms_warm_1024"] / r["micro_decode_ms_warm_1024"], 2)
                                     if base and base["micro_decode_ms_warm_1024"] and r["micro_decode_ms_warm_1024"]
                                     else None)
    if rows:
        with open(RES / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    cols = ["stage", "makespan_s", "e2e_speedup", "accuracy", "micro_decode_ms_warm_1024",
            "micro_decode_ms_cold_1024", "micro_decode_speedup", "roofline_frac_1024", "gpu_busy_ratio"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join("-" if r[c] is None else str(r[c]) for c in cols))

    if len(sys.argv) > 1:
        bd = load(RES / sys.argv[1] / "breakdown.json")
        if bd:
            print(f"\nAmdahl breakdown of {sys.argv[1]} (e2e):")
            for k, v in bd.items():
                print(f"  {k:12s} {v['seconds']:9.1f}s {100 * v['frac']:6.1f}%  "
                      f"max overall speedup if removed: {1 / max(1e-9, 1 - v['frac']):.2f}x")


if __name__ == "__main__":
    main()
