"""L2 profile with torch.profiler.

1) Aggregate op tables over the first K tasks (CPU-time and CUDA-time views) + GPU busy ratio.
2) A short chrome trace (one generate call, few tokens) for visual inspection in Perfetto.

    python bench/profile_torch.py --stage stage00_baseline --engine naive --runner serial --tasks 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentopt.agent import AgentState  # noqa: E402
from agentopt.engines import make_engine  # noqa: E402
from agentopt.runner import RUNNERS  # noqa: E402
from agentopt.timing import Timer  # noqa: E402
from bench.run_bench import load_tasks  # noqa: E402


def kernel_stats(prof):
    """GPU busy time = union of kernel intervals (handles overlap); also kernel count."""
    ivs = sorted(
        (e.time_range.start, e.time_range.end) for e in prof.events() if e.device_type == DeviceType.CUDA
    )
    busy, cur_s, cur_e = 0, None, None
    for s, e in ivs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy / 1e6, len(ivs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--engine", default="naive")
    ap.add_argument("--runner", default="serial")
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--trace-tokens", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    out = ROOT / "results" / args.stage / "profile"
    out.mkdir(parents=True, exist_ok=True)

    engine = make_engine(args.engine)
    runner = RUNNERS[args.runner]
    tasks = load_tasks(args.tasks + 1)
    runner(engine, tasks[:1], Timer(), args.max_new_tokens)  # warmup

    # 1) aggregate
    timer = Timer()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        _, turns = runner(engine, tasks[1:], timer, args.max_new_tokens)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    busy_s, n_kernels = kernel_stats(prof)
    gen_tokens = sum(t["gen_len"] for t in turns)
    ka = prof.key_averages()
    (out / "top_ops_by_cuda.txt").write_text(ka.table(sort_by="self_cuda_time_total", row_limit=40))
    (out / "top_ops_by_cpu.txt").write_text(ka.table(sort_by="self_cpu_time_total", row_limit=40))
    summary = {
        "profiled_tasks": args.tasks,
        "wall_s_with_profiler": wall,
        "gpu_kernel_busy_s": busy_s,
        "gpu_busy_ratio": busy_s / wall,
        "kernels": n_kernels,
        "gen_tokens": gen_tokens,
        "kernels_per_generated_token": n_kernels / max(1, gen_tokens),
        "breakdown_with_profiler": timer.breakdown(wall),
    }
    (out / "torch_profile_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(ka.table(sort_by="self_cuda_time_total", row_limit=25))
    print(ka.table(sort_by="self_cpu_time_total", row_limit=25))

    # 2) short trace of one generate call on a real first-turn prompt
    st = AgentState(tasks[1])
    ids = engine.tok.encode(st.prompt_text(engine.tok), add_special_tokens=False)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=False) as prof2:
        engine.generate(ids, args.trace_tokens, Timer())
    prof2.export_chrome_trace(str(out / "torch_trace_generate.json"))


if __name__ == "__main__":
    main()
