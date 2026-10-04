"""One-off probe of the agent workload: tool-use rate / turns / accuracy (not a stage benchmark)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentopt.engines import make_engine
from agentopt.runner import run_serial
from agentopt.timing import Timer
from bench.run_bench import load_tasks

import agentopt.agent as A
VARIANTS = {
    "short": (
        "You are a math assistant that solves problems by writing Python code. "
        "First call the python tool with a short program (at most 15 lines) that computes the answer "
        "and prints only the final result. "
        "After you see the output, reply with the final answer as \\boxed{N} where N is a plain number "
        "(call the tool again if the output shows a mistake). Be concise."
    ),
    "short_comment": (
        "You are a math assistant that solves problems by writing Python code. "
        "First call the python tool with a short program that computes the answer step by step "
        "(one variable per quantity, with a brief comment) and ends with a single print() of the final result. "
        "After you see the output, reply with the final answer as \\boxed{N} where N is a plain number "
        "(call the tool again if the output shows a mistake). Be concise."
    ),
}
# argv: N [variant]
A.SYSTEM_PROMPT = VARIANTS[sys.argv[2]] if len(sys.argv) > 2 else A.SYSTEM_PROMPT
print("ARGS", sys.argv[1:], flush=True)
e = make_engine("naive")
tasks = load_tasks(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
t0 = time.perf_counter()
res, turns = run_serial(e, tasks, Timer())
print("decode ms/tok per turn:", [round(1000 * t["decode_s"] / max(1, t["gen_len"]), 1) for t in turns])
print(json.dumps({"acc": sum(r["correct"] for r in res) / len(res), "tool_calls": sum(r["tool_calls"] for r in res),
      "avg_turns": len(turns) / len(res), "gen_tokens": sum(t["gen_len"] for t in turns),
      "max_prompt": max(t["prompt_len"] for t in turns), "not_stopped": sum(not t["stopped"] for t in turns),
      "finish": {k: sum(r["finish_reason"] == k for r in res) for k in {r["finish_reason"] for r in res}},
      "time_s": round(time.perf_counter() - t0, 1)}))
for r in res[:3] + [r for r in res if not r["correct"]][:3]:
    print("=" * 30, r["task_id"], "gold", r["gold"], "pred", r["pred"])
    for m in r["messages"][2:]:
        print(f"[{m['role']}] {m['content'][:500]}")
