"""Schedulers that drive AgentStates through an engine.

Stage 00: run_serial -- one task at a time, one turn at a time.
"""
import time

from .agent import AgentState
from .tools import execute


def run_serial(engine, tasks, timer, max_new_tokens: int = 512, max_turns: int = 6):
    tok = engine.tok
    results, turns = [], []
    for task in tasks:
        st = AgentState(task, max_turns=max_turns)
        t_task = time.perf_counter()
        while not st.done:
            with timer("template"):
                ids = tok.encode(st.prompt_text(tok), add_special_tokens=False)
            gen, gstats = engine.generate(ids, max_new_tokens, timer)
            with timer("detokenize"):
                text = tok.decode(gen, skip_special_tokens=False)
            call = st.observe(text)
            tool_s = 0.0
            if call is not None:
                with timer("tool") as t_tool:
                    result = execute(*call)
                tool_s = t_tool.elapsed
                st.add_tool_result(result)
            turns.append({
                "task_id": task["id"], "turn": st.turns, "prompt_len": len(ids), "gen_len": len(gen),
                "prefill_s": gstats["prefill_s"], "decode_s": gstats["decode_s"], "tool_s": tool_s,
                "stopped": gstats["stopped"],
            })
        results.append({
            "task_id": task["id"], "gold": task["answer"], "pred": st.answer, "correct": st.correct,
            "turns": st.turns, "tool_calls": st.tool_calls, "finish_reason": st.finish_reason,
            "latency_s": time.perf_counter() - t_task, "messages": st.messages,
        })
    return results, turns


RUNNERS = {"serial": run_serial}
