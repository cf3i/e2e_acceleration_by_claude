"""Schedulers that drive AgentStates through an engine.

run_serial     -- one task at a time, one turn at a time (phase 1; any engine with generate()).
run_continuous -- continuous batching (phase 2; BatchEngine): up to S agents in flight, one slot each. Every
                  loop iteration admits new tasks into free slots (prefill), runs ONE decode step for all slots,
                  and finishes turns whose row emitted a stop token (detokenize, tool call, next-turn prefill).
                  With forced-token prefill a turn starts without a first token; it arrives from a later step.
"""
import collections
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


def run_continuous(engine, tasks, timer, max_new_tokens: int = 512, max_turns: int = 6):
    tok, stop_ids = engine.tok, engine.stop_ids
    queue, free = collections.deque(tasks), list(range(engine.num_slots))[::-1]
    engine.reset_stats()
    if getattr(engine, "prefix_reuse", False) and len(tasks) >= 2:
        # Shared prefix = what all first-turn prompts have in common (system prompt + tool schema + chat header).
        firsts = [tok.encode(AgentState(t).prompt_text(tok), add_special_tokens=False) for t in tasks[:4]]
        n = next((i for i, xs in enumerate(zip(*firsts)) if len(set(xs)) > 1), min(map(len, firsts)) - 1)
        with timer("prefill"):
            engine.set_shared_prefix(firsts[0][:n])
    live = {}  # slot -> per-turn state of the agent occupying it
    results, turns = [], []

    def start_turn(slot, st, t_task):
        with timer("template"):
            ids = tok.encode(st.prompt_text(tok), add_special_tokens=False)
        with timer("prefill") as t_pre:
            first = engine.submit(slot, ids)
        live[slot] = {"st": st, "t_task": t_task, "ids": ids, "gen": [] if first is None else [first],
                      "prefill_s": t_pre.elapsed, "decode_s": 0.0,
                      "limit": min(max_new_tokens, engine.room(len(ids)))}
        if first is not None and (first in stop_ids or live[slot]["limit"] <= 1):
            finish_turn(slot)

    def finish_turn(slot):
        run = live.pop(slot)
        st, gen = run["st"], run["gen"]
        stopped = gen[-1] in stop_ids
        if stopped:
            gen = gen[:-1]
        with timer("detokenize"):
            text = tok.decode(gen, skip_special_tokens=False)
        call = st.observe(text)
        tool_s = 0.0
        if call is not None:
            with timer("tool") as t_tool:
                result = execute(*call)
            tool_s = t_tool.elapsed
            st.add_tool_result(result)
        turns.append({"task_id": st.task["id"], "turn": st.turns, "prompt_len": len(run["ids"]),
                      "gen_len": len(gen), "prefill_s": run["prefill_s"], "decode_s": run["decode_s"],
                      "tool_s": tool_s, "stopped": stopped})
        if st.done:
            engine.release(slot)
            free.append(slot)
            results.append({
                "task_id": st.task["id"], "gold": st.task["answer"], "pred": st.answer, "correct": st.correct,
                "turns": st.turns, "tool_calls": st.tool_calls, "finish_reason": st.finish_reason,
                "latency_s": time.perf_counter() - run["t_task"], "messages": st.messages,
            })
        else:
            start_turn(slot, st, run["t_task"])

    while queue or live:
        while free and queue:
            start_turn(free.pop(), AgentState(queue.popleft(), max_turns=max_turns), time.perf_counter())
        if not live:
            continue
        with timer("decode") as t_dec:
            out = engine.step()
        share = t_dec.elapsed / len(live)  # amortized decode time per live sequence
        for slot in list(live):
            run = live[slot]
            run["decode_s"] += share
            if slot not in out:  # still consuming its forced suffix
                continue
            run["gen"].append(out[slot])
            if out[slot] in stop_ids or len(run["gen"]) >= run["limit"]:
                finish_turn(slot)
    return results, turns


RUNNERS = {"serial": run_serial, "continuous": run_continuous}
