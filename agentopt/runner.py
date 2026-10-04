"""Schedulers that drive AgentStates through an engine.

run_serial     -- one task at a time, one turn at a time (phase 1; any engine with generate()).
run_continuous -- continuous batching (phase 2; BatchEngine): up to S agents in flight, one slot each. Every
                  loop iteration admits new tasks into free slots (prefill), runs ONE decode step for all slots,
                  and finishes turns whose row emitted a stop token (detokenize, tool call, next-turn prefill).
                  With forced-token prefill a turn starts without a first token; it arrives from a later step.
continuous_async -- same, but tool calls run in a thread pool: the agent's slot is suspended (state saved) while
                  its tool runs and the other rows keep decoding; finished tools are picked up between steps.
"""
import collections
import concurrent.futures as cf
import functools
import os
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


def _timed_execute(call):
    t0 = time.perf_counter()
    result = execute(*call)
    return result, time.perf_counter() - t0


def run_continuous(engine, tasks, timer, max_new_tokens: int = 512, max_turns: int = 6,
                   async_tools: bool = False, tool_workers: int = 8):
    tok, stop_ids = engine.tok, engine.stop_ids
    pool = cf.ThreadPoolExecutor(tool_workers) if async_tools else None
    pending = {}  # future -> (slot, agent state, t_task, turn record) for tools in flight
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
        rec = {"task_id": st.task["id"], "turn": st.turns, "prompt_len": len(run["ids"]),
               "gen_len": len(gen), "prefill_s": run["prefill_s"], "decode_s": run["decode_s"],
               "tool_s": 0.0, "stopped": stopped}
        turns.append(rec)
        if call is not None and pool is not None:
            engine.suspend(slot)
            pending[pool.submit(_timed_execute, call)] = (slot, st, run["t_task"], rec)
            return
        if call is not None:
            with timer("tool") as t_tool:
                result = execute(*call)
            rec["tool_s"] = t_tool.elapsed
            st.add_tool_result(result)
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

    def collect(futures):
        for fut in futures:
            slot, st, t_task, rec = pending.pop(fut)
            result, rec["tool_s"] = fut.result()
            st.add_tool_result(result)
            start_turn(slot, st, t_task)  # st cannot be done here: a tool call never ends the episode

    while queue or live or pending:
        if pending:
            collect([f for f in list(pending) if f.done()])
        while free and queue:
            start_turn(free.pop(), AgentState(queue.popleft(), max_turns=max_turns), time.perf_counter())
        if not live:
            if pending:
                with timer("tool_wait"):
                    done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                collect(done)
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
    if pool is not None:
        pool.shutdown()
    return results, turns


class PromptEncoder:
    """Tokenizes an agent's prompt incrementally: the rendered conversation of turn k+1 extends the one of turn k,
    so only the text after the last common special token is encoded (text is split at special tokens before BPE,
    so the tokens before such a cut are unchanged). AGENTOPT_CHECK_TOK=1 compares with a full encode."""

    def __init__(self, tok, sep="<|im_end|>"):
        self.tok, self.sep = tok, sep
        self.sep_id = tok.convert_tokens_to_ids(sep)
        self.check = os.environ.get("AGENTOPT_CHECK_TOK") == "1"
        self.cache = {}  # id(agent state) -> (text, ids)

    def __call__(self, st):
        text = st.prompt_text(self.tok)
        ids = None
        prev = self.cache.get(id(st))
        if prev is not None and prev[2] is st:
            ptext, pids, _ = prev
            p = ptext.rfind(self.sep)
            while p >= 0 and not text.startswith(ptext[:p + len(self.sep)]):
                p = ptext.rfind(self.sep, 0, p)
            if p >= 0:
                cut = p + len(self.sep)
                n = ptext.count(self.sep, 0, cut)
                k = [i for i, t in enumerate(pids) if t == self.sep_id][n - 1] + 1
                ids = pids[:k] + self.tok.encode(text[cut:], add_special_tokens=False)
        if ids is None:
            ids = self.tok.encode(text, add_special_tokens=False)
        if self.check:
            assert ids == self.tok.encode(text, add_special_tokens=False), "incremental tokenization mismatch"
        self.cache[id(st)] = (text, ids, st)
        return ids

    def forget(self, st):
        self.cache.pop(id(st), None)


def run_v3(engine, tasks, timer, max_new_tokens: int = 512, max_turns: int = 6, tool_workers: int = 8):
    """Phase 3 (V3Engine): turns are submitted without blocking; every `step()` runs one mixed forward (decode
    rows + prefill chunks) and returns the tokens of the previous step plus the slots whose turn ended. Tool calls
    run in a thread pool. New tasks are admitted while a slot is free, the KV cache can take them and the prefill
    backlog is under two steps' token budget (more cannot be fed sooner; tokenizing prompts early would only delay
    the next step)."""
    tok = engine.tok
    encode = PromptEncoder(tok)
    # sections must not synchronize the device: the engine keeps one step in flight while the CPU works (a sync
    # at the end of every step() would leave the GPU idle during the next step's scheduling and the CPU work)
    timer.sync = False
    pool = cf.ThreadPoolExecutor(tool_workers)
    pending = {}  # future -> (slot, agent state, t_task, turn record)
    queue, free = collections.deque(tasks), list(range(engine.num_slots))[::-1]
    engine.reset_stats()
    if len(tasks) >= 2:
        firsts = [tok.encode(AgentState(t).prompt_text(tok), add_special_tokens=False) for t in tasks[:4]]
        n = next((i for i, xs in enumerate(zip(*firsts)) if len(set(xs)) > 1), min(map(len, firsts)) - 1)
        with timer("prefill"):
            engine.register_prefix(firsts[0][:n])
    live, results, turns = {}, [], []
    nxt = None  # tokenized first prompt of queue[0]

    def start_turn(slot, st, t_task, ids=None):
        if ids is None:
            with timer("template"):
                ids = encode(st)
        limit = min(max_new_tokens, engine.room(len(ids)))
        engine.submit(slot, ids, limit)
        live[slot] = {"st": st, "t_task": t_task, "ids": ids, "gen": [], "t0": time.perf_counter(), "decode_s": 0.0}

    def finish_turn(slot):
        run = live.pop(slot)
        st, gen = run["st"], run["gen"]
        stopped = bool(gen) and gen[-1] in engine.stop_ids
        if stopped:
            gen = gen[:-1]
        with timer("detokenize"):
            text = tok.decode(gen, skip_special_tokens=False)
        call = st.observe(text)
        rec = {"task_id": st.task["id"], "turn": st.turns, "prompt_len": len(run["ids"]), "gen_len": len(gen),
               "prefill_s": 0.0, "decode_s": run["decode_s"], "turn_s": time.perf_counter() - run["t0"],
               "tool_s": 0.0, "stopped": stopped}
        turns.append(rec)
        if call is not None:
            pending[pool.submit(_timed_execute, call)] = (slot, st, run["t_task"], rec)
            return
        engine.release(slot)
        free.append(slot)
        encode.forget(st)
        results.append({
            "task_id": st.task["id"], "gold": st.task["answer"], "pred": st.answer, "correct": st.correct,
            "turns": st.turns, "tool_calls": st.tool_calls, "finish_reason": st.finish_reason,
            "latency_s": time.perf_counter() - run["t_task"], "messages": st.messages,
        })

    def collect(futures):
        for fut in futures:
            slot, st, t_task, rec = pending.pop(fut)
            result, rec["tool_s"] = fut.result()
            st.add_tool_result(result)
            start_turn(slot, st, t_task)

    while queue or live or pending:
        if pending:
            collect([f for f in list(pending) if f.done()])
        while free and queue and engine.prefill_backlog() < 2 * engine.TB:
            if nxt is None:
                st = AgentState(queue[0], max_turns=max_turns)
                with timer("template"):
                    nxt = (st, encode(st))
            if not engine.can_admit(len(nxt[1]), max_new_tokens):
                break
            queue.popleft()
            start_turn(free.pop(), nxt[0], time.perf_counter(), nxt[1])
            nxt = None
        if not live:
            if pending:
                with timer("tool_wait"):
                    done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                collect(done)
            elif queue:
                raise RuntimeError("cannot admit a task into an empty engine (KV cache too small)")
            continue
        with timer("decode") as t_step:
            out, done = engine.step()
        share = t_step.elapsed / len(live)
        for slot, t in out.items():
            run = live.get(slot)
            if run is not None:
                run["gen"].append(t)
        for run in live.values():
            run["decode_s"] += share
        for slot in done:
            if slot in live:
                finish_turn(slot)
    engine.drain()
    pool.shutdown()
    return results, turns


RUNNERS = {"serial": run_serial, "v3": run_v3, "continuous": run_continuous,
           "continuous_async": functools.partial(run_continuous, async_tools=True)}
