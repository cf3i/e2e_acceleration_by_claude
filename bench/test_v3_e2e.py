"""End-to-end checks of the v3 engine.

1. Teacher-forced top-1 agreement on reference first-turn answers (bf16 reference engine and p2_s10 transcripts):
   the v3 model's argmax at every answer position vs the reference token.
2. Greedy generation of first turns, several sessions at once: (a) CUDA graphs (all steps) + token budget 2048,
   (b) eager + token budget 61 (prompts split into many chunks, mixed with decode rows), (c) one session at a
   time. All three should give (near-)identical tokens; also compared with the p2_s10 transcript.

    python bench/test_v3_e2e.py [--n 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentopt.agent import AgentState  # noqa: E402
from agentopt.tools import TOOLS  # noqa: E402
from agentopt.v3.engine import V3Engine  # noqa: E402


def load(name):
    return {r["task_id"]: r for r in map(json.loads, open(ROOT / "results" / name / "transcripts.jsonl"))}


def generate(eng, prompts, max_new, one_at_a_time=False):
    outs = {i: [] for i in range(len(prompts))}
    groups = [[i] for i in range(len(prompts))] if one_at_a_time else [list(range(len(prompts)))]
    for g in groups:
        for i in g:
            eng.submit(i, prompts[i], max_new)
        live = set(g)
        while live:
            out, done = eng.step()
            for slot, t in out.items():
                outs[slot].append(t)
            live -= done
        eng.drain()
        for i in g:
            eng.release(i)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=96)
    args = ap.parse_args()
    t0 = time.time()
    eng = V3Engine(num_slots=16, buckets=(1, 2, 4, 8, 16))
    print(f"graphs: {sorted(eng.graphs)}")
    print(f"engine up in {time.time() - t0:.0f}s, KV blocks {eng.m.NB}", flush=True)
    tok = eng.tok
    tasks = [json.loads(line) for line in open(ROOT / "data/gsm8k_128.jsonl")][:args.n]
    refs = {"bf16": load("ref_ours_bf16_r3"), "p2_s10": load("ref_ours_p2s10_r3")}

    # 1. teacher-forced agreement on first-turn answers
    for name, ref in refs.items():
        agree = total = 0
        for t in tasks:
            msgs = ref[t["id"]]["messages"]
            p = tok.apply_chat_template(msgs[:2], tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
            full = tok.apply_chat_template(msgs[:3], tools=TOOLS, tokenize=False, enable_thinking=False)
            pi, fi = tok.encode(p, add_special_tokens=False), tok.encode(full, add_special_tokens=False)
            assert fi[:len(pi)] == pi
            ans = fi[len(pi):]
            ans = ans[:ans.index(tok.convert_tokens_to_ids("<|im_end|>")) + 1]
            pred = eng.teacher_force(fi[:len(pi) + len(ans)])
            pred = pred[len(pi) - 1:len(pi) - 1 + len(ans)]
            agree += sum(a == b for a, b in zip(pred, ans))
            total += len(ans)
        print(f"teacher-forced top-1 agreement vs {name}: {agree}/{total} = {100 * agree / total:.2f}%", flush=True)

    # 2. greedy generation consistency
    prompts = [tok.encode(AgentState(t).prompt_text(tok), add_special_tokens=False) for t in tasks]
    ga = generate(eng, prompts, args.max_new)
    graphs, eng.graphs, eng.TB = eng.graphs, {}, 61
    gb = generate(eng, prompts, args.max_new)
    eng.graphs, eng.TB = graphs, 2048
    gc = generate(eng, prompts, args.max_new, one_at_a_time=True)
    stats = dict(eng.stats)
    for i, t in enumerate(tasks):
        ref = tok.encode(refs["p2_s10"][t["id"]]["messages"][2]["content"], add_special_tokens=False)

        def lcp(x, y):
            return next((k for k, (a, b) in enumerate(zip(x, y)) if a != b), min(len(x), len(y)))
        print(f"task {t['id']:4d}: len {len(ga[i])}  lcp(graph,eager61)={lcp(ga[i], gb[i])} "
              f"lcp(graph,single)={lcp(ga[i], gc[i])} lcp(graph,p2_s10)={lcp(ga[i], ref)}/{len(ref)}")
    print(repr(tok.decode(ga[0])))
    print(stats)


if __name__ == "__main__":
    main()
