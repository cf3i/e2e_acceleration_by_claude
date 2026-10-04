"""Generality workload: the same 128 questions, each preceded by a long, task-specific block of solved
GSM8K-train examples (a few-shot / retrieved-context pattern).

- Prompt lengths are spread uniformly over 1k–7k tokens.
- Different tasks share nothing beyond the system prompt, so the shared-prefix and short-suffix shortcuts don't
  apply. Optimizations must hold for long, varied prompts.
- Deterministic (seed 0).

    python bench/make_longctx.py   # -> data/gsm8k_longctx_128.jsonl
"""
import json
import random
import re
from pathlib import Path

import datasets
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/mnt/data/u/feic/pjs/model/Qwen3.5-27B"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    train = datasets.load_dataset("openai/gsm8k", "main", split="train")
    ex = [f"Q: {r['question']}\nA: {re.sub(r'<<[^>]*>>', '', r['answer']).replace('####', 'Answer:')}" for r in train]
    ex_len = [len(tok.encode(e)) + 2 for e in ex]
    rng = random.Random(0)
    tasks = [json.loads(line) for line in open(ROOT / "data/gsm8k_128.jsonl")]
    out, lens = [], []
    for t in tasks:
        target, picked, n = rng.uniform(1000, 7000), [], 0
        while n < target:
            i = rng.randrange(len(ex))
            picked.append(ex[i])
            n += ex_len[i]
        q = ("Here are some solved example problems for reference:\n\n" + "\n\n".join(picked)
             + f"\n\nNow solve this problem:\n{t['question']}")
        out.append({**t, "question": q})
        lens.append(len(tok.encode(q)))
    with open(ROOT / "data/gsm8k_longctx_128.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in out)
    lens.sort()
    print(f"{len(out)} tasks; question tokens min {lens[0]}, median {lens[len(lens) // 2]}, max {lens[-1]}, "
          f"total {sum(lens)}")


if __name__ == "__main__":
    main()
