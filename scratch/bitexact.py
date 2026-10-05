"""Greedy tokens of a few mixed prefill/decode sessions (for bit-exactness checks between revisions)."""
import json, random, sys
sys.path.insert(0, '.')
from agentopt.v3.engine import V3Engine
eng = V3Engine(num_slots=16)
rng = random.Random(0)
lens = [3000, 1500, 700, 230, 40, 9, 2600, 1100]
gen = {i: [] for i in range(len(lens))}
for i, n in enumerate(lens):
    eng.submit(i, [rng.randrange(1000, 60000) for _ in range(n)], 160)
active = set(gen)
while active:
    out, done = eng.step()
    for s, t in out.items():
        gen[s].append(t)
    active -= done
out, done = eng.drain()
for s, t in out.items():
    gen[s].append(t)
json.dump(gen, open(sys.argv[1], 'w'))
print({s: len(v) for s, v in gen.items()})
