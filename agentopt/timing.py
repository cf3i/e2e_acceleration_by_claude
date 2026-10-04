"""L1 application-level timing.

Every timed section is also an NVTX range, so the same names show up in nsys timelines.
Sections must not be nested: the breakdown is "sum of sections + other = makespan".
"""
import collections
import contextlib
import time

import torch

_CUDA = torch.cuda.is_available()


class Section:
    __slots__ = ("elapsed",)

    def __init__(self):
        self.elapsed = 0.0


class Timer:
    def __init__(self, sync: bool = True):
        # sync=True makes GPU sections honest (wait for queued kernels before stopping the clock).
        self.sync = sync and _CUDA
        self.totals = collections.defaultdict(float)
        self.counts = collections.defaultdict(int)

    @contextlib.contextmanager
    def __call__(self, name: str):
        sec = Section()
        if _CUDA:
            torch.cuda.nvtx.range_push(name)
        t0 = time.perf_counter()
        try:
            yield sec
        finally:
            if self.sync:
                torch.cuda.synchronize()
            sec.elapsed = time.perf_counter() - t0
            self.totals[name] += sec.elapsed
            self.counts[name] += 1
            if _CUDA:
                torch.cuda.nvtx.range_pop()

    def breakdown(self, makespan: float) -> dict:
        rows = {k: {"seconds": v, "count": self.counts[k], "frac": v / makespan} for k, v in self.totals.items()}
        other = makespan - sum(self.totals.values())
        rows["other"] = {"seconds": other, "count": 0, "frac": other / makespan}
        return dict(sorted(rows.items(), key=lambda kv: -kv[1]["seconds"]))
