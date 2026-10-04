"""Phase-2 engine: continuous batching over a fixed set of slots (one in-flight agent turn per slot).

API used by the continuous runner:
    prefill(slot, ids) -> first token     batch-1 prefill of one sequence into its slot (eager)
    step() -> list[int] (one per slot)     one decode step for ALL slots: one CUDA-graph replay
    release(slot)                          park the row (its outputs are ignored from now on)

The decode step is the phase-1 recipe (compiled forward + GQA-folded decode attention, captured in one CUDA
graph) at batch S: row r = slot r, its own token / position / mask row, so rows can be at any length and a
sequence can join or leave between any two steps. Weights are read once per step for all S rows, which is the
whole point: at batch 1 a step is bound by reading 54 GB of weights; at batch S the same read serves S tokens.
Idle rows still compute (their outputs are discarded); they are parked at a scratch position (see slot_cache).
"""
import os
import time

import torch

from .naive import NaiveEngine
from .slot_cache import SlotCache


class BatchEngine(NaiveEngine):
    def __init__(self, num_slots: int = 32, max_len: int = 2048, compile_decode: bool = True,
                 gqa_decode_attn: bool = True, **kw):
        super().__init__(**kw)
        self.options.update(num_slots=num_slots, max_len=max_len, compile_decode=compile_decode,
                            gqa_decode_attn=gqa_decode_attn)
        if gqa_decode_attn:
            from .decode_attention import install
            install(self.model)
        self.decode_model = self.model
        if compile_decode:
            os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))
            torch._dynamo.config.cache_size_limit = 1024
            torch._dynamo.config.accumulated_cache_size_limit = 4096
            self.decode_model = torch.compile(self.model, dynamic=False, fullgraph=False)
        self.num_slots, self.max_len, self.park = num_slots, max_len, max_len - 1
        S, dev = num_slots, self.device
        self.tok_buf = torch.zeros((S, 1), dtype=torch.long, device=dev)
        self.pos_buf = torch.full((S,), self.park, dtype=torch.long, device=dev)
        self.out_buf = torch.zeros((S,), dtype=torch.long, device=dev)
        self.mask_buf = torch.zeros((S, 1, 1, max_len), dtype=torch.bool, device=dev)
        self.rows = torch.arange(S, device=dev)
        self.trues = torch.ones((S,), dtype=torch.bool, device=dev)  # device-side value: no H2D copy in capture
        self.host_buf = torch.zeros((S,), dtype=torch.long, pin_memory=True)
        self.event = torch.cuda.Event()
        self.cache = SlotCache(self.model.config, S, max_len, self.pos_buf)
        t0 = time.perf_counter()
        self._capture()
        print(f"batch decode graph (S={S}) captured in {time.perf_counter() - t0:.1f}s, "
              f"cache {self.cache.memory_bytes() / 2**30:.1f} GiB", flush=True)

    def _step(self):
        self.mask_buf.view(self.num_slots, self.max_len).index_put_((self.rows, self.pos_buf), self.trues)
        out = self.decode_model(input_ids=self.tok_buf, position_ids=self.pos_buf.view(-1, 1),
                                attention_mask=self.mask_buf, past_key_values=self.cache, use_cache=True)
        self.out_buf.copy_(out.logits[:, -1].argmax(-1))
        self.tok_buf.copy_(self.out_buf.view(-1, 1))
        self.pos_buf.add_(1).clamp_(max=self.park)

    @torch.inference_mode()
    def _capture(self):
        self.prefill(0, list(range(16)))  # allocates the slot storage (lazy, from HF's own tensors)
        self.cache.begin_decode()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()
        for slot in range(self.num_slots):
            self.release(slot)

    @torch.inference_mode()
    def prefill(self, slot: int, input_ids: list[int]) -> int:
        n = len(input_ids)
        assert n < self.park, "prompt exceeds max_len"
        self.cache.begin_prefill(slot, 0)
        out = self.model(input_ids=torch.tensor([input_ids], device=self.device),
                         position_ids=torch.arange(n, device=self.device).view(1, -1),
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.cache.begin_decode()
        next_id = out.logits[0, -1].argmax()
        self.mask_buf[slot].zero_()
        self.mask_buf[slot, ..., :n] = True
        self.tok_buf[slot] = next_id
        self.pos_buf[slot] = n
        return int(next_id)

    def release(self, slot: int):
        self.mask_buf[slot].zero_()
        self.pos_buf[slot] = self.park

    def room(self, slot_len: int) -> int:
        """Tokens a sequence of length slot_len may still generate before hitting the scratch position."""
        return self.park - slot_len

    @torch.inference_mode()
    def step(self) -> list[int]:
        self.graph.replay()
        self.host_buf.copy_(self.out_buf, non_blocking=True)
        self.event.record()
        self.event.synchronize()
        return self.host_buf.tolist()

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, timer, ignore_eos: bool = False):
        """Single-sequence generation on slot 0 (for check_equiv / microbench compatibility)."""
        assert len(input_ids) + max_new_tokens <= self.park, "context exceeds max_len"
        stop_ids = set() if ignore_eos else self.stop_ids
        with timer("prefill") as t_pre:
            gen = [self.prefill(0, input_ids)]
        with timer("decode") as t_dec:
            while gen[-1] not in stop_ids and len(gen) < max_new_tokens:
                gen.append(self.step()[0])
        self.release(0)
        stopped = gen[-1] in stop_ids
        if stopped:
            gen = gen[:-1]
        return gen, {"prefill_s": t_pre.elapsed, "decode_s": t_dec.elapsed, "stopped": stopped}
