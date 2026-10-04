"""Phase-2 engine: continuous batching over a fixed set of slots (one in-flight agent turn per slot).

API used by the continuous runner:
    prefill(slot, ids) -> first token     batch-1 prefill of one sequence into its slot (eager)
    submit(slot, ids) -> first token|None  start a turn: eager prefill, or (force_max) queue a short suffix
    step() -> {slot: token}                one decode step for ALL slots: one CUDA-graph replay; returns the
                                           new token of every row that is generating (not consuming a suffix)
    release(slot)                          park the row (its outputs are ignored from now on)
    suspend(slot)                          park the row but keep its state (agent waiting for a tool); the
                                           next submit() on the slot resumes it

The decode step is the phase-1 recipe (compiled forward + GQA-folded decode attention, captured in one CUDA
graph) at batch S: row r = slot r, its own token / position / mask row, so rows can be at any length and a
sequence can join or leave between any two steps. Weights are read once per step for all S rows, which is the
whole point: at batch 1 a step is bound by reading 54 GB of weights; at batch S the same read serves S tokens.
Idle rows still compute (their outputs are discarded); they are parked at a scratch position (see slot_cache).

Prefix reuse (optional, `prefix_reuse`): the engine remembers which tokens each slot's state covers (prompt +
every token fed by decode). A prefill whose ids start with those tokens only processes the new suffix (agent
turn k+1 = turn k + its answer + the tool result). A fresh slot whose ids start with the registered shared prefix
(system prompt + tool schema, `set_shared_prefix`) gets that prefix's state copied in and prefills the rest.

Forced-token prefill (optional, `force_max`): an eager batch-1 prefill costs ~105 ms per call almost regardless
of length (host-bound, 64 layers of HF dispatch + Triton launches) and stalls every other row. A reusable suffix
of at most `force_max` tokens is instead fed through the decode graph one token per step: the graph takes the
row's input from a forced-token buffer (filled by the host before each replay) instead of its last output, and
the row's outputs are ignored until the last suffix token has been fed. Same state update as decoding, no
extra call, the other rows keep decoding.

Suspend (for asynchronous tools): every replay updates the Gated-DeltaNet states of ALL rows in place, parked
or not, so a slot whose agent is waiting for a tool would have its recurrent state overwritten by garbage. KV is
safe (a parked row writes only to the scratch position). suspend() saves the slot's conv/recurrent states and
parks the row; resume (inside submit) copies them back and moves the row back to its real end.
"""
import collections
import os
import time

import torch

from .naive import NaiveEngine
from .slot_cache import SlotCache


class BatchEngine(NaiveEngine):
    def __init__(self, num_slots: int = 32, max_len: int = 2048, compile_decode: bool = True,
                 gqa_decode_attn: bool = True, prefix_reuse: bool = False, force_max: int = 0, **kw):
        super().__init__(**kw)
        self.options.update(num_slots=num_slots, max_len=max_len, compile_decode=compile_decode,
                            gqa_decode_attn=gqa_decode_attn, prefix_reuse=prefix_reuse, force_max=force_max)
        assert prefix_reuse or not force_max, "forced-token prefill only feeds suffixes of reusable state"
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
        self.force_dev = torch.zeros((2, S), dtype=torch.long, device=dev)  # [token, on] per row, read in-graph
        self.force_host = torch.zeros((2, S), dtype=torch.long, pin_memory=True)
        self.event = torch.cuda.Event()
        self.cache = SlotCache(self.model.config, S, max_len, self.pos_buf)
        self.prefix_reuse = prefix_reuse
        if prefix_reuse:
            from .hybrid_prefill import install as install_extend
            install_extend(self.model, self.cache)
        self.slot_tokens = [None] * S  # tokens whose state slot s holds (None = nothing reusable)
        self.next_tok = [None] * S     # token the next decode step feeds into slot s
        self.live = set()
        self.force_max = force_max
        self.force_q = [collections.deque() for _ in range(S)]  # suffix tokens still to feed into slot s
        self.suspended = {}  # slot -> saved linear-attention states while its agent waits for a tool
        self.shared = None             # (tokens, snapshot) of the shared prefix
        self.reset_stats()
        t0 = time.perf_counter()
        self._capture()
        print(f"batch decode graph (S={S}) captured in {time.perf_counter() - t0:.1f}s, "
              f"cache {self.cache.memory_bytes() / 2**30:.1f} GiB", flush=True)

    def _step(self):
        self.mask_buf.view(self.num_slots, self.max_len).index_put_((self.rows, self.pos_buf), self.trues)
        tok = torch.where(self.force_dev[1].view(-1, 1) > 0, self.force_dev[0].view(-1, 1), self.tok_buf)
        out = self.decode_model(input_ids=tok, position_ids=self.pos_buf.view(-1, 1),
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

    def reset_stats(self):
        self.stats = {"prefill_calls": 0, "prefill_tokens": 0, "reused_turn": 0, "reused_shared": 0,
                      "forced_tokens": 0,
                      "decode_steps": 0, "live_row_steps": 0}

    @staticmethod
    def _is_prefix(p, ids):
        return p is not None and len(p) < len(ids) and ids[:len(p)] == p

    @torch.inference_mode()
    def set_shared_prefix(self, ids: list[int]):
        """Prefill `ids` once and keep a snapshot of the resulting state for every later fresh prefill."""
        self.shared = None
        self.prefill(0, ids)
        self.shared = (list(ids), self.cache.snapshot(0, len(ids)))
        self.release(0)

    @torch.inference_mode()
    def prefill(self, slot: int, input_ids: list[int]) -> int:
        n = len(input_ids)
        assert n < self.park, "prompt exceeds max_len"
        start = 0
        if self.prefix_reuse and self._is_prefix(self.slot_tokens[slot], input_ids):
            start = len(self.slot_tokens[slot])
            self.stats["reused_turn"] += start
        elif self.prefix_reuse and self.shared is not None and self._is_prefix(self.shared[0], input_ids):
            self.cache.restore(slot, self.shared[1])
            start = len(self.shared[0])
            self.stats["reused_shared"] += start
        self.stats["prefill_calls"] += 1
        self.stats["prefill_tokens"] += n - start
        self.cache.begin_prefill(slot, start, n - start)
        out = self.model(input_ids=torch.tensor([input_ids[start:]], device=self.device),
                         position_ids=torch.arange(start, n, device=self.device).view(1, -1),
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.cache.begin_decode()
        next_id = int(out.logits[0, -1].argmax())
        self.mask_buf[slot].zero_()
        self.mask_buf[slot, ..., :n] = True
        self.tok_buf[slot] = next_id
        self.pos_buf[slot] = n
        self.slot_tokens[slot], self.next_tok[slot] = list(input_ids), next_id
        self.live.add(slot)
        return next_id

    @torch.inference_mode()
    def submit(self, slot: int, input_ids: list[int]):
        """Start a turn on `slot`. Returns the first generated token (eager prefill) or None (the suffix is
        queued as forced tokens; the first generated token comes out of a later step())."""
        if slot in self.suspended:
            self._resume(slot)
        if not self.force_max:
            return self.prefill(slot, input_ids)
        n = len(input_ids)
        assert n < self.park, "prompt exceeds max_len"
        if self._is_prefix(self.slot_tokens[slot], input_ids):
            start = len(self.slot_tokens[slot])  # slot state, mask and position already end at `start`
            kind = "reused_turn"
        elif self.shared is not None and self._is_prefix(self.shared[0], input_ids):
            start, kind = len(self.shared[0]), "reused_shared"
        else:
            return self.prefill(slot, input_ids)
        if n - start > self.force_max:
            return self.prefill(slot, input_ids)
        if kind == "reused_shared":
            self.cache.restore(slot, self.shared[1])
            self.mask_buf[slot].zero_()
            self.mask_buf[slot, ..., :start] = True
            self.pos_buf[slot] = start
            self.slot_tokens[slot] = list(input_ids[:start])
        self.stats[kind] += start
        self.stats["forced_tokens"] += n - start
        self.force_q[slot] = collections.deque(input_ids[start:])
        self.next_tok[slot] = None
        self.live.add(slot)
        return None

    @torch.inference_mode()
    def suspend(self, slot: int):
        """Take a slot out of decoding while keeping its state (its tokens / KV stay, its row is parked)."""
        assert not self.force_q[slot], "cannot suspend while consuming a suffix"
        self.suspended[slot] = [(l.conv_states[slot].clone(), l.recurrent_states[slot].clone())
                                if l is not None else None for l in self.cache.layers]
        self.pos_buf[slot] = self.park
        self.live.discard(slot)

    def _resume(self, slot: int):
        for l, saved in zip(self.cache.layers, self.suspended.pop(slot)):
            if saved is not None:
                l.conv_states[slot].copy_(saved[0])
                l.recurrent_states[slot].copy_(saved[1])
        self.mask_buf[slot, ..., self.park] = False  # the parked steps marked the scratch position
        self.pos_buf[slot] = len(self.slot_tokens[slot])
        self.live.add(slot)

    def release(self, slot: int):
        self.suspended.pop(slot, None)
        self.force_q[slot].clear()
        self.mask_buf[slot].zero_()
        self.pos_buf[slot] = self.park
        self.live.discard(slot)
        self.slot_tokens[slot] = self.next_tok[slot] = None

    def room(self, slot_len: int) -> int:
        """Tokens a sequence of length slot_len may still generate before hitting the scratch position."""
        return self.park - slot_len

    @torch.inference_mode()
    def step(self) -> dict[int, int]:
        forcing = [s for s in self.live if self.force_q[s]]
        if forcing or self.force_host[1].any():
            self.force_host.zero_()
            for s in forcing:
                self.force_host[0, s], self.force_host[1, s] = self.force_q[s][0], 1
            self.force_dev.copy_(self.force_host, non_blocking=True)  # host buffer reused only after the sync
        self.graph.replay()
        self.host_buf.copy_(self.out_buf, non_blocking=True)
        self.event.record()
        self.event.synchronize()
        out = self.host_buf.tolist()
        self.stats["decode_steps"] += 1
        self.stats["live_row_steps"] += len(self.live)
        emitted = {}
        for s in self.live:  # the step fed the head of the forced queue, else next_tok[s], into slot s
            q = self.force_q[s]
            self.slot_tokens[s].append(q.popleft() if q else self.next_tok[s])
            if not q:
                self.next_tok[s] = emitted[s] = out[s]
        return emitted

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
