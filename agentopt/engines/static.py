"""Phase-1 engine with a preallocated StaticCache and (optionally) a CUDA-graph-captured decode step.

StaticCache: KV for the 16 full-attention layers is allocated once at `max_cache_len` and written in place
(the write position is a device tensor), the Gated-DeltaNet conv/recurrent states are fixed-size and updated
in place. Every tensor address is stable across tokens and across requests, which is what CUDA graphs need.

Attention mask: with a StaticCache the full-attention layers see all `max_cache_len` KV slots. For a single
query token HF skips building a mask (it assumes the KV holds only real tokens), which would let decode attend
to unfilled slots. So decode passes an explicit 4D bool mask (used as-is by HF) kept in a static buffer that the
step itself extends by one position. Prefill (q_len = n < max_cache_len) gets a correct causal mask from HF.

CUDA graph: one decode step = model forward + argmax + "feed the token back" + "advance the position", all
recorded once and replayed per token, so the ~thousands of kernel launches and all Python/HF dispatch for the
64 layers become one `graph.replay()`. The host still reads the token each step for the stop check.
"""
import torch
from transformers import StaticCache

from .naive import NaiveEngine


class StaticEngine(NaiveEngine):
    def __init__(self, max_cache_len: int = 4096, cuda_graph: bool = False, **kw):
        super().__init__(**kw)
        self.options.update(static_cache=True, max_cache_len=max_cache_len, cuda_graph=cuda_graph)
        self.max_cache_len = max_cache_len
        self.cache = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
        # Static decode-step buffers: current token, its position, and the produced next token.
        self.tok_buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.pos_buf = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.next_buf = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.mask_buf = torch.zeros((1, 1, 1, max_cache_len), dtype=torch.bool, device=self.device)
        self.graph = None
        if cuda_graph:
            self._capture()

    def _step(self):
        """One decode step on the static buffers; leaves tok_buf/pos_buf/mask_buf ready for the next step.
        (The model takes positions from the cache's device-side length; pos_buf mirrors it for the mask.)"""
        self.mask_buf.index_fill_(-1, self.pos_buf, True)
        out = self.model(input_ids=self.tok_buf, attention_mask=self.mask_buf, past_key_values=self.cache,
                         use_cache=True)
        self.next_buf.copy_(out.logits[0, -1].argmax().view(1))
        self.tok_buf.copy_(self.next_buf.view(1, 1))
        self.pos_buf.add_(1)

    @torch.inference_mode()
    def _capture(self):
        # Initialize every cache layer with a dummy prefill (lazy allocation must happen before capture), then
        # warm up on a side stream (Triton autotune / cuBLAS workspaces), then record one step.
        self._prefill(list(range(16)))
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
        self.cache.reset()

    def _prefill(self, input_ids: list[int]) -> int:
        self.cache.reset()
        n = len(input_ids)
        ids = torch.tensor([input_ids], device=self.device)
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        next_id = out.logits[0, -1].argmax()
        self.mask_buf.zero_()
        self.mask_buf[..., :n] = True
        self.tok_buf.copy_(next_id.view(1, 1))
        self.pos_buf.fill_(n)
        return int(next_id)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, timer, ignore_eos: bool = False):
        assert len(input_ids) + max_new_tokens <= self.max_cache_len, "context exceeds max_cache_len"
        stop_ids = set() if ignore_eos else self.stop_ids
        with timer("prefill") as t_pre:
            next_id = self._prefill(input_ids)
        gen = [next_id]
        with timer("decode") as t_dec:
            while next_id not in stop_ids and len(gen) < max_new_tokens:
                if self.graph is not None:
                    self.graph.replay()
                else:
                    self._step()
                next_id = int(self.next_buf)
                gen.append(next_id)
        stopped = gen[-1] in stop_ids
        if stopped:
            gen = gen[:-1]
        return gen, {"prefill_s": t_pre.elapsed, "decode_s": t_dec.elapsed, "stopped": stopped}
