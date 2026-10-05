"""Qwen3.5-27B text forward over a flat batch of token rows (v3).

One forward = one engine step over a fixed-shape row layout (so every step can be a CUDA graph replay):
rows [0, P) are prefill chunks of up to `npre` sequences (consecutive rows per sequence), rows [P, P + D) are
decode rows (one token of one sequence each), rows beyond are padding; P and D are device values (row0 = P), the
shapes depend only on the layout (T rows, npre). Decode-only layouts have npre = 0 (P = 0). All sequence state
lives in pools owned by the model:
  - Gated-DeltaNet recurrent state  rec[layer]  [S + 1, 48, 128, 128] bf16 (slot S = scratch)
  - Gated-DeltaNet conv window      conv[layer] [S + 1, 3, 10240] bf16
  - full-attention KV, paged        k/v[layer]  [num_blocks * BS, 4, 256] bf16 (block 0 = scratch)
and per slot a block table [S + 1, max_blocks] (device, persistent) and the last sampled token `last_tok`, which
the next step's decode rows read on the device (so a step can be launched before the previous step's tokens
reach the host).

Weights: read straight from the checkpoint; every Linear of the decoder (not lm_head) as FP8 e4m3 with one
per-tensor scale (amax / 448, computed like torchao's per-tensor recipe); activations quantized per token by the
producing kernel (see kernels.py).
"""
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from . import kernels as K

H, I = 5120, 17408
HV, HK, DK, DV = 48, 16, 128, 128
CONV_C = 2 * HK * DK + HV * DV  # 10240
# Mixed steps: the prefill-row kernels (gdn kkt + chunk, attention prefill) and the decode-row kernels (gdn_decode,
# attention decode) are independent (disjoint output rows, different state slots); "gdn" / "attn" run the prefill
# ones on a side stream concurrently with the decode ones (parallel branches of the step's CUDA graph). Bit-exact;
# small mixed steps (B=40, 40 prefill rows) 20.6 -> 20.0 ms; big ones are power-capped, the sections shorten but
# the step does not (energy per step is the bound).
OVERLAP = set(filter(None, os.environ.get("AGENTOPT_OVERLAP", "gdn,attn").split(",")))
CONV_FUSE = os.environ.get("AGENTOPT_CONV_FUSE", "1") != "0"
# prefill-row attention: FA3 (flash_attn_3 hopper build, hdim 256 paged; 1.35-1.5x our Triton kernel) if installed
ATTN_FA3 = K.FA3 is not None and os.environ.get("AGENTOPT_ATTN_FA3", "1") != "0"
# No cuBLASLt workspace: large-M FP8 GEMMs then use the same nvjet kernels with static tile scheduling instead of a
# dynamic scheduler whose counter needs a memset node before every GEMM (the memset breaks the PDL chain: ~7 us of
# gap per GEMM, ~3 ms per 2k-token step). Same speed or faster per GEMM, small-M kernels unchanged. Read by torch at
# the first cuBLASLt call.
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "0")
HQ, HKV, HD, ROT = 24, 4, 256, 64
GROUPS = {"gate_up": ("mlp.", ["gate_proj", "up_proj"]),
          "gdn_in": ("linear_attn.", ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]),
          "attn_in": ("self_attn.", ["q_proj", "k_proj", "v_proj"])}


def quantize_fp8(w: torch.Tensor):
    """Per-tensor e4m3 (torchao recipe: scale = amax / 448 in the weight dtype, then x / scale, clamp, cast)."""
    scale = (w.abs().amax() / 448.0).float()
    q = (w.float() / scale).clamp(-448.0, 448.0).to(K.FP8)
    return q, scale


@dataclass
class Step:
    """Device views of one step's metadata (int32; see module docstring) and the layout sizes."""
    T: int                    # rows (incl. padding)
    npre: int                 # max prefill sequences (0: decode-only layout)
    dmax: int                 # max decode rows (programs of the decode kernels, from row0)
    nl: int                   # sampled rows
    tok: torch.Tensor         # [T] token id, or -1: take last_tok[in_slot]
    in_slot: torch.Tensor     # [T]
    pos: torch.Tensor         # [T]
    slot_map: torch.Tensor    # [T] KV row (block * BS + offset) written by the token
    row_start: torch.Tensor   # [T] first row of the row's sequence (token-parallel conv)
    row_seq_slot: torch.Tensor  # [T] slot of the row's sequence (scratch for padding)
    row_slot: torch.Tensor    # [T + dmax] slot of decode rows, -1 otherwise
    row0: torch.Tensor        # [1] first decode row (= P)
    seq_slot: torch.Tensor    # [ns] conv-state sequences (decode-only: one per row)
    seq_start: torch.Tensor   # [ns]
    seq_len: torch.Tensor     # [ns] (0: padding)
    logit_row: torch.Tensor   # [nl] row whose logits are sampled
    sample_slot: torch.Tensor  # [nl] slot whose last_tok receives the sample (scratch if none)
    # prefill (npre > 0)
    p_slot: torch.Tensor | None = None     # [npre + 1] slots of the prefill sequences (last ones: padding)
    p_seqk: torch.Tensor | None = None     # [npre + 1] their lengths after this step (0: padding)
    cu: torch.Tensor | None = None         # [npre + 2] row offsets
    chunk_idx: torch.Tensor | None = None  # [nt, 2] (sequence, chunk) of each 64-chunk; padding -> empty seq
    tiles: torch.Tensor | None = None      # [4, n_tiles] attention tiles: tok0, ntok, pos0, slot


class Model:
    def __init__(self, path: str, num_slots: int, block_size: int = 64, max_blocks_per_seq: int = 512,
                 device: str = "cuda"):
        self.path, self.dev = Path(path), device
        self.side = torch.cuda.Stream(device, priority=-1)  # side stream of mixed steps (OVERLAP)
        cfg = json.loads((self.path / "config.json").read_text())["text_config"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.layer_types = cfg["layer_types"]
        self.S, self.BS = num_slots, block_size
        self._load()
        S1 = num_slots + 1
        n_lin = sum(t == "linear_attention" for t in self.layer_types)
        n_att = len(self.layer_types) - n_lin
        self.rec = torch.zeros((n_lin, S1, HV, DK, DV), dtype=torch.bfloat16, device=device)
        self.conv = torch.zeros((n_lin, S1, 3, CONV_C), dtype=torch.bfloat16, device=device)
        self.n_att = n_att
        self.block_table = torch.zeros((S1, max_blocks_per_seq), dtype=torch.int32, device=device)
        self.last_tok = torch.zeros((S1,), dtype=torch.long, device=device)

    def kv_block_bytes(self):
        return 2 * self.n_att * self.BS * HKV * HD * 2

    def alloc_kv(self, num_blocks: int):
        self.NB = num_blocks
        self.kc = torch.zeros((self.n_att, num_blocks * self.BS, HKV, HD), dtype=torch.bfloat16, device=self.dev)
        self.vc = torch.zeros_like(self.kc)

    # ------------------------------------------------------------------ weights
    def _load(self):
        idx = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        pre = "model.language_model."
        files = {}
        for name, f in idx.items():
            if name.startswith(pre) or name == "lm_head.weight":
                files.setdefault(f, []).append(name)
        # linears sharing an input are stacked into one buffer per group, allocated up front (merging after the
        # load would leave the freed per-linear weights as fragmentation)
        shapes = {}
        for f, names in files.items():
            with safe_open(str(self.path / f), framework="pt", device="cpu") as fh:
                for n in names:
                    shapes[n.removeprefix(pre)] = fh.get_slice(n).get_shape()
        self.group_buf, dest = {}, {}
        for i, kind in enumerate(self.layer_types):
            for gname, (sub, members) in GROUPS.items():
                if gname != "gate_up" and (gname == "gdn_in") != (kind == "linear_attention"):
                    continue
                names = [f"layers.{i}.{sub}{m}.weight" for m in members]
                rows = [shapes[n][0] for n in names]
                buf = torch.empty((sum(rows), shapes[names[0]][1]), dtype=K.FP8, device=self.dev)
                self.group_buf[(i, gname)] = buf
                o = 0
                for n, r in zip(names, rows):
                    dest[n] = buf[o:o + r]
                    o += r
        raw = {}  # quantized as they are read: the bf16 checkpoint (54 GB) never sits on the GPU at once
        for f, names in files.items():
            with safe_open(str(self.path / f), framework="pt", device=self.dev) as fh:
                for n in names:
                    t = fh.get_tensor(n)
                    short = n.removeprefix(pre)
                    if t.ndim == 2 and short.startswith("layers.") and "conv1d" not in short:
                        t = quantize_fp8(t)
                        if short in dest:
                            dest[short].copy_(t[0])
                            t = (dest[short], t[1])
                    raw[short] = t
                    del t
            torch.cuda.empty_cache()
        self.embed = raw.pop("embed_tokens.weight")
        self.lm_head = raw.pop("lm_head.weight")
        self.final_w1 = (1.0 + raw.pop("norm.weight").float()).contiguous()
        self.layers = []
        for i, kind in enumerate(self.layer_types):
            p = f"layers.{i}."
            g = lambda n: raw.pop(p + n)  # noqa: E731
            L = {"kind": kind,
                 "in_w1": (1.0 + g("input_layernorm.weight").float()).contiguous(),
                 "post_w1": (1.0 + g("post_attention_layernorm.weight").float()).contiguous()}
            names = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
            if kind == "linear_attention":
                names += ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
                          "linear_attn.in_proj_a", "linear_attn.out_proj"]
                L["conv_w"] = g("linear_attn.conv1d.weight").float().reshape(CONV_C, 4).contiguous()
                L["A_log"] = g("linear_attn.A_log").float().contiguous()
                L["dt_bias"] = g("linear_attn.dt_bias").float().contiguous()
                L["gnorm_w"] = g("linear_attn.norm.weight").float().contiguous()
            else:
                names += ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                L["qn_w1"] = (1.0 + g("self_attn.q_norm.weight").float()).contiguous()
                L["kn_w1"] = (1.0 + g("self_attn.k_norm.weight").float()).contiguous()
            for n in names:
                L[n.split(".")[1]] = g(n + ".weight")
            for gname, (_, members) in GROUPS.items():
                if (i, gname) in self.group_buf:
                    parts = [L.pop(n) for n in members]
                    svec = torch.cat([sc.expand(q.shape[0]) for q, sc in parts]).contiguous()
                    L[gname] = (self.group_buf[(i, gname)], svec, parts)
            self.layers.append(L)
        assert not raw, f"unused weights: {list(raw)[:5]}"
        self.one = torch.ones((), dtype=torch.float32, device=self.dev)

    def mm(self, xq, w):
        q, s = w
        return torch._scaled_mm(xq, q.t(), self.one, s, out_dtype=torch.bfloat16)

    def mm_group(self, xq, G, conv=None):
        """G = (stacked fp8 weight, per-column scale vector, per-linear (weight view, scale)): linears sharing an
        input, weights and per-tensor scales unchanged. Outputs: column slices of one small-M GEMM, or per-linear
        cuBLAS calls. conv: see K.mm_fp8 (None if the GEMM cannot take it)."""
        w, svec, parts = G
        cfg = K.mm_fp8_cfg(xq.shape[0], w.shape[0])
        if cfg is None:
            assert conv is None
            return [self.mm(xq, p) for p in parts]
        y = K.mm_fp8(xq, w, svec, cfg, conv)
        out, o = [], 0
        for q, _ in parts:
            out.append(y[:, o:o + q.shape[0]])
            o += q.shape[0]
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, b: Step) -> torch.Tensor:
        """Runs one step; returns the sampled token of every sampled row [nl] (greedy), also stored in last_tok."""
        ids = torch.where(b.tok >= 0, b.tok, self.last_tok[b.in_slot])
        res = self.embed[ids]
        if b.npre:
            pre = (b.cu.long(), b.chunk_idx.long())
            page_table = self.block_table[b.p_slot.long()] if ATTN_FA3 else None
        else:
            pre = page_table = None
        delta = dscale = None
        li = ai = 0
        for L in self.layers:
            xq, xs = K.add_rmsnorm_quant(res, L["in_w1"], self.eps, delta, dscale)
            if L["kind"] == "linear_attention":
                delta, dscale = self._gdn(L, li, xq, xs, b, pre)
                li += 1
            else:
                delta, dscale = self._attn(L, ai, xq, xs, b, page_table)
                ai += 1
            xq, xs = K.add_rmsnorm_quant(res, L["post_w1"], self.eps, delta, dscale)
            hq, hs = K.silu_mul_quant(*self.mm_group(xq, L["gate_up"]), xs)
            delta, dscale = self.mm(hq, L["down_proj"]), hs
        h = K.add_rmsnorm_quant(res, self.final_w1, self.eps, delta, dscale, quant=False)
        logits = h[b.logit_row] @ self.lm_head.t()
        nxt = logits.argmax(-1)
        self.last_tok[b.sample_slot] = nxt
        return nxt

    def _branch(self, kind):
        """Context for the prefill-row kernels: on the side stream (forked from the current one) if kind overlaps."""
        if kind not in OVERLAP:
            return contextlib.nullcontext()
        self.side.wait_stream(torch.cuda.current_stream())
        return torch.cuda.stream(self.side)

    def _join(self, kind, b: Step):
        if kind in OVERLAP and b.npre:
            torch.cuda.current_stream().wait_stream(self.side)

    def _gdn(self, L, li, xq, xs, b: Step, pre):
        # decode-only steps with the small-M GEMM: conv1d in its epilogue (columns [0, CONV_C) come out as SiLU(conv),
        # q / k not normalized yet)
        fuse = CONV_FUSE and not b.npre and K.mm_fp8_cfg(b.T, L["gdn_in"][0].shape[0]) is not None
        conv = (xs, L["conv_w"], self.conv[li], b.row_seq_slot) if fuse else None
        qkv_raw, z_raw, b_raw, a_raw = self.mm_group(xq, L["gdn_in"], conv)
        ns = b.seq_slot.shape[0]
        if fuse:
            q, k, v = qkv_raw[:, :HK * DK], qkv_raw[:, HK * DK:2 * HK * DK], qkv_raw[:, 2 * HK * DK:]
            o = torch.empty((b.T, HV, DV), dtype=torch.bfloat16, device=self.dev)
        elif b.npre:
            q, k, v = K.conv1d_rows(qkv_raw, xs, L["conv_w"], self.conv[li], b.row_start, b.row_seq_slot,
                                    b.seq_slot, b.seq_start, b.seq_len, ns)
            g, beta = K.gdn_gates(a_raw, b_raw, xs, L["A_log"], L["dt_bias"])
            cu, ci = pre
            o = torch.empty((b.T, HV, DV), dtype=torch.bfloat16, device=self.dev)
            with self._branch("gdn"):
                K.gdn_prefill(q, k, v, g, beta, self.rec[li], b.p_slot, cu, ci, b.npre + 1, o=o)
        else:
            q, k, v = K.conv1d_varlen(qkv_raw, xs, L["conv_w"], self.conv[li], b.seq_slot, b.seq_start, b.seq_len,
                                      ns)
            o = torch.empty((b.T, HV, DV), dtype=torch.bfloat16, device=self.dev)
        K.gdn_decode(q, k, v, a_raw, b_raw, xs, L["A_log"], L["dt_bias"], self.rec[li], b.row_slot, b.row0, b.dmax,
                     o, norm=fuse)
        self._join("gdn", b)
        oq, os_ = K.gated_norm_quant(o, z_raw, xs, L["gnorm_w"], self.eps)
        return self.mm(oq, L["out_proj"]), os_

    def _attn(self, L, ai, xq, xs, b: Step, page_table):
        q_raw, k_raw, v_raw = self.mm_group(xq, L["attn_in"])
        q = K.qkv_prep(q_raw, k_raw, v_raw, xs, b.pos, b.slot_map, L["qn_w1"], L["kn_w1"], self.kc[ai],
                       self.vc[ai], self.eps, self.theta)
        o = torch.empty((b.T, HQ, HD), dtype=torch.bfloat16, device=self.dev)
        if b.npre:
            t = b.tiles
            with self._branch("attn"):
                if ATTN_FA3:
                    K.attn_prefill_fa3(q, self.kc[ai], self.vc[ai], page_table, b.p_seqk, b.cu, b.T, o, self.BS)
                else:
                    K.attn_prefill(q, self.kc[ai], self.vc[ai], self.block_table, t[0], t[1], t[2], t[3], t.shape[1],
                                   o, self.BS)
        K.attn_decode(q, self.kc[ai], self.vc[ai], self.block_table, b.row_slot, b.pos, b.row0, b.dmax, o, self.BS)
        self._join("attn", b)
        oq, os_ = K.attn_gate_quant(o, q_raw, xs)
        return self.mm(oq, L["o_proj"]), os_
