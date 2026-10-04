"""Slot cache for continuous batching: one fixed slot per in-flight sequence, each at its own position.

HF's StaticCache keeps one write position per layer shared by the whole batch, so batch rows cannot be at
different lengths and a new sequence cannot be prefilled into one row while the others are mid-decode. This cache
is duck-typed to exactly what HF's Qwen3.5 code calls (mask creation + full-attention `update` + Gated-DeltaNet
state accessors) and has two modes:

- prefill(slot, start): batch 1, writes the new tokens of one sequence into its slot (KV at [start, start + n),
  conv / recurrent states of the slot) and returns that slot's KV prefix for attention.
- decode: batch = all slots; row r is slot r. Each row writes its KV at its own position `pos[r]` (scatter), and
  attention gets the whole [S, kv_heads, max_len, d] buffer with a per-row mask. Linear-attention states are
  read and updated for all slots in place.

Prefill can also *extend* a slot (start > 0): the new tokens attend to the slot's KV prefix, and the
Gated-DeltaNet layers continue from the slot's conv window / recurrent state (see hybrid_prefill.py). That is
what cross-turn reuse and the shared system-prompt prefix are built on. Unlike KV, a recurrent state exists only
at the position where it was saved, so reuse is possible exactly at a slot's current end or at a snapshot.

Storage is allocated lazily from the first prefill, so dtypes/shapes are whatever HF's layers produce. All
layers of a kind share one tensor ([layers, S, ...]) so that moving a row between slots is a few copies.
Rows without a live sequence are parked at position `max_len - 1` (a scratch slot real sequences never reach).

Decode can run on the first `B` rows only (`set_rows(B)`, used while capturing one CUDA graph per batch bucket):
everything decode reads or writes is then a [:B] view of the same storage.
"""
import torch


class _LinearSlots:
    """What the Gated-DeltaNet layer reads directly: all slots' states (decode: the first B rows)."""
    def __init__(self, cache):
        self.cache = cache
        self.conv_full = None
        self.rec_full = None

    def _rows(self, t):
        return t if t is None or self.cache.mode == "prefill" else t[:self.cache.B]

    @property
    def conv_states(self):
        return self._rows(self.conv_full)

    @property
    def recurrent_states(self):
        return self._rows(self.rec_full)


class SlotCache:
    is_compileable = True

    def __init__(self, config, num_slots: int, max_len: int, pos: torch.Tensor):
        self.num_slots, self.max_len = num_slots, max_len
        self.pos = pos  # (S,) device tensor owned by the engine: decode write position of every row
        self.rows = torch.arange(num_slots, device=pos.device)
        self.B = num_slots  # decode rows (graph bucket being captured)
        self.layer_types = config.layer_types
        self.attn_idx = [i for i, t in enumerate(self.layer_types) if t != "linear_attention"]
        self.lin_idx = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        self.layers = [_LinearSlots(self) if t == "linear_attention" else None for t in self.layer_types]
        self.keys = [None] * len(self.layer_types)
        self.values = [None] * len(self.layer_types)
        self.big = {}  # "k" / "v" / "conv" / "rec" -> [layers, S, ...] storage behind the per-layer views
        self.mode, self.slot, self.start, self.n_new = "decode", 0, 0, 0
        self.prev_conv = None  # extend mode: the slot's conv window before this prefill (current layer)
        self.recurrent_inplace = False  # decode kernel already wrote the new states into our buffers

    # --- mode switches (host side, outside the captured graph) ---
    def begin_prefill(self, slot: int, start: int = 0, n_new: int = 0):
        self.mode, self.slot, self.start, self.n_new = "prefill", slot, start, n_new

    @property
    def extending(self) -> bool:
        return self.mode == "prefill" and self.start > 0

    def begin_decode(self):
        self.mode = "decode"

    # --- what HF mask creation calls (prefill only; decode passes a ready 4D mask) ---
    def get_seq_length(self, layer_idx: int = 0):
        return self.start

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
        return self.start + query_length, 0

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        return self.mode == "decode"

    def set_rows(self, B: int):
        self.B = B

    def _alloc(self, name, like, idx, tail):
        big = like.new_zeros((len(idx), self.num_slots, *tail))
        self.big[name] = big
        return {li: big[j] for j, li in enumerate(idx)}

    def swap_rows(self, a: int, b: int):
        """Exchange everything stored for slots a and b (KV and linear-attention states)."""
        for big in self.big.values():
            tmp = big[:, a].clone()
            big[:, a].copy_(big[:, b])
            big[:, b].copy_(tmp)

    # --- full-attention layers ---
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.keys[layer_idx] is None:
            tail = (key_states.shape[1], self.max_len, key_states.shape[-1])
            for li, t in self._alloc("k", key_states, self.attn_idx, tail).items():
                self.keys[li] = t
            for li, t in self._alloc("v", value_states, self.attn_idx, tail).items():
                self.values[li] = t
        k, v = self.keys[layer_idx], self.values[layer_idx]
        if self.mode == "prefill":
            s, a, n = self.slot, self.start, key_states.shape[2]
            k[s, :, a:a + n] = key_states[0]
            v[s, :, a:a + n] = value_states[0]
            return k[s:s + 1, :, :a + n], v[s:s + 1, :, :a + n]
        B = self.B
        k[self.rows[:B], :, self.pos[:B]] = key_states[:, :, 0]
        v[self.rows[:B], :, self.pos[:B]] = value_states[:, :, 0]
        return k[:B], v[:B]

    # --- Gated-DeltaNet layers ---
    def update_conv_state(self, conv_states, layer_idx, **kwargs):
        layer = self.layers[layer_idx]
        if layer.conv_full is None:
            for li, t in self._alloc("conv", conv_states, self.lin_idx, conv_states.shape[1:]).items():
                self.layers[li].conv_full = t
        if self.mode == "prefill":
            new = conv_states[0]  # last K inputs of the new tokens, zero-padded on the left if n_new < K
            if self.extending:
                # The window must continue the slot's previous inputs, not zero padding.
                self.prev_conv = layer.conv_full[self.slot].clone()
                k = min(self.n_new, new.shape[-1])
                new = torch.cat([self.prev_conv, new[:, new.shape[-1] - k:]], -1)[:, -new.shape[-1]:]
            layer.conv_full[self.slot].copy_(new)
            return layer.conv_full[self.slot:self.slot + 1]
        layer.conv_states.copy_(conv_states)
        return layer.conv_states

    def update_recurrent_state(self, recurrent_states, layer_idx, **kwargs):
        layer = self.layers[layer_idx]
        if layer.rec_full is None:
            for li, t in self._alloc("rec", recurrent_states, self.lin_idx, recurrent_states.shape[1:]).items():
                self.layers[li].rec_full = t
        if self.mode == "prefill":
            layer.rec_full[self.slot].copy_(recurrent_states[0])
            return layer.rec_full[self.slot:self.slot + 1]
        if not self.recurrent_inplace:
            layer.recurrent_states.copy_(recurrent_states)
        return layer.recurrent_states

    # --- snapshots (shared prefix) ---
    def snapshot(self, slot: int, n: int) -> dict:
        """Copy of everything slot `slot` holds for its first n tokens (valid only if n == its current end)."""
        return {"n": n,
                "kv": [(k[slot, :, :n].clone(), v[slot, :, :n].clone()) if k is not None else None
                       for k, v in zip(self.keys, self.values)],
                "lin": [(l.conv_full[slot].clone(), l.rec_full[slot].clone()) if l is not None else None
                        for l in self.layers]}

    def restore(self, slot: int, snap: dict):
        n = snap["n"]
        for k, v, kv in zip(self.keys, self.values, snap["kv"]):
            if kv is not None:
                k[slot, :, :n].copy_(kv[0])
                v[slot, :, :n].copy_(kv[1])
        for l, lin in zip(self.layers, snap["lin"]):
            if lin is not None:
                l.conv_full[slot].copy_(lin[0])
                l.rec_full[slot].copy_(lin[1])

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.big.values())
