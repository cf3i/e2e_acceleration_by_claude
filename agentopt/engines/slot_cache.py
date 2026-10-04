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

Storage is allocated lazily from the first prefill, so dtypes/shapes are whatever HF's layers produce.
Rows without a live sequence are parked at position `max_len - 1` (a scratch slot real sequences never reach).
"""
import torch


class _LinearSlots:
    """What the Gated-DeltaNet layer reads directly in decode: all slots' states."""
    def __init__(self):
        self.conv_states = None
        self.recurrent_states = None


class SlotCache:
    is_compileable = True

    def __init__(self, config, num_slots: int, max_len: int, pos: torch.Tensor):
        self.num_slots, self.max_len = num_slots, max_len
        self.pos = pos  # (S,) device tensor owned by the engine: decode write position of every row
        self.rows = torch.arange(num_slots, device=pos.device)
        self.layer_types = config.layer_types
        self.layers = [_LinearSlots() if t == "linear_attention" else None for t in self.layer_types]
        self.keys = [None] * len(self.layer_types)
        self.values = [None] * len(self.layer_types)
        self.mode, self.slot, self.start = "decode", 0, 0

    # --- mode switches (host side, outside the captured graph) ---
    def begin_prefill(self, slot: int, start: int = 0):
        self.mode, self.slot, self.start = "prefill", slot, start

    def begin_decode(self):
        self.mode = "decode"

    # --- what HF mask creation calls (prefill only; decode passes a ready 4D mask) ---
    def get_seq_length(self, layer_idx: int = 0):
        return self.start

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0):
        return self.start + query_length, 0

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        return self.mode == "decode"

    # --- full-attention layers ---
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.keys[layer_idx] is None:
            shape = (self.num_slots, key_states.shape[1], self.max_len, key_states.shape[-1])
            self.keys[layer_idx] = key_states.new_zeros(shape)
            self.values[layer_idx] = value_states.new_zeros(shape)
        k, v = self.keys[layer_idx], self.values[layer_idx]
        if self.mode == "prefill":
            s, a, n = self.slot, self.start, key_states.shape[2]
            k[s, :, a:a + n] = key_states[0]
            v[s, :, a:a + n] = value_states[0]
            return k[s:s + 1, :, :a + n], v[s:s + 1, :, :a + n]
        k[self.rows, :, self.pos] = key_states[:, :, 0]
        v[self.rows, :, self.pos] = value_states[:, :, 0]
        return k, v

    # --- Gated-DeltaNet layers ---
    def update_conv_state(self, conv_states, layer_idx, **kwargs):
        layer = self.layers[layer_idx]
        if layer.conv_states is None:
            layer.conv_states = conv_states.new_zeros((self.num_slots, *conv_states.shape[1:]))
        if self.mode == "prefill":
            layer.conv_states[self.slot].copy_(conv_states[0])
            return layer.conv_states[self.slot:self.slot + 1]
        layer.conv_states.copy_(conv_states)
        return layer.conv_states

    def update_recurrent_state(self, recurrent_states, layer_idx, **kwargs):
        layer = self.layers[layer_idx]
        if layer.recurrent_states is None:
            layer.recurrent_states = recurrent_states.new_zeros((self.num_slots, *recurrent_states.shape[1:]))
        if self.mode == "prefill":
            layer.recurrent_states[self.slot].copy_(recurrent_states[0])
            return layer.recurrent_states[self.slot:self.slot + 1]
        layer.recurrent_states.copy_(recurrent_states)
        return layer.recurrent_states

    def memory_bytes(self) -> int:
        ts = [t for t in self.keys + self.values if t is not None]
        ts += [t for l in self.layers if l is not None for t in (l.conv_states, l.recurrent_states) if t is not None]
        return sum(t.numel() * t.element_size() for t in ts)
