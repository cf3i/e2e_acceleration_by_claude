"""v3 engine: token-budget scheduler over the v3 model (one forward per step, decode + prefill chunks mixed).

Per step (= one `step()` call):
  - prefilling sessions contribute chunks of their pending prompt tokens (FCFS, at most NPRE sequences, within
    the token budget minus the decode rows); a session whose last chunk is scheduled samples at that chunk's last
    row and decodes from the next step on;
  - every decoding session contributes one row; its input token is the one the previous step sampled, read on the
    device from `last_tok` (so step n+1 is launched before step n's tokens reach the host);
  - the rows are padded to a layout bucket (decode-only buckets by row count; mixed buckets by row count with
    NPRE prefill sequences), and every step replays the CUDA graph captured for its bucket.
`step()` launches step n+1, then reads step n's sampled tokens and returns them. A turn that ends (stop token or
limit) at step n has its last token fed by step n+1 as well, so a session's state always covers prompt + all
generated tokens incl. the stop token; the sample step n+1 produces for it is discarded.

Sessions own a slot (recurrent/conv state) and KV blocks. A new turn whose prompt extends the tokens already fed
continues from the state; otherwise the session restarts from the registered shared prefix (state snapshot +
shared full KV blocks) or from scratch.
"""
import collections
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from .. import MODEL_PATH
from . import kernels as K
from .model import Model, Step

IDLE, PREFILL, DECODE = 0, 1, 2
CHUNK = 64  # GDN chunk size


class Layout:
    """Fixed sizes of one step bucket and the offsets of its int32 metadata arrays in the flat meta buffer."""

    def __init__(self, T: int, npre: int, num_slots: int, nl: int | None = None):
        self.T, self.npre = T, npre
        self.dmax = min(T, num_slots)
        self.ns = npre + self.dmax if npre else T
        self.nl = nl or (min(T, npre + self.dmax) if npre else T)
        self.nt = T // CHUNK + npre + 1 if npre else 0
        self.n_tiles = T // K.PREFILL_TQ + npre + 1 if npre else 0
        f = [("tok", T), ("in_slot", T), ("pos", T), ("slot_map", T), ("row_start", T), ("row_seq_slot", T),
             ("row_slot", T + self.dmax), ("row0", 1), ("seq_slot", self.ns), ("seq_start", self.ns),
             ("seq_len", self.ns), ("logit_row", self.nl), ("sample_slot", self.nl)]
        if npre:
            f += [("p_slot", npre + 1), ("cu", npre + 2), ("chunk_idx", 2 * self.nt),
                  ("tiles", 4 * self.n_tiles)]
        self.off, o = {}, 0
        for name, n in f:
            self.off[name] = (o, n)
            o += n
        self.size = o

    def views(self, buf):
        v = {name: buf[o:o + n] for name, (o, n) in self.off.items()}
        if self.npre:
            v["chunk_idx"] = v["chunk_idx"].reshape(self.nt, 2)
            v["tiles"] = v["tiles"].reshape(4, self.n_tiles)
        return v

    def step(self, buf) -> Step:
        return Step(T=self.T, npre=self.npre, dmax=self.dmax, nl=self.nl, **self.views(buf))


class Session:
    def __init__(self, slot):
        self.slot = slot
        self.hist = []        # tokens fed (or scheduled to be fed) whose values the host knows
        self.n_pos = 0        # tokens scheduled into the state (>= len(hist) by the in-flight sampled ones)
        self.todo = []        # prompt tokens still to feed
        self.blocks = []      # KV blocks, in order
        self.n_shared = 0     # leading blocks owned by the shared prefix
        self.phase = IDLE
        self.turn = 0
        self.gen = 0
        self.limit = 0
        self.t_submit = 0.0
        self.force = None     # replay: the tokens this turn emits instead of the sampled ones


class V3Engine:
    def __init__(self, model_path: str = MODEL_PATH, num_slots: int = 128, token_budget: int = 2048,
                 block_size: int = 64, max_len: int = 32768, mem_headroom_gb: float = 4.0,
                 buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128),
                 mixed_buckets=(64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536,
                                1792, 2048),
                 npre: int = 16, cuda_graph: bool = True, device: str = "cuda", **_):
        self.options = {"num_slots": num_slots, "token_budget": token_budget, "block_size": block_size,
                        "buckets": list(buckets), "mixed_buckets": list(mixed_buckets), "npre": npre,
                        "cuda_graph": cuda_graph}
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.stop_ids = {self.im_end, self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.num_slots, self.TB, self.BS, self.dev = num_slots, token_budget, block_size, device
        self.NPRE = npre
        self.max_blocks = max_len // block_size
        self.m = Model(model_path, num_slots, block_size, self.max_blocks, device)
        self.scratch = num_slots
        self.sessions = {}
        self.prefix = None    # (tokens, blocks, rec snapshot, conv snapshot)
        self.buckets = sorted(b for b in buckets if b <= num_slots)
        if self.buckets[-1] < num_slots:
            self.buckets.append(num_slots)
        self.mixed_buckets = sorted(b for b in mixed_buckets if b < token_budget) + [token_budget]
        self.layouts = {}
        tmax = max(Layout(self.TB, npre, num_slots).size, Layout(self.TB, 1, num_slots, nl=self.TB).size)
        self.meta_dev = torch.zeros(tmax, dtype=torch.int32, device=device)
        self.meta_host = [torch.zeros(tmax, dtype=torch.int32).pin_memory() for _ in range(2)]
        self.meta_ev = [None, None]
        self.out_host = [torch.zeros(tmax, dtype=torch.long).pin_memory() for _ in range(2)]
        self.flip = 0
        self.inflight = None  # (event, out_host, [(slot, turn)])
        self.graphs = {}
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        nb = int((free - mem_headroom_gb * 2**30) // self.m.kv_block_bytes())
        self.m.alloc_kv(nb)  # before the capture: the graphs bake in the KV pointers (their pool: headroom)
        self.free_blocks = list(range(nb - 1, 0, -1))  # block 0 = scratch
        self.refcnt = np.zeros(nb, dtype=np.int32)
        if cuda_graph:
            self._capture()
        self.reset_stats()

    # ------------------------------------------------------------------ stats
    def reset_stats(self):
        self.stats = collections.Counter()
        self.trace = []
        self.trace_ev = []  # (start, end) CUDA events of every launched step

    def gpu_times(self):
        """GPU (start, end) of every traced step in seconds, relative to the first step's start."""
        if not self.trace_ev:
            return []
        torch.cuda.synchronize()
        e0 = self.trace_ev[0][0]
        return [(e0.elapsed_time(a) / 1e3, e0.elapsed_time(b) / 1e3) for a, b in self.trace_ev]

    # ------------------------------------------------------------------ KV blocks
    def _alloc(self, s: Session, n_tokens: int, updates: list):
        need = -(-n_tokens // self.BS) - len(s.blocks)
        if need > len(self.free_blocks):
            raise RuntimeError(f"out of KV blocks (need {need}, free {len(self.free_blocks)})")
        for _ in range(need):
            b = self.free_blocks.pop()
            self.refcnt[b] = 1
            updates.append((s.slot * self.max_blocks + len(s.blocks), b))
            s.blocks.append(b)

    def _free(self, s: Session):
        for b in s.blocks:
            self.refcnt[b] -= 1
            if self.refcnt[b] == 0:
                self.free_blocks.append(b)
        s.blocks, s.n_shared = [], 0

    def kv_free_tokens(self) -> int:
        """Free KV tokens net of what running turns may still claim (pending prompt + remaining limit)."""
        claim = sum(-(-(s.n_pos + len(s.todo) + max(0, s.limit - s.gen)) // self.BS) - len(s.blocks)
                    for s in self.sessions.values() if s.phase != IDLE)
        return (len(self.free_blocks) - claim) * self.BS

    def can_admit(self, n_tokens: int, max_new: int = 512) -> bool:
        return self.kv_free_tokens() >= n_tokens + max_new

    def prefill_backlog(self) -> int:
        """Prompt tokens submitted but not yet fed."""
        return sum(len(s.todo) for s in self.sessions.values() if s.phase == PREFILL)

    def room(self, n_tokens: int) -> int:
        return self.max_blocks * self.BS - n_tokens - 1

    # ------------------------------------------------------------------ sessions
    def _reset_state(self, s: Session, ids):
        self._free(s)
        m = self.m
        if self.prefix is not None and ids[:len(self.prefix[0])] == self.prefix[0]:
            ptoks, pblocks, rec, conv = self.prefix
            m.rec[:, s.slot].copy_(rec)
            m.conv[:, s.slot].copy_(conv)
            for i, b in enumerate(pblocks):
                self.refcnt[b] += 1
                s.blocks.append(b)
            m.block_table[s.slot, :len(pblocks)] = torch.tensor(pblocks, dtype=torch.int32).to(self.dev, non_blocking=True)
            s.n_shared = len(pblocks)
            s.hist = list(ptoks)
            self.stats["prefix_hit_tokens"] += len(ptoks)
        else:
            m.rec[:, s.slot].zero_()
            m.conv[:, s.slot].zero_()
            s.hist = []
        s.n_pos = len(s.hist)

    def submit(self, slot: int, input_ids: list[int], max_new: int = 512, force: list[int] | None = None):
        """Starts a turn: the session will feed input_ids (beyond what it already fed) and generate up to max_new.
        force (replay, for timing A/Bs): the turn emits these tokens (then <|im_end|>) instead of the sampled ones,
        so the run follows a recorded trajectory whatever the numerics. The device still feeds the sampled tokens
        (the state holds them: same work per step, wrong values, fine for timing); the host sees the forced ones."""
        s = self.sessions.get(slot)
        if s is None:
            s = self.sessions[slot] = Session(slot)
        assert s.phase == IDLE, f"slot {slot} busy"
        n = len(s.hist)
        if n and n < len(input_ids) and input_ids[:n] == s.hist and s.n_pos == n:
            self.stats["reused_tokens"] += n
        else:
            if n:
                self.stats["state_resets"] += 1
            self._reset_state(s, input_ids)
        s.todo = list(input_ids[len(s.hist):])
        assert s.todo, "empty turn"
        s.phase, s.gen, s.limit, s.turn, s.force = PREFILL, 0, max_new, s.turn + 1, force
        s.t_submit = time.perf_counter()

    def suspend(self, slot: int):
        pass  # state stays in the slot

    def release(self, slot: int):
        s = self.sessions.pop(slot, None)
        if s is not None:
            self._free(s)

    def register_prefix(self, ids: list[int]):
        """Prefills the shared prefix (truncated to whole KV blocks) once; later sessions start from its state."""
        n = len(ids) // self.BS * self.BS
        if n == 0:
            return
        if self.prefix is not None:
            for b in self.prefix[1]:
                self.refcnt[b] -= 1
                if self.refcnt[b] == 0:
                    self.free_blocks.append(b)
            self.prefix = None
        slot = next(i for i in range(self.num_slots) if i not in self.sessions)
        self.submit(slot, ids[:n], 0)
        s = self.sessions[slot]
        while s.phase != IDLE:
            self.step()
        self.drain()
        torch.cuda.synchronize()
        blocks = list(s.blocks)
        for b in blocks:
            self.refcnt[b] += 1
        self.prefix = (list(ids[:n]), blocks, self.m.rec[:, slot].clone(), self.m.conv[:, slot].clone())
        self.release(slot)

    set_shared_prefix = register_prefix

    # ------------------------------------------------------------------ scheduling
    def _layout(self, T, npre, nl=None):
        key = (T, npre, nl)
        lay = self.layouts.get(key)
        if lay is None:
            lay = self.layouts[key] = Layout(T, npre, self.num_slots, nl)
        return lay

    def _schedule(self):
        """Picks the next step's rows: (decode sessions, prefill chunks [(session, tokens, pos0)], block updates)."""
        updates = []
        dec = [s for s in self.sessions.values() if s.phase == DECODE]
        dec.sort(key=lambda s: s.slot)
        for s in dec:
            self._alloc(s, s.n_pos + 1, updates)
        budget = self.TB - len(dec)
        pre = []
        if budget > 0:
            for s in sorted((s for s in self.sessions.values() if s.phase == PREFILL), key=lambda s: s.t_submit):
                c = min(len(s.todo), budget)
                if c <= 0 or len(pre) == self.NPRE:
                    break
                self._alloc(s, s.n_pos + c, updates)
                toks, s.todo = s.todo[:c], s.todo[c:]
                pre.append((s, toks, s.n_pos))
                budget -= c
        return dec, pre, updates

    def _fill(self, a, lay, dec, pre, all_logits=False):
        """Writes the step's metadata (all of the layout, padding included) into the host array a.
        Returns the sampled (slot, turn) list, in sampled-row order. Advances the sessions' positions."""
        v = lay.views(a)
        BS, sc = self.BS, self.scratch
        P = sum(len(t) for _, t, _ in pre)
        D = len(dec)
        T = lay.T
        assert P + D <= T and D <= lay.dmax and len(pre) <= lay.npre
        v["tok"][:] = 0
        v["in_slot"][:] = sc
        v["pos"][:] = 0
        v["slot_map"][:] = 0
        v["row_start"][:] = np.arange(T)
        v["row_seq_slot"][:] = sc
        v["row_slot"][:] = -1
        v["row0"][0] = P
        v["seq_slot"][:] = sc
        v["seq_start"][:] = 0
        v["seq_len"][:] = 0
        v["logit_row"][:] = 0
        v["sample_slot"][:] = sc
        samples = []
        nl = 0
        if lay.npre:
            v["p_slot"][:] = sc
            v["cu"][:] = P
            ci = v["chunk_idx"]
            ci[:, 0], ci[:, 1] = lay.npre, 0
            tl_ = v["tiles"]
            tl_[0], tl_[1], tl_[2], tl_[3] = 0, 0, 0, sc
            r = nc = nt = 0
            tq = K.PREFILL_TQ
            for i, (s, toks, p0) in enumerate(pre):
                c = len(toks)
                pos = np.arange(p0, p0 + c)
                blk = np.asarray(s.blocks, dtype=np.int64)
                rows = slice(r, r + c)
                v["tok"][rows] = toks
                v["pos"][rows] = pos
                v["slot_map"][rows] = blk[pos // BS] * BS + pos % BS
                v["row_start"][rows] = r
                v["row_seq_slot"][rows] = s.slot
                v["seq_slot"][i], v["seq_start"][i], v["seq_len"][i] = s.slot, r, c
                v["p_slot"][i], v["cu"][i] = s.slot, r
                k = -(-c // CHUNK)
                ci[nc:nc + k, 0], ci[nc:nc + k, 1] = i, np.arange(k)
                nc += k
                o = np.arange(0, c, tq)
                tl_[0, nt:nt + len(o)] = r + o
                tl_[1, nt:nt + len(o)] = np.minimum(tq, c - o)
                tl_[2, nt:nt + len(o)] = p0 + o
                tl_[3, nt:nt + len(o)] = s.slot
                nt += len(o)
                s.hist.extend(toks)
                s.n_pos += c
                if not s.todo:  # last chunk: decode from the next step on (limit 0: just feed the tokens)
                    s.phase = DECODE if s.limit > 0 else IDLE
                    if s.phase == DECODE and not all_logits:
                        v["logit_row"][nl], v["sample_slot"][nl] = r + c - 1, s.slot
                        samples.append((s.slot, s.turn))
                        nl += 1
                r += c
            if all_logits:
                v["logit_row"][:P] = np.arange(P)
                nl = P
        if D:
            rows = np.arange(P, P + D)
            slots = np.fromiter((s.slot for s in dec), np.int32, D)
            pos = np.fromiter((s.n_pos for s in dec), np.int64, D)
            v["tok"][rows] = -1
            v["in_slot"][rows] = slots
            v["pos"][rows] = pos
            v["slot_map"][rows] = [s.blocks[p // BS] * BS + p % BS for s, p in zip(dec, pos.tolist())]
            v["row_seq_slot"][rows] = slots
            v["row_slot"][rows] = slots
            j0 = lay.npre
            v["seq_slot"][j0:j0 + D], v["seq_start"][j0:j0 + D], v["seq_len"][j0:j0 + D] = slots, rows, 1
            v["logit_row"][nl:nl + D], v["sample_slot"][nl:nl + D] = rows, slots
            for s in dec:
                samples.append((s.slot, s.turn))
                s.n_pos += 1
        return samples

    # ------------------------------------------------------------------ step
    def step(self) -> tuple[dict[int, int], set[int]]:
        """Launches the next step, then returns (slot -> token sampled by the previous step, slots whose turn ended)."""
        launched = self._launch()
        out, done = self._collect()
        self.inflight = launched
        if launched is not None:
            self.trace[-1] += (time.perf_counter(),)
        return out, done

    def drain(self):
        out, done = self._collect()
        self.inflight = None
        return out, done

    def _apply_updates(self, updates):
        if updates:
            u = torch.tensor(updates, dtype=torch.long).to(self.dev, non_blocking=True)
            self.m.block_table.view(-1)[u[:, 0]] = u[:, 1].int()

    def _launch(self):
        if not any(s.phase in (PREFILL, DECODE) for s in self.sessions.values()):
            return None
        t0 = time.perf_counter()
        dec, pre, updates = self._schedule()
        D, P = len(dec), sum(len(t) for _, t, _ in pre)
        if self.graphs:
            if P:
                lay = self._layout(next(b for b in self.mixed_buckets if b >= P + D), self.NPRE)
            else:
                lay = self._layout(next(b for b in self.buckets if b >= D), 0)
        else:  # eager: exact sizes
            lay = self._layout(P + D, len(pre)) if P else self._layout(D, 0)
        h = self.meta_host[self.flip]
        ev = self.meta_ev[self.flip]
        if ev is not None:
            ev.synchronize()  # the H2D copy that last used this pinned buffer has run
        samples = self._fill(h.numpy(), lay, dec, pre)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        self._apply_updates(updates)
        self.meta_dev[:lay.size].copy_(h[:lay.size], non_blocking=True)
        g = self.graphs.get((lay.T, lay.npre))
        if g is not None:
            g[0].replay()
            nxt = g[1]
            self.stats["graph_steps"] += 1
        else:
            nxt = self.m.forward(lay.step(self.meta_dev))
            self.stats["eager_steps"] += 1
        ev = torch.cuda.Event()
        ev.record()
        self.meta_ev[self.flip] = ev
        oh = self.out_host[self.flip]
        n = len(samples)
        oh[:n].copy_(nxt[:n], non_blocking=True)
        ev2 = torch.cuda.Event(enable_timing=True)
        ev2.record()
        self.trace_ev.append((ev0, ev2))
        self.flip ^= 1
        self.stats["steps"] += 1
        self.stats["decode_rows"] += D
        self.stats["prefill_rows"] += P
        self.stats["padded_rows"] += lay.T - P - D
        self.trace.append((t0, time.perf_counter(), lay.T if lay.npre else -lay.T, D, P))
        return ev2, oh, samples

    def teacher_force(self, ids: list[int], slot: int = 0) -> list[int]:
        """Greedy next token at every position of ids (fed from an empty state as prefill chunks); for tests."""
        assert self.inflight is None
        self.release(slot)
        s = self.sessions[slot] = Session(slot)
        self._reset_state(s, [])
        s.limit = 0
        preds = []
        for c0 in range(0, len(ids), self.TB):
            toks = ids[c0:c0 + self.TB]
            s.todo = ids[c0 + len(toks):]
            s.phase = PREFILL
            updates = []
            self._alloc(s, c0 + len(toks), updates)
            lay = self._layout(len(toks), 1, len(toks))
            h = self.meta_host[0]
            self._fill(h.numpy(), lay, [], [(s, toks, c0)], all_logits=True)
            self._apply_updates(updates)
            self.meta_dev[:lay.size].copy_(h[:lay.size])
            preds += self.m.forward(lay.step(self.meta_dev)).tolist()
        self.release(slot)
        return preds

    def _collect(self):
        out, done = {}, set()
        if self.inflight is None:
            return out, done
        ev, oh, samples = self.inflight
        ev.synchronize()
        toks = oh[:len(samples)].tolist()
        for (slot, turn), t in zip(samples, toks):
            s = self.sessions.get(slot)
            if s is None or s.turn != turn or s.phase != DECODE:
                continue  # turn already ended (this is the sample after its last token)
            if s.force is not None:
                t = s.force[s.gen] if s.gen < len(s.force) else self.im_end
            s.hist.append(t)
            s.gen += 1
            out[slot] = t
            if t in self.stop_ids or s.gen >= s.limit:
                s.phase = IDLE
                done.add(slot)
        return out, done

    # ------------------------------------------------------------------ CUDA graphs
    def _capture(self):
        """One graph per layout bucket (shared memory pool, largest first). Each is warmed up eagerly on a
        realistic dummy step on the scratch slot (Triton autotuning sees real work), then captured."""
        m, sc = self.m, self.scratch
        dummy = Session(sc)
        dummy.blocks = [0] * self.max_blocks
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        lays = [self._layout(T, self.NPRE) for T in sorted(self.mixed_buckets, reverse=True)]
        lays += [self._layout(T, 0) for T in sorted(self.buckets, reverse=True)]
        h = self.meta_host[0]
        with torch.cuda.stream(stream):
            for lay in lays:
                D = min(lay.dmax, lay.T // 2) if lay.npre else lay.T
                P = lay.T - D
                dummy.n_pos, dummy.phase, dummy.todo = 0, PREFILL, []
                pre = [(dummy, [0] * P, 0)] if P else []
                self._fill(h.numpy(), lay, [dummy] * D, pre)
                v = lay.views(h.numpy())
                v["slot_map"][:] = 0  # everything on the scratch block
                v["row_slot"][v["row_slot"] >= 0] = sc
                self.meta_dev[:lay.size].copy_(h[:lay.size])
                b = lay.step(self.meta_dev)
                for _ in range(2):
                    m.forward(b)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    nxt = m.forward(b)
                self.graphs[(lay.T, lay.npre)] = (g, nxt)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        m.rec[:, sc].zero_()
        m.conv[:, sc].zero_()
