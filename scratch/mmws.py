"""Skinny-M FP8 GEMM y[M,N] = x[M,K] w[N,K]^T * s[N]: TMA descriptors + warp specialization (+ persistent) in
Triton 3.6 vs cuBLAS / current mm_fp8 (graph loop over weight copies > L2)."""
import sys, os, itertools, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from triton.language.extra.cuda import gdc_wait
from agentopt.v3 import kernels as K
from benchutil import bench_loop
F8 = torch.float8_e4m3fn
triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device='cuda'))


@triton.jit
def _mm_tma(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            STAGES: tl.constexpr, WS: tl.constexpr, SWAP: tl.constexpr):
    gdc_wait()
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    wd = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    if SWAP:
        acc = tl.zeros((BN, BM), tl.float32)
    else:
        acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES, warp_specialize=WS):
        x = xd.load([pid_m * BM, k0])
        w = wd.load([pid_n * BN, k0])
        if SWAP:
            acc = tl.dot(w, tl.trans(x), acc, max_num_imprecise_acc=128)
        else:
            acc = tl.dot(x, tl.trans(w), acc, max_num_imprecise_acc=128)
    n = pid_n * BN + tl.arange(0, BN)
    m = pid_m * BM + tl.arange(0, BM)
    s = tl.load(s_ptr + n, mask=n < N, other=0.0)
    if SWAP:
        out = (acc * s[:, None]).to(tl.bfloat16)
        tl.store(y_ptr + m.to(tl.int64)[None, :] * N + n[:, None], out, mask=(m < M)[None, :] & (n < N)[:, None])
    else:
        out = (acc * s[None, :]).to(tl.bfloat16)
        tl.store(y_ptr + m.to(tl.int64)[:, None] * N + n[None, :], out, mask=(m < M)[:, None] & (n < N)[None, :])


one = torch.ones((), device='cuda')
shapes = [("gate_up", 34816, 5120), ("gdn_in", 16480, 5120), ("down", 5120, 17408), ("out", 5120, 6144)]
Ms = [int(a) for a in sys.argv[1:]] or [32, 64, 128]
for name, N, Kd in shapes:
    ws = [(torch.randn(N, Kd, device='cuda') * .5).to(F8) for _ in range(max(2, int(1.5e9 // (N * Kd))))]
    sv = torch.full((N,), 0.01, device='cuda'); s0 = sv[0]
    floor = N * Kd / 3.1e12
    for M in Ms:
        x = torch.randn(M, Kd, device='cuda').to(F8)
        y = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
        tcb = bench_loop([lambda w=w: torch._scaled_mm(x, w.t(), one, s0, out_dtype=torch.bfloat16) for w in ws], iters=12)
        cur = K.mm_fp8_cfg(M, N)
        tcur = bench_loop([lambda w=w: K.mm_fp8(x, w, sv, cur) for w in ws], iters=12) if cur else float('nan')
        rows = []
        BMs = sorted({min(max(16, triton.next_power_of_2(M)), 128), 64} | ({128} if M > 64 else set()))
        for BM, BN, BK, st, ws_, nw, swap in itertools.product(BMs, (64, 128, 256), (128, 256), (3, 4, 6), (0, 1), (4, 8), (0, 1)):
            if (BN + BM) * BK * st > 200000 or (swap and BM > 64) or (not swap and BM < 64):
                continue
            try:
                t = bench_loop([lambda w=w: _mm_tma[(triton.cdiv(N, BN), triton.cdiv(M, BM))](
                    x, w, sv, y, M, N, K=Kd, BM=BM, BN=BN, BK=BK, STAGES=st, WS=bool(ws_), SWAP=bool(swap),
                    num_warps=nw, launch_pdl=True) for w in ws], iters=12, reps=3)
            except Exception as e:
                continue
            rows.append((t, (BM, BN, BK, st, ws_, nw, swap)))
        rows.sort()
        ref = (x.float() @ ws[0].float().t()) * 0.01
        if rows:
            BM, BN, BK, st, ws_, nw, swap = rows[0][1]
            _mm_tma[(triton.cdiv(N, BN), triton.cdiv(M, BM))](x, ws[0], sv, y, M, N, K=Kd, BM=BM, BN=BN, BK=BK, STAGES=st,
                                                            WS=bool(ws_), SWAP=bool(swap), num_warps=nw)
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
        print(f"{name:8s} M={M:4d}: floor {floor*1e6:5.1f} | cublas {tcb*1e6:6.1f} | cur {tcur*1e6:6.1f} | tma " +
              " ; ".join(f"{t*1e6:6.1f} {c}" for t, c in rows[:3]) + f" err {err:.1e}", flush=True)
