"""Large-M FP8 GEMM y[M,N] = x[M,K] w[N,K]^T * s[N] (gate_up of a 2048-row mixed step): Triton TMA, persistent
or not, warp-specialized or not, vs cuBLAS. Loop over weight copies (> L2), back to back as in the model.
CUBLASLT_WORKSPACE_SIZE=0 python scratch/mmbig.py [M ...]"""
import sys, itertools, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from benchutil import bench_loop
F8 = torch.float8_e4m3fn
NSM = torch.cuda.get_device_properties(0).multi_processor_count
triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device='cuda'))


@triton.jit
def _tile(t, M, N, BM: tl.constexpr, BN: tl.constexpr, GM: tl.constexpr):
    # grouped order: GM row blocks share the weight columns while they are in L2
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    grp = t // (GM * nn)
    fm = grp * GM
    gs = tl.minimum(nm - fm, GM)
    return fm + (t % (GM * nn)) % gs, (t % (GM * nn)) // gs


@triton.jit
def _mm_big(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            STAGES: tl.constexpr, WS: tl.constexpr, GM: tl.constexpr, NPROG: tl.constexpr, FLAT: tl.constexpr,
            IMP: tl.constexpr):
    xd = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    wd = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    yd = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    ntiles = tl.cdiv(M, BM) * tl.cdiv(N, BN)
    for t in tl.range(tl.program_id(0), ntiles, NPROG, flatten=FLAT, warp_specialize=WS):
        pm, pn = _tile(t, M, N, BM, BN, GM)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            acc = tl.dot(xd.load([pm * BM, k0]), wd.load([pn * BN, k0]).T, acc, max_num_imprecise_acc=IMP)
        s = tl.load(s_ptr + pn * BN + tl.arange(0, BN))
        yd.store([pm * BM, pn * BN], (acc * s[None, :]).to(tl.bfloat16))


if __name__ == "__main__":
    one = torch.ones((), device='cuda')
    Ms = [int(a) for a in sys.argv[1:]] or [2048]
    for name, N, Kd in [("gate_up", 34816, 5120)]:
        ws = [(torch.randn(N, Kd, device='cuda') * .5).to(F8) for _ in range(max(2, int(1.0e9 // (N * Kd))))]
        sv = torch.full((N,), 0.01, device='cuda'); s0 = sv[0]
        for M in Ms:
            x = torch.randn(M, Kd, device='cuda').to(F8)
            y = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
            tcb = bench_loop([lambda w=w: torch._scaled_mm(x, w.t(), one, s0, out_dtype=torch.bfloat16) for w in ws], iters=8)
            fl = 2 * M * N * Kd
            ref = (x[:256].float() @ ws[-1].float().t()) * 0.01
            yc = torch._scaled_mm(x, ws[-1].t(), one, s0, out_dtype=torch.bfloat16)
            ec = ((yc[:256].float() - ref).abs().max() / ref.abs().max()).item()
            print(f"{name} M={M}: cublas {tcb * 1e6:7.1f} us {fl / tcb / 1e12:6.0f} TFLOPS err {ec:.1e}", flush=True)
            rows = []
            for BM, BN, BK, st, ws_, nw, gm, flat, imp in itertools.product((128,), (128, 256), (64, 128), (3, 4), (1,), (8,), (8,), (0,), (128,)):
                if (BN + BM) * BK * st + BM * BN * 2 > 225000 or (ws_ and flat):
                    continue
                nprog = NSM
                try:
                    f = lambda w: _mm_big[(nprog,)](x, w, sv, y, M, N, K=Kd, BM=BM, BN=BN, BK=BK, STAGES=st, WS=bool(ws_),
                                                     GM=gm, NPROG=nprog, FLAT=bool(flat), IMP=imp, num_warps=nw, num_stages=st)
                    t = bench_loop([lambda w=w: f(w) for w in ws], iters=8, reps=3)
                except Exception as e:
                    print("  fail", (BM, BN, BK, st, ws_, nw, gm, flat), str(e)[:120], flush=True)
                    continue
                f(ws[-1])
                err = ((y[:256].float() - ref).abs().max() / ref.abs().max()).item()
                print(f"  {(BM, BN, BK, st, ws_, nw, gm, flat, imp)}: {t * 1e6:7.1f} us {fl / t / 1e12:6.0f} TFLOPS err {err:.1e}", flush=True)
