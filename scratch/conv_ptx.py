import sys, torch, triton, collections, re
sys.path.insert(0, '.')
from agentopt.v3 import kernels as K
T, C = 2048, 10240; dev = 'cuda'; bf = torch.bfloat16
x = torch.randn(T, C, device=dev, dtype=bf); xs = torch.rand(T, device=dev); w = torch.randn(C, 4, device=dev)
state = torch.randn(130, 3, C, device=dev, dtype=bf)
rs = torch.zeros(T, dtype=torch.int32, device=dev)
outs = K._conv_outs(T, dev)
ck = K._conv1d_rows_kernel[(T // 16, C // 128)](x, xs, w, state, *outs, rs, rs, T, C, QK=2048, VC=C - 4096, BT=16, BLOCK_C=128, DK=128, num_warps=4)
ptx = ck.asm['ptx']
cnt = collections.Counter(re.findall(r'^\s*(cvt[\w.]*|ex2[\w.]*|div[\w.]*|rcp[\w.]*|ld\.global[\w.]*|fma[\w.]*|mul\.f32|bar[\w.]*|shfl[\w.]*)', ptx, re.M))
for k, v in sorted(cnt.items()): print(k, v)
open('/tmp/conv.ptx', 'w').write(ptx)
sass = ck.asm.get('sass')
if sass:
    c2 = collections.Counter(re.findall(r'\s([A-Z][A-Z0-9_]+)(?:\.[\w.]+)?\s', sass))
    print(sorted(c2.items(), key=lambda a: -a[1])[:40])
