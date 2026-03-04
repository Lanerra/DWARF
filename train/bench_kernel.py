"""
Kernel benchmark: DSQGAttentionV3 vs PyTorch scaled_dot_product_attention.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/bench_kernel.py
"""
import sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, 'kernels')
from dsqg_attention_v3 import dsqg_attention_v3

DEVICE  = 'cuda'
N_ITERS = 50
N_WARM  = 10

def sync(): torch.cuda.synchronize()

def bench(fn):
    for _ in range(N_WARM): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(N_ITERS): fn()
    sync()
    return (time.perf_counter() - t0) / N_ITERS * 1e3

def flops_sdpa(B, H, N, HD):
    return B * H * 4 * N * N * HD

def flops_dsqg(B, H, N, HD, J=44):
    return B * H * N * J * 4 * HD

configs = [
    (1, 8,  512, 32),
    (1, 8, 1024, 32),
    (4, 8, 1024, 32),
    (8, 8, 2048, 32),
    (1, 8, 4096, 32),
    (1, 8, 8192, 32),
    (1, 8,16384, 32),
]

print('=' * 96)
print(f'{"Config":28s}  {"SDPA fwd":>9} {"DSQG fwd":>9} {"fwd×":>6}  '
      f'{"SDPA f+b":>9} {"DSQG f+b":>9} {"f+b×":>6}  {"FLOP×":>6}')
print('-' * 96)

for (B, H, N, HD) in configs:
    tag = f'B={B} H={H} N={N:>5} HD={HD}'
    try:
        def mk():
            q = torch.randn(B,H,N,HD,device=DEVICE,dtype=torch.bfloat16)
            k = torch.randn(B,H,N,HD,device=DEVICE,dtype=torch.bfloat16)
            v = torch.randn(B,H,N,HD,device=DEVICE,dtype=torch.bfloat16)
            pb= torch.zeros(44,H, device=DEVICE,dtype=torch.float32)
            se= torch.zeros(44,HD,device=DEVICE,dtype=torch.float32)
            return q,k,v,pb,se
        q,k,v,pb,se = mk()

        tf_s = bench(lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True))
        tf_d = bench(lambda: dsqg_attention_v3(q,k,v,pb,se))

        # fwd+bwd
        def fb_sdpa():
            qr=q.detach().requires_grad_(True)
            kr=k.detach().requires_grad_(True)
            vr=v.detach().requires_grad_(True)
            F.scaled_dot_product_attention(qr,kr,vr,is_causal=True).sum().backward()
        def fb_dsqg():
            qr=q.detach().requires_grad_(True)
            kr=k.detach().requires_grad_(True)
            vr=v.detach().requires_grad_(True)
            dsqg_attention_v3(qr,kr,vr,pb,se).sum().backward()

        tb_s = bench(fb_sdpa)
        tb_d = bench(fb_dsqg)

        fr = flops_sdpa(B,H,N,HD) / flops_dsqg(B,H,N,HD)
        print(f'{tag:28s}  {tf_s:>8.2f}ms {tf_d:>8.2f}ms {tf_s/tf_d:>5.1f}x  '
              f'{tb_s:>8.2f}ms {tb_d:>8.2f}ms {tb_s/tb_d:>5.1f}x  {fr:>5.0f}x')
    except Exception as e:
        print(f'{tag:28s}  ERROR: {e}')

print('=' * 96)
print('FLOP× = theoretical ops ratio (SDPA/DSQG). speedup× = actual wall time.')
print('Gap between FLOP× and speedup× = memory-bound overhead in DSQG kernel.')
