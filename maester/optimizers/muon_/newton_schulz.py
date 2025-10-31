import torch
import triton
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
from triton.testing import do_bench

from maester.optimizers.muon_.syrk import syrk

@torch.compile(fullgraph=True)
def nsloop_torch(X: Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    '''
    When compiled down, inductor produces the following steps:
    1. A = matmul X with reinterpret_tensor(X)
    2. (triton) read A -> write b*A and c*A
    3. B = addmm(b*A, c*A, A)
    4. (triton) read X -> write a*X (this is stupid)
    5. X = addmm(a*X, B, X)
    '''
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

@torch.compile(dynamic=True)
def nsloop_triton_lazy(X: Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    '''
    1. do X@X.mT. A is newly allocated.
    2. do b*A.clone() + c*A@A. B points to the memory allocated by A.clone().
    3. return newly allocated a*X + B@X.
    '''
    X = X.contiguous()
    f = torch.baddbmm if X.ndim > 2 else torch.addmm
    for _ in range(steps):
        A = syrk(X)
        B = f(A, A, A, beta=b, alpha=c) # <--- will perform better on small shapes
        # B = b*A + syrk(A, c, trans_r=False) # <--- directly match torch impl
        # B = syrk(A, c, 1, out=b*A, trans_r=False) # <--- avoid fma differences
        # B = syrk(A, c, b, out=A.clone(), trans_r=False) # <--- fastest implementation
        X = f(X, B, X, beta=a, alpha=1)
    return X

@torch.compile(dynamic=True)
def nsloop_triton_inplace(X: Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    X = X.contiguous()
    f = X.baddbmm_ if X.ndim > 2 else X.addmm_
    # I believe it is not possible to do this without at least 3 temp buffers:
    A = torch.empty(*X.shape[:-1], X.size(-2), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    x = torch.empty_like(X)

    for _ in range(steps):
        syrk(X, out=A) # will overwrite A
        syrk(A, c, b, out=B.copy_(A), trans_r=False) # will read and writeadd to B
        f(B, x.copy_(X), beta=a, alpha=1)
    return X

def zeropower_via_newtonschulz(G, steps=10, eps=1e-7, f_iter=nsloop_torch):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
        # DTensor will NaN for sharded compute on Shard(1)
        if isinstance(X, DTensor):
            p = [Shard(0) if isinstance(p, Shard) else p for p in X._spec.placements]
            X = X.redistribute(placements=p)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps) # ensure top singular value <= 1
    X = f_iter(X,steps)
    return X if G.size(-2) <= G.size(-1) else X.mT

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M","K"],
        x_vals=[256*i for i in range(1,33)],
        ylabel="TFLOP/s (pegged to eager case)",
        line_arg="provider",
        line_vals=['triton-lazy','eager-0','triton-inplace'],
        line_names=['Triton (lazy)','Naive (compiled)','Triton (inplace)'],
        styles=[('green','-'),('blue','-'),('green',':')],
        plot_name="Newton-Schulz (10 iters)",
        args={},
    )
)
def benchmark(M: int, K: int, provider: str, *, steps=10):
    a = torch.randn(M,K,dtype=torch.bfloat16,device='cuda')
    mm_cost = (M*K*M) + (M*M*M) + (M*M*K)
    perf = lambda m: steps*2*mm_cost*1e-12 / (m*1e-3)
    match provider:
        case 'eager-0': f_iter = nsloop_torch
        case 'triton-lazy': f_iter = nsloop_triton_lazy
        case 'triton-inplace': f_iter = nsloop_triton_inplace

    f = lambda: zeropower_via_newtonschulz(a, steps, f_iter=f_iter)
    ms, min_ms, max_ms = triton.testing.do_bench(f, quantiles=[0.5,0.2,0.8])
    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == '__main__':
    benchmark.run(show_plots=False, save_path='./matplotlib-newton_schulz', print_data=True)

    torch.manual_seed(0)
    __import__('lovely_tensors').monkey_patch()
    TEST_SHAPES = [
        (4096,4096),   # typical 7b w_q
        (16384,6144),  # mistral 22b (transposed)
        (1294,929),    # malform unpadded tensor where full mm perf > syrk perf
        (4,512,512),   # worst possible situation (batched and very small)
    ]
    for s in TEST_SHAPES:
        with torch.device('cuda'): G = torch.randn(s, dtype=torch.bfloat16)

        # Print raw results in grey
        print("\033[90m",end='')
        print(o_torch := zeropower_via_newtonschulz(G, f_iter=nsloop_torch))
        print(o_tritL := zeropower_via_newtonschulz(G, f_iter=nsloop_triton_lazy))
        print(o_tritI := zeropower_via_newtonschulz(G, f_iter=nsloop_triton_inplace))
        print("\033[0m",end='')

        # print divergence with torch (generally OK)
        print(f'{o_torch-o_tritL=}')
        print(f'{o_torch-o_tritI=}')

        # print speed differences
        print('\ttorch (eager):\t', t_torch := do_bench(lambda: zeropower_via_newtonschulz(G, f_iter=nsloop_torch)))
        print('\ttriton (lazy):\t', t_tritL := do_bench(lambda: zeropower_via_newtonschulz(G, f_iter=nsloop_triton_lazy)))
        print('\ttritoninplace:\t', t_tritI := do_bench(lambda: zeropower_via_newtonschulz(G, f_iter=nsloop_triton_inplace)))
        speedratio = max(t_torch/t_tritL, t_torch/t_tritI) # best of lazy & inplace
        print("\t\033[91m" if speedratio < 1.0 else "\t\033[92m", f'>>>{speedratio=:.2f}<<<', "\033[0m")