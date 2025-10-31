import torch
import torch.nn.functional as F
from torch import Tensor as TT

import triton
from triton import testing as ttesting, language as tl

# This can probably be improved, especially for small shapes.
def get_syrk_autotune_config(): return [
    triton.Config(
        dict(BLOCK_SIZE_M=m, BLOCK_SIZE_K=k, GROUP_SIZE_M=g),
        num_stages=stages, num_warps=warps,
    )
    for m,k,g,stages,warps in [
        (128, 64, 8, 3, 8),
        (128, 32, 8, 4, 4,),
        (64, 32, 8, 4, 4,),
        (64, 32, 8, 5, 2,),
        (32, 32, 8, 4, 4,),
        (32, 32, 8, 3, 8,),
        (32, 32, 8, 2, 16,),
        (32, 16, 8, 2, 16,),

        # # Good config for fp8 inputs.
        # (256, 128, 8, 3, 8,),
        # (256, 128, 8, 4, 4,),
        # (128, 128, 8, 3, 8,),
        # (128, 128, 8, 4, 4,),
        # (128, 64, 8, 4, 4,),
        # (64, 64, 8, 4, 4,),
        # (64, 128, 8, 4, 4,),
    ]
]
@triton.autotune(
    configs=get_syrk_autotune_config(),
    key=['M', 'K', 'stride_a_m', 'stride_a_k', 'stride_c_m', 'stride_c_n'],
    restore_value=['c_ptr'],
)
@triton.jit
def syrk_kernel(
    a_ptr, c_ptr, # A is M x K, C is M x M, user guarantees K >= M
    M, K,
    # NOTE: kernel generally assumes both A and C are row-major
    stride_a_m, stride_a_k,
    stride_c_m, stride_c_n,
    # typical mm kernel meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # scalar factors for the update
    alpha: tl.constexpr, beta: tl.constexpr,
    # if disabled, we compute A@A instead, assuming A is symmetric.
    trans_r: tl.constexpr,
):
    # In SYRK, C is MxM. We conceptually tile C into square blocks of size BLOCK_SIZE_M.
    pid = tl.program_id(0)
    programs_per_dim = tl.cdiv(M, BLOCK_SIZE_M)

    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    groupsize = GROUP_SIZE_M * programs_per_dim
    gid = pid // groupsize
    lid = pid % groupsize
    groupR0_pid_m = gid * GROUP_SIZE_M
    group_size_m = min(programs_per_dim - groupR0_pid_m, GROUP_SIZE_M)
    pid_m = groupR0_pid_m + (lid % group_size_m)
    pid_n = (lid // group_size_m)

    # Skip blocks in the upper triangle: we only compute if row >= col.
    if pid_m < pid_n: return

    # block pointers of dubious correctness (numerically correct)
    L_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M,K),
        strides=(stride_a_m, stride_a_k),
        offsets=(pid_m*BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    R_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(K,M),
        strides=(stride_a_k, stride_a_m) if trans_r else (stride_a_m, stride_a_k),
        offsets=(0, pid_n*BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
        order=(0, 1) if trans_r else (1, 0),
    )
    C_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M,M),
        strides=(stride_c_m, stride_c_n),
        offsets=(pid_m*BLOCK_SIZE_M, pid_n*BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_M),
        order=(1, 0)
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    # Loop over K dimension in tiles.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        l = tl.load(L_block_ptr,padding_option="zero", boundary_check=(0,1))
        r = tl.load(R_block_ptr,padding_option="zero", boundary_check=(0,1))

        # TODO: determine the point at which fast accum becomes inaccurate.
        acc = tl.dot(l, r, acc)
        # acc += tl.dot(l,r) if pid_m != pid_n else tl.dot(l.to(tl.float16),r.to(tl.float16))

        L_block_ptr = tl.advance(L_block_ptr, (0, BLOCK_SIZE_K))
        R_block_ptr = tl.advance(R_block_ptr, (BLOCK_SIZE_K, 0))

    # alpha/beta epilogue
    acc = alpha * acc
    if beta != 0: # load block from C and calculate acc += beta*C
        acc = tl.fma(beta, tl.load(C_block_ptr, padding_option="zero", boundary_check=(0,1)).to(tl.float32), acc)

    # save the computed block
    tl.store(C_block_ptr, acc.to(c_ptr.dtype.element_ty), boundary_check=(0,1))

    # For off-diagonal blocks (i.e. pid_m != pid_n), mirror the result to the symmetric location.
    if pid_m > pid_n:
        # The mirror block is located at (pid_n, pid_m), and we are currently at (pid_m, pid_n), so:
        jump_m = (pid_n-pid_m) * BLOCK_SIZE_M
        jump_n = (pid_m-pid_n) * BLOCK_SIZE_M
        C_block_ptr = tl.advance(C_block_ptr, (jump_m, jump_n))
        tl.store(C_block_ptr, tl.trans(acc).to(c_ptr.dtype.element_ty), boundary_check=(0,1))

@torch.compiler.disable()
def syrk_launch(a: TT, c: TT, α: float, β: float, trans_r: bool) -> TT:
    assert a.ndim == 2
    M,K = a.shape[-2:]
    # TODO: fix this causing compile perf issues
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(M, meta['BLOCK_SIZE_M']), 1,1)
    syrk_kernel[grid](
        a, c,
        M, K,
        a.stride(-2), a.stride(-1),
        c.stride(-2), c.stride(-1),
        alpha=α, beta=β,
        trans_r=trans_r,
    )
    return c

def syrk(A: TT, α: float=1., β: float=0., out: TT | None=None, trans_r: bool=True):
    ''' SYRK(...) computes:
      * out = α*A@A^T + β*out, if trans_r is True
      * out = α*A@A   + β*out, if trans_r is False (where we assume A is symmetric)
    If out is not provided and β=0, `out` will be dynamically allocated.
    '''
    # Validate input A
    assert A.is_floating_point() and A.size(-1) >= A.size(-2)
    if not trans_r:
        assert A.size(-2) == A.size(-1), "A must be a symmetric matrix."
        assert A.dtype == torch.bfloat16, "A@A will overflow fp16 (or fp8)."
    assert A.is_contiguous(), "A must be contiguous."

    # Validate/create out
    if out is None:
        assert β == 0, "if beta is nonzero, out must be provided."
        out = torch.empty(*A.shape[:-1], A.size(-2), device=A.device, dtype=A.dtype)
    assert A.size(-2) == out.size(-2) == out.size(-1) and out.is_contiguous()

    # A@A.mT will have high magnitudes along the diagonal, so fp8 output should never be used.
    if out.dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        raise ValueError("fp8 syrk output will always be inaccurate along the diagonal.")
    
    # TODO: we don't have a batched kernel so we just loop over batch dim if need be
    for a,c in zip(A.view(-1, *A.shape[-2:]), out.view(-1, *out.shape[-2:])):
        syrk_launch(a, c, α, β, trans_r)
    return out

def unscaled_mm_fp16out(a,b,_scale=torch.tensor(1.0, device='cuda')):
    return torch._scaled_mm(a,b,_scale,_scale,out_dtype=torch.float16)

def test_syrk(dtype=torch.float16):
    with torch.random.fork_rng(['cuda']):
        torch.manual_seed(0)
        D = 512
        a = torch.randn(D,D,dtype=torch.float16).to(dtype)
        # always use fp16 as numerical basis of comparison
        torch_output = F.linear(a.half(), a.half())
        triton_output = syrk(a).to(torch.float16)
        print(triton_output)
        print(torch_output)
        print(triton_output - torch_output) # for fp16 D=512, this is all_zeros.
        # TODO: I noticed that for fp8, the diagonal blocks are off by 1e-2 instead of ~1e-4:
        pooled_pt = F.avg_pool2d(torch_output[None,None], kernel_size=64, stride=64)
        pooled_tr = F.avg_pool2d(triton_output[None,None],kernel_size=64, stride=64)
        print((pooled_pt-pooled_tr).v)
        # further check A@A.
        torch_output = torch_output.bfloat16() # <-- will inf if not
        symm_pt = torch_output@torch_output
        symm_tr = syrk(torch_output, trans_r=False)
        print(symm_pt - symm_tr)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M","K"],
        x_vals=[(256*i, 256*i) for i in range(1,17)],
        ylabel="Hypothetical dense TFLOPs",
        line_arg="provider_dtype",
        # line_vals=['triton-fp8','triton-fp16','eager-fp8','eager-fp16'],
        # line_names=['Triton FP8','Triton FP16','Eager FP8','Eager FP16'],
        line_vals=['triton-fp16','eager-fp16'],
        line_names=['Triton FP16','Eager FP16'],
        # line_vals=['triton-fp8','eager-fp8'],
        # line_names=['Triton FP8','Eager FP8'],
        styles=[('green','-'),('green',':'),('red','-'),('red',':')],
        plot_name="A@A.mT perf",
        args={},
    )
)
def benchmark(M: int, K: int, provider_dtype: str, *, trans_r: bool=True):
    provider,dtype = provider_dtype.split('-')
    dtype = dict(fp8=torch.float8_e4m3fn, fp16=torch.float16)[dtype]
    a = torch.randn(M,K,dtype=torch.float16).to(dtype)
    a /= a.norm(dim=(-2, -1), keepdim=True) + 1e-7

    if trans_r:
        f_pt = (lambda: unscaled_mm_fp16out(a,a.mT)) if dtype==torch.float8_e4m3fn else lambda: F.linear(a,a)
        f = f_pt if provider == 'eager' else lambda: syrk(a)
        perf = lambda m: 2*M*M*K*1e-12 / (m*1e-3)
    else:
        a = F.linear(a,a).bfloat16().mT
        f_pt = lambda: torch.matmul(a,a) + 1*torch.zeros(*a.shape)
        f = f_pt if provider == 'eager' else lambda: syrk(a.contiguous(), β=1.0, out=torch.zeros(*a.shape), trans_r=False)
        perf = lambda m: 2*M*M*M*1e-12 / (m*1e-3)

    ms, min_ms, max_ms = triton.testing.do_bench(f, quantiles=[0.5,0.2,0.8])

    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == "__main__":
    __import__("lovely_tensors").monkey_patch()
    with torch.device('cuda'):
        benchmark.run(show_plots=True, print_data=True)

        # Run original tests
        torch.set_printoptions(linewidth=200)
        test_syrk(torch.float16)
        test_syrk(torch.float8_e4m3fn)