import torch
from torch import Tensor
from flash_attn_2_cuda import fwd, bwd

@torch.library.custom_op("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(q: Tensor, 
                        k: Tensor, 
                        v: Tensor, 
                        dropout_p: float = 0.0, 
                        softmax_scale: float | None = None,
                        causal: bool = False, 
                        return_softmax: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = fwd(
        q, k, v, None, dropout_p, softmax_scale, causal, return_softmax, None
    )
    if S_dmask is None:
        batch_size, seqlen, nheads, headdim = q.shape
        S_dmask = torch.empty(batch_size, nheads, seqlen, seqlen, dtype=q.dtype, device=q.device)
    return out.clone(), q.clone(), k.clone(), v.clone(), out_padded.clone(), softmax_lse.clone(), S_dmask, rng_state

@_flash_attn_forward.register_fake
def _(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, return_softmax=False):
    batch_size, seqlen, nheads, headdim = q.shape
    return (torch.empty_like(q), 
            torch.empty_like(q), 
            torch.empty_like(k), 
            torch.empty_like(v), 
            torch.empty(batch_size, seqlen, nheads, headdim, dtype=q.dtype, device=q.device), # out_padded
            torch.empty(batch_size, nheads, seqlen, dtype=torch.float32, device=q.device), # softmax_lse
            torch.empty(batch_size, nheads, seqlen, seqlen, dtype=q.dtype, device=q.device), # S_dmask
            torch.empty(batch_size, dtype=torch.int64, device=q.device) # rng_state
            )

@torch.library.custom_op("flash_attn::_flash_attn_backward", mutates_args=(), device_types="cuda")
def _flash_attn_backward(dout: Tensor, 
                         q: Tensor, 
                         k: Tensor, 
                         v: Tensor, 
                         out: Tensor, 
                         softmax_lse: Tensor, 
                         dq: Tensor, 
                         dk: Tensor, 
                         dv: Tensor,
                         dropout_p: float = 0.0, 
                         softmax_scale: float | None = None, 
                         causal: bool = False, 
                         rng_state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, = bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p,
        softmax_scale, causal, None, rng_state
    )
    return dq.clone(), dk.clone(), dv.clone(), softmax_d

@_flash_attn_backward.register_fake
def _(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, rng_state):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(softmax_lse),
    )


def flash_attn_func(q: Tensor, k: Tensor, v: Tensor, 
                    dropout_p: float = 0.0, softmax_scale: float | None = None, 
                    causal: bool = False, return_softmax: bool = False) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_softmax=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
        q, k, v, dropout_p, softmax_scale, causal=causal,
        return_softmax=return_softmax and dropout_p > 0
    )
    return out if not return_softmax else (out, softmax_lse, S_dmask)

def _backward(ctx, dout, *args):
    q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv, ctx.dropout_p, ctx.softmax_scale, ctx.causal,
        rng_state=rng_state
    )
    dq = dq[..., :dout.shape[-1]]  # We could have padded the head dimension
    dk = dk[..., :dout.shape[-1]]
    dv = dv[..., :dout.shape[-1]]
    return dq, dk, dv, None, None, None, None, None, None, None, None

def setup_context(ctx, inputs, output):
    q, k, v, dropout_p, softmax_scale, causal, return_softmax = inputs
    _, _, _, _, out_padded, softmax_lse, S_dmask, rng_state = output
    ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
    ctx.dropout_p = dropout_p
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal

_flash_attn_forward.register_autograd(_backward, setup_context=setup_context)