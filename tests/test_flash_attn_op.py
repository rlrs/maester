from maester.models.flash_attn_op import _flash_attn_forward, _flash_attn_backward
from flash_attn import flash_attn_func
import torch

import pytest
import numpy as np

@pytest.mark.parametrize("batch_size,seqlen,nheads,headdim,dropout_p,causal", [
    (2, 128, 8, 64, 0.0, False),
    (2, 1024, 24, 128, 0.0, True),
])
def test_opcheck(batch_size, seqlen, nheads, headdim, dropout_p, causal):
    q = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    torch.library.opcheck(_flash_attn_forward, (q, k, v, dropout_p, None, causal, False))

@pytest.mark.parametrize("batch_size,seqlen,nheads,headdim,dropout_p,causal", [
    (1, 16, 8, 8, 0.0, False),
    (1, 16, 8, 8, 0.0, True),
])
def test_gradcheck(batch_size, seqlen, nheads, headdim, dropout_p, causal):
    q = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    torch.autograd.gradcheck(_flash_attn_forward, (q, k, v, dropout_p, None, causal, False))

@pytest.mark.parametrize("batch_size,seqlen,nheads,headdim,dropout_p,causal", [
    (2, 128, 8, 64, 0.0, False),
    (2, 1024, 24, 128, 0.0, True),
    (1, 256, 16, 32, 0.01, True),
    (4, 64, 4, 128, 0.05, False),
])
def test_flash_attn_forward(batch_size, seqlen, nheads, headdim, dropout_p, causal):
    torch.manual_seed(42)
    
    q = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16)
    
    softmax_scale = 1.0 / (headdim ** 0.5)
    
    # Ground truth
    out_ref = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    
    # Test implementation
    out, _, _, _, _, _, _, _ = _flash_attn_forward(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    
    assert torch.allclose(out, out_ref, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("batch_size,seqlen,nheads,headdim,dropout_p,causal", [
    (2, 128, 8, 64, 0.0, False),
    (2, 1024, 24, 128, 0.0, True),
    (1, 256, 16, 32, 0.001, True),
    (4, 64, 4, 128, 0.005, False),
])
def test_flash_attn_backward(batch_size, seqlen, nheads, headdim, dropout_p, causal):
    torch.manual_seed(42)
    
    q = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device='cuda', dtype=torch.float16, requires_grad=True)
    
    softmax_scale = 1.0 / (headdim ** 0.5)
    
    # Ground truth
    out_ref = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad, k.grad, v.grad = None, None, None
    
    # Test implementation
    out, q_, k_, v_, out_padded, softmax_lse, _, rng_state = _flash_attn_forward(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    dout = torch.ones_like(out)
    
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(dout, q_, k_, v_, out_padded, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, rng_state)
    
    assert torch.allclose(dq, dq_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dk, dk_ref, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dv, dv_ref, rtol=1e-3, atol=1e-3)


