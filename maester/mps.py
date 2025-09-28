import torch
from types import SimpleNamespace

device_type = "mps"

# monkey patch torch
torch.cuda.get_device_properties = lambda x: SimpleNamespace(name="mps", total_memory=0)
torch.mps.is_initialized = lambda : True
torch.cuda.reset_peak_memory_stats = lambda : None
torch.cuda.set_device = lambda x: None
torch.cuda.synchronize = torch.mps.synchronize

# monkey patch flex attention
import torch.nn.attention.flex_attention as _fam
_cbm, _fa = _fam.create_block_mask, _fam.flex_attention
_fam.create_block_mask = lambda *a, **kw: kw.setdefault("device", "mps") and _cbm(*a, **kw)
_fam.flex_attention = lambda q, k, v, block_mask=None, **kw: _fa(q.to("cpu"), k.to("cpu"), v.to("cpu"), block_mask=block_mask.to("cpu"), **kw).to("mps")

# monkey patch to allow triton to be imported on MPS
triton, tl = SimpleNamespace(), SimpleNamespace()
triton.autotune = lambda *args, **kwargs: lambda f: f
triton.jit = lambda *args, **kwargs: lambda f: f
triton.Config = lambda *args, **kwargs: None
tl.constexpr = lambda x: x

# monkey patch to enable gpu memory monitor
def build_gpu_memory_monitor():
    from .metrics import GPUMemStats
    gpu_memory_monitor = SimpleNamespace()
    gpu_memory_monitor.device_name = "mps"
    gpu_memory_monitor.device_index = None
    gpu_memory_monitor.device_capacity_gib = 0.0
    gpu_memory_monitor.get_peak_stats = lambda : GPUMemStats(0.0, 0.0, 0.0, 0.0, 0, 0)
    gpu_memory_monitor.reset_peak_stats = lambda : None
    return gpu_memory_monitor
