"""
For the compatibility with mmcv 1.7.3 -> https://github.com/rathaROG/mmcv/releases/tag/v1.7.3
"""

import torch
import torch.nn.parallel._functions as torch_parallel_fns
import mmcv.parallel._functions as mmcv_parallel_fns
from mmcv.parallel import distributed as mmcv_dist

# --- Patch #1: Fix _get_stream to accept int device IDs ---
_orig_get_stream = torch_parallel_fns._get_stream

def _get_stream_compat(device):
    if isinstance(device, int):
        device = torch.device("cpu") if device == -1 else torch.device("cuda", device)
    return _orig_get_stream(device)

torch_parallel_fns._get_stream = _get_stream_compat
mmcv_parallel_fns._get_stream = _get_stream_compat


# --- Patch #2: Guard DDP attrs used by _run_ddp_forward ---
def ensure_ddp_attrs(ddp_module):
    if not hasattr(ddp_module, "_use_replicated_tensor_module"):
        ddp_module._use_replicated_tensor_module = False
    if not hasattr(ddp_module, "_replicated_tensor_module"):
        ddp_module._replicated_tensor_module = None
    return ddp_module


# Patch MMDistributedDataParallel.__init__ so every instance gets the attrs
_orig_mmddp_init = mmcv_dist.MMDistributedDataParallel.__init__

def _mmddp_init_compat(self, *args, **kwargs):
    _orig_mmddp_init(self, *args, **kwargs)
    ensure_ddp_attrs(self)

mmcv_dist.MMDistributedDataParallel.__init__ = _mmddp_init_compat


# Optional helper for wrap manually
def wrap_mmddp(model, **ddp_kwargs):
    ddp = mmcv_dist.MMDistributedDataParallel(model, **ddp_kwargs)
    return ensure_ddp_attrs(ddp)
