"""channels_last helper (Blackwell sm_120)."""
from __future__ import annotations

import torch
import torch.nn as nn


def apply_channels_last(model: nn.Module) -> nn.Module:
    """Move conv-heavy modules to channels_last before the first forward.

    On Blackwell (sm_120) the channels_last memory format lets cuDNN pick the
    NHWC-fused conv kernels, which is the single biggest conv speedup on the
    5090.  Must be applied BEFORE the first forward so the layout sticks.
    """
    return model.to(memory_format=torch.channels_last)
