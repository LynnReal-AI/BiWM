# Copied+slimmed from minWM/HY15 (imports rewritten to hunyuan.* local dependencies; see the hunyuan/ module notes)
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT; see https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE

import torch.nn as nn


def get_activation_layer(act_type):
    """Get activation layer for the given act_type."""
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        # Approximate `tanh` requires torch >= 1.13
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")
