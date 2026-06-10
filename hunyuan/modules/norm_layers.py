# Copied+slimmed from minWM/HY15 (imports rewritten to hunyuan.* local dependencies; see the hunyuan/ module documentation)
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT; see https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """Initialize the RMSNorm normalization layer."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """Apply the RMSNorm normalization to the input tensor."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def reset_parameters(self):
        if hasattr(self, "weight"):
            self.weight.fill_(1)

    def forward(self, x):
        """Forward pass through the RMSNorm layer."""
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """Get the normalization layer for the given norm_layer name."""
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")
