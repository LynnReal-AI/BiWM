"""Fail-closed FlashAttention-3 policy for H3 training entry points."""

from importlib.metadata import PackageNotFoundError, version

import torch


REQUIRED_H3_TRAINING_ATTENTION_BACKEND = "_flash_3"
REQUIRED_FA3_VERSION = "3.0.0"


def require_h3_fa3_config(requested: str) -> str:
    """Reject a production H3 training configuration that is not explicit FA3."""
    if requested != REQUIRED_H3_TRAINING_ATTENTION_BACKEND:
        raise RuntimeError(
            "H3 fine-tuning requires --attention_backend _flash_3; "
            f"got {requested!r}. Non-FA3 backends are permitted only by "
            "train.py's no-save --attention_benchmark_only diagnostic contract."
        )
    return REQUIRED_H3_TRAINING_ATTENTION_BACKEND


def require_h3_fa3_training_backend(requested: str, device: torch.device) -> str:
    """Validate the mandatory H3 fine-tuning backend and installed kernel target."""
    require_h3_fa3_config(requested)
    if device.type != "cuda":
        raise RuntimeError("H3 FA3 fine-tuning requires a CUDA Hopper device")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(
            "the installed H3 FA3 wheel targets sm90a (H100/H200); "
            f"got capability {torch.cuda.get_device_capability(device)}"
        )
    try:
        installed_version = version("flash-attn-3")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "flash-attn-3 is not installed; run scripts/fa3_build_install.sh"
        ) from error
    if installed_version != REQUIRED_FA3_VERSION:
        raise RuntimeError(
            f"H3 fine-tuning requires flash-attn-3=={REQUIRED_FA3_VERSION}; "
            f"got {installed_version}"
        )
    from diffusers.utils import is_flash_attn_3_available

    if not is_flash_attn_3_available():
        raise RuntimeError(
            "Diffusers cannot import flash_attn_3; run scripts/fa3_build_install.sh"
        )
    return REQUIRED_H3_TRAINING_ATTENTION_BACKEND
