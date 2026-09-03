"""Regression tests for Wan DMD's classifier-free guidance target."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Keep the regression test runnable directly from a source checkout, where
# pytest's import path may contain ``tests/`` but not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.wan import dmd_core


def test_calc_dmd_loss_anchors_cfg_at_unconditional_prediction(monkeypatch):
    """The real-score x0 target must use the standard CFG interpolation."""

    cond_context = torch.tensor([1.0])
    neg_context = torch.tensor([2.0])
    x0_gen = torch.full((1, 1, 1, 2, 2), 10.0, requires_grad=True)

    # Keep the test deterministic and independent of a model checkpoint.  With
    # sigma=1, returning x_t - target makes velocity_into_x0 recover ``target``.
    monkeypatch.setattr(
        dmd_core,
        "_draw_dmd_sigma",
        lambda args, B, device, dtype, denoised_to=0.0: torch.ones(
            B, device=device, dtype=dtype
        ),
    )

    def fake_velocity(model, x_t, sigma, context, seq_len, **kwargs):
        if model == "real" and context is cond_context:
            target = 2.0
        elif model == "real" and context is neg_context:
            target = 1.0
        else:
            target = 0.0
        return x_t - target

    monkeypatch.setattr(dmd_core, "_dit_velocity_field", fake_velocity)

    args = SimpleNamespace(real_guidance_scale=5.0)
    loss, _ = dmd_core.calc_dmd_loss(
        "real",
        "fake",
        x0_gen,
        cond_context,
        neg_context,
        None,
        args,
        torch.device("cpu"),
    )

    # Standard CFG: 1 + 5 * (2 - 1) = 6.  The old conditional anchor yielded
    # 7, which changes the normalized DMD gradient and this loss (1.125).
    assert loss.item() == pytest.approx(1.125)
