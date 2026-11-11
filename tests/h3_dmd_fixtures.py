"""Explicit CPU-only tiny H3 fixtures, never production attention defaults."""
import numpy as np
import torch
from h3.framepack import FramePackH3
from h3.dmd.memory import enable_checkpoint_input_offload

TINY = dict(num_attention_heads=1, attention_head_dim=128, hidden_size=128,
            num_layers=2, num_refiner_layers=1, ffn_dim=256,
            time_embed_hidden_dim=128, time_embed_dim=64)

def tiny(cls, *, offload=False):
    model = cls(**TINY)
    model.install_camera_processors()
    if isinstance(model, FramePackH3):
        model.install_framepack()
    model.set_attention_backend('native')  # Explicit CPU fixture only.
    for name, p in model.named_parameters():
        fp32 = name.startswith(('proj_in.', 'proj_out.', 'audio_proj_in.', 'audio_proj_out.',
                                'time_embedder.', 'history_compressor.'))
        p.data = p.data.to(torch.float32 if fp32 else torch.bfloat16)
    if offload:
        enable_checkpoint_input_offload(model)
    return model

def sample():
    views = torch.eye(4).repeat(17, 1, 1)
    views[:, 0, 3] = torch.linspace(0, 2, 17)
    return dict(target_latents=torch.randn(24, 17, 2, 2).bfloat16(),
                anchor_latents=torch.randn(24, 1, 2, 2).bfloat16(),
                prompt_embeds=torch.randn(5, 5120).bfloat16(),
                text_token_tags=torch.tensor([1, 0, 0, 1, 1]),
                camera_viewmats=views, camera_K=torch.eye(3).repeat(17, 1, 1))

def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right, (left, right)
