# SPDX-License-Identifier: Apache-2.0
"""Native bidirectional H3 score; no FramePack history or GT target values.

Conditioned and captionless/no-camera calls share weights, noisy video, anchor,
and H3 time. This preserves the BiWM joint text+camera CFG direction while
retaining H3's independent first-image conditioning on both branches.
"""
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

from h3.layout import build_layout, patchify, row_timesteps, unpatchify
from h3.model import BiWMH3Transformer, CameraControl, get_dtype

from .core import run_cfg, velocity_into_x0


def image_prefix(sample):
    """Keep the audited causal Qwen first-image prefix, remove later caption.

    Same prefix rule as the native SolarWM captionless layout. This does not
    re-encode a negative caption or invent/zero image embeddings. Only use with
    first-target-image-only caches, never reference-video/HAND Qwen caches.
    """
    embeds, tags = sample['prompt_embeds'], sample['text_token_tags']
    if embeds.ndim != 2 or embeds.shape != (len(tags), 5120) or tags.ndim != 1:
        raise ValueError('Native Qwen embedding/tag geometry differs')
    if not bool(((tags == 0) | (tags == 1)).all()):
        raise ValueError('Video/audio or unknown Qwen modality in camera-only cache')
    vision = (tags == 0).nonzero().flatten()
    if not len(vision) or not bool((vision[1:] - vision[:-1] == 1).all()):
        raise ValueError('Require exactly one contiguous first-image vision block')
    begin, end = int(vision[0]), int(vision[-1]) + 1
    if not begin or not bool((tags[:begin] == 1).all()) or not bool((tags[end:] == 1).all()):
        raise ValueError('Expected native first-image prefix followed by caption')
    return dict(sample, prompt_embeds=embeds[:end], text_token_tags=tags[:end])


class FullScoreH3(BiWMH3Transformer):
    def forward(self, sample, noisy, anchor, target_t, *, camera_conditioned=True):
        """Full noisy video + Qwen + anchor; never read target/history values."""
        batch, channels, frames, height, width = noisy.shape
        if batch != 1 or channels != 24 or anchor.shape != (1, 24, 1, height, width):
            raise ValueError('Native H3 score expects one aligned video/anchor')
        if sample['camera_viewmats'].shape[0] < frames or sample['camera_K'].shape[0] < frames:
            raise ValueError('Full score cannot outlive its physical camera trajectory')
        layout, rows = build_layout(sample['text_token_tags'], frames, height, width)
        times, inverse = row_timesteps(layout, target_t, rows)
        video_indices, text_indices = layout['video_indices'], layout['text_indices']
        frame_ids = torch.cat((torch.zeros(rows, dtype=torch.long, device=noisy.device),
                               torch.arange(frames, device=noisy.device).repeat_interleave(rows)))
        control = None
        if camera_conditioned:
            control = CameraControl(video_indices,
                                    sample['camera_viewmats'].unsqueeze(0).index_select(1, frame_ids),
                                    sample['camera_K'].unsqueeze(0).index_select(1, frame_ids))
        video = self.proj_in(torch.cat((patchify(anchor), patchify(noisy)), 1).to(get_dtype(self.proj_in)))
        text = self.token_refiner(self.context_embedder(
            sample['prompt_embeds'].unsqueeze(0).to(get_dtype(self.context_embedder))))
        packed = text.new_zeros(batch, len(layout['position_ids']), text.shape[-1])
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        temb = self.time_embedder(self.time_proj(times).to(get_dtype(self.time_embedder)))
        adaln = inverse * MINIMAX_H3_MODALITY_NUM + layout['token_tags']
        rotary = self.rope(layout['position_ids'])
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed = self._gradient_checkpointing_func(block, packed, temb, adaln, rotary, control)
            else:
                packed = block(packed, temb, adaln, rotary, control)
        packed = self.norm_out(packed, temb, inverse).to(get_dtype(self.proj_out))
        video = self.proj_out(packed).index_select(1, video_indices)[:, rows:]
        return unpatchify(video, frames, height, width)


@torch.no_grad()
def real_and_fake_x0(teacher, critic, sample, noisy, sigma, anchor, guidance_scale):
    """Detached full-video scores on the exact same noisy data and anchor."""
    if torch.as_tensor(sigma).numel() != 1:
        raise ValueError('Native single-video score requires scalar sigma')
    time = 1 - torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32).reshape(())
    conditional = teacher(sample, noisy, anchor, time, camera_conditioned=True)
    unconditional = teacher(image_prefix(sample), noisy, anchor, time, camera_conditioned=False)
    real = run_cfg(velocity_into_x0(conditional, noisy, sigma),
                   velocity_into_x0(unconditional, noisy, sigma), guidance_scale)
    fake = velocity_into_x0(critic(sample, noisy, anchor, time, camera_conditioned=True), noisy, sigma)
    return real, fake


@torch.no_grad()
def sample_score(model, sample, *, points, shift, guidance_scale, seed):
    """Dense native full-video teacher baseline; shape-only GT, no history.

    CFG1 has the exact noise order/Euler arithmetic of h3.flow.sample_video.
    Other CFG scales retain first-image/anchor in both text+camera branches.
    """
    if points < 2 or shift <= 0:
        raise ValueError('Explicit dense schedule points/positive shift required')
    target = sample['target_latents']
    rng = torch.Generator(device=target.device).manual_seed(seed)
    def noise(shape):
        return torch.randn(shape, device=target.device, dtype=torch.float32, generator=rng)
    noisy = noise((1, *target.shape))
    anchor = .999 * sample['anchor_latents'].float().unsqueeze(0)
    anchor = anchor + .001 * noise(anchor.shape)
    sigmas = torch.linspace(1, 0, points, device=target.device)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    negative = image_prefix(sample) if guidance_scale != 1 else None
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        velocity = model(sample, noisy, anchor, 1 - sigma, camera_conditioned=True)
        if negative is not None:
            unconditional = model(negative, noisy, anchor, 1 - sigma, camera_conditioned=False)
            velocity = run_cfg(velocity, unconditional, guidance_scale)
        noisy += (sigma - next_sigma) * velocity
    if not torch.isfinite(noisy).all():
        raise RuntimeError('Nonfinite native full-video teacher sample')
    return noisy
