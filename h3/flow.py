"""The only Stage 1 objective: data-ward video flow matching, shift=3."""
import torch
import torch.nn.functional as F

from .layout import build_layout, patchify, row_timesteps, unpatchify


def predict_velocity(model, sample, noisy, target_t, anchor):
    clean = sample["target_latents"]
    _, frames, height, width = clean.shape
    layout, anchor_rows = build_layout(sample["text_token_tags"], frames, height, width)
    times, inverse = row_timesteps(layout, target_t, anchor_rows)
    frame_ids = torch.cat((torch.zeros(anchor_rows, device=clean.device, dtype=torch.long),
                           torch.arange(frames, device=clean.device).repeat_interleave(anchor_rows)))
    video, _ = model(
        hidden_states=torch.cat((patchify(anchor), patchify(noisy)), dim=1),
        audio_hidden_states=clean.new_empty(1, 0, 32),
        encoder_hidden_states=sample["prompt_embeds"].unsqueeze(0),
        timestep=times, timestep_indices=inverse, **layout,
        cam_viewmats=sample["camera_viewmats"].unsqueeze(0),
        cam_K=sample["camera_K"].unsqueeze(0), camera_frame_ids=frame_ids,
        return_dict=False)
    return video[:, anchor_rows:]


def flow_matching_loss(model, sample, shift=3.0):
    clean = sample["target_latents"].float().unsqueeze(0)
    anchor = sample["anchor_latents"].float().unsqueeze(0)
    noise = torch.randn_like(clean)
    u = torch.rand((), device=clean.device)
    sigma = shift * u / (1 + (shift - 1) * u)
    target_t = 1 - sigma
    anchor = .999 * anchor + .001 * torch.randn_like(anchor)
    prediction = predict_velocity(model, sample, target_t * clean + sigma * noise, target_t, anchor)
    return F.mse_loss(prediction.float(), patchify(clean - noise))


@torch.no_grad()
def sample_video(model, sample, *, steps=30, shift=3.0, seed=42):
    """29 Euler updates for 30 shifted time points, matching stage0.5_v7."""
    if steps < 2:
        raise ValueError("At least two schedule points are required")
    clean = sample["target_latents"]
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    def noise(shape):
        return torch.randn(shape, device=clean.device, dtype=torch.float32, generator=generator)
    latent = noise((1, *clean.shape))
    anchor = .999 * sample["anchor_latents"].float().unsqueeze(0)
    anchor = anchor + .001 * noise(anchor.shape)
    sigmas = torch.linspace(1, 0, steps, device=clean.device)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        prediction = predict_velocity(model, sample, latent, 1 - sigma, anchor)
        latent += (sigma - next_sigma) * unpatchify(prediction, *clean.shape[1:])
    return latent
