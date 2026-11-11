# SPDX-License-Identifier: Apache-2.0
"""Wan BiWM's AR self-forcing/DMD math adapted to native H3 velocities.

Source: BiWM b032ad5 pipelines/wan/dmd_core.py. No HAND, auxiliary objectives,
new GPU launch or training budget. Generator/score model setup stays separate.
"""
import torch
import torch.nn.functional as F

from h3.layout import unpatchify
from pipelines.common.dmd_algo import run_cfg


def _sigma_view(sigma, value):
    sigma = torch.as_tensor(sigma, device=value.device, dtype=value.dtype)
    if sigma.ndim == 1:
        if len(sigma) != value.shape[0]:
            raise ValueError('Sigma batch differs')
        sigma = sigma.view(-1, *([1] * (value.ndim - 1)))
    if sigma.ndim not in (0, value.ndim):
        raise ValueError('Sigma must be scalar or per-batch')
    return sigma


def velocity_into_x0(velocity, noisy, sigma):
    """H3 is data-minus-noise, opposite to Wan's velocity convention."""
    if velocity.shape != noisy.shape:
        raise ValueError('Velocity/noisy geometry differs')
    return noisy + _sigma_view(sigma, noisy) * velocity


def critic_inputs(generated, noise, sigma):
    """No generator graph in the critic's data/noise target or score input."""
    if generated.shape != noise.shape:
        raise ValueError('Generated/noise geometry differs')
    clean = generated.detach().float()
    noise = noise.detach().float()
    fraction = _sigma_view(sigma, clean)
    return (1 - fraction) * clean + fraction * noise, clean - noise


def critic_loss(velocity, target):
    return F.mse_loss(velocity.float(), target.detach().float())


def generator_loss(generated, real_x0, fake_x0):
    """Same normalized detached DMD target as Wan BiWM, without extra losses."""
    if not generated.shape == real_x0.shape == fake_x0.shape:
        raise ValueError('DMD score/generator geometry differs')
    with torch.no_grad():
        difference = fake_x0.float() - real_x0.float()
        axes = tuple(range(1, generated.ndim))
        normalizer = (generated.float() - real_x0.float()).abs().mean(axes, keepdim=True)
        direction = torch.nan_to_num(difference / (normalizer + 1e-8))
    target = (generated.double() - direction.double()).detach()
    return .5 * F.mse_loss(generated.double(), target), direction


def make_schedule(start_sigmas, shift):
    """Explicit schedule/shift; no hidden phase3 numerical configuration."""
    points = tuple(float(value) for value in start_sigmas) + (0.,)
    if shift <= 0 or points[0] != 1 or points[-2] <= 0:
        raise ValueError('Require positive shift, starts from1, and a final positive start')
    if any(not 0 <= value <= 1 for value in points) or any(a <= b for a, b in zip(points, points[1:])):
        raise ValueError('Sigma schedule must decrease strictly from1 to0')
    return tuple(shift * value / (1 + (shift - 1) * value) for value in points)


def student_rollout(model, sample, schedule, blocks, grad_step, *, rng, train_mode=True,
                    record_replay=False):
    """Generated-only FramePack history; one synchronized exit per native chunk.

    The caller supplies the same schedule, block count and gradient-exit index
    on every FSDP rank. Non-hit maps re-noise their x0 prediction. Between chunks
    the history is detached and quantized to the phase2 cache/history dtype.
    Each chunk gets its own immutable canvas: mutating a shared history buffer
    would invalidate compressor/autograd-checkpoint saved tensors at backward.
    """
    device = sample['target_latents'].device
    channels, total, height, width = sample['target_latents'].shape
    if channels != 24 or blocks < 1 or 7 + 5 * (blocks - 1) > total:
        raise ValueError('Requested chunks exceed native H3 geometry')
    if (len(schedule) < 2 or schedule[0] != 1 or schedule[-1] != 0
        or any(not 0 <= value <= 1 for value in schedule)
        or any(a <= b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError('Require an explicit decreasing schedule from1 to0')
    if train_mode and not 0 <= grad_step < len(schedule) - 1:
        raise ValueError('Gradient exit must be explicit and synchronized')
    def noise(shape):
        return torch.randn(shape, device=device, dtype=torch.float32, generator=rng)
    anchor = .999 * sample['anchor_latents'].detach().float().unsqueeze(0)
    anchor = anchor + .001 * noise(anchor.shape)
    generated, history, replay = [], [], []
    outer_grad = torch.is_grad_enabled()
    for block in range(blocks):
        start, count = (0, 7) if block == 0 else (7 + (block - 1) * 5, 5)
        zeros = sample['target_latents'].new_zeros(channels, total - start, height, width)
        if history:
            past = torch.cat(history, dim=2)[0].to(zeros.dtype)
            canvas = torch.cat((past, zeros), dim=1)
        else:
            canvas = zeros
        current = dict(sample, target_latents=canvas.detach())
        noisy = noise((1, channels, count, height, width))
        for index, (sigma, next_sigma) in enumerate(zip(schedule, schedule[1:])):
            hit = train_mode and index == grad_step
            if hit and record_replay:
                replay.append(dict(start=start, count=count, sigma=sigma,
                    sample={key: value.detach().to('cpu', copy=True) for key, value in current.items()},
                    noisy=noisy.detach().to('cpu', copy=True), anchor=anchor.detach().to('cpu', copy=True),
                    cpu_rng=torch.get_rng_state().clone(),
                    cuda_rng=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None))
            with torch.set_grad_enabled(outer_grad and hit):
                velocity = model(current, noisy, anchor, noisy.new_tensor(1 - sigma), start)
                prediction = velocity_into_x0(unpatchify(velocity, count, height, width), noisy, sigma)
            if hit:
                break
            if index < len(schedule) - 2:
                noisy = (1 - next_sigma) * prediction + next_sigma * noise(prediction.shape)
        generated.append(prediction)
        history.append(prediction.detach())
    meta = dict(
        blocks=blocks, latent_frames=7 + 5 * (blocks - 1),
        grad_step=grad_step if train_mode else -1,
        denoised_to=float(schedule[grad_step + 1]) if train_mode else 0.)
    if record_replay:
        meta['replay_inputs'] = replay
    return torch.cat(generated, dim=2), meta


def replay_student_chunk(model, trace, device):
    """One native gradient-exit map; caller controls RNG/no_sync/backward."""
    sample = {key: value.to(device) for key, value in trace['sample'].items()}
    noisy, anchor = trace['noisy'].to(device), trace['anchor'].to(device)
    sigma, count = trace['sigma'], trace['count']
    velocity = model(sample, noisy, anchor, noisy.new_tensor(1 - sigma), trace['start'])
    return velocity_into_x0(unpatchify(velocity, count, noisy.shape[-2], noisy.shape[-1]), noisy, sigma)
