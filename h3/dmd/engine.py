# SPDX-License-Identifier: Apache-2.0
"""Real 5C/1G updates shared by the H200 trainer and tiny CPU numerical gate."""
from dataclasses import asdict
from contextlib import nullcontext
import time

import torch
import torch.distributed as dist

from h3.distributed import clip_sharded_grad_norm
from .core import critic_inputs, critic_loss, generator_loss, replay_student_chunk, student_rollout
from .score import real_and_fake_x0
from .memory import move_fp32_optimizer_state


class Engine:
    def __init__(self, generator, teacher, critic, gopt, copt, counters, sample,
                 schedule, config, device, streams):
        self.generator, self.teacher, self.critic = generator, teacher, critic
        self.gopt, self.copt, self.counters = gopt, copt, counters
        self.sample, self.schedule, self.config = sample, schedule, config
        self.device, self.streams = device, streams
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('Real teacher must be frozen before constructing the engine')
        identities = [{id(p) for p in model.parameters()} for model in (generator, teacher, critic)]
        if any(identities[i] & identities[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('DMD model roles must not share parameter objects')
        if counters.effective_batch % self.world:
            raise ValueError('Effective batch must divide into actual DP world size')
        self.accumulation = counters.effective_batch // self.world
        self.generator.train()
        self.critic.train()
        self.teacher.eval()

    def exit_step(self):
        choice = torch.randint(len(self.schedule) - 1, (), device=self.device,
                               generator=self.streams['control'])
        dist.broadcast(choice, src=0)
        return int(choice)

    def score_sigma(self, denoised_to):
        """BiWM draw: U -> explicit shift -> clamp, optional exit-time floor."""
        cfg = self.config['score_sigma']
        lower = max(cfg['min'], denoised_to) if cfg['ts_schedule'] else cfg['min']
        if not 0 <= lower <= cfg['max'] <= 1 or cfg['shift'] <= 0:
            raise ValueError('Invalid explicit score sigma bounds/exit floor')
        sigma = torch.rand((), device=self.device, generator=self.streams['control'])
        sigma = cfg['shift'] * sigma / (1 + (cfg['shift'] - 1) * sigma)
        sigma = sigma.clamp(lower, cfg['max'])
        dist.broadcast(sigma, src=0)
        return sigma

    def noise(self, value, role='score'):
        return torch.randn(value.shape, device=value.device, dtype=torch.float32,
                           generator=self.streams[role])

    def anchor(self, sample):
        anchor = sample['anchor_latents'].float().unsqueeze(0)
        return .999 * anchor + .001 * self.noise(anchor)

    def finite(self, value):
        good = torch.isfinite(value.detach()).all().int()
        dist.all_reduce(good, op=dist.ReduceOp.MIN)
        if not int(good):
            raise RuntimeError('Nonfinite DMD value on one or more ranks')

    def replay_backward(self, generated, traces):
        """Exact full-output cotangent, one chunk graph at a time, one DP sync.

        History is already detached in the original rollout. Replay visits the
        independent chunk maps in reverse creation order and uses no_sync until
        the final backward, retaining one gradient reduction per microbatch.
        """
        cotangent = generated.grad
        if cotangent is None or len(traces) != self.config['blocks']:
            raise ValueError('Full DMD output cotangent/native chunk trace missing')
        self.finite(cotangent)
        devices = [self.device] if self.device.type == 'cuda' else []
        for index, trace in enumerate(reversed(traces)):
            sync = (self.generator.no_sync() if index + 1 < len(traces)
                    and hasattr(self.generator, 'no_sync') else nullcontext())
            with sync, torch.random.fork_rng(devices=devices):
                torch.set_rng_state(trace['cpu_rng'])
                if self.device.type == 'cuda':
                    torch.cuda.set_rng_state(trace['cuda_rng'], self.device)
                prediction = replay_student_chunk(self.generator, trace, self.device)
                start, count = trace['start'], trace['count']
                reference = generated.detach()[:, :, start:start + count]
                same = prediction.new_tensor(int(torch.equal(prediction.detach(), reference)), dtype=torch.int)
                dist.all_reduce(same, op=dist.ReduceOp.MIN)
                if not int(same):
                    raise RuntimeError('Chunk replay differs from its original no-grad prediction')
                torch.autograd.backward(prediction, cotangent[:, :, start:start + count])
                del prediction, reference
        generated.grad = None

    def update(self, role):
        if role not in ('critic', 'generator'):
            raise ValueError('Unknown DMD update role')
        is_critic = role == 'critic'
        if not is_critic and self.counters.critic != self.counters.critic_warmup + 5 * (self.counters.generator + 1):
            raise ValueError('A generator Adam requires five fresh critic Adams')
        optimizer = self.copt if is_critic else self.gopt
        model = self.critic if is_critic else self.generator
        optimizer.zero_grad(set_to_none=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic()
        moved_optimizer_bytes = 0
        park_critic = self.config.get('critic_optimizer_offload_for_generator', False)
        if park_critic:
            moved_optimizer_bytes = move_fp32_optimizer_state(self.copt, to_cpu=not is_critic)
            if not is_critic and moved_optimizer_bytes and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        cursor = self.counters.critic_samples if is_critic else self.counters.generator_samples
        values = torch.zeros(3, device=self.device, dtype=torch.float64)
        samples, exits, sigmas = [], [], []
        chunk_replay = not is_critic and self.config.get('generator_chunk_replay', False)
        for micro in range(self.accumulation):
            absolute = cursor + micro * self.world + self.rank
            identity, sample = self.sample(role, absolute)
            samples.append(identity)
            step = self.exit_step()
            # Critic uses the same randomized self-forcing exit distribution,
            # with gradients disabled. Full four-map rollouts are eval-only.
            with torch.set_grad_enabled(not is_critic and not chunk_replay):
                generated, meta = student_rollout(self.generator, sample, self.schedule,
                    self.config['blocks'], step, rng=self.streams[role], train_mode=True,
                    record_replay=chunk_replay)
            if chunk_replay:
                generated.requires_grad_(True)
            self.finite(generated)
            sigma = self.score_sigma(meta['denoised_to'])
            noisy, target = critic_inputs(generated, self.noise(generated), sigma)
            anchor = self.anchor(sample)
            if is_critic:
                velocity = self.critic(sample, noisy, anchor, 1 - sigma, camera_conditioned=True)
                objective = critic_loss(velocity, target)
                direction_mean = objective.new_zeros(())
            else:
                real, fake = real_and_fake_x0(self.teacher, self.critic, sample, noisy, sigma,
                                              anchor, self.config['real_guidance_scale'])
                self.finite(real)
                self.finite(fake)
                objective, direction = generator_loss(generated, real, fake)
                direction_mean = direction.abs().mean().detach()
            self.finite(objective)
            (objective / self.accumulation).backward()
            if chunk_replay:
                self.replay_backward(generated, meta['replay_inputs'])
            values += torch.stack((objective.detach().double(), sigma.detach().double(),
                                   direction_mean.double())) / self.accumulation
            exits.append(step)
            sigmas.append(float(sigma))
            del generated, noisy, target, anchor, objective
            if is_critic:
                del velocity
            else:
                del real, fake, direction
        norm = clip_sharded_grad_norm(model.parameters(), self.config['clip'][role])
        self.finite(norm)
        optimizer.step()
        if is_critic:
            self.counters.critic_updated()
        else:
            self.counters.generator_updated()
        optimizer.zero_grad(set_to_none=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        seconds = time.monotonic() - started
        local = float(values[0])
        dist.all_reduce(values)
        values /= self.world
        return dict(role=role, rank=self.rank, world_size=self.world,
            **asdict(self.counters), accumulation=self.accumulation, sample_ids=samples,
            sample_cursor_before=cursor, loss=local, global_loss=float(values[0]),
            score_sigma_mean=float(values[1]), DMD_direction_abs_mean=float(values[2]),
            grad_norm=float(norm), grad_exits=exits, score_sigmas=sigmas,
            critic_optimizer_parked_on_CPU=park_critic and not is_critic,
            optimizer_state_moved_bytes=moved_optimizer_bytes,
            generator_backward_mode='chunk_replay' if chunk_replay else 'original_graph',
            blocks=self.config['blocks'], latent_frames=7 + 5 * (self.config['blocks'] - 1),
            seconds=seconds, peak_GiB=torch.cuda.max_memory_allocated(self.device) / 2**30
            if self.device.type == 'cuda' else None)

    def cycle(self):
        self.counters.validate(boundary=True)
        critic = [self.update('critic') for _ in range(5)]
        generator = self.update('generator')
        self.counters.validate(boundary=True)
        return critic, generator
