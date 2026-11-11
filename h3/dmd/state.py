# SPDX-License-Identifier: Apache-2.0
"""Complete same-world G/critic checkpointing at actual 1G/5C boundaries."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from pipelines.h3.train_stage1 import checkpoint_context

SCHEMA = 'biwm.h3.dmd.complete.v1'


@dataclass
class Counters:
    generator: int = 0
    critic: int = 0
    generator_samples: int = 0
    critic_samples: int = 0
    critic_warmup: int = 0
    effective_batch: int = 8

    def validate(self, *, boundary=False):
        values = asdict(self)
        if any(type(v) is not int or v < 0 for v in values.values()) or not self.effective_batch:
            raise ValueError('DMD counters must be nonnegative integers and batch positive')
        if self.generator_samples != self.generator * self.effective_batch:
            raise ValueError('Generator data cursor disagrees with actual updates')
        if self.critic_samples != self.critic * self.effective_batch:
            raise ValueError('Critic data cursor disagrees with actual updates')
        if self.generator and self.critic < self.critic_warmup + 5 * self.generator:
            raise ValueError('Generator has fewer than five fresh critic updates')
        if self.critic > self.critic_warmup + 5 * (self.generator + 1):
            raise ValueError('More than five fresh critic updates before generator')
        if boundary and self.critic != self.critic_warmup + 5 * self.generator:
            raise ValueError('Checkpoint is not a complete 1G/5C boundary')

    def critic_updated(self):
        candidate = Counters(**asdict(self))
        candidate.critic += 1
        candidate.critic_samples += self.effective_batch
        candidate.validate()
        self.__dict__.update(candidate.__dict__)

    def generator_updated(self):
        if self.critic != self.critic_warmup + 5 * (self.generator + 1):
            raise ValueError('Require five actual new critic updates before generator')
        self.generator += 1
        self.generator_samples += self.effective_batch
        self.validate(boundary=True)


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_rng(device, generators):
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                generators={key: value.get_state() for key, value in generators.items()})


def restore_rng(state, device, generators):
    if set(state['generators']) != set(generators):
        raise ValueError('Dedicated DMD RNG streams differ')
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        if state['cuda'] is None:
            raise ValueError('CUDA RNG missing from GPU checkpoint')
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)
    elif state['cuda'] is not None:
        raise ValueError('GPU checkpoint cannot resume as CPU training')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    for name, generator in generators.items():
        generator.set_state(state['generators'][name].cpu())


def _all(values):
    outputs = [None] * dist.get_world_size()
    dist.all_gather_object(outputs, values)
    return outputs


def save_checkpoint(output, generator, critic, generator_optimizer, critic_optimizer,
                    counters, config, device, *, rng_streams):
    """Frozen teacher is pinned by config source hashes, never online state."""
    counters.validate(boundary=True)
    rank, world = dist.get_rank(), dist.get_world_size()
    digest = config_digest(config)
    headers = _all(dict(counters=asdict(counters), config_sha256=digest))
    if any(header != headers[0] for header in headers):
        raise ValueError('Ranks disagree on completed updates/configuration')
    destination = Path(output) / f'checkpoint_{counters.generator:06d}'
    if (destination / 'COMPLETE.json').exists():
        raise ValueError('Refuse to overwrite a complete DMD checkpoint')
    destination.mkdir(parents=True, exist_ok=True)
    rng = capture_rng(device, rng_streams)
    with checkpoint_context(generator):
        generator_state = generator.state_dict()
    with checkpoint_context(critic):
        critic_state = critic.state_dict()
    state = dict(schema=SCHEMA, rank=rank, world_size=world, config_sha256=digest,
                 counters=asdict(counters), generator=generator_state, critic=critic_state,
                 generator_optimizer=generator_optimizer.state_dict(),
                 critic_optimizer=critic_optimizer.state_dict(), rng=rng)
    path = destination / f'rank{rank:03d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    del generator_state, critic_state, state
    files = _all(dict(rank=rank, path=path.name, bytes=path.stat().st_size, sha256=file_digest(path)))
    if rank == 0:
        receipt = dict(schema=SCHEMA, world_size=world, config_sha256=digest,
                       counters=asdict(counters), source_weight='Live', files=files)
        marker = destination / 'COMPLETE.tmp'
        marker.write_text(json.dumps(receipt, indent=2) + '\n')
        marker.replace(destination / 'COMPLETE.json')
    dist.barrier()
    return destination


def load_checkpoint(path, generator, critic, generator_optimizer, critic_optimizer,
                    config, device, *, rng_streams):
    """Validate all rank files before mutating either role or optimizer."""
    path = Path(path)
    rank, world = dist.get_rank(), dist.get_world_size()
    state, error = None, None
    try:
        complete = json.loads((path / 'COMPLETE.json').read_text())
        digest = config_digest(config)
        if complete['schema'] != SCHEMA or complete['world_size'] != world:
            raise ValueError('DMD checkpoint schema/world differs')
        if complete['config_sha256'] != digest:
            raise ValueError('DMD scientific/data/source configuration differs')
        counters = Counters(**complete['counters'])
        counters.validate(boundary=True)
        if len(complete['files']) != world or sorted(f['rank'] for f in complete['files']) != list(range(world)):
            raise ValueError('DMD checkpoint rank manifest is incomplete')
        for item in complete['files']:
            if item['path'] != f"rank{item['rank']:03d}.pt":
                raise ValueError('Invalid DMD checkpoint rank path')
            if (path / item['path']).stat().st_size != item['bytes'] or not item['bytes']:
                raise ValueError('DMD checkpoint rank file is missing or truncated')
        item = next(f for f in complete['files'] if f['rank'] == rank)
        if file_digest(path / item['path']) != item['sha256']:
            raise ValueError('DMD checkpoint rank SHA differs')
        state = torch.load(path / item['path'], map_location='cpu', weights_only=False)
        if (state['schema'], state['rank'], state['world_size'], state['config_sha256'], state['counters']) != (
                SCHEMA, rank, world, digest, asdict(counters)):
            raise ValueError('DMD rank payload disagrees with complete manifest')
        if set(state['rng']['generators']) != set(rng_streams):
            raise ValueError('Dedicated DMD RNG streams differ')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = _all(error)
    if any(errors):
        raise ValueError('Cannot restore complete DMD state: ' + repr(errors))
    with checkpoint_context(generator):
        generator.load_state_dict(state['generator'], strict=True)
    with checkpoint_context(critic):
        critic.load_state_dict(state['critic'], strict=True)
    generator_optimizer.load_state_dict(state['generator_optimizer'])
    critic_optimizer.load_state_dict(state['critic_optimizer'])
    restore_rng(state['rng'], device, rng_streams)
    dist.barrier()
    return counters
