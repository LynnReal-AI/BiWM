# SPDX-License-Identifier: Apache-2.0
"""Read original SolarWM camera WDS without duplicating full latent caches."""
import hashlib
import json
from pathlib import Path
import tarfile

import torch
from safetensors.torch import load
from torch.utils.data import Dataset

from h3.data import validate_sample
def invert(value):
    out = torch.zeros_like(value)
    rotation = value[..., :3, :3].transpose(-1, -2)
    out[..., :3, :3] = rotation
    out[..., :3, 3] = -torch.einsum('...ij,...j->...i', rotation, value[..., :3, 3])
    out[..., 3, 3] = 1.
    return out



class SolarWMCameraDataset(Dataset):
    def __init__(self, manifest, release_root):
        self.manifest = Path(manifest).resolve()
        self.root = Path(release_root).resolve()
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError('Empty SolarWM camera manifest')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        shard = (self.root / row['shard']).resolve()
        if not shard.is_relative_to(self.root):
            raise ValueError('SolarWM shard is outside the declared release')
        with tarfile.open(shard, 'r:') as archive:
            member = archive.getmember(row['h3_preencoded_member'])
            if not member.isfile():
                raise ValueError('Expected a regular safetensors member')
            blob = archive.extractfile(member).read()
        if hashlib.blake2s(blob).hexdigest() != row['tensor_digest']:
            raise ValueError('SolarWM tensor digest differs')
        source = load(blob)
        c2w = source['camera_c2w'].float()
        if c2w.shape != (source['target_latents'].shape[1], 4, 4):
            raise ValueError('Expected native latent-aligned SolarWM camera cache')
        relative = invert(c2w[:1]) @ c2w
        relative[0] = torch.eye(4)
        sample = {name: source[name] for name in (
            'target_latents', 'anchor_latents', 'prompt_embeds', 'text_token_tags')}
        sample.update(camera_viewmats=invert(relative).contiguous(), camera_K=source['camera_K'].float())
        return validate_sample(sample)


class GlobalSampleOrder:
    """Same shuffled, nonoverlapping global samples for DP4/accum2 or DP8/accum1.

    No mutable reader cursor: step/epoch and a dedicated CPU generator determine
    the order without changing model/rollout RNG. Restore the optimizer step
    along with the original manifest/hash/seed for an exact same-world resume.
    """
    def __init__(self, length, seed):
        if length < 1:
            raise ValueError('Empty sample order')
        self.length, self.seed = length, seed
        self.epoch, self.order = None, None

    def __call__(self, absolute_index):
        if absolute_index < 0:
            raise ValueError('Negative sample index')
        epoch, index = divmod(absolute_index, self.length)
        if self.epoch != epoch:
            generator = torch.Generator(device='cpu').manual_seed(self.seed + epoch)
            self.order = torch.randperm(self.length, generator=generator).tolist()
            self.epoch = epoch
        return self.order[index]
