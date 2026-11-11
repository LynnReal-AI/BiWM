# SPDX-License-Identifier: Apache-2.0
"""Small balanced-by-domain, unique-source SolarWM subset for authorized smoke."""
import argparse, gzip, hashlib, json, tarfile
from collections import defaultdict
from pathlib import Path
import torch
from safetensors.torch import load, save_file
from h3.data import validate_sample


from h3.framepack_data import invert


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--release_root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output, root = Path(a.output), Path(a.release_root)
    output.mkdir(parents=True, exist_ok=False)
    rows, seen = defaultdict(list), set()
    with gzip.open(a.index, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            source = row['sample_id'].split('/h3-')[0]
            domain = row['sample_id'].split('/')[0]
            if len(rows[domain]) >= 18 or source in seen:
                continue
            seen.add(source)
            rows[domain].append(row)
    metadata = []
    manifests = {'train': [], 'validation': []}
    for domain, selected in sorted(rows.items()):
        for index, row in enumerate(selected):
            split = 'validation' if index >= 16 else 'train'
            with tarfile.open(root / row['shard']) as tar:
                blob = tar.extractfile(row['h3_preencoded_member']).read()
                if hashlib.blake2s(blob).hexdigest() != row['tensor_digest']:
                    raise ValueError('SolarWM tensor digest differs')
                manifest_blob = tar.extractfile(row['manifest_member']).read()
                if hashlib.blake2s(manifest_blob).hexdigest() != row['manifest_digest']:
                    raise ValueError('SolarWM manifest digest differs')
                manifest = json.loads(manifest_blob)
            source = load(blob)
            c2w = source['camera_c2w'].float()
            if len(c2w) != source['target_latents'].shape[1]:
                raise ValueError('Expected native latent-aligned SolarWM camera cache')
            relative = invert(c2w[:1])[0].unsqueeze(0) @ c2w
            relative[0] = torch.eye(4)
            sample = {key: source[key] for key in ['target_latents', 'anchor_latents', 'prompt_embeds', 'text_token_tags']}
            sample.update(camera_viewmats=invert(relative).contiguous(), camera_K=source['camera_K'].float())
            validate_sample(sample)
            name = f'{domain}_{index:03d}.safetensors'
            save_file({k:v.contiguous() for k,v in sample.items()}, str(output / name))
            manifests[split].append(dict(sample_id=row['sample_id'],path=name))
            metadata.append(dict(sample_id=row['sample_id'], split=split, source_row=row,
                                 source_manifest=manifest, gt_rgb_availability='not extracted; check raw source before delivery'))
    for split, values in manifests.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
    (output / 'PROVENANCE.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(train=len(manifests['train']),validation=len(manifests['validation']),
                          domains={k:len(v) for k,v in rows.items()})),flush=True)


if __name__ == '__main__':
    main()
