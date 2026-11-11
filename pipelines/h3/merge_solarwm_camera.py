# SPDX-License-Identifier: Apache-2.0
"""CPU shard-by-shard FP32 merge of the original official camera LoRA."""
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(32 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    base, adapter, output = Path(a.base), Path(a.adapter), Path(a.output)
    receipt = output / 'MERGE_COMPLETE.json'
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior['base'] != str(base) or prior['adapter'] != str(adapter):
            raise ValueError('Existing merge has a different source')
        return
    source = torch.load(adapter / 'ema.pt', map_location='cpu', mmap=True, weights_only=True)
    if source['schema'] != 'solarwm.minimax-h3-ema.v1':
        raise ValueError('Unexpected official camera release schema')
    def canonical(key):
        key = '.'.join(part for part in key.split('.') if part not in {
            '_fsdp_wrapped_module', '_checkpoint_wrapped_module'})
        return key.removeprefix('base_model.model.').replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')
    values = {canonical(k): v for k, v in source['shadow'].items()}
    expected = {k for k in values if k.endswith('.lora_A.weight')}
    if len(expected) != 312 or len(values) != 624:
        raise ValueError('Official camera LoRA must have exactly 312 A/B pairs')
    destination = output / 'transformer'
    destination.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / 'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())
    expected_weights = {k.removesuffix('.lora_A.weight') + '.weight' for k in expected}
    if not expected_weights <= set(index['weight_map']):
        raise ValueError(f'Camera adapter/base keys differ: {sorted(expected_weights-set(index["weight_map"]))[:8]}')
    used = set()
    shards = []
    for filename in sorted(set(index['weight_map'].values())):
        target = destination / filename
        tensors = {}
        with safe_open(base / 'transformer' / filename, framework='pt', device='cpu') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                prefix = key.removesuffix('.weight')
                lora_a, lora_b = prefix + '.lora_A.weight', prefix + '.lora_B.weight'
                if key.endswith('.weight') and lora_a in values:
                    A, B = values[lora_a], values[lora_b]
                    if A.shape[0] != 384 or B.shape[1] != 384:
                        raise ValueError('Expected rank=alpha=384')
                    tensor = (tensor.float() + B.float() @ A.float()).to(tensor.dtype)
                    used.add(lora_a)
                tensors[key] = tensor.contiguous()
            temporary = target.with_suffix('.tmp')
            save_file(tensors, str(temporary), metadata={'format': 'pt'})
            temporary.replace(target)
        shards.append(dict(name=filename, sha256=digest(target)))
        print(json.dumps(dict(merged=filename, adapter_modules=len(used))), flush=True)
        del tensors
    if used != expected:
        raise ValueError(f'Unmatched official adapter keys: {sorted(expected-used)[:10]}')
    for filename in ['config.json', 'diffusion_pytorch_model.safetensors.index.json']:
        (destination / filename).write_bytes((base / 'transformer' / filename).read_bytes())
    receipt.write_text(json.dumps(dict(base=str(base), adapter=str(adapter), official_release_step=10500,
                                      official_source_weight='published EMA only; not project v7 EMA',
                                      destination_weight='merged Live initialization', rank=384, alpha=384,
                                      adapter_sha256=digest(adapter / 'ema.pt'), modules=len(used),
                                      merge_compute='FP32, then original checkpoint dtype', shards=shards), indent=2)+'\n')


if __name__ == '__main__':
    main()
