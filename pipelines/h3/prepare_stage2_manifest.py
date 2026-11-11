# SPDX-License-Identifier: Apache-2.0
"""Build a source-disjoint full camera manifest; never extract/duplicate latents."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def source_id(sample_id):
    # Exclude sibling 160-frame crops of a held-out dataset source clip too.
    return sample_id.split('/h3-')[0].split('__160f')[0]


def build_manifest(index, pilot_provenance, output):
    index, pilot_provenance, output = map(Path, (index, pilot_provenance, output))
    output.mkdir(parents=True, exist_ok=False)
    heldout = {source_id(row['sample_id']) for row in json.loads(pilot_provenance.read_text())
               if row['split'] == 'validation'}
    if not heldout:
        raise ValueError('A fixed validation source split is required')
    counts, excluded = Counter(), Counter()
    temporary = output / 'train.jsonl.tmp'
    with gzip.open(index, 'rt') as reader, temporary.open('w') as writer:
        for line in reader:
            row = json.loads(line)
            domain = row['sample_id'].split('/')[0]
            if source_id(row['sample_id']) in heldout:
                excluded[domain] += 1
                continue
            selected = {key: row[key] for key in (
                'sample_id', 'shard', 'h3_preencoded_member', 'tensor_digest')}
            writer.write(json.dumps(selected, separators=(',', ':')) + '\n')
            counts[domain] += 1
    if not sum(counts.values()):
        raise ValueError('No training windows after source split')
    temporary.replace(output / 'train.jsonl')
    def sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    receipt = dict(index=str(index.resolve()), index_sha256=sha256(index),
                   pilot_provenance=str(pilot_provenance.resolve()),
                   pilot_provenance_sha256=sha256(pilot_provenance),
                   train_manifest_sha256=sha256(output / 'train.jsonl'),
                   training_windows=sum(counts.values()), training_domains=dict(counts),
                   excluded_windows=sum(excluded.values()), excluded_domains=dict(excluded),
                   validation_source_ids=sorted(heldout), latent_files_duplicated=0,
                   source_group_rule='strip H3 window and derived __160f crop suffix; dataset source clip, not a scene-level split',
                   status='Manifest prepared only; no training launched')
    (output / 'PROVENANCE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--pilot-provenance', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.index, args.pilot_provenance, args.output)))
