"""Expand H3 paths and pin reviewed Stage 3 inputs before CUDA startup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        value = os.path.expandvars(value)
        if re.search(r'\$(?:\{[^}]+\}|[A-Za-z_]\w*)', value):
            raise ValueError('Unset path variable in H3 configuration: ' + value)
    return value


def prepare(config):
    config = expand(config)
    source, data = config['source'], config['data']
    for key in ('phase2_checkpoint', 'selection', 'score_bundle'):
        source[key] = str(Path(source[key]).resolve())
    checkpoint = Path(source['phase2_checkpoint'])
    phase2_config = checkpoint.parent / 'executed_config.json'
    phase2 = json.loads(phase2_config.read_text())
    selected = json.loads(Path(source['selection']).read_text())
    complete = json.loads((checkpoint / 'COMPLETE.json').read_text())
    cases = config['validation']['indices']
    if (Path(selected['checkpoint']).resolve() != checkpoint
        or selected['source_config_sha256'] != digest(phase2_config)
        or selected['source_weight'] != 'Live' or selected['accepted_for_DMD'] is not True
        or selected['model_step'] != complete['step']
        or sorted(selected['reviewed_cases']) != sorted(cases) or len(set(cases)) != 8):
        raise ValueError('Phase2 source selection does not match the reviewed eight cases')
    source['selection_sha256'] = digest(source['selection'])
    source['score_bundle_sha256'] = digest(Path(source['score_bundle']) / 'COMPLETE.json')
    source['camera_merge_sha256'] = digest(Path(phase2['initialization_path']) / 'MERGE_COMPLETE.json')
    for key in ('train', 'validation'):
        data[key + '_manifest'] = str(Path(data[key + '_manifest']).resolve())
        data[key + '_manifest_sha256'] = digest(data[key + '_manifest'])
    data['release_root'] = str(Path(data['release_root']).resolve())
    if 'train' in config:
        for key in ('teacher_quality_review', 'teacher_media_verification', 'teacher_complete'):
            source[key] = str(Path(source[key]).resolve())
        review = json.loads(Path(source['teacher_quality_review']).read_text())
        media = json.loads(Path(source['teacher_media_verification']).read_text())
        teacher = json.loads(Path(source['teacher_complete']).read_text())
        media_sha, teacher_sha = (digest(source[key]) for key in
                                  ('teacher_media_verification', 'teacher_complete'))
        if (review['accepted_for_finite_DMD_pilot'] is not True
            or review['source_phase2_step'] != complete['step']
            or sorted(review['reviewed_cases']) != sorted(cases)
            or review['teacher_CFG'] != config['dmd']['real_guidance_scale']
            or review['source_media_verification_sha256'] != media_sha
            or review['source_teacher_complete_sha256'] != teacher_sha
            or media['all_passed'] is not True or media['cases'] != 8
            or teacher['cases'] != 8 or teacher['frames'] != 158
            or teacher['source_step'] != complete['step']
            or teacher['G_updates'] != 0 or teacher['C_updates'] != 0):
            raise ValueError('Teacher review/media/source/CFG do not match this finite pilot')
        if (config['train']['max_generator_steps'] > review.get('approved_max_generator_step', 2)
            or config['train']['max_critic_steps'] > review.get('approved_max_critic_step', 15)):
            raise ValueError('Training budget exceeds the explicitly reviewed finite budget')
        source['teacher_quality_review_sha256'] = digest(source['teacher_quality_review'])
        source['teacher_media_verification_sha256'] = media_sha
        source['teacher_complete_sha256'] = teacher_sha
        config['validation']['baseline_root'] = str(Path(config['validation']['baseline_root']).resolve())
    return config


def write_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Existing run configuration differs; use a new output directory')
        return
    with path.open('x') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')


def continuation_contract(config):
    """Separate reviewed budget/provenance updates from the training contract.

    Source weights, data/order/RNG, model/score/precision and optimizer settings
    remain strict. The new run independently verifies its reviewed source files
    before loading the old checkpoint with the old config digest.
    """
    value = json.loads(json.dumps(config))
    for key in ('continuation', 'checkpoint_steps', 'validation_steps', 'status',
                'stage', 'deliberate_differences', 'stop_conditions'):
        value.pop(key, None)
    for key in ('max_generator_steps', 'max_critic_steps', 'initial_generator_step'):
        value['train'].pop(key, None)
    for key in ('selection', 'selection_sha256', 'teacher_quality_review',
                'teacher_quality_review_sha256', 'teacher_media_verification',
                'teacher_media_verification_sha256', 'teacher_complete', 'teacher_complete_sha256'):
        value['source'].pop(key, None)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = prepare(json.loads(Path(args.input).read_text()))
    write_config(config, args.output)


if __name__ == '__main__':
    main()
