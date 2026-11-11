"""Independently decode and hash eight native H3 teacher or DMD validation cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from .configure_stage3 import digest, write_config


def verify(root, config, step):
    root = Path(root).resolve()
    cases = config['validation']['indices']
    if len(cases) != 8 or len(set(cases)) != 8:
        raise ValueError('Require eight fixed cases')
    height, width = config['data']['resolution']
    if (height, width, config['data']['pixel_frames']) != (768, 1344, 158):
        raise ValueError('Require native158-frame768p validation')
    inputs = []
    for case in cases:
        directory = root / 'validation' / f'case{case:03d}'
        marker = json.loads((directory / 'COMPLETE.json').read_text())
        if (marker['case_id'] != case or marker['frames'] != 158
            or marker['model_step'] != step or marker['full_decode'] is not True):
            raise ValueError('Validation marker belongs to another case/step/span')
        for name, panels in zip(('baseline.mp4', f'step{step}.mp4',
                                f'GT_step{step}.mp4', f'GT_baseline_step{step}.mp4'), (1, 1, 2, 3)):
            inputs.append((directory / name, panels))
    def inspect(item):
        path, panels = item
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'json', str(path)]))['streams'][0]
        if (int(probe['nb_read_frames']), probe['height'], probe['width']) != (158, height, width * panels):
            raise ValueError('MP4 geometry/frame count differs: ' + str(path))
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2',
                        '-i', str(path), '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL)
        return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                    sha256=digest(path), frames=158, width=width * panels, height=height, full_decode=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(inspect, inputs))
    return dict(at=time.time(), root=str(root), model_step=step, cases=8,
                case_ids=cases, files=files, all_passed=True,
                scope='MP4 frame/geometry/full-decode/SHA integrity only; not quality or optimizer-update acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.input, json.loads(Path(args.config).read_text()), args.step)
    write_config(result, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}))


if __name__ == '__main__':
    main()
