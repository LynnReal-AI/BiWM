"""Export a selected phase2 Live on its saved world size before DMD startup."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from h3.h3_fa3_policy import require_h3_fa3_training_backend
from h3.fa3_autograd import verify_fa3_backward
from pipelines.h3.train_stage1 import configure_lora
from h3.framepack import FramePackH3
from pipelines.h3.train_stage2 import wrap
from h3.dmd.initialization import export_score_bundle, load_phase2_generator, require_phase2_selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    require_h3_fa3_training_backend('_flash_3', device)
    gate = verify_fa3_backward(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=90))
    if 'H200' not in torch.cuda.get_device_name(device):
        raise ValueError('Authorized phase2 export requires H200')
    result = [None, None]
    if dist.get_rank() == 0:
        try:
            result[0], complete = require_phase2_selection(args.checkpoint, args.selection, dist.get_world_size())
        except Exception as exc:
            result[1] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(result, src=0)
    if result[1]:
        raise ValueError('Phase2 selection failed: ' + result[1])
    config = result[0]
    model = FramePackH3.strict_from_pretrained(config['initialization_path'], subfolder='transformer',
        torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    model.install_framepack()
    configure_lora(model, argparse.Namespace(lora_rank=config['train']['lora_rank'],
                                            lora_alpha=config['train']['lora_alpha']))
    model.set_attention_backend('_flash_3')
    model = wrap(model, device)
    step = load_phase2_generator(args.checkpoint, model, dist.get_world_size(), device)
    export_score_bundle(model, config, args.output, source_checkpoint=args.checkpoint)
    if dist.get_rank() == 0:
        print(json.dumps(dict(event='selected_phase2_export', source_step=step,
            output=str(Path(args.output).resolve()), FA3_gate=gate, DMD_updates=0)), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
