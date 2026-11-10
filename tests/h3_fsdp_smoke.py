"""Run with torchrun --standalone --nproc_per_node=2 tests/h3_fsdp_smoke.py.

CPU/Gloo test of mixed-precision FSDP save, restore and another optimizer update.
The production training entrypoint still requires Hopper/FA3.
"""
import json
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist

from h3.distributed import clip_sharded_grad_norm
from h3.flow import flow_matching_loss
from h3.optimizer import FP32MasterAdamW
from pipelines.h3.train_stage1 import wrap_fsdp, save_checkpoint, resume_checkpoint
from test_h3 import tiny_model, sample


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cpu")
    try:
        torch.manual_seed(13)
        model = tiny_model()
        for name, parameter in model.named_parameters():
            fp32 = name.startswith(("proj_in.", "proj_out.", "audio_proj_in.",
                                    "audio_proj_out.", "time_embedder."))
            parameter.data = parameter.data.to(torch.float32 if fp32 else torch.bfloat16)
        model = wrap_fsdp(model, device)
        optimizer = FP32MasterAdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        values = sample()
        flow_matching_loss(model, values).backward()
        clip_sharded_grad_norm(model.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = [tempfile.mkdtemp(prefix="biwm_h3_fsdp_") if rank == 0 else None]
        dist.broadcast_object_list(output, src=0)
        output = Path(output[0])
        config = dict(test="two-rank CPU mixed precision", world_size=world)
        save_checkpoint(output, model, optimizer, 1, config, rank, world, device)
        before = [p.detach().clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        step = resume_checkpoint(output / "checkpoint_000001", model, optimizer,
                                 config, rank, world, device)
        assert step == 1
        for parameter, expected in zip(model.parameters(), before):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        flow_matching_loss(model, values).backward()
        clip_sharded_grad_norm(model.parameters())
        optimizer.step()
        print(json.dumps(dict(rank=rank, world_size=world, cpu_fsdp_resume="passed")), flush=True)
        dist.barrier()
        if rank == 0:
            shutil.rmtree(output)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
