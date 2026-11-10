"""Export a complete Stage 1 checkpoint as Live diffusers weights or PEFT LoRA.

Launch using torchrun with exactly the world size of the saved checkpoint.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType

from h3.h3_fa3_policy import require_h3_fa3_training_backend
from h3.model import BiWMH3Transformer
from pipelines.h3.train_stage1 import checkpoint_context, configure_lora, wrap_fsdp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--merge_lora", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    require_h3_fa3_training_backend("_flash_3", device)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    checkpoint = Path(args.checkpoint)
    receipt = json.loads((checkpoint / "COMPLETE.json").read_text())
    if receipt["world_size"] != dist.get_world_size():
        raise ValueError("Export requires the saved FSDP world size")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Export output must be a new directory")
    state = torch.load(checkpoint / f"rank{rank:03d}.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    del state["optimizer"]
    plain = BiWMH3Transformer.strict_from_pretrained(
        config["model_path"], subfolder="transformer", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True)
    if config["mode"] == "lora":
        configure_lora(plain, argparse.Namespace(**config))
    model = wrap_fsdp(plain, device)
    with checkpoint_context(model):
        model.load_state_dict(state["model"], strict=True)
    del state
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full = model.state_dict()
    if rank == 0:
        # A CPU model avoids ever gathering the complete 33B transformer on one GPU.
        from accelerate import init_empty_weights
        with init_empty_weights():
            exported = BiWMH3Transformer.from_config(plain.config)
            if config["mode"] == "lora":
                configure_lora(exported, argparse.Namespace(**config))
        exported.load_state_dict(full, strict=True, assign=True)
        if config["mode"] == "lora" and not args.merge_lora:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file
            output.mkdir(parents=True)
            save_file({k: v.contiguous() for k, v in get_peft_model_state_dict(exported).items()},
                      str(output / "adapter_model.safetensors"))
            exported.peft_config["default"].save_pretrained(output)
        else:
            if config["mode"] == "lora":
                # Fuse in FP32 on CPU, then restore the native mixed precision.
                exported.float()
                exported.fuse_lora(safe_fusing=True)
                exported.unload_lora()
                for name, parameter in exported.named_parameters():
                    fp32 = name.startswith(("proj_in.", "proj_out.", "audio_proj_in.",
                                            "audio_proj_out.", "time_embedder."))
                    parameter.data = parameter.data.to(torch.float32 if fp32 else torch.bfloat16)
            exported.save_pretrained(output / "transformer", safe_serialization=True)
        (output / "SOURCE.json").write_text(json.dumps(dict(
            source_checkpoint=str(checkpoint.resolve()), source_weight="Live",
            step=receipt["step"], mode=config["mode"], merged=args.merge_lora), indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
