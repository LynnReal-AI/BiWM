#!/usr/bin/env python3
# NVFP4 deployment weight export CLI: QAT-trained generator BF16 ckpt -> packed-FP4 safetensors
# (each 2D .weight -> uint8 packed FP4 + FP8-E4M3 block scale + global scale), for nvfp4_kernel.py.
# Usage:
#   python pipelines/common/nvfp4_pack_ckpt.py --in_ckpt <bf16.safetensors> --out <fp4.safetensors>
import argparse

import safetensors.torch

from pipelines.common.nvfp4_pack import NVFP4PackConfig, transform_state_dict_to_nvfp4


def main():
    ap = argparse.ArgumentParser("NVFP4 weight export (QAT bf16 → packed FP4)")
    ap.add_argument("--in_ckpt", required=True,
                    help="QAT-trained generator weights (.safetensors, BF16 full-precision state_dict)")
    ap.add_argument("--out", required=True, help="output packed-FP4 safetensors path")
    ap.add_argument("--block_size", type=int, default=16, help="only 16 is supported (LightX2V compatible)")
    ap.add_argument("--skip_keys", type=str, nargs="*",
                    default=["text_embedding", "time_embedding", "time_projection", "head"],
                    help="key substrings to skip quantization (aligned with training-time QAT nvfp4_skip_modules)")
    args = ap.parse_args()

    sd = safetensors.torch.load_file(args.in_ckpt, device="cpu")
    cfg = NVFP4PackConfig(enabled=True, block_size=args.block_size,
                            skip_keys=tuple(args.skip_keys or ()))
    out_sd = transform_state_dict_to_nvfp4(sd, cfg)
    safetensors.torch.save_file(
        out_sd, args.out,
        metadata={"quantization": "nvfp4", "block_size": str(args.block_size),
                  "format": "block16-packed-nvfp4"})


if __name__ == "__main__":
    main()
