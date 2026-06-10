#!/usr/bin/bash
# =============================================================================
# Wan 2.1 1.3B - FP8-E4M3 quantized DMD inference (thin wrapper over infer_dmd.sh)
# =============================================================================
#   - Default: --fp8 fake-quant (numerically aligned, any GPU, NO real speedup).
#   - Real FP8 GEMM (Hopper / H200 speedup, torch._scaled_mm): FP8_KERNEL=1
#         (auto-adds --torch_compile; real bandwidth gain needs batch>1, see --batch_size in infer_stage2.py).
#   - Weight-only quant (skip activation quant): FP8_WEIGHT_ONLY=1
#   Use the QAT generator weights from scripts/wan21/stage3_fp8_optional.sh.
#   All infer_dmd.sh env vars apply (GENERATOR_CKPT / PROMPT / MODE / ACTION_FRAMES / ...).
# Example:  GENERATOR_CKPT=... FP8_KERNEL=1 bash scripts/wan21/infer_dmd_fp8.sh
# =============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

_Q="--fp8"
[ -n "${FP8_KERNEL}" ] && _Q="${_Q} --fp8_kernel --torch_compile"
[ -n "${FP8_WEIGHT_ONLY}" ] && _Q="${_Q} --fp8_weight_only"

export EXTRA_ARGS="${_Q} ${EXTRA_ARGS:-}"
export OUTPUT="${OUTPUT:-./outputs/wan21_dmd_infer_fp8.mp4}"
exec bash "$HERE/infer_dmd.sh"
