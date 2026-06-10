#!/usr/bin/bash
# =============================================================================
# Wan 2.1 1.3B - NVFP4 quantized DMD inference (thin wrapper over infer_dmd.sh)
# =============================================================================
#   - Default: --nvfp4 fake-quant (numerically aligned; NO real speedup off Blackwell).
#   - Real FP4 kernel (Blackwell / RTX 5090 speedup):
#         NVFP4_KERNEL=<packed-FP4 .safetensors exported by pipelines/common/nvfp4_pack_ckpt.py>
#   - W4A4 (also quantize activations, quality drops; default is W4A16): NVFP4_QUANT_ACT=1
#   Use the QAT generator weights from scripts/wan21/stage3_nvfp4_optional.sh.
#   All infer_dmd.sh env vars apply (GENERATOR_CKPT / PROMPT / MODE / ACTION_FRAMES / ...).
# Example:  GENERATOR_CKPT=... NVFP4_KERNEL=gen_nvfp4.safetensors bash scripts/wan21/infer_dmd_nvfp4.sh
# =============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

_Q="--nvfp4"
[ -n "${NVFP4_KERNEL}" ] && _Q="${_Q} --nvfp4_kernel ${NVFP4_KERNEL}"
[ -n "${NVFP4_QUANT_ACT}" ] && _Q="${_Q} --nvfp4_quantize_activations"

export EXTRA_ARGS="${_Q} ${EXTRA_ARGS:-}"
export OUTPUT="${OUTPUT:-./outputs/wan21_dmd_infer_nvfp4.mp4}"
exec bash "$HERE/infer_dmd.sh"
