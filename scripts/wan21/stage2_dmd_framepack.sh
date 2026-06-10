#!/usr/bin/bash
# =============================================================================
# Stage 2 DMD distillation + FramePack history compression (yume1.5 pyramid multi-scale learnable conv) — video_real dataset
# =============================================================================
# This script = the FramePack entry of stage2_dmd.sh (thin wrapper, reuses the same training logic/data/inherited weights).
#   dataset/inherited weights already changed to video_real in stage2_dmd.sh (this entry inherits automatically).
#   History mechanism: --dmd_history_mode framepack
#     · adaptively selects a tier by history latent frame count (tiny/small/medium/large/xlarge), light near / heavy far;
#     · multi-scale spatial conv {1,2,4,8,16}x + frame compression 2x_f are all [learnable parameters], initialized by the main patch_embedding
#       via trilinear upsampling, trained end-to-end with DMD (TRAIN_MODULES=all already includes history_encoder);
#     · no LR branch (FramePack tokens are already the fully compressed history), uniformly returns (mem_tokens, mem_indices_grid).
#   Implementation: pipelines/wan/history_compress_framepack.py
#
# * Compression tier vs history length relation (K=4 -> history T=4*block count):
#     2x@T>=3(~0.4s, blocks>=1)  4x@T>=7(blocks>=2)  8x@T>=23(blocks>=6)  16x@T>=87  32x@T>=343
#   current default NUM_CHUNKS=5 -> max history T=16 (small tier, up to 4x). To trigger 8x or higher, increase NUM_CHUNKS>=6.
#
# Usage:
#   bash scripts/finetune/stage2_dmd_framepack.sh
#   NUM_CHUNKS=6 bash scripts/finetune/stage2_dmd_framepack.sh     # trigger a higher compression tier
# =============================================================================

export HISTORY_MODE=framepack
# Separate output dir per mode, to avoid ckpts overwriting each other
export LOG_NAME=${LOG_NAME:-"Wan22_5B_stage2_dmd_framepack_video_real_480p_77f"}

# Pass through all logic to the base script (env vars STAGE1_DIR / NUM_CHUNKS / GENERATOR_CKPT etc. still take effect)
exec bash "$(dirname "$0")/stage2_dmd.sh" "$@"
