#!/usr/bin/bash
# =============================================================================
# Wan 2.1 1.3B - Stage 1 sanity inference (non-autoregressive, 50-step CFG, whole clip at once)
# =============================================================================
# Entry: pipelines/wan/infer_stage1.py
#   Verifies the Stage-1 cam-text teacher (bidirectional, full-clip). Camera via ACTION_FRAMES (e.g. "w-8, a-11").
# All values below are env-overridable.
# =============================================================================
set -e

WAN_BASE=${WAN_BASE:-./ckpts/Wan2.1-T2V-1.3B}

STAGE1_DIR=${STAGE1_DIR:-"./logs/wan21/stage1"}
if [ -z "${GENERATOR_CKPT}" ]; then
    _STEP=$(ls -d ${STAGE1_DIR}/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
    GENERATOR_CKPT="${STAGE1_DIR}/checkpoint-${_STEP}"
fi
if [ ! -e "${GENERATOR_CKPT}" ]; then
    echo "[ERROR] stage1 checkpoint not found: ${GENERATOR_CKPT}"
    echo "        set GENERATOR_CKPT=<checkpoint dir> or STAGE1_DIR=<stage1 output dir>"
    exit 1
fi

PROMPT=${PROMPT:-"A cinematic scene, smooth camera movement, highly detailed."}
ACTION_FRAMES=${ACTION_FRAMES:-}           # e.g. "w-8, a-11"; empty = random / None
NUM_STEPS=${NUM_STEPS:-50}
CFG=${CFG:-5.0}
NUM_FRAMES=${NUM_FRAMES:-77}
NUM_HEIGHT=${NUM_HEIGHT:-480}
NUM_WIDTH=${NUM_WIDTH:-832}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}
SEED=${SEED:-0}
FPS=${FPS:-24}
OUTPUT=${OUTPUT:-./outputs/wan21_stage1_infer.mp4}

EXTRA=()
[ -n "${ACTION_FRAMES}" ] && EXTRA+=(--action_frames "${ACTION_FRAMES}")

python pipelines/wan/infer_stage1.py \
    --generator_ckpt "${GENERATOR_CKPT}" \
    --wan_base "${WAN_BASE}" \
    --prompt "${PROMPT}" \
    --num_steps ${NUM_STEPS} \
    --cfg ${CFG} \
    --num_frames ${NUM_FRAMES} \
    --num_height ${NUM_HEIGHT} \
    --num_width ${NUM_WIDTH} \
    --sigma_shift ${SIGMA_SHIFT} \
    --seed ${SEED} \
    --fps ${FPS} \
    --output "${OUTPUT}" \
    "${EXTRA[@]}" ${EXTRA_ARGS}
