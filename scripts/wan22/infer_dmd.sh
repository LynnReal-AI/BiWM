#!/usr/bin/bash
# =============================================================================
# Wan 2.2 TI2V-5B - Inference: chunk-by-chunk 4-step autoregressive rollout (Stage 2 DMD generator)
# =============================================================================
# Entry: pipelines/wan/infer_stage2.py
#   - t2v by default; set MODE=i2v + I2V_VIDEO=<mp4> for image/video-to-video continuation.
#   - Camera: discrete cam-text via ACTION_FRAMES (e.g. "w-8, a-11"); leave empty for random/None.
# All values below are env-overridable, e.g.:  GENERATOR_CKPT=... ACTION_FRAMES="w-8, a-11" bash scripts/wan22/infer_dmd.sh
# =============================================================================
set -e

WAN_BASE=${WAN_BASE:-./ckpts/Wan2.2-TI2V-5B}

# generator checkpoint: pass GENERATOR_CKPT=<ckpt dir>, or auto-pick the latest under STAGE2_DIR
STAGE2_DIR=${STAGE2_DIR:-"./logs/wan22/stage2"}
if [ -z "${GENERATOR_CKPT}" ]; then
    _STEP=$(ls -d ${STAGE2_DIR}/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
    GENERATOR_CKPT="${STAGE2_DIR}/checkpoint-${_STEP}"
fi
if [ ! -e "${GENERATOR_CKPT}" ]; then
    echo "[ERROR] generator checkpoint not found: ${GENERATOR_CKPT}"
    echo "        set GENERATOR_CKPT=<checkpoint dir> or STAGE2_DIR=<stage2 output dir>"
    exit 1
fi

MODE=${MODE:-t2v}                          # t2v | i2v
PROMPT=${PROMPT:-"A cinematic scene, smooth camera movement, highly detailed, photorealistic."}
ACTION_FRAMES=${ACTION_FRAMES:-}           # e.g. "w-8, a-11"; empty = random / None
I2V_VIDEO=${I2V_VIDEO:-}                    # for MODE=i2v: path to the conditioning mp4
DURATION_SEC=${DURATION_SEC:-5.0}
FPS=${FPS:-24}
CHUNK_SIZE=${CHUNK_SIZE:-4}
MAX_CHUNKS=${MAX_CHUNKS:-5}
SINK_CHUNKS=${SINK_CHUNKS:-1}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}
NUM_HEIGHT=${NUM_HEIGHT:-480}
NUM_WIDTH=${NUM_WIDTH:-832}
SEED=${SEED:-42}
OUTPUT=${OUTPUT:-./outputs/wan22_dmd_infer.mp4}

EXTRA=()
[ -n "${ACTION_FRAMES}" ] && EXTRA+=(--action_frames "${ACTION_FRAMES}")
[ -n "${I2V_VIDEO}" ] && EXTRA+=(--i2v_video "${I2V_VIDEO}")

python pipelines/wan/infer_stage2.py \
    --generator_ckpt "${GENERATOR_CKPT}" \
    --wan_base "${WAN_BASE}" \
    --mode "${MODE}" \
    --prompt "${PROMPT}" \
    --duration_sec ${DURATION_SEC} \
    --fps ${FPS} \
    --chunk_size ${CHUNK_SIZE} \
    --max_chunks ${MAX_CHUNKS} \
    --sink_chunks ${SINK_CHUNKS} \
    --sigma_shift ${SIGMA_SHIFT} \
    --num_height ${NUM_HEIGHT} \
    --num_width ${NUM_WIDTH} \
    --seed ${SEED} \
    --output "${OUTPUT}" \
    "${EXTRA[@]}" ${EXTRA_ARGS}
