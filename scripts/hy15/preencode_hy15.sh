#!/usr/bin/bash
# =============================================================================
# HY1.5 pre-encoding (DLC single card) — pre-compute the VAE latent + text embed for both the video_real + videos datasets.
# During training, set DATA_MODE=preenc + PREENCODED_DIR in stage1_hy15.sh to read directly, skipping per-step online encoding (faster).
# Entry: pipelines/data_preprocess/preencode_hy15.py
# Output: dataset/preenc_hy15/{video_real,videos}/<vid_id>.pt  (resumable, skips if already exists)
# =============================================================================
set -e

BIWM=.
CKPT=${CKPT:-${BIWM}/ckpts/HunyuanVideo-1.5}
VAE_PATH=${VAE_PATH:-${CKPT}/vae}
TEXT_ENCODER_PATH=${TEXT_ENCODER_PATH:-${CKPT}/text_encoder/llm}
NUM_FRAMES=${NUM_FRAMES:-77}; NUM_HEIGHT=${NUM_HEIGHT:-480}; NUM_WIDTH=${NUM_WIDTH:-832}
OUT_ROOT=${OUT_ROOT:-${BIWM}/dataset/preenc_hy15}

# Distributed sharding (DLC single card → world=1; multi-card: torchrun injects RANK/WORLD_SIZE for auto sharding)
export RANK=${RANK:-0}; export WORLD_SIZE=${WORLD_SIZE:-1}; export LOCAL_RANK=${LOCAL_RANK:-0}

# ===== env: biwm-hy15 (torch2.9/transformers4.56, supports qwen2_5_vl), falls back to biwm-wan =====
_CONDA_ROOT=${CONDA_PREFIX%/*}
if [ -d "${_CONDA_ROOT}/biwm-hy15" ]; then CONDA_ENV_DIR="${_CONDA_ROOT}/biwm-hy15"; else CONDA_ENV_DIR="${_CONDA_ROOT}/biwm-wan"; fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_DIR}/bin/python}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib:${LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:${BIWM}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
cd "${BIWM}"

LOG_DIR="${BIWM}/logs/hy15/preenc"; mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/preenc_node${RANK}_$(date +%H%M%S).log") 2>&1

echo "=============================================="
echo "HY1.5 pre-encoding  env=${CONDA_ENV_DIR}  RANK=${RANK}/${WORLD_SIZE}"
echo "VAE=${VAE_PATH}  MLLM=${TEXT_ENCODER_PATH}  resolution=${NUM_HEIGHT}x${NUM_WIDTH} frames=${NUM_FRAMES}"
echo "output root directory=${OUT_ROOT}"
echo "start: $(date '+%F %T')"
echo "=============================================="

# Detect the allocated GPU count (DLC injects KUBERNETES_CONTAINER_RESOURCE_GPU); single card → single process, multi-card → torchrun sharding (idx%world==rank) for speedup
NGPU=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ -z "${NGPU}" ] || [ "${NGPU}" -lt 1 ] 2>/dev/null && NGPU=1
echo "[env] Detected GPU count NGPU=${NGPU}"

run_one () {
    local NAME=$1 VDIR=$2 CJSON=$3
    echo ">>> [$(date '+%T')] pre-encoding ${NAME}: ${VDIR} (NGPU=${NGPU})"
    local ARGS=(--video_dir "${VDIR}" --caption_json "${CJSON}" --output_dir "${OUT_ROOT}/${NAME}"
                --vae_path "${VAE_PATH}" --text_encoder_path "${TEXT_ENCODER_PATH}"
                --num_frames ${NUM_FRAMES} --num_height ${NUM_HEIGHT} --num_width ${NUM_WIDTH})
    if [ "${NGPU}" -gt 1 ]; then
        "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=${NGPU} --master_port=$((29700 + RANDOM % 100)) \
            pipelines/data_preprocess/preencode_hy15.py "${ARGS[@]}"
    else
        "${PYTHON_BIN}" -m pipelines.data_preprocess.preencode_hy15 "${ARGS[@]}"
    fi
    echo ">>> [$(date '+%T')] ${NAME} done"
}

# First video_real (5066, used by stage1), then videos (19824)
run_one video_real "${BIWM}/dataset/video_real" "${BIWM}/dataset/videos_real.json"
run_one videos     "${BIWM}/dataset/videos"     "${BIWM}/dataset/videos_syn.json"

echo "=============================================="
echo "All pre-encoding done: $(date '+%F %T')  exit code $?"
echo "Output: ${OUT_ROOT}/video_real/*.pt , ${OUT_ROOT}/videos/*.pt"
echo "For training: DATA_MODE=preenc PREENCODED_DIR=${OUT_ROOT}/video_real bash scripts/finetune/stage1_hy15.sh"
echo "=============================================="
