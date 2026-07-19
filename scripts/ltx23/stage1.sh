#!/usr/bin/env bash
# LTX-Video 2.3 Stage 1: per-frame camera-text fine-tuning.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29617}

if [[ -n "${KUBERNETES_CONTAINER_RESOURCE_GPU:-}" ]]; then
    NPROC_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE=$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)
else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || true)
    [[ ${NPROC_PER_NODE} -gt 0 ]] || NPROC_PER_NODE=8
fi

LTX2_ROOT=${LTX2_ROOT:-${ROOT_DIR}/ckpts/LTX-Video-2.3}
LTX2_CKPT=${LTX2_CKPT:-${LTX2_ROOT}/ltx-2.3-22b-dev.safetensors}
GEMMA_PATH=${GEMMA_PATH:-${LTX2_ROOT}/google/gemma-3-12b-it-qat-q4_0-unquantized}
VIDEO_DIR=${BIWM_VIDEO_DIR:-${ROOT_DIR}/dataset/video_real}
CAPTION_JSON=${BIWM_CAPTION_JSON:-${ROOT_DIR}/dataset/videos_real.json}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/logs/ltx23/stage1}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/logs}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log") 2>&1

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200000}

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    pipelines/ltx23/train_stage1.py \
    --data_mode live \
    --camera_mode camtext \
    --biwm_video_dir "${VIDEO_DIR}" \
    --biwm_caption_json "${CAPTION_JSON}" \
    --pretrained_model_path "${LTX2_CKPT}" \
    --vae_path "${LTX2_CKPT}" \
    --gemma_path "${GEMMA_PATH}" \
    --training_mode mixed \
    --i2v_rate 0.8 \
    --i2v_cond_latent_frames 1 \
    --num_frames 77 \
    --num_height 480 \
    --num_width 832 \
    --seed 42 \
    --gradient_checkpointing \
    --optimizer adamw \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --betas 0.9,0.999 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 20 \
    --max_grad_norm 1.0 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --cp_size "${CP_SIZE:-1}" \
    --sigma_shift 3.0 \
    --validation_shift 5.0 \
    --max_train_steps "${MAX_TRAIN_STEPS:-20000}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS:-100}" \
    --validation_interval "${VALIDATION_INTERVAL:-50}" \
    --first_validation_step "${FIRST_VALIDATION_STEP:-50}" \
    --diffusion_sampling_steps 50 \
    --cfg_scale 6.0 \
    --fps 24 \
    --dataloader_num_workers "${NUM_WORKERS:-2}" \
    --log_interval 1 \
    --output_dir "${OUTPUT_DIR}"
