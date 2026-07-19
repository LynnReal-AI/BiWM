#!/usr/bin/env bash
# HunyuanVideo-1.5 Stage 1: per-latent-frame discrete camera-action fine-tuning.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29615}

if [[ -n "${KUBERNETES_CONTAINER_RESOURCE_GPU:-}" ]]; then
    NPROC_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE=$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)
else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || true)
    [[ ${NPROC_PER_NODE} -gt 0 ]] || NPROC_PER_NODE=8
fi

CP_SIZE=${CP_SIZE:-1}
DATA_MODE=${DATA_MODE:-live}
VIDEO_DIR=${BIWM_VIDEO_DIR:-${ROOT_DIR}/dataset/videos}
CAPTION_JSON=${BIWM_CAPTION_JSON:-${ROOT_DIR}/dataset/videos_syn.json}
PREENCODED_DIR=${PREENCODED_DIR:-${ROOT_DIR}/dataset/preenc_hy15/videos}

HY15_ROOT=${HY15_ROOT:-${ROOT_DIR}/ckpts/HunyuanVideo-1.5}
TRANSFORMER_DIR=${HY15_TRANSFORMER_DIR:-${HY15_ROOT}/transformer/480p_t2v}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/logs/hy15/stage1_action}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/logs}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log") 2>&1

PREENCODE_ARGS=()
if [[ ${DATA_MODE} == preenc ]]; then
    PREENCODE_ARGS+=(--preencoded_dir "${PREENCODED_DIR}")
fi

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
    pipelines/hy15/train_hunyuan.py \
    --data_mode "${DATA_MODE}" \
    "${PREENCODE_ARGS[@]}" \
    --camera_mode action \
    --use_discrete_action \
    --biwm_video_dir "${VIDEO_DIR}" \
    --biwm_caption_json "${CAPTION_JSON}" \
    --pretrained_model_path "${TRANSFORMER_DIR}" \
    --vae_path "${HY15_ROOT}/vae" \
    --text_encoder_path "${HY15_ROOT}/text_encoder/llm" \
    --training_mode mixed \
    --i2v_rate 0.4 \
    --v2v_rate 0.4 \
    --v2v_block_k 4 \
    --num_frames 77 \
    --num_height 480 \
    --num_width 832 \
    --seed 42 \
    --gradient_checkpointing \
    --optimizer muon \
    --learning_rate 2e-5 \
    --weight_decay 1e-4 \
    --betas 0.9,0.999 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 20 \
    --max_grad_norm 1.0 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --cp_size "${CP_SIZE}" \
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
