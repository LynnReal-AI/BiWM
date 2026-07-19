#!/usr/bin/env bash
# HunyuanVideo-1.5 Stage 2: DMD distillation with discrete camera actions.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29616}

if [[ -n "${KUBERNETES_CONTAINER_RESOURCE_GPU:-}" ]]; then
    NPROC_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE=$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)
else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || true)
    [[ ${NPROC_PER_NODE} -gt 0 ]] || NPROC_PER_NODE=8
fi

latest_checkpoint() {
    local directory=$1
    local checkpoint
    while IFS= read -r checkpoint; do
        if [[ -f "${checkpoint}/diffusion_pytorch_model.safetensors" || -f "${checkpoint}/model.pt" ]]; then
            printf '%s\n' "${checkpoint}"
            return 0
        fi
    done < <(find "${directory}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V -r)
    return 1
}

TASK_TYPE=${TASK_TYPE:-t2v}
NUM_BLOCKS=${NUM_BLOCKS:-5}
HY15_ROOT=${HY15_ROOT:-${ROOT_DIR}/ckpts/HunyuanVideo-1.5}
TRANSFORMER_DIR=${HY15_TRANSFORMER_DIR:-${HY15_ROOT}/transformer/480p_t2v}
VIDEO_DIR=${BIWM_VIDEO_DIR:-${ROOT_DIR}/dataset/videos}
CAPTION_JSON=${BIWM_CAPTION_JSON:-${ROOT_DIR}/dataset/videos_syn.json}
STAGE1_DIR=${STAGE1_DIR:-${ROOT_DIR}/logs/hy15/stage1_action}

STAGE1_CHECKPOINT=$(latest_checkpoint "${STAGE1_DIR}" || true)
GENERATOR_CKPT=${HY15_GENERATOR_CKPT:-${STAGE1_CHECKPOINT}}
REAL_SCORE_CKPT=${HY15_REAL_SCORE_CKPT:-${STAGE1_CHECKPOINT}}
FAKE_SCORE_CKPT=${HY15_FAKE_SCORE_CKPT:-${GENERATOR_CKPT}}

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/logs/hy15/stage2_dmd_action}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/logs}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log") 2>&1

DMD_EXTRA_ARGS=()
[[ ${DMD_TS_SCHEDULE:-1} == 0 ]] || DMD_EXTRA_ARGS+=(--dmd_ts_schedule)
[[ ${DMD_EULER_ROLLOUT:-0} == 1 ]] && DMD_EXTRA_ARGS+=(--dmd_euler_rollout)
[[ ${DMD_USE_GAN:-0} == 1 ]] && DMD_EXTRA_ARGS+=(--dmd_use_gan --dmd_gan_weight "${DMD_GAN_WEIGHT:-0.01}" --dmd_disc_lr "${DMD_DISC_LR:-1e-4}")

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
    pipelines/hy15/train_dmd.py \
    --data_mode live \
    --camera_mode action \
    --use_discrete_action \
    --training_mode "${TASK_TYPE}" \
    --biwm_video_dir "${VIDEO_DIR}" \
    --biwm_caption_json "${CAPTION_JSON}" \
    --pretrained_model_path "${TRANSFORMER_DIR}" \
    --vae_path "${HY15_ROOT}/vae" \
    --text_encoder_path "${HY15_ROOT}/text_encoder/llm" \
    --generator_ckpt "${GENERATOR_CKPT}" \
    --real_score_ckpt "${REAL_SCORE_CKPT}" \
    --fake_score_ckpt "${FAKE_SCORE_CKPT}" \
    --gradient_checkpointing \
    --num_frames 77 \
    --num_height 480 \
    --num_width 832 \
    --fps 24.0 \
    --validation_interval "${VALIDATION_INTERVAL:-50}" \
    --first_validation_step "${FIRST_VALIDATION_STEP:-1}" \
    --dmd_denoising_sigmas 1.0 0.75 0.5 0.25 \
    --dmd_block_K 4 \
    --dmd_num_blocks "${NUM_BLOCKS}" \
    --dmd_timestep_shift 5.0 \
    --dfake_gen_update_ratio 5 \
    --real_guidance_scale 5.0 \
    --dmd_generator_lr 1e-5 \
    --dmd_critic_lr 8e-6 \
    --weight_decay 1e-4 \
    --max_grad_norm 5.0 \
    --dmd_critic_warmup_steps "${CRITIC_WARMUP_STEPS:-20}" \
    "${DMD_EXTRA_ARGS[@]}" \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --log_interval 1 \
    --max_train_steps "${MAX_TRAIN_STEPS:-50000000}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS:-50}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42
