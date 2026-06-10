#!/usr/bin/bash

# =============================================================================
# HunyuanVideo-1.5 (HY15) stage1 - text-controlled camera training (video_real, PRoPE removed)
# =============================================================================
# Dataset: dataset/video_real/{id}/gen.mp4  (whole clip has one 81-class camera action)
#   caption + action in dataset/videos_real.json
# Camera: removes PRoPE/discrete action injection, switches to [cam-text text control] — translate the clip's camera action into a
#       sentence of English (e.g. "Camera moves left. Camera pitches down.") and concatenate it into the caption, with the MLLM text stream controlling camera motion.
# Text encoder: MLLM = Qwen2.5-VL-7B (not T5)
# Training: full parameters; optimizer Muon (consistent with minWM); flow-matching (rand→shift, train sigma_shift=3.0=official train_timestep_shift);
#       sampler FlowMatchDiscreteScheduler(euler); validation shift=5.0(=official validation_timestep_shift)+cfg6.0; grad_accum=1;
#       before training (step0) run a validation once to confirm the base inference path is correct; afterwards output mp4 every 50 steps.
# Weights: all from ckpts/HunyuanVideo-1.5 (transformer 480p_t2v / vae / text_encoder llm), no dependency on minWM.
# Video: dataloader decodes mp4 in real time → online VAE + MLLM encoding inside the loop (--data_mode live).
# =============================================================================

set -e

# ===== Auto-detect distributed environment =====
if [ -n "$KUBERNETES_CONTAINER_RESOURCE_GPU" ]; then
    NPROC_PER_NODE=$KUBERNETES_CONTAINER_RESOURCE_GPU
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NPROC_PER_NODE" -eq 0 ]; then
        NPROC_PER_NODE=8
    fi
fi

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29613}

# ===== Frame count / resolution (77 frames → HY15 VAE 16x spatial / 4x temporal → latent 20×30×52) =====
NUM_FRAMES=77
NUM_HEIGHT=480
NUM_WIDTH=832

# ===== Sequence-parallel CP (Ulysses, same name as Wan script) =====
#   CP_SIZE=1 off (pure DP); >1 must divide total GPU count, and divide the latent token count (20×30×52=31200, divisible by 2/4/8).
#   Cards in the same CP group jointly compute a 1/CP shard of one sequence (intra-block all-to-all full-sequence attention), DP dim = total GPU // CP_SIZE.
CP_SIZE=${CP_SIZE:-1}

# ===== Dataset (video_real, env-overridable) =====
BIWM=.
BIWM_VIDEO_DIR=${BIWM_VIDEO_DIR:-${BIWM}/dataset/video_real}
BIWM_CAPTION_JSON=${BIWM_CAPTION_JSON:-${BIWM}/dataset/videos_real.json}

# ===== Data mode (live=online VAE+MLLM encoding every step; preenc=read pre-encoded .pt, faster; see preencode_hy15.sh) =====
DATA_MODE=${DATA_MODE:-live}
PREENCODED_DIR=${PREENCODED_DIR:-${BIWM}/dataset/preenc_hy15/video_real}
PREENC_ARG=""
[ "${DATA_MODE}" = "preenc" ] && PREENC_ARG="--preencoded_dir ${PREENCODED_DIR}"

# ===== Weights (ckpts/HunyuanVideo-1.5; t2v uses 480p_t2v) =====
CKPT=${CKPT:-${BIWM}/ckpts/HunyuanVideo-1.5}
HY15_TRANSFORMER_DIR=${HY15_TRANSFORMER_DIR:-${CKPT}/transformer/480p_t2v}

# ===== Logging =====
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
LOG_NAME=${LOG_NAME:-"HY15_stage1_camtext_video_real_480p_77f"}
LOG_DIR="${BIWM}/logs/hy15/stage1"
OUTPUT_DIR=${OUTPUT_DIR:-"${BIWM}/logs/hy15/stage1"}
mkdir -p ${LOG_DIR} ${OUTPUT_DIR}

LOG_FILE="${LOG_DIR}/train_node${NODE_RANK}_$(date +%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "HunyuanVideo-1.5 stage1 - text-controlled camera (video_real, PRoPE removed)"
echo "Log file: ${LOG_FILE}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo "Nodes: $NNODES, GPUs per node: $NPROC_PER_NODE, total GPUs: $TOTAL_GPUS"
echo "Resolution: ${NUM_HEIGHT}x${NUM_WIDTH}, frames=${NUM_FRAMES}"
echo "Camera: cam-text text control (PRoPE removed), optimizer Muon, full-parameter training"
echo "CP (sequence parallel): CP_SIZE=${CP_SIZE} (1=pure DP; >1 = Ulysses same group jointly computes one sequence)"
echo "Dataset: ${BIWM_VIDEO_DIR}"
echo "transformer: ${HY15_TRANSFORMER_DIR} (480p_t2v)"
echo "MLLM/VAE: ${CKPT}/text_encoder/llm (Qwen2.5-VL) / ${CKPT}/vae"
echo "=============================================="

# ===== Environment config (biwm-hy15: biwm-wan clone + torch2.9.1/transformers4.56 supports qwen2_5_vl) =====
_CONDA_ROOT=${CONDA_PREFIX%/*}
if [ -d "${_CONDA_ROOT}/biwm-hy15" ]; then CONDA_ENV_DIR="${_CONDA_ROOT}/biwm-hy15"; else CONDA_ENV_DIR="${_CONDA_ROOT}/biwm-wan"; fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_DIR}/bin/python}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib:${LD_LIBRARY_PATH}"
echo "[env] CONDA_ENV_DIR=${CONDA_ENV_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:${BIWM}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200000

cd "${BIWM}"

# ===== Launch training =====
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pipelines/hy15/train_hunyuan.py \
    --data_mode ${DATA_MODE} \
    ${PREENC_ARG} \
    --camera_mode camtext \
    --biwm_video_dir "${BIWM_VIDEO_DIR}" \
    --biwm_caption_json "${BIWM_CAPTION_JSON}" \
    --pretrained_model_path "${HY15_TRANSFORMER_DIR}" \
    --vae_path "${CKPT}/vae" \
    --text_encoder_path "${CKPT}/text_encoder/llm" \
    --training_mode t2v \
    --num_frames ${NUM_FRAMES} \
    --num_height ${NUM_HEIGHT} \
    --num_width ${NUM_WIDTH} \
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
    --cp_size ${CP_SIZE} \
    --sigma_shift 3.0 \
    --validation_shift 5.0 \
    --max_train_steps 20000 \
    --checkpointing_steps 100 \
    --validation_interval 50 \
    --first_validation_step 50 \
    --diffusion_sampling_steps 50 \
    --cfg_scale 6.0 \
    --fps 24 \
    --dataloader_num_workers 2 \
    --log_interval 1 \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Training finished"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code: $EXIT_CODE"
echo "=============================================="
exit $EXIT_CODE
