#!/usr/bin/bash

# =============================================================================
# Wan 2.2 TI2V-5B - BiWM discrete camera text training (dataset/videos)
# =============================================================================
# Dataset: dataset/videos/{id}_{token string}/gen.mp4  (77 frames, 832x480, 24fps)
#   caption + pose in dataset/videos_syn.json (index == 6-digit id)
# Camera: pure discrete cam_text cross-attn (81cls), action_labels parsed directly from action_frames,
#       at latent level (20 latent frames = 1 init + 19 motion segments), no continuous Plucker.
#   token→label:  w/s/a/d=translate fwd/bwd/left/right(9/18/36/27)  up/down=pitch(1/2)  right/left=yaw(3/4)
# Training mode: hybrid (i2v 0.5 / t2v 0.5 / v2v 0.0)
# Weights: trained from scratch on the Wan2.1-T2V-1.3B base (no resume)
# Video: dataloader decodes mp4 in real time → VAE encoding
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
MASTER_PORT=${MASTER_PORT:-29505}

# ===== Training mode (i2v 0.5 / t2v 0.5 / v2v 0.0) =====
TRAINING_MODE="hybrid"
I2V_PROB=0.0
V2V_PROB=0.0
# t2v_prob = 1.0 - i2v - v2v = 0.5
I2V_COND_LATENT_FRAMES=1

# ===== Frame count (77 frames @ 24fps ≈ 3.2s → 20 latent frames) =====
MAX_FRAMES=77
VALIDATION_FRAMES=77

# ===== Resolution: 832x480 (÷32 OK: 832/32=26, 480/32=15) =====
NUM_HEIGHT=480
NUM_WIDTH=832

# ===== BiWM dataset paths (can be overridden by wrapper script env, e.g. stage1_video_real.sh) =====
BIWM_VIDEO_DIR=${BIWM_VIDEO_DIR:-./dataset/videos}
BIWM_CAPTION_JSON=${BIWM_CAPTION_JSON:-./dataset/videos_syn.json}

# ===== Logging =====
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
LOG_NAME=${LOG_NAME:-"wan21_1.3b_camtext_480p_77f"}
LOG_DIR="./logs/wan21/stage1"
OUTPUT_DIR="./logs/wan21/stage1"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/train_node${NODE_RANK}_$(date +%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "Wan 2.2 TI2V-5B - BiWM discrete camera text training"
echo "Log file: ${LOG_FILE}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo "Nodes: $NNODES, GPUs per node: $NPROC_PER_NODE, total GPUs: $TOTAL_GPUS"
echo "Resolution: ${NUM_HEIGHT}x${NUM_WIDTH}, fps=24, frames=${MAX_FRAMES}"
echo "Training mode: ${TRAINING_MODE} (i2v=${I2V_PROB}, t2v=0.5, v2v=${V2V_PROB})"
echo "Camera: pure discrete cam_text cross-attn (81cls), action_labels from action_frames, no Plucker"
echo "Dataset: ${BIWM_VIDEO_DIR}"
echo "caption/pose: ${BIWM_CAPTION_JSON}"
echo "Weights: Wan2.1-T2V-1.3B base (from scratch, no resume)"
echo "=============================================="

# ===== Environment config =====
PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV_DIR="${CONDA_PREFIX}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib:${LD_LIBRARY_PATH}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200000

WAN_MODEL_PATH=./ckpts/Wan2.1-T2V-1.3B

# ===== Launch training =====
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pipelines/wan/train_stage1.py \
    --biwm_video_dir "${BIWM_VIDEO_DIR}" \
    --biwm_caption_json "${BIWM_CAPTION_JSON}" \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --lr_warmup_steps=20 \
    --lr_scheduler constant_with_warmup \
    --gradient_accumulation_steps=4 \
    --weight_decay=0.001 \
    --max_train_epochs=10 \
    --checkpointing_steps=50 \
    --pretrained_model_path "${WAN_MODEL_PATH}" \
    --wan_version 2.1 \
    --vae_path "${WAN_MODEL_PATH}" \
    --text_encoder_path "${WAN_MODEL_PATH}" \
    --wan_model_type ti2v \
    --output_dir="${OUTPUT_DIR}" \
    --num_frames ${MAX_FRAMES} \
    --num_height ${NUM_HEIGHT} \
    --num_width ${NUM_WIDTH} \
    --training_mode "${TRAINING_MODE}" \
    --i2v_cond_latent_frames ${I2V_COND_LATENT_FRAMES} \
    --i2v_prob ${I2V_PROB} \
    --v2v_prob ${V2V_PROB} \
    --diffusion_sampling_steps 50 \
    --cfg_scale 5.0 \
    --validation_height ${NUM_HEIGHT} \
    --validation_width ${NUM_WIDTH} \
    --validation_frames ${VALIDATION_FRAMES} \
    --validation_interval 50 \
    --first_validation_step 51 \
    --dataloader_num_workers 4 \
    --prefetch_factor 2 \
    --learning_rate=5e-5 \
    --cp_size 1 \
    --fps 24.0 \
    --use_camera_control \
    --train_sigma_shift 5.0 \
    --val_sigma_shift 5.0 \
    --skip_validation_before_train

EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Training finished"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code: $EXIT_CODE"
echo "=============================================="
exit $EXIT_CODE
