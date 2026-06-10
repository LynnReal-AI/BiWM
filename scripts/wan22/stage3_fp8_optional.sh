#!/usr/bin/bash
# ★★ This script = stage2_dmd.sh + FP8(E4M3) quantization-aware training (QAT) ★★  —— for Hopper/H200 (with FP8 tensor cores)
#   The generator's nn.Linear is fake-quantized to FP8 (per-tensor, weight+activation), running FP8 throughout (student);
#   additionally adds BF16↔FP8 self-distillation KL (flow_kl + x0_kl) to align quantized outputs to full precision, fighting quantization degradation.
#   ★ Mutually exclusive with NVFP4 (NVFP4=Blackwell/5090, FP8=Hopper/H200). Real inference speedup on H200 uses torch._scaled_mm.
# =============================================================================
# Wan 2.2 TI2V-5B - Stage 2: DMD distillation (50-step → 4-step) on the BiWM discrete-camera dataset
# =============================================================================
# Perform DMD distillation on top of stage1 (discrete-camera cam-text), compressing multi-step sampling into a 4-step autoregressive rollout.
#   Entry: pipelines/wan/train_stage2.py --dmd_distill
#
# Dataset: dataset/videos/{id}_{token-string}/gen.mp4 (77 frames→20 latent), caption+pose in videos_syn.json
#   Camera: pure discrete cam-text cross-attn (81cls), full_labels[1, M*K] parsed directly from action_frames (same as stage1),
#         sliced per block and injected into generator/teacher/critic/DMD-loss/validation(joystick), no continuous Plucker.
#
# ★ Stage 3 (QAT) three-model inheritance:
#   - real_score (teacher, frozen): first-stage stage1 weights (data score CFG, unchanged)
#   - fake_score (critic, online full params): starts from first-stage stage1 weights (aligns to generator distribution after warmup)
#   - generator (student, online full params): ★ second-stage DMD weights, quantization-aware fine-tuned on top
#
# (per user request):
#   - GAN loss / GT-reg / EMA all disabled (DMD_GAN/DMD_GT_REG/DMD_EMA emptied)
#   - history_encoder not involved (code already skips build/load/train), normal rollout for DMD
#
# (per user request): train [two blocks], and [do not use HistoryEncoder].
#   - num_chunks=2, chunk_size=4 latent → total 8 latent = 29 frames. Autoregressive 2 blocks.
#   - No HE mechanism: block2 prepends block1's already-generated [clean latent] as condition frames (cond_latent_frames) to the sequence,
#     continuing via the model's native I2V timestep-separation path (switch --dmd_use_he off by default; HE used only if passed). No extra HE params introduced.
# =============================================================================

set -e

# ===== GPU detection (DLC compatible) =====
if [ -n "$KUBERNETES_CONTAINER_RESOURCE_GPU" ]; then
    NPROC_PER_NODE=$KUBERNETES_CONTAINER_RESOURCE_GPU
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$NPROC_PER_NODE" -eq 0 ] && NPROC_PER_NODE=8
fi
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29531}
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))

# ===== Resolution (480p, H/W must be divisible by 32 = VAE16x × patchify2x) =====
NUM_HEIGHT=480
NUM_WIDTH=832

# ===== chunk config (train two blocks, no HistoryEncoder) =====
#   CHUNK_SIZE = number of latent frames per chunk
#   NUM_CHUNKS = number of chunks (2 = autoregressive two blocks: block1 cold start + block2 conditioned on block1 clean frames)
#   total latent = NUM_CHUNKS × CHUNK_SIZE = 8, NUM_FRAMES = (8-1)*4+1 = 29 frames
#   (the underlying entry is still --dmd_block_K=CHUNK_SIZE / --dmd_num_blocks=NUM_CHUNKS)
#   no HE: do not pass --dmd_use_he (off by default) → block2 uses cond_latent_frames condition frames for autoregression, no HE params.
CHUNK_SIZE=4
NUM_CHUNKS=5
TOTAL_LATENT=$(( NUM_CHUNKS * CHUNK_SIZE ))    # 8
NUM_FRAMES=$(( (TOTAL_LATENT - 1) * 4 + 1 ))   # 29

# ===== 4-step generator schedule =====
DMD_SIGMAS="1.0 0.75 0.5 0.25"    # 4 start points (before shift; trailing 0.0 implicit) = 4-step
SIGMA_SHIFT=5.0                   # same as stage1
DFAKE_GEN_RATIO=5                 # update critic N times, gen 1 time (Self-Forcing=5)
REAL_GUIDANCE=5.0
GEN_LR=1e-5
CRITIC_LR=2e-6

# ===== Speedup: critic warmup + ts_schedule =====
DMD_CRITIC_WARMUP_STEPS=50
DMD_TS_SCHEDULE="--dmd_ts_schedule"

# ===== best-file DMD trick (per user request, GAN / GT-reg / EMA all temporarily disabled) =====
DMD_GAN=""        # disable GAN loss
DMD_GT_REG=""     # disable GT-latent regression
DMD_EMA=""        # disable EMA

# ===== ★ FP8 quantization-aware training (QAT) — H200 =====
FP8_KL_WEIGHT=${FP8_KL_WEIGHT:-0.03}
FP8_X0_WEIGHT=${FP8_X0_WEIGHT:-0.25}
FP8_FLOW_WEIGHT=${FP8_FLOW_WEIGHT:-1.0}
FP8_WARMUP=${FP8_WARMUP:-0}
FP8="--dmd_fp8 --fp8_kl_weight ${FP8_KL_WEIGHT} --fp8_flow_weight ${FP8_FLOW_WEIGHT} \
     --fp8_x0_weight ${FP8_X0_WEIGHT} --fp8_warmup_steps ${FP8_WARMUP}"

# ===== Trainable-module whitelist (all = unfreeze all generator DiT params; no history_encoder anymore) =====
TRAIN_MODULES="all"

# ===== BiWM dataset paths =====
BIWM_VIDEO_DIR=./dataset/videos
BIWM_CAPTION_JSON=./dataset/videos_syn.json

# ===== Stage 3 inherited weights (third stage, quantization-aware training) =====
#   ★ generator inherits the [second-stage DMD] output (the generator of stage2_dmd.sh), quantization-aware fine-tuned on top (QAT);
#   ★ real_score/fake_score (teacher/critic) use the original data-score model = first-stage stage1 (data-distribution target unchanged,
#     the 4-step distilled generator is not suitable to serve again as a 50-step teacher; if you really need teacher to also use stage2, set REAL_SCORE_CKPT to override).
# --- generator: second-stage DMD output (stage2_dmd.sh OUTPUT_DIR), automatically take the latest checkpoint ---
STAGE2_DIR=${STAGE2_DIR:-"./logs/wan22/stage2"}
if [ -z "${STAGE2_CKPT_STEP}" ]; then
    STAGE2_CKPT_STEP=$(ls -d ${STAGE2_DIR}/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
fi
STAGE2_CKPT_DIR="${STAGE2_DIR}/checkpoint-${STAGE2_CKPT_STEP}"
GENERATOR_CKPT=${GENERATOR_CKPT:-"${STAGE2_CKPT_DIR}"}
# --- teacher/critic: first-stage stage1 (data-score model) ---
STAGE1_DIR=${STAGE1_DIR:-"./logs/wan22/stage1"}
if [ -z "${STAGE1_CKPT_STEP}" ]; then
    STAGE1_CKPT_STEP=$(ls -d ${STAGE1_DIR}/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
fi
REAL_SCORE_CKPT=${REAL_SCORE_CKPT:-"${STAGE1_DIR}/checkpoint-${STAGE1_CKPT_STEP}/diffusion_pytorch_model.safetensors"}

if [ ! -f "${GENERATOR_CKPT}/diffusion_pytorch_model.safetensors" ]; then
    echo "[ERROR] Cannot find second-stage DMD generator: ${GENERATOR_CKPT}/diffusion_pytorch_model.safetensors"
    echo "        Please run stage2_dmd.sh first, or specify via STAGE2_DIR=<stage2 output dir> / GENERATOR_CKPT=<ckpt dir>"
    exit 1
fi
if [ ! -f "${REAL_SCORE_CKPT}" ]; then
    echo "[ERROR] Cannot find teacher(stage1) weights: ${REAL_SCORE_CKPT}"
    echo "        Specify via STAGE1_DIR=<stage1 output dir> or REAL_SCORE_CKPT=<...safetensors>"
    exit 1
fi

# pretrained = original Wan22 5B base model (architecture + VAE + text encoder; DiT weights are then overwritten by the stage1 ckpt)
WAN_BASE=./ckpts/Wan2.2-TI2V-5B

# ===== Log =====
LOG_NAME="Wan22_5B_stage3_fp8_biwm_480p_77f"
LOG_DIR="./logs/wan22/stage3"
OUTPUT_DIR="./logs/wan22/stage3"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_node${NODE_RANK}_$(date +%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "Wan 2.2 5B - Stage 3 FP8 quantization-aware training (QAT, inheriting second-stage DMD) on BiWM"
echo "  rollout          : num_chunks=${NUM_CHUNKS} × chunk_size=${CHUNK_SIZE} = ${TOTAL_LATENT} latent (NUM_FRAMES=${NUM_FRAMES})"
echo "  4-step sigmas    : ${DMD_SIGMAS} (shift=${SIGMA_SHIFT})"
echo "  camera           : pure discrete cam-text (81cls), full_labels from action_frames (injected per block)"
echo "  train modules    : ${TRAIN_MODULES}  dfake/gen=${DFAKE_GEN_RATIO}  gen_lr=${GEN_LR} critic_lr=${CRITIC_LR}"
echo "  best-trick        : GAN=[${DMD_GAN:-off}] GT_REG=[${DMD_GT_REG:-off}] EMA=[${DMD_EMA:-off}]"
echo "  FP8 QAT (H200)    : on (KL w=${FP8_KL_WEIGHT} flow=${FP8_FLOW_WEIGHT} x0=${FP8_X0_WEIGHT} warmup=${FP8_WARMUP})"
echo "  inherit(Stage3)   : generator←second-stage DMD, teacher/critic←first-stage stage1"
echo "    generator_ckpt : ${GENERATOR_CKPT}   (stage2 DMD)"
echo "    real_score_ckpt: ${REAL_SCORE_CKPT}   (stage1 teacher)"
echo "  base arch/VAE/TE : ${WAN_BASE}"
echo "  dataset          : ${BIWM_VIDEO_DIR}"
echo "  GPUs             : ${NPROC_PER_NODE} x ${NNODES} = ${TOTAL_GPUS}"
echo "  Log file         : ${LOG_FILE}"
echo "=============================================="

# ===== Environment (biwm-wan) =====
PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV_DIR="${CONDA_PREFIX}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_DIR}/lib:${LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Offline cluster: GAN discriminator DINO uses local weights, disable network to avoid timeouts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DOWNLOAD_TIMEOUT=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200000

${PYTHON_BIN} -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pipelines/wan/train_stage2.py \
    --dmd_distill \
    --generator_ckpt "${GENERATOR_CKPT}" \
    --real_score_ckpt "${REAL_SCORE_CKPT}" \
    --dmd_num_blocks ${NUM_CHUNKS} \
    --dmd_block_K ${CHUNK_SIZE} \
    --dmd_block_counts ${NUM_CHUNKS} \
    --dmd_block_count_probs "1.0" \
    --dmd_denoising_sigmas ${DMD_SIGMAS} \
    --dmd_timestep_shift ${SIGMA_SHIFT} \
    --dfake_gen_update_ratio ${DFAKE_GEN_RATIO} \
    --real_guidance_scale ${REAL_GUIDANCE} \
    --dmd_generator_lr ${GEN_LR} \
    --dmd_critic_lr ${CRITIC_LR} \
    --dmd_critic_warmup_steps ${DMD_CRITIC_WARMUP_STEPS} \
    ${DMD_TS_SCHEDULE} \
    ${DMD_GAN} \
    ${DMD_GT_REG} \
    ${DMD_EMA} \
    ${FP8} \
    --train_modules ${TRAIN_MODULES} \
    --offload_text_encoder \
    --use_biwm_camera_dataset \
    --biwm_video_dir "${BIWM_VIDEO_DIR}" \
    --biwm_caption_json "${BIWM_CAPTION_JSON}" \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --max_grad_norm 5.0 \
    --checkpointing_steps=50 \
    --pretrained_model_path "${WAN_BASE}" \
    --vae_path "${WAN_BASE}" \
    --text_encoder_path "${WAN_BASE}" \
    --wan_model_type ti2v \
    --output_dir="${OUTPUT_DIR}" \
    --num_frames ${NUM_FRAMES} \
    --num_height ${NUM_HEIGHT} \
    --num_width ${NUM_WIDTH} \
    --training_mode "t2v" \
    --use_camera_control \
    --train_sigma_shift ${SIGMA_SHIFT} \
    --val_sigma_shift ${SIGMA_SHIFT} \
    --master_weight_type bf16 \
    --enable_memory_efficient_attention \
    --dataloader_num_workers 4 \
    --prefetch_factor 2 \
    --cp_size 1 \
    --fps 24.0 \
    --validation_interval 200 \
    --first_validation_step 100
