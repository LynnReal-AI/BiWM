#!/usr/bin/bash
# =============================================================================
# Wan 2.2 TI2V-5B - Stage 2: DMD distillation (50-step -> 4-step) on the video_real real-game dataset
# =============================================================================
# DMD distillation on top of stage1 (discrete-camera cam-text), compressing multi-step sampling into a 4-step autoregressive rollout.
#   Entry: pipelines/wan/train_stage2.py --dmd_distill
#
# dataset switched to video_real (real game clips, 81-class combined-camera action_label).
# Dataset: dataset/video_real/{id}_{trans}_{rot}/gen.mp4 (77 frames -> 20 latent), caption+action_label
#         in videos_real.json; camera = a constant 81-class combined action for the whole clip (loader expands via action_label).
#   Camera: pure discrete cam-text cross-attn (81cls), full_labels[1, M*K] sliced and injected per block into
#         generator/teacher/critic/DMD-loss/validation(joystick), no continuous Plucker.
#
# Three models (DMD2), all inheriting the latest video_real stage1 weights:
#   - real_score (teacher, frozen): stage1 weights, provides data score (CFG)
#   - fake_score (critic, online full-param): starts from stage1 weights, learns the generator's current distribution
#   - generator (student, online full-param): stage1 weights, 4-step single-chunk rollout
#
# (per user request):
#   - GAN loss / GT-reg / EMA all disabled (DMD_GAN/DMD_GT_REG/DMD_EMA left empty)
#   - does not involve history_encoder (code skips build/load/train), normal rollout does DMD
#
# (per user request): train [two blocks], and [do not use HistoryEncoder].
#   - num_chunks=2, chunk_size=4 latent -> total 8 latent = 29 frames. 2-block autoregressive.
#   - no HE mechanism: block2 takes block1's already-generated [clean latent] as condition frames (cond_latent_frames) prepended to the sequence,
#     continuing via the model's native I2V timestep-separation path (switch --dmd_use_he off by default; only uses HE if passed). Adds no extra HE parameters.
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
MASTER_PORT=${MASTER_PORT:-29525}
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))

# ===== Resolution (480p, H/W must be divisible by 32 = VAE16x x patchify2x) =====
NUM_HEIGHT=480
NUM_WIDTH=832

# ===== chunk config (train two blocks, no HistoryEncoder) =====
#   CHUNK_SIZE = number of latent frames per chunk
#   NUM_CHUNKS = number of chunks (2 = 2-block autoregressive: block1 cold-start + block2 conditioned on block1 clean frames)
#   total latent = NUM_CHUNKS x CHUNK_SIZE = 8, NUM_FRAMES = (8-1)*4+1 = 29 frames
#   (underlying entry is still --dmd_block_K=CHUNK_SIZE / --dmd_num_blocks=NUM_CHUNKS)
#   no HE: do not pass --dmd_use_he (off by default) -> block2 autoregresses on cond_latent_frames condition frames, no HE params.
CHUNK_SIZE=${CHUNK_SIZE:-4}
NUM_CHUNKS=${NUM_CHUNKS:-5}
TOTAL_LATENT=$(( NUM_CHUNKS * CHUNK_SIZE ))    # 8
NUM_FRAMES=$(( (TOTAL_LATENT - 1) * 4 + 1 ))   # 29

# ===== Block count (per user request [fixed 5-block training], disable warmup + disable mix-M) =====
#   BLOCK_WARMUP_STEPS=0 -> no M=1 cold-start; DMD_BLOCK_COUNTS="5"/prob"1.0" -> constant M=5 (consistent with dataset/videos).
#   (to enable warmup: BLOCK_WARMUP_STEPS=300; to enable mix-M: DMD_BLOCK_COUNTS="1 2 3 4 5" PROBS="0.3 0.2 0.2 0.15 0.15")
BLOCK_WARMUP_STEPS=${BLOCK_WARMUP_STEPS:-0}
DMD_BLOCK_COUNTS="5"
DMD_BLOCK_COUNT_PROBS="1.0"

# ===== History compression mechanism (--dmd_history_mode none|framepack) =====
#   none      : no compression, already-generated [clean latent] used as condition frames (cond_latent_frames) for autoregressive continuation (original behavior/default);
#   framepack : yume1.5 pyramid multi-scale spatiotemporal downsampling (light near / heavy far, learnable multi-scale conv) compresses history -> mem tokens (no LR);
#               implementation in pipelines/wan/history_compress_framepack.py (refactored version, trained end-to-end with DMD).
#   switch: HISTORY_MODE=framepack bash scripts/finetune/stage2_dmd.sh
HISTORY_MODE=${HISTORY_MODE:-none}

# ===== 4-step generator schedule =====
DMD_SIGMAS="1.0 0.75 0.5 0.25"    # 4 start points (before shift; trailing 0.0 implied) = 4-step
SIGMA_SHIFT=5.0                   # consistent with stage1
DFAKE_GEN_RATIO=6                 # critic updates N times, gen 1 time (5->6, critic tracks more tightly)
REAL_GUIDANCE=5.0
GEN_LR=1e-5   # switched back to 1e-5 per user request
CRITIC_LR=2e-6
DISC_LR=1e-4   # GAN discriminator independent LR (previously wrongly used critic 2e-6 -> discriminator could not learn, disc=2.0 GAN ineffective)

# ===== Speedup: critic warmup + ts_schedule =====
DMD_CRITIC_WARMUP_STEPS=50
DMD_TS_SCHEDULE="--dmd_ts_schedule"

# ===== best-file DMD trick (per user request GAN / GT-reg / EMA all temporarily disabled) =====
# enable GAN to resist collapse — at step200 the generator collapses to RGB noise; pure DMD2 with no regularization is too fragile.
#   ProjectedDiscriminator(DINO-ViT-S, local weights ADD/models/weights/, offline usable), hinge, boosts sharpness + resists high-frequency collapse.
DMD_GAN=""        # GAN disabled per user request (rely on SFT+forward-KL to resist collapse instead); to enable change back to "--dmd_use_gan --dmd_gan_weight 0.01"
DMD_GT_REG=""     # disable GT-latent regression (if still drifting, add --dmd_use_gt_reg --dmd_gt_reg_weight 0.05)
DMD_EMA=""        # disable EMA

# ===== SFT + forward-KL anchoring (* , port from yume ltx_dmd.sh) — resists DMD collapse/mode-shrink, preserves long video and motion =====
#   SFT: low-sigma flow-matching velocity MLE on the [full real video (20 frames)] (strong data anchoring); * uses full video length, decoupled from dynamic M rollout,
#        so even warmup M=1 learns all 20 frames -> both resists collapse and preserves long-video modeling.
#   teacher-FKL: forward-KL along the teacher's dense ODE trajectory (mass-covering, prevents mode-shrink/low-motion; larger teacher_steps = more accurate but slower).
#   real-FKL: high-sigma regression of the real video back to x0 (mass-covering toward real motion), complements low-sigma SFT.
# per user request temporarily disable all SFT / forward-KL (run pure DMD + block-count curriculum + mix-M only to see the baseline)
SFT=""        # disable SFT (to enable: --dmd_sft_weight 0.1 --dmd_sft_sigma_max 0.125)
FKL=""        # disable teacher forward-KL (to enable: --dmd_fkl_weight 0.1 --dmd_fkl_steps 1 --dmd_fkl_teacher_steps 12)
REAL_FKL=""   # disable real forward-KL (to enable: --dmd_real_fkl_weight 0.1 --dmd_real_fkl_sigma_min 0.25 --dmd_real_fkl_sigma_max 0.65)

# ===== Trainable module whitelist (all = generator DiT fully unfrozen; no history_encoder anymore) =====
TRAIN_MODULES="all"

# ===== Dataset path (switched to video_real real-game dataset, env-overridable) =====
BIWM_VIDEO_DIR=${BIWM_VIDEO_DIR:-./dataset/video_real}
BIWM_CAPTION_JSON=${BIWM_CAPTION_JSON:-./dataset/videos_real.json}

# ===== Inherit latest stage1 weights (real/fake/generator all start from here) =====
# stage1 output dir (OUTPUT_DIR of stage1_video_real.sh); can be overridden with env var STAGE1_DIR.
# changed to this video_real stage1 run, auto-pick latest ckpt (currently checkpoint-1580)
STAGE1_DIR=${STAGE1_DIR:-"./logs/wan21/stage1"}
# Auto-pick the checkpoint-N with the largest step (use STAGE1_CKPT_STEP to specify an exact step)
if [ -z "${STAGE1_CKPT_STEP}" ]; then
    STAGE1_CKPT_STEP=$(ls -d ${STAGE1_DIR}/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
fi
STAGE1_CKPT_DIR="${STAGE1_DIR}/checkpoint-${STAGE1_CKPT_STEP}"
# Allow direct override via env vars (GENERATOR_CKPT / REAL_SCORE_CKPT)
GENERATOR_CKPT=${GENERATOR_CKPT:-"${STAGE1_CKPT_DIR}"}
REAL_SCORE_CKPT=${REAL_SCORE_CKPT:-"${STAGE1_CKPT_DIR}/diffusion_pytorch_model.safetensors"}

if [ -z "${STAGE1_CKPT_STEP}" ] || [ ! -f "${REAL_SCORE_CKPT}" ]; then
    echo "[ERROR] Cannot find latest stage1 weights: ${REAL_SCORE_CKPT}"
    echo "        Please run stage1_pretrain.sh first, or specify via env vars:"
    echo "        STAGE1_DIR=<stage1 output dir> or GENERATOR_CKPT=<ckpt dir> REAL_SCORE_CKPT=<...safetensors>"
    exit 1
fi

# pretrained = original Wan22 5B base model (architecture + VAE + text encoder; DiT weights are then overridden by the stage1 ckpt)
WAN_BASE=./ckpts/Wan2.1-T2V-1.3B

# ===== Logging (LOG_NAME can be overridden by wrapper scripts to separate none/framepack output dirs) =====
LOG_NAME=${LOG_NAME:-"Wan22_5B_stage2_dmd_video_real_480p_77f"}
LOG_DIR="./logs/wan21/stage2"
OUTPUT_DIR="./logs/wan21/stage2"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/train_node${NODE_RANK}_$(date +%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "Wan 2.2 5B - Stage 2 DMD distillation (50-step -> 4-step) on BiWM"
echo "  rollout          : num_chunks=${NUM_CHUNKS} × chunk_size=${CHUNK_SIZE} = ${TOTAL_LATENT} latent (NUM_FRAMES=${NUM_FRAMES})"
echo "  block count      : fixed M=${DMD_BLOCK_COUNTS} (warmup=${BLOCK_WARMUP_STEPS}, mix-M off), 5-block training throughout"
echo "  history_mode     : ${HISTORY_MODE}  (none=clean condition frames / framepack=pyramid multi-scale)"
echo "  4-step sigmas    : ${DMD_SIGMAS} (shift=${SIGMA_SHIFT})"
echo "  camera           : pure discrete cam-text (81cls), full_labels from action_frames (injected per block)"
echo "  train modules    : ${TRAIN_MODULES}  dfake/gen=${DFAKE_GEN_RATIO}  gen_lr=${GEN_LR} critic_lr=${CRITIC_LR}"
echo "  best-trick        : GAN=[${DMD_GAN:-off}] GT_REG=[${DMD_GT_REG:-off}] EMA=[${DMD_EMA:-off}]"
echo "  inherit stage1 weights : ${STAGE1_CKPT_DIR}  (real/fake/generator all start from here)"
echo "    generator_ckpt : ${GENERATOR_CKPT}"
echo "    real_score_ckpt: ${REAL_SCORE_CKPT}"
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
# Offline cluster: GAN discriminator DINO uses local weights, disable networking to avoid timeouts
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
    --dmd_history_mode ${HISTORY_MODE} \
    --dmd_block_counts ${DMD_BLOCK_COUNTS} \
    --dmd_block_count_probs ${DMD_BLOCK_COUNT_PROBS} \
    --dmd_block_warmup_steps ${BLOCK_WARMUP_STEPS} \
    --dmd_block_warmup_count 1 \
    --dmd_denoising_sigmas ${DMD_SIGMAS} \
    --dmd_timestep_shift ${SIGMA_SHIFT} \
    --dfake_gen_update_ratio ${DFAKE_GEN_RATIO} \
    --real_guidance_scale ${REAL_GUIDANCE} \
    --dmd_generator_lr ${GEN_LR} \
    --dmd_critic_lr ${CRITIC_LR} \
    --dmd_disc_lr ${DISC_LR} \
    --dmd_critic_warmup_steps ${DMD_CRITIC_WARMUP_STEPS} \
    ${DMD_TS_SCHEDULE} \
    ${DMD_GAN} \
    ${DMD_GT_REG} \
    ${DMD_EMA} \
    ${SFT} \
    ${FKL} \
    ${REAL_FKL} \
    --train_modules ${TRAIN_MODULES} \
    --offload_text_encoder \
    --use_biwm_camera_dataset \
    --biwm_video_dir "${BIWM_VIDEO_DIR}" \
    --biwm_caption_json "${BIWM_CAPTION_JSON}" \
    --seed 42 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 5.0 \
    --checkpointing_steps=50 \
    --pretrained_model_path "${WAN_BASE}" \
    --wan_version 2.1 \
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
    --dataloader_num_workers ${NUM_WORKERS:-4} \
    --prefetch_factor 2 \
    --cp_size 1 \
    --fps 24.0 \
    --validation_interval 100 \
    --first_validation_step 100
