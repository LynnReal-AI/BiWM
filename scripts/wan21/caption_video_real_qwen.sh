#!/usr/bin/bash
# =============================================================================
# Qwen3-VL-235B captioning for video_real (★ only describe the static scene, strictly forbid camera motion/camera movement/person actions)
# =============================================================================
#   - 235B/439GB MoE: device_map="auto" single instance [model parallel] spread across 8 cards (not 8-way data parallel);
#   - therefore it is [single process] processing 5066 entries serially, each taking [start/middle/end 3 frames] → only describing static content (appearance/environment/objects/art style);
#   - resumable: skip entries that already have a caption; save back to videos_real.json every 20 entries;
#   - uses subject_ref/.venv (transformers 5.9, supports qwen3_vl_moe), reproducing ground_test.py's 235B usage.
#   Entry: pipelines/data_preprocess/caption_video_real_qwen.py
#   Usage: bash scripts/finetune/caption_video_real_qwen.sh
# =============================================================================
set -e

BASE=.
SR=./ckpts/subject_ref
# 235B needs qwen3_vl_moe from transformers>=4.57; uses subject_ref's venv
PYTHON_BIN=${PYTHON_BIN:-"${SR}/.venv/bin/python"}
MODEL=${MODEL:-"${SR}/Qwen3-VL-235B-full"}
VIDEO_DIR=${VIDEO_DIR:-"${BASE}/dataset/video_real"}
JSON=${JSON:-"${BASE}/dataset/videos_real.json"}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:${BASE}"
# device_map=auto will use all visible GPUs; to specify: export CUDA_VISIBLE_DEVICES=0,1,...,7

cd "${BASE}"
echo "=============================================="
echo "video_real caption (Qwen3-VL-235B, static description, no motion/camera movement/person actions)"
echo "  python=${PYTHON_BIN}"
echo "  model =${MODEL}"
echo "  video =${VIDEO_DIR}"
echo "  json  =${JSON}"
echo "  GPUs  =${CUDA_VISIBLE_DEVICES:-all visible} (device_map=auto model parallel)"
echo "=============================================="

"${PYTHON_BIN}" pipelines/data_preprocess/caption_video_real_qwen.py \
    --video_dir "${VIDEO_DIR}" --json "${JSON}" --model "${MODEL}"

echo "[done] caption written back to ${JSON}"
