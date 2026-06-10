#!/usr/bin/bash
# =============================================================================
# Qwen3-VL-235B captioning for video_real (* describe only the static scene; camera motion/camera movement/character actions strictly forbidden)
# =============================================================================
#   - 235B/439GB MoE: device_map="auto" single instance [model parallel] spread across 8 cards (not 8-way data parallel);
#   - so it is a [single process] processing the 5066 clips serially, taking [start/middle/end 3 frames] per clip -> describe only static content (appearance/environment/objects/art style);
#   - resumable: skip entries that already have a caption; save back to videos_real.json every 20 clips;
#   - uses subject_ref/.venv (transformers 5.9, supports qwen3_vl_moe), reproducing the 235B usage in ground_test.py.
#   Entry: pipelines/data_preprocess/caption_video_real_qwen.py
#   Usage: bash scripts/finetune/caption_video_real_qwen.sh
# =============================================================================
set -e

BASE=.
SR=./ckpts/subject_ref
# 235B requires qwen3_vl_moe from transformers>=4.57; use subject_ref's venv
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
echo "video_real caption (Qwen3-VL-235B, static description, no motion/camera movement/character actions)"
echo "  python=${PYTHON_BIN}"
echo "  model =${MODEL}"
echo "  video =${VIDEO_DIR}"
echo "  json  =${JSON}"
echo "  GPUs  =${CUDA_VISIBLE_DEVICES:-all visible} (device_map=auto model parallel)"
echo "=============================================="

"${PYTHON_BIN}" pipelines/data_preprocess/caption_video_real_qwen.py \
    --video_dir "${VIDEO_DIR}" --json "${JSON}" --model "${MODEL}"

echo "[done] caption written back to ${JSON}"
