#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
: "${H3_MODEL_PATH:?Set H3_MODEL_PATH to an H3 diffusers checkpoint root}"
: "${H3_MANIFEST:?Set H3_MANIFEST to the preencoded camera JSONL manifest}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new training run directory}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON_BIN:-python}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-8}" --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29531}" \
  -m pipelines.h3.train_stage1 \
  --model_path "$H3_MODEL_PATH" --manifest "$H3_MANIFEST" --output_dir "$OUTPUT_DIR" \
  --mode "${TRAINING_MODE:-lora}" --learning_rate "${LEARNING_RATE:-1e-5}" \
  --max_steps "${MAX_TRAIN_STEPS:-1000}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-100}" --attention_backend _flash_3 "$@"
