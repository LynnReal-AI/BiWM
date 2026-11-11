#!/usr/bin/env bash
set -euo pipefail
: "${H3_CONFIG:?Set H3_CONFIG to the reviewed DMD JSON configuration}"
: "${OUTPUT_DIR:?Set a new run directory, or the same directory for exact resume}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OUTPUT_DIR
export H3_DMD_BASELINE_ROOT="${H3_DMD_BASELINE_ROOT:-$OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pipelines.h3.configure_stage3 --input "$H3_CONFIG" --output "$OUTPUT_DIR/config.json"
export BIWM_RUN="$OUTPUT_DIR"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
    -m pipelines.h3.train_stage3
