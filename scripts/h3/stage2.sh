#!/usr/bin/env bash
set -euo pipefail
: "${H3_CONFIG:?Set H3_CONFIG to the pilot or full JSON configuration}"
: "${OUTPUT_DIR:?Set a new run directory, or the same directory for exact resume}"
export H3_CONFIG OUTPUT_DIR
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import hashlib,json,os,re
from pathlib import Path
def expand(value):
    if isinstance(value,dict):return {key:expand(item) for key,item in value.items()}
    if isinstance(value,list):return [expand(item) for item in value]
    if isinstance(value,str):
        value=os.path.expandvars(value)
        if re.search(r'\$\{[^}]+\}',value):raise ValueError('Unset path variable in H3 configuration: '+value)
    return value
config=expand(json.loads(Path(os.environ['H3_CONFIG']).read_text()))
manifest=Path(config['data'].get('train_manifest',str(Path(config['data_path'])/'train.jsonl')))
config['data']['train_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
output=Path(os.environ['OUTPUT_DIR']);output.mkdir(parents=True,exist_ok=True)
path=output/'config.json'
if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Existing run configuration differs; use a new output directory')
path.write_text(json.dumps(config,indent=2)+'\n')
PY
export BIWM_RUN="$OUTPUT_DIR"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
    -m pipelines.h3.train_stage2
