# HunyuanVideo-1.5 (HY15) Training Environment Setup Guide

> For stage1 training of HunyuanVideo-1.5 within biWM (`pipelines/hy15/train_hunyuan.py`,
> model in `hunyuan/`). This document lists, one by one, the **environment pitfalls** we actually ran into; follow it to avoid detours.
>
> **Hardware**: NVIDIA H200 (sm_90, ~140GB per GPU) ×8, CUDA toolkit 12.x (`nvcc` must be available, needed to compile flash-attn).

---

## 0. One-line conclusion (verified working combination)

| Component | Version | Notes |
|---|---|---|
| python | 3.10 | |
| **torch** | **2.9.1+cu128** | ★ must match transformers / flash-attn |
| **transformers** | **4.56.0** | qwen2_5_vl needs ≥4.49; 4.56 pairs with torch 2.9 |
| diffusers | 0.35.0 | |
| tokenizers | 0.22.2 | comes with transformers |
| torchvision | 0.24.1 | pairs with torch 2.9.1 |
| flash-attn | 2.8.3 (**compiled on-site for torch 2.9.1**) | optional but strongly recommended; if not installed, falls back to torch SDPA |
| accelerate / numpy / torchdata / torchao | 1.13.0 / 2.2.6 / 0.11.0 / 0.15.0 | see `minWM/requirements.txt` |

This is exactly the combination in HunyuanVideo-1.5's official `requirements.txt` (`torch==2.9.1` + `transformers==4.56.0`). **Do not mix old torch + new transformers, or vice versa** (see pitfalls ①②③).

---

## 1. Installation steps

```bash
# (1) create a new python 3.10 environment (or clone from an existing one)
conda create -n hy15 python=3.10 -y
conda activate hy15

# (2) install per HunyuanVideo-1.5's official requirements (incl. torch 2.9.1 / transformers 4.56.0 / diffusers 0.35)
pip install -r minWM/requirements.txt        # torch will pull a ~2GB cu12 wheel

# (3) flash-attn —— must be compiled on-site for the current torch (see pitfall ③), don't use one compiled for a different torch
#     needs nvcc; compilation takes 10-40min (limit concurrency to avoid OOM during compilation)
export MAX_JOBS=16
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
```

> Domestic/offline cluster: HF weights via mirror (`export HF_ENDPOINT=https://hf-mirror.com` and **clear all upper/lower-case http(s)_proxy**,
> otherwise 401); pip via direct pypi or a Tsinghua mirror.

---

## 2. Key pitfalls (in order of appearance, each one actually hit)

### ① transformers too old → doesn't recognize `qwen2_5_vl`
HunyuanVideo-1.5's text encoder is **MLLM = Qwen2.5-VL-7B-Instruct** (not T5).
```
ValueError: The checkpoint ... model type `qwen2_5_vl` but Transformers does not recognize this architecture
```
**Cause**: `transformers < 4.49` has no `qwen2_5_vl`. **Fix**: upgrade to ≥4.49 (we use 4.56.0).

### ② transformers too new (4.51+) with old torch (≤2.5) → `torch.int1`
```
AttributeError: module 'torch' has no attribute 'int1'   → Failed to import transformers.models.t5
```
**Cause**: transformers 4.51+ references the sub-byte dtype (`torch.int1`) that only exists in torch 2.6+.
**Fix**: either upgrade torch to 2.9.1 (recommended, pairs with transformers 4.56), or downgrade transformers to 4.49.0 (which can also import with torch 2.5.1). **The versions must match**.

### ③ flash-attn ABI mismatch / diffusers imports it unconditionally → crash on startup
```
ImportError: flash_attn_2_cuda...so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs
```
**Cause**: flash-attn is a C++ extension **compiled for one specific torch version**. A flash_attn compiled with torch 2.5.1 hits an undefined symbol when loaded under torch 2.9.
**And** `diffusers 0.35`'s `attention_dispatch.py` does `find_spec("flash_attn")` and **imports it unconditionally** if found—so one bad flash_attn makes `import diffusers` crash outright.
**Fix** (one of two):
- **Compile on-site**: `pip install flash-attn --no-build-isolation` (recompile for the current torch 2.9.1, recommended, faster + saves memory);
- **Uninstall**: `pip uninstall -y flash-attn` (`find_spec` returns None, diffusers skips it; the model's `is_flash2_available()` returns False → falls back to torch SDPA, runs but slower).

> Conclusion: **flash-attn must be compiled against the same torch version**. Whenever you change torch, you must recompile flash-attn.

### ④ flash-attn compiles slowly / needs nvcc
flash-attn compiles kernels for all CUDA architectures from source, ~150-250 `.o` files, 10-40min. To speed up:
- `export MAX_JOBS=16` (limit concurrency, too large causes OOM during compilation);
- compile only the target architecture: `export TORCH_CUDA_ARCH_LIST="9.0"` (H200=Hopper, saves 2/3 of the time).
- compilation can run on any node (just install into the shared conda env, GPU nodes can use it).

### ⑤ runs even without flash-attn (torch 2.9 SDPA + gradient checkpointing)
The model's attention uses `F.scaled_dot_product_attention`; torch 2.9 ships with a memory-efficient backend;
combined with **gradient checkpointing** (`hunyuan/modules/model.py`'s `forward_bi` already wraps all 54 double_blocks with
`torch.utils.checkpoint`, and `stage1_hy15.sh` defaults to `--gradient_checkpointing`), 480×832×77f
(latent 20×30×52 ≈ 31k tokens) fits on an H200. **Without gradient checkpointing it OOMs (137G)**.

### ⑥ gradient checkpointing's `CheckpointError: different metadata`
```
torch.utils.checkpoint.CheckpointError: Recomputed values ... have different metadata than during the forward pass
```
**Cause**: under autocast/SDPA the tensor stride has benign tiny differences when recomputed. **Fix**: pass
`determinism_check="none"` to the checkpoint call (also set `use_reentrant=False`). Already fixed in the model.

---

## 3. Verification (self-check after install)

```python
import torch, transformers, diffusers
print(torch.__version__, transformers.__version__, diffusers.__version__)
# expected: 2.9.1+cu128  4.56.0  0.35.0

# qwen2_5_vl registered
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
assert "qwen2_5_vl" in M

# flex_attention (used by stage2 DMD teacher-forcing) —— this API only exists in torch 2.9
from torch.nn.attention.flex_attention import BlockMask
import inspect; assert "seq_lengths" in inspect.signature(BlockMask.from_kv_blocks).parameters

# flash-attn (optional)
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    print("flash-attn OK")
except Exception as e:
    print("flash-attn unavailable (will fall back to torch SDPA):", e)

# model + VAE + MLLM(TextEncoder) import (at the repo root, with PYTHONPATH containing the repo root)
from hunyuan.modules.model import ARHunyuanVideo_1_5_DiffusionTransformer
from hunyuan.vae import AutoencoderKLConv3D
from hunyuan.text_encoder import TextEncoder, PROMPT_TEMPLATE
print("hunyuan model/VAE/MLLM import OK")
```

---

## 4. Weights (not environment, but often stepped on together)

`tencent/HunyuanVideo-1.5` only contains **transformer + vae + upsampler + scheduler**;
**MLLM/SigLIP/ByT5 must be downloaded separately** (see the official `checkpoints-download.md`):

```bash
export HF_ENDPOINT=https://hf-mirror.com   # domestic
hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/HunyuanVideo-1.5
# MLLM text encoder (HunyuanMLLM not released, official recommends using Qwen2.5-VL instead)
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/llm
hf download google/byt5-small --local-dir ./ckpts/HunyuanVideo-1.5/text_encoder/byt5-small
# SigLIP vision encoder (only needed for i2v; gated, requires HF token + application)
# hf download black-forest-labs/FLUX.1-Redux-dev --local-dir ./ckpts/HunyuanVideo-1.5/vision_encoder/siglip --token <YOUR_HF_TOKEN>
```

> biWM's **t2v live training** only needs transformer + vae + Qwen2.5-VL; byt5/vision are fed zeros,
> **no gated SigLIP needed, and no modelscope Glyph needed** (see `make_live_batch` in `fastvideo/train_hunyuan.py`).

---

## 5. Quick reference: error → fix

| Error | Fix |
|---|---|
| `model type qwen2_5_vl ... not recognize` | upgrade transformers ≥4.49 (use 4.56) |
| `torch has no attribute 'int1'` | match torch/transformers (torch 2.9.1 + tf 4.56; or downgrade tf to 4.49) |
| `flash_attn...undefined symbol` | recompile flash-attn for the current torch, or uninstall it and fall back to SDPA |
| `import diffusers` crashes in flash_attn | same as above (a bad flash_attn pollutes the diffusers import) |
| CUDA OOM 137G | enable `--gradient_checkpointing` (mandatory) |
| `CheckpointError: different metadata` | add `determinism_check="none"` to the checkpoint call |
| `Can't pickle ... lambda` | use a module-level function (not a lambda) for the DataLoader collate_fn |
| `ProcessGroupNCCL ... no GPUs found` | launch on a node with GPUs (login/web nodes have none) |
