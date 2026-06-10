# 🎮 BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression

> ***The first full-stack open-source framework for interactive video world models under the bidirectional autoregressive paradigm.***

<p align="center">
  <a href="https://arxiv.org/abs/2606.10135"><img src="https://img.shields.io/badge/arXiv-2606.10135-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2603.25730"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-FFD21E?style=for-the-badge&logoColor=black" alt="HF Paper"></a>
  <a href="assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-7C3AED?style=for-the-badge" alt="License"></a>
</p>

**BiWM** is our contribution to the world-model community: a **full-stack open-source framework** that turns a pretrained **bidirectional** video diffusion backbone into an **action / camera-controllable interactive video world model** under the **bidirectional autoregressive** paradigm — jointly optimizing generation quality and inference speed, in just **two training stages** (vs. four in causal-attention pipelines), converging in a few hundred steps on 8×H200 GPUs. We hope more researchers and developers join us in growing the community together.

## ▶️ Demo

https://github.com/user-attachments/assets/e0f8de57-bc5e-4377-9db5-1dc581eacf03

## 🌟 Highlights

- **Bidirectional attention over history (not causal attention).** The rollout is autoregressive — generated chunk by chunk, so it *is* causal in time. The distinction is the **attention**: like Yume-1.5 and Matrix-Game-3.0, each step attends to the current chunk **and its history with full bidirectional attention**, instead of the **unidirectional (causal) attention** that prior causal world models use to read history. This full-attention history interaction gives self-correcting error propagation and stable long-horizon rollout, avoiding the error accumulation that makes causal-attention pipelines trail in quality. Existing open-source frameworks (e.g., minWM) use only causal-attention models.
- **Two stages instead of four.** From a pretrained backbone: (1) inject camera control by fine-tuning, then (2) a few-step **Distribution Matching Distillation (DMD)** stage that turns the backbone into an action/camera-controllable world model. No separate autoregressive-training or causal-initialization stages.
- **One recipe, many backbones.** A single recipe spans **Wan2.1-1.3B, Wan2.2-5B, HunyuanVideo-1.5-8B, and LTX-2.3-22B**, and also supports secondary fine-tuning of existing bidirectional models.
  - *Recommended backbone (by overall quality):* **Wan2.2 TI2V-5B > LTX-2.3-22B > HunyuanVideo-1.5 > Wan2.1-1.3B**.
- **Real-world camera control.** BiWM enables real-world camera control where minWM loses controllability, via discrete text camera actions decoupled from scene content.
- **t2v / i2v / v2v in one model.** Stage 1 mixes text-to-video, image-to-video and video-to-video in a *single* training run (`--training_mode hybrid`). i2v and v2v are just t2v conditioned on a few clean leading latent frames, so **i2v and v2v emerge naturally on top of t2v** — even under the bidirectional autoregressive setup — with no extra models, losses, or stages.
- **Long rollouts.** Pluggable history compression (**FramePack-style** and **PackForcing-style**) for long-horizon generation.
- **Efficient inference.** Optional **NVFP4** 4-bit training/inference pipeline (and FP8-E4M3 for Hopper/H200).
- **Quality-preserving distillation.** To counter DMD's mode-seeking degradation, BiWM adds **GAN** and **mass-covering forward-KL** objectives that preserve scene dynamics.

## 🧩 Pipeline

### Stage 1 — Camera-control fine-tuning
From a pretrained bidirectional video backbone, inject **discrete text camera control** (81 combined camera actions, `action_label = trans*9 + rot`) via per-latent-frame cross-attention, decoupled from the scene caption.

FramePack-style history compression can also be used here in Stage 1 (not only Stage 2): the compressed history is fed as a conditioning prefix so Stage 1 can train on **longer videos** — i.e. **v2v training** (continue from a clean history clip), keeping the token budget bounded as the horizon grows.

### Stage 2 — Few-step DMD distillation (bidirectional → autoregressive)
A Distribution Matching Distillation stage with three models inheriting the Stage-1 weights:
- **real_score** (teacher, frozen) — provides the data score (CFG).
- **fake_score** (critic, online) — learns the current generator distribution.
- **generator** (student, online) — few-step **chunk-by-chunk autoregressive** rollout.

Optional GAN loss, mass-covering forward-KL, and pluggable history compression (`none` / `framepack`).

### (Optional) Quantized inference
Wrap the generator with **NVFP4** (Blackwell FP4 kernels) or **FP8-E4M3** (Hopper) quantization, QAT + self-distillation KL alignment for real inference speedup.

## 🪶 Minimal by design

BiWM is intentionally **minimal**: it reuses the native backbone architecture as much as possible and adds camera control / history / distillation as small, switchable modules — no heavy framework, no deep abstraction layers. Taking **Wan2.2 TI2V-5B** as the example, the **entire pipeline is just a handful of core `.py` files**:

| File | Role |
|---|---|
| `pipelines/wan/train_stage1.py` | **Stage 1** training entry — cam-text multi-step diffusion fine-tuning |
| `pipelines/wan/train_stage2.py` | **Stage 2/3** training entry (`--dmd_distill`) — DMD distillation loop |
| `pipelines/wan/dmd_core.py` | DMD core — autoregressive generator rollout + DMD loss |
| `pipelines/wan/infer_stage2.py` | Inference — chunk-by-chunk 4-step autoregressive rollout |
| `wan/modules/model.py` | The `WanModel` — cam-text injection, 3D axial RoPE, history mem prefix |
| `pipelines/dataset/biwm_camera_text_dataset.py` | Discrete camera-text dataset |

Everything else is optional (history compression, quantization, GAN, other backbones). Read these few files and you understand the whole framework.

## 🧰 Environment setup

BiWM spans multiple backbones with different dependency stacks. Use **one conda environment per
backbone family** — the Wan and HunyuanVideo-1.5 stacks need different `torch` / `transformers` /
`diffusers` versions and must not be mixed.

### Wan2.1 (1.3B) & Wan2.2 (TI2V-5B)

> **Wan2.1 and Wan2.2 run the exact same BiWM pipeline** — identical entrypoints, training loop, and
> `wan/` model code. The **only differences are the VAE** (Wan2.1 VAE vs. Wan2.2 VAE) **and a few
> model-loading parameters** (`--pretrained_model_path` / `--vae_path` / `--text_encoder_path` point
> at the respective base checkpoint). Choose the base weights for the size you want; everything else
> is the same — which is why both share one environment below.

Both Wan backbones share a single environment (same `wan/` model code and training pipeline).
Verified combination (CUDA 12.4, sm_80/sm_90 — A100 / H100 / H200):

| Component | Version |
|---|---|
| python | 3.10 |
| torch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| transformers | 4.44.2 |
| diffusers | 0.31.0 |
| accelerate | 1.13.0 |
| tokenizers | 0.19.1 |
| numpy | 1.26.4 |
| peft | 0.19.1 |
| flash-attn | 2.8.3 (built for torch 2.5.1) |
| torchao | 0.17.0 (NVFP4 / FP8 quantization) |

```bash
conda create -n biwm-wan python=3.10 -y
conda activate biwm-wan
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.44.2 diffusers==0.31.0 accelerate==1.13.0 \
            tokenizers==0.19.1 numpy==1.26.4 peft==0.19.1 torchao==0.17.0 \
            easydict decord imageio imageio-ffmpeg opencv-python
pip install ninja packaging wheel
pip install flash-attn==2.8.3 --no-build-isolation   # build against the installed torch
```

> **NVFP4** needs Blackwell (e.g. RTX 5090) for the real FP4 kernel; **FP8-E4M3** needs Hopper
> (`torch._scaled_mm`). On other GPUs the quantized graphs still run in fake-quant mode for QAT,
> just without the real-kernel speedup.

### HunyuanVideo-1.5 (HY15)

HY15's text encoder is a **Qwen2.5-VL MLLM** (not T5), which forces a newer stack
(CUDA 12.8, Hopper H200). Verified combination:

| Component | Version |
|---|---|
| python | 3.10 |
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| transformers | 4.56.0 (≥4.49 for `qwen2_5_vl`) |
| diffusers | 0.35.0 |
| torchao | 0.15.0 |
| flash-attn | 2.8.3 (built for torch 2.9.1) |

```bash
conda create -n biwm-hy15 python=3.10 -y
conda activate biwm-hy15
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.0 diffusers==0.35.0 torchao==0.15.0
export MAX_JOBS=16 TORCH_CUDA_ARCH_LIST="9.0"   # H200 = Hopper
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation     # MUST be rebuilt for torch 2.9.1
```

> **Do not mix versions:** `transformers < 4.49` does not know `qwen2_5_vl`; `transformers ≥ 4.51`
> on `torch ≤ 2.5` fails with `torch has no attribute 'int1'`; a `flash-attn` built for a different
> torch poisons `import diffusers` (0.35 imports it unconditionally if present) — rebuild flash-attn
> whenever torch changes, or uninstall it to fall back to torch SDPA. See
> [`hunyuan/ENV_SETUP.md`](hunyuan/ENV_SETUP.md) for the full setup guide (flash-attn build flags,
> weight download, gradient-checkpointing OOM fixes, error→fix table).

> **Offline / China clusters:** fetch weights through the HF mirror
> (`export HF_ENDPOINT=https://hf-mirror.com` and **clear all upper/lower-case `http(s)_proxy`**,
> else 401); install pip packages from PyPI directly or a mirror.

## 🔤 Data encoding: online vs. pre-encoded

Every backbone **except LTX** can be trained in two interchangeable data modes — encode on the fly,
or load embeddings/latents precomputed to disk. **Text embeddings and VAE latents follow the same
two-mode choice**; pre-encoding trades disk for a large training speedup (no per-step text-encoder /
VAE forward pass).

- **Online (live)** — the dataloader decodes each `.mp4` and runs the **text encoder + VAE live**
  every step. Simplest to start; needs the encoder/VAE weights resident in GPU memory.
- **Pre-encoded (offline cache)** — a one-off preprocessing pass runs the text encoder + VAE once,
  writes `{text_embedding, vae_latent, action_label, caption}` to disk, and training just loads them.

### HunyuanVideo-1.5

Toggled by a single flag — `--data_mode live` vs. `--data_mode preenc`:

| Mode | Flag | Text encoder (Qwen2.5-VL MLLM) | VAE | Data read from |
|---|---|---|---|---|
| Online | `--data_mode live` | `--text_encoder_path ckpts/HunyuanVideo-1.5/text_encoder/llm` | `--vae_path ckpts/HunyuanVideo-1.5/vae` | raw `.mp4` (e.g. `dataset/video_real`) |
| Pre-encoded | `--data_mode preenc --preencoded_dir dataset/preenc_hy15/<set>` | (used once, offline) | (used once, offline) | `dataset/preenc_hy15/{videos,video_real}/*.pt` |

Build the cache with `scripts/hy15/preencode_hy15.sh` → `pipelines/data_preprocess/preencode_hy15.py`
(each `.pt` holds `latent`, `prompt_embed`, `prompt_embed_capt`, `action_label`, `caption`, `vid_id`).

### Wan2.1 / Wan2.2

| Mode | Text encoder | VAE | Data read from |
|---|---|---|---|
| Online | T5 / UMT5 from `--text_encoder_path <base_ckpt>` | `--vae_path <base_ckpt>` (Wan2.1 VAE / Wan2.2 VAE) | raw `.mp4` `dataset/videos` (+ `videos_syn.json`) via `BiwmCameraTextDataset` |
| Pre-encoded | (used once, offline) | (used once, offline) | LMDB of precomputed latents + prompt embeds |

The Wan online path is `pipelines/dataset/biwm_camera_text_dataset.py` (`BiwmCameraTextDataset`). The
Wan **pre-encoded** path follows the LMDB latent-cache recipe from
[minWM](https://github.com/shengshu-ai/minWM) (`Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py`
→ `Wan21/wan_utils/dataset.py`): VAE-encode the videos once into a single LMDB
(`latents` / `prompts` rows), then train off that cache. Point `--vae_path` at the matching VAE
(Wan2.1 `Wan2.1_VAE.pth` or the Wan2.2 VAE) when building it.

> **LTX** does not yet support either preprocessing mode (code release pending).

## 🧭 Getting started — full reproduction (Wan2.1 / Wan2.2)

End-to-end recipe on **Wan2.2-TI2V-5B**. **Wan2.1 runs the identical pipeline** — same entrypoints, training loop and `wan/` model code; only the **base weights / VAE** differ. Use the scripts under `scripts/wan21/` (same flags) and point the `--pretrained_model_path` / `--vae_path` / `--text_encoder_path` at your Wan2.1 checkpoint.

All commands run from the repo root, in the `biwm-wan` conda env (see [Environment setup](#️-environment-setup)). 8×H200 is the reference config; scale `--nproc_per_node` / `WORLD_SIZE` to your node.

### Step 1 — Download base weights

```bash
# (China clusters: export HF_ENDPOINT=https://hf-mirror.com and clear http(s)_proxy)
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ckpts/Wan2.2-TI2V-5B
```

All Wan scripts expect `./ckpts/Wan2.2-TI2V-5B`, which must contain the DiT weights, the VAE (`Wan2.2_VAE.pth`), the T5/UMT5 text encoder (`models_t5_umt5-xxl-enc-bf16.pth`) and tokenizer (`google/umt5-xxl/`). For Wan2.1, download the Wan2.1 base + `Wan2.1_VAE.pth` and adjust the paths.

### Step 2 — Prepare data

**Option A — download our released dataset from 🤗 Hugging Face** ([`shaohao011/BiWM`](https://huggingface.co/datasets/shaohao011/BiWM)). Two `.tar` archives + their caption/pose JSONs:

```bash
# (China clusters: export HF_ENDPOINT=https://hf-mirror.com and clear http(s)_proxy)
hf download shaohao011/BiWM --repo-type dataset --local-dir dataset
cd dataset
tar -xf videos_syn.tar       # → dataset/videos/<id>_<token>/gen.mp4   (synthetic, 19823 clips)
tar -xf videos_real.tar      # → dataset/video_real/<id>_.../gen.mp4   (real game, 5066 clips)
cd ..
# yields: dataset/{videos, video_real, videos_syn.json, videos_real.json}
```

- **`videos_syn`** (synthetic) is sourced from the dataset released by [minWM](https://github.com/shengshu-ai/minWM).
- **`videos_real`** is real game footage with per-clip 81-class combined-camera labels.

**Option B — bring your own** — BiWM is data-agnostic; arrange clips in this layout (one directory per clip, named `<6-digit id>_<camera-token-string>`):

```
dataset/videos/000013_up4right8s7/gen.mp4      # 77 frames, 832×480, 24fps
dataset/videos_syn.json                   # list, index == 6-digit id
   → [{"caption": "...static scene only...", "action_frames": "up-4, right-8, s-7"}, ...]
```

Two dataset flavors (read by `pipelines/dataset/biwm_camera_text_dataset.py`):
- **`dataset/videos`** + `videos_syn.json` — synthetic clips with per-segment `action_frames` (single-axis tokens `w/s/a/d` = translate, `up/down` = pitch, `right/left` = yaw; `token-N` = action lasts N latent frames; segments sum to 19).
- **`dataset/video_real`** + `videos_real.json` — real clips, one **constant 81-class** combined action per clip, written as a numeric `action_frames` segment `"<action_label>-<n>"` (e.g. `"38-19"`; `action_label = trans*9 + rot`).

`caption` describes **only the static scene** (no camera/character motion) — camera is carried entirely by the discrete `action_frames` field, decoupled from the caption.

### Step 3 (optional) — Auto-caption with a VLM

If your clips have no captions, generate static-only captions with Qwen3-VL (writes `caption` back into the json in place; resumable):

```bash
bash scripts/wan22/caption_video_real_qwen.sh    # → pipelines/data_preprocess/caption_video_real_qwen.py
```

### Step 4 — Stage 1: camera-control fine-tuning

Online encoding (raw `.mp4` → VAE + T5 encoded live each step — no offline cache needed; for the pre-encoded LMDB path see [Data encoding](#-data-encoding-online-vs-pre-encoded)).

**One script, two interchangeable datasets** — pick the dataset purely via env vars (no separate script):

```bash
# Synthetic clips (dataset/videos + videos_syn.json) — the default
bash scripts/wan22/stage1_pretrain.sh

# Real game footage (dataset/video_real + videos_real.json) — same script, switch via env
BIWM_VIDEO_DIR=./dataset/video_real \
BIWM_CAPTION_JSON=./dataset/videos_real.json \
LOG_NAME=Wan22_5B_video_real \
  bash scripts/wan22/stage1_pretrain.sh
```

Outputs to `logs/wan22/stage1/checkpoint-*/`. The dataset is selected entirely by the env knobs `BIWM_VIDEO_DIR` / `BIWM_CAPTION_JSON` (+ `LOG_NAME`); `stage2_dmd.sh` takes the same knobs (it defaults to `dataset/video_real`).

> **Limited GPU memory?** Pass `--use_lora` to fine-tune low-rank adapters instead of the full DiT (parameter-efficient; far less memory than full-parameter training).

> **⭐ Mixed t2v / i2v / v2v — our feature.** Stage 1 takes `--training_mode hybrid` and, **per training step**, randomly picks the task by changing only the number of clean *conditioning* latent frames on the same video: `--i2v_prob` → i2v (`--i2v_cond_latent_frames`, default 1 clean frame), `--v2v_prob` → v2v (`--v2v_cond_ratio` clean prefix), the rest → t2v (0 clean frames). So **i2v and v2v are t2v with a conditioning prefix and emerge naturally on top of t2v** — one model, no extra stages, and the behavior carries through to the bidirectional autoregressive student in Stage 2. The shipped scripts default to `--i2v_prob 0.0 --v2v_prob 0.0` (pure t2v); set e.g. `--i2v_prob 0.3 --v2v_prob 0.2` to train all three jointly.

### Step 5 — Stage 2: DMD distillation (50-step bidirectional → 4-step autoregressive)

Three models (teacher/critic/generator) all inherit the Stage-1 weights; the generator becomes a chunk-by-chunk 4-step autoregressive world model.

```bash
# point at your Stage-1 run; the script auto-picks the latest checkpoint-N
STAGE1_DIR=./logs/wan22/stage1 bash scripts/wan22/stage2_dmd.sh   # STAGE1_DIR defaults to ./logs/wan22/stage1
```

It calls `pipelines/wan/train_stage2.py --dmd_distill` with `--generator_ckpt <ckpt dir>` + `--real_score_ckpt <ckpt>/diffusion_pytorch_model.safetensors`. Pluggable history compression via `--dmd_history_mode none|framepack` (thin wrapper: `stage2_dmd_framepack.sh`); enable the GAN objective with `--dmd_use_gan` (+ `--dmd_gan_weight`) — it's just an extra loss, not a separate script.

### Step 6 (optional) — Stage 3: quantization-aware distillation

QAT on top of the Stage-2 generator (teacher/critic stay at the Stage-1 50-step data-score model):

```bash
STAGE2_DIR=./logs/wan22/stage2 STAGE1_DIR=./logs/wan22/stage1 \
  bash scripts/wan22/stage3_nvfp4_optional.sh      # NVFP4 (Blackwell) — or stage3_fp8_optional.sh (Hopper/H200)
```

### Step 7 — Inference

```bash
# autoregressive 4-step rollout (the distilled world model)
python pipelines/wan/infer_stage2.py \
    --generator_ckpt ./logs/wan22/stage2/checkpoint-XXXX \
    --wan_base ./ckpts/Wan2.2-TI2V-5B \
    --mode t2v --prompt "..." --action_frames 'w-8, right-12, s-6' \
    --duration_sec 60 --chunk_size 4 --max_chunks 5 --sink_chunks 1 \
    --output ./outputs/dmd_infer.mp4
```

- **i2v**: `--mode i2v --i2v_video <clip.mp4>`; random camera: leave `--action_frames` empty; constant action: `--action_label <0-80>`.
- **Quantized inference**: add `--nvfp4 --nvfp4_kernel` (Blackwell) or `--fp8 --fp8_kernel` (Hopper); optionally `--torch_compile`.
- **Sanity-check Stage 1** (non-autoregressive 50-step CFG teacher, whole clip at once): `python pipelines/wan/infer_stage1.py --generator_ckpt <stage1 ckpt> --wan_base ./ckpts/Wan2.2-TI2V-5B --prompt "..." --action_frames '...'`.
- **Batch cases**: `python pipelines/data_preprocess/build_infer_cases.py --out_dir <dir>` generates case JSONs, then pass `--cases_json <file> --case_index <i>`.

## 🗺️ Roadmap

Feature coverage per backbone (✅ open-sourced · 🚧 in progress · ⬜ planned):

| Backbone | Camera-control fine-tuning | DMD (Self-Forcing) distillation | FramePack-style compression | PackForcing-style compression | NVFP4 / FP8 (QAT + inference) |
|---|:--:|:--:|:--:|:--:|:--:|
| **Wan2.1 (1.3B)** | ✅ | ✅ | ✅ | ⬜ | ✅ |
| **Wan2.2 (5B)** | ✅ | ✅ | ✅ | ⬜ | ✅ |
| **HunyuanVideo-1.5** | ✅ | 🚧 | ⬜ | ⬜ | ⬜ |
| **LTX** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

- **Wan2.1 (1.3B) & Wan2.2 (5B):** full pipeline open-sourced — camera-control fine-tuning, DMD
  (Self-Forcing) distillation, FramePack-style history compression, and NVFP4 / FP8 quantized
  distillation + inference. PackForcing-style compression: code releasing soon.
- **HunyuanVideo-1.5:** camera-control fine-tuning open-sourced; DMD distillation in progress.
- **PackForcing-style history compression:** code releasing soon.
- **LTX:** to be released.

## 🗂️ Repository layout

| Path | Purpose |
|---|---|
| `pipelines/wan/` | Wan backbone: training entry, DMD core, history compression, autoregressive inference |
| `pipelines/hy15/` | HunyuanVideo-1.5 backbone adaptation |
| `pipelines/common/` | Shared components: optimizer (muon), quantization (NVFP4 / FP8), control utilities |
| `pipelines/common/camera_control.py` | Continuous-camera → discrete-text conversion: `c2w_to_action_labels` maps a continuous extrinsics (c2w) sequence to the 81-class discrete camera actions used as cam-text |
| `pipelines/dataset/`, `pipelines/data_preprocess/` | Discrete camera-text dataset, VAE/text pre-encoding |
| `wan/`, `hunyuan/` | Backbone model definitions (cam-text injection, 3D RoPE, history mem prefix) |
| `ADD/` | Adversarial diffusion distillation (ProjectedDiscriminator) for the GAN objective |
| `scripts/wan21/`, `scripts/wan22/`, `scripts/hy15/` | Training / distillation / inference shell entrypoints per backbone |

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{rui2026biwm,
  title={BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression},
  author={Rui, Shaohao and Mao, Xiaofeng and Zhang, Zhanyu and Lin, Peijia and Zhu, Yansong and Zhang, Yibo and Wan, Haibin and Ma, Weijie},
  journal={arXiv preprint arXiv:2606.10135},
  year={2026}
}

@article{mao2026packforcing,
  title={Packforcing: Short video training suffices for long video sampling and long context inference},
  author={Mao, Xiaofeng and Rui, Shaohao and Ying, Kaining and Zheng, Bo and Li, Chuanhao and Chi, Mingmin and Zhang, Kaipeng},
  journal={arXiv preprint arXiv:2603.25730},
  year={2026}
}
```

## 💬 Community

Join our WeChat group for discussion, questions, and updates:

<p align="center">
  <img src="assets/wechat.jpg" alt="BiWM WeChat group" width="280">
</p>

## 🙏 Acknowledgements

BiWM builds upon and is grateful to these excellent open-source projects:
[FastVideo](https://github.com/hao-ai-lab/FastVideo),
[Yume-1.5](https://github.com/stdstu12/YUME),
[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5),
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
[Wan2.2](https://github.com/Wan-Video/Wan2.2),
[LTX-Video](https://github.com/Lightricks/LTX-Video),
and [minWM](https://github.com/shengshu-ai/minWM).

## 📜 License

Released under the [Apache License 2.0](LICENSE).
