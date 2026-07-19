# 🎮 BiWM: Bidirectional Autoregressive Video World Models

> **The first open-source training framework for bidirectional autoregressive video world models.**

<p align="center">
  <a href="https://arxiv.org/abs/2606.10135"><img src="https://img.shields.io/badge/arXiv-2606.10135-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2603.25730"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-FFD21E?style=for-the-badge&logoColor=black" alt="HF Paper"></a>
  <a href="assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-7C3AED?style=for-the-badge" alt="License"></a>
</p>

BiWM turns a pretrained bidirectional video diffusion model into an action-controllable,
autoregressive world model in two training stages:

1. **Camera-control fine-tuning** with 81 discrete camera actions.
2. **Few-step DMD distillation** for chunk-by-chunk autoregressive generation.

Unlike causal-attention pipelines, BiWM keeps full bidirectional attention inside each current
chunk and its history. It supports t2v, i2v, and v2v conditioning in one model.

## Demo

https://github.com/user-attachments/assets/e0f8de57-bc5e-4377-9db5-1dc581eacf03

## Supported backbones

| Backbone | Stage 1 | Stage 2 DMD | Quantized inference |
|---|:--:|:--:|:--:|
| Wan2.1-1.3B | ✅ | ✅ | FP8 / NVFP4 |
| Wan2.2-TI2V-5B | ✅ | ✅ | FP8 / NVFP4 |
| HunyuanVideo-1.5-8B | ✅ cam-text + discrete action | ✅ | — |
| LTX-Video 2.3-22B | ✅ | ✅ | — |

## Setup

Use separate environments for Wan and HY15/LTX23.

### Wan2.1 / Wan2.2

```bash
conda create -n biwm-wan python=3.10 -y
conda activate biwm-wan
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.44.2 diffusers==0.31.0 accelerate==1.13.0 \
  tokenizers==0.19.1 numpy==1.26.4 peft==0.19.1 torchao==0.17.0 \
  easydict decord einops safetensors imageio imageio-ffmpeg opencv-python
```

### HunyuanVideo-1.5 / LTX-2.3

```bash
conda create -n biwm-hy15 python=3.10 -y
conda activate biwm-hy15
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.0 diffusers==0.35.0 torchao==0.15.0 \
  decord einops safetensors imageio imageio-ffmpeg opencv-python
```

HY15 uses Qwen2.5-VL; LTX23 uses Gemma-3. See [hunyuan/ENV_SETUP.md](hunyuan/ENV_SETUP.md)
for HY15-specific setup and troubleshooting.

## Data

Download the released dataset:

```bash
hf download shaohao011/BiWM --repo-type dataset --local-dir dataset
tar -xf dataset/videos_syn.tar -C dataset
tar -xf dataset/videos_real.tar -C dataset
```

Expected layout:

```text
dataset/
├── videos/                 # synthetic clips
├── videos_syn.json
├── video_real/             # real-game clips
└── videos_real.json
```

Each JSON record contains a scene-only `caption` and `action_frames`. Camera action
`action_label = translation * 9 + rotation`, producing 81 combined actions.

## Training

All scripts run from any working directory, resolve the repository root automatically, and allow
paths to be overridden through environment variables.

### Wan2.2

```bash
bash scripts/wan22/stage1_pretrain.sh
STAGE1_DIR=./logs/wan22/stage1 bash scripts/wan22/stage2_dmd.sh
```

Wan2.1 uses the corresponding scripts under `scripts/wan21/`.

### HunyuanVideo-1.5 discrete actions

Place the base weights under `ckpts/HunyuanVideo-1.5`, then run:

```bash
bash scripts/hy15/stage1_action.sh
STAGE1_DIR=./logs/hy15/stage1_action bash scripts/hy15/stage2_dmd_action.sh
```

The action path injects per-latent-frame discrete actions into AdaLN while keeping captions free of
camera descriptions.

### LTX-Video 2.3

```bash
export LTX2_CKPT=./ckpts/LTX-Video-2.3/ltx-2.3-22b-dev.safetensors
export GEMMA_PATH=./ckpts/LTX-Video-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized
bash scripts/ltx23/stage1.sh
STAGE1_DIR=./logs/ltx23/stage1 bash scripts/ltx23/stage2_dmd.sh
```

Common overrides: `BIWM_VIDEO_DIR`, `BIWM_CAPTION_JSON`, `OUTPUT_DIR`,
`PYTHON_BIN`, `MAX_TRAIN_STEPS`, and `CHECKPOINTING_STEPS`.

## Inference

Wan distilled-model inference:

```bash
python pipelines/wan/infer_stage2.py \
  --generator_ckpt ./logs/wan22/stage2/checkpoint-XXXX \
  --wan_base ./ckpts/Wan2.2-TI2V-5B \
  --mode t2v \
  --prompt "A character explores a forest" \
  --action_frames 'w-8, right-12, s-6' \
  --output ./outputs/dmd_infer.mp4
```

Add `--nvfp4 --nvfp4_kernel` on Blackwell or `--fp8 --fp8_kernel` on Hopper.

## Repository layout

| Path | Purpose |
|---|---|
| `pipelines/wan/` | Wan Stage 1, DMD, inference, and compression |
| `pipelines/hy15/` | HY15 Stage 1 and DMD |
| `pipelines/ltx23/` | LTX23 Stage 1 and DMD |
| `pipelines/common/` | Shared DMD, optimizer, GAN, control, and quantization code |
| `wan/`, `hunyuan/`, `ltx23/` | Backbone implementations |
| `scripts/` | Public training and inference entrypoints |

## Citation

```bibtex
@article{rui2026biwm,
  title={BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression},
  author={Rui, Shaohao and Mao, Xiaofeng and Zhang, Zhanyu and Lin, Peijia and Zhu, Yansong and Zhang, Yibo and Wan, Haibin and Ma, Weijie},
  journal={arXiv preprint arXiv:2606.10135},
  year={2026}
}
```

## Acknowledgements

Built upon [FastVideo](https://github.com/hao-ai-lab/FastVideo),
[minWM](https://github.com/shengshu-ai/minWM),
[Wan](https://github.com/Wan-Video/Wan2.2),
[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), and
[LTX-Video](https://github.com/Lightricks/LTX-Video).

## License

Released under the [Apache License 2.0](LICENSE).
