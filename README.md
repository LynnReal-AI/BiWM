# 🎮 BiWM：双向自回归视频世界模型

[English](README_EN.md)

> **首个开源双向自回归视频世界模型训练框架。**

BiWM 是一个极简的 Video World Model 训练框架：将预训练双向视频扩散模型分两阶段训练为动作可控、分块自回归的世界模型。

1. Stage 1：相机控制微调，支持 81 类离散相机动作。
2. Stage 2：少步 DMD 蒸馏，实现逐块自回归生成。

模型在当前块与历史块内保留完整双向注意力，并统一支持 T2V、I2V 和 V2V 条件。

<p align="center">
  <a href="https://arxiv.org/abs/2606.10135"><img src="https://img.shields.io/badge/arXiv-2606.10135-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-7C3AED?style=for-the-badge" alt="License"></a>
</p>

## Demo

https://github.com/user-attachments/assets/e0f8de57-bc5e-4377-9db5-1dc581eacf03

## 支持模型

| Backbone | Stage 1 | Stage 2 DMD | 量化推理 |
|---|:---:|:---:|:---:|
| Wan2.1-1.3B | ✅ | ✅ | FP8 / NVFP4 |
| Wan2.2-TI2V-5B | ✅ | ✅ | FP8 / NVFP4 |
| HunyuanVideo-1.5-8B | ✅ 相机文本 / 离散动作 | ✅ | — |
| LTX-Video 2.3-22B | ✅ | ✅ | — |

## 安装

Wan 与 HY15/LTX23 建议使用独立环境。

```bash
# Wan2.1 / Wan2.2
conda create -n biwm-wan python=3.10 -y
conda activate biwm-wan
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.44.2 diffusers==0.31.0 accelerate==1.13.0 \
  tokenizers==0.19.1 numpy==1.26.4 peft==0.19.1 torchao==0.17.0 \
  easydict decord einops safetensors imageio imageio-ffmpeg opencv-python
```

```bash
# HunyuanVideo-1.5 / LTX-Video 2.3
conda create -n biwm-hy15 python=3.10 -y
conda activate biwm-hy15
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.0 diffusers==0.35.0 torchao==0.15.0 \
  decord einops safetensors imageio imageio-ffmpeg opencv-python
```

HY15 使用 Qwen2.5-VL，LTX23 使用 Gemma-3。HY15 细节见 [环境说明](hunyuan/ENV_SETUP.md)。

## 数据

```bash
hf download shaohao011/BiWM --repo-type dataset --local-dir dataset
tar -xf dataset/videos_syn.tar -C dataset
tar -xf dataset/videos_real.tar -C dataset
```

每条 JSON 数据包含场景 `caption` 与 `action_frames`；`action_label = translation * 9 + rotation`，共 81 种组合动作。

## 训练

```bash
# Wan2.2；Wan2.1 使用 scripts/wan21 下的对应脚本
bash scripts/wan22/stage1_pretrain.sh
STAGE1_DIR=./logs/wan22/stage1 bash scripts/wan22/stage2_dmd.sh

# HunyuanVideo-1.5 离散动作
bash scripts/hy15/stage1_action.sh
STAGE1_DIR=./logs/hy15/stage1_action bash scripts/hy15/stage2_dmd_action.sh

# LTX-Video 2.3
export LTX2_CKPT=./ckpts/LTX-Video-2.3/ltx-2.3-22b-dev.safetensors
export GEMMA_PATH=./ckpts/LTX-Video-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized
bash scripts/ltx23/stage1.sh
STAGE1_DIR=./logs/ltx23/stage1 bash scripts/ltx23/stage2_dmd.sh
```

常用覆盖变量：`BIWM_VIDEO_DIR`、`BIWM_CAPTION_JSON`、`OUTPUT_DIR`、`PYTHON_BIN`、`MAX_TRAIN_STEPS`、`CHECKPOINTING_STEPS`。

## 引用

```bibtex
@article{rui2026biwm,
  title={BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression},
  author={Rui, Shaohao and Mao, Xiaofeng and Zhang, Zhanyu and Lin, Peijia and Zhu, Yansong and Zhang, Yibo and Wan, Haibin and Ma, Weijie},
  journal={arXiv preprint arXiv:2606.10135},
  year={2026}
}
```

## 致谢与许可

基于 [FastVideo](https://github.com/hao-ai-lab/FastVideo)、[minWM](https://github.com/shengshu-ai/minWM)、[Wan](https://github.com/Wan-Video/Wan2.2)、[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) 与 [LTX-Video](https://github.com/Lightricks/LTX-Video) 构建。

项目采用 [Apache License 2.0](LICENSE)。
