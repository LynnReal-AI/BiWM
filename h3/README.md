# MiniMax-H3 in BiWM

BiWM now includes a **camera-only MiniMax-H3 Stage 1 training implementation**,
adapted from the `stage0.5_v7` training source. This release provides native
bidirectional flow matching with ProPE camera injection. FramePack v2v adaptation
and autoregressive DMD are subsequent stages and are not included in this release.

The H3 roadmap has three stages:

1. Camera-conditioned flow matching (this release).
2. Merge the SolarWM camera LoRA into the base, then train a fresh LoRA for
   FramePack v2v history conditioning.
3. Use the adapted model for bidirectional autoregressive rollout and DMD,
   following the Wan2.2-TI2V-5B BiWM rollout strategy.

These names are specific to the H3 release; other backbones retain their two-stage
interfaces. H3 Stage 2 is not the existing AnyFlow/MeanFlow training recipe.

## Scope

The original `stage0.5_v7` experiment trained the full 33B transformer with HAND
conditions and auxiliary losses. This implementation intentionally simplifies it:

- Input: cached Qwen(first RGB frame + caption), an independently encoded VAE
  first-frame anchor, target video latents, and aligned camera poses.
- No HAND inputs, reference-video Qwen encoding, ROI, LPIPS, DINO, audio tokens,
  auxiliary losses, or extra action-embedding modules.
- Native H3 MM-RoPE on head dimensions `[0:96)`, camera ProPE on `[96:128)`,
  first-frame-relative W2C, fixed Wan focal intrinsics, and logd4 translation.
  Camera transforms select anchor/target rows explicitly; Qwen rows are excluded.
- Native temporal grid `5/3 * [1,4,4,4,4]`; anchor shares target frame 0's position
  and camera. Keep source frames/poses aligned without frame repetition/interpolation.
- Flow target `v = x0 - noise`; noisy target `t*x0 + (1-t)*noise`, with
  `sigma=3*u/(1+2*u)` and native H3 clean-time `t=1-sigma`.
- Qwen embeddings stay noise-free and receive target `t` through native AdaLN;
  anchor is `.999*anchor + .001*noise`, with time `.999`. MSE covers the complete
  target latent sequence, including its native tail.
- Explicit `_flash_3` through `h3/h3_fa3_policy.py`; no fallback. BF16 blocks,
  FP32 input/output/time projections, FP32 master AdamW, constant LR `1e-5`,
  betas `(0.9,0.95)`, weight decay `.01`, gradient clip `1`, no warmup or EMA.
- `h3/fa3_autograd.py` retains the native FA3 backward used by the original
  training entry. Diffusers 0.40's local FA3 custom op only wraps the forward.
  Before loading H3, every rank verifies exact forward parity and finite Q/K/V
  gradients against a small FP32 reference. CPU tests cannot replace this gate.

The original run used 24 H200s and SP2/DP12. This release uses FSDP with DP only
(SP1), batch size 1 per GPU, configurable accumulation and full or LoRA training.
LoRA defaults to rank=alpha=384 on all 312 Q/K/V/O and FFN linears. An 8×H200
launch is provided; its capacity and quality require GPU validation for the chosen
sequence length. CPU checks do not establish full-run equivalence to v7.

## Environment

Use a separate Python 3.10 environment with CUDA-enabled PyTorch and an H3-capable
Diffusers build:

```bash
pip install -r h3/requirements.txt
# Install flash-attn-3==3.0.0 compiled for Hopper/sm90a and your CUDA/PyTorch build.
python -c 'from diffusers import MiniMaxH3Transformer3DModel; import flash_attn_3'
```

Obtain the base weights from [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3).
The checkpoint root must contain `transformer/`. No model weights, datasets,
machine-specific absolute paths or cluster credentials are shipped with this code.

## Data

The trainer consumes a JSONL manifest, with paths relative to the manifest:

```json
{"sample_id":"clip_0001","path":"clip_0001.safetensors"}
```

Each file contains exactly these tensors (no batch dimension):

| Key | Shape / meaning |
|---|---|
| `target_latents` | `[24,T,H,W]`, native H3 VAE encoding of the full target |
| `anchor_latents` | `[24,1,H,W]`, independent encoding of target RGB frame 0 |
| `prompt_embeds` | `[N,5120]`, native Qwen first-image + caption embeddings |
| `text_token_tags` | `[N]`, int64; preserve Qwen image=0/text=1 tags |
| `camera_viewmats` | `[T,4,4]`, first-relative W2C before logd4 |
| `camera_K` | `[T,3,3]`, aligned source intrinsics, validated for provenance |

Use complete native lengths `T=5*n+2`, `n>=1`; for example 124 RGB frames encode
to 37 latents. Native 768×1344 uses latent `H=48,W=84`. The trainer does not
load Qwen or VAE during optimization. Preencode using the native H3 components;
do not copy a HAND-conditioned Qwen cache into this format or replace the full
target's latent0 with the separately encoded anchor.

Existing caches can be stripped to this format with:

```bash
python -m pipelines.h3.prepare_stage1 --input source.jsonl --output camera_cache
```

Source rows contain `path`, optional `camera_path` (a safetensors file providing
the two already aligned camera tensors), and an audited
`qwen_skip_reference_videos: true`. This converter removes unused cache fields;
it does not encode raw RGB or guess camera trajectories from discrete labels.
ProPE uses fixed focal K at runtime; actual poses still have to match the clip.

## Training and resume

Run from the repository root, or invoke the shell entrypoint from any directory:

```bash
H3_MODEL_PATH=/path/to/H3 \
H3_MANIFEST=/path/to/camera_cache/manifest.jsonl \
OUTPUT_DIR=/path/to/new/run \
NPROC_PER_NODE=8 TRAINING_MODE=lora \
bash scripts/h3/stage1.sh
```

Use `TRAINING_MODE=full` for full fine-tuning. Overrides include `LEARNING_RATE`,
`MAX_TRAIN_STEPS`, `GRAD_ACCUM`, `CHECKPOINTING_STEPS`, and `PYTHON_BIN`.
This workspace puts actual runs under
`hand_control_world/solarwm_h3_repro/outputs_official_v2/` with short version names.

Resume by passing `--resume /path/to/run/checkpoint_000100` with the same model,
dataset, mode, FSDP world size, batch and optimizer settings. Complete checkpoints
contain Live model shards, optimizer masters/moments, all rank RNGs, step and
configuration. A deterministic sample cursor is reconstructed from the complete
step. `COMPLETE.json` is written only after every rank has saved successfully.

Export using the saved world size:

```bash
torchrun --standalone --nproc_per_node=8 -m pipelines.h3.export_stage1 \
  --checkpoint /path/to/run/checkpoint_000100 --output /path/to/new/export
# Add --merge_lora to export the merged transformer for a fresh LoRA stage.
```

Exports always use Live weights. Merged exports contain `transformer/`; reuse
the base checkpoint's tokenizer, processor, text encoder and VAE for conditioning
and decoding. Export/merge needs sufficient CPU RAM for the 33B model.
Load an exported adapter with `model.load_lora_adapter(export_path, prefix=None)`
on the same base, then explicitly select `_flash_3` before training.

## Sampling and validation

`h3.flow.sample_video` implements CFG1 native Euler sampling (30 time points,
29 updates, shift3). It uses target tensor shape only to choose output geometry;
target values are not passed as conditioning. Decode output latents with the
native H3 VAE, and compare fixed-case/seed videos against original RGB.
The minimal trainer saves checkpoints; video validation is run separately.

```bash
python -m unittest discover -s tests -p 'test_h3*.py'
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/h3_fsdp_smoke.py
```

GPU execution, full FSDP resume/export, generation quality and later stages must
be assessed separately from CPU unit checks. Source hashes and extraction scope
are recorded in [SOURCE.json](SOURCE.json).

Validation on 2026-11-10: 11 CPU unit checks passed; two CPU/Gloo ranks restored
mixed BF16/FP32 FSDP model/optimizer/RNG and completed another update. A reduced
H3 topology also verified discovery of all 312 LoRA targets, adapter state
materialization from meta tensors, and FP32 CPU merge/removal of adapters.
No 33B Hopper training or video-quality result is claimed by these checks.
