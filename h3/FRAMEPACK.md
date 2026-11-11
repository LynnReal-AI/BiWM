# H3 Stage 2: FramePack v2v

Local release preparation. Final training/checkpoint selection is pending;
this branch has not been published.

Stage 2 merges the original SolarWM camera LoRA, then trains a fresh H3 LoRA and
the existing BiWM FramePack compressor. The only objective is current-chunk
flow matching. Training uses clean GT history; validation rolls out entirely
from generated history. ProPE covers anchor, target and compressed history at
their physical camera-frame indices. There are no HAND conditions or auxiliary
losses, and no KV cache.

The first chunk contains 7 latents (22 RGB frames); later chunks contain 5 latents
(17 RGB frames). All 9 chunks cover 47 latents / 158 RGB frames at 1344×768. Qwen encodes
the target first image and caption only; the independent VAE anchor keeps its
native `.999` time. H3 MM-RoPE and ProPE suffixes retain the Stage1 contract.

The modules are extracted from the frozen training source recorded in
[FRAMEPACK_SOURCE.json](FRAMEPACK_SOURCE.json). Four-rank CPU tests compare the
original/public loss, gradients, FP32 Adam, checkpoint-next-update and 47-latent
rollout bitwise. These tests do not establish H200 memory capacity or quality.

GPU validation on 2026-11-11 used one 8×H200 allocation and the actual 33B model.
The 10-update pilot completed, followed by full-data continuation on 629800 windows
with a fixed budget of 1000 additional updates. Complete eight-rank Live110/260/510
checkpoints and both intermediate 8-case, 158-frame validations have been produced;
final Live1010 training and validation are still pending. Each intermediate set
contains 32 MP4s, including GT comparisons, with frame/geometry and complete decode
checks. Static frame review shows mixed improvements over Live10 and persistent
late camera/geometry drift; it does not establish convergence. The four-GPU
accumulation2 path has CPU coverage but has not completed a 33B H200 update.

## Camera initialization and data

Use the H3 environment from [README.md](README.md), plus the `ffmpeg` and
`ffprobe` system binaries. The merge command below accepts the original
SolarWM `ema.pt` release schema with 312 rank-384 pairs. This external published
camera checkpoint is the selected initialization; it does not inherit the
workspace's gesture-trained checkpoints. The merge computes in FP32 and
retains each original base-weight dtype.

```bash
export H3_BASE=/path/to/MiniMax-H3
export SOLARWM_CAMERA_LORA=/path/to/official/SolarWM-camera-release
export H3_CAMERA_MERGED=/path/to/new/camera_merged
python -m pipelines.h3.merge_solarwm_camera \
  --base "$H3_BASE" --adapter "$SOLARWM_CAMERA_LORA" --output "$H3_CAMERA_MERGED"
```

Supply the original SolarWM camera WDS index and release shards. Preparation
keeps cached Qwen/anchor/video tensors and first-relative W2C poses; it does not
encode raw RGB, interpolate frames, or infer camera poses from action labels.
These preparation commands expect 47 already latent-aligned camera rows. Native
H3 uses pixel offsets `[0,1,5,9,13]` in each 17-frame period, ending at pixel154
for a 158-frame clip. The published checkpoint manifest also describes raw
158-row camera artifacts; the native reader aligns those upstream. Already
aligned caches must not be uniformly resampled or truncated again. All14 held-out
camera caches were compared bitwise with that native reader, across7domains.

The published SolarWM profile includes encoded silence audio; this BiWM video
path omits audio rows as described in the Stage1 scope. Camera/data agreement
does not establish equivalence to the original complete audiovisual input.
The small functional pilot selects 16 training and 2 validation source clips per domain.
The full manifest excludes all windows and derived 160-frame siblings of the
14 validation source groups. This is a dataset-source-clip split, not a scene
or unseen-pretraining benchmark.

```bash
export SOLARWM_INDEX=/path/to/train-index.jsonl.gz
export SOLARWM_RELEASE_ROOT=/path/to/releases-v1
export H3_CACHE_DIR=/path/to/new/prepared_pilot
python -m pipelines.h3.prepare_stage2_camera \
  --index "$SOLARWM_INDEX" --release_root "$SOLARWM_RELEASE_ROOT" --output "$H3_CACHE_DIR"
python -m pipelines.h3.prepare_stage2_manifest \
  --index "$SOLARWM_INDEX" --pilot-provenance "$H3_CACHE_DIR/PROVENANCE.json" \
  --output /path/to/new/full_manifest
export H3_TRAIN_MANIFEST=/path/to/new/full_manifest/train.jsonl
```

## Pilot, continuation and validation

Both example configurations use LoRA rank=alpha=384, all 312 QKVO/FFN targets,
FP32 master AdamW at 1e-5, shift3, and effective batch8. The trainer explicitly
requires H200/FA3. Eight GPUs use accumulation1; four use accumulation2. Exact
checkpoint resume and pilot-to-full continuation require the saved world size.
The ordered global sample batches agree across topologies, but floating-point
summation and model RNG differ.

The entrypoint checks fixed recipe fields before initializing CUDA: shift3,
7/5-latent chunks, GT history, SP1, one-video microbatches, effective batch8,
FA3, a trainable compressor, constant-LR FP32-master AdamW and CFG1 validation.
Changing these descriptive fields does not select another implementation;
unsupported values are rejected. Learning rate, Adam hyperparameters, clip,
LoRA rank/alpha, update budget and validation schedule points are read at runtime.

```bash
export OUTPUT_DIR=/path/to/new/h3_pilot
H3_CONFIG=h3/configs/stage2_pilot.json NPROC_PER_NODE=8 bash scripts/h3/stage2.sh
export H3_PILOT_RUN="$OUTPUT_DIR"
export OUTPUT_DIR=/path/to/new/h3_full
H3_CONFIG=h3/configs/stage2_full.json NPROC_PER_NODE=8 bash scripts/h3/stage2.sh
```

The launcher expands the explicit path variables and hashes the actual training
manifest. The full example restores Live10, FP32 Adam and RNG, then performs
1000 additional updates through global1010 on the new full-data sample order.
It retains the source model/conditioning/loss/optimizer contract. Every complete
checkpoint includes all FSDP rank files and a final `COMPLETE.json` marker.

The pilot delivers 8 cases at 39 frames. The full configuration uses 8 fixed cases
across 7 domains, saves checkpoints at 110/260/510/1010, and delivers 158-frame
comparisons at 260/510/final1010. Each case contains raw generated MP4s,
`GT | result`, and `GT | baseline | result`. GT is explicitly marked as a native
VAE reconstruction when original RGB is unavailable. The VAE uses instance-only
FP16 SDPA with FP32 weights; the DiT remains on FA3. Native MP4 frame counts and
full decoding must pass before a case is complete. Training loss and file
integrity do not replace a visual quality review.

H3 Stage 3 DMD will be published after actual GPU training and validation.
