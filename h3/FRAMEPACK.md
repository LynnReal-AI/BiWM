# H3 Stage 2: FramePack v2v

Local release preparation. Final training/checkpoint selection is pending;
this branch has not been published.

Stage 2 merges the original SolarWM camera LoRA, then trains a fresh H3 LoRA and
the existing BiWM FramePack compressor. The only objective is current-chunk
flow matching. Training uses clean GT history; validation rolls out entirely
from generated history. ProPE covers anchor, target and compressed history at
their physical camera-frame indices. There are no HAND conditions or auxiliary
losses, and no KV cache.

The native first chunk is7latents/22RGB frames; subsequent chunks are5latents/
17RGB frames. All9chunks cover47latents/158RGB frames at1344×768. Qwen encodes
the target first image and caption only; the independent VAE anchor keeps its
native `.999` time. H3 MM-RoPE and ProPE suffixes retain the Stage1 contract.

The modules are extracted from the frozen training source recorded in
[FRAMEPACK_SOURCE.json](FRAMEPACK_SOURCE.json). Four-rank CPU tests compare the
original/public loss, gradients, FP32 Adam, checkpoint-next-update and47latent
rollout bitwise. These tests do not establish H200 memory capacity or quality.

## Camera initialization and data

Use the H3 environment from [README.md](README.md), plus the `ffmpeg` and
`ffprobe` system binaries. The merge command below accepts the original
SolarWM `ema.pt` release schema with312rank384pairs. This external published
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
The small functional pilot selects16train and2validation source clips/domain.
The full manifest excludes all windows and derived160-frame siblings of the
14validation source groups. This is a dataset-source-clip split, not a scene
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

Both example configurations use LoRA rank=alpha384, all312QKVO/FFN targets,
FP32 master AdamW at1e-5, shift3, and effective batch8. The trainer explicitly
requires H200/FA3. Eight GPUs use accumulation1; four use accumulation2. Exact
checkpoint resume and pilot-to-full continuation require the saved world size.
The ordered global sample batches agree across topologies, but floating-point
summation and model RNG differ.

```bash
export OUTPUT_DIR=/path/to/new/h3_pilot
H3_CONFIG=h3/configs/stage2_pilot.json NPROC_PER_NODE=8 bash scripts/h3/stage2.sh
export H3_PILOT_RUN="$OUTPUT_DIR"
export OUTPUT_DIR=/path/to/new/h3_full
H3_CONFIG=h3/configs/stage2_full.json NPROC_PER_NODE=8 bash scripts/h3/stage2.sh
```

The launcher expands the explicit path variables and hashes the actual training
manifest. The full example restores Live10, FP32 Adam and RNG, then performs
1000additional updates throughglobal1010 on the new full-data sample order.
It retains the source model/conditioning/loss/optimizer contract. Every complete
checkpoint includes all FSDP rank files and a final `COMPLETE.json` marker.

The pilot delivers8cases at39frames. The full configuration uses8fixed cases
across7domains, saves checkpoints at110/260/510/1010, and delivers158frame
comparisons at260/510/final1010. Each case contains raw generated MP4s,
`GT | result`, and `GT | baseline | result`. GT is explicitly marked as a native
VAE reconstruction when original RGB is unavailable. The VAE uses instance-only
FP16 SDPA with FP32 weights; the DiT remains on FA3. Native MP4 frame counts and
full decoding must pass before a case is complete. Training loss and file
integrity do not replace a visual quality review.

H3 Stage3DMD is released separately after actual GPU training and validation.
