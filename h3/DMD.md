# H3 Stage 3: bidirectional autoregressive DMD

This is the camera-only H3 adaptation of the BiWM Wan2.2-TI2V-5B self-forcing
DMD route. On 2026-11-12 the latest native eight-H200 run restored G20/C105 and
completed **30 new G / 150 new C updates to G50/C255**, without extra warmup.
All 1,440 new rank receipts passed exact manifest-order, cursor, synchronized
exit/sigma/loss and finite-gradient audits. G21/C110 and G50/C255 each contain
all eight complete rank payloads (561,936,242,960 bytes). G21/G35/G50 each deliver
eight native 158-frame cases and 32 MP4s; all 96 passed independent
frame/geometry/full-decode/SHA checks. G35 retains latents, without a full
checkpoint. A separate native eight-GPU G50 restore reproduced both models,
both FP32 Adam states, all RNG and cursors bitwise, adding zero updates.
The final G50 global loss was 0.0663971072 (slowest rank 147.70 seconds);
C255 was 0.0309326160 (83.54 seconds). Scalar losses do not establish quality.

Native GT-VAE-reconstruction / CFG1-teacher / G20 / G50 four-panel MP4s were
created for all eight cases. Static review covered six overview frames, all
eight adjacent chunk-boundary pairs and the native final frame per case.
G50 preserves the late store display that G20 blurred/replaced and retains a
snow-scene character to frame157, whereas G20 and teacher lose it. However,
the retained figure has wrong identity, clothing and action. Bathroom vanity
omission, invented textures, camera and layout errors remain. Some cases have
no clear overall gain. No hard cut was established in the sampled boundary
pairs; this is not an exhaustive temporal/perceptual or quantitative camera
control evaluation. Teacher and AR use different RNG paths, so this is not a
paired-noise comparison. GT is VAE reconstruction, not original RGB.
**Quality acceptance and convergence remain unestablished.**
Source extraction and evidence digests are recorded in [DMD_SOURCE.json](DMD_SOURCE.json).

Historical G20 run (2026-11-11): snapshot `20c7849` completed the G3..20 updates,
checkpoints and validation tensors, then media delivery failed because the
shared renderer did not accept teacher/DMD label arguments. Commit `b5c86a3`
repairs that API and checks the real FM, teacher and DMD signatures. A separate
native VAE-only process decoded G3/G10/G20, adding zero G/C updates; original
failed and recovered receipts remain distinct. **The G50 run used snapshot
`bcb583c` with this fix, exited successfully and needed no recovery.** Native
VAE decoding uses FP32 weights, FP16 autocast and explicit instance-local SDPA;
DiT training retains `_flash_3`.

The generator keeps the selected Stage 2 LoRA384 and FramePack compressor on the
original SolarWM camera-merged base. Export folds that Stage 2 adapter into two
independent native score models: a frozen teacher and a full trainable critic.
The generator retains its original unfused adapter tensors and compressor.
The export computes the merge in FP32, retaining the native weight dtypes.
Phase 3 starts fresh G/C optimizers; this transfer is distinct from resuming a
complete DMD checkpoint.

All roles cover native 158 RGB frames / 47 latents at 1344×768. Student rollout
uses 9 chunks, first 7 and then 5 latents. History contains detached generated
latents, compressed and packed afresh for each visible context. Qwen uses the
first target image and caption; the independent VAE anchor and continuous W2C
camera ProPE keep their native H3 clocks. There is no HAND, audio-token path,
auxiliary loss, EMA or KV cache. The native teacher/critic score the full video
without FramePack. Target GT values are used for validation comparisons, not
as student rollout history or score conditioning.

The four student starts `[1,.75,.5,.25]`, shifted by 3, produce sigma points
`[1,.9,.75,.5,0]`. Training synchronizes a randomized gradient exit across ranks.
The score sigma follows `U → shift3 → clamp[.02,.98]`, with the configured exit
floor. H3 predicts `v=x0-noise` at clean-time `t=1-sigma`, so
`x0_hat=noisy+sigma*v`. Real and fake scores share the noisy sample and anchor.
The DMD field is `(fake_x0-real_x0)/(mean(abs(generated-real_x0))+1e-8)`;
its detached-target MSE surrogate supplies the generator gradient. The critic
fits the same generated distribution with flow matching. The example performs
five warmup C Adams and five fresh actual C Adams before each G Adam.

The default native teacher is conditional CFG1 with 30 schedule points/shift3.
On the selected local source, CFG5 introduced severe sampled color/geometry
artifacts; CFG1 passed a limited eight-case source review. Residual viewpoint,
layout and character errors remain. This is a source-specific choice, not a
universal H3 guidance rule or a convergence result.

The DiT requires Hopper `_flash_3` and the native backward gate. Blocks use BF16
with native FP32 projection/time paths, FP32 gradient reduction/master/moments,
and FP64 gradient norms. G/C use LR `1e-5/2e-6`, betas `(0,.999)`, eps `1e-8`,
weight decay 0 and clip 5. Checkpoint recomputation inputs are stored on CPU.
During G, idle critic FP32 Adam slots also reside on CPU and return to their
parameter devices before the next C update. Live critic weights stay on GPU.
This changes storage location while preserving the forward/backward order,
state values, RNG, Adam formula and update count.

Keeping nine generator graphs exceeded native H200 memory even after parking
the critic Adam. The current candidate records immutable CPU inputs at each
gradient exit during the same no-grad rollout, computes the original full-output
DMD cotangent once, then replays one chunk at a time in reverse creation order.
FSDP `no_sync` covers all but the final chunk, retaining one gradient reduction
per microbatch and one clip/Adam per G. Replay must reproduce each prediction
bitwise, and per-chunk RNG is forked/restored. Four-rank tiny CPU runs match the
original two-update graph's parameters, both FP32 Adams, RNG and data cursors
bitwise. A full native eight-H200 zero-update backward probe passed with exact
replay predictions and peak allocated memory 98.78 GiB per GPU. Its lazy G Adam
was not stepped or allocated; up to 7.66 GiB of estimated Adam storage and another
10 GiB of GPU margin were included in the capacity check. This establishes
backward capacity with a nonzero test cotangent. The subsequent actual DMD run
completed G1 and G2 with peak allocated memory 98.82 and 106.49 GiB, respectively;
the critic reached 109.36 GiB. It retained the full native model/span/resolution
and transferred idle critic FP32 Adam slots between CPU/GPU without dropping state.
The extracted public engine matches the frozen experimental engine bitwise in
four-rank tiny CPU tests. Native GPU restore also ran this public implementation;
the original 2G/15C run used the frozen experimental entrypoint. The public
entrypoint subsequently continued the complete G2/C15 state through another
native eight-H200 1G/5C cycle to G3/C20, without optimizer/RNG/data resets or
additional critic warmup. G3 used 183.24 seconds with 106.47 GiB peak allocated
memory. That continuation then reached G20/C105, with 864 all-rank receipts
covering exactly G3..20 and C16..105. Data order, cursors, synchronized exits,
score sigmas and finite losses/gradients were audited. Neither these integrity
gates nor a scalar loss establish converged video quality.

## Select and export Stage 2 Live

Finish the bounded [Stage 2 run](FRAMEPACK.md), including its final eight native
validation cases, before selecting a single Live source. An intermediate source
is permitted only with its own matched 158-frame validation. Review all eight
cases and create an explicit selection JSON containing:

```json
{
  "checkpoint": "/path/to/phase2/checkpoint_001010",
  "model_step": 1010,
  "source_weight": "Live",
  "source_config_sha256": "SHA256 of phase2/executed_config.json",
  "accepted_for_DMD": true,
  "reviewed_cases": [0,2,4,6,8,10,12,13]
}
```

The acceptance field records the actual visual review; a complete marker or low
FM loss alone does not establish it. Use the case indices in your source config.
Hash metadata with `pipelines.h3.configure_stage3.digest(path)` or `sha256sum`.
Export with the saved FSDP world size:

```bash
export H3_PHASE2_CHECKPOINT=/path/to/phase2/checkpoint_001010
export H3_PHASE2_SELECTION=/path/to/reviewed_selection.json
export H3_SCORE_BUNDLE=/path/to/new/phase3_initialization
torchrun --standalone --nproc_per_node=8 -m pipelines.h3.prepare_stage3 \
  --checkpoint "$H3_PHASE2_CHECKPOINT" --selection "$H3_PHASE2_SELECTION" \
  --output "$H3_SCORE_BUNDLE"
```

The bundle contains the native merged score weights/config/index, the original
unfused generator adapter and FP32 compressor, plus a file/SHA manifest. It does
not copy the original camera-merged generator base or package Qwen/VAE weights.
Those paths remain pinned in the selected Stage 2 configuration.

## Native teacher preflight

Keep the same Stage 2 SolarWM train/validation manifests and cached release root.
The full training split retains its validation-source exclusions.

```bash
export H3_TRAIN_MANIFEST=/path/to/full_manifest/train.jsonl
export H3_VALIDATION_MANIFEST=/path/to/prepared/validation.jsonl
export SOLARWM_RELEASE_ROOT=/path/to/releases-v1
export H3_TEACHER_RUN=/path/to/new/native_teacher
export H3_TEACHER_CONFIG=/path/to/teacher_config.json
python -m pipelines.h3.configure_stage3 \
  --input h3/configs/stage3_teacher.json --output "$H3_TEACHER_CONFIG"
torchrun --standalone --nproc_per_node=8 -m pipelines.h3.evaluate_stage3_teacher \
  --config "$H3_TEACHER_CONFIG" --output "$H3_TEACHER_RUN"
python -m pipelines.h3.verify_stage3_media \
  --input "$H3_TEACHER_RUN" --config "$H3_TEACHER_CONFIG" --step 1010 \
  --output "$H3_TEACHER_RUN/MEDIA_VERIFICATION.json"
```

Use the selected source step in the verification command. Teacher preflight
performs zero G/C updates. Each case delivers raw teacher and Phase 2 AR baseline
videos, `GT | teacher`, and `GT | baseline | teacher`. GT is explicitly a native
VAE reconstruction when original RGB is unavailable. The independent verifier
checks all 32 MP4s for actual frame count, geometry, full decode and SHA.

Review the generated videos and create a separate teacher-review JSON with the
following fields and actual values:

```json
{
  "accepted_for_finite_DMD_pilot": true,
  "source_phase2_step": 1010,
  "reviewed_cases": [0,2,4,6,8,10,12,13],
  "teacher_CFG": 1,
  "source_teacher_complete_sha256": "SHA256 of native_teacher/COMPLETE.json",
  "source_media_verification_sha256": "SHA256 of MEDIA_VERIFICATION.json",
  "approved_max_generator_step": 2,
  "approved_max_critic_step": 15
}
```

## Bounded DMD pilot and exact resume

```bash
export H3_TEACHER_REVIEW=/path/to/reviewed_teacher.json
export H3_TEACHER_MEDIA="$H3_TEACHER_RUN/MEDIA_VERIFICATION.json"
export OUTPUT_DIR=/path/to/new/h3_dmd_pilot
H3_CONFIG=h3/configs/stage3_pilot.json NPROC_PER_NODE=8 bash scripts/h3/stage3.sh
```

The launcher expands paths and pins source/manifest/review hashes before CUDA.
The example budget is 2G/15C, including five C warmup updates. Eight H200s use
DP8/SP1/accumulation1/effective batch8. The four-GPU full-critic capacity has
not been established; the example enables eight GPUs only.

Completed updates require matching receipts from every actual rank in
`step_receipts/`. Complete G1/C10 and G2/C15 checkpoints include both live
models, two FP32 Adams, CPU/CUDA/Python/NumPy and named RNG states, and independent
G/C data cursors. The teacher is immutable and pinned by source hashes. Partial
rank files, G0 baseline generation, model loading and teacher preflight do not
count as optimizer updates. Exact resume uses the same output directory and
unchanged configuration; unsaved later receipts require an explicit recovery
run rather than an implicit rewind.

G0 uses the same fixed cases/seeds and four-map sampler. First/final G latents
are saved at their actual model steps. After freeing all three DiTs and Adams,
the native VAE delivers eight-case 158-frame MP4 comparisons for both steps.
The VAE uses its own FP16/SDPA instance with FP32 weights; the DiT remains FA3.
Verify G1 under `validation_step000001/` and G2 at the run root with
`pipelines.h3.verify_stage3_media`, using the run's `executed_config.json` and
the corresponding `--step`. Full state/media integrity, generator memory,
camera/geometry/temporal quality and any longer training decision are separate
acceptance items. A two-update functional pilot is not convergence.

Reproduce the native temporal-span CPU protocol gate with four Gloo ranks:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
BIWM_DMD_DRAFT=h3/configs/stage3_pilot.json \
BIWM_DMD_CPU_OUT=/path/to/new/cpu_dmd_gate \
torchrun --standalone --nproc_per_node=4 tests/h3_dmd_smoke.py
python -m unittest discover -s tests -p test_h3_dmd_config.py
```

This uses tiny spatial2x2/rank4-LoRA CPU models with explicit native CPU
attention. It exercises the production engine through5warmupC/2G/15C,
all47latents/9chunks, the medium-history compressor, two FP32 Adams, named
RNG/data cursors and checkpoint-next-cycle bitwise resume. It does not measure
33B H200 memory, actual production-model updates, decoded video or convergence.

After a complete native checkpoint is available, verify restore without taking
another G/C optimizer step:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
torchrun --standalone --nproc_per_node=8 tests/h3_dmd_restore.py \
  --checkpoint /path/to/h3_dmd_pilot/checkpoint_000002 \
  --output /path/to/new/restore_gate
```

The restore gate compares native G/C shards, both FP32 Adam states, every RNG
stream and data cursors against the saved rank payloads. Device relocation is
allowed; shapes, shard identity, dtypes and values must match. It needs the same
large-model CPU RAM/GPU/storage capacity as a real restore. Its comparator and
entrypoint continuation have tiny CPU coverage. The native eight-GPU restore
gate passed for the G2/C15 pilot, G20/C105 and G50/C255 checkpoints; this does not execute a next GPU
optimizer step or establish next-cycle bitwise equivalence on production GPUs.
