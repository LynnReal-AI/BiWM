# BiWM Stage 1 — camera-control fine-tuning (HunyuanVideo-1.5 training entry).
# wan-style single-file entry (arg / distributed / model build / data / training loop / checkpoint),
# directly driving biWM/hunyuan's ARHunyuanVideo_1_5_DiffusionTransformer native dual-stream forward_bi.
# Text encoder is an MLLM (Qwen2.5-VL), not T5. Camera is controlled via text (camtext) or PRoPE.
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from hunyuan.modules.model import (ARHunyuanVideo_1_5_DiffusionTransformer,
                                   MMDoubleStreamBlock)
from hunyuan.configs.hunyuan_t2v import HUNYUAN_T2V_CONFIG
from pipelines.dataset.hy15_camera_dataset import make_camera_plucker_loader

# HunyuanVideo-1.5 VAE geometry: spatial 16x / temporal 4x / latent 32ch / scaling_factor 1.03682
HY15_LATENT_CHANNEL_COUNT = 32
HY15_VAE_SPACE_STRIDE = 16
HY15_VAE_TIME_STRIDE = 4
HY15_VAE_SCALE_COEFF = 1.03682

# 81 discrete camera actions → camera text (action_label = trans*9 + rot). One constant action per clip,
# concatenated into the caption so the text stream controls camera motion (replacing PRoPE).
_TRANS_PHRASES =["Camera does not move.", "Camera moves forward.", "Camera moves backward.",
               "Camera moves right.", "Camera moves left.", "Camera moves forward-right.",
               "Camera moves forward-left.", "Camera moves backward-right.", "Camera moves backward-left."]
_ROT_PHRASES = ["Camera does not rotate.", "Camera pitches up.", "Camera pitches down.",
             "Camera yaws right.", "Camera yaws left.", "Camera pitches up and yaws right.",
             "Camera pitches up and yaws left.", "Camera pitches down and yaws right.",
             "Camera pitches down and yaws left."]


def action_into_camtext(a: int) -> str:
    a = int(a)
    return f"{_TRANS_PHRASES[a // 9]} {_ROT_PHRASES[a % 9]}"


# Customized system prompt: camera description inside <camera>...</camera>, static scene elsewhere (no motion/camera).
# Decouples the camera control signal from static content in the text. crop_start=-1 crops out system-prompt tokens.
CAMTEXT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for a game-world video generator. The prompt has a "
    "<camera>...</camera> tag that describes ONLY the camera motion (translation and rotation, "
    "e.g. moving/panning/tilting), followed by text that describes ONLY the static scene "
    "(characters, objects, environment, art style, lighting) with NO motion. Read the camera "
    "movement from the <camera> tag and the static appearance from the rest."
)
CAMTEXT_PROMPT_FORMAT = {
    "template": [
        {"role": "system", "content": CAMTEXT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": "{}"},
    ],
    "crop_start": -1,
}


def assemble_camtext_prompt(cam_text: str, caption: str) -> str:
    """Assemble <camera>camera</camera> + static caption (camera first/wrapped in tag; caption is static-only, no motion/camera)."""
    cap = (caption or "").rstrip()
    return f"<camera>{cam_text}</camera> {cap}".strip()


def on_main_rank():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rank0_print(*a, **k):
    if on_main_rank():
        print(*a, **k, flush=True)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world


# --- Model build + weight loading ---
def _allowed_init_keys():
    import inspect
    return {p for p in inspect.signature(
        ARHunyuanVideo_1_5_DiffusionTransformer.__init__).parameters if p != "self"}


def _read_transformer_config(weights_dir):
    """Build from config.json in full (filtered to __init__-accepted keys), else placeholder config (random init).
    Must be full so the real ckpt's in_channels/patch_size/text_states_dim/mm_double_blocks_depth/etc. all match."""
    cj = os.path.join(weights_dir, "config.json") if weights_dir and os.path.isdir(weights_dir) else None
    if cj and os.path.exists(cj):
        accepted = _allowed_init_keys()
        file_cfg = json.load(open(cj))
        cfg = {k: v for k, v in file_cfg.items() if not k.startswith("_") and k in accepted}
        rank0_print(f"[HY15] full build from config.json: hidden={cfg.get('hidden_size')}, "
               f"double={cfg.get('mm_double_blocks_depth')}, in_ch={cfg.get('in_channels')}, "
               f"patch={cfg.get('patch_size')}, text_dim={cfg.get('text_states_dim')}, "
               f"vision_dim={cfg.get('vision_states_dim')}, concat={cfg.get('concat_condition')}")
        return cfg
    rank0_print("[HY15] no config.json, using placeholder HUNYUAN_T2V_CONFIG (random-init smoke test)")
    return dict(HUNYUAN_T2V_CONFIG)


def _read_sd_shape_tolerant(model, sd):
    """Load only weights with matching name and shape; skip (report) mismatched shapes or extras
    (strict=False does not skip shape mismatches, so filter ourselves)."""
    msd = model.state_dict()
    keep, mism = {}, []
    for k, v in sd.items():
        if k in msd and tuple(msd[k].shape) == tuple(v.shape):
            keep[k] = v
        elif k in msd:
            mism.append((k, tuple(v.shape), tuple(msd[k].shape)))
    missing, unexpected = model.load_state_dict(keep, strict=False)
    return missing, unexpected, mism


def _read_safetensors_dir(model_dir):
    """Load diffusers-format transformer weights (single / sharded safetensors)."""
    import glob
    import safetensors.torch as st
    sd = {}
    idx = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(idx):
        weight_map = json.load(open(idx)).get("weight_map", {})
        for shard in sorted(set(weight_map.values())):
            sd.update(st.load_file(os.path.join(model_dir, shard), device="cpu"))
        return sd
    single = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    if os.path.exists(single):
        return st.load_file(single, device="cpu")
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    for s in shards:
        sd.update(st.load_file(s, device="cpu"))
    if not sd:
        raise FileNotFoundError(f"no safetensors under {model_dir}")
    return sd


def make_hy15_transformer(args, device, dtype=torch.bfloat16):
    tdir = args.pretrained_model_path
    cfg = _read_transformer_config(tdir)
    rank0_print(f"[HY15] building ARHunyuanVideo_1_5_DiffusionTransformer (hidden={cfg['hidden_size']}, "
           f"double_blocks={cfg['mm_double_blocks_depth']})")
    model = ARHunyuanVideo_1_5_DiffusionTransformer(**cfg)
    model._biwm_config = dict(cfg)   # store build config; store_checkpoint writes config.json from it
    # whether img_in takes the concat condition (2*C+1): i2v ckpt usually True; placeholder t2v config False (only feeds C)
    args._concat_condition = bool(cfg.get("concat_condition", False))
    # camera_mode: camtext (default) = drop PRoPE, camera via text (full fine-tune); prope = add PRoPE (+ optional discrete action)
    if args.camera_mode == "prope":
        model.add_prope_parameters()
        if args.use_discrete_action:
            model.add_discrete_action_parameters()
        rank0_print("[HY15] camera_mode=prope (PRoPE viewmats/Ks + discrete action)")
    else:
        rank0_print("[HY15] camera_mode=camtext (drop PRoPE; camera via text cam-text concatenated into caption; full fine-tuning)")

    # Load real weights (dual-stream-only; single-stream keys fall into unexpected, strict=False)
    if tdir and os.path.isdir(tdir):
        try:
            sd = _read_safetensors_dir(tdir)
            missing, unexpected, mism = _read_sd_shape_tolerant(model, sd)
            rank0_print(f"[HY15] weight loading: missing={len(missing)} unexpected={len(unexpected)} "
                   f"shape-mismatch-skipped={len(mism)}")
            if mism:
                rank0_print(f"[HY15]  skipped shape mismatches (first 5): {mism[:5]}")
            del sd
        except FileNotFoundError:
            rank0_print(f"[HY15] transformer weights not found ({tdir}), random init.")
    else:
        rank0_print(f"[HY15] no transformer weight directory, random init.")

    model = model.to(device=device, dtype=dtype)
    if args.gradient_checkpointing:
        # set the flag directly (forward_bi applies torch.utils.checkpoint to double_blocks)
        model.gradient_checkpointing = True
        rank0_print("[HY15] gradient checkpointing ON (forward_bi per-block checkpoint)")
    return model


def enclose_fsdp(model, device, args):
    import functools
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        MixedPrecision, ShardingStrategy,
                                        BackwardPrefetch)
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={MMDoubleStreamBlock})
    rank0_print("[HY15] FSDP FULL_SHARD wrap on MMDoubleStreamBlock")
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.bfloat16,
                                       buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        sync_module_states=True,
    )


# --- Conditional latent (i2v concat + mask) ---
def _blend_by_mask(t0, t1, mask, dim=2):
    idx = torch.nonzero(mask).squeeze(1)
    out = t0.clone()
    sl = [slice(None)] * out.dim()
    sl[dim] = idx
    out[tuple(sl)] = t1[tuple(sl)]
    return out


def fetch_task_mask(training_mode, t_lat, device):
    m = torch.zeros(t_lat, device=device)
    if training_mode == "i2v":
        m[0] = 1.0
    return m


def stage_cond_latents(training_mode, image_cond, latents):
    B, C, T, H, W = latents.shape
    if image_cond is not None and training_mode == "i2v":
        latents_concat = image_cond.repeat(1, 1, T, 1, 1).clone()
        latents_concat[:, :, 1:, :, :] = 0.0
    else:
        latents_concat = torch.zeros_like(latents)
    mask = fetch_task_mask(training_mode, T, latents.device)
    zeros = torch.zeros(B, 1, T, H, W, device=latents.device, dtype=latents.dtype)
    ones = torch.ones(B, 1, T, H, W, device=latents.device, dtype=latents.dtype)
    mask_concat = _blend_by_mask(zeros, ones, mask, dim=2)
    return torch.cat([latents_concat, mask_concat], dim=1)  # (B, C+1, T, H, W)


# --- live real-time encoding (t2v): HunyuanVideo VAE + MLLM(Qwen2.5-VL); byt5/vision fed zeros ---
BYT5_MAX_TOKENS = 256
BYT5_WIDTH = 1472


def init_hy15_vae(vae_dir, device, dtype=torch.bfloat16):
    """Load HunyuanVideo VAE (AutoencoderKLConv3D, 16x spatial / 4x temporal / 32ch)."""
    from hunyuan.vae import AutoencoderKLConv3D
    cand = os.path.join(vae_dir, "vae") if os.path.isdir(os.path.join(vae_dir, "vae")) else vae_dir
    vae = AutoencoderKLConv3D.from_pretrained(cand, torch_dtype=dtype).to(device).eval().requires_grad_(False)
    vae._sf = float(getattr(vae.config, "scaling_factor", None) or HY15_VAE_SCALE_COEFF)
    rank0_print(f"[HY15-live] VAE ready {cand} (scaling_factor={vae._sf})")
    return vae


def init_mllm(llm_dir, device, dtype=torch.bfloat16, max_length=1000):
    """Load HY15's MLLM text encoder (Qwen2.5-VL, extract hidden state with the video prompt template)."""
    from hunyuan.text_encoder import TextEncoder
    # use the customized system-prompt template (CAMTEXT_PROMPT_FORMAT): camera in <camera>, caption static-only.
    te = TextEncoder(
        text_encoder_type="llm", tokenizer_type="llm",
        text_encoder_path=llm_dir, tokenizer_path=llm_dir,
        max_length=max_length, text_encoder_precision="bf16",
        prompt_template=CAMTEXT_PROMPT_FORMAT, prompt_template_video=CAMTEXT_PROMPT_FORMAT,
        # official takes the hidden state of the LLM's 3rd-from-last layer (skip=2); last-layer (None) → text features OOD → color noise
        hidden_state_skip_layer=2, apply_final_norm=False,
        device=device,
    )
    te = te.to(device).eval().requires_grad_(False)
    rank0_print(f"[HY15-live] MLLM(Qwen2.5-VL) ready {llm_dir}")
    return te


@torch.no_grad()
def mllm_embed(te, captions, device):
    """captions:List[str] -> (prompt_embeds [B,L,3584], prompt_mask [B,L]). video template."""
    be = te.text2tokens(captions, data_type="video", max_length=te.max_length)
    out = te.encode(be, data_type="video", device=device)
    return out.hidden_state, out.attention_mask


_NEG_EMBED_STORE = {}


@torch.no_grad()
def embed_neg_prompt(te, device):
    """CFG uncond/neg = empty string "" encoded with the original li-dit-encode-video-json template (= minWM HY1.5 neg_prompt),
    consistent with the base model's CFG calibration. Cached. Temporarily swaps the template for encoding then restores it."""
    if "neg" in _NEG_EMBED_STORE:
        pe, pm = _NEG_EMBED_STORE["neg"]
        return pe.to(device), pm.to(device)
    from hunyuan.text_encoder import PROMPT_TEMPLATE
    saved = te.prompt_template_video
    try:
        te.prompt_template_video = dict(PROMPT_TEMPLATE["li-dit-encode-video-json"])  # original template
        pe, pm = mllm_embed(te, [""], device)
    finally:
        te.prompt_template_video = saved                       # restore customized <camera> template
    _NEG_EMBED_STORE["neg"] = (pe.detach(), pm.detach())
    return pe, pm


# --- Validation video saving (torchvision write_video crf18 + GT|Gen side-by-side) ---
def _frames_into_uint8(vid):
    """[B,3,T,H,W]∈[-1,1] → list[np.uint8 HWC] (takes batch 0)."""
    f = ((vid[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu().numpy()
    f = (f * 255).astype(np.uint8)
    return [f[i] for i in range(f.shape[0])]


def store_video(video_frames, output_path, fps=16):
    """Same as Wan store_video: torchvision write_video (h264, crf18, veryfast)."""
    from torchvision.io import write_video
    video_tensor = torch.from_numpy(np.stack(video_frames))
    write_video(output_path, video_tensor, fps=int(fps), options={'crf': '18', 'preset': 'veryfast'})


def stitch_videos_with_labels(left_frames, right_frames, output_path,
                               left_label="Ground Truth", right_label="Generated",
                               fps=16, font_scale=0.7, label_height=40):
    """Same as Wan stitch_videos_with_labels: side-by-side + top text labels → torchvision write_video."""
    from torchvision.io import write_video
    if len(left_frames) != len(right_frames):
        m = min(len(left_frames), len(right_frames))
        left_frames, right_frames = left_frames[:m], right_frames[:m]
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
    combined_frames = []
    for lf, rf in zip(left_frames, right_frames):
        lh, lw = lf.shape[:2]; rh, rw = rf.shape[:2]
        th_ = min(lh, rh)
        if th_ % 2 != 0:
            th_ -= 1
        if has_cv2:
            if lh != th_:
                lw2 = int(lw * th_ / lh); lw2 -= lw2 % 2
                lf = cv2.resize(lf, (lw2, th_), interpolation=cv2.INTER_LINEAR)
            if rh != th_:
                rw2 = int(rw * th_ / rh); rw2 -= rw2 % 2
                rf = cv2.resize(rf, (rw2, th_), interpolation=cv2.INTER_LINEAR)
            lhn, lwn = lf.shape[:2]; rhn, rwn = rf.shape[:2]
            lL = np.zeros((lhn + label_height, lwn, 3), dtype=np.uint8); lL[label_height:] = lf
            rL = np.zeros((rhn + label_height, rwn, 3), dtype=np.uint8); rL[label_height:] = rf
            font = cv2.FONT_HERSHEY_SIMPLEX; thick = 2
            (tw, tht), _ = cv2.getTextSize(left_label, font, font_scale, thick)
            cv2.putText(lL, left_label, ((lwn - tw) // 2, (label_height + tht) // 2), font, font_scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
            (tw, tht), _ = cv2.getTextSize(right_label, font, font_scale, thick)
            cv2.putText(rL, right_label, ((rwn - tw) // 2, (label_height + tht) // 2), font, font_scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
            combined = np.concatenate([lL, rL], axis=1)
        else:
            combined = np.concatenate([lf, rf], axis=1)
        combined_frames.append(combined)
    write_video(output_path, torch.from_numpy(np.stack(combined_frames)),
                fps=int(fps), options={'crf': '18', 'preset': 'veryfast'})


@torch.no_grad()
def assemble_live_batch(item, vae, te, args, device, caption_only=False):
    """One tuple from BiwmCamCaptionData(video_real) → dict used by run_one_step.
    camtext mode (default): one camera action per clip → one camera-text sentence concatenated into the caption.
    prope mode: feeds viewmats/Ks/per-frame action. byt5/vision/image_cond fed zeros (t2v).
    caption_only=True (stage2 DMD training): text uses only the caption, no camera text."""
    pixel_values = item["pixel_values"]                     # [T,C,H,W] ∈[-1,1]
    caption = item["caption"]
    action_labels = item["action_labels"]                   # [T_lat]
    # VAE encode: [T,C,H,W] → [1,C,T,H,W]
    x = pixel_values.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype=next(vae.parameters()).dtype)
    latent = (vae.encode(x).latent_dist.sample() * vae._sf).to(torch.bfloat16)   # [1,32,T_lat,h,w]
    B, C, T_lat, h, w = latent.shape
    # this clip's camera action (constant): take the non-zero mode (first frame init=0)
    al = torch.as_tensor(action_labels).reshape(-1).long()
    _nz = al[al > 0]
    clip_action = int(torch.bincount(_nz).argmax()) if _nz.numel() > 0 else 0
    # text → MLLM; CFG dropout. camera goes into <camera>...</camera>, caption static-only.
    cap = caption.rstrip() if caption else ""
    if caption_only:                                         # stage2 DMD training: pure caption, no camera introduced (no <camera>)
        full_text = cap
    elif args.camera_mode == "camtext":
        full_text = assemble_camtext_prompt(action_into_camtext(clip_action), cap)
    else:
        full_text = cap
    if random.random() < args.training_cfg_rate:
        full_text = ""                                       # unconditional (CFG training)
    pe, pm = mllm_embed(te, [full_text], device)            # [1,L,3584], [1,L]
    byt5 = torch.zeros(1, BYT5_MAX_TOKENS, BYT5_WIDTH, device=device, dtype=torch.bfloat16)
    byt5_m = torch.zeros(1, BYT5_MAX_TOKENS, device=device, dtype=torch.bfloat16)
    vis = torch.zeros(1, 1, 1, device=device, dtype=torch.bfloat16)
    img_cond = torch.zeros(1, C, 1, h, w, device=device, dtype=torch.bfloat16)
    vm = ks = action_t = None   # camtext mode: camera carried by text, no continuous viewmats/Ks
    return {
        "latent": latent, "prompt_embed": pe, "prompt_mask": pm,
        "viewmats": vm, "Ks": ks, "action": action_t,
        "byt5_text_states": byt5, "byt5_text_mask": byt5_m, "vision_states": vis,
        "image_cond": img_cond, "i2v_mask": torch.ones_like(latent),
        "caption": caption, "clip_action": clip_action,   # for per-rank re-encoding of cam-text during validation
    }


# --- Single training step (flow matching, native forward_bi) ---
def run_one_step(model, batch, args, device, step):
    latents = batch["latent"].to(device, dtype=torch.bfloat16)         # (B,C,T,H,W)
    prompt_embed = batch["prompt_embed"].to(device, dtype=torch.bfloat16)
    prompt_mask = batch["prompt_mask"].to(device, dtype=torch.bfloat16)
    byt5_states = batch["byt5_text_states"].to(device, dtype=torch.bfloat16)
    byt5_mask = batch["byt5_text_mask"].to(device, dtype=torch.bfloat16)
    vision_states = batch["vision_states"].to(device, dtype=torch.bfloat16)
    image_cond = batch["image_cond"].to(device, dtype=torch.bfloat16)
    i2v_mask = batch["i2v_mask"].to(device, dtype=torch.bfloat16)
    # camera: None in camtext mode (via text); only present in prope mode
    _vm = batch.get("viewmats"); _ks = batch.get("Ks"); _act = batch.get("action")
    viewmats = _vm.to(device, dtype=torch.float32) if _vm is not None else None
    Ks = _ks.to(device, dtype=torch.float32) if _ks is not None else None
    action = _act.to(device).long() if _act is not None else None

    B, C, T_lat, H, W = latents.shape
    task = args.training_mode

    # flow-matching sigma sampling aligned with Wan stage1: u~U[0,1] then shift (no logit-normal)
    noise = torch.randn_like(latents)
    u = torch.rand(B, device=device)
    sh = args.sigma_shift
    sigma = sh * u / (1.0 + (sh - 1.0) * u)                 # (B,) in [0,1]
    sig_b = sigma.view(B, 1, 1, 1, 1).to(latents.dtype)
    noisy = (1.0 - sig_b) * latents + sig_b * noise
    timestep = (sigma * 1000.0).repeat_interleave(T_lat).to(torch.bfloat16)   # (B*T_lat,)
    # text-stream timestep constantly 0 (text not noised, aligned with minWM bidir)
    timestep_txt = torch.zeros(B, device=device, dtype=torch.bfloat16)        # (B,)
    target = noise - latents                                # flow-matching velocity target

    # concat_condition model takes [noisy, cond(=latents_concat+mask)] (2C+1); otherwise only noisy (C)
    if getattr(args, "_concat_condition", False):
        cond = stage_cond_latents(task, image_cond if task == "i2v" else None, noisy)
        hidden_states = torch.cat([noisy, cond], dim=1)     # (B, 2C+1, T, H, W)
    else:
        hidden_states = noisy                               # (B, C, T, H, W)

    extra_kwargs = {"byt5_text_states": byt5_states, "byt5_text_mask": byt5_mask}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        pred, _ = model(
            bi_inference=True,
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_txt=timestep_txt,
            text_states=prompt_embed,
            text_states_2=None,
            encoder_attention_mask=prompt_mask,
            vision_states=vision_states if task == "i2v" else None,
            viewmats=viewmats,
            Ks=Ks,
            action=action,
            extra_kwargs=extra_kwargs,
            mask_type=task,
            return_dict=False,
        )
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
        diff = (pred.float() * i2v_mask - target.float() * i2v_mask) ** 2
        loss = diff.sum() / torch.clamp(i2v_mask.sum(), min=1.0)
    return loss


# --- checkpoint ---
def store_checkpoint(model, out_dir, step):
    """Save checkpoint (format aligned with Wan): FSDP FULL_STATE_DICT →
      checkpoint-{step}/diffusion_pytorch_model.safetensors + config.json.
    Isomorphic to the base weight directory, usable directly as --pretrained_model_path for resume/inference."""
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        FullStateDictConfig, StateDictType)
    import safetensors.torch as st
    fcfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fcfg):
        sd = model.state_dict()
    if on_main_rank():
        d = os.path.join(out_dir, f"checkpoint-{step}")
        os.makedirs(d, exist_ok=True)
        # weights: safetensors; make contiguous to avoid non-contiguous storage errors
        st.save_file({k: v.contiguous() for k, v in sd.items()},
                     os.path.join(d, "diffusion_pytorch_model.safetensors"))
        # config.json: build-time cfg; under FSDP/checkpoint wrap, unwrap layer by layer to get the attribute
        m = model
        for attr in ("_fsdp_wrapped_module", "_checkpoint_wrapped_module", "module"):
            m = getattr(m, attr, m)
        bcfg = getattr(model, "_biwm_config", None) or getattr(m, "_biwm_config", None)
        if bcfg is not None:
            try:
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump({k: v for k, v in bcfg.items()
                               if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
                              f, indent=4)
            except Exception as e:
                rank0_print(f"[HY15] config.json save failed (skipped): {type(e).__name__}: {e}")
        rank0_print(f"[HY15] saved checkpoint (safetensors+config) -> {d}")


# --- Validation: sample-denoise from noise → VAE decode → save mp4 (t2v) ---
# Representative camera actions for validation (each rank samples a different camera move):
#   9=forward 18=backward 27=right 36=left 1=pitch_up 2=pitch_down 3=yaw_right 4=yaw_left
_EVAL_ACTIONS = [9, 18, 27, 36, 1, 2, 3, 4]


@torch.no_grad()
def execute_validation(model, vae, mllm, val_batch, args, step, device, rank):
    """Each rank samples a different camera action (camtext: re-encode caption+cam-text), denoise from noise → VAE decode
    → overlay joystick (control) → each rank saves its own mp4. Sampler aligned with minWM."""
    from hunyuan.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
    model.eval()
    task = args.training_mode
    latent = val_batch["latent"].to(device, dtype=torch.bfloat16)
    B, C, T_lat, H, W = latent.shape

    # under CP the same group uses the same action + noise seed and saves only once (rank0 within group);
    #   val_id is the CP group id (without CP, the global rank).
    from pipelines.utils.parallel_states import nccl_state, fetch_sequence_parallel_state
    if fetch_sequence_parallel_state():
        val_id = nccl_state.group_id            # each CP group samples a different action
        save_this = (nccl_state.rank_within_group == 0)
    else:
        val_id = rank                          # pure DP: one action per rank (original behavior)
        save_this = True

    # communication self-check: val_same_input=True → all ranks use the same action + caption + seed
    #   (if outputs still differ it is a communication/FSDP problem).
    _same = getattr(args, "val_same_input", False)
    _vid = 0 if _same else val_id
    # validation action uses the GT clip's own real action clip_action (so GT|Gen|joystick are the same action),
    #   aligned with training assemble_live_batch. val_same_input still uses the fixed action 9.
    if _same:
        act = _EVAL_ACTIONS[0]
    else:
        act = int(val_batch.get("clip_action", _EVAL_ACTIONS[val_id % len(_EVAL_ACTIONS)]))
    if args.camera_mode == "camtext" and mllm is not None:
        _cap = ("A medieval castle on a green hill under a clear blue sky" if _same
                else str(val_batch.get("caption", "")))
        full_text = assemble_camtext_prompt(action_into_camtext(act), _cap)  # <camera>cam</camera> + static caption
        pe, pm = mllm_embed(mllm, [full_text], device)
        viewmats = Ks = action = None
    else:                                    # prope: use cached condition + cached camera
        pe = val_batch["prompt_embed"].to(device, torch.bfloat16)
        pm = val_batch["prompt_mask"].to(device, torch.bfloat16)
        viewmats = val_batch["viewmats"].to(device, torch.float32) if val_batch.get("viewmats") is not None else None
        Ks = val_batch["Ks"].to(device, torch.float32) if val_batch.get("Ks") is not None else None
        action = val_batch["action"].to(device).long() if val_batch.get("action") is not None else None
    extra = {"byt5_text_states": val_batch["byt5_text_states"].to(device, torch.bfloat16),
             "byt5_text_mask": val_batch["byt5_text_mask"].to(device, torch.bfloat16)}
    img_cond = val_batch["image_cond"].to(device, torch.bfloat16)

    # CFG: uncond/neg = empty string encoded with the original li-dit template (= minWM neg_prompt). guidance<=1 disables CFG.
    guidance = float(getattr(args, "cfg_scale", 1.0))
    pe_u = pm_u = None
    if guidance > 1.0 and mllm is not None:
        from pipelines.common.dmd_algo import run_cfg
        pe_u, pm_u = embed_neg_prompt(mllm, device)

    g = torch.Generator(device=device).manual_seed(42 + _vid)  # same noise for the same CP group
    x = torch.randn(latent.shape, generator=g, device=device, dtype=torch.bfloat16)
    sched = FlowMatchDiscreteScheduler(shift=args.validation_shift, reverse=True, solver="euler")
    sched.set_timesteps(args.diffusion_sampling_steps, device=device)

    def _pred(_pe, _pm, _hidden, _ts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(bi_inference=True, hidden_states=_hidden, timestep=_ts.expand(B * T_lat),
                           timestep_txt=torch.zeros(B, device=device, dtype=torch.bfloat16),  # text-stream timestep constantly 0
                           text_states=_pe, text_states_2=None,
                           encoder_attention_mask=_pm, vision_states=None,
                           viewmats=viewmats, Ks=Ks, action=action,
                           extra_kwargs=extra, mask_type=task, return_dict=False)
        return out

    for t in sched.timesteps:
        ts = t.to(device, torch.bfloat16)
        if getattr(args, "_concat_condition", False):
            condlat = stage_cond_latents(task, img_cond if task == "i2v" else None, x)
            hidden = torch.cat([x, condlat], dim=1)
        else:
            hidden = x
        pred = _pred(pe, pm, hidden, ts)                        # conditional
        if pe_u is not None:                                    # CFG: uncond + scale*(cond-uncond)
            pred = run_cfg(pred, _pred(pe_u, pm_u, hidden, ts), guidance)
        x = sched.step(pred, t, x).prev_sample
    # decode generated video (no overlay) + GT (VAE reconstruction of val_batch latent) for side-by-side comparison
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vid = vae.decode((x / vae._sf).to(next(vae.parameters()).dtype)).sample   # [B,3,T,H,W]
    frames = _frames_into_uint8(vid)                            # list[np.uint8 HWC]
    gt_frames = None
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gt_vid = vae.decode((latent / vae._sf).to(next(vae.parameters()).dtype)).sample
        gt_frames = _frames_into_uint8(gt_vid)
    except Exception as e:
        rank0_print(f"[HY15][val] GT decode skipped: {type(e).__name__}: {e}")

    if not save_this:    # CP: each group saves only once (rank0 within group); pure DP: every rank saves
        model.train(); return

    d = os.path.join(args.output_dir, "validation"); os.makedirs(d, exist_ok=True)
    fps = int(getattr(args, "fps", 24))
    cam_txt = action_into_camtext(act)
    name_prefix = f"step_{step:07d}_{task}_camtext_video_real_rank_{rank:03d}_act{act}"
    # 1) generated video (no overlay)
    store_video(frames, os.path.join(d, f"validation_{name_prefix}.mp4"), fps=fps)
    # 2) validation metadata txt
    try:
        with open(os.path.join(d, f"validation_{name_prefix}_prompt.txt"), "w", encoding="utf-8") as mf:
            mf.write(f"Step: {step}\nRank: {rank}\nVal id(CP group/rank): {val_id}\n")
            mf.write(f"Action label: {act}\nCamera text: {cam_txt}\n")
            mf.write(f"Prompt: {full_text if (args.camera_mode=='camtext' and mllm is not None) else '<preencoded>'}\n")
            mf.write(f"CFG Scale: {guidance}\nSampling Steps: {args.diffusion_sampling_steps}\n")
            mf.write(f"Num Frames(latent): {T_lat}\nResolution: {W*HY15_VAE_SPACE_STRIDE}x{H*HY15_VAE_SPACE_STRIDE}\n")
            mf.write(f"FPS: {fps}\nSource: video_real\nTask Type: {task}\nCamera: camtext(<camera>tag)\n")
    except Exception as e:
        rank0_print(f"[HY15][val] metadata write skipped: {type(e).__name__}: {e}")
    # 3) GT|Gen side-by-side + joystick overlay
    try:
        from pipelines.common.control_overlay import superimpose_control_video as add_control_overlay
        al = torch.full((T_lat,), int(act), dtype=torch.long)
        gen_j = add_control_overlay(frames, al, vae_temporal_stride=HY15_VAE_TIME_STRIDE)
        if gt_frames is not None:
            gt_j = add_control_overlay(gt_frames, al, vae_temporal_stride=HY15_VAE_TIME_STRIDE)
            stitch_videos_with_labels(gt_j, gen_j, os.path.join(d, f"combined_control_{name_prefix}.mp4"),
                                       "GT", "Generated", fps=fps)
        else:                                                # without GT, save only the generated video with joystick
            store_video(gen_j, os.path.join(d, f"control_{name_prefix}.mp4"), fps=fps)
    except Exception as e:
        rank0_print(f"[HY15][val] control/combined skipped: {type(e).__name__}: {e}")
    if on_main_rank():
        rank0_print(f"[HY15][val] step {step}: one action per (CP group/rank); saved validation_*/combined_control_* (act{act}) -> {d}")
    model.train()


# --- main ---
def _live_batch_merge(b):
    """live DataLoader collate: batch_size=1, directly return that tuple (module-level, picklable to workers)."""
    return b[0]


def main(args):
    local_rank, rank, world = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    # sequence parallelism (CP, Ulysses). cp_size==1: pure DP. cp_size>1: chunk the sequence (dim=1)
    #   + per-block all-to-all full-sequence attention + final all_gather.
    from pipelines.utils.parallel_states import setup_sequence_parallel_state
    setup_sequence_parallel_state(args.cp_size)
    if args.cp_size > 1:
        rank0_print(f"[HY15] CP (sequence parallelism) on: cp_size={args.cp_size}, "
               f"DP groups={world // args.cp_size}; same sample within a group, different samples across groups")

    model = make_hy15_transformer(args, device)
    model.train()
    model = enclose_fsdp(model, device, args)

    vae = mllm = None
    if args.data_mode == "preencoded":
        dataset, loader = make_camera_plucker_loader(
            json_path=args.json_path, causal=False, window_frames=args.window_frames,
            batch_size=args.train_batch_size, num_data_workers=args.dataloader_num_workers,
            drop_last=True, drop_first_row=False, seed=args.seed,
            cfg_rate=args.training_cfg_rate, i2v_rate=args.i2v_rate, training_mode=args.training_mode,
        )
    elif args.data_mode == "preenc":
        # biWM preencoding (.pt from preencode_hy15.py): training reads latent+prompt_embed directly (skips per-step
        #   online VAE+MLLM encoding). Validation still needs VAE (decode) + MLLM (re-encode cam-text), so still loaded.
        from pipelines.dataset.biwm_camera_text_dataset import BiwmCachedFeatureData
        from torch.utils.data import DataLoader
        dataset = BiwmCachedFeatureData(args.preencoded_dir,
                                        caption_only=getattr(args, "preenc_caption_only", False))
        live_sampler = None
        if world > 1:
            from torch.utils.data.distributed import DistributedSampler
            live_sampler = DistributedSampler(dataset, num_replicas=world // args.cp_size,
                                              rank=rank // args.cp_size, shuffle=True,
                                              seed=args.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=1, sampler=live_sampler,
                            shuffle=(live_sampler is None),
                            num_workers=args.dataloader_num_workers, collate_fn=_live_batch_merge)
        vae = init_hy15_vae(args.vae_path, device)
        llm_dir = args.text_encoder_path or os.path.join(
            os.path.dirname(args.vae_path.rstrip("/")), "text_encoder", "llm")
        mllm = init_mllm(llm_dir, device)
    else:  # live: mp4 → online VAE + MLLM(Qwen2.5-VL) encoding (t2v: byt5/vision fed zeros)
        from pipelines.dataset.biwm_camera_text_dataset import BiwmCamCaptionData
        from torch.utils.data import DataLoader
        dataset = BiwmCamCaptionData(
            video_dir=args.biwm_video_dir, caption_json=args.biwm_caption_json,
            width=args.num_width, height=args.num_height, num_frames=args.num_frames,
            vae_temporal_factor=HY15_VAE_TIME_STRIDE)
        # CP-group-aware sampling: DP dim = world // cp_size, dp_rank = rank // cp_size. Same CP group → same sample;
        #   different CP groups → different samples. cp_size==1 degenerates to per-rank DP.
        live_sampler = None
        if world > 1:
            from torch.utils.data.distributed import DistributedSampler
            _dp_size = world // args.cp_size
            _dp_rank = rank // args.cp_size
            live_sampler = DistributedSampler(dataset, num_replicas=_dp_size, rank=_dp_rank,
                                              shuffle=True, seed=args.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=1, sampler=live_sampler,
                            shuffle=(live_sampler is None),
                            num_workers=args.dataloader_num_workers,
                            collate_fn=_live_batch_merge)
        vae = init_hy15_vae(args.vae_path, device)
        llm_dir = args.text_encoder_path or os.path.join(
            os.path.dirname(args.vae_path.rstrip("/")), "text_encoder", "llm")
        mllm = init_mllm(llm_dir, device)

    params = [p for p in model.parameters() if p.requires_grad]
    betas = tuple(float(x) for x in args.betas.split(","))
    if args.optimizer == "muon":
        # MuonOpt: Newton-Schulz orthogonalized momentum for ≥2D params, 1D/embed/head fall back to AdamW
        from pipelines.common.muon import build_muon_optimizer
        optimizer = build_muon_optimizer(model, lr=args.learning_rate,
                                       weight_decay=args.weight_decay, adamw_betas=betas)
        rank0_print("[HY15] optimizer = MuonOpt (consistent with minWM; 2D→Newton-Schulz, 1D→AdamW backup)")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate,
                                      weight_decay=args.weight_decay, betas=betas)
        rank0_print("[HY15] optimizer = AdamW")
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                             num_warmup_steps=args.lr_warmup_steps,
                             num_training_steps=args.max_train_steps)

    rank0_print(f"[HY15] start training: max_steps={args.max_train_steps}, "
           f"trainable={sum(p.numel() for p in params)/1e9:.3f}B, world={world}, opt={args.optimizer}")
    os.makedirs(args.output_dir, exist_ok=True)

    _epoch = [0]
    _live_sampler = locals().get("live_sampler", None)

    def _fetch(it):
        """Get the next batch dict; live mode skips bad samples and encodes online. Returns (batch, it).
        On epoch change, call set_epoch on the sampler so the same CP group reshuffles consistently."""
        while True:
            try:
                raw = next(it)
            except StopIteration:
                _epoch[0] += 1
                if _live_sampler is not None:
                    _live_sampler.set_epoch(_epoch[0])
                it = iter(loader)
                raw = next(it)
            if args.data_mode == "live":
                if isinstance(raw, tuple) and raw and raw[0] == "skip":
                    continue
                return assemble_live_batch(raw, vae, mllm, args, device), it
            return raw, it

    # Validation: needs VAE decode; live mode already loaded it, preencoded mode loads on demand
    if args.validation_interval > 0 and vae is None:
        vae = init_hy15_vae(args.vae_path, device)
    val_batch = None    # fixed validation sample (takes the first batch)

    from collections import deque
    step_times = deque(maxlen=50)
    step, loader_iter = 0, iter(loader)
    # pre-training validation (step 0): run execute_validation once before training to confirm the base weights +
    #   validation path are correct as a whole. Takes the first batch as the fixed validation sample.
    if (getattr(args, "validate_before_training", False)
            and args.validation_interval > 0 and vae is not None):
        _vb, loader_iter = _fetch(loader_iter)
        val_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in _vb.items()}
        rank0_print("[HY15][val] === pre-training validation (step 0, base weights) ===")
        try:
            execute_validation(model, vae, mllm, val_batch, args, 0, device, rank)
        except Exception as e:
            rank0_print(f"[HY15][val] step 0 pre-training validation failed (skipped): {type(e).__name__}: {e}")
        model.train()
    while step < args.max_train_steps:
        _t0 = time.perf_counter()
        optimizer.zero_grad()
        total = 0.0
        for _ in range(args.gradient_accumulation_steps):
            batch, loader_iter = _fetch(loader_iter)
            loss = run_one_step(model, batch, args, device, step) / args.gradient_accumulation_steps
            loss.backward()
            total += loss.item()
        gnorm = model.clip_grad_norm_(args.max_grad_norm) if hasattr(model, "clip_grad_norm_")\
            else torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        lr_sched.step()
        step += 1
        step_times.append(time.perf_counter() - _t0)
        if on_main_rank() and step % args.log_interval == 0:
            st = step_times[-1]
            avg = sum(step_times) / len(step_times)
            mem = torch.cuda.max_memory_allocated() / 1024**3
            try:
                _ep = step * args.gradient_accumulation_steps / max(1, len(loader))
            except TypeError:
                _ep = 0.0
            rank0_print(f"[Step {step}/{args.max_train_steps}] loss={total:.4f}  epoch={_ep:.2f}  "
                   f"grad={float(gnorm):.4f}  lr={lr_sched.get_last_lr()[0]:.2e}  "
                   f"time={st:.2f}s  avg={avg:.2f}s  mem={mem:.1f}GB")
            torch.cuda.reset_peak_memory_stats()
        if (args.validation_interval > 0 and vae is not None and step >= args.first_validation_step
                and (step % args.validation_interval == 0 or step == args.first_validation_step)):
            try:
                # each validation takes a new clip (new caption + clip_action) so prompt/action varies across steps.
                #   Under CP, _fetch is lockstep within the group; consumes only 1 extra training sample per interval.
                _vbatch, loader_iter = _fetch(loader_iter)
                execute_validation(model, vae, mllm, _vbatch, args, step, device, rank)
            except Exception as e:
                rank0_print(f"[HY15][val] step {step} validation failed (skipped): {type(e).__name__}: {e}")
            model.train()
        if step % args.checkpointing_steps == 0:
            store_checkpoint(model, args.output_dir, step)

    store_checkpoint(model, args.output_dir, args.max_train_steps)
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_options():
    p = argparse.ArgumentParser("HunyuanVideo-1.5 stage1 (native forward_bi)")
    # data
    p.add_argument("--data_mode", choices=["preencoded", "live", "preenc"], default="preencoded",
                   help="preencoded=minWM .pt; live=mp4 online VAE+MLLM; preenc=biWM preencoded .pt (preencode_hy15.py, faster)")
    p.add_argument("--preencoded_dir", type=str, default="",
                   help="preenc mode: preencode_hy15.py output directory (contains per-clip .pt)")
    p.add_argument("--preenc_caption_only", action="store_true",
                   help="preenc mode uses caption-only text (stage2 DMD); otherwise uses <camera> text (stage1)")
    p.add_argument("--json_path", type=str, required=False,
                   help="preencoded train_index.json (data_mode=preencoded)")
    # live mode (mp4 → online VAE+MLLM encoding)
    p.add_argument("--biwm_video_dir", type=str, default="",
                   help="live: video directory (dataset/videos or dataset/video_real)")
    p.add_argument("--biwm_caption_json", type=str, default="",
                   help="live: caption/pose json (videos_syn.json / videos_real.json)")
    p.add_argument("--text_encoder_path", type=str, default="",
                   help="live: MLLM(Qwen2.5-VL) directory; if empty, uses <vae_path>/../text_encoder/llm")
    p.add_argument("--num_frames", type=int, default=77)
    p.add_argument("--num_height", type=int, default=480)
    p.add_argument("--num_width", type=int, default=832)
    p.add_argument("--training_mode", choices=["t2v", "i2v"], default="t2v")
    p.add_argument("--window_frames", type=int, default=20)
    p.add_argument("--i2v_rate", type=float, default=0.0)
    p.add_argument("--training_cfg_rate", type=float, default=0.0)
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    # model / weights
    p.add_argument("--pretrained_model_path", type=str, default="",
                   help="HY15 DiT transformer directory (config.json + safetensors)")
    p.add_argument("--vae_path", type=str, default="")
    p.add_argument("--camera_mode", choices=["camtext"], default="camtext",
                   help="camtext=camera via text (cam-text concatenated into caption, full fine-tuning); PRoPE continuous camera not supported for now")
    p.add_argument("--use_discrete_action", action="store_true", default=True)
    p.add_argument("--no_discrete_action", dest="use_discrete_action", action="store_false")
    # flow matching
    p.add_argument("--sigma_shift", type=float, default=3.0,
                   help="training flow-matching sigma shift (=official train.py train_timestep_shift=3.0)")
    p.add_argument("--validation_shift", type=float, default=5.0,
                   help="validation/inference sampler shift (=official validation_timestep_shift=5.0); decoupled from sigma_shift")
    # optim
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--betas", type=str, default="0.9,0.999")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                   help="muon=consistent with minWM (2D Newton-Schulz + 1D AdamW backup); adamw=ordinary")
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps", type=int, default=20)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # sequence parallelism (CP/Ulysses): 1=off (pure DP); >1 must divide world and the latent token count.
    p.add_argument("--cp_size", type=int, default=1,
                   help="CP (sequence parallelism) group size, Ulysses; 1=off; DP dim=world//cp_size")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--max_train_steps", type=int, default=20000)
    # io
    p.add_argument("--output_dir", type=str, default="./logs/hy15/stage1")
    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # validation (official: sampler shift=5.0, cfg=6.0, 50 steps)
    p.add_argument("--validation_interval", type=int, default=50, help="validate every N steps; 0=off")
    p.add_argument("--val_same_input", action="store_true", default=False,
                   help="communication self-check: all ranks use the same action+caption+seed")
    p.add_argument("--first_validation_step", type=int, default=50,
                   help="earliest step for the first validation; must be <= validation_interval")
    p.add_argument("--validate_before_training", action="store_true", default=False,
                   help="run one validation before training (step 0) to confirm the base weights + validation path")
    p.add_argument("--diffusion_sampling_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=6.0,
                   help="validation CFG guidance strength; >1 enables cond+uncond dual forward, 1.0=no CFG")
    p.add_argument("--fps", type=float, default=24.0,
                   help="validation video save fps; HY15 official=24")
    return p.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main(parse_options())
