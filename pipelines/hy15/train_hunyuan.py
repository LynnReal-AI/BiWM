# BiWM HunyuanVideo-1.5 Stage 1 training entry (cam-text, discrete action, or PRoPE).
# =============================================================================
# HunyuanVideo-1.5 (HY15) stage1 训练入口 —— wan 风格单入口, 原生 forward_bi
# =============================================================================
# 与 fastvideo/wan_model_2_2.py 同样的"单文件入口"风格(arg / 分布式 / 模型构建 / 数据 /
# 训练循环 / checkpoint), 但 **不套 WanModel 适配层**: 直接驱动 biWM/hunyuan 的
# ARHunyuanVideo_1_5_DiffusionTransformer 原生双流 forward_bi。
#
# 文本编码器是 MLLM(Qwen2.5-VL-7B), 不是 T5:
#   - 默认(--data_mode preencoded): 读 minWM 预编码 .pt(已含真实 MLLM/byt5/SigLIP 的
#     prompt_embeds/byt5_text_states/vision_states), 训练循环里**不跑编码器**;
#   - 可选(--data_mode live): 走 mp4 数据集 + 在线 MLLM 编码(见 _encode_text_mllm, 待接 Qwen2.5-VL)。
#
# 相机(双流逐帧对应, 见 model.forward_bi):
#   - 离散: 逐帧 action[B,T] → action_in 加到每帧 AdaLN 调制向量 vec (zero-init);
#   - 连续 PRoPE: viewmats/Ks 逐 token camera-rope (img_attn_prope_proj, zero-init)。
#
# 权重: --pretrained_model_path 指向 ckpts/HunyuanVideo-1.5/transformer/{480p_i2v|480p_t2v}
#       (diffusers 格式 config.json + safetensors); 双流-only 模型对真 ckpt strict=False best-effort。
#       VAE: hunyuan/vae AutoencoderKLConv3D (仅 live 编码 / 验证解码用)。
#
# ⚠️ 本文件按"先写对、后跑通"的约定写; 真权重/数据齐了再在 DLC 上冒烟调。
# =============================================================================
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from hunyuan.modules.model import (ARHunyuanVideo_1_5_DiffusionTransformer,
                                   MMDoubleStreamBlock)
from hunyuan.configs.hunyuan_t2v import HUNYUAN_T2V_CONFIG
from pipelines.dataset.hy15_camera_dataset import make_camera_plucker_loader as build_camera_plucker_dataloader

# HunyuanVideo-1.5 VAE 几何 (实测 ckpts/HunyuanVideo-1.5/vae/config.json):
#   空间 16x / 时间 4x / latent 32ch / scaling_factor 1.03682 (体积压缩 1024x, 净压缩 96x)
HY15_LATENT_CHANNELS = 32
HY15_VAE_SPATIAL_STRIDE = 16
HY15_VAE_TEMPORAL_STRIDE = 4
HY15_VAE_SCALING_FACTOR = 1.03682

# 81 类离散相机动作 → 相机文本 (action_label = trans*9 + rot)。video_real 每 clip 一个常量动作,
# 故整段共享一句相机文本, 拼进 caption 由文本流控制相机运动(取代 PRoPE)。
_TRANS_TEXT = ["Camera does not move.", "Camera moves forward.", "Camera moves backward.",
               "Camera moves right.", "Camera moves left.", "Camera moves forward-right.",
               "Camera moves forward-left.", "Camera moves backward-right.", "Camera moves backward-left."]
_ROT_TEXT = ["Camera does not rotate.", "Camera pitches up.", "Camera pitches down.",
             "Camera yaws right.", "Camera yaws left.", "Camera pitches up and yaws right.",
             "Camera pitches up and yaws left.", "Camera pitches down and yaws right.",
             "Camera pitches down and yaws left."]


def action_to_camtext(a: int) -> str:
    a = int(a)
    return f"{_TRANS_TEXT[a // 9]} {_ROT_TEXT[a % 9]}"


# 定制系统提示词 —— 相机描述放进 <camera>...</camera>, 其余只写静态画面(无运动/相机)。
#   设计: 把"相机控制信号"与"静态内容"在文本里显式解耦, 让 MLLM 把相机语义集中在 <camera> 段;
#   数据集 caption 已是静态-only(严禁运动/运镜)。uncond(空串)经此模板=系统提示+空用户内容(分布内,
#   不退化)→ 可配 CFG(对齐 minWM 推理 cfg6+空串neg)。crop_start=-1 自动裁掉系统提示 token, 保留用户内容
#   (含 <camera> 段)的 hidden(其表征已在因果注意力里吸收了系统提示上下文)。
CAMTEXT_SYS_PROMPT = (
    "You are a helpful assistant for a game-world video generator. The prompt has a "
    "<camera>...</camera> tag that describes ONLY the camera motion (translation and rotation, "
    "e.g. moving/panning/tilting), followed by text that describes ONLY the static scene "
    "(characters, objects, environment, art style, lighting) with NO motion. Read the camera "
    "movement from the <camera> tag and the static appearance from the rest."
)
CAMTEXT_PROMPT_TEMPLATE = {
    "template": [
        {"role": "system", "content": CAMTEXT_SYS_PROMPT},
        {"role": "user", "content": "{}"},
    ],
    "crop_start": -1,
}


def build_camtext_prompt(cam_text: str, caption: str) -> str:
    """拼成 <camera>相机</camera> + 静态caption (相机在前/tag包裹; caption 只静态, 无运动/相机)。"""
    cap = (caption or "").rstrip()
    return f"<camera>{cam_text}</camera> {cap}".strip()


# =============================================================================
# 小工具
# =============================================================================
def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def mprint(*a, **k):
    if is_main():
        print(*a, **k, flush=True)


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1 and not dist.is_initialized():
        from datetime import timedelta
        # NCCL watchdog 默认 10min, 重步(首个 gen-update +371s / 验证)易误杀 → 提到 2h。
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(seconds=7200))
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world


# =============================================================================
# 模型构建 + 权重加载
# =============================================================================
def _accepted_init_keys():
    import inspect
    return {p for p in inspect.signature(
        ARHunyuanVideo_1_5_DiffusionTransformer.__init__).parameters if p != "self"}


def _load_transformer_config(weights_dir):
    """有 config.json 则**全量**用它构建(只过滤 __init__ 接受的键), 否则用占位配置(随机初始化)。
    ★ 必须全量: 真 ckpt 的 in_channels/patch_size/text_states_dim(3584,Qwen2.5-VL)/vision_states_dim(1152,SigLIP)/
      mm_double_blocks_depth(54)/rope_theta 等都得对上, 否则 load_state_dict 形状不匹配。"""
    cj = os.path.join(weights_dir, "config.json") if weights_dir and os.path.isdir(weights_dir) else None
    if cj and os.path.exists(cj):
        accepted = _accepted_init_keys()
        file_cfg = json.load(open(cj))
        cfg = {k: v for k, v in file_cfg.items() if not k.startswith("_") and k in accepted}
        mprint(f"[HY15] 用 config.json 全量构建: hidden={cfg.get('hidden_size')}, "
               f"double={cfg.get('mm_double_blocks_depth')}, in_ch={cfg.get('in_channels')}, "
               f"patch={cfg.get('patch_size')}, text_dim={cfg.get('text_states_dim')}, "
               f"vision_dim={cfg.get('vision_states_dim')}, concat={cfg.get('concat_condition')}")
        return cfg
    mprint("[HY15] 无 config.json, 用占位 HUNYUAN_T2V_CONFIG (随机初始化冒烟)")
    return dict(HUNYUAN_T2V_CONFIG)


def _load_sd_shape_tolerant(model, sd):
    """只加载与当前模型同名同形状的权重; 形状不一致或多余的跳过(报告)。
    strict=False 不会跳过形状不匹配 → 必须自己过滤。"""
    msd = model.state_dict()
    keep, mism = {}, []
    for k, v in sd.items():
        if k in msd and tuple(msd[k].shape) == tuple(v.shape):
            keep[k] = v
        elif k in msd:
            mism.append((k, tuple(v.shape), tuple(msd[k].shape)))
    missing, unexpected = model.load_state_dict(keep, strict=False)
    return missing, unexpected, mism


def _load_safetensors_dir(model_dir):
    """加载 diffusers 格式 transformer 权重(单 / 分片 safetensors)。"""
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


def build_hy15_transformer(args, device, dtype=torch.bfloat16):
    tdir = args.pretrained_model_path
    cfg = _load_transformer_config(tdir)
    mprint(f"[HY15] 构建 ARHunyuanVideo_1_5_DiffusionTransformer (hidden={cfg['hidden_size']}, "
           f"double_blocks={cfg['mm_double_blocks_depth']})")
    model = ARHunyuanVideo_1_5_DiffusionTransformer(**cfg)
    model._biwm_config = dict(cfg)   # 存构建配置, save_checkpoint 据此写 config.json(同 Wan, 使 ckpt 可作 pretrained_model_path 直接加载)
    # img_in 是否吃 concat 条件(2*C+1): i2v ckpt 通常 True; 占位 t2v config False(只喂 C)
    args._concat_condition = bool(cfg.get("concat_condition", False))
    # 相机参数(zero-init → 初期 no-op): PRoPE 逐 token + 离散逐帧 action
    # camera_mode:
    #   camtext(默认)=取消 PRoPE, 相机由文本(cam-text 拼进 caption)控制 → 不加任何相机参数, 全参训练基座 DiT(missing 应为 0);
    #   action      =【时间步调控相机/逐帧调控】取消 PRoPE, 不在 caption 写相机文本; 逐 latent 帧的 81 类离散 action
    #                经 TimestepEmbedder(action_in, zero-init) 加到 AdaLN 调制 vec 上(与 timestep 同一注入路径), 实现逐帧相机控制;
    #   prope       = 旧行为, 加 PRoPE(viewmats/Ks)(+可选离散 action)。
    if args.camera_mode == "prope":
        model.add_prope_parameters()
        if args.use_discrete_action:
            model.add_discrete_action_parameters()
        mprint("[HY15] camera_mode=prope (PRoPE viewmats/Ks + 离散 action)")
    elif args.camera_mode == "action":
        model.add_discrete_action_parameters()   # 仅加逐帧离散 action(TimestepEmbedder→AdaLN vec), 不加 PRoPE
        mprint("[HY15] camera_mode=action (时间步调控/逐帧离散 action 经 TimestepEmbedder 注入 AdaLN vec; 取消 PRoPE; caption 无相机文本)")
    else:
        mprint("[HY15] camera_mode=camtext (取消 PRoPE; 相机走文本 cam-text 拼进 caption; 全参训练)")

    # 加载真权重 (双流-only, 对真 ckpt 的单流 key 会落入 unexpected, strict=False)
    if tdir and os.path.isdir(tdir):
        try:
            sd = _load_safetensors_dir(tdir)
            missing, unexpected, mism = _load_sd_shape_tolerant(model, sd)
            mprint(f"[HY15] 权重加载: missing={len(missing)} unexpected={len(unexpected)} "
                   f"shape-mismatch-skipped={len(mism)}")
            if mism:
                mprint(f"[HY15]  跳过形状不匹配(前5): {mism[:5]}")
            del sd
        except FileNotFoundError:
            mprint(f"[HY15] 未找到 transformer 权重({tdir}), 随机初始化。")
    else:
        mprint(f"[HY15] 无 transformer 权重目录, 随机初始化。")

    model = model.to(device=device, dtype=dtype)
    if args.gradient_checkpointing:
        # 直接置标志(forward_bi 据此对 double_blocks 做 torch.utils.checkpoint); 不依赖 diffusers API
        model.gradient_checkpointing = True
        mprint("[HY15] gradient checkpointing ON (forward_bi 逐 block checkpoint)")
    return model


def wrap_fsdp(model, device, args):
    import functools
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        MixedPrecision, ShardingStrategy,
                                        BackwardPrefetch)
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={MMDoubleStreamBlock})
    mprint("[HY15] FSDP FULL_SHARD wrap on MMDoubleStreamBlock")
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


# =============================================================================
# 条件 latent (i2v concat + mask) —— 复刻 minWM bidir 的 _prepare_cond_latents
# =============================================================================
def _merge_by_mask(t0, t1, mask, dim=2):
    idx = torch.nonzero(mask).squeeze(1)
    out = t0.clone()
    sl = [slice(None)] * out.dim()
    sl[dim] = idx
    out[tuple(sl)] = t1[tuple(sl)]
    return out


def resolve_sample_task(args):
    """每条样本解析具体 task: mixed → 按 v2v_rate/i2v_rate 掷骰 v2v/i2v/t2v; 非 mixed → 恒为 training_mode。
    i2v/v2v 改【时间维 context】—— 三者都是"前 n_ctx 帧 clean(σ=0) 拼时间维"的特例
      (t2v=0 / i2v=1 / v2v=多帧); 掷骰顺序: r<v2v_rate→v2v, <v2v_rate+i2v_rate→i2v, 余 t2v。"""
    if args.training_mode != "mixed":
        return args.training_mode
    r = random.random()
    v2v_r = float(getattr(args, "v2v_rate", 0.0))
    i2v_r = float(getattr(args, "i2v_rate", 0.0))
    if r < v2v_r:
        return "v2v"
    if r < v2v_r + i2v_r:
        return "i2v"
    return "t2v"


def sample_n_ctx(task, t_lat, block_k=4):
    """该样本的 clean context 帧数(时间维, σ=0): t2v=0 / i2v=1 / v2v=随机 block 对齐{K,2K,...}。
    i2v 是 v2v 的特例(n_ctx=1); v2v 的 n_ctx 对齐 stage2 block_K 粒度,
      至少留 block_k 帧目标(n_ctx ≤ t_lat-1)。对齐 stage2 自回归历史长度{0,K,2K,...}。"""
    if task == "i2v":
        return 1
    if task == "v2v":
        k = max(1, int(block_k))
        max_blocks = max(1, (t_lat - k) // k)          # 至少留 k 帧目标
        nb = random.randint(1, max_blocks)
        return min(nb * k, t_lat - 1)
    return 0                                            # t2v


def get_task_mask(training_mode, t_lat, device):
    m = torch.zeros(t_lat, device=device)
    if training_mode == "i2v":
        m[0] = 1.0
    return m


def prepare_cond_latents(training_mode, image_cond, latents):
    B, C, T, H, W = latents.shape
    if image_cond is not None and training_mode == "i2v":
        latents_concat = image_cond.repeat(1, 1, T, 1, 1).clone()
        latents_concat[:, :, 1:, :, :] = 0.0
    else:
        latents_concat = torch.zeros_like(latents)
    mask = get_task_mask(training_mode, T, latents.device)
    zeros = torch.zeros(B, 1, T, H, W, device=latents.device, dtype=latents.dtype)
    ones = torch.ones(B, 1, T, H, W, device=latents.device, dtype=latents.dtype)
    mask_concat = _merge_by_mask(zeros, ones, mask, dim=2)
    return torch.cat([latents_concat, mask_concat], dim=1)  # (B, C+1, T, H, W)


# =============================================================================
# live 实时编码 (t2v): HunyuanVideo VAE + MLLM(Qwen2.5-VL); byt5/vision 喂零
# =============================================================================
# t2v 下: byt5(游戏片段无渲染文字 glyph) 与 vision_states(t2v 屏蔽) 均喂零 →
#   不需要 gated 的 SigLIP / modelscope 的 Glyph, 只用已下载的 VAE + Qwen2.5-VL。
BYT5_MAX_LEN = 256
BYT5_DIM = 1472


def load_hy15_vae(vae_dir, device, dtype=torch.bfloat16):
    """加载 HunyuanVideo VAE(AutoencoderKLConv3D, 16x空间/4x时间/32ch)。"""
    from hunyuan.vae import AutoencoderKLConv3D
    cand = os.path.join(vae_dir, "vae") if os.path.isdir(os.path.join(vae_dir, "vae")) else vae_dir
    vae = AutoencoderKLConv3D.from_pretrained(cand, torch_dtype=dtype).to(device).eval().requires_grad_(False)
    vae._sf = float(getattr(vae.config, "scaling_factor", None) or HY15_VAE_SCALING_FACTOR)
    mprint(f"[HY15-live] VAE 就绪 {cand} (scaling_factor={vae._sf})")
    return vae


def load_mllm(llm_dir, device, dtype=torch.bfloat16, max_length=1000):
    """加载 HY15 的 MLLM 文本编码器(Qwen2.5-VL, 用 video prompt template 抽 hidden state)。"""
    from hunyuan.text_encoder import TextEncoder
    # 用【定制】系统提示词模板(CAMTEXT_PROMPT_TEMPLATE) —— 相机描述进 <camera>...</camera>,
    #   caption 只静态。带模板编码(对齐 minWM 推理): cond/uncond(空串)都在模型分布内, 空串neg 不退化 → 可配 CFG。
    te = TextEncoder(
        text_encoder_type="llm", tokenizer_type="llm",
        text_encoder_path=llm_dir, tokenizer_path=llm_dir,
        max_length=max_length, text_encoder_precision="bf16",
        prompt_template=CAMTEXT_PROMPT_TEMPLATE, prompt_template_video=CAMTEXT_PROMPT_TEMPLATE,
        # 真凶修复: 官方取 LLM 倒数第3层(skip=2)的 hidden state; 我们之前默认 None(最后一层)
        #   → 文本特征整段 OOD → 彩色噪声(cfg/shift/timestep_txt 怎么调都救不了, 因编码层就错)。
        hidden_state_skip_layer=2, apply_final_norm=False,
        device=device,
    )
    te = te.to(device).eval().requires_grad_(False)
    mprint(f"[HY15-live] MLLM(Qwen2.5-VL) 就绪 {llm_dir}")
    return te


@torch.no_grad()
def mllm_encode(te, captions, device):
    """captions:List[str] -> (prompt_embeds [B,L,3584], prompt_mask [B,L])。video 模板。"""
    be = te.text2tokens(captions, data_type="video", max_length=te.max_length)
    out = te.encode(be, data_type="video", device=device)
    return out.hidden_state, out.attention_mask


_NEG_EMB_CACHE = {}


@torch.no_grad()
def encode_neg_prompt(te, device):
    """CFG 的 uncond/neg = 空串 ""【用原始 li-dit-encode-video-json 模板】编码(= minWM HY1.5 neg_prompt;
    见 minWM generate_negative_prompts.py)。cond 走定制 <camera> 模板, 但 neg 用原始模板的标准空 neg ——
    与 base 模型 CFG 校准一致, 空 neg 不退化。结果缓存(空串固定, 只算一次)。临时换模板编码后即恢复。"""
    if "neg" in _NEG_EMB_CACHE:
        pe, pm = _NEG_EMB_CACHE["neg"]
        return pe.to(device), pm.to(device)
    from hunyuan.text_encoder import PROMPT_TEMPLATE
    saved = te.prompt_template_video
    try:
        te.prompt_template_video = dict(PROMPT_TEMPLATE["li-dit-encode-video-json"])  # 原始模板
        pe, pm = mllm_encode(te, [""], device)
    finally:
        te.prompt_template_video = saved                       # 恢复定制 <camera> 模板
    _NEG_EMB_CACHE["neg"] = (pe.detach(), pm.detach())
    return pe, pm


# =============================================================================
# 验证视频保存 (★ 与 wan_model_2_2 保持一致: torchvision write_video crf18 + GT|Gen 左右拼接)
# =============================================================================
def _frames_to_uint8(vid):
    """[B,3,T,H,W]∈[-1,1] → list[np.uint8 HWC] (取 batch 0)。"""
    f = ((vid[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu().numpy()
    f = (f * 255).astype(np.uint8)
    return [f[i] for i in range(f.shape[0])]


def save_video(video_frames, output_path, fps=16):
    """同 Wan save_video: torchvision write_video (h264, crf18, veryfast)。"""
    from torchvision.io import write_video
    video_tensor = torch.from_numpy(np.stack(video_frames))
    write_video(output_path, video_tensor, fps=int(fps), options={'crf': '18', 'preset': 'veryfast'})


def combine_videos_with_labels(left_frames, right_frames, output_path,
                               left_label="Ground Truth", right_label="Generated",
                               fps=16, font_scale=0.7, label_height=40):
    """同 Wan combine_videos_with_labels: 左右拼接 + 顶部文字标签 → torchvision write_video。"""
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
def make_live_batch(item, vae, te, args, device, caption_only=False):
    """Convert one public ``BiwmCamCaptionData`` sample into a training batch.
    camtext 模式(默认, 取消 PRoPE): 整段一个相机动作 → 一句相机文本拼进 caption, 相机由文本控制;
      不喂 viewmats/Ks/action。CFG: 按 training_cfg_rate 丢弃整段文本(含相机)→无条件(对齐 minWM)。
    prope 模式: 旧行为, 喂 viewmats/Ks/逐帧 action。byt5/vision/image_cond 喂零(t2v)。
    ★ caption_only=True (stage2 DMD 训练用): 文本只用 caption, 不拼相机文本(t2v DMD 不引入相机;
      相机控制在 stage1 已学进权重, stage2 蒸馏只对 caption-only 分布)。"""
    pixel_values = item["pixel_values"]                       # [T,C,H,W] ∈[-1,1]
    caption = item["caption"]
    action_labels = item["action_labels"]                     # [T_lat]
    # VAE 编码: [T,C,H,W] → [1,C,T,H,W]
    x = pixel_values.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype=next(vae.parameters()).dtype)
    latent = (vae.encode(x).latent_dist.sample() * vae._sf).to(torch.bfloat16)   # [1,32,T_lat,h,w]
    B, C, T_lat, h, w = latent.shape
    # 该 clip 的相机动作(常量): 取非零众数(video_real 整段同一动作; 首帧 init=0)
    al = torch.as_tensor(action_labels).reshape(-1).long()
    _nz = al[al > 0]
    clip_action = int(torch.bincount(_nz).argmax()) if _nz.numel() > 0 else 0
    # 文本 → MLLM; CFG dropout。相机描述进 <camera>...</camera>(在前/tag包裹),
    #   caption 只静态; 见 build_camtext_prompt + CAMTEXT_PROMPT_TEMPLATE。cam 在前+因果LLM → 相机渗透整段表征。
    cap = caption.rstrip() if caption else ""
    if caption_only:                                         # stage2 DMD 训练: 纯 caption, 不引入相机(无 <camera>)
        full_text = cap
    elif args.camera_mode == "camtext":
        full_text = build_camtext_prompt(action_to_camtext(clip_action), cap)
    else:                                                     # action(时间步调控) / prope: caption 纯静态, 相机不进文本(走逐帧 action)
        full_text = cap
    if random.random() < args.training_cfg_rate:
        full_text = ""                                       # 无条件(CFG 训练)
    pe, pm = mllm_encode(te, [full_text], device)            # [1,L,3584], [1,L]
    byt5 = torch.zeros(1, BYT5_MAX_LEN, BYT5_DIM, device=device, dtype=torch.bfloat16)
    byt5_m = torch.zeros(1, BYT5_MAX_LEN, device=device, dtype=torch.bfloat16)
    vis = torch.zeros(1, 1, 1, device=device, dtype=torch.bfloat16)
    # image_cond 恒取该 clip 首帧【干净 latent】(便宜, latent 已解码)。
    #   是否真正使用由 task 决定: i2v 时 prepare_cond_latents 经 concat 通道喂首帧 + mask[0]=1; t2v 时传 None 不用。
    #   这样 mixed 训练里同一条数据可被随机当 i2v 或 t2v 用, 无需重编码。原先恒喂零 → i2v 学不到首帧条件。
    img_cond = latent[:, :, :1, :, :].clone()               # [1,C,1,h,w] 首帧干净 latent
    # 每条样本的 task: mixed 模式按 i2v_rate 概率掷骰 (i2v else t2v); 否则恒为 training_mode。
    task_this = resolve_sample_task(args)
    # 相机: camtext 模式不喂(走文本); action 模式喂【逐帧 action】(无 PRoPE); prope 模式喂 viewmats/Ks/逐帧 action
    vm = ks = action_t = None
    if args.camera_mode in ("prope", "action"):
        action_t = al.reshape(1, -1).to(device)              # ★ 逐帧 action(保留 pose_str 的每段不同动作, 不塌缩成常量)
        if action_t.shape[1] != T_lat:
            # nearest 对齐到 latent 帧数, 保离散类别不被插值破坏
            action_t = F.interpolate(action_t.float().unsqueeze(1), size=T_lat, mode="nearest").squeeze(1).long()
    if args.camera_mode == "prope":                          # PRoPE 额外喂连续 viewmats/Ks(action 模式不喂)
        if item.get("viewmats") is not None and item.get("Ks") is not None:
            vm = item["viewmats"].unsqueeze(0).to(device, torch.float32)
            ks = item["Ks"].unsqueeze(0).to(device, torch.float32)
        else:
            vm = torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(1, T_lat, 1, 1)
            ks = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, T_lat, 1, 1)
    return {
        "latent": latent, "prompt_embed": pe, "prompt_mask": pm,
        "viewmats": vm, "Ks": ks, "action": action_t,
        "byt5_text_states": byt5, "byt5_text_mask": byt5_m, "vision_states": vis,
        "image_cond": img_cond, "i2v_mask": torch.ones_like(latent),
        "task": task_this,                                 # ★ 本条样本的 i2v/t2v (mixed 掷骰结果); train_one_step 据此用 image_cond
        "caption": caption, "clip_action": clip_action,   # 验证 per-rank 重编 cam-text 用
    }


# =============================================================================
# 单步训练 (flow matching, 原生 forward_bi)
# =============================================================================
def train_one_step(model, batch, args, device, step):
    latents = batch["latent"].to(device, dtype=torch.bfloat16)         # (B,C,T,H,W)
    prompt_embed = batch["prompt_embed"].to(device, dtype=torch.bfloat16)
    prompt_mask = batch["prompt_mask"].to(device, dtype=torch.bfloat16)
    byt5_states = batch["byt5_text_states"].to(device, dtype=torch.bfloat16)
    byt5_mask = batch["byt5_text_mask"].to(device, dtype=torch.bfloat16)
    vision_states = batch["vision_states"].to(device, dtype=torch.bfloat16)
    image_cond = batch["image_cond"].to(device, dtype=torch.bfloat16)
    i2v_mask = batch["i2v_mask"].to(device, dtype=torch.bfloat16)
    # 相机: camtext 模式为 None(走文本, 不喂 prope/action); prope 模式才有
    _vm = batch.get("viewmats"); _ks = batch.get("Ks"); _act = batch.get("action")
    viewmats = _vm.to(device, dtype=torch.float32) if _vm is not None else None
    Ks = _ks.to(device, dtype=torch.float32) if _ks is not None else None
    action = _act.to(device).long() if _act is not None else None

    B, C, T_lat, H, W = latents.shape
    # task 来自 make_live_batch 掷骰(mixed: v2v/i2v/t2v); 兜底回退 training_mode。
    task = batch.get("task", args.training_mode)
    # i2v/v2v 改【时间维 context】(对齐 stage2 自回归历史消费) ——
    #   前 n_ctx 帧 clean(σ=0)拼在序列前(时间维 dim=2), 其余帧按采样 σ 加噪; 逐帧 timestep(context=0);
    #   channel cond 恒走 t2v(零通道, 与 stage2 dit_velocity 一致); loss 只算目标帧(context 给定不算)。
    #   i2v=n_ctx 1 / v2v=n_ctx∈{K,2K..} / t2v=0。
    n_ctx = sample_n_ctx(task, T_lat, getattr(args, "v2v_block_k", 4))

    # --- flow-matching: 目标帧 sigma 采样对齐 Wan stage1 —— rand→shift ---
    noise = torch.randn_like(latents)
    u = torch.rand(B, device=device)
    sh = args.sigma_shift
    sigma = sh * u / (1.0 + (sh - 1.0) * u)                 # (B,) 目标帧 σ
    # ★ 逐帧 σ: 前 n_ctx 帧=0(clean context), 其余=sigma → noisy 中 context 帧即 clean latent
    per_frame_sigma = sigma.view(B, 1).expand(B, T_lat).clone()   # (B,T_lat)
    if n_ctx > 0:
        per_frame_sigma[:, :n_ctx] = 0.0
    sig_f = per_frame_sigma.view(B, 1, T_lat, 1, 1).to(latents.dtype)
    noisy = (1.0 - sig_f) * latents + sig_f * noise
    timestep = (per_frame_sigma * 1000.0).reshape(-1).to(torch.bfloat16)   # (B*T_lat,) 逐帧
    # ★ 文本流时间步恒为 0 (文本不加噪, 对齐 minWM bidir / stage2)
    timestep_txt = torch.zeros(B, device=device, dtype=torch.bfloat16)        # (B,)
    target = noise - latents                                # flow-matching velocity 目标

    # --- channel cond: 恒走 t2v(零通道+零mask), 与 stage2 dit_velocity 一致; i2v/v2v 条件已在【时间维】 ---
    if getattr(args, "_concat_condition", False):
        cond = prepare_cond_latents("t2v", None, noisy)     # 零 (B, C+1, T, H, W)
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
            vision_states=None,                             # 时间维 context, 不用 SigLIP vision
            viewmats=viewmats,
            Ks=Ks,
            action=action,
            extra_kwargs=extra_kwargs,
            mask_type="t2v",                                # 走 t2v 路径(channel cond 零); context 靠逐帧 timestep 区分
            return_dict=False,
        )
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
        # ★ loss 只算目标帧(前 n_ctx 帧是给定 clean context, 不算 loss)
        loss_mask = torch.ones_like(latents)
        if n_ctx > 0:
            loss_mask[:, :, :n_ctx] = 0.0
        diff = (pred.float() * loss_mask - target.float() * loss_mask) ** 2
        loss = diff.sum() / torch.clamp(loss_mask.sum(), min=1.0)
    return loss


# =============================================================================
# checkpoint
# =============================================================================
def save_checkpoint(model, out_dir, step):
    """保存 checkpoint —— 格式对齐 Wan(pipelines/utils/checkpoint.py:save_checkpoint):
    FSDP FULL_STATE_DICT(offload_to_cpu, rank0_only) → 存
      checkpoint-{step}/diffusion_pytorch_model.safetensors  (safetensors, 非 model.pt)
      checkpoint-{step}/config.json                           (构建配置, 同 base 格式)
    这样 checkpoint 与 base 权重目录同构, 可直接 --pretrained_model_path checkpoint-{step} 续训/推理
    (走 _load_safetensors_dir + _load_transformer_config)。"""
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        FullStateDictConfig, StateDictType)
    import safetensors.torch as st
    fcfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fcfg):
        sd = model.state_dict()
    if is_main():
        d = os.path.join(out_dir, f"checkpoint-{step}")
        os.makedirs(d, exist_ok=True)
        # 权重: safetensors(同 Wan diffusion_pytorch_model.safetensors); 连续化避免非连续存储报错
        st.save_file({k: v.contiguous() for k, v in sd.items()},
                     os.path.join(d, "diffusion_pytorch_model.safetensors"))
        # config.json(同 Wan): 用构建时 cfg(来自 json, 可序列化); FSDP/checkpoint wrap 下逐层解包取属性
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
                mprint(f"[HY15] config.json 保存失败(跳过): {type(e).__name__}: {e}")
        mprint(f"[HY15] saved checkpoint (safetensors+config) -> {d}")


# =============================================================================
# 验证: 从噪声采样去噪 → VAE 解码 → 存 mp4 (t2v, 固定 val 样本)
# =============================================================================
# 验证用的代表性相机动作 (rank → 不同动作, 各 rank 独立采样不同运镜; 参考 wan 每 rank 独立)
#   9=forward 18=backward 27=right 36=left 1=pitch_up 2=pitch_down 3=yaw_right 4=yaw_left
_VAL_ACTIONS = [9, 18, 27, 36, 1, 2, 3, 4]


@torch.no_grad()
def run_validation(model, vae, mllm, val_batch, args, step, device, rank):
    """每 rank 独立采样一个不同的相机动作(camtext: 重编 caption+cam-text), 从噪声去噪 → VAE 解码
    → 叠摇杆(joystick) overlay → 各 rank 存各自 mp4。采样器对齐 minWM。"""
    import numpy as np
    from hunyuan.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
    model.eval()
    # mixed 模式验证用 i2v 展示新学的首帧条件能力 (val_batch.image_cond 已恒为 GT 首帧 latent)。
    task = "i2v" if args.training_mode == "mixed" else args.training_mode
    latent = val_batch["latent"].to(device, dtype=torch.bfloat16)
    B, C, T_lat, H, W = latent.shape

    # CP 下同一序列被组内各 rank 分片协同计算 → 同组必须同动作+同噪声种子,
    #   且每组只存一次(组内 rank0)。val_id 用 CP 组号(无 CP 时即 global rank)。
    from pipelines.utils.parallel_states import nccl_state as nccl_info, fetch_sequence_parallel_state as get_sequence_parallel_state
    if get_sequence_parallel_state():
        val_id = nccl_info.group_id            # 每个 CP 组采一个不同动作
        save_this = (nccl_info.rank_within_group == 0)
    else:
        val_id = rank                          # 纯 DP: 每 rank 一个动作(原行为)
        save_this = True

    # 通信自检开关 —— val_same_input=True 时所有 rank 用【完全相同】的 动作+caption+seed,
    #   若各 rank 输出仍不同 → 才是通信/FSDP问题; 若完全相同 → 通信无问题(差异纯来自各 rank 不同输入)。
    _same = getattr(args, "val_same_input", False)
    _vid = 0 if _same else val_id
    # 该 (rank/组) 的相机动作 + 文本条件
    # 验证动作必须用【GT clip 自己的真实动作 clip_action】, 而非任意采样的 _VAL_ACTIONS ——
    #   否则 GT(原始clip,真实动作) 与 Generated(采样动作) 不是同一动作: 会出现 "GT画面明显右偏但摇杆overlay画成直行"
    #   的错位(GT|Gen 无法对比)。改用 clip_action 后: Gen 用真实动作生成、GT 本就是真实动作、摇杆也是真实动作 → 三者一致,
    #   且与训练 make_live_batch(line 432 用 clip_action) 对齐。val_same_input(通信自检) 仍用固定动作 9。
    if _same:
        act = _VAL_ACTIONS[0]
    else:
        act = int(val_batch.get("clip_action", _VAL_ACTIONS[val_id % len(_VAL_ACTIONS)]))
    if args.camera_mode == "camtext" and mllm is not None:
        _cap = ("A medieval castle on a green hill under a clear blue sky" if _same
                else str(val_batch.get("caption", "")))
        full_text = build_camtext_prompt(action_to_camtext(act), _cap)  # <camera>cam</camera> + 静态caption
        pe, pm = mllm_encode(mllm, [full_text], device)
        viewmats = Ks = action = None
    else:                                    # prope: 用缓存的条件 + 缓存相机
        pe = val_batch["prompt_embed"].to(device, torch.bfloat16)
        pm = val_batch["prompt_mask"].to(device, torch.bfloat16)
        viewmats = val_batch["viewmats"].to(device, torch.float32) if val_batch.get("viewmats") is not None else None
        Ks = val_batch["Ks"].to(device, torch.float32) if val_batch.get("Ks") is not None else None
        action = val_batch["action"].to(device).long() if val_batch.get("action") is not None else None
    extra = {"byt5_text_states": val_batch["byt5_text_states"].to(device, torch.bfloat16),
             "byt5_text_mask": val_batch["byt5_text_mask"].to(device, torch.bfloat16)}
    img_cond = val_batch["image_cond"].to(device, torch.bfloat16)

    # CFG: uncond/neg = 空串 ""【用原始 li-dit 模板】编码(= minWM HY1.5 neg_prompt, encode_neg_prompt 缓存);
    #   cond 走定制 <camera> 模板, neg 走原始模板的标准空 neg(与 base CFG 校准一致)。guidance<=1 不做 CFG。
    guidance = float(getattr(args, "cfg_scale", 1.0))
    pe_u = pm_u = None
    if guidance > 1.0 and mllm is not None:
        from pipelines.common.dmd_algo import run_cfg as apply_cfg
        pe_u, pm_u = encode_neg_prompt(mllm, device)            # neg = 原始模板编码空串(同 minWM)

    g = torch.Generator(device=device).manual_seed(42 + _vid)  # 同CP组同噪声; val_same_input 时所有rank同噪声(=42)
    x = torch.randn(latent.shape, generator=g, device=device, dtype=torch.bfloat16)
    sched = FlowMatchDiscreteScheduler(shift=args.validation_shift, reverse=True, solver="euler")
    sched.set_timesteps(args.diffusion_sampling_steps, device=device)

    # 验证改【时间维 context】(对齐训练/stage2) —— i2v: 首帧 clean GT(σ=0)作 context,
    #   逐帧 timestep(context=0, 目标=t), 每步保持 context 帧=clean GT; channel cond 走 t2v(零)。
    n_ctx_val = 1 if task == "i2v" else 0
    if n_ctx_val > 0:
        x[:, :, :n_ctx_val] = latent[:, :, :n_ctx_val]         # context = clean GT 首帧

    def _pred(_pe, _pm, _hidden, _ts_flat):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(bi_inference=True, hidden_states=_hidden, timestep=_ts_flat,
                           timestep_txt=torch.zeros(B, device=device, dtype=torch.bfloat16),  # 文本流时间步恒0(对齐minWM)
                           text_states=_pe, text_states_2=None,
                           encoder_attention_mask=_pm, vision_states=None,
                           viewmats=viewmats, Ks=Ks, action=action,
                           extra_kwargs=extra, mask_type="t2v", return_dict=False)
        return out

    for t in sched.timesteps:
        # 逐帧 timestep: context 帧=0, 目标帧=t
        pf_ts = torch.full((B, T_lat), float(t), device=device, dtype=torch.bfloat16)
        if n_ctx_val > 0:
            pf_ts[:, :n_ctx_val] = 0.0
        ts_flat = pf_ts.reshape(-1)
        if getattr(args, "_concat_condition", False):
            condlat = prepare_cond_latents("t2v", None, x)     # 零(条件已在时间维 context)
            hidden = torch.cat([x, condlat], dim=1)
        else:
            hidden = x
        pred = _pred(pe, pm, hidden, ts_flat)                  # 条件
        if pe_u is not None:                                   # CFG: uncond + scale*(cond-uncond)
            pred = apply_cfg(pred, _pred(pe_u, pm_u, hidden, ts_flat), guidance)
        x = sched.step(pred, t, x).prev_sample
        if n_ctx_val > 0:
            x[:, :, :n_ctx_val] = latent[:, :, :n_ctx_val]     # 保持 context 帧干净(只更新目标帧)
    # 解码【生成视频】(无 overlay) + 【GT】(val_batch latent 的 VAE 重建) —— 与 Wan 一致, 用于左右对比
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vid = vae.decode((x / vae._sf).to(next(vae.parameters()).dtype)).sample   # [B,3,T,H,W]
    frames = _frames_to_uint8(vid)                            # list[np.uint8 HWC]
    gt_frames = None
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gt_vid = vae.decode((latent / vae._sf).to(next(vae.parameters()).dtype)).sample
        gt_frames = _frames_to_uint8(gt_vid)
    except Exception as e:
        mprint(f"[HY15][val] GT 解码跳过: {type(e).__name__}: {e}")

    if not save_this:    # CP: 每组只存一次(组内 rank0); 纯 DP: 每 rank 都存
        model.train(); return

    d = os.path.join(args.output_dir, "validation"); os.makedirs(d, exist_ok=True)
    fps = int(getattr(args, "fps", 24))
    cam_txt = action_to_camtext(act)
    # ★ 命名/数量/类型 对齐 Wan: validation_{prefix}.mp4(生成,无overlay) + _prompt.txt + combined_joystick_{prefix}.mp4(GT|Gen+摇杆)
    name_prefix = f"step_{step:07d}_{task}_camtext_video_real_rank_{rank:03d}_act{act}"
    # 1) 生成视频 (无 overlay, torchvision write_video crf18)
    save_video(frames, os.path.join(d, f"validation_{name_prefix}.mp4"), fps=fps)
    # 2) 验证元信息 txt (同 Wan)
    try:
        with open(os.path.join(d, f"validation_{name_prefix}_prompt.txt"), "w", encoding="utf-8") as mf:
            mf.write(f"Step: {step}\nRank: {rank}\nVal id(CP组/rank): {val_id}\n")
            mf.write(f"Action label: {act}\nCamera text: {cam_txt}\n")
            mf.write(f"Prompt: {full_text if (args.camera_mode=='camtext' and mllm is not None) else '<preencoded>'}\n")
            mf.write(f"CFG Scale: {guidance}\nSampling Steps: {args.diffusion_sampling_steps}\n")
            mf.write(f"Num Frames(latent): {T_lat}\nResolution: {W*HY15_VAE_SPATIAL_STRIDE}x{H*HY15_VAE_SPATIAL_STRIDE}\n")
            mf.write(f"FPS: {fps}\nSource: video_real\nTask Type: {task}\nCamera: camtext(<camera>tag)\n")
    except Exception as e:
        mprint(f"[HY15][val] metadata 写入跳过: {type(e).__name__}: {e}")
    # 3) GT|Gen 左右拼接 + 摇杆 overlay (同 Wan combined_joystick)
    try:
        from pipelines.common.control_overlay import superimpose_control_video as add_joystick_overlay
        # ★ action(时间步调控)模式: 相机逐帧变化 → 摇杆 overlay 用【逐帧 action】序列; 否则(camtext/常量)用 clip_action 常量
        if args.camera_mode == "action" and action is not None:
            al = action.detach().reshape(-1)[:T_lat].cpu().long()
            if al.numel() < T_lat:
                al = torch.cat([al, torch.full((T_lat - al.numel(),), int(act), dtype=torch.long)])
        else:
            al = torch.full((T_lat,), int(act), dtype=torch.long)
        gen_j = add_joystick_overlay(frames, al, vae_temporal_stride=HY15_VAE_TEMPORAL_STRIDE)
        if gt_frames is not None:
            gt_j = add_joystick_overlay(gt_frames, al, vae_temporal_stride=HY15_VAE_TEMPORAL_STRIDE)
            combine_videos_with_labels(gt_j, gen_j, os.path.join(d, f"combined_joystick_{name_prefix}.mp4"),
                                       "GT", "Generated", fps=fps)
        else:                                                # 无 GT 时只存带摇杆的生成视频
            save_video(gen_j, os.path.join(d, f"joystick_{name_prefix}.mp4"), fps=fps)
    except Exception as e:
        mprint(f"[HY15][val] joystick/combined 跳过: {type(e).__name__}: {e}")
    if is_main():
        mprint(f"[HY15][val] step {step}: 每(CP组/rank)一个动作; 存 validation_*/combined_joystick_* (act{act}) -> {d}")
    model.train()


# =============================================================================
# main
# =============================================================================
def _live_collate(b):
    """live DataLoader collate: batch_size=1, 直接返回那条 tuple (模块级, 可 pickle 给 worker)。"""
    return b[0]


def main(args):
    local_rank, rank, world = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    # 序列并行(CP, Ulysses)。复用 biWM 既有 nccl_info(与 Wan 共用 SP 组源);
    #   HY15 模型(forward_bi/attention) 通过 hunyuan/distributed.get_parallel_state() 读它。
    #   cp_size==1: sp_enabled=False, 模型所有 SP 分支跳过 = 纯 DP(行为同原单卡)。
    #   cp_size>1 : 同组各 rank chunk(dim=1) 序列 + 每 block all-to-all 全序列 attention + 末 all_gather。
    from pipelines.utils.parallel_states import setup_sequence_parallel_state as initialize_sequence_parallel_state, nccl_state as nccl_info
    initialize_sequence_parallel_state(args.cp_size)
    if args.cp_size > 1:
        mprint(f"[HY15] CP(序列并行) on: cp_size={args.cp_size}, "
               f"DP groups={world // args.cp_size}; 同组同样本, 跨组不同样本")

    model = build_hy15_transformer(args, device)
    model.train()
    model = wrap_fsdp(model, device, args)

    vae = mllm = None
    if args.data_mode == "preencoded":
        dataset, loader = build_camera_plucker_dataloader(
            json_path=args.json_path, causal=False, window_frames=args.window_frames,
            batch_size=args.train_batch_size, num_data_workers=args.dataloader_num_workers,
            drop_last=True, drop_first_row=False, seed=args.seed,
            cfg_rate=args.training_cfg_rate, i2v_rate=args.i2v_rate, task_type=args.training_mode,
        )
    elif args.data_mode == "preenc":
        # biWM 预编码(preencode_hy15.py 产的 .pt) —— 训练步直接读 latent+prompt_embed,
        #   跳过每步在线 VAE+MLLM 编码(stage1 提速根因)。验证仍需 VAE(解码)+MLLM(重编 cam-text), 故仍加载。
        from pipelines.dataset.biwm_camera_text_dataset import BiwmCachedFeatureData as BiwmPreencodedDataset
        from torch.utils.data import DataLoader
        dataset = BiwmPreencodedDataset(args.preencoded_dir,
                                        caption_only=getattr(args, "preenc_caption_only", False))
        live_sampler = None
        if world > 1:
            from torch.utils.data.distributed import DistributedSampler
            live_sampler = DistributedSampler(dataset, num_replicas=world // args.cp_size,
                                              rank=rank // args.cp_size, shuffle=True,
                                              seed=args.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=1, sampler=live_sampler,
                            shuffle=(live_sampler is None),
                            num_workers=args.dataloader_num_workers, collate_fn=_live_collate)
        vae = load_hy15_vae(args.vae_path, device)
        llm_dir = args.text_encoder_path or os.path.join(
            os.path.dirname(args.vae_path.rstrip("/")), "text_encoder", "llm")
        mllm = load_mllm(llm_dir, device)
    else:  # live: mp4 → 在线 VAE + MLLM(Qwen2.5-VL) 编码 (t2v: byt5/vision 喂零)
        from pipelines.dataset.biwm_camera_text_dataset import BiwmCamCaptionData
        from torch.utils.data import DataLoader
        dataset = BiwmCamCaptionData(
            video_dir=args.biwm_video_dir, caption_json=args.biwm_caption_json,
            width=args.num_width, height=args.num_height, num_frames=args.num_frames,
            vae_temporal_factor=HY15_VAE_TEMPORAL_STRIDE)
        # CP 组感知采样 —— DP 维度 = world // cp_size, dp_rank = rank // cp_size。
        #   同一 CP 组(dp_rank 相同)的各 rank 必须拿到**同一条样本**(它们各算序列的 1/cp 分片);
        #   不同 CP 组拿不同样本(数据并行)。cp_size==1 时退化为普通逐 rank DP(num_replicas=world)。
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
                            collate_fn=_live_collate)
        vae = load_hy15_vae(args.vae_path, device)
        llm_dir = args.text_encoder_path or os.path.join(
            os.path.dirname(args.vae_path.rstrip("/")), "text_encoder", "llm")
        mllm = load_mllm(llm_dir, device)

    params = [p for p in model.parameters() if p.requires_grad]
    betas = tuple(float(x) for x in args.betas.split(","))
    if args.optimizer == "muon":
        # ★ 与 minWM 对齐: Muon(≥2D 参数 Newton-Schulz 正交化动量, 1D/embed/head 退 AdamW)
        from pipelines.common.muon import build_muon_optimizer
        optimizer = build_muon_optimizer(model, lr=args.learning_rate,
                                       weight_decay=args.weight_decay, adamw_betas=betas)
        mprint("[HY15] optimizer = Muon (与 minWM 一致; 2D→Newton-Schulz, 1D→AdamW backup)")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate,
                                      weight_decay=args.weight_decay, betas=betas)
        mprint("[HY15] optimizer = AdamW")
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                             num_warmup_steps=args.lr_warmup_steps,
                             num_training_steps=args.max_train_steps)

    mprint(f"[HY15] start training: max_steps={args.max_train_steps}, "
           f"trainable={sum(p.numel() for p in params)/1e9:.3f}B, world={world}, opt={args.optimizer}")
    os.makedirs(args.output_dir, exist_ok=True)

    _epoch = [0]
    _live_sampler = locals().get("live_sampler", None)

    def _fetch(it):
        """取下一个 batch dict; live 模式跳过坏样本并在线编码。返回 (batch, it)。
        换 epoch 时对 CP/DP 采样器 set_epoch, 保证同 CP 组重洗一致、跨 epoch 顺序变化。"""
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
                if ((isinstance(raw, dict) and raw.get("skip")) or
                        (isinstance(raw, tuple) and raw and raw[0] == "skip")):
                    continue
                return make_live_batch(raw, vae, mllm, args, device), it
            return raw, it

    # 验证: 需要 VAE 解码; live 模式已加载, preencoded 模式按需加载
    if args.validation_interval > 0 and vae is None:
        vae = load_hy15_vae(args.vae_path, device)
    val_batch = None    # 固定验证样本(取第一条 batch)

    from collections import deque
    step_times = deque(maxlen=50)
    step, loader_iter = 0, iter(loader)
    # 训练前验证(step 0) —— 在权重被训练污染前先跑一遍 run_validation,
    #   确认 base 权重 + 验证路径(MLLM camtext 编码 / FlowMatch 采样器 / VAE 解码 / joystick overlay)整体正确。
    #   取首个 batch 作固定验证样本(后续循环 val_batch 已非 None → 不再重复缓存)。
    if (getattr(args, "validate_before_training", False)
            and args.validation_interval > 0 and vae is not None):
        _vb, loader_iter = _fetch(loader_iter)
        val_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in _vb.items()}
        mprint("[HY15][val] === 训练前验证 (step 0, base 权重) ===")
        try:
            run_validation(model, vae, mllm, val_batch, args, 0, device, rank)
        except Exception as e:
            mprint(f"[HY15][val] step 0 训练前验证失败(跳过): {type(e).__name__}: {e}")
        model.train()
    while step < args.max_train_steps:
        _t0 = time.perf_counter()
        optimizer.zero_grad()
        total = 0.0
        for _ in range(args.gradient_accumulation_steps):
            batch, loader_iter = _fetch(loader_iter)
            loss = train_one_step(model, batch, args, device, step) / args.gradient_accumulation_steps
            loss.backward()
            total += loss.item()
        gnorm = model.clip_grad_norm_(args.max_grad_norm) if hasattr(model, "clip_grad_norm_") \
            else torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        lr_sched.step()
        step += 1
        step_times.append(time.perf_counter() - _t0)
        # 逐步日志 (对齐 wan_model_2_2 tqdm postfix: loss/epoch/step/time/grad/lr + 显存)
        if is_main() and step % args.log_interval == 0:
            st = step_times[-1]
            avg = sum(step_times) / len(step_times)
            mem = torch.cuda.max_memory_allocated() / 1024**3
            try:
                _ep = step * args.gradient_accumulation_steps / max(1, len(loader))
            except TypeError:
                _ep = 0.0
            mprint(f"[Step {step}/{args.max_train_steps}] loss={total:.4f}  epoch={_ep:.2f}  "
                   f"grad={float(gnorm):.4f}  lr={lr_sched.get_last_lr()[0]:.2e}  "
                   f"time={st:.2f}s  avg={avg:.2f}s  mem={mem:.1f}GB")
            torch.cuda.reset_peak_memory_stats()
        if (args.validation_interval > 0 and vae is not None and step >= args.first_validation_step
                and (step % args.validation_interval == 0 or step == args.first_validation_step)):
            try:
                # 每次验证取一条【新】clip(新 caption+新 clip_action), 不复用固定样本 ——
                #   不同 step 的验证 prompt/动作会变化(看多样性)。CP 下 _fetch 同组 lockstep, 各 rank 一致;
                #   每 validation_interval 仅多消耗 1 条训练样本(可忽略)。代价: 无法纵向对比同一 prompt 的进步。
                _vbatch, loader_iter = _fetch(loader_iter)
                run_validation(model, vae, mllm, _vbatch, args, step, device, rank)
            except Exception as e:
                mprint(f"[HY15][val] step {step} 验证失败(跳过): {type(e).__name__}: {e}")
            model.train()
        if step % args.checkpointing_steps == 0:
            save_checkpoint(model, args.output_dir, step)

    save_checkpoint(model, args.output_dir, args.max_train_steps)
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser("HunyuanVideo-1.5 stage1 (native forward_bi)")
    # data
    p.add_argument("--data_mode", choices=["preencoded", "live", "preenc"], default="preencoded",
                   help="preencoded=minWM .pt; live=mp4在线VAE+MLLM; preenc=biWM预编码.pt(preencode_hy15.py,提速)")
    p.add_argument("--preencoded_dir", type=str, default="",
                   help="preenc 模式: preencode_hy15.py 输出目录(含 per-clip .pt)")
    p.add_argument("--preenc_caption_only", action="store_true",
                   help="preenc 模式用 caption-only 文本(stage2 DMD); 否则用 <camera> 文本(stage1)")
    p.add_argument("--json_path", type=str, required=False,
                   help="预编码 train_index.json (data_mode=preencoded)")
    # live 模式 (mp4 → 在线 VAE+MLLM 编码)
    p.add_argument("--biwm_video_dir", type=str, default="",
                   help="live: 视频目录 (dataset/videos 或 dataset/video_real)")
    p.add_argument("--biwm_caption_json", type=str, default="",
                   help="live: caption/pose json (preencode_input.json / video_real_input.json)")
    p.add_argument("--text_encoder_path", type=str, default="",
                   help="live: MLLM(Qwen2.5-VL) 目录; 空则取 <vae_path>/../text_encoder/llm")
    p.add_argument("--num_frames", type=int, default=77)
    p.add_argument("--num_height", type=int, default=480)
    p.add_argument("--num_width", type=int, default=832)
    p.add_argument("--training_mode", choices=["t2v", "i2v", "v2v", "mixed"], default="t2v",
                   help="t2v/i2v/v2v 固定; mixed=逐样本按 --v2v_rate/--i2v_rate 掷骰 v2v/i2v/t2v(时间维 clean context)")
    p.add_argument("--window_frames", type=int, default=20)
    p.add_argument("--i2v_rate", type=float, default=0.0,
                   help="mixed 模式 i2v 概率(时间维首帧 clean context, n_ctx=1)")
    p.add_argument("--v2v_rate", type=float, default=0.0,
                   help="mixed 模式 v2v 概率(时间维多帧 clean context); 掷骰: r<v2v_rate→v2v, <v2v_rate+i2v_rate→i2v, 余 t2v")
    p.add_argument("--v2v_block_k", type=int, default=4,
                   help="v2v 的 clean context 帧数按此粒度对齐(=stage2 dmd_block_K), n_ctx∈{K,2K,...}")
    p.add_argument("--training_cfg_rate", type=float, default=0.0)  # 默认不做训练期文本丢弃
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    # model / weights
    p.add_argument("--pretrained_model_path", type=str, default="",
                   help="HY15 DiT transformer 目录(config.json + safetensors)")
    p.add_argument("--vae_path", type=str, default="")
    p.add_argument("--camera_mode", choices=["camtext", "action", "prope"], default="camtext",
                   help="camtext=相机走文本(cam-text拼caption); action=时间步调控/逐帧离散action经TimestepEmbedder注入AdaLN(caption无相机文本); prope=PRoPE viewmats/Ks+离散action")
    p.add_argument("--use_discrete_action", action="store_true", default=True)
    p.add_argument("--no_discrete_action", dest="use_discrete_action", action="store_false")
    # flow matching
    p.add_argument("--sigma_shift", type=float, default=3.0,
                   help="训练 flow-matching sigma shift (=官方 train.py train_timestep_shift=3.0)")
    p.add_argument("--validation_shift", type=float, default=5.0,
                   help="验证/推理采样器 shift (=官方 train.py validation_timestep_shift=5.0); 与训练 sigma_shift 解耦")
    # (已移除 --logit_mean/--logit_std: sigma 改用 Wan 同款 rand→shift, 见 train_one_step)
    # optim
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--betas", type=str, default="0.9,0.999")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                   help="muon=与 minWM 一致(2D Newton-Schulz + 1D AdamW backup); adamw=普通")
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps", type=int, default=20)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # 序列并行(CP/Ulysses): 与 Wan 脚本同名 --cp_size。1=关(纯DP); >1 须整除 world 且整除 latent token 数。
    p.add_argument("--cp_size", type=int, default=1,
                   help="CP(序列并行)组大小, Ulysses; 1=关; 同组同样本协同算一条序列, DP 维=world//cp_size")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--max_train_steps", type=int, default=20000)
    # io
    p.add_argument("--output_dir", type=str, default="./logs/hy15/stage1")
    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # validation (官方对齐: 采样器 shift=validation_shift=5.0, cfg=6.0, 50 步)
    p.add_argument("--validation_interval", type=int, default=50, help="每 N 步验证一次; 0=关")
    p.add_argument("--val_same_input", action="store_true", default=False,
                   help="通信自检: 所有rank用相同动作+caption+seed; 输出仍不同才是通信/FSDP问题")
    p.add_argument("--first_validation_step", type=int, default=50,
                   help="首次验证的最早步; 须 <= validation_interval, 否则首个 %%interval==0 的步会被跳过(如51会跳过step50→首验落到100)")
    p.add_argument("--validate_before_training", action="store_true", default=False,
                   help="训练前(step 0)先跑一次验证, 在权重被训练污染前确认 base 权重+验证路径(MLLM camtext 编码/采样器/VAE 解码)整体正确")
    p.add_argument("--diffusion_sampling_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=6.0,
                   help="验证 CFG 引导强度(=minWM); >1 启用 cond+uncond 双前向, 1.0=不做 CFG")
    p.add_argument("--fps", type=float, default=24.0,
                   help="验证视频保存 fps; HY15 官方=24 (77帧→3.2s; minWM hy15_inference 的 8 是慢放)")
    return p.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main(parse_args())
