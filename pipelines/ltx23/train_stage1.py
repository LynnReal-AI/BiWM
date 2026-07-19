# BiWM LTX-Video 2.3 Stage 1 training entry.
# =============================================================================
# LTX-Video 2.3 (LTX2.3) stage1 训练入口 —— wan 风格单入口 (HY15 的结构孪生)
# =============================================================================
# 本文件是 fastvideo/hy15_model.py 的【结构孪生】: main() 循环 / Muon 优化器 / FSDP /
# checkpoint 保存格式 / 分布式初始化 / 数据集 loader / argparse / 验证节奏
# 全部与 HY15 保持一致, 只在三处不同:
#   1) 模型加载: LTX23Model (ltx23.modules.model_ltx_2_3), 从 .safetensors metadata['config'] 构建;
#      改用【正确】的 LTX23Model —— 它内置【逐帧 cam-text cross-attn】(同 Wan2.2-5B):
#      precompute_cam_text_embeddings 预编码 81 类相机文本存 _cam_text_raw, _forward 按 action_labels[b,t]
#      逐 latent 帧 gather cam-text + caption 拼 per_frame_context 做 cross-attn。旧的 LTX2Model 无此能力。
#   2) 训练目标接线: LTX flow-matching (velocity = noise - latents), 走 LTX forward 接口;
#   3) i2v 条件机制: 不是 concat 通道, 而是 forward 传 cond_latent_frames=N —— 模型内部把前 N
#      个 latent 帧的 timestep 置 σ=0(clean), 见 model_ltx_2_3.py:998-1001。
#
# 相机控制: 【逐帧离散 cam-text via action_labels】(与 Wan2.2-5B 一致, 不再拼进 prompt):
#   make_live_batch 用 caption-ONLY 当 prompt, 相机由逐 latent 帧 action_labels 经模型内置
#   precompute_cam_text_embeddings 注入 cross-attn。不用 plucker/连续相机; cam-text 路径仅由
#   precompute(置 _use_cam_text_cross_attn + _cam_text_raw) + 传入 action_labels 触发,
#   不依赖 enable_camera_control (见 _forward 的 _use_cam_text guard, model_ltx_2_3.py:1128-1130)。
#
# 文本编码器: Gemma-3 (AVGemmaTextEncoderModel + GemmaFeaturesExtractorProjLinear +
#   Embeddings1DConnector + LTXVGemmaTokenizer + Gemma3ForConditionalGeneration)。
#   一句 prompt → text_encoder(prompt).video_encoding [1, L, cross_attention_dim] (22B=4096)。
#   LTX 的 model.forward 只吃 context(embedding list), 没有 attention mask。
#
# 权重: --pretrained_model_path 指向 LTX-2.3 的 .safetensors 文件(含 transformer/vae/text 三部分);
#   transformer 配置从该文件 metadata['config']['transformer'] 读, 构建逻辑搬自 utils/ltx_wrapper.py
#   (self-contained, 不 import ltx_wrapper)。
#
# ★ precompute 接线: Gemma 文本编码器(load_text_encoder)加载后、FSDP wrap 之前, 在【未 wrap 的】model 上
#   调用一次 model.precompute_cam_text_embeddings(encode_fn, device, dtype); encode_fn 用 Gemma tokenizer
#   求每句真实 token 长度 S_i, 再 te(t).video_encoding[0][:S_i] → List[Tensor[S_i, text_dim]]。
#   存下的 _cam_text_raw 是 model 的普通 list 属性(非 nn 参数), FSDP wrap 后仍存活(与参考实现一致)。
#
# ⚠️ 本文件按"先写对、后跑通"的约定写; 真权重/数据齐了再在 DLC 上冒烟调。
# =============================================================================
import argparse
import inspect
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

from ltx23.modules.model_ltx_2_3 import LTX23Model, LTX23AttentionBlock
from pipelines.dataset.biwm_camera_text_dataset import BiwmCamCaptionData

# LTX-2.3 VAE 几何 (见 utils/ltx_wrapper.py LTXVAEWrapper):
#   空间 32x / 时间 8x / latent 128ch (无独立 scaling_factor: per_channel_statistics 内置进 VAE)
LTX_VAE_LATENT_CHANNELS = 128
LTX_VAE_SPATIAL_STRIDE = 32
LTX_VAE_TEMPORAL_STRIDE = 8

# 81 类离散相机动作 → 相机文本 (action_label = trans*9 + rot), 仅供验证 txt 标签人读展示。
# 相机注入【不再走 prompt】—— 改用 LTX23Model 内置的逐帧 cam-text cross-attn:
#   model.precompute_cam_text_embeddings 用 wan.modules.model.ACTION_TEXT_TABLE(81 句) 预编码存 _cam_text_raw,
#   _forward 按逐 latent 帧 action_labels[b,t] gather cam-text + caption 拼 per_frame_context 做 cross-attn。
#   故此处只保留 action_to_camtext 用于验证元信息显示, 不再拼进 prompt(build_camtext_prompt 已删)。
_TRANS_TEXT = ["Camera does not move.", "Camera moves forward.", "Camera moves backward.",
               "Camera moves right.", "Camera moves left.", "Camera moves forward-right.",
               "Camera moves forward-left.", "Camera moves backward-right.", "Camera moves backward-left."]
_ROT_TEXT = ["Camera does not rotate.", "Camera pitches up.", "Camera pitches down.",
             "Camera yaws right.", "Camera yaws left.", "Camera pitches up and yaws right.",
             "Camera pitches up and yaws left.", "Camera pitches down and yaws right.",
             "Camera pitches down and yaws left."]


def action_to_camtext(a: int) -> str:
    """仅供验证 txt 人读展示 (相机注入实际走 action_labels + 模型内置 ACTION_TEXT_TABLE, 非此句)。"""
    a = int(a)
    return f"{_TRANS_TEXT[a // 9]} {_ROT_TEXT[a % 9]}"


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
# 模型构建 + 权重加载 (LTX23Model, 搬自 utils/ltx_wrapper.py, self-contained)
# =============================================================================
def convert_checkpoint_state_dict(state_dict: dict, model_keys: set) -> dict:
    """LTX-2 ckpt key → 当前模型 key (搬自 utils/ltx_wrapper.convert_checkpoint_state_dict)。
    跳过 audio 权重; 处理 'model.diffusion_model.' 前缀 + transformer_blocks→blocks; 否则直用。"""
    converted = {}
    for ckpt_key, value in state_dict.items():
        if any(x in ckpt_key for x in ['audio_', 'av_ca_', '_a2v_', '_v2a_']):
            continue
        if ckpt_key.startswith('model.diffusion_model.'):
            model_key = ckpt_key[len('model.diffusion_model.'):]
            model_key = model_key.replace('transformer_blocks.', 'blocks.')
        else:
            model_key = ckpt_key
        if model_key in model_keys:
            converted[model_key] = value
    num_total = len(model_keys)
    num_loaded = len(converted)
    ratio = (num_loaded / num_total * 100.0) if num_total else 0.0
    mprint(f"[LTX23] Loaded {num_loaded}/{num_total} weights ({ratio:.2f}%)")
    return converted


def _read_safetensors_config(ckpt_path, key='transformer'):
    """读 transformer 配置 dict, 兼容两种来源:
    1) LTX base 单文件 *.safetensors: 配置在 metadata['config']['transformer'] (LTX-2.3 把全配置塞进 metadata)。
    2) 本项目 checkpoint 目录: 顶层 config.json (save_checkpoint 写的 model._biwm_config, 即过滤后的 transformer 配置本身)。
       加目录分支, 使 stage1 产物可直接作 --pretrained_model_path 续训 (stage2/resume 需要)。"""
    import safetensors
    cfg = {}
    if not ckpt_path or not os.path.exists(ckpt_path):
        return cfg
    if os.path.isdir(ckpt_path):
        cfg_json = os.path.join(ckpt_path, "config.json")
        if os.path.isfile(cfg_json):
            with open(cfg_json) as f:
                cfg = json.load(f)            # 已是 transformer 配置本身(非嵌套), 直接返回
        return cfg
    if ckpt_path.endswith('.safetensors'):
        with safetensors.safe_open(ckpt_path, framework='pt') as f:
            metadata = f.metadata() or {}
            full = json.loads(metadata.get('config', '{}'))
            cfg = full.get(key, {}) if key else full
    return cfg


def _resolve_weight_file(ckpt_path):
    """返回真正的权重文件路径: 目录→其下 diffusion_pytorch_model.safetensors(或 model.pt); 文件→自身。"""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    if os.path.isdir(ckpt_path):
        cand = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")
        if os.path.isfile(cand):
            return cand
        cand_pt = os.path.join(ckpt_path, "model.pt")
        return cand_pt if os.path.isfile(cand_pt) else None
    return ckpt_path


def build_ltx23_transformer(args, device, dtype=torch.bfloat16):
    """构建 LTX23Model (搬自 utils/ltx_wrapper.LTXDiffusionWrapper._load_model)。
    配置从 ckpt(.safetensors) metadata['config']['transformer'] 读, 映射 attention_type/rope_type 枚举,
    过滤到 LTX23Model.__init__ 接受的键 → 构建 → convert_checkpoint_state_dict + load_state_dict(strict=False)。
    LTX23Model 内置【逐帧 cam-text cross-attn】, 相机经 action_labels + precompute 注入,
      不启用 plucker/连续相机; cam-text 路径仅由 precompute(_use_cam_text_cross_attn + _cam_text_raw) +
      传入 action_labels 触发, 不依赖 enable_camera_control(其值随 ckpt config, 默认 False 即可)。"""
    from ltx23.modules.attention import AttentionFunction
    from ltx23.modules.rope import LTXRopeType
    import safetensors.torch

    ckpt = args.pretrained_model_path
    raw_cfg = _read_safetensors_config(ckpt, key='transformer')
    # ★ 保存【原始字符串配置】供 save_checkpoint 写 config.json (枚举值不可 json 序列化, 故存原始 dict)。
    orig_cfg = dict(raw_cfg)

    # attention_type: 优先 flash_attn(无则 DEFAULT); rope_type: LTXRopeType.SPLIT (与 apply_rotary_emb 一致)
    try:
        from flash_attn import flash_attn_func as _fa  # noqa: F401
        _has_flash = True
    except ImportError:
        _has_flash = False
    attention_map = {
        'flash_attention_3': AttentionFunction.FLASH_ATTENTION_3 if _has_flash else AttentionFunction.DEFAULT,
        'xformers': AttentionFunction.XFORMERS,
        'pytorch': AttentionFunction.PYTORCH,
        'default': AttentionFunction.DEFAULT,
    }
    cfg = dict(raw_cfg)
    cfg['attention_type'] = attention_map.get(cfg.get('attention_type', 'default'), AttentionFunction.DEFAULT)
    rope_map = {'split': LTXRopeType.SPLIT, 'interleaved': LTXRopeType.INTERLEAVED}
    if 'rope_type' in cfg:
        cfg['rope_type'] = rope_map.get(cfg.get('rope_type', 'split'), LTXRopeType.SPLIT)

    # 过滤到 LTX23Model.__init__ 接受的键
    valid_params = set(inspect.signature(LTX23Model.__init__).parameters.keys())
    filtered_config = {k: v for k, v in cfg.items() if k in valid_params}
    mprint(f"[LTX23] 构建 LTX23Model: {len(filtered_config)} 个配置键; "
           f"cross_attention_dim={filtered_config.get('cross_attention_dim')}, "
           f"num_layers={filtered_config.get('num_layers')}, "
           f"enable_camera_control={filtered_config.get('enable_camera_control', False)}")
    if args.camera_mode == "prope":
        raise NotImplementedError("[LTX23] camera_mode=prope 不支持; LTX2.3 stage1 仅支持逐帧 cam-text(action_labels 控相机)")
    mprint("[LTX23] camera_mode=camtext (相机走逐帧 cam-text via action_labels; 模型内置 cross-attn; 全参训练)")

    model = LTX23Model(**filtered_config)
    # 存【原始字符串配置】(只取 json 可序列化的), save_checkpoint 据此写 config.json,
    #   使 checkpoint 可作 --pretrained_model_path 续训/推理 (与 HY15 一致)。
    model._biwm_config = {k: v for k, v in orig_cfg.items()
                          if isinstance(v, (str, int, float, bool, list, dict, type(None)))}

    model_keys = set(name for name, _ in model.named_parameters())
    model_keys.update(name for name, _ in model.named_buffers())

    # 加载真权重 (base 单 .safetensors 或 checkpoint 目录下 diffusion_pytorch_model.safetensors;
    #   audio key 被 convert 跳过, 单流缺失走 strict=False)
    wfile = _resolve_weight_file(ckpt)
    if wfile and os.path.exists(wfile):
        if wfile.endswith('.safetensors'):
            state_dict = safetensors.torch.load_file(wfile, device='cpu')
        else:
            state_dict = torch.load(wfile, map_location='cpu')
            state_dict = state_dict.get('state_dict', state_dict)
        converted = convert_checkpoint_state_dict(state_dict, model_keys)
        del state_dict
        if converted:
            missing, unexpected = model.load_state_dict(converted, strict=False)
            mprint(f"[LTX23] 权重加载: missing={len(missing)} unexpected={len(unexpected)}")
        del converted
        import gc; gc.collect()
    else:
        mprint(f"[LTX23] 无 transformer 权重({ckpt}), 随机初始化。")

    mprint(f"[LTX23] LTX23Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    model = model.to(device=device, dtype=dtype)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        mprint("[LTX23] gradient checkpointing ON")
    return model


def wrap_fsdp(model, device, args):
    import functools
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        MixedPrecision, ShardingStrategy,
                                        BackwardPrefetch)
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={LTX23AttentionBlock})
    mprint("[LTX23] FSDP FULL_SHARD wrap on LTX23AttentionBlock")
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
# i2v 条件 (LTX: cond_latent_frames, 非 concat 通道) + 逐帧 loss mask
# =============================================================================
def resolve_sample_task(args):
    """每条样本解析具体 task: mixed → 以 i2v_rate 概率为 i2v, 否则 t2v; 非 mixed → 恒为 training_mode。
    支持 i2v/t2v 混合训练 (--training_mode mixed --i2v_rate 0.8)。"""
    if args.training_mode == "mixed":
        return "i2v" if random.random() < float(getattr(args, "i2v_rate", 0.0)) else "t2v"
    return args.training_mode


def resolve_cond_n(task, args):
    """该样本的条件帧数 (latent 空间): i2v → args.i2v_cond_latent_frames(默认 1); t2v → 0。
    LTX 的 i2v 不是 concat 通道, 而是把前 cond_n 个 latent 帧的 timestep 置 σ=0 (forward 传 cond_latent_frames)。"""
    if task == "i2v":
        return int(getattr(args, "i2v_cond_latent_frames", 1))
    return 0


def frame_loss_mask(B, C, T, H, W, cond_n, device, dtype):
    """逐帧 loss mask: 前 cond_n 个 latent 帧(条件/clean)置 0, 其余置 1。
    i2v 不在 clean 条件帧上计 loss(它们被强制等于 GT, 无监督意义); t2v cond_n=0 → 全 1。"""
    m = torch.ones(B, C, T, H, W, device=device, dtype=dtype)
    if cond_n > 0:
        m[:, :, :cond_n, :, :] = 0.0
    return m


# =============================================================================
# live 实时编码 (LTX): LTX-2 VAE(128ch/8x时间/32x空间) + Gemma-3 文本编码器
# =============================================================================
class _LTXVAE:
    """LTX-2 VAE 封装 (搬自 utils/ltx_wrapper.LTXVAEWrapper, self-contained)。
    encode: pixel [B,F,3,H,W] → latent [B,128,F',H',W'] (F'=(F-1)//8+1, H'=H//32, W'=W//32);
    decode: latent [B,128,F',H',W'] → pixel [B,3,F,H,W]。per_channel_statistics 已内置进 VAE 权重,
    故无独立 scaling_factor(直接 encode/decode, 不乘 _sf)。"""

    def __init__(self, encoder, decoder, dtype):
        self.encoder = encoder
        self.decoder = decoder
        self.dtype = dtype

    def parameters(self):
        return self.encoder.parameters()


def load_ltx_vae(ckpt_path, device, dtype=torch.bfloat16):
    """加载 LTX-2 VAE encoder+decoder (从 .safetensors 的 vae.* 权重; 搬自 LTXVAEWrapper.__init__)。"""
    from ltx23.modules.vae import create_video_encoder, create_video_decoder
    import safetensors
    config = _read_safetensors_config(ckpt_path, key=None)  # 整个 config(VAE 构建吃完整 config)
    encoder = create_video_encoder(config)
    decoder = create_video_decoder(config)
    if ckpt_path and os.path.exists(ckpt_path):
        import gc
        with safetensors.safe_open(ckpt_path, framework='pt', device='cpu') as f:
            all_keys = f.keys()
            enc_sd, dec_sd = {}, {}
            for k in all_keys:
                if k.startswith('vae.encoder.'):
                    enc_sd[k.replace('vae.encoder.', '')] = f.get_tensor(k)
                elif k.startswith('vae.decoder.'):
                    dec_sd[k.replace('vae.decoder.', '')] = f.get_tensor(k)
                elif k.startswith('vae.per_channel_statistics.'):
                    _kk = k.replace('vae.', '')
                    enc_sd[_kk] = f.get_tensor(k)
                    dec_sd[_kk] = f.get_tensor(k)
            if enc_sd:
                encoder.load_state_dict(enc_sd, strict=False)
            if dec_sd:
                decoder.load_state_dict(dec_sd, strict=False)
            del enc_sd, dec_sd
        gc.collect()
    encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    decoder = decoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    mprint(f"[LTX23-live] VAE 就绪 (128ch/8x时间/32x空间) {ckpt_path}")
    return _LTXVAE(encoder, decoder, dtype)


@torch.no_grad()
def ltx_vae_encode(vae, pixel):
    """pixel [B,F,3,H,W] ∈[-1,1] → latent [B,128,F',H',W'] (内部 permute 到 [B,3,F,H,W] 喂 encoder)。"""
    if pixel.dim() == 5 and pixel.shape[2] == 3:
        pixel = pixel.permute(0, 2, 1, 3, 4)  # [B,3,F,H,W]
    return vae.encoder(pixel.to(vae.dtype))    # [B,128,F',H',W']


@torch.no_grad()
def ltx_vae_decode(vae, latent):
    """latent [B,128,F',H',W'] → pixel [B,3,F,H,W]。"""
    if latent.dim() == 5 and latent.shape[1] != 3 and latent.shape[2] == 3:
        latent = latent.permute(0, 2, 1, 3, 4)
    return vae.decoder(latent.to(vae.dtype))   # [B,3,F,H,W]


def load_text_encoder(ckpt_path, gemma_path, device, dtype=torch.bfloat16):
    """加载 LTX-2 Gemma-3 文本编码器 (搬自 utils/ltx_wrapper.LTXTextEncoderWrapper.__init__, self-contained)。
    一句 prompt → text_encoder(prompt).video_encoding [1, L, cross_attention_dim] (22B=4096)。"""
    from ltx23.modules.text_encoder import (
        AVGemmaTextEncoderModel, LTXVGemmaTokenizer,
        GemmaFeaturesExtractorProjLinear, Embeddings1DConnector,
    )
    from ltx23.modules.rope import LTXRopeType
    from transformers import Gemma3ForConditionalGeneration
    import safetensors

    tf_cfg = _read_safetensors_config(ckpt_path, key='transformer')
    rope_type = LTXRopeType(tf_cfg.get('rope_type', 'split'))
    pe_max_pos = tf_cfg.get('connector_positional_embedding_max_pos', [1])
    _vid_heads = int(tf_cfg.get('connector_num_attention_heads', 32))
    _vid_head_dim = int(tf_cfg.get('connector_attention_head_dim', 128))
    _num_layers = int(tf_cfg.get('connector_num_layers', 8))
    _num_registers = int(tf_cfg.get('connector_num_learnable_registers', 128))
    _cross_dim = int(tf_cfg.get('cross_attention_dim', 4096))
    _use_video_key = _cross_dim > 3840  # 22B(4096) uses video_aggregate_embed + bias
    feature_extractor = GemmaFeaturesExtractorProjLinear(
        out_dim=_cross_dim, bias=_use_video_key, use_video_key=_use_video_key)
    embeddings_connector = Embeddings1DConnector(
        num_attention_heads=_vid_heads, attention_head_dim=_vid_head_dim,
        positional_embedding_max_pos=pe_max_pos, rope_type=rope_type)
    audio_connector = None  # video-only

    tokenizer = LTXVGemmaTokenizer(gemma_path)
    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    te = AVGemmaTextEncoderModel(
        feature_extractor, embeddings_connector, audio_connector,
        tokenizer=tokenizer, model=gemma_model, dtype=dtype,
        use_v2_norm=_use_video_key, gemma_embedding_dim=3840)

    # 从 ckpt 选择性加载 text_embedding_projection.* + video_embeddings_connector
    if ckpt_path and os.path.exists(ckpt_path):
        import gc
        with safetensors.safe_open(ckpt_path, framework='pt', device='cpu') as f:
            all_keys = f.keys()
            fe_keys = [k for k in all_keys if k.startswith('text_embedding_projection.')]
            if fe_keys:
                fe_w = {k.replace('text_embedding_projection.', ''): f.get_tensor(k) for k in fe_keys}
                te.feature_extractor_linear.load_state_dict(fe_w, strict=False)
                del fe_w
            ec_keys = [k for k in all_keys if 'video_embeddings_connector' in k]
            if ec_keys:
                ec_w = {k.replace('model.diffusion_model.video_embeddings_connector.', ''): f.get_tensor(k)
                        for k in ec_keys}
                te.embeddings_connector.load_state_dict(ec_w, strict=False)
                del ec_w
        gc.collect()
    te = te.to(device).eval().requires_grad_(False)
    te.dtype = dtype   # AVGemmaTextEncoderModel 无 .dtype 属性, 显式挂上供 ltx_encode 用(否则首个 caption 编码 AttributeError)
    mprint(f"[LTX23-live] 文本编码器(Gemma-3) 就绪 {gemma_path} (cross_dim={_cross_dim})")
    return te


@torch.no_grad()
def ltx_encode(te, prompts, device):
    """prompts:List[str] -> prompt_embeds [B, L, cross_attention_dim]。
    LTX 的 model.forward 只吃 context(embedding 列表), 没有 attention mask (对齐 mllm_encode 但无 mask)。"""
    embs = [te(p).video_encoding for p in prompts]   # 每个 [1, L, C]
    return torch.cat(embs, dim=0).to(device=device, dtype=te.dtype)


def make_cam_text_encode_fn(te, device, dtype=torch.bfloat16):
    """返回 encode_fn(texts:List[str]) -> List[Tensor[S_i, text_dim]] (与参考编码函数一致)。
    precompute_cam_text_embeddings 内部从 wan.modules.model.ACTION_TEXT_TABLE 取 81 句相机文本传入。
    每句: 用 Gemma tokenizer 求【真实 token 数 S_i】(weight>0), 再 te(t).video_encoding[0][:S_i] 截掉
    padding/learnable registers, 只留真实 token 的 raw text_dim 嵌入(caption_projection 投影在 forward 里做)。"""
    @torch.no_grad()
    def _encode_fn(texts):
        results = []
        for t in texts:
            token_pairs = te.tokenizer.tokenize_with_weights(t)["gemma"]
            attn_mask = torch.tensor([p[1] for p in token_pairs])
            s_i = int(attn_mask.gt(0).sum().item())          # 真实 token 数
            emb = te(t).video_encoding[0]                    # [L, text_dim] (含 registers/pad)
            results.append(emb[:s_i].to(device=device, dtype=dtype))
        return results
    return _encode_fn


_NEG_EMB_CACHE = {}


@torch.no_grad()
def encode_neg_prompt(te, device):
    """CFG 的 uncond/neg = 空串 "" (Gemma 无需特殊模板, 直接编码空串)。结果缓存(空串固定, 只算一次)。
    返回 prompt_embeds [1, L, C] (LTX context 无 attention mask)。"""
    if "neg" in _NEG_EMB_CACHE:
        return _NEG_EMB_CACHE["neg"].to(device)
    pe = ltx_encode(te, [""], device)
    _NEG_EMB_CACHE["neg"] = pe.detach()
    return pe


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
    相机【不再拼进 prompt】—— prompt 恒为 caption-ONLY, 相机由【逐 latent 帧 action_labels】
      经 LTX23Model 内置 cam-text cross-attn 注入 (与 Wan2.2-5B stage1 一致)。CFG: 按 training_cfg_rate
      把 prompt 丢成空串 (相机仍由 action_labels 控制, 与文本解耦)。caption_only 参数现等价于默认行为(保留兼容)。
    LTX: VAE 128ch/8x时间/32x空间; image_cond = 首帧干净 latent (i2v 由 cond_latent_frames 机制使用, 非 concat)。"""
    pixel_values = item["pixel_values"]                       # [T,C,H,W] ∈[-1,1]
    caption = item["caption"]
    action_labels = item["action_labels"]                     # [T_lat] discrete camera actions
    # LTX VAE 编码: [T,C,H,W] → pixel [1,F,3,H,W] → latent [1,128,F',h,w]
    x = pixel_values.unsqueeze(0).to(device, dtype=vae.dtype)   # [1,F,C,H,W]
    latent = ltx_vae_encode(vae, x).to(torch.bfloat16)         # [1,128,T_lat,h,w]
    B, C, T_lat, h, w = latent.shape
    # 该 clip 的相机动作(常量, 仅供验证显示): 取非零众数(video_real 整段同一动作; 首帧 init=0)
    al_flat = torch.as_tensor(action_labels).reshape(-1).long()
    _nz = al_flat[al_flat > 0]
    clip_action = int(torch.bincount(_nz).argmax()) if _nz.numel() > 0 else 0
    # ★ 逐帧 action_labels [1, T_lat] —— 对齐到当前 latent 帧数 (长度不符则 nearest 重采样, 保离散类别;
    #   镜像 Wan stage1 wan_model_2_2.py:1713-1727)。喂 model.forward 的 action_labels=, 逐帧控相机。
    al = al_flat.unsqueeze(0)                                # [1, L]
    if al.shape[1] != T_lat:
        al = F.interpolate(al.float().unsqueeze(1), size=T_lat, mode='nearest').squeeze(1).long()
    al = al.to(device=device, dtype=torch.long)             # [1, T_lat]
    # 文本 → Gemma (caption-ONLY, 不含相机); CFG dropout 丢成空串。
    full_text = caption.rstrip() if caption else ""
    if random.random() < args.training_cfg_rate:
        full_text = ""                                       # 无条件(CFG 训练; 相机仍由 action_labels 控)
    pe = ltx_encode(te, [full_text], device)                 # [1,L,C] (LTX context 无 mask)
    # ★ image_cond 恒取该 clip 首帧【干净 latent】(便宜, latent 已编码)。
    #   是否真正使用由 task 决定: i2v 时 train_one_step 把 noisy 首帧换回 clean + 传 cond_latent_frames; t2v 不用。
    #   mixed 训练里同一条数据可被随机当 i2v 或 t2v 用, 无需重编码。
    img_cond = latent[:, :, :1, :, :].clone()               # [1,C,1,h,w] 首帧干净 latent
    # 每条样本的 task: mixed 模式按 i2v_rate 概率掷骰 (i2v else t2v); 否则恒为 training_mode。
    task_this = resolve_sample_task(args)
    return {
        "latent": latent, "prompt_embed": pe,
        "image_cond": img_cond,
        "action_labels": al,                               # ★ 逐帧相机 [1, T_lat]; train_one_step/forward 用
        "task": task_this,                                 # ★ 本条样本的 i2v/t2v (mixed 掷骰结果); train_one_step 据此用 cond_latent_frames
        "caption": caption, "clip_action": clip_action,   # 验证 per-rank 重编 cam-text 用
    }


# =============================================================================
# 单步训练 (flow matching, LTX forward; i2v 走 cond_latent_frames)
# =============================================================================
def train_one_step(model, batch, args, device, step):
    latents = batch["latent"].to(device, dtype=torch.bfloat16)         # (B,C,T,H,W)
    prompt_embed = batch["prompt_embed"].to(device, dtype=torch.bfloat16)
    image_cond = batch["image_cond"].to(device, dtype=torch.bfloat16)  # 首帧干净 latent [B,C,1,h,w]
    # ★ 逐帧相机 action_labels [B, T_lat] —— 始终传 (t2v/i2v 都按逐帧 cam-text 控相机; t2v 仅无 clean 首帧)。
    action_labels = batch.get("action_labels", None)
    if action_labels is not None:
        action_labels = action_labels.to(device=device, dtype=torch.long)

    B, C, T_lat, H, W = latents.shape
    # task 来自该条样本 make_live_batch 的掷骰结果 (mixed 模式 i2v/t2v 混合); 兜底回退 training_mode。
    task = batch.get("task", args.training_mode)
    cond_n = resolve_cond_n(task, args)                     # i2v: i2v_cond_latent_frames(默认1); t2v: 0
    cond_n = min(cond_n, T_lat)

    # --- flow-matching: sigma 采样对齐 Wan stage1 —— u~U[0,1] 再 shift (与 HY15 一致) ---
    noise = torch.randn_like(latents)
    u = torch.rand(B, device=device)
    sh = args.sigma_shift
    sigma = sh * u / (1.0 + (sh - 1.0) * u)                 # (B,) in [0,1]
    sig_b = sigma.view(B, 1, 1, 1, 1).to(latents.dtype)
    noisy = (1.0 - sig_b) * latents + sig_b * noise        # 全帧加噪
    target = noise - latents                                # flow-matching velocity 目标

    # --- i2v: 前 cond_n 个 latent 帧换回【干净 latent】, 并传 cond_latent_frames=cond_n 给 forward ---
    #   LTX i2v 不是 concat 通道: 模型内部把前 cond_n 帧的 timestep 置 σ=0 (见 model.py:903-910)。
    #   故输入这些帧必须是 clean latent(否则与 σ=0 不自洽)。t2v: cond_n=0, 不改 noisy。
    if cond_n > 0:
        noisy = noisy.clone()
        noisy[:, :, :cond_n] = latents[:, :, :cond_n]      # 用 GT clean latent (首帧来自 image_cond 同源)
        if image_cond is not None and image_cond.shape[2] >= 1:
            noisy[:, :, :1] = image_cond[:, :, :1].to(noisy.dtype)

    # --- LTX forward: x_list=[noisy_i [C,F,H,W]], ctx_list=[prompt_embed_i [L,C]],
    #     seq_len=T*H*W (patch_size=1), t=per-BATCH sigma[B] (≤1, 模型直接接受); 返回 [B,C,F,H,W]。
    #     ★ action_labels=[B,T_lat] 逐帧相机 —— LTX23Model 内部按 action_labels[b,t] gather cam-text+caption
    #       拼 per_frame_context 做 cross-attn (model_ltx_2_3.py:1123-1198); t2v/i2v 都传(逐帧相机为监督信号)。 ---
    x_list = [noisy[i] for i in range(B)]
    ctx_list = [prompt_embed[i] for i in range(B)]
    seq_len = T_lat * H * W
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        pred = model(
            x_list,
            t=sigma.to(torch.bfloat16),                    # per-batch sigma[B] in [0,1]
            context=ctx_list,
            seq_len=seq_len,
            action_labels=action_labels,                   # ★ 逐帧 cam-text 控相机
            cond_latent_frames=cond_n,
            fps=float(getattr(args, "fps", 24.0)),
        )                                                  # [B,C,F,H,W]
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
        # 逐帧 loss mask: 前 cond_n 个条件帧(clean, 被强制等 GT)不计 loss; 其余计 (t2v cond_n=0 → 全帧)
        lmask = frame_loss_mask(B, C, T_lat, H, W, cond_n, device, torch.float32)
        diff = (pred.float() * lmask - target.float() * lmask) ** 2
        loss = diff.sum() / torch.clamp(lmask.sum(), min=1.0)
    return loss


# =============================================================================
# checkpoint
# =============================================================================
def save_checkpoint(model, out_dir, step):
    """保存 checkpoint —— 格式对齐 HY15/Wan(pipelines/utils/checkpoint.py:save_checkpoint):
    FSDP FULL_STATE_DICT(offload_to_cpu, rank0_only) → 存
      checkpoint-{step}/diffusion_pytorch_model.safetensors  (safetensors)
      checkpoint-{step}/config.json                           (构建配置, model._biwm_config, 已只取 json 可序列化)
    ✓ 本 ckpt 目录(顶层 config.json + diffusion_pytorch_model.safetensors)可直接作
       --pretrained_model_path 续训/推理 —— build_ltx23_transformer 的 _read_safetensors_config/
       _resolve_weight_file 已支持【目录】与【base 单 .safetensors】两种来源。"""
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                        FullStateDictConfig, StateDictType)
    import safetensors.torch as st
    fcfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fcfg):
        sd = model.state_dict()
    if is_main():
        d = os.path.join(out_dir, f"checkpoint-{step}")
        os.makedirs(d, exist_ok=True)
        # 权重: safetensors; 连续化避免非连续存储报错
        st.save_file({k: v.contiguous() for k, v in sd.items()},
                     os.path.join(d, "diffusion_pytorch_model.safetensors"))
        # config.json: 用构建时 cfg(LTX 原始字符串配置, 已过滤为 json 可序列化); FSDP/checkpoint wrap 下逐层解包取属性
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
                mprint(f"[LTX23] config.json 保存失败(跳过): {type(e).__name__}: {e}")
        mprint(f"[LTX23] saved checkpoint (safetensors+config) -> {d}")


# =============================================================================
# 验证: 从噪声采样去噪 → VAE 解码 → 存 mp4 (t2v, 固定 val 样本)
# =============================================================================
# 验证用的代表性相机动作 (rank → 不同动作, 各 rank 独立采样不同运镜; 参考 wan 每 rank 独立)
#   9=forward 18=backward 27=right 36=left 1=pitch_up 2=pitch_down 3=yaw_right 4=yaw_left
_VAL_ACTIONS = [9, 18, 27, 36, 1, 2, 3, 4]


@torch.no_grad()
def run_validation(model, vae, mllm, val_batch, args, step, device, rank):
    """每 rank 独立采样一个不同的相机动作(camtext: 重编 caption+cam-text), 从噪声去噪 → LTX VAE 解码
    → 叠摇杆(joystick) overlay → 各 rank 存各自 mp4。采样器: LTX flow-matching euler (手写, x0=xt-σ*flow)。"""
    import numpy as np
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
    # 逐帧相机 action_labels [1, T_lat] —— 相机走 action_labels(非 prompt)。
    #   优先用 val_batch 的逐帧 action_labels(与 GT clip 真实运镜对齐); 缺失则把 act 展开成常量逐帧。
    _val_al = val_batch.get("action_labels", None)
    if _val_al is not None and torch.is_tensor(_val_al):
        action_labels = _val_al.to(device=device, dtype=torch.long)
        if action_labels.dim() == 1:
            action_labels = action_labels.unsqueeze(0)
        if action_labels.shape[1] != T_lat:
            action_labels = F.interpolate(action_labels.float().unsqueeze(1), size=T_lat,
                                          mode='nearest').squeeze(1).long()
    else:
        action_labels = torch.full((B, T_lat), int(act), dtype=torch.long, device=device)
    # 文本: caption-ONLY (相机不在 prompt 里, 由 action_labels 控)。
    if mllm is not None:
        _cap = ("A medieval castle on a green hill under a clear blue sky" if _same
                else str(val_batch.get("caption", "")))
        full_text = (_cap or "").rstrip()
        pe = ltx_encode(mllm, [full_text], device)              # [1,L,C]
    else:
        pe = val_batch["prompt_embed"].to(device, torch.bfloat16)
        full_text = "<preencoded>"
    # i2v 条件帧数: i2v → i2v_cond_latent_frames(默认1); t2v → 0。首帧 clean latent 来自 val_batch.image_cond。
    cond_n = min(resolve_cond_n(task, args), T_lat)
    img_cond = val_batch["image_cond"].to(device, torch.bfloat16)

    # CFG: uncond/neg = 空串 "" (Gemma 直接编码空串, encode_neg_prompt 缓存)。guidance<=1 不做 CFG。
    guidance = float(getattr(args, "cfg_scale", 1.0))
    pe_u = None
    if guidance > 1.0 and mllm is not None:
        from pipelines.common.dmd_algo import run_cfg as apply_cfg
        pe_u = encode_neg_prompt(mllm, device)                  # neg = 空串 Gemma 编码

    g = torch.Generator(device=device).manual_seed(42 + _vid)  # 同CP组同噪声; val_same_input 时所有rank同噪声(=42)
    x = torch.randn(latent.shape, generator=g, device=device, dtype=torch.bfloat16)
    # i2v: 首帧固定为 clean cond latent (与训练一致)
    if cond_n > 0 and img_cond is not None and img_cond.shape[2] >= 1:
        x[:, :, :1] = img_cond[:, :, :1].to(x.dtype)

    # --- LTX flow-matching euler 采样 (手写): sigma 1→0 (validation_shift), x0 = xt - σ*flow ---
    #   sigma schedule: u∈linspace(1,0,steps+1), shift 同训练(sh*u/(1+(sh-1)u))。
    n_steps = int(args.diffusion_sampling_steps)
    sh = float(getattr(args, "validation_shift", 5.0))
    u_grid = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    sigmas = (sh * u_grid / (1.0 + (sh - 1.0) * u_grid))         # [steps+1], 1→0

    def _flow(_pe, _x, _sig):
        """LTX forward → flow velocity [B,C,F,H,W]。_sig: per-batch sigma[B] in [0,1]。
        ★ action_labels=[B,T_lat] 逐帧相机, 经 LTX23Model 内置 cam-text cross-attn 注入(同训练)。"""
        x_list = [_x[i] for i in range(B)]
        ctx_list = [_pe[i] for i in range(_pe.shape[0])]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x_list, t=_sig.to(torch.bfloat16), context=ctx_list,
                         seq_len=T_lat * H * W, action_labels=action_labels,
                         cond_latent_frames=cond_n,
                         fps=float(getattr(args, "fps", 24.0)))

    for i in range(n_steps):
        sig = sigmas[i].view(1).expand(B).contiguous()
        sig_next = sigmas[i + 1]
        flow = _flow(pe, x, sig)                                 # 条件
        if pe_u is not None:                                    # CFG: uncond + scale*(cond-uncond)
            flow = apply_cfg(flow, _flow(pe_u, x, sig), guidance)
        # euler: x_{next} = x + (σ_next - σ_cur) * flow  (flow = dx/dσ = noise - x0)
        x = x + (sig_next - sigmas[i]) * flow
        if cond_n > 0 and img_cond is not None and img_cond.shape[2] >= 1:
            x[:, :, :1] = img_cond[:, :, :1].to(x.dtype)       # i2v: 每步保持首帧 clean

    # 解码【生成视频】(无 overlay) + 【GT】(val_batch latent 的 LTX VAE 重建), 用于左右对比
    with torch.autocast("cuda", dtype=torch.bfloat16):
        vid = ltx_vae_decode(vae, x)                            # [B,3,T,H,W]
    frames = _frames_to_uint8(vid)                            # list[np.uint8 HWC]
    gt_frames = None
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gt_vid = ltx_vae_decode(vae, latent)
        gt_frames = _frames_to_uint8(gt_vid)
    except Exception as e:
        mprint(f"[LTX23][val] GT 解码跳过: {type(e).__name__}: {e}")

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
            mf.write(f"Action label(clip): {act}\nCamera text(display): {cam_txt}\n")
            mf.write(f"Prompt(caption-only): {full_text if mllm is not None else '<preencoded>'}\n")
            mf.write(f"CFG Scale: {guidance}\nSampling Steps: {args.diffusion_sampling_steps}\n")
            mf.write(f"Num Frames(latent): {T_lat}\nResolution: {W*LTX_VAE_SPATIAL_STRIDE}x{H*LTX_VAE_SPATIAL_STRIDE}\n")
            mf.write(f"FPS: {fps}\nSource: video_real\nTask Type: {task}\nCamera: per-frame cam-text via action_labels\n")
    except Exception as e:
        mprint(f"[LTX23][val] metadata 写入跳过: {type(e).__name__}: {e}")
    # 3) GT|Gen 左右拼接 + 摇杆 overlay (同 Wan combined_joystick)
    try:
        from pipelines.common.control_overlay import superimpose_control_video as add_joystick_overlay
        al = torch.full((T_lat,), int(act), dtype=torch.long)
        gen_j = add_joystick_overlay(frames, al, vae_temporal_stride=LTX_VAE_TEMPORAL_STRIDE)
        if gt_frames is not None:
            gt_j = add_joystick_overlay(gt_frames, al, vae_temporal_stride=LTX_VAE_TEMPORAL_STRIDE)
            combine_videos_with_labels(gt_j, gen_j, os.path.join(d, f"combined_joystick_{name_prefix}.mp4"),
                                       "GT", "Generated", fps=fps)
        else:                                                # 无 GT 时只存带摇杆的生成视频
            save_video(gen_j, os.path.join(d, f"joystick_{name_prefix}.mp4"), fps=fps)
    except Exception as e:
        mprint(f"[LTX23][val] joystick/combined 跳过: {type(e).__name__}: {e}")
    if is_main():
        mprint(f"[LTX23][val] step {step}: 每(CP组/rank)一个动作; 存 validation_*/combined_joystick_* (act{act}) -> {d}")
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

    # ★ 序列并行(CP, Ulysses) 状态初始化 —— cp_size==1 时为 no-op(纯 DP, 行为同原单卡)。
    #   ⚠️ LTX23Model 用自带 attention(不读 biWM nccl_info 的 SP 分支), 故 cp_size>1 当前不被 LTX 支持;
    #     脚本默认 CP_SIZE=1。保留此调用以与 HY15 结构对齐(验证 CP 组/存图逻辑仍复用)。
    from pipelines.utils.parallel_states import setup_sequence_parallel_state as initialize_sequence_parallel_state, nccl_state as nccl_info
    initialize_sequence_parallel_state(args.cp_size)
    if args.cp_size > 1:
        mprint(f"[LTX23] CP(序列并行) on: cp_size={args.cp_size}, "
               f"DP groups={world // args.cp_size}; 同组同样本, 跨组不同样本")

    model = build_ltx23_transformer(args, device)
    model.train()

    # 文本编码器(Gemma)必须在 FSDP wrap【之前】加载 —— precompute 要在【未 wrap 的】
    #   model 上跑 (precompute 存的 _cam_text_raw 是普通 list 属性, FSDP wrap 后随 model 存活)。
    mllm = load_text_encoder(args.pretrained_model_path if not args.vae_path else args.vae_path,
                             args.gemma_path, device)
    # ★ 逐帧 cam-text cross-attn 预编码 (一次): 用 Gemma 把 81 类 ACTION_TEXT_TABLE 相机文本编码成
    #   _cam_text_raw (raw text_dim, 按真实 token 截断), 并置 _use_cam_text_cross_attn=True。
    #   之后 _forward 收到 action_labels 即按逐 latent 帧 gather cam-text + caption 注入 cross-attn。
    _cam_encode_fn = make_cam_text_encode_fn(mllm, device, torch.bfloat16)
    model.precompute_cam_text_embeddings(_cam_encode_fn, device, torch.bfloat16)
    mprint("[LTX23] 逐帧 cam-text 预编码完成 (_cam_text_raw 已存; FSDP wrap 后随 model 存活)")

    model = wrap_fsdp(model, device, args)

    vae = None
    # live: mp4 → 在线 LTX VAE + Gemma 文本编码 (相机走 cam-text 文本)
    dataset = BiwmCamCaptionData(
        video_dir=args.biwm_video_dir, caption_json=args.biwm_caption_json,
        width=args.num_width, height=args.num_height, num_frames=args.num_frames,
        vae_temporal_factor=LTX_VAE_TEMPORAL_STRIDE)
    # ★ CP 组感知采样 —— DP 维度 = world // cp_size, dp_rank = rank // cp_size。
    #   同一 CP 组(dp_rank 相同)的各 rank 必须拿到**同一条样本**; 不同 CP 组拿不同样本(数据并行)。
    from torch.utils.data import DataLoader
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
    vae = load_ltx_vae(args.pretrained_model_path if not args.vae_path else args.vae_path, device)

    params = [p for p in model.parameters() if p.requires_grad]
    betas = tuple(float(x) for x in args.betas.split(","))
    if args.optimizer == "muon":
        # ★ 与 minWM 对齐: Muon(≥2D 参数 Newton-Schulz 正交化动量, 1D/embed/head 退 AdamW)
        from pipelines.common.muon import build_muon_optimizer
        optimizer = build_muon_optimizer(model, lr=args.learning_rate,
                                       weight_decay=args.weight_decay, adamw_betas=betas)
        mprint("[LTX23] optimizer = Muon (与 minWM 一致; 2D→Newton-Schulz, 1D→AdamW backup)")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate,
                                      weight_decay=args.weight_decay, betas=betas)
        mprint("[LTX23] optimizer = AdamW")
    from diffusers.optimization import get_scheduler
    lr_sched = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                             num_warmup_steps=args.lr_warmup_steps,
                             num_training_steps=args.max_train_steps)

    mprint(f"[LTX23] start training: max_steps={args.max_train_steps}, "
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
            if ((isinstance(raw, dict) and raw.get("skip")) or
                    (isinstance(raw, tuple) and raw and raw[0] == "skip")):
                continue
            return make_live_batch(raw, vae, mllm, args, device), it

    val_batch = None    # 固定验证样本(取第一条 batch)

    from collections import deque
    step_times = deque(maxlen=50)
    step, loader_iter = 0, iter(loader)
    # ★ 训练前验证(step 0) —— 在权重被训练污染前先跑一遍 run_validation,
    #   确认 base 权重 + 验证路径(Gemma camtext 编码 / LTX euler 采样器 / VAE 解码 / joystick overlay)整体正确。
    if (getattr(args, "validate_before_training", False)
            and args.validation_interval > 0 and vae is not None):
        _vb, loader_iter = _fetch(loader_iter)
        val_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in _vb.items()}
        mprint("[LTX23][val] === 训练前验证 (step 0, base 权重) ===")
        try:
            run_validation(model, vae, mllm, val_batch, args, 0, device, rank)
        except Exception as e:
            mprint(f"[LTX23][val] step 0 训练前验证失败(跳过): {type(e).__name__}: {e}")
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
                mprint(f"[LTX23][val] step {step} 验证失败(跳过): {type(e).__name__}: {e}")
            model.train()
        if step % args.checkpointing_steps == 0:
            save_checkpoint(model, args.output_dir, step)

    save_checkpoint(model, args.output_dir, args.max_train_steps)
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser("LTX-Video 2.3 stage1 (cam-text, native LTX forward)")
    # data
    p.add_argument("--data_mode", choices=["live"], default="live",
                   help="live=mp4在线 LTX VAE + Gemma 编码 (LTX23 仅支持 live)")
    p.add_argument("--biwm_video_dir", type=str, default="",
                   help="live: 视频目录 (dataset/videos 或 dataset/video_real)")
    p.add_argument("--biwm_caption_json", type=str, default="",
                   help="live: caption/pose json (preencode_input.json / video_real_input.json)")
    p.add_argument("--text_encoder_path", type=str, default="",
                   help="(LTX 未使用; Gemma 路径见 --gemma_path)")
    p.add_argument("--gemma_path", type=str, default="",
                   help="Gemma-3 目录 (live 编码必需, e.g. .../google/gemma-3-12b-it-qat-q4_0-unquantized)")
    p.add_argument("--num_frames", type=int, default=77)
    p.add_argument("--num_height", type=int, default=480)
    p.add_argument("--num_width", type=int, default=832)
    p.add_argument("--training_mode", choices=["t2v", "i2v", "mixed"], default="t2v",
                   help="t2v/i2v 固定; mixed=按 --i2v_rate 概率逐样本混合 i2v/t2v (验证用 i2v 展示)")
    p.add_argument("--i2v_rate", type=float, default=0.0)
    p.add_argument("--i2v_cond_latent_frames", type=int, default=1,
                   help="i2v 条件帧数 (latent 空间): 这些帧 timestep 置 σ=0 (cond_latent_frames), 不计 loss")
    p.add_argument("--training_cfg_rate", type=float, default=0.0)  # 默认不做训练期文本丢弃
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    # model / weights
    p.add_argument("--pretrained_model_path", type=str, default="",
                   help="LTX-2.3 .safetensors 文件 (含 transformer/vae/text 三部分, 配置在 metadata['config'])")
    p.add_argument("--vae_path", type=str, default="",
                   help="VAE/Gemma 权重所在 .safetensors (默认同 --pretrained_model_path)")
    p.add_argument("--camera_mode", choices=["camtext", "prope"], default="camtext",
                   help="camtext=相机走【逐帧 cam-text via action_labels】(LTX23Model 内置 cross-attn, prompt 仅 caption); prope=不支持(报错)")
    p.add_argument("--use_discrete_action", action="store_true", default=True)
    p.add_argument("--no_discrete_action", dest="use_discrete_action", action="store_false")
    # flow matching
    p.add_argument("--sigma_shift", type=float, default=3.0,
                   help="训练 flow-matching sigma shift (u~U[0,1]→shift, 同 Wan/HY15)")
    p.add_argument("--validation_shift", type=float, default=5.0,
                   help="验证采样器 shift; 与训练 sigma_shift 解耦")
    # (已移除 --logit_mean/--logit_std: sigma 改用 Wan 同款 rand→shift, 见 train_one_step)
    # optim
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--betas", type=str, default="0.9,0.999")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="adamw",
                   help="adamw=与 Wan2.2-5B 一致(默认); muon=minWM 风格(2D Newton-Schulz + 1D AdamW backup)")
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
    p.add_argument("--output_dir", type=str, default="./logs/ltx23/stage1")
    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # validation (官方对齐: 采样器 shift=validation_shift=5.0, cfg=6.0, 50 步)
    p.add_argument("--validation_interval", type=int, default=50, help="每 N 步验证一次; 0=关")
    p.add_argument("--val_same_input", action="store_true", default=False,
                   help="通信自检: 所有rank用相同动作+caption+seed; 输出仍不同才是通信/FSDP问题")
    p.add_argument("--first_validation_step", type=int, default=50,
                   help="首次验证的最早步; 须 <= validation_interval, 否则首个 step%%interval==0 的步会被跳过(如51会跳过step50→首验落到100)")
    p.add_argument("--validate_before_training", action="store_true", default=False,
                   help="训练前(step 0)先跑一次验证, 在权重被训练污染前确认 base 权重+验证路径(MLLM camtext 编码/采样器/VAE 解码)整体正确")
    p.add_argument("--diffusion_sampling_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=6.0,
                   help="验证 CFG 引导强度(=minWM); >1 启用 cond+uncond 双前向, 1.0=不做 CFG")
    p.add_argument("--fps", type=float, default=24.0,
                   help="验证视频保存 fps + LTX forward 时间位置归一化的 fps (LTX positions 按 fps 归一)")
    return p.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main(parse_args())
