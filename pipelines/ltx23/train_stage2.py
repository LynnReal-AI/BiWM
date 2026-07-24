"""Stage 2 DMD distillation for the LTX-Video 2.3 backbone."""

import argparse
import gc
import os
import random
import time
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pipelines.ltx23.train_stage1 import (
    init_distributed, mprint, is_main,
    build_ltx23_transformer, wrap_fsdp,
    load_ltx_vae, ltx_vae_encode, ltx_vae_decode,
    load_text_encoder, ltx_encode, encode_neg_prompt,
    make_live_batch, resolve_sample_task, resolve_cond_n, frame_loss_mask,
    save_checkpoint,
    make_cam_text_encode_fn,
    _read_safetensors_config, _resolve_weight_file, LTX_VAE_TEMPORAL_STRIDE,
    # 验证视频保存工具 (与 stage1 同款, torchvision write_video crf18)
    save_video, _frames_to_uint8, LTX_VAE_SPATIAL_STRIDE, _VAL_ACTIONS,
)
from ltx23.modules.model import LTX23AttentionBlock


# =============================================================================
# flow-matching 小工具 (与 stage1 一致; LTX flow 约定 x0 = xt - sigma*v)
# =============================================================================
def pred_x0_from_flow(noisy, v, sigma):
    """x0 = x_t - sigma * v (LTX flow-matching, 与 stage1 验证 _flow 一致; sigma 广播)。"""
    return (noisy.float() - sigma.float() * v.float()).to(noisy.dtype)


def shift_sigma(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


def build_dit_from_ckpt(ckpt_path, pretrained_fallback, device, dtype, trainable,
                        camera_mode="camtext", grad_ckpt=False):
    """Build a trainable or frozen model from a Stage 1 or base checkpoint."""
    # ckpt 自身有 config 则用 ckpt 作 pretrained_model_path; 否则退回 base(.safetensors) 取 config + 权重。
    use_path = ckpt_path
    if not ckpt_path or not _read_safetensors_config(ckpt_path, key='transformer'):
        if pretrained_fallback and _read_safetensors_config(pretrained_fallback, key='transformer'):
            mprint(f"[DMD-LTX23] ckpt({ckpt_path}) 无 config, 退回 base 取 config+权重: {pretrained_fallback}")
            use_path = pretrained_fallback
    shim = types.SimpleNamespace(
        pretrained_model_path=use_path,
        camera_mode=camera_mode,
        gradient_checkpointing=bool(grad_ckpt and trainable),
    )
    model = build_ltx23_transformer(shim, device, dtype)
    model.requires_grad_(trainable)
    model.train(trainable)
    return model


# =============================================================================
# Gemma 显存 offload —— stage2 是 3×22B(gen+critic 可训 + teacher 冻结)
#   + 12B Gemma, 8×H100 会 OOM(Gemma 24GB/卡不分片是大头)。做法: precompute 后, 用 GPU Gemma
#   把【所有 caption】预编码成 context 嵌入存 CPU(显存换内存, 节点 RAM 充裕), 释放 Gemma 省 ~24GB;
#   训练每步按 caption 查表→搬回 GPU(嵌入很小, 传输可忽略)。保留【全 padded context】与 stage1 一致,
#   不改 DMD 算法; L 无关(存多大都行)。CPU 编码太慢故不做 miss 兜底, 未命中退空串(理论不会发生)。
# =============================================================================
class _CachedEnc:
    __slots__ = ("video_encoding",)
    def __init__(self, v): self.video_encoding = v


class _CachedGemma:
    """drop-in 替换 te: te(text).video_encoding 走 CPU 缓存(查不到退 "" 并告警)。带 .dtype 供 ltx_encode。"""
    def __init__(self, cache_cpu, device, dtype):
        self._cache = cache_cpu; self.device = device; self.dtype = dtype; self._warned = set()
    def __call__(self, text):
        key = text if text in self._cache else (text.rstrip() if text.rstrip() in self._cache else "")
        if key == "" and text.strip() and text not in self._warned:
            self._warned.add(text)
            print(f"[CachedGemma][WARN] caption 未预编码, 退空串: {text[:60]!r}", flush=True)
        return _CachedEnc(self._cache[key].to(self.device, self.dtype))


def _build_caption_cache_and_free_gemma(mllm, args, device):
    """GPU Gemma 预编码所有 caption(.rstrip())+ "" → CPU dict, 释放 Gemma, 返回 _CachedGemma。"""
    import json as _json, time as _t
    caps = set([""])
    try:
        data = _json.load(open(args.biwm_caption_json))
        for e in (data if isinstance(data, list) else data.values()):
            c = (e.get("caption", "") if isinstance(e, dict) else "") or ""
            caps.add(c.rstrip())
    except Exception as ex:
        mprint(f"[DMD-LTX23] 读 caption json 失败({ex}); 只缓存空串(其余退空串)。")
    mprint(f"[DMD-LTX23] Gemma offload: GPU 预编码 {len(caps)} 条 caption → CPU ...")
    cache = {}; t0 = _t.time()
    with torch.no_grad():
        for i, c in enumerate(caps):
            cache[c] = mllm(c).video_encoding.detach().to("cpu", torch.bfloat16)
            if i and i % 500 == 0:
                mprint(f"  ...{i}/{len(caps)} ({_t.time()-t0:.0f}s)")
    _shape = tuple(next(iter(cache.values())).shape)
    mprint(f"[DMD-LTX23] 预编码完成 {len(cache)} 条(单条 {_shape}), 用时 {_t.time()-t0:.0f}s; 释放 Gemma 省 ~24GB/卡")
    del mllm; gc.collect(); torch.cuda.empty_cache()
    return _CachedGemma(cache, device, torch.bfloat16)


def _precompute_cam_text(model, te, device):
    """逐帧 cam-text cross-attn 预编码 (镜像 ltx23_model.main():860-861)。
    用 Gemma 把 81 类 ACTION_TEXT_TABLE 相机文本编码成 _cam_text_raw 存进 model, 并置
    _use_cam_text_cross_attn=True。必须在 FSDP wrap 【之前】对【每个】未 wrap 模型各调一次:
      之后 _forward 收到 action_labels 即按逐 latent 帧 gather cam-text + caption 注入 cross-attn。"""
    _fn = make_cam_text_encode_fn(te, device, torch.bfloat16)
    model.precompute_cam_text_embeddings(_fn, device, torch.bfloat16)


# =============================================================================
# 调一次 LTX2Model 取 velocity —— 替换 HY15 dit_velocity(forward_bi); LTX 走 cond_latent_frames
# =============================================================================
def ltx_velocity(model, x, sigma, ctx, cond_latent_frames=0, action_labels=None, fps=24.0):
    """LTX 原生 forward 取 flow velocity (与 stage1 train_one_step/_flow 同款调用)。
    x:[B,C,F,H,W]; sigma:[B] (per-batch scalar, ∈[0,1]); ctx:[B,L,C] (Gemma context, 无 attention mask)。
    cond_latent_frames>0 时模型内部把前 N 个 latent 帧的 timestep 置 σ=0(clean 前缀/首帧),
      这取代了 HY15 的 per_frame_sigma 逐帧档机制: 前缀帧 σ=0、其余帧用传入的 sigma。
    ★ action_labels:[B,F] 逐 latent 帧离散相机 label (None=纯caption) —— 与 Wan2.2/stage1 一致,
      LTX23Model 内部按 action_labels[b,t] gather cam-text + caption 拼 per_frame_context 做 cross-attn。
    返回 v:[B,C,F,H,W] (flow velocity = noise - x0)。"""
    B, C, Fl, H, W = x.shape
    x_list = [x[i] for i in range(B)]
    ctx_list = [ctx[i] for i in range(ctx.shape[0])]
    seq_len = Fl * H * W
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        v = model(
            x_list,
            t=sigma.to(torch.bfloat16),
            context=ctx_list,
            seq_len=seq_len,
            cond_latent_frames=int(cond_latent_frames),
            action_labels=action_labels,
            fps=float(fps),
        )                                                  # [B,C,F,H,W]
    return v


def _bcast_int(val, device):
    """rank0 的 int broadcast 到所有 rank (保证 grad_step 各 rank 一致)。"""
    if (not dist.is_initialized()) or dist.get_world_size() == 1:
        return val
    t = torch.tensor([int(val)], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


def _band_sigma(B, device, lo, hi):
    """对齐 minWM —— renoise σ 在 generator 留梯度步窄带 [lo, hi] 内均匀采样
    (lo=sigmas[k_i+1], hi=sigmas[k_i]; 均为 shift 后 σ)。real/fake 在同一窄带比较 x0, score 匹配才准。"""
    u = torch.rand(B, device=device)
    return (lo + u * (hi - lo)).clamp(0.0, 1.0)


def velocity_to_x0(v, x_t, sigma):
    """flow matching: x0 = x_t - σ·v. sigma 标量 or [B] 广播 (镜像 dmd_wan.velocity_to_x0)。"""
    if torch.is_tensor(sigma) and sigma.dim() > 0:
        sigma = sigma.view(-1, *([1] * (x_t.dim() - 1)))
    return x_t - sigma * v


# =============================================================================
# DMD generator and critic losses. CFG applies camera text only to the
# conditional branch so guidance preserves both caption and camera control.
# =============================================================================
def compute_dmd_loss(real_score, fake_score, x0_gen, caption_emb, neg_caption_emb,
                     full_action_labels, sig_lo, sig_hi, args, device):
    """DMD generator loss (镜像 dmd_wan.py:408-458)。real_score(frozen teacher, CFG) +
    fake_score(critic), 把 x0_gen 当完整视频: caption + 整段逐帧相机, 无 history。
    sig_lo/sig_hi: generator 留梯度步窄带; teacher/critic 在同一窄带加噪比较 x0。"""
    B, C, Fn, H, W = x0_gen.shape
    fps = float(getattr(args, "fps", 24.0))
    with torch.no_grad():
        sigma = _band_sigma(B, device, sig_lo, sig_hi).to(x0_gen.dtype)
        noise = torch.randn_like(x0_gen)
        s5 = sigma.view(B, 1, 1, 1, 1)
        x_t = (1 - s5) * x0_gen + s5 * noise

        # real score (独立 frozen teacher) + CFG: cond=caption+逐帧相机; uncond=neg caption(action_labels=None)
        v_real_c = ltx_velocity(real_score, x_t, sigma, caption_emb,
                                action_labels=full_action_labels, fps=fps)
        v_real_u = ltx_velocity(real_score, x_t, sigma, neg_caption_emb,
                                action_labels=None, fps=fps)
        x0_real_c = velocity_to_x0(v_real_c, x_t, sigma)
        x0_real_u = velocity_to_x0(v_real_u, x_t, sigma)
        x0_real = x0_real_c + (x0_real_c - x0_real_u) * args.real_guidance_scale

        # fake score (独立 critic)
        v_fake = ltx_velocity(fake_score, x_t, sigma, caption_emb,
                              action_labels=full_action_labels, fps=fps)
        x0_fake = velocity_to_x0(v_fake, x_t, sigma)

        # DMD grad (eq.7/8): (fake-real) / mean|x0_gen - real|
        grad = (x0_fake - x0_real)
        normalizer = (x0_gen - x0_real).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
        grad = grad / (normalizer + 1e-8)
        grad = torch.nan_to_num(grad)

    target = (x0_gen.double() - grad.double()).detach()
    dmd_loss = 0.5 * F.mse_loss(x0_gen.double(), target)
    _log = {"dmd_grad_abs": grad.abs().mean().item(), "dmd_sigma": float(sigma.mean().item())}
    return dmd_loss, _log


def compute_critic_loss(fake_score, x0_gen, caption_emb, full_action_labels,
                        sig_lo, sig_hi, args, device):
    """fake_score (独立 critic, 全参在线更新) 学 denoise generator 输出: flow matching velocity loss
    (镜像 dmd_wan.py:575-593)。cond=caption+逐帧相机(action_labels=full_action_labels)。"""
    x0 = x0_gen.detach()
    B, C, Fn, H, W = x0.shape
    fps = float(getattr(args, "fps", 24.0))
    sigma = _band_sigma(B, device, sig_lo, sig_hi).to(x0.dtype)
    noise = torch.randn_like(x0)
    s5 = sigma.view(B, 1, 1, 1, 1)
    x_t = (1 - s5) * x0 + s5 * noise
    v_fake = ltx_velocity(fake_score, x_t, sigma, caption_emb,
                          action_labels=full_action_labels, fps=fps)
    target_v = (noise - x0)                              # flow matching velocity
    critic_loss = F.mse_loss(v_fake.float(), target_v.float())
    return critic_loss, {"critic_sigma": float(sigma.mean().item())}


# =============================================================================
# 自回归 (block-AR) generator rollout —— LTX 版: clean 前缀走 cond_latent_frames(非 per_frame_sigma)
# =============================================================================
def generator_block_rollout(gen, noise_full, cond, full_action_labels, args, device):
    """逐 block 自回归生成 x0_gen —— LTX cond_latent_frames 机制(取代 HY15 per_frame_sigma + forward_bi block mask):
      block b 喂 x_in = cat([clean历史(0:s0) | 当前块(s0:s1, σ=sg)], dim=2), 传 cond_latent_frames=s0;
      模型内部把前 s0 帧 timestep 置 σ=0(干净历史), 其余帧用 sg → 只取当前块 velocity。
      - 历史 = 已生成的 x0 块(detached), 作 clean 前缀; RoPE 时间真实递增(当前块落 s0..s1-1)。
      - K(latent 帧/block)= args.dmd_block_K (LTX 无 forward_bi 4 帧 hardcode 约束, 纯 args 驱动)。
      - self-forcing: 每 block few-step, 仅【留梯度步 grad_step】留梯度并 break; 历史 detach 不跨 block 串图。
      ★ 逐帧相机 (镜像 dmd_wan 'none' 模式 line~360): block b 喂【历史+当前】的 action_labels =
        full_action_labels[:, 0:s1] (与 x_in 帧一一对应; 前 s0 帧历史 + 当前 Kb 帧)。
      ★ _outer_grad 修复 (NCCL 死锁, 镜像 dmd_wan.generator_rollout:374-386): 非命中步 set_grad_enabled(False)
        去噪后【还原到外层 grad 状态 _outer_grad】(critic rollout 在 no_grad 下调用), 不硬关 grad 制造孤儿图。
    返回 (x0_gen:[B,C,Tfull,H,W] 逐 block 带梯度, sig_lo, sig_hi: renoise 窄带下/上限)。"""
    B, C, Tf, H, W = noise_full.shape
    cond_emb = cond["prompt_embed"]
    fps = float(getattr(args, "fps", 24.0))
    K = args.dmd_block_K
    M = args.dmd_num_blocks if args.dmd_num_blocks > 0 else max(1, Tf // K)
    # 防止 M*K 超过 Tf(=latent 帧数)产生空块。LTX VAE 8x 时间 → 77帧→T_lat=10,
    #   若 sh 沿用 HY15 的 M=5/K=4(=20, HY15 VAE 4x→20 帧正好) 会越界产生空块 → v_full[:,:,-Kb:]
    #   在 Kb<0 时变成错误切片(0 vs 8 崩溃)。截到覆盖 Tf 所需块数 ceil(Tf/K)。
    M = min(M, max(1, (Tf + K - 1) // K))
    sigmas = [shift_sigma(s, args.dmd_timestep_shift) for s in args.dmd_denoising_sigmas]
    n_steps = len(sigmas)
    if getattr(args, "dmd_ts_schedule", False):
        grad_step = _bcast_int(random.randint(0, n_steps - 1), device)
    else:
        grad_step = n_steps - 1
    sched_next = sigmas[1:] + [0.0]
    sig_lo = float(sched_next[grad_step])                 # 窄带下限 = grad_step 去噪到的 σ (对齐 minWM)
    sig_hi = float(sigmas[grad_step])                     # 窄带上限 = grad_step 当前 σ
    _outer_grad = torch.is_grad_enabled()                 # 外层 grad 状态 (critic rollout 在 no_grad 下调)
    generated = []                                        # 已生成的 clean x0 块(detached), 作历史前缀
    x0_blocks = []
    for b in range(M):
        s0, s1 = b * K, min((b + 1) * K, Tf)
        Kb = s1 - s0
        if Kb <= 0:                                          # ★ 空块(s0>=Tf)直接停: 避免 -Kb 切片错位
            break
        prefix = torch.cat(generated, dim=2).detach() if generated else None  # [B,C,s0,H,W] 干净历史
        # 逐帧相机: 历史+当前帧的 action_labels (与 x_in 帧对齐); 无相机时 None (纯 caption)
        al_full = full_action_labels[:, 0:s1] if full_action_labels is not None else None
        blk_noisy = noise_full[:, :, s0:s1]
        x0_blk = None
        for i, sg in enumerate(sigmas):
            exit_flag = (i == grad_step)
            # 非命中步禁梯度、命中步带梯度(但跟随外层 _outer_grad, 不在 no_grad rollout 里硬开 grad)
            torch.set_grad_enabled(exit_flag and _outer_grad)
            # Prefix clean history before the current noisy block.
            if prefix is not None:
                x_in = torch.cat([prefix, blk_noisy], dim=2)            # [B,C,s0+Kb,H,W]
                cond_n = s0
            else:
                x_in = blk_noisy
                cond_n = 0
            sig_b = torch.full((B,), float(sg), device=device)          # per-batch scalar σ
            v_full = ltx_velocity(gen, x_in, sig_b, cond_emb, cond_latent_frames=cond_n,
                                  action_labels=al_full, fps=fps)
            v_blk = v_full[:, :, -Kb:]                                  # 只取当前块 velocity
            x0_blk = pred_x0_from_flow(blk_noisy, v_blk,
                                       torch.full((B, 1, 1, 1, 1), float(sg), device=device))
            torch.set_grad_enabled(_outer_grad)                         # ★ 还原外层 grad 状态(不硬关)
            if exit_flag:
                break
            sg_n = sigmas[i + 1]                                            # 桥接到下一步(detach)
            if getattr(args, "dmd_euler_rollout", False):
                blk_noisy = (blk_noisy + v_blk * (sg_n - sg)).detach()                       # euler 确定性
            else:
                blk_noisy = ((1 - sg_n) * x0_blk + sg_n * torch.randn_like(x0_blk)).detach()  # cm 重加噪
        generated.append(x0_blk.detach())                                  # 作下一 block 历史(detach)
        x0_blocks.append(x0_blk)
    return torch.cat(x0_blocks, dim=2), sig_lo, sig_hi


# =============================================================================
# 验证: 逐 block 自回归采样 (全局共享 caption + 每窗口自己的 cam-text → 看相机随窗口变化)
# =============================================================================
@torch.no_grad()
def run_validation_dmd(gen, vae, mllm, val_sample, args, step, device, rank):
    """generator few-step 逐 block rollout 采样 (train_mode=False, 固定 M 块):
      - 文本: 全局共享 caption, 每个【窗口/block】拼自己的 cam-text(window_actions[b]) → 相机随新窗口变;
      - LTX cond_latent_frames 历史机制; 与训练 rollout 同结构, 但 no_grad + 逐块换文本。
    LTX VAE 解码 + joystick overlay 存 mp4。"""
    import numpy as np
    gen.eval()
    task = args.training_mode
    B, C, Tf, h, w = val_sample["latent"].shape
    fps_v = float(getattr(args, "fps", 24.0))
    K = args.dmd_block_K
    M = args.dmd_num_blocks if args.dmd_num_blocks > 0 else max(1, Tf // K)
    M = min(M, max(1, (Tf + K - 1) // K))   # 同训练 rollout —— 截 M 到 ceil(Tf/K), 防空块崩溃(LTX T_lat=10)
    caption = str(val_sample.get("caption", "")).rstrip()

    # 每个窗口一个相机动作(demo: 展示相机随窗口切换); 各 rank 起点不同 → 各 rank 一条不同的运镜序列
    window_actions = [_VAL_ACTIONS[(rank + b) % len(_VAL_ACTIONS)] for b in range(M)]
    # 相机走【逐帧 action_labels】(与训练/stage1 一致), prompt 恒 caption-ONLY。
    #   每个窗口一个常量动作 → 该窗口 K 帧的 action_labels 全填该动作。
    pe_cap = ltx_encode(mllm, [caption], device)                          # [1,L,C] caption-only context
    block_al = [torch.full((B, K), int(a), dtype=torch.long, device=device) for a in window_actions]

    sigmas = [shift_sigma(s, args.dmd_timestep_shift) for s in args.dmd_denoising_sigmas]
    g = torch.Generator(device=device).manual_seed(42 + rank)
    noise_full = torch.randn(val_sample["latent"].shape, generator=g, device=device, dtype=torch.bfloat16)
    # ★ 与训练 rollout 一致 —— 每 block 喂 [clean历史 | 当前块], cond_latent_frames=s0(前缀帧 σ=0)。
    generated = []
    al_so_far = []                                                        # 累积逐帧 action_labels(历史+当前)
    for b in range(M):
        s0, s1 = b * K, min((b + 1) * K, Tf)
        Kb = s1 - s0
        if Kb <= 0:                                                       # ★ 空块(s0>=Tf)直接停, 同训练 rollout
            break
        prefix = torch.cat(generated, dim=2) if generated else None       # [B,C,s0,h,w] 干净历史
        al_so_far.append(block_al[b][:, :Kb])                             # 当前窗口逐帧动作
        al_full = torch.cat(al_so_far, dim=1)                            # [B, s1] 历史+当前帧相机
        blk_noisy = noise_full[:, :, s0:s1]
        x0_blk = None
        for i, sg in enumerate(sigmas):
            if prefix is not None:
                x_in = torch.cat([prefix, blk_noisy], dim=2)
                cond_n = s0
            else:
                x_in = blk_noisy
                cond_n = 0
            sig_b = torch.full((B,), float(sg), device=device)
            v_full = ltx_velocity(gen, x_in, sig_b, pe_cap, cond_latent_frames=cond_n,
                                  action_labels=al_full, fps=fps_v)
            v_blk = v_full[:, :, -Kb:]
            x0_blk = pred_x0_from_flow(blk_noisy, v_blk,
                                       torch.full((B, 1, 1, 1, 1), float(sg), device=device))
            if i < len(sigmas) - 1:
                sg_n = sigmas[i + 1]
                if getattr(args, "dmd_euler_rollout", False):
                    blk_noisy = blk_noisy + v_blk * (sg_n - sg)
                else:
                    blk_noisy = (1 - sg_n) * x0_blk + sg_n * torch.randn_like(x0_blk)
        generated.append(x0_blk)
    clean = torch.cat(generated, dim=2)                               # [B,C,Tfull,h,w]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        vid = ltx_vae_decode(vae, clean)                              # [B,3,T,H,W]
    frames = _frames_to_uint8(vid)                            # list[np.uint8 HWC] (无 overlay)
    d = os.path.join(args.output_dir, "validation"); os.makedirs(d, exist_ok=True)
    fps = int(getattr(args, "fps", 24))
    # ★ 保存格式/类型对齐 Wan/stage1: torchvision write_video crf18; 命名 validation_/joystick_{prefix}
    prefix = f"step_{step:07d}_{task}_camtext_video_real_rank_{rank:03d}_dmd"
    save_video(frames, os.path.join(d, f"validation_{prefix}.mp4"), fps=fps)   # 生成(无overlay)
    try:
        from pipelines.common.control_overlay import superimpose_control_video as add_joystick_overlay
        per_lat_act = torch.tensor([window_actions[min(t // K, M - 1)] for t in range(Tf)], dtype=torch.long)
        frames_j = add_joystick_overlay(frames, per_lat_act, vae_temporal_stride=LTX_VAE_TEMPORAL_STRIDE)
        save_video(frames_j, os.path.join(d, f"joystick_{prefix}.mp4"), fps=fps)  # 叠摇杆(逐窗动作)
    except Exception as e:
        mprint(f"[DMD-LTX23][val] joystick overlay 跳过: {type(e).__name__}: {e}")
    try:
        with open(os.path.join(d, f"validation_{prefix}_prompt.txt"), "w", encoding="utf-8") as mf:
            mf.write(f"Step: {step}\nRank: {rank}\nTask: {task}\nSource: video_real\n")
            mf.write(f"AR blocks(M): {M}  block_K: {K}\nWindow actions(逐窗相机): {window_actions}\n")
            mf.write(f"Caption: {caption}\nSampling sigmas: {args.dmd_denoising_sigmas}\nFPS: {fps}\n")
    except Exception as e:
        mprint(f"[DMD-LTX23][val] metadata 跳过: {type(e).__name__}: {e}")
    if is_main():
        mprint(f"[DMD-LTX23][val] step {step}: AR rollout {M}块, 窗口动作={window_actions} -> {d}/validation_{prefix}.mp4")


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
    dtype = torch.bfloat16
    task = args.training_mode

    # Build the student, critic, and teacher from the camera-text backbone.
    _gc = bool(getattr(args, "gradient_checkpointing", False))
    _base = args.pretrained_model_path
    generator = build_dit_from_ckpt(args.generator_ckpt or _base, _base,
                                    device, dtype, trainable=True,
                                    camera_mode=args.camera_mode, grad_ckpt=_gc)
    # ★ critic 必须开 grad_ckpt 修 OOM(单次 fwd 存满全部层激活 + generator rollout 图, H200 也放不下)。
    fake_score = build_dit_from_ckpt(args.fake_score_ckpt or args.generator_ckpt or _base, _base,
                                     device, dtype, trainable=True,
                                     camera_mode=args.camera_mode, grad_ckpt=_gc)
    real_score = build_dit_from_ckpt(args.real_score_ckpt or args.generator_ckpt or _base, _base,
                                     device, dtype, trainable=False,
                                     camera_mode=args.camera_mode)
    real_score.eval()

    # ---- 逐帧 cam-text 预编码 (镜像 ltx23_model.main:855-862) ----
    #   文本编码器(Gemma)必须在 FSDP wrap【之前】加载, 并对【每个未 wrap 的】模型各调一次
    #   precompute_cam_text_embeddings: 用 Gemma 把 81 类 ACTION_TEXT_TABLE 相机文本编码成 _cam_text_raw
    #   存进各 model, 并置 _use_cam_text_cross_attn=True; FSDP wrap 后随 model 存活。
    #   之后 _forward 收到 action_labels 即按逐 latent 帧 gather cam-text + caption 注入 cross-attn。
    vae = load_ltx_vae(args.pretrained_model_path if not args.vae_path else args.vae_path, device)
    mllm = load_text_encoder(args.pretrained_model_path if not args.vae_path else args.vae_path,
                             args.gemma_path, device)
    for _m in (generator, fake_score, real_score):
        _precompute_cam_text(_m, mllm, device)
    mprint("[DMD-LTX23] 逐帧 cam-text 预编码完成 (generator/fake_score/real_score 各一次; FSDP 前)")

    # ---- ★ Gemma offload 省显存 (默认关; 仅 H100 等紧显存时开 DMD_GEMMA_OFFLOAD=1) ----
    #   真正的 OOM 是【加载阶段】3×22B 先各自 .to(GPU) 峰值 132GB(H100-80G 装不下; H200-141G 可以)。
    #   offload 只省稳态 Gemma 24GB, 救不了加载峰值。H200 上不需要, 故默认走原路径(已验证可加载)。
    if os.environ.get("DMD_GEMMA_OFFLOAD", "0") == "1":
        mllm = _build_caption_cache_and_free_gemma(mllm, args, device)

    # ---- FSDP (各自包; LTX23AttentionBlock 为 wrap 单元, 见 ltx23_model.wrap_fsdp) ----
    if world > 1:
        generator = wrap_fsdp(generator, device, args)
        fake_score = wrap_fsdp(fake_score, device, args)
        real_score = wrap_fsdp(real_score, device, args)

    gen_opt = torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad],
                                lr=args.dmd_generator_lr, weight_decay=args.weight_decay)
    critic_opt = torch.optim.AdamW([p for p in fake_score.parameters() if p.requires_grad],
                                   lr=args.dmd_critic_lr, weight_decay=args.weight_decay)

    # ---- 稳定化开关日志 ----
    _critic_warmup = int(getattr(args, "dmd_critic_warmup_steps", 0))
    mprint(f"[DMD-LTX23] 稳定化: critic_warmup={_critic_warmup} "
           f"ts_schedule={getattr(args,'dmd_ts_schedule',False)}")

    # ---- 数据 ----
    #   live(默认, 对齐 stage1): video_real mp4 → 在线 LTX VAE + Gemma 编码; 训练 caption-only(不引入相机)。
    from pipelines.dataset.biwm_camera_text_dataset import BiwmCamCaptionData
    from torch.utils.data import DataLoader
    dataset = BiwmCamCaptionData(
        video_dir=args.biwm_video_dir, caption_json=args.biwm_caption_json,
        width=args.num_width, height=args.num_height, num_frames=args.num_frames,
        vae_temporal_factor=LTX_VAE_TEMPORAL_STRIDE)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=args.dataloader_num_workers, collate_fn=_live_collate)
    # vae / mllm 已在 FSDP wrap 前加载 (precompute 用), 此处不再重复加载。

    denoise_sigmas = [shift_sigma(s, args.dmd_timestep_shift) for s in args.dmd_denoising_sigmas]
    mprint(f"[DMD-LTX23] start: camera_mode={args.camera_mode}(训练 {task} caption-only), "
           f"gen few-step sigmas(shift前)={args.dmd_denoising_sigmas} "
           f"→(shift后){[round(s,3) for s in denoise_sigmas]}, "
           f"update_ratio={args.dfake_gen_update_ratio}, world={world}")
    os.makedirs(args.output_dir, exist_ok=True)

    def get_cond_live(b):
        """live: b 是 make_live_batch 返回的 dict(caption_only=True → 训练纯 caption)。
        LTX cond 只需 prompt_embed(Gemma context, 无 attention mask)。"""
        return dict(prompt_embed=b["prompt_embed"], image_cond=b.get("image_cond"), task=b.get("task", task))

    def _fetch_live(it):
        while True:
            try:
                raw = next(it)
            except StopIteration:
                it = iter(loader); raw = next(it)
            if ((isinstance(raw, dict) and raw.get("skip")) or
                    (isinstance(raw, tuple) and raw and raw[0] == "skip")):
                continue
            return make_live_batch(raw, vae, mllm, args, device, caption_only=True), it

    val_sample = None       # 缓存首条(含 caption/clip_action), 验证用
    # ★ 预编码 CFG 负向 = 空串编码(=stage1/minWM), 供 real_score uncond 用(LTX context 无 mask)。
    _neg_pe = encode_neg_prompt(mllm, device) if mllm is not None else None
    step, it = 0, iter(loader)
    _ratio = max(1, int(args.dfake_gen_update_ratio))   # critic 每步更新; gen 每 _ratio 步更新(对齐 Wan)
    _gen_upd = 0                                         # gen 更新计数(供梯度累积窗口)
    _hb = lambda tag, t: mprint(f"[DMD-LTX23][hb] step {step} {tag} +{time.perf_counter()-t:.1f}s") \
        if is_main() and (step < 5 or step % args.log_interval == 0) else None
    while step < args.max_train_steps:
        _t = time.perf_counter()
        if is_main() and (step < 5 or step % args.log_interval == 0):
            mprint(f"[DMD-LTX23][hb] step {step} 进入循环, 取数据...")
        batch, it = _fetch_live(it)
        if val_sample is None:
            val_sample = {"latent": batch["latent"].detach(),
                          "caption": batch.get("caption", ""),
                          "image_cond": batch["image_cond"].detach()}
        cond = get_cond_live(batch)
        latent = batch["latent"].to(device, dtype)
        B = latent.shape[0]
        # 逐帧离散相机 action_labels [1, T_lat] (make_live_batch 返回; video_real 整段同一动作)。
        #   rollout 整段帧数 = M*K (M=block 数, K=每块 latent 帧); 长度不符则 nearest 重采样到 M*K (保离散类别)。
        full_action_labels = batch.get("action_labels", None)
        if full_action_labels is not None:
            full_action_labels = full_action_labels.to(device=device, dtype=torch.long)
            _K = args.dmd_block_K
            _M = args.dmd_num_blocks if args.dmd_num_blocks > 0 else max(1, latent.shape[2] // _K)
            # rollout 实际只生成 min(M*K, T_lat) 帧(见 generator_block_rollout 的 M 截断);
            #   action_labels 须对齐到此, 否则会把已对齐 T_lat 的逐帧 label 错误上采样(丢后半段动作)。
            _MK = min(_M * _K, latent.shape[2])
            if full_action_labels.shape[1] != _MK:
                full_action_labels = F.interpolate(full_action_labels.float().unsqueeze(1), size=_MK,
                                                   mode='nearest').squeeze(1).long()

        _accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        # ===== 1) critic(fake_score) 更新【每步 1 次】—— 用自己的 fresh rollout(no_grad), 对齐 Wan/minWM。
        _hb("取数据完成, 开始 critic rollout", _t)
        critic_opt.zero_grad()
        with torch.no_grad():
            x0_c, _lo_c, _hi_c = generator_block_rollout(generator, torch.randn_like(latent),
                                                         cond, full_action_labels, args, device)
        x0_c = x0_c.detach()
        # ★ critic: flow-matching velocity loss; cond=caption+逐帧相机(action_labels) (镜像 dmd_wan.compute_critic_loss)
        c_loss, _ = compute_critic_loss(fake_score, x0_c, cond["prompt_embed"],
                                        full_action_labels, _lo_c, _hi_c, args, device)
        c_loss.backward()
        (fake_score.clip_grad_norm_(args.max_grad_norm) if hasattr(fake_score, "clip_grad_norm_")
         else torch.nn.utils.clip_grad_norm_(fake_score.parameters(), args.max_grad_norm))
        critic_opt.step()
        _hb("critic 完成", _t)

        # ===== 2) generator 更新【每 _ratio 步 1 次, warmup 后】—— 自己的 fresh rollout(带梯度), 对齐 Wan。
        g_loss = None
        _is_gen_step = (step >= _critic_warmup) and (step % _ratio == 0)
        if _is_gen_step:
            _accum_first = (_gen_upd % _accum == 0)
            _accum_last = (_gen_upd % _accum == _accum - 1)
            if _accum_first:
                gen_opt.zero_grad()
            x0_gen, _sig_lo, _sig_hi = generator_block_rollout(generator, torch.randn_like(latent),
                                                               cond, full_action_labels, args, device)
            cond_emb = cond["prompt_embed"]
            # ★ CFG uncond —— 用【空串编码 neg-prompt】(=stage1/minWM); 无 mllm 兜底退回全零文本。
            if _neg_pe is not None:
                neg_emb = _neg_pe.to(device, x0_gen.dtype).expand(B, -1, -1)
            else:
                neg_emb = torch.zeros_like(cond_emb)
            # The conditional branch uses camera text; the negative branch does not.
            g_loss, _ = compute_dmd_loss(real_score, fake_score, x0_gen, cond_emb, neg_emb,
                                         full_action_labels, _sig_lo, _sig_hi, args, device)
            (g_loss / _accum).backward()              # 累积: 除以窗口大小(=1 时不变)
            if _accum_last:                            # 窗口末才 clip + step
                (generator.clip_grad_norm_(args.max_grad_norm) if hasattr(generator, "clip_grad_norm_")
                 else torch.nn.utils.clip_grad_norm_(generator.parameters(), args.max_grad_norm))
                gen_opt.step()
            _gen_upd += 1                              # gen 更新计数(累积窗口用)
        _hb("generator 更新完成(本步结束)", _t)

        step += 1
        if is_main() and step % args.log_interval == 0:
            _gmsg = (f"g_loss={g_loss.item():.4f}" if g_loss is not None
                     else f"gen=(critic warmup {step}/{_critic_warmup})")
            mprint(f"step {step}/{args.max_train_steps}  {_gmsg}  c_loss={c_loss.item():.4f}")
        # ---- 验证: 逐 block 自回归采样, 全局共享 caption + 每窗口自己的 cam-text ----
        if (args.validation_interval > 0 and vae is not None
                and val_sample is not None and step >= args.first_validation_step
                and (step % args.validation_interval == 0 or step == args.first_validation_step)):
            try:
                run_validation_dmd(generator, vae, mllm, val_sample, args, step, device, rank)
            except Exception as e:
                mprint(f"[DMD-LTX23][val] step {step} 验证失败(跳过): {type(e).__name__}: {e}")
            generator.train()
        if step % args.checkpointing_steps == 0:
            save_checkpoint(generator, args.output_dir, step)
    save_checkpoint(generator, args.output_dir, args.max_train_steps)
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser("LTX-Video 2.3 stage2 DMD distillation")
    p.add_argument("--json_path", type=str, required=False)
    p.add_argument("--training_mode", choices=["t2v", "i2v"], default="t2v")
    p.add_argument("--window_frames", type=int, default=20)
    p.add_argument("--i2v_rate", type=float, default=0.0)
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    p.add_argument("--data_mode", choices=["live"], default="live",
                   help="live=video_real mp4 在线 LTX VAE+Gemma(训练 caption-only)")
    p.add_argument("--camera_mode", choices=["camtext", "prope"], default="camtext",
                   help="camtext=相机走文本(全参基座); prope=不支持(LTX23 仅 camtext)")
    p.add_argument("--biwm_video_dir", type=str, default="", help="live: video_real 目录")
    p.add_argument("--biwm_caption_json", type=str, default="", help="live: video_real_input.json")
    p.add_argument("--vae_path", type=str, default="", help="live: LTX VAE 权重 .safetensors(默认同 pretrained)")
    p.add_argument("--text_encoder_path", type=str, default="", help="(LTX 未用; Gemma 见 --gemma_path)")
    p.add_argument("--gemma_path", type=str, default="",
                   help="Gemma-3 目录 (live 编码必需, e.g. .../google/gemma-3-12b-it-qat-q4_0-unquantized)")
    p.add_argument("--num_frames", type=int, default=77)
    p.add_argument("--num_height", type=int, default=480)
    p.add_argument("--num_width", type=int, default=832)
    p.add_argument("--training_cfg_rate", type=float, default=0.0)   # make_live_batch 需要(DMD 不丢文本)
    p.add_argument("--sigma_shift", type=float, default=5.0)         # 占位(make_live_batch 对齐)
    p.add_argument("--i2v_cond_latent_frames", type=int, default=1,
                   help="i2v 条件帧数 (latent 空间): 前 N 帧 timestep 置 σ=0 (cond_latent_frames)")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--validation_interval", type=int, default=100, help="每 N 步验证一次; 0=关")
    p.add_argument("--first_validation_step", type=int, default=100,
                   help="首次验证最早步; 须 <= validation_interval")
    # weights
    p.add_argument("--pretrained_model_path", type=str, default="",
                   help="LTX-2.3 base .safetensors (config fallback + VAE/Gemma 权重来源)")
    p.add_argument("--generator_ckpt", type=str, default="", help="stage1 ckpt 目录(含 config.json+权重)")
    p.add_argument("--fake_score_ckpt", type=str, default="")
    p.add_argument("--real_score_ckpt", type=str, default="")
    p.add_argument("--use_discrete_action", action="store_true", default=True)
    p.add_argument("--no_discrete_action", dest="use_discrete_action", action="store_false")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                   help="开激活重算(generator/fake_score): 省显存修 OOM(对齐 stage1)")
    # dmd (命名对齐 Wan/HY15)
    p.add_argument("--dmd_denoising_sigmas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25],
                   help="generator 每 block few-step 去噪的 sigma 列表(shift 前, [0,1]); 末尾隐含 0.0 = 4-step")
    p.add_argument("--dmd_block_K", type=int, default=4,
                   help="自回归 block 粒度(latent 帧/块); LTX 无 forward_bi mask 约束, 纯 args 驱动")
    p.add_argument("--dmd_num_blocks", type=int, default=0,
                   help="block 数(自回归块数); 0=按 latent 长度自动 = T_lat//dmd_block_K")
    p.add_argument("--dmd_timestep_shift", type=float, default=5.0)
    p.add_argument("--dfake_gen_update_ratio", type=int, default=5,
                   help="critic 更新次数 / generator 1 次")
    p.add_argument("--real_guidance_scale", type=float, default=6.0, help="real_score CFG")
    # optim
    p.add_argument("--dmd_generator_lr", type=float, default=2e-6)
    p.add_argument("--dmd_critic_lr", type=float, default=4e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="generator 梯度累积步数(对齐 Wan); =1 不累积")
    p.add_argument("--cp_size", type=int, default=1,
                   help="CP(序列并行)组大小; LTX2Model 当前不读 biWM SP, 保持 1(纯 DP)")
    p.add_argument("--max_train_steps", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default="./logs/ltx23/stage2_dmd")
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # ===== 稳定化机制 (对齐 Wan/HY15 stage2_dmd; 默认全关, sh 可逐项开) =====
    p.add_argument("--dmd_critic_warmup_steps", type=int, default=0,
                   help="前 N 步只训 critic, 之后再开 generator 更新 (0=关)")
    p.add_argument("--dmd_ts_schedule", action="store_true", default=False,
                   help="开 ts_schedule: rollout 随机留梯度步(Self-Forcing) + critic/DMD 加噪下限=denoised_to")
    p.add_argument("--dmd_euler_rollout", action="store_true", default=False,
                   help="rollout 桥接用 euler 确定性步而非 cm 重加噪 (诊断 cm 是否拖累)")
    p.add_argument("--dmd_min_step", type=float, default=0.0, help="critic/DMD 加噪 sigma 下限(0~1)")
    p.add_argument("--dmd_max_step", type=float, default=1.0, help="critic/DMD 加噪 sigma 上限(0~1)")
    # SFT / forward-KL 锚定 (LTX23 暂留 stub 参数对齐 sh; loss 项未接, 默认权重 0)
    p.add_argument("--dmd_sft_weight", type=float, default=0.0, help="(stub) SFT 权重, 0=关")
    p.add_argument("--dmd_sft_sigma_max", type=float, default=0.5)
    p.add_argument("--dmd_fkl_weight", type=float, default=0.0, help="(stub) 教师 forward-KL 权重, 0=关")
    p.add_argument("--dmd_fkl_steps", type=int, default=1)
    p.add_argument("--dmd_fkl_teacher_steps", type=int, default=12)
    p.add_argument("--dmd_real_fkl_weight", type=float, default=0.0, help="(stub) 真实数据 forward-KL 权重, 0=关")
    p.add_argument("--dmd_real_fkl_sigma_min", type=float, default=0.25)
    p.add_argument("--dmd_real_fkl_sigma_max", type=float, default=0.65)
    return p.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main(parse_args())
