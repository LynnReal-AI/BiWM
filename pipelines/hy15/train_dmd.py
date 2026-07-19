# BiWM HunyuanVideo-1.5 Stage 2 DMD training entry.
# 【死锁根因修复】generator_block_rollout 留梯度改为跟随【外层】grad 开关
#   (exit_flag and torch.is_grad_enabled(), 对齐 Wan dmd_wan.py:374 的 _outer_grad)。
#   原硬写 set_grad_enabled(exit_flag) 会让 critic 的 no_grad rollout 在 grad_step 处建出孤儿图(随后 detach 永不 backward),
#   FSDP 残留 generator 的 post-backward(reduce-scatter) hook; 叠加 checkpoint 存档(rank0_only gather 单改 rank0 handle)后,
#   存档后首个 generator.backward() rank0 不发 reduce-scatter → 其余 rank 永久等 → NCCL 死锁(确定性卡 step=checkpointing_steps)。
# validation/checkpoint 后补 dist.barrier()(+empty_cache)(辅助; 单靠它不够, 真因见上)。
# 训练文本由 caption-only 改为 camtext(相机+caption, 相机在前)。
#   generator/critic/teacher 三者共用同一 cond → 全部带相机文本; DMD 在【相机条件分布】上蒸馏,
#   不再把 stage1 学到的相机控制蒸掉。数据每 clip 单动作 → 整段一句相机文本, 各 block 共用(对应"block特定")。
#   验证仍逐 block 切相机文本(测泛化, 见 run_validation_dmd)。
# =============================================================================
# HunyuanVideo-1.5 (HY15) stage2 DMD 蒸馏 —— wan 风格单入口, 原生 forward_bi
# =============================================================================
# 参照 fastvideo/dmd_wan.py 的风格 + minWM ar_hunyuan_dmd_distill_pipeline 的算法,
# 把 stage1 的 HY15 多步 DiT 蒸馏成少步 generator (DMD2)。
#
# 三模型(均为 biWM/hunyuan 原生 ARHunyuanVideo_1_5_DiffusionTransformer, 走 forward_bi):
#   - generator (student, 可训)   : 从 stage1 ckpt 初始化, few-step 去噪
#   - fake_score (critic, 可训)    : 学 generator 当前分布, 给 fake score
#   - real_score (teacher, 冻结)   : 原 stage1 base, 给 real score (CFG)
# DMD2: KL 梯度 grad=(fake_x0-real_x0) 归一化 → generator loss=0.5*MSE(x0_gen,(x0_gen-grad).detach());
#       critic 对 generator 样本做去噪 (flow-matching) → critic loss。critic 更新 N 次 / generator 1 次。
#
# 复用 stage1: build/config/数据集/cond 拼装 都从 pipelines.hy15.train_hunyuan import (DRY)。
# ⚠️ 依赖 stage1 ckpt; 本文件按"先写对、后跑通"约定写, 权重/数据齐了再 DLC 调。
# =============================================================================
import argparse
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pipelines.hy15.train_hunyuan import (init_distributed, mprint, is_main,
                                  _load_transformer_config, _load_safetensors_dir,
                                  _load_sd_shape_tolerant,
                                  prepare_cond_latents, build_camera_plucker_dataloader,
                                  # stage2 全量刷新到 camtext/live, 复用 stage1 组件(DRY)
                                  load_hy15_vae, load_mllm, mllm_encode, make_live_batch, encode_neg_prompt,
                                  action_to_camtext, build_camtext_prompt, _live_collate,
                                  save_video, _frames_to_uint8, HY15_VAE_SPATIAL_STRIDE,
                                  HY15_VAE_TEMPORAL_STRIDE, BYT5_MAX_LEN, BYT5_DIM, _VAL_ACTIONS)
from hunyuan.modules.model import (ARHunyuanVideo_1_5_DiffusionTransformer,
                                   MMDoubleStreamBlock)
from pipelines.common.dmd_algo import (
    run_cfg as apply_cfg,
    calc_kl_gradient as compute_kl_gradient,
    dmd_student_loss as dmd_generator_loss,
    dmd_fake_score_loss as dmd_critic_loss,
)


# =============================================================================
# flow-matching 小工具 (与 stage1 一致)
# =============================================================================
def pred_x0_from_flow(noisy, v, sigma):
    """x0 = x_t - sigma * v"""
    return (noisy.float() - sigma.float() * v.float()).to(noisy.dtype)


def shift_sigma(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


# =============================================================================
# 构建一个原生 HY15 DiT, 从 stage1 ckpt(或 base 权重目录)加载
# =============================================================================
def build_dit_from_ckpt(ckpt_path, transformer_dir, device, dtype, use_action, trainable,
                        camera_mode="camtext", grad_ckpt=False):
    """camera_mode=camtext(默认): 取消 PRoPE/离散action, 相机走文本 → 不加 prope/action 参数(全参基座);
       camera_mode=action: 取消 PRoPE, 逐帧离散 action 经 TimestepEmbedder 注入 AdaLN;
       camera_mode=prope: 旧行为, 加 prope(+可选 action) 参数。"""
    cfg = _load_transformer_config(transformer_dir)
    model = ARHunyuanVideo_1_5_DiffusionTransformer(**cfg)
    model._biwm_config = dict(cfg)   # 供 _save 写 config.json(对齐 Wan/stage1, ckpt 与 base 同构)
    if camera_mode == "prope":
        model.add_prope_parameters()
        if use_action:
            model.add_discrete_action_parameters()
    elif camera_mode == "action":
        model.add_discrete_action_parameters()
    concat_condition = bool(cfg.get("concat_condition", False))

    sd = None
    if ckpt_path and os.path.isfile(ckpt_path):          # 显式给 .pt 文件(legacy stage1 输出)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mprint(f"[DMD] load legacy .pt {ckpt_path}")
    elif ckpt_path and os.path.isdir(ckpt_path):
        try:                                             # ★ 优先 safetensors(stage1/Wan 新格式 diffusion_pytorch_model.safetensors)
            sd = _load_safetensors_dir(ckpt_path)
            mprint(f"[DMD] load safetensors {ckpt_path}")
        except FileNotFoundError:                        # 退回 legacy model.pt(旧 stage1 输出)
            cand = os.path.join(ckpt_path, "model.pt")
            if os.path.isfile(cand):
                sd = torch.load(cand, map_location="cpu", weights_only=True)
                mprint(f"[DMD] load legacy {cand}")
    if sd is not None:
        miss, unexp, mism = _load_sd_shape_tolerant(model, sd)
        mprint(f"[DMD]  -> missing={len(miss)} unexpected={len(unexp)} shape-skip={len(mism)}")
        del sd
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(trainable)
    model.train(trainable)
    if grad_ckpt and trainable:          # 开激活重算(forward_bi 据此 checkpoint double_blocks),
        model.gradient_checkpointing = True   #   省可训模型(generator/fake_score)的激活显存, 修 OOM
    return model, concat_condition


# =============================================================================
# 调一次 DiT 取 velocity (拼 cond + 逐帧 timestep), 复用 stage1 的 concat 逻辑
# =============================================================================
def dit_velocity(model, noisy, sigma_scalar, cond, concat_condition, task, drop_text=False,
                 clean_x=None, aug_timesteps=None, per_frame_sigma=None):
    """整段双向 forward_bi 取 velocity。
    noisy:[B,C,T,H,W]; sigma_scalar:[B](整段一个 sigma) 或 per_frame_sigma:[B,T](逐帧 sigma, block rollout 用)。
    clean_x/aug_timesteps: 双向自回归 self-forcing 的 clean 前缀(整段 clean 估计) + 其逐帧噪声档(≈0)。
    返回 v:[B,C,T,H,W]。"""
    B, C, T, H, W = noisy.shape
    if concat_condition:
        cond_lat = prepare_cond_latents(task, cond.get("image_cond") if task == "i2v" else None, noisy)
        hidden = torch.cat([noisy, cond_lat], dim=1)
        clean_in = None
        if clean_x is not None:
            clean_lat = prepare_cond_latents(task, cond.get("image_cond") if task == "i2v" else None, clean_x)
            clean_in = torch.cat([clean_x, clean_lat], dim=1)
    else:
        hidden = noisy
        clean_in = clean_x
    if per_frame_sigma is not None:                       # [B,T] → 逐帧 timestep [B*T]
        timestep = (per_frame_sigma * 1000.0).reshape(-1).to(torch.bfloat16)
        _sig_txt = per_frame_sigma.amax(dim=1)            # 当前块的去噪σ(历史帧=0,当前块=sg → max=sg)
    else:
        timestep = (sigma_scalar * 1000.0).repeat_interleave(T).to(torch.bfloat16)
        _sig_txt = sigma_scalar
    # 文本时间调制。time_in 与图像 t 共用(model.py:1088 vs 1267) → 缩放约定 σ*1000。
    #   官方/base 约定: 跟随去噪σ。但【stage1 训练时 timestep_txt=0】(hy15_model.py:495), generator 学的是
    #   vec_txt=time_in(0); 采样喂 σ 会 OOD 塌黑(实测 mean170→19)。→ 随 stage1 用 0(自洽); 要改 σ 须 stage1 同改重训。
    _ = _sig_txt
    timestep_txt = torch.zeros(noisy.shape[0], device=noisy.device, dtype=torch.bfloat16)
    text_states = torch.zeros_like(cond["prompt_embed"]) if drop_text else cond["prompt_embed"]
    extra = {"byt5_text_states": cond["byt5_text_states"], "byt5_text_mask": cond["byt5_text_mask"]}
    v, _ = model(
        bi_inference=True, hidden_states=hidden, timestep=timestep, timestep_txt=timestep_txt,
        text_states=text_states, text_states_2=None, encoder_attention_mask=cond["prompt_mask"],
        vision_states=cond.get("vision_states") if task == "i2v" else None,
        viewmats=cond.get("viewmats"), Ks=cond.get("Ks"), action=cond.get("action"),  # camtext 下均 None
        extra_kwargs=extra, mask_type=task, return_dict=False,
        clean_x=clean_in, aug_timesteps=aug_timesteps,
    )
    return v


def _slice_cond_to_frames(cond, n_frames):
    """DMD block rollout 只喂 [历史+当前块], 逐帧相机条件也必须切到同样的时间长度。"""
    out = dict(cond)
    for key in ("action", "viewmats", "Ks"):
        v = out.get(key)
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] != n_frames:
            out[key] = v[:, :n_frames]
    return out


def _bcast_int(val, device):
    """rank0 的 int broadcast 到所有 rank (保证 grad_step 各 rank 一致)。"""
    if (not dist.is_initialized()) or dist.get_world_size() == 1:
        return val
    t = torch.tensor([int(val)], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


def _rand_sigma(B, device, shift, denoised_to=0.0, args=None):
    """critic/DMD 加噪用随机 sigma (shift 后)。
    对齐 Wan _sample_dmd_sigma — clamp 到 [s_min, s_max];
      --dmd_ts_schedule 开时下限抬到 generator 这步去噪到的 denoised_to (信号聚焦到该噪声档及以上)。"""
    s_min = float(getattr(args, "dmd_min_step", 0.0)) if args is not None else 0.0
    s_max = float(getattr(args, "dmd_max_step", 1.0)) if args is not None else 1.0
    if args is not None and getattr(args, "dmd_ts_schedule", False):
        s_min = max(s_min, float(denoised_to))
    u = torch.rand(B, device=device)
    s = shift_sigma(u, shift)
    return s.clamp(s_min, s_max)


def _band_sigma(B, device, lo, hi):
    """对齐 minWM —— renoise σ 在 generator 留梯度步窄带 [lo, hi] 内均匀采样
    (lo=sigmas[k_i+1], hi=sigmas[k_i]; 均为 shift 后 σ)。real/fake 在同一窄带比较 x0, score 匹配才准。"""
    u = torch.rand(B, device=device)
    return (lo + u * (hi - lo)).clamp(0.0, 1.0)


# =============================================================================
# 双向自回归 (block-AR) generator rollout —— 块内双向、块间因果(forward_bi clean_x + 4帧 block mask)
# =============================================================================
def generator_block_rollout(gen, noise_full, cond, concat_condition, args, device):
    """逐 block 自回归生成 x0_gen —— 重写, 对齐 Wan dmd_wan.py 双向 AR(none-HE):
      block b 只喂 [clean 历史(0:s0, σ=0) | 当前块(s0:s1, σ=sg)] (不喂整段未来零帧),
      【全双向 attention】(不传 clean_x → is_tf=False → 无 tf_block_mask), 只取当前块 velocity。
      - 历史 = 已生成的 x0 块(detached), σ=0 干净条件; 逐帧 timestep 历史=0、当前=sg。
      - RoPE 时间真实递增: x_in 序列长 s0+Kb, 当前块落绝对位置 s0..s1-1。
      - self-forcing: 每 block few-step, 仅【留梯度步 grad_step】留梯度并 break; 历史 detach 不跨 block 串图。
        ★ 对齐 dmd_wan.py generator_rollout: 训练期每 block 在随机 grad_step 早退, 该步 x0 同时作 loss 与历史
        (随机 grad_step → 模型见到各噪声档历史, Self-Forcing 鲁棒); 验证期(run_validation_dmd)走完整 4 步取净 x0。
    返回 (x0_gen:[B,C,Tfull,H,W] 逐 block 带梯度, sig_lo, sig_hi: renoise 窄带下/上限)。"""
    B, C, Tf, H, W = noise_full.shape
    K = args.dmd_block_K
    M = args.dmd_num_blocks if args.dmd_num_blocks > 0 else max(1, Tf // K)
    sigmas = [shift_sigma(s, args.dmd_timestep_shift) for s in args.dmd_denoising_sigmas]
    n_steps = len(sigmas)
    if getattr(args, "dmd_ts_schedule", False):
        grad_step = _bcast_int(random.randint(0, n_steps - 1), device)
    else:
        grad_step = n_steps - 1
    sched_next = sigmas[1:] + [0.0]
    sig_lo = float(sched_next[grad_step])                 # 窄带下限 = grad_step 去噪到的 σ (对齐 minWM)
    sig_hi = float(sigmas[grad_step])                     # 窄带上限 = grad_step 当前 σ
    generated = []                                        # 已生成的 clean x0 块(detached), 作历史
    x0_blocks = []
    # 【死锁根因修复】对齐 Wan dmd_wan.py:374 —— 留梯度必须跟随【外层】grad 开关。
    #   critic rollout 在 `with torch.no_grad()` 下调用本函数(line ~565), 若这里硬写 set_grad_enabled(exit_flag)
    #   会在 grad_step 处【覆盖外层 no_grad 重新开梯度】, 建出一张随后被 detach、永不 backward 的孤儿图 →
    #   FSDP 给 generator 注册了 post-backward(reduce-scatter) hook 却永不触发, handle 残留"待 backward"状态。
    #   平时无害, 但 checkpoint 存档(FSDP FULL_STATE_DICT rank0_only gather 只动 rank0 的 handle)后, 该残留与
    #   rank0 被单独改过的 handle 叠加 → 存档后首个真正 generator.backward() 在 rank0 不发 reduce-scatter,
    #   其余 rank 发了却永久等 rank0 → NCCL 死锁(确定性卡在 step=checkpointing_steps 首个 gen 更新)。
    #   修复: exit_flag and _outer_grad —— critic(外层 no_grad)恒不建图; generator(外层有 grad)才在 grad_step 留梯度。
    _outer_grad = torch.is_grad_enabled()
    for b in range(M):
        s0, s1 = b * K, min((b + 1) * K, Tf)
        Kb = s1 - s0
        prefix = torch.cat(generated, dim=2).detach() if generated else None  # [B,C,s0,H,W] 干净历史
        blk_noisy = noise_full[:, :, s0:s1]
        x0_blk = None
        for i, sg in enumerate(sigmas):
            exit_flag = (i == grad_step)
            with torch.set_grad_enabled(exit_flag and _outer_grad):   # 跟随外层(critic no_grad 恒不建图)
                # 只喂 [历史(σ=0) | 当前块(σ=sg)]; 全双向(无 clean_x/tf mask)
                if prefix is not None:
                    x_in = torch.cat([prefix, blk_noisy], dim=2)            # [B,C,s0+Kb,H,W]
                    sig_frame = torch.cat([torch.zeros(B, s0, device=device),
                                           torch.full((B, Kb), float(sg), device=device)], dim=1)
                else:
                    x_in = blk_noisy
                    sig_frame = torch.full((B, Kb), float(sg), device=device)
                cond_i = _slice_cond_to_frames(cond, x_in.shape[2])
                v_full = dit_velocity(gen, x_in, None, cond_i, concat_condition, args.training_mode,
                                      per_frame_sigma=sig_frame)            # clean_x=None → is_tf=False → 全双向
                v_blk = v_full[:, :, -Kb:]                                  # 只取当前块 velocity
                x0_blk = pred_x0_from_flow(blk_noisy, v_blk, torch.full((B, 1, 1, 1, 1), float(sg), device=device))
            if exit_flag:
                break
            sg_n = sigmas[i + 1]                                            # 桥接到下一步(detach)
            if getattr(args, "dmd_euler_rollout", False):
                blk_noisy = (blk_noisy + v_blk * (sg_n - sg)).detach()                       # euler 确定性
            else:
                blk_noisy = ((1 - sg_n) * x0_blk + sg_n * torch.randn_like(x0_blk)).detach()  # cm 重加噪
        generated.append(x0_blk.detach())                                  # 作下一 block 历史(detach, =grad_step x0, 对齐 dmd_wan.py)
        x0_blocks.append(x0_blk)
    return torch.cat(x0_blocks, dim=2), sig_lo, sig_hi


# =============================================================================
# SFT + forward-KL 锚定 (HY15 原生 dit_velocity 版, 算法对齐 Wan dmd_wan.py)
#   抗 DMD reverse-KL 缩模/塌缩, 保长视频与运动。real = 本步真实视频 latent (整段, 与动态 rollout 解耦)。
# =============================================================================
def _hy15_sft_loss(gen, gt_latent, cond, cc, args, device):
    """SFT: 在【完整真实视频 latent】上低σ flow-matching velocity MLE (强数据锚定, 低σ精修)。"""
    x0 = gt_latent                                          # [B,C,T,H,W] 完整真实视频
    B = x0.shape[0]
    s = random.uniform(float(getattr(args, "dmd_min_step", 0.0)), float(args.dmd_sft_sigma_max))
    ss = shift_sigma(s, args.dmd_timestep_shift)
    eps = torch.randn_like(x0)
    sv = torch.full((B, 1, 1, 1, 1), float(ss), device=device)
    x_t = (1.0 - sv) * x0 + sv * eps
    v = dit_velocity(gen, x_t, torch.full((B,), float(ss), device=device), cond, cc, args.training_mode)
    return F.mse_loss(v.float(), (eps - x0).float())        # velocity MLE


def _hy15_real_fkl_loss(gen, gt_latent, cond, cc, args, device):
    """real forward-KL: 高σ 把【完整真实视频】加噪 → student pred_x0 → MSE 回归真实 x0 (mass-covering 朝真实运动)。"""
    x0 = gt_latent
    B = x0.shape[0]
    s = random.uniform(float(args.dmd_real_fkl_sigma_min), float(args.dmd_real_fkl_sigma_max))
    ss = shift_sigma(s, args.dmd_timestep_shift)
    eps = torch.randn_like(x0)
    sv = torch.full((B, 1, 1, 1, 1), float(ss), device=device)
    x_t = (1.0 - sv) * x0 + sv * eps
    v = dit_velocity(gen, x_t, torch.full((B,), float(ss), device=device), cond, cc, args.training_mode)
    x0_pred = pred_x0_from_flow(x_t, v, sv)
    return F.mse_loss(x0_pred.float(), x0.float())


def _hy15_teacher_fkl_loss(gen, real_score, cond, cc, latent, args, device, dtype):
    """teacher forward-KL (HiAR PD, data-free): 沿教师(real_score+CFG)在线密集 ODE 轨迹取 save 点,
    弦外插得 teacher pred_x0, student 在同点回归之 (mass-covering 防 reverse-KL 缩模/低运动)。"""
    B, C, T, H, W = latent.shape
    shift = float(args.dmd_timestep_shift)
    cfg = float(args.real_guidance_scale)
    ts = max(4, int(args.dmd_fkl_teacher_steps))
    ts_list = [shift_sigma(1.0 - i / ts, shift) for i in range(ts + 1)]   # σ 1→0 (+shift)
    k = max(1, min(int(args.dmd_fkl_steps), ts))
    stride = max(1, ts // 4)
    save_steps = sorted(set(min(j * stride, ts) for j in range(k + 1)))
    save_set, max_step = set(save_steps), save_steps[-1]
    saves = {}
    lat = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    with torch.no_grad():                                   # 教师密集 ODE rollout (CFG)
        for i in range(max_step):
            if i in save_set:
                saves[i] = lat.detach()
            si = ts_list[i]
            sc = torch.full((B,), float(si), device=device)
            v_c = dit_velocity(real_score, lat, sc, cond, cc, args.training_mode, drop_text=False)
            v_u = dit_velocity(real_score, lat, sc, cond, cc, args.training_mode, drop_text=True)
            v = v_u + cfg * (v_c - v_u)
            lat = lat + (ts_list[i + 1] - si) * v
        saves[max_step] = lat.detach()
    loss = None
    for j in range(len(save_steps) - 1):                    # k 个 segment 的 x0 回归
        a, bb = save_steps[j], save_steps[j + 1]
        x_in, x_tgt = saves[a], saves[bb]
        s_in, s_tgt = ts_list[a], ts_list[bb]
        teacher_x0 = (x_in.double() - s_in * (x_tgt.double() - x_in.double()) / (s_tgt - s_in)).float()
        sc = torch.full((B,), float(s_in), device=device)
        v_s = dit_velocity(gen, x_in, sc, cond, cc, args.training_mode)
        x0_pred = x_in.float() - s_in * v_s.float()
        e = F.mse_loss(x0_pred, teacher_x0)
        loss = e if loss is None else loss + e
    return loss / (len(save_steps) - 1)


# =============================================================================
# 验证: 逐 block 自回归采样 (全局共享 caption + 每窗口自己的 cam-text → 看相机随窗口变化)
# =============================================================================
@torch.no_grad()
def run_validation_dmd(gen, vae, mllm, val_sample, args, concat_condition, step, device, rank):
    """generator few-step 逐 block rollout 采样:
      - 文本: 全局共享 caption, 但每个【窗口/block】拼自己的 cam-text(window_actions[b]) → 相机随新窗口而变;
      - 块内双向、块间因果(clean 前缀 + forward_bi); 与训练 rollout 同结构, 但 no_grad + 逐块换文本。
    存 mp4(逐窗口动作叠摇杆 overlay)。"""
    import numpy as np
    gen.eval()
    task = args.training_mode
    B, C, Tf, h, w = val_sample["latent"].shape
    K = args.dmd_block_K
    M = args.dmd_num_blocks if args.dmd_num_blocks > 0 else max(1, Tf // K)
    caption = str(val_sample.get("caption", "")).rstrip()

    # 每个窗口一个相机动作(demo: 展示相机随窗口切换); 各 rank 起点不同 → 各 rank 一条不同的运镜序列
    window_actions = [_VAL_ACTIONS[(rank + b) % len(_VAL_ACTIONS)] for b in range(M)]
    if args.camera_mode == "action":
        block_text = [mllm_encode(mllm, [caption], device) for _ in range(M)]
        full_action = torch.zeros(1, Tf, device=device, dtype=torch.long)
        for b, a in enumerate(window_actions):
            s0, s1 = b * K, min((b + 1) * K, Tf)
            full_action[:, s0:s1] = int(a)
    else:
        # camtext 模式: 每个窗口重编 <camera>相机</camera> + 静态caption。
        block_text = []
        full_action = None
        for a in window_actions:
            ft = build_camtext_prompt(action_to_camtext(a), caption)
            block_text.append(mllm_encode(mllm, [ft], device))

    # t2v 占位条件(同 make_live_batch): byt5/vision/image_cond 喂零
    byt5 = torch.zeros(1, BYT5_MAX_LEN, BYT5_DIM, device=device, dtype=torch.bfloat16)
    byt5_m = torch.zeros(1, BYT5_MAX_LEN, device=device, dtype=torch.bfloat16)
    vis = torch.zeros(1, 1, 1, device=device, dtype=torch.bfloat16)
    img_cond = torch.zeros(1, C, 1, h, w, device=device, dtype=torch.bfloat16)

    sigmas = [shift_sigma(s, args.dmd_timestep_shift) for s in args.dmd_denoising_sigmas]
    g = torch.Generator(device=device).manual_seed(42 + rank)
    noise_full = torch.randn(val_sample["latent"].shape, generator=g, device=device, dtype=torch.bfloat16)
    # 与训练 rollout 一致 —— 每 block 只喂 [clean历史(σ=0) | 当前块(σ=sg)], 全双向(无 clean_x/tf mask)。
    generated = []
    for b in range(M):
        s0, s1 = b * K, min((b + 1) * K, Tf)
        Kb = s1 - s0
        pe, pm = block_text[b]
        cond_b = dict(prompt_embed=pe, prompt_mask=pm, byt5_text_states=byt5, byt5_text_mask=byt5_m,
                      vision_states=vis, image_cond=img_cond, viewmats=None, Ks=None,
                      action=full_action[:, :s1] if full_action is not None else None)
        prefix = torch.cat(generated, dim=2) if generated else None       # [B,C,s0,H,W] 干净历史
        blk_noisy = noise_full[:, :, s0:s1]
        x0_blk = None
        for i, sg in enumerate(sigmas):
            if prefix is not None:
                x_in = torch.cat([prefix, blk_noisy], dim=2)
                sig_frame = torch.cat([torch.zeros(B, s0, device=device),
                                       torch.full((B, Kb), float(sg), device=device)], dim=1)
            else:
                x_in = blk_noisy
                sig_frame = torch.full((B, Kb), float(sg), device=device)
            v_full = dit_velocity(gen, x_in, None, cond_b, concat_condition, task, per_frame_sigma=sig_frame)
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
    clean = torch.cat(generated, dim=2)                               # [B,C,Tfull,H,W]

    vid = vae.decode((clean / vae._sf).to(next(vae.parameters()).dtype)).sample   # [B,3,T,H,W]
    frames = _frames_to_uint8(vid)                            # list[np.uint8 HWC] (无 overlay)
    d = os.path.join(args.output_dir, "validation"); os.makedirs(d, exist_ok=True)
    fps = int(getattr(args, "fps", 24))
    # ★ 保存格式/类型对齐 Wan/stage1: torchvision write_video crf18; 命名 validation_/joystick_{prefix}
    prefix = f"step_{step:07d}_{task}_camtext_video_real_rank_{rank:03d}_dmd"
    save_video(frames, os.path.join(d, f"validation_{prefix}.mp4"), fps=fps)   # 生成(无overlay)
    try:
        from pipelines.common.control_overlay import superimpose_control_video as add_joystick_overlay
        per_lat_act = torch.tensor([window_actions[min(t // K, M - 1)] for t in range(Tf)], dtype=torch.long)
        frames_j = add_joystick_overlay(frames, per_lat_act, vae_temporal_stride=HY15_VAE_TEMPORAL_STRIDE)
        save_video(frames_j, os.path.join(d, f"joystick_{prefix}.mp4"), fps=fps)  # 叠摇杆(逐窗动作)
    except Exception as e:
        mprint(f"[DMD][val] joystick overlay 跳过: {type(e).__name__}: {e}")
    # euler 读出已【移除】—— 它在 FSDP 里多跑 30 次 forward + 包 try/except,
    #   某 rank OOM 进 except 跳出会和其余 rank 的 all-gather 不匹配 → NCCL 超时崩(step50 首个 gen+val 撞一起实测)。
    #   euler 质量监控改用独立单卡 test_sample_hy15.py(无 FSDP, 安全)。验证只保留 cm(少步学生 deliverable)。
    try:
        with open(os.path.join(d, f"validation_{prefix}_prompt.txt"), "w", encoding="utf-8") as mf:
            mf.write(f"Step: {step}\nRank: {rank}\nTask: {task}\nSource: video_real\n")
            mf.write(f"AR blocks(M): {M}  block_K: {K}\nWindow actions(逐窗相机): {window_actions}\n")
            mf.write(f"Caption: {caption}\nSampling sigmas: {args.dmd_denoising_sigmas}\nFPS: {fps}\n")
    except Exception as e:
        mprint(f"[DMD][val] metadata 跳过: {type(e).__name__}: {e}")
    if is_main():
        mprint(f"[DMD][val] step {step}: AR rollout {M}块, 窗口动作={window_actions} -> {d}/validation_{prefix}.mp4")


# =============================================================================
# main
# =============================================================================
def main(args):
    local_rank, rank, world = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    dtype = torch.bfloat16
    task = args.training_mode

    # ---- 三模型 (camtext: 不加 prope/action, 全参基座; 相机走文本) ----
    _gc = bool(getattr(args, "gradient_checkpointing", False))
    generator, cc = build_dit_from_ckpt(args.generator_ckpt, args.pretrained_model_path,
                                        device, dtype, args.use_discrete_action, trainable=True,
                                        camera_mode=args.camera_mode, grad_ckpt=_gc)
    # critic 必须开 grad_ckpt —— 实测关掉后 critic 阶段 OOM(单次 fwd 存满 54 层激活~40GB
    #   + generator 保留的 rollout 图, 140GB H200 也放不下)。critic 慢是 6×8.5B fwd+bwd 固有, 提速靠降
    #   --dfake_gen_update_ratio 或 CP, 不能靠关 grad_ckpt。
    fake_score, _ = build_dit_from_ckpt(args.fake_score_ckpt or args.generator_ckpt,
                                        args.pretrained_model_path, device, dtype,
                                        args.use_discrete_action, trainable=True,
                                        camera_mode=args.camera_mode, grad_ckpt=_gc)
    real_score, _ = build_dit_from_ckpt(args.real_score_ckpt or args.generator_ckpt,
                                        args.pretrained_model_path, device, dtype,
                                        args.use_discrete_action, trainable=False,
                                        camera_mode=args.camera_mode)
    real_score.eval()

    # ---- FSDP (各自包) ----
    import functools
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, MixedPrecision,
                                        ShardingStrategy)
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    pol = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={MMDoubleStreamBlock})
    mp_pol = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
    def wrap(m):
        return FSDP(m, auto_wrap_policy=pol, mixed_precision=mp_pol,
                    sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=device,
                    use_orig_params=True, sync_module_states=True)
    if world > 1:
        generator, fake_score, real_score = wrap(generator), wrap(fake_score), wrap(real_score)

    gen_opt = torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad],
                                lr=args.dmd_generator_lr, weight_decay=args.weight_decay)
    critic_opt = torch.optim.AdamW([p for p in fake_score.parameters() if p.requires_grad],
                                   lr=args.dmd_critic_lr, weight_decay=args.weight_decay)

    # ---- 稳定化开关日志 ----
    _critic_warmup = int(getattr(args, "dmd_critic_warmup_steps", 0))
    _use_sft = float(getattr(args, "dmd_sft_weight", 0.0)) > 0.0
    _use_fkl = float(getattr(args, "dmd_fkl_weight", 0.0)) > 0.0
    _use_rfkl = float(getattr(args, "dmd_real_fkl_weight", 0.0)) > 0.0
    mprint(f"[DMD] 稳定化: critic_warmup={_critic_warmup} ts_schedule={getattr(args,'dmd_ts_schedule',False)} "
           f"SFT={'开' if _use_sft else '关'} "
           f"teacherFKL={'开' if _use_fkl else '关'} realFKL={'开' if _use_rfkl else '关'}")

    # ---- 数据 ----
    #   live(默认, 对齐 stage1): video_real mp4 → 在线 VAE + MLLM(Qwen2.5-VL) 编码; 训练 caption-only(不引入相机)。
    #   preencoded(旧): 读 train_index.json 预编码 .pt。
    vae = mllm = None
    if args.data_mode == "live":
        from pipelines.dataset.biwm_camera_text_dataset import BiwmCamCaptionData
        from torch.utils.data import DataLoader
        dataset = BiwmCamCaptionData(
            video_dir=args.biwm_video_dir, caption_json=args.biwm_caption_json,
            width=args.num_width, height=args.num_height, num_frames=args.num_frames,
            vae_temporal_factor=HY15_VAE_TEMPORAL_STRIDE)
        loader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=args.dataloader_num_workers, collate_fn=_live_collate)
        vae = load_hy15_vae(args.vae_path, device)
        llm_dir = args.text_encoder_path or os.path.join(
            os.path.dirname(args.vae_path.rstrip("/")), "text_encoder", "llm")
        mllm = load_mllm(llm_dir, device)
    else:
        dataset, loader = build_camera_plucker_dataloader(
            json_path=args.json_path, causal=False, window_frames=args.window_frames,
            batch_size=args.train_batch_size, num_data_workers=args.dataloader_num_workers,
            drop_last=True, drop_first_row=False, seed=args.seed, cfg_rate=0.0,
            i2v_rate=args.i2v_rate, task_type=task)

    denoise_sigmas = [shift_sigma(s, args.dmd_timestep_shift) for s in args.dmd_denoising_sigmas]
    _cond_desc = "caption-only + 逐帧 action" if args.camera_mode == "action" else "camtext: 相机+caption"
    mprint(f"[DMD] start: data_mode={args.data_mode}, camera_mode={args.camera_mode}(训练 {_cond_desc}), "
           f"gen few-step sigmas(shift前)={args.dmd_denoising_sigmas} "
           f"→(shift后){[round(s,3) for s in denoise_sigmas]}, "
           f"update_ratio={args.dfake_gen_update_ratio}, world={world}")
    os.makedirs(args.output_dir, exist_ok=True)

    def get_cond_preenc(batch):
        return dict(
            prompt_embed=batch["prompt_embed"].to(device, dtype),
            prompt_mask=batch["prompt_mask"].to(device, dtype),
            byt5_text_states=batch["byt5_text_states"].to(device, dtype),
            byt5_text_mask=batch["byt5_text_mask"].to(device, dtype),
            vision_states=batch["vision_states"].to(device, dtype),
            image_cond=batch["image_cond"].to(device, dtype),
            viewmats=batch["viewmats"].to(device, torch.float32),
            Ks=batch["Ks"].to(device, torch.float32),
            action=batch["action"].to(device).long() if args.use_discrete_action else None,
        )

    def get_cond_live(b):
        """live: b 是 make_live_batch 返回的 dict。
        action 模式: 文本 caption-only, 相机通过 b["action"] 逐帧注入; camtext 模式: 相机拼进文本。"""
        return dict(
            prompt_embed=b["prompt_embed"], prompt_mask=b["prompt_mask"],
            byt5_text_states=b["byt5_text_states"], byt5_text_mask=b["byt5_text_mask"],
            vision_states=b["vision_states"], image_cond=b["image_cond"],
            viewmats=b.get("viewmats"), Ks=b.get("Ks"),
            action=b.get("action") if args.use_discrete_action else None,
        )

    def _fetch_live(it):
        while True:
            try:
                raw = next(it)
            except StopIteration:
                it = iter(loader); raw = next(it)
            if ((isinstance(raw, dict) and raw.get("skip")) or
                    (isinstance(raw, tuple) and raw and raw[0] == "skip")):
                continue
            # action 模式必须保持文本纯 caption; 相机只走逐帧离散 action 注入。
            return make_live_batch(raw, vae, mllm, args, device,
                                   caption_only=(args.camera_mode == "action")), it

    val_sample = None       # 缓存首条(含 caption/clip_action), 验证用
    # 预编码 CFG 负向 = 空串编码(=stage1/minWM), 供 real_score uncond 用(非全零)。
    _neg_pe = _neg_pm = None
    if mllm is not None:
        _neg_pe, _neg_pm = encode_neg_prompt(mllm, device)
    step, it = 0, iter(loader)
    _ratio = max(1, int(args.dfake_gen_update_ratio))   # critic 每步更新; gen 每 _ratio 步更新(对齐 Wan)
    _gen_upd = 0                                         # gen 更新计数(供梯度累积窗口)
    # 分阶段心跳(前5步逐步打印, 之后按 log_interval) —— 定位卡在哪一阶段(取数据/rollout/critic/gen)。
    _hb = lambda tag, t: mprint(f"[DMD][hb] step {step} {tag} +{time.perf_counter()-t:.1f}s") \
        if is_main() and (step < 5 or step % args.log_interval == 0) else None
    while step < args.max_train_steps:
        _t = time.perf_counter()
        if is_main() and (step < 5 or step % args.log_interval == 0):
            mprint(f"[DMD][hb] step {step} 进入循环, 取数据...")
        if args.data_mode == "live":
            batch, it = _fetch_live(it)
            if val_sample is None:
                val_sample = {"latent": batch["latent"].detach(),
                              "caption": batch.get("caption", ""),
                              "image_cond": batch["image_cond"].detach()}
        else:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader); batch = next(it)
        cond = get_cond_live(batch) if args.data_mode == "live" else get_cond_preenc(batch)
        latent = batch["latent"].to(device, dtype)
        B = latent.shape[0]

        _accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        # ===== 1) critic(fake_score) 更新【每步 1 次】—— 用自己的 fresh rollout(no_grad), 对齐 Wan/minWM。
        #   (原写法: 单次 rollout 上 critic 连更 5 次 = 同一样本重复喂 critic, 错; 现每步 1 次 fresh 数据。)
        _hb("取数据完成, 开始 critic rollout", _t)
        critic_opt.zero_grad()
        with torch.no_grad():
            x0_c, _lo_c, _hi_c = generator_block_rollout(generator, torch.randn_like(latent), cond, cc, args, device)
        x0_c = x0_c.detach()
        sig_c = _band_sigma(B, device, _lo_c, _hi_c)
        noise_c = torch.randn_like(x0_c)
        noisy_c = (1 - sig_c.view(B, 1, 1, 1, 1)) * x0_c + sig_c.view(B, 1, 1, 1, 1) * noise_c
        v_fake = dit_velocity(fake_score, noisy_c, sig_c, cond, cc, task)
        c_loss = dmd_critic_loss(v_fake, noise_c, x0_c)
        c_loss.backward()
        (fake_score.clip_grad_norm_(args.max_grad_norm) if hasattr(fake_score, "clip_grad_norm_")
         else torch.nn.utils.clip_grad_norm_(fake_score.parameters(), args.max_grad_norm))
        critic_opt.step()
        _hb("critic 完成", _t)

        # ===== 2) generator 更新【每 _ratio 步 1 次, warmup 后】—— 自己的 fresh rollout(带梯度), 对齐 Wan。
        g_loss = None
        _l_sft = _l_fkl = _l_rfkl = float("nan")
        _is_gen_step = (step >= _critic_warmup) and (step % _ratio == 0)
        if _is_gen_step:
            _accum_first = (_gen_upd % _accum == 0)
            _accum_last = (_gen_upd % _accum == _accum - 1)
            if _accum_first:
                gen_opt.zero_grad()
            x0_gen, _sig_lo, _sig_hi = generator_block_rollout(generator, torch.randn_like(latent), cond, cc, args, device)
            sig_g = _band_sigma(B, device, _sig_lo, _sig_hi)
            noise_g = torch.randn_like(x0_gen)
            sgv = sig_g.view(B, 1, 1, 1, 1)
            noisy_g = (1 - sgv) * x0_gen + sgv * noise_g
            # CFG uncond cond —— 用【空串编码 neg-prompt】替换文本(=stage1/minWM), 非全零。
            if _neg_pe is not None:
                _neg_cond = dict(cond)
                _neg_cond["prompt_embed"] = _neg_pe.to(device, x0_gen.dtype).expand(B, -1, -1)
                _neg_cond["prompt_mask"] = _neg_pm.to(device).expand(B, -1)
                _u_kw = {}
            else:                                              # 无 mllm 兜底: 退回全零(drop_text)
                _neg_cond, _u_kw = cond, {"drop_text": True}
            with torch.no_grad():
                v_real_c = dit_velocity(real_score, noisy_g, sig_g, cond, cc, task, drop_text=False)
                v_real_u = dit_velocity(real_score, noisy_g, sig_g, _neg_cond, cc, task, **_u_kw)
                v_real = apply_cfg(v_real_c, v_real_u, args.real_guidance_scale)
                real_x0 = pred_x0_from_flow(noisy_g, v_real, sgv)
                v_fake = dit_velocity(fake_score, noisy_g, sig_g, cond, cc, task)
                fake_x0 = pred_x0_from_flow(noisy_g, v_fake, sgv)
                kl_grad = compute_kl_gradient(fake_x0, real_x0, x0_gen.detach(), normalize_mode="global")
            g_loss = dmd_generator_loss(x0_gen, kl_grad)
            # --- SFT / forward-KL 锚定 (用完整真实视频 latent, 与动态 rollout 解耦) ---
            if _use_sft:
                _sft = _hy15_sft_loss(generator, latent, cond, cc, args, device)
                g_loss = g_loss + float(args.dmd_sft_weight) * _sft.to(g_loss.dtype)
                _l_sft = float(_sft.detach())
            if _use_rfkl:
                _rfkl = _hy15_real_fkl_loss(generator, latent, cond, cc, args, device)
                g_loss = g_loss + float(args.dmd_real_fkl_weight) * _rfkl.to(g_loss.dtype)
                _l_rfkl = float(_rfkl.detach())
            if _use_fkl:
                _fkl = _hy15_teacher_fkl_loss(generator, real_score, cond, cc, latent, args, device, dtype)
                g_loss = g_loss + float(args.dmd_fkl_weight) * _fkl.to(g_loss.dtype)
                _l_fkl = float(_fkl.detach())
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
            _ex = ""
            if _use_sft:  _ex += f" sft={_l_sft:.4f}"
            if _use_fkl:  _ex += f" fkl={_l_fkl:.4f}"
            if _use_rfkl: _ex += f" rfkl={_l_rfkl:.4f}"
            mprint(f"step {step}/{args.max_train_steps}  {_gmsg}  c_loss={c_loss.item():.4f}{_ex}")
        # ---- 验证: 逐 block 自回归采样, 全局共享 caption + 每窗口自己的 cam-text ----
        # first_validation_step 那一步必触发(即使不是 validation_interval 倍数) ——
        #   原来只 step%interval==0 时验证, 导致 first_validation_step=1 形同虚设(首验落到 step=interval)。
        if (args.validation_interval > 0 and args.data_mode == "live" and vae is not None
                and val_sample is not None and step >= args.first_validation_step
                and (step % args.validation_interval == 0 or step == args.first_validation_step)):
            try:
                run_validation_dmd(generator, vae, mllm, val_sample, args, cc, step, device, rank)
            except Exception as e:
                mprint(f"[DMD][val] step {step} 验证失败(跳过): {type(e).__name__}: {e}")
            generator.train()
            # 验证后 barrier —— 各 rank 重新对齐, 防 no_grad 推理时 FSDP unshard 的
            #   残留状态串进下一步训练 collective(对齐 Wan hrnet_pretrain_wan 验证/存档后 dist.barrier 的写法)。
            if dist.is_initialized():
                dist.barrier()
        if step % args.checkpointing_steps == 0:
            # 【死锁修复】checkpoint 存档(FSDP FULL_STATE_DICT, rank0_only+offload_to_cpu)
            #   会临时 unshard generator 参数; rank0 因 CPU offload + 落盘比其余 rank 慢, 无 barrier 时 rank0 会在
            #   reshard 尚未完成时进入下一步 generator forward → rank0 不注册 post-backward reduce-scatter hook →
            #   下一步 generator.backward() rank0 无梯度、不发 reduce-scatter, 其余 rank 发了却永久等 rank0 → NCCL 死锁
            #   (实测卡在 step=checkpointing_steps 存档后的首个 generator 更新, 伴随 "clip_grad_norm_ on rank0 with no gradients")。
            #   修复: 存档前后 empty_cache + 存档后 dist.barrier 强制所有 rank reshard 完成再继续(各 rank hook 注册一致)。
            torch.cuda.empty_cache()
            _save(generator, args.output_dir, step)
            if dist.is_initialized():
                dist.barrier()
            torch.cuda.empty_cache()
    _save(generator, args.output_dir, args.max_train_steps)
    if dist.is_initialized():
        dist.destroy_process_group()


def _save(model, out_dir, step):
    """对齐 Wan stage2(hrnet_pretrain_wan 用 save_checkpoint)/stage1: 存
      checkpoint-{step}/diffusion_pytorch_model.safetensors + config.json  (替代 generator-{step}/model.pt)
    与 base/stage1 同构, 可直接 --generator_ckpt/--pretrained_model_path 加载续训或推理。"""
    import json as _json
    import safetensors.torch as st
    try:
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                            FullStateDictConfig, StateDictType)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            sd = model.state_dict()
    except Exception:
        sd = model.state_dict()
    if is_main():
        d = os.path.join(out_dir, f"checkpoint-{step}"); os.makedirs(d, exist_ok=True)
        st.save_file({k: v.contiguous() for k, v in sd.items()},
                     os.path.join(d, "diffusion_pytorch_model.safetensors"))
        m = model
        for attr in ("_fsdp_wrapped_module", "_checkpoint_wrapped_module", "module"):
            m = getattr(m, attr, m)
        bcfg = getattr(model, "_biwm_config", None) or getattr(m, "_biwm_config", None)
        if bcfg is not None:
            try:
                with open(os.path.join(d, "config.json"), "w") as f:
                    _json.dump({k: v for k, v in bcfg.items()
                                if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
                               f, indent=4)
            except Exception as e:
                mprint(f"[DMD] config.json 保存失败(跳过): {type(e).__name__}: {e}")
        mprint(f"[DMD] saved generator (safetensors+config) -> {d}")


def parse_args():
    p = argparse.ArgumentParser("HY15 stage2 DMD distillation")
    p.add_argument("--json_path", type=str, required=False)
    # 命名与 Wan stage2_dmd.sh 对齐(统一训练框架, 不能一模型一组参数)。
    p.add_argument("--training_mode", choices=["t2v", "i2v"], default="t2v")  # was --task_type
    p.add_argument("--window_frames", type=int, default=20)
    p.add_argument("--i2v_rate", type=float, default=0.0)
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    # 全量刷新到 camtext/live(对齐 stage1)
    p.add_argument("--data_mode", choices=["live", "preencoded"], default="live",
                   help="live=video_real mp4 在线 VAE+MLLM(训练 caption-only); preencoded=旧 train_index.json")
    p.add_argument("--camera_mode", choices=["camtext", "action", "prope"], default="camtext",
                   help="camtext=相机走文本; action=逐帧离散action注入AdaLN且文本纯caption; prope=旧 viewmats/Ks+离散action")
    p.add_argument("--biwm_video_dir", type=str, default="", help="live: video_real 目录")
    p.add_argument("--biwm_caption_json", type=str, default="", help="live: video_real_input.json")
    p.add_argument("--vae_path", type=str, default="", help="live: HY15 VAE 目录")
    p.add_argument("--text_encoder_path", type=str, default="", help="live: MLLM(Qwen2.5-VL) 目录")
    p.add_argument("--num_frames", type=int, default=77)
    p.add_argument("--num_height", type=int, default=480)
    p.add_argument("--num_width", type=int, default=832)
    p.add_argument("--training_cfg_rate", type=float, default=0.0)   # make_live_batch 需要(DMD 不丢文本)
    p.add_argument("--sigma_shift", type=float, default=5.0)         # 占位(make_live_batch/验证调度器对齐)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--validation_interval", type=int, default=100, help="每 N 步验证一次; 0=关")
    p.add_argument("--first_validation_step", type=int, default=100,
                   help="首次验证最早步; 须 <= validation_interval")
    # weights
    p.add_argument("--pretrained_model_path", type=str, default="", help="DiT config/base 目录")
    p.add_argument("--generator_ckpt", type=str, default="", help="stage1 ckpt (model.pt 或目录)")
    p.add_argument("--fake_score_ckpt", type=str, default="")
    p.add_argument("--real_score_ckpt", type=str, default="")
    p.add_argument("--use_discrete_action", action="store_true", default=True)
    p.add_argument("--no_discrete_action", dest="use_discrete_action", action="store_false")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                   help="开激活重算(generator/fake_score): forward_bi checkpoint double_blocks, 省显存修 OOM(对齐 stage1)")
    # dmd (命名对齐 Wan: --dmd_denoising_sigmas / --dmd_block_K / --dmd_num_blocks / --dmd_timestep_shift /
    #      --dfake_gen_update_ratio / --real_guidance_scale)
    p.add_argument("--dmd_denoising_sigmas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25],
                   help="generator 每 block few-step 去噪的 sigma 列表(shift 前, [0,1]); 末尾隐含 0.0 = 4-step")
    p.add_argument("--dmd_block_K", type=int, default=4,
                   help="自回归 block 粒度(latent 帧/块); 须与 model.forward_bi teacher-forcing mask 一致(hardcode=4)")
    p.add_argument("--dmd_num_blocks", type=int, default=0,
                   help="block 数(自回归块数); 0=按 latent 长度自动 = T_lat//dmd_block_K")
    p.add_argument("--dmd_timestep_shift", type=float, default=5.0)  # was --dmd_time_shift
    p.add_argument("--dfake_gen_update_ratio", type=int, default=5,  # was --dmd_update_ratio
                   help="critic 更新次数 / generator 1 次")
    p.add_argument("--real_guidance_scale", type=float, default=6.0,  # was --dmd_guidance_scale
                   help="real_score CFG")
    # optim (命名对齐 Wan: --dmd_generator_lr / --dmd_critic_lr)
    p.add_argument("--dmd_generator_lr", type=float, default=2e-6)
    p.add_argument("--dmd_critic_lr", type=float, default=4e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="generator 梯度累积步数(对齐 Wan); =1 不累积。critic/disc 仍每轮 step, 仅 generator 在窗口内累积")
    p.add_argument("--max_train_steps", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default="./logs/hy15/stage2_dmd")
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # ===== 与 Wan stage2_dmd.sh / dmd_wan.py 对齐的稳定化机制 (默认全关, sh 可逐项开) =====
    # critic 热身: 前 N 步只更新 critic, generator 暂不更新 (让 critic 先学准 generator 分布再开 KL 梯度)
    p.add_argument("--dmd_critic_warmup_steps", type=int, default=0,
                   help="前 N 步只训 critic, 之后再开 generator 更新 (0=关)")
    # ts_schedule: critic/DMD 加噪 sigma 下限 = generator 这步去噪到的 denoised_to (信号聚焦到该噪声档及以上)
    p.add_argument("--dmd_ts_schedule", action="store_true", default=False,
                   help="开 ts_schedule: rollout 随机留梯度步(Self-Forcing) + critic/DMD 加噪下限=denoised_to")
    p.add_argument("--dmd_euler_rollout", action="store_true", default=False,
                   help="rollout 桥接用 euler 确定性步(x+=v·dt)而非 cm 重加噪 —— 临时关 cm(诊断 cm 是否拖累训练)")
    p.add_argument("--dmd_min_step", type=float, default=0.0, help="critic/DMD 加噪 sigma 下限(0~1, shift前语义同Wan)")
    p.add_argument("--dmd_max_step", type=float, default=1.0, help="critic/DMD 加噪 sigma 上限(0~1)")
    # SFT (完整真实视频低σ velocity MLE, 强数据锚定抗塌缩)
    p.add_argument("--dmd_sft_weight", type=float, default=0.0, help="SFT 权重, 0=关")
    p.add_argument("--dmd_sft_sigma_max", type=float, default=0.5, help="SFT 仅在低σ∈shift(U[0,此值])施加")
    # teacher forward-KL (HiAR PD, data-free 沿教师密集ODE轨迹回归, mass-covering 防缩模/低运动)
    p.add_argument("--dmd_fkl_weight", type=float, default=0.0, help="教师 forward-KL 权重, 0=关")
    p.add_argument("--dmd_fkl_steps", type=int, default=1, help="forward-KL segment 数 k")
    p.add_argument("--dmd_fkl_teacher_steps", type=int, default=12, help="教师在线密集 ODE 步数")
    # real forward-KL (高σ 把真实视频回归 x0, 朝真实运动)
    p.add_argument("--dmd_real_fkl_weight", type=float, default=0.0, help="真实数据 forward-KL 权重, 0=关")
    p.add_argument("--dmd_real_fkl_sigma_min", type=float, default=0.25, help="real-FKL 高σ下限")
    p.add_argument("--dmd_real_fkl_sigma_max", type=float, default=0.65, help="real-FKL 高σ上限")
    return p.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main(parse_args())
