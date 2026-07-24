#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import torch
import torch.nn.functional as F
import torch.distributed as dist

WAN_PATCH_H, WAN_PATCH_W = 2, 2


def register_dmd_args(parser):
    g = parser.add_argument_group("dmd")
    g.add_argument("--dmd_distill", action="store_true", default=False,
                   help="enable Stage3 DMD distillation")
    g.add_argument("--generator_ckpt", type=str, default=None,
                   help="generator init ckpt (Stage2 checkpoint-N directory, with LoRA + history_encoder)")
    g.add_argument("--real_score_ckpt", type=str, default=None,
                   help="real/fake (critic) base weights safetensors (teacher). If not given, use pretrained_model_path")
    g.add_argument("--dmd_denoising_sigmas", type=float, nargs="+",
                   default=[1.0, 0.75, 0.5, 0.25],
                   help="list of denoising sigma start points per generator block (trailing 0.0 implied). 4 entries = 4-step")
    g.add_argument("--dmd_num_blocks", type=int, default=4,
                   help="autoregressive block count M (under dynamic DMD acts as the **maximum** block count: VRAM/frame budget/validation length computed from it)")
    g.add_argument("--dmd_block_K", type=int, default=16, help="latent frame count K per block")
    # dynamic DMD — random total block count M per training step (history = M-1 blocks)
    g.add_argument("--dmd_block_counts", type=int, nargs="+", default=None,
                   help="dynamic DMD total block-count candidates (e.g. 1 2 3 4 5). None=fixed use dmd_num_blocks (old behavior)")
    g.add_argument("--dmd_block_count_probs", type=float, nargs="+", default=None,
                   help="sampling probability for each candidate block count (same length as dmd_block_counts, auto-normalized internally). "
                        "biased toward large M = more compute on the longest history")
    # block-count curriculum (warmup): for the first N steps fix cold start, then sample by block counts
    g.add_argument("--dmd_block_warmup_steps", type=int, default=0,
                   help="for the first N steps fix single-block (or dmd_block_warmup_count blocks) cold start, only after that sample by normal block count. 0=off")
    g.add_argument("--dmd_block_warmup_count", type=int, default=1,
                   help="block count fixed during warmup (default 1 = pure t2v no-history cold start)")
    # DMD timestep sampling (teacher noising)
    g.add_argument("--dmd_min_step", type=int, default=20)
    g.add_argument("--dmd_max_step", type=int, default=980)
    g.add_argument("--dmd_timestep_shift", type=float, default=5.0)
    g.add_argument("--real_guidance_scale", type=float, default=5.0,
                   help="real_score CFG guidance strength")
    g.add_argument("--dfake_gen_update_ratio", type=int, default=5,
                   help="update critic N times, update generator once")
    g.add_argument("--dmd_generator_lr", type=float, default=2e-6)
    g.add_argument("--dmd_critic_lr", type=float, default=4e-6)
    g.add_argument("--gen_train_qkv_only", action="store_true", default=False,
                   help="[compat old flag] = --train_modules qkv. generator DiT only trains attention QKV (self+cross)")
    # generator trainable-module whitelist; takes priority over gen_train_qkv_only
    g.add_argument("--train_modules", type=str, nargs="+", default=None,
                   choices=["qkv", "patch_embedding", "history_encoder", "all"],
                   help="generator trainable-module whitelist (multiple allowed): qkv / patch_embedding / history_encoder / all")
    g.add_argument("--dmd_ts_schedule", action="store_true", default=False,
                   help="enable ts_schedule: lower bound of DMD/critic noising sigma = denoised_to the generator reached (signal focusing)")
    g.add_argument("--dmd_critic_warmup_steps", type=int, default=0,
                   help="for the first N steps only train the critic, not the generator (DMD signal warmup, 0=off)")
    # optional GT-latent regression anchor (opt-in, default off)
    g.add_argument("--dmd_use_gt_reg", action="store_true", default=False, help="DMD loss adds +w·MSE(x0_gen, gt_latent)")
    g.add_argument("--dmd_gt_reg_weight", type=float, default=0.1, help="GT-latent regression weight (best uses 0.1)")
    # SFT + forward-KL anchoring: resists DMD reverse-KL mode-shrinkage/collapse, preserves long video and motion
    g.add_argument("--dmd_sft_weight", type=float, default=0.0, help="SFT (full real-video low-σ velocity MLE) weight, 0=off")
    g.add_argument("--dmd_sft_sigma_max", type=float, default=0.5, help="SFT applied only at σ∈shift(U[sigma_min,this value]) (low-σ refinement)")
    g.add_argument("--dmd_fkl_weight", type=float, default=0.0, help="teacher forward-KL (HiAR PD) weight, 0=off")
    g.add_argument("--dmd_fkl_steps", type=int, default=1, help="number of segments k for forward-KL (paper=1)")
    g.add_argument("--dmd_fkl_teacher_steps", type=int, default=12, help="teacher online dense ODE step count (paper 48, take small e.g. 12 online)")
    g.add_argument("--dmd_real_fkl_weight", type=float, default=0.0, help="real-data forward-KL (high-σ regression toward real x0) weight, 0=off")
    g.add_argument("--dmd_real_fkl_sigma_min", type=float, default=0.25, help="real-FKL high-σ lower bound")
    g.add_argument("--dmd_real_fkl_sigma_max", type=float, default=0.65, help="real-FKL high-σ upper bound")
    # history compression mechanism (none=prepend clean latents as cond frames; framepack=compress to mem tokens)
    g.add_argument("--dmd_history_mode", type=str, default=None,
                   choices=["none", "framepack"],
                   help="DMD multi-block history compression mechanism: none/framepack")
    return parser


def velocity_into_x0(v, x_t, sigma):
    """flow matching: x0 = x_t - σ·v.  sigma scalar or [B] broadcast."""
    if torch.is_tensor(sigma) and sigma.dim() > 0:
        sigma = sigma.view(-1, *([1] * (x_t.dim() - 1)))
    return x_t - sigma * v


def _draw_dmd_sigma(args, B, device, dtype, denoised_to=0.0):
    """random sigma for teacher noising, broadcast from rank0.
    Uniform over [0,1] -> timestep_shift -> clamp [s_min, s_max].
    With --dmd_ts_schedule on, raise the lower bound to denoised_to (signal focusing)."""
    s_max = args.dmd_max_step / 1000.0
    s_min = args.dmd_min_step / 1000.0
    if getattr(args, 'dmd_ts_schedule', False):
        s_min = max(s_min, float(denoised_to))
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        sigma = torch.rand(B, device=device)
        shift = args.dmd_timestep_shift
        if shift > 1:
            sigma = shift * sigma / (1 + (shift - 1) * sigma)
        sigma = sigma.clamp(s_min, s_max)
    else:
        sigma = torch.zeros(B, device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(sigma, src=0)
    return sigma.to(dtype)


def _broadcast_int(val, device):
    """broadcast rank0's int to all ranks (ensures grad_step is consistent)."""
    if (not dist.is_initialized()) or dist.get_world_size() == 1:
        return val
    t = torch.tensor([val], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


def _draw_dmd_block_count(args, device):
    """dynamic DMD: random total block count M (history = M-1 blocks), broadcast from rank0
    so each FSDP rank does the same number of forwards (otherwise the collective deadlocks)."""
    counts = list(getattr(args, 'dmd_block_counts', None) or [int(args.dmd_num_blocks)])
    probs = getattr(args, 'dmd_block_count_probs', None)
    if probs and len(probs) == len(counts):
        m = random.choices(counts, weights=list(probs), k=1)[0]
    else:
        m = random.choice(counts)
    return _broadcast_int(int(m), device)


def determine_history_mode(args) -> str:
    """Resolve the DMD multi-block history compression mechanism: returns 'none'|'framepack' (via --dmd_history_mode)."""
    return getattr(args, 'dmd_history_mode', None) or 'none'


def _dit_velocity_field(transformer, x_5d, sigma, context_emb, seq_len,
                          mem_tokens=None, mem_indices=None,
                          target_t_indices=None, action_labels=None, cond_latent_frames=0):
    """uniform wrapper for the Wan transformer forward -> velocity [B,48,F,H,W]. Supports batch B>1.
    cond_latent_frames>0: the first N frames are clean conditioning frames (t=0), only later frames denoised by σ."""
    t_in = (sigma if torch.is_tensor(sigma) else torch.tensor([sigma], device=x_5d.device))
    t_in = (t_in.view(-1)[:1].float() * 1000.0).to(x_5d.device)
    x_list = list(x_5d)
    ctx_list = list(context_emb) if context_emb.dim() == 3 else [context_emb]
    out = transformer(
        x=x_list, t=t_in, context=ctx_list, seq_len=seq_len,
        cond_latent_frames=cond_latent_frames,
        history_kv_tokens=mem_tokens, history_indices_grid=mem_indices,
        target_t_indices=target_t_indices,
        action_labels=action_labels,
    )
    return torch.stack(out) if isinstance(out, (list, tuple)) else out


def student_rollout(generator, history_encoder, args, caption_emb,
                      full_action_labels, latent_shape, device, dtype, train_mode=True,
                      num_blocks=None):
    """autoregressively roll out M blocks from pure noise (K frames per block, 4-step).
      train_mode=True : keep gradient at a random step per block, break at that step (Self-Forcing trick, DMD loss at that x0).
      train_mode=False: run the full 4-step per block, final x0 = full denoise (validation/inference).
    Between blocks the history encoder compresses all generated frames (detached) as mem; the graph is not chained across blocks.

    Args:
        latent_shape: (C=48, H_lat, W_lat)
        full_action_labels: [1, M*K] full discrete camera labels (None=caption only)
        num_blocks: explicit total block count M (None=auto: train->sample, eval->dmd_num_blocks)
    Returns:
        x0_gen [1,48,M*K,H,W] with gradient, gradient_mask (None), denoised_to_sigma (float), M (int).
    """
    C, H, W = latent_shape
    K = args.dmd_block_K
    if num_blocks is not None:
        M = int(num_blocks)
    elif train_mode:
        M = _draw_dmd_block_count(args, device)
    else:
        M = int(args.dmd_num_blocks)
    tpf = (H // WAN_PATCH_H) * (W // WAN_PATCH_W)
    # 4-step schedule + timestep_shift
    sched = list(args.dmd_denoising_sigmas) + [0.0]
    _shift = float(getattr(args, 'dmd_timestep_shift', 5.0))
    if _shift and _shift != 1.0:
        sched = [(_shift * s / (1 + (_shift - 1) * s)) for s in sched]
    n_steps = len(sched) - 1

    # train: randomly pick one step per block to keep gradient (consistent across ranks); eval: full 4-step no grad
    grad_step = _broadcast_int(random.randint(0, n_steps - 1), device) if train_mode else -1
    # DMD noising lower bound = sigma the generator denoised to (next sigma after grad_step); last step -> 0 = full range
    denoised_to_sigma = float(sched[grad_step + 1]) if (train_mode and 0 <= grad_step < n_steps) else 0.0

    # multi-block history compression: none / framepack
    hist_mode = determine_history_mode(args)
    if hist_mode != 'none' and history_encoder is None:
        raise RuntimeError(f"dmd_history_mode={hist_mode} but history_encoder=None: please build the compressor by mode in "
                           f"setup_wan_model")
    use_compress = (hist_mode != 'none') and (history_encoder is not None)

    generated_detached = []   # next block's history (detached, not chained into the graph)
    x0_blocks = []            # with gradient, concatenated into x0_gen for DMD

    for b in range(M):
        n_hist = b * K
        # build this block's velocity closure _vel(x_cur, σ) -> current K-frame velocity, per history mechanism
        if b == 0:
            # block0 has no history
            seq_len0 = K * tpf
            gen_t0 = torch.arange(0, K, device=device, dtype=torch.float32)
            al_b0 = full_action_labels[:, 0:K] if full_action_labels is not None else None
            def _vel(x_cur, s, _sl=seq_len0, _gt=gen_t0, _al=al_b0):
                return _dit_velocity_field(generator, x_cur, s, caption_emb, _sl,
                                             target_t_indices=_gt, action_labels=_al)
        elif use_compress:
            # compress all generated frames -> (mem_tokens, mem_indices_grid)
            hist = torch.cat(generated_detached, dim=2).contiguous()
            mem_tokens, mem_indices = history_encoder(hist)
            n_mem = mem_tokens.shape[1] if mem_tokens is not None else 0
            seq_len_he = n_mem + K * tpf
            gen_t_he = torch.arange(n_hist, n_hist + K, device=device, dtype=torch.float32)
            al_b_he = full_action_labels[:, b * K:(b + 1) * K] if full_action_labels is not None else None
            def _vel(x_cur, s, _sl=seq_len_he, _mem=mem_tokens, _mi=mem_indices, _gt=gen_t_he, _al=al_b_he):
                return _dit_velocity_field(generator, x_cur, s, caption_emb, _sl,
                                             mem_tokens=_mem, mem_indices=_mi,
                                             target_t_indices=_gt, action_labels=_al)
        else:
            # HE-free autoregression: prepend generated clean latents as cond frames (t=0); only the current K frames denoised by σ
            prefix = torch.cat(generated_detached, dim=2).contiguous()   # [1,C,n_hist,H,W] clean (detached)
            seq_len_nh = (n_hist + K) * tpf
            al_full = full_action_labels[:, 0:(b + 1) * K] if full_action_labels is not None else None  # history + current
            def _vel(x_cur, s, _pre=prefix, _sl=seq_len_nh, _cf=n_hist, _al=al_full, _nh=n_hist):
                x_in = torch.cat([_pre, x_cur], dim=2)                   # [1,C,n_hist+K,H,W]
                v_full = _dit_velocity_field(generator, x_in, s, caption_emb, _sl,
                                               cond_latent_frames=_cf, action_labels=_al)
                return v_full[:, :, _nh:, :, :]                          # only the current K-frame velocity

        # 4-step denoise (Self-Forcing few-step scheme): non-hit steps denoise grad-off + re-noise to next sigma;
        # the grad_step is denoised once with gradient, x0 given to DMD, then break.
        x_t = torch.randn(1, C, K, H, W, device=device, dtype=dtype)
        x0 = None
        _outer_grad = torch.is_grad_enabled()        # outer grad state (critic rollout runs under no_grad)
        for i in range(n_steps):
            s_cur = sched[i]
            exit_flag = train_mode and (i == grad_step)

            if not exit_flag:
                torch.set_grad_enabled(False)
                v = _vel(x_t, s_cur)
                # transformer outputs float32; convert back to dtype to avoid a conv dtype mismatch in the bf16 history_encoder
                x0 = velocity_into_x0(v, x_t, s_cur).to(dtype)
                torch.set_grad_enabled(_outer_grad)
                # re-noise to the next sigma level (flow forward noising, fresh ε each step)
                if i < n_steps - 1:
                    s_nxt = sched[i + 1]
                    x_t = ((1 - s_nxt) * x0 + s_nxt * torch.randn_like(x0)).to(dtype)
            else:
                v = _vel(x_t, s_cur)
                x0 = velocity_into_x0(v, x_t, s_cur).to(dtype)
                break

        x0_blocks.append(x0)
        generated_detached.append(x0.detach())

    x0_gen = torch.cat(x0_blocks, dim=2)                 # [1,48,M*K,H,W] with gradient
    return x0_gen, None, denoised_to_sigma, M


def calc_dmd_loss(real_score, fake_score, x0_gen, caption_emb, neg_caption_emb,
                     full_action_labels, args, device, gradient_mask=None, denoised_to=0.0,
                     gt_latent=None, gt_reg_weight=0.0):
    """DMD generator loss. real_score (frozen teacher, CFG) + fake_score (critic) treat x0_gen as a complete
    video (t2v + caption + full camera, no history). denoised_to: noising-sigma lower bound when ts_schedule is on."""
    B, C, Fn, H, W = x0_gen.shape
    tpf = (H // WAN_PATCH_H) * (W // WAN_PATCH_W)
    seq_len = Fn * tpf
    full_t = torch.arange(0, Fn, device=device, dtype=torch.float32)

    with torch.no_grad():
        sigma = _draw_dmd_sigma(args, B, device, x0_gen.dtype, denoised_to=denoised_to)
        noise = torch.randn_like(x0_gen)
        s5 = sigma.view(B, 1, 1, 1, 1)
        x_t = (1 - s5) * x0_gen + s5 * noise

        # real score (frozen teacher) + CFG. cond = caption + camera; uncond = pure neg caption (action_labels=None),
        # so CFG amplifies the joint caption+camera direction (else camera cancels out on both sides).
        v_real_c = _dit_velocity_field(real_score, x_t, sigma, caption_emb, seq_len,
                                         target_t_indices=full_t, action_labels=full_action_labels)
        v_real_u = _dit_velocity_field(real_score, x_t, sigma, neg_caption_emb, seq_len,
                                         target_t_indices=full_t, action_labels=None)
        x0_real_c = velocity_into_x0(v_real_c, x_t, sigma)
        x0_real_u = velocity_into_x0(v_real_u, x_t, sigma)
        x0_real = x0_real_c + (x0_real_c - x0_real_u) * args.real_guidance_scale

        # fake score (critic)
        v_fake = _dit_velocity_field(fake_score, x_t, sigma, caption_emb, seq_len,
                                       target_t_indices=full_t, action_labels=full_action_labels)
        x0_fake = velocity_into_x0(v_fake, x_t, sigma)

        # DMD grad: (fake-real) / mean|x0_gen - real|
        grad = (x0_fake - x0_real)
        normalizer = (x0_gen - x0_real).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
        grad = grad / (normalizer + 1e-8)
        grad = torch.nan_to_num(grad)

    target = (x0_gen.double() - grad.double()).detach()
    if gradient_mask is not None:
        dmd_loss = 0.5 * F.mse_loss(x0_gen.double()[gradient_mask], target[gradient_mask])
    else:
        dmd_loss = 0.5 * F.mse_loss(x0_gen.double(), target)
    _log = {"dmd_grad_abs": grad.abs().mean().item(), "dmd_sigma": float(sigma.mean().item())}
    # GT-latent regression anchor: +w·MSE(x0_gen, gt_latent). gt_latent frame count must = Fn.
    if gt_latent is not None and gt_reg_weight > 0:
        _reg = F.mse_loss(x0_gen.double(), gt_latent.double())
        dmd_loss = dmd_loss + gt_reg_weight * _reg
        _log["gt_reg"] = float(_reg.item())
    return dmd_loss, _log


# SFT + forward-KL anchoring: resists DMD reverse-KL mode-shrinkage/collapse, preserves long video and motion
def _warp_sigma(s, shift):
    """timestep_shift: push more sampling toward the high-noise end (consistent with _draw_dmd_sigma)."""
    return (shift * s / (1 + (shift - 1) * s)) if (shift and shift > 1) else s


def calc_sft_loss(generator, gt_latent, caption_emb, full_action_labels, args, device):
    """diffusion SFT: low-σ flow-matching velocity MLE on the full real-video latent (strong data anchor),
    decoupled from the dynamic-M rollout (uses gt_latent's full T_full)."""
    x0 = gt_latent                                              # [1,C,T_full,H,W] full real video
    B, C, T, H, W = x0.shape
    tpf = (H // WAN_PATCH_H) * (W // WAN_PATCH_W)
    seq_len = T * tpf
    full_t = torch.arange(0, T, device=device, dtype=torch.float32)
    al = full_action_labels[:, :T] if full_action_labels is not None else None
    s = random.uniform(args.dmd_min_step / 1000.0, float(args.dmd_sft_sigma_max))   # low σ
    ss = _warp_sigma(s, float(getattr(args, 'dmd_timestep_shift', 5.0)))
    eps = torch.randn_like(x0)
    x_t = (1.0 - ss) * x0 + ss * eps
    v = _dit_velocity_field(generator, x_t, ss, caption_emb, seq_len,
                            target_t_indices=full_t, action_labels=al)
    return F.mse_loss(v.float(), (eps - x0).float())            # velocity MLE (flow matching)


def calc_real_fkl_loss(generator, gt_latent, caption_emb, full_action_labels, args, device):
    """real-data forward-KL: high-σ noise the full real video -> student pred_x0 -> MSE regression toward real x0
    (mass-covering, preventing low-motion collapse; complementary to the low-σ SFT)."""
    x0 = gt_latent
    B, C, T, H, W = x0.shape
    tpf = (H // WAN_PATCH_H) * (W // WAN_PATCH_W)
    seq_len = T * tpf
    full_t = torch.arange(0, T, device=device, dtype=torch.float32)
    al = full_action_labels[:, :T] if full_action_labels is not None else None
    s = random.uniform(float(args.dmd_real_fkl_sigma_min), float(args.dmd_real_fkl_sigma_max))   # high σ
    ss = _warp_sigma(s, float(getattr(args, 'dmd_timestep_shift', 5.0)))
    eps = torch.randn_like(x0)
    x_t = (1.0 - ss) * x0 + ss * eps
    v = _dit_velocity_field(generator, x_t, ss, caption_emb, seq_len,
                            target_t_indices=full_t, action_labels=al)
    x0_pred = velocity_into_x0(v, x_t, ss)
    return F.mse_loss(x0_pred.float(), x0.float())


def calc_teacher_fkl_loss(generator, real_score, caption_emb, neg_caption_emb,
                             full_action_labels, latent_shape, total_frames, args, device, dtype):
    """teacher forward-KL (data-free): take save points along the teacher's (real_score+CFG) dense ODE trajectory,
    secant-extrapolate to teacher pred_x0, and the student regresses its x0 to it (mass-covering)."""
    C, H, W = latent_shape
    T = int(total_frames)
    tpf = (H // WAN_PATCH_H) * (W // WAN_PATCH_W)
    seq_len = T * tpf
    full_t = torch.arange(0, T, device=device, dtype=torch.float32)
    al = full_action_labels[:, :T] if full_action_labels is not None else None
    shift = float(getattr(args, 'dmd_timestep_shift', 5.0))
    cfg = float(getattr(args, 'real_guidance_scale', 5.0))
    ts = max(4, int(args.dmd_fkl_teacher_steps))
    ts_list = [_warp_sigma(1.0 - i / ts, shift) for i in range(ts + 1)]   # σ 1->0 (+shift)
    k = max(1, min(int(args.dmd_fkl_steps), ts))
    stride = max(1, ts // 4)
    save_steps = sorted(set(min(j * stride, ts) for j in range(k + 1)))
    save_set, max_step = set(save_steps), save_steps[-1]
    saves = {}
    lat = torch.randn(1, C, T, H, W, device=device, dtype=dtype)
    with torch.no_grad():                                       # teacher dense ODE rollout (CFG)
        for i in range(max_step):
            if i in save_set:
                saves[i] = lat.detach()
            si = ts_list[i]
            v_c = _dit_velocity_field(real_score, lat, si, caption_emb, seq_len,
                                      target_t_indices=full_t, action_labels=al)
            v_u = _dit_velocity_field(real_score, lat, si, neg_caption_emb, seq_len,
                                      target_t_indices=full_t, action_labels=None)
            v = v_u + cfg * (v_c - v_u)
            lat = lat + (ts_list[i + 1] - si) * v
        saves[max_step] = lat.detach()
    loss = None
    for j in range(len(save_steps) - 1):                        # x0 regression of k segments
        a, b = save_steps[j], save_steps[j + 1]
        x_in, x_tgt = saves[a], saves[b]
        s_in, s_tgt = ts_list[a], ts_list[b]
        teacher_x0 = (x_in.double() - s_in * (x_tgt.double() - x_in.double()) / (s_tgt - s_in)).float()
        v_s = _dit_velocity_field(generator, x_in, s_in, caption_emb, seq_len,
                                  target_t_indices=full_t, action_labels=al)
        x0_pred = x_in.float() - s_in * v_s.float()
        e = F.mse_loss(x0_pred, teacher_x0)
        loss = e if loss is None else loss + e
    return loss / (len(save_steps) - 1)


def calc_critic_loss(fake_score, x0_gen, caption_emb, full_action_labels, args, device, denoised_to=0.0):
    """fake_score (critic, online) learns to denoise the generator output: flow-matching velocity loss.
    denoised_to: noising-sigma lower bound when ts_schedule is on."""
    x0 = x0_gen.detach()
    B, C, Fn, H, W = x0.shape
    tpf = (H // WAN_PATCH_H) * (W // WAN_PATCH_W)
    seq_len = Fn * tpf
    full_t = torch.arange(0, Fn, device=device, dtype=torch.float32)

    sigma = _draw_dmd_sigma(args, B, device, x0.dtype, denoised_to=denoised_to)
    noise = torch.randn_like(x0)
    s5 = sigma.view(B, 1, 1, 1, 1)
    x_t = (1 - s5) * x0 + s5 * noise

    v_fake = _dit_velocity_field(fake_score, x_t, sigma, caption_emb, seq_len,
                                   target_t_indices=full_t, action_labels=full_action_labels)
    target_v = (noise - x0)                              # flow matching velocity
    critic_loss = F.mse_loss(v_fake.float(), target_v.float())
    return critic_loss, {"critic_sigma": float(sigma.mean().item())}


@torch.no_grad()
def data_teacher_sample(real_score, caption_emb, neg_caption_emb, full_action_labels,
                        latent_shape, total_frames, device, dtype, args,
                        num_steps=50, cfg=5.0):
    """full num_steps-step CFG sampling of real_score (teacher; t2v whole segment in one pass, no history).
    Used in validation to check the teacher signal. cond = caption + camera; uncond = neg caption (action_labels=None).
    flow ODE: x_{s+1} = x_s + v·(s_next - s_cur), sigma 1->0 (+timestep_shift)."""
    C, H, W = latent_shape
    tpf = (H // WAN_PATCH_H) * (W // WAN_PATCH_W)
    seq_len = total_frames * tpf
    full_t = torch.arange(0, total_frames, device=device, dtype=torch.float32)
    sig = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    shift = float(getattr(args, 'dmd_timestep_shift', 5.0))
    if shift > 1:
        sig = shift * sig / (1 + (shift - 1) * sig)
    x_t = torch.randn(1, C, total_frames, H, W, device=device, dtype=dtype)
    for i in range(num_steps):
        s_cur = sig[i].item()
        s_nxt = sig[i + 1].item()
        v_c = _dit_velocity_field(real_score, x_t, s_cur, caption_emb, seq_len,
                                    target_t_indices=full_t, action_labels=full_action_labels)
        if cfg > 1.0:
            v_u = _dit_velocity_field(real_score, x_t, s_cur, neg_caption_emb, seq_len,
                                        target_t_indices=full_t, action_labels=None)   # uncond does not append camera
            v = v_u + cfg * (v_c - v_u)
        else:
            v = v_c
        x_t = (x_t.float() + v.float() * (s_nxt - s_cur)).to(dtype)
    return x_t   # clean x0 (whole segment generated by the teacher)
