# HunyuanWanAdapter: makes the HY15 DiT expose a WanModel-compatible interface
import torch
import torch.nn as nn

from .modules.model import ARHunyuanVideo_1_5_DiffusionTransformer


class HunyuanWanAdapter(nn.Module):
    """Wraps ARHunyuanVideo_1_5_DiffusionTransformer into a WanModel-style interface."""

    def __init__(self, cfg: dict, device=None, dtype=None):
        super().__init__()
        self.model = ARHunyuanVideo_1_5_DiffusionTransformer(**cfg)
        if device is not None or dtype is not None:
            self.model = self.model.to(device=device, dtype=dtype)
        # PRoPE is unconditional inside block.forward_bi (uses img_attn_prope_proj), so it must be added first (zero-init→no-op)
        self.model.add_prope_parameters()
        # Attach the discrete action (zero-init→no-op) so action_labels can be injected
        self.model.add_discrete_action_parameters()

        self.dim = self.model.hidden_size
        self.patch_size = self.model.patch_size
        self._device = device
        self._dtype = dtype

    # ---------------- forward (WanModel-compatible) ----------------
    def forward(self, x, t, context, seq_len=None, y=None,
                action_labels=None, viewmats=None, Ks=None,
                cond_latent_frames=0, **_unused):
        # x: List[Tensor[C,F,H,W]] → [B,C,F,H,W]
        hidden_states = torch.stack(x, dim=0)
        B, C, F, H, W = hidden_states.shape
        ps = self.patch_size
        tt, th, tw = F // ps[0], H // ps[1], W // ps[2]
        dev, dt = hidden_states.device, hidden_states.dtype

        # timestep: HY15 expects [B*tt] (per latent frame), timestep_txt: [B]
        t = t.to(dev)
        timestep = t.repeat_interleave(tt)                       # [B*tt]
        timestep_txt = t                                          # [B]

        # context: List[Tensor[L,4096]] → pad to [B,Lmax,4096] + mask[B,Lmax]
        text_states, encoder_attention_mask = self._pad_context(context, B, dev, dt)

        # viewmats/Ks: PRoPE is unconditional, synthesize identity matrices + default pinhole if missing
        viewmats, Ks = self._prepare_camera(viewmats, Ks, B, tt, dev)

        # action: [B, tt] (forward_bi internally reshape(-1)→[B*tt]); align to tt
        action = self._align_action(action_labels, B, tt, dev)

        img, _ = self.model.forward_bi(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_txt=timestep_txt,
            text_states=text_states,
            text_states_2=None,
            encoder_attention_mask=encoder_attention_mask,
            vision_states=None,
            viewmats=viewmats,
            Ks=Ks,
            action=action,
        )  # img: [B, C, F, H, W]

        return [img[i].float() for i in range(img.shape[0])]

    # ---------------- camera control interface (WanModel-compatible) ----------------
    def add_prope_camera_control(self, patches_x=None, patches_y=None,
                                 image_width=None, image_height=None,
                                 cross_norm=False, cross_norm_scale=1.0,
                                 cn_warmup_steps=0):
        # HY15's PRoPE is per-token (camera per-token), it does not need patches/image sizes;
        # here we only ensure the prope parameters are added (idempotent) and store the config for interface symmetry.
        self.model.add_prope_parameters()
        self._prope_cfg = dict(patches_x=patches_x, patches_y=patches_y,
                               image_width=image_width, image_height=image_height,
                               cross_norm=cross_norm, cross_norm_scale=cross_norm_scale,
                               cn_warmup_steps=cn_warmup_steps)

    # ---------------- internal utilities ----------------
    @staticmethod
    def _pad_context(context, B, dev, dt):
        if torch.is_tensor(context):
            context = [context[i] for i in range(context.shape[0])]
        assert len(context) == B, f"context length {len(context)} != B {B}"
        lens = [c.shape[0] for c in context]
        Lmax = max(lens)
        D = context[0].shape[-1]
        text_states = torch.zeros(B, Lmax, D, device=dev, dtype=dt)
        mask = torch.zeros(B, Lmax, device=dev, dtype=torch.bool)
        for i, c in enumerate(context):
            L = c.shape[0]
            text_states[i, :L] = c.to(dev, dt)
            mask[i, :L] = True
        return text_states, mask

    @staticmethod
    def _prepare_camera(viewmats, Ks, B, tt, dev):
        if viewmats is None:
            viewmats = torch.eye(4, device=dev).reshape(1, 1, 4, 4).repeat(B, tt, 1, 1)
        else:
            viewmats = viewmats.to(dev, torch.float32)
            if viewmats.dim() == 3:
                viewmats = viewmats.unsqueeze(0)
            viewmats = HunyuanWanAdapter._align_frames(viewmats, tt)
        if Ks is None:
            Ks = torch.eye(3, device=dev).reshape(1, 1, 3, 3).repeat(B, tt, 1, 1)
        else:
            Ks = Ks.to(dev, torch.float32)
            if Ks.dim() == 3:
                Ks = Ks.unsqueeze(0)
            Ks = HunyuanWanAdapter._align_frames(Ks, tt)
        return viewmats, Ks

    @staticmethod
    def _align_frames(m, tt):
        # m: [B, T, ...]; truncate or pad with the last frame to tt
        T = m.shape[1]
        if T == tt:
            return m
        if T > tt:
            return m[:, :tt]
        pad = m[:, -1:].repeat(1, tt - T, *([1] * (m.dim() - 2)))
        return torch.cat([m, pad], dim=1)

    @staticmethod
    def _align_action(action_labels, B, tt, dev):
        if action_labels is None:
            return None
        a = action_labels
        if not torch.is_tensor(a):
            a = torch.as_tensor(a)
        a = a.to(dev).long()
        if a.dim() == 1:
            a = a.unsqueeze(0)
        T = a.shape[1]
        if T != tt:
            a = HunyuanWanAdapter._align_frames(a.unsqueeze(-1), tt).squeeze(-1)
        return a
