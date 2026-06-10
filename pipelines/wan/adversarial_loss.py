# >>> BiWM Stage 2 (optional) — GAN adversarial loss <<<
"""
Adversarial loss for video generation training.
Hinge-based GAN loss with ProjectedDiscriminator.

Input shape: [B, C, T, H, W] or [C, T, H, W] (velocity / x0 space).
Internally does channel projection (latent_channels → 48) + frame flattening.

Distributed strategy: each rank independently holds its own copy of the discriminator,
at initialization broadcasts rank 0 weights to keep them consistent,
and updates uniformly after allreducing gradients each step.
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class GanCoach:
    """
    Wraps ProjectedDiscriminator for adversarial training.

    Distributed: each rank holds a full copy, allreduce gradients to keep in sync.

    Usage:
        adv = GanCoach(device)
        # In training loop (x0 space):
        d_loss = adv.discriminator_step(real_x0, fake_x0)
        g_loss = adv.generator_loss(fake_x0)
        total_loss = diffusion_loss + adv.gan_weight * g_loss
    """

    def __init__(self, device, fsdp_kwargs=None, lr=1e-4, gan_weight=0.01,
                 c_dim=384, max_grad_norm=1.0, latent_channels=128):
        from ADD.models.discriminator import ProjectedDiscriminator

        self.device = device
        self.gan_weight = gan_weight
        self.max_grad_norm = max_grad_norm

        # Channel projection: latent_channels → 48 (ProjectedDiscriminator expects 48)
        self.latent_channels = latent_channels
        self._disc_in_channels = 48
        if latent_channels != self._disc_in_channels:
            self.channel_proj = nn.Conv2d(
                latent_channels, self._disc_in_channels, 1, bias=False
            ).to(device=device, dtype=torch.bfloat16)
            nn.init.xavier_uniform_(self.channel_proj.weight)
        else:
            self.channel_proj = None

        # Create discriminator (no FSDP, each rank holds a full copy)
        self.discriminator = ProjectedDiscriminator(c_dim=c_dim).train().to(device)

        # ===== Distributed sync: broadcast rank 0 weights to all ranks =====
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self._world_size > 1:
            for p in self.discriminator.parameters():
                dist.broadcast(p.data, src=0)
            if self.channel_proj is not None:
                for p in self.channel_proj.parameters():
                    dist.broadcast(p.data, src=0)
        # Collect all trainable params
        _all_params = [p for p in self.discriminator.parameters() if p.requires_grad]
        if self.channel_proj is not None:
            _all_params += list(self.channel_proj.parameters())

        # Optimizer (Adam8bit for memory efficiency)
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.Adam8bit(
                _all_params,
                lr=lr, betas=(0, 0.999), weight_decay=0.001, eps=1e-8,
            )
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                _all_params,
                lr=lr, betas=(0, 0.999), weight_decay=0.001,
            )

    def _allreduce_grads(self):
        """AllReduce discriminator + channel_proj gradients, keeping weights consistent across ranks."""
        if self._world_size <= 1:
            return
        for p in self.discriminator.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if self.channel_proj is not None:
            for p in self.channel_proj.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _prepare_input(self, latent):
        """Prepare latent for discriminator.

        Input: [B, C, T, H, W] or [C, T, H, W]
        Output: [B*T, disc_ch, H, W] — all frames flattened into batch, channel projected
        """
        if latent.dim() == 5:
            B, C, T, H, W = latent.shape
            x = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        elif latent.dim() == 4:
            C, T, H, W = latent.shape
            x = latent.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
        else:
            x = latent

        # Project channels (e.g. 128→48)
        if self.channel_proj is not None:
            x = self.channel_proj(x)
        return x

    def discriminator_step(self, real_latent, fake_latent):
        """
        Update discriminator with hinge loss.
        Args:
            real_latent: ground truth x0 [B, C, T, H, W]
            fake_latent: predicted x0 [B, C, T, H, W] (detached)
        Returns: d_loss value
        """
        self.optimizer.zero_grad()

        real_input = self._prepare_input(real_latent).detach()
        fake_input = self._prepare_input(fake_latent).detach()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_real, pred_real_f, _ = self.discriminator(real_input, None)
            pred_fake, pred_fake_f, _ = self.discriminator(fake_input, None)

        pred_real = torch.cat(pred_real, dim=1)
        pred_fake = torch.cat(pred_fake, dim=1)
        pred_real_f = torch.cat(pred_real_f, dim=1)
        pred_fake_f = torch.cat(pred_fake_f, dim=1)

        # Hinge loss
        loss_real = (torch.mean(torch.relu(1.0 - pred_real)) +
                     torch.mean(torch.relu(1.0 - pred_real_f)))
        loss_fake = (torch.mean(torch.relu(1.0 + pred_fake)) +
                     torch.mean(torch.relu(1.0 + pred_fake_f)))

        loss_d = (loss_real + loss_fake) / 2.0
        loss_d.backward()

        # AllReduce gradients to keep the discriminator consistent across ranks
        self._allreduce_grads()

        if hasattr(self.discriminator, 'clip_grad_norm_'):
            self.discriminator.clip_grad_norm_(self.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_d.item()

    def generator_loss(self, fake_latent):
        """
        Compute generator adversarial loss (fool the discriminator).
        Args:
            fake_latent: predicted x0 [B, C, T, H, W] (must keep gradients)
        Returns: gan_loss tensor (differentiable)
        """
        fake_input = self._prepare_input(fake_latent)

        # Freeze the discriminator: avoid FSDP double-backward conflict + save memory
        self.discriminator.requires_grad_(False)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_fake, pred_fake_f, _ = self.discriminator(fake_input, None)
        self.discriminator.requires_grad_(True)

        pred_fake = torch.cat(pred_fake, dim=1)
        pred_fake_f = torch.cat(pred_fake_f, dim=1)

        gan_loss = -torch.mean(pred_fake) - torch.mean(pred_fake_f)
        return gan_loss

    def state_dict(self):
        sd = {
            'discriminator': self.discriminator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.channel_proj is not None:
            sd['channel_proj'] = self.channel_proj.state_dict()
        return sd

    def load_state_dict(self, state):
        if 'discriminator' in state:
            self.discriminator.load_state_dict(state['discriminator'])
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        if 'channel_proj' in state and self.channel_proj is not None:
            self.channel_proj.load_state_dict(state['channel_proj'])
