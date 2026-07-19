"""Small hinge-GAN helpers shared by the backbone-specific DMD trainers."""

import torch


def _discriminator_forward(discriminator, latent):
    """Run a 5D video latent through the frame-wise discriminator."""
    frames = latent.squeeze(0).permute(1, 0, 2, 3).contiguous()
    logits, feature_logits, _ = discriminator(frames, None)
    return torch.cat(logits, dim=1), torch.cat(feature_logits, dim=1)


def discriminator_hinge_loss(discriminator, fake, real):
    """Hinge loss used to update the discriminator."""
    pred_fake, pred_fake_features = _discriminator_forward(discriminator, fake.detach())
    pred_real, pred_real_features = _discriminator_forward(discriminator, real.detach())
    real_loss = torch.relu(1.0 - pred_real).mean() + torch.relu(1.0 - pred_real_features).mean()
    fake_loss = torch.relu(1.0 + pred_fake).mean() + torch.relu(1.0 + pred_fake_features).mean()
    return 0.5 * (real_loss + fake_loss)


def generator_hinge_loss(discriminator, fake):
    """Adversarial loss used to update the generator."""
    pred_fake, pred_fake_features = _discriminator_forward(discriminator, fake)
    return -pred_fake.mean() - pred_fake_features.mean()
