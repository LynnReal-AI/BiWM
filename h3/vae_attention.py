# SPDX-License-Identifier: Apache-2.0
"""Instance-only VAE SDPA selection; never change the DiT/global FA3 backend."""
def configure_vae_attention(vae):
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
    before = _AttentionBackendRegistry.get_active_backend()
    count = 0
    for module in vae.modules():
        processor = getattr(module, 'processor', None)
        if processor is not None and hasattr(processor, '_attention_backend'):
            processor._attention_backend = AttentionBackendName.NATIVE
            count += 1
    assert count > 0, 'No VAE attention processors were configured'
    assert _AttentionBackendRegistry.get_active_backend() == before
    return count
