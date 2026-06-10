# model config dict used by the HY15 adapter layer
HUNYUAN_T2V_CONFIG = dict(
    patch_size=[1, 2, 2],
    in_channels=16,
    concat_condition=False,
    out_channels=16,
    hidden_size=2048,
    heads_num=16,
    mlp_width_ratio=4.0,
    mlp_act_type="gelu_tanh",
    mm_double_blocks_depth=20,
    mm_single_blocks_depth=0,
    rope_dim_list=[16, 56, 56],
    qkv_bias=True,
    qk_norm=True,
    qk_norm_type="rms",
    guidance_embed=False,
    text_projection="single_refiner",
    use_attention_mask=True,
    text_states_dim=4096,      # aligned to the wan T5 context dim, the adapter's context[L,4096] can be fed directly
    attn_mode="flash",         # when flash is unavailable at runtime, maybe_fallback_attn_mode automatically falls back to torch
    glyph_byT5_v2=False,
    vision_projection="none",
    use_cond_type_embedding=False,
)

# tiny model for smoke tests
HUNYUAN_SMALL_CONFIG = dict(
    patch_size=[1, 2, 2],
    in_channels=16,
    concat_condition=False,
    out_channels=16,
    hidden_size=256,
    heads_num=4,
    mlp_width_ratio=4.0,
    mlp_act_type="gelu_tanh",
    mm_double_blocks_depth=2,
    mm_single_blocks_depth=0,
    rope_dim_list=[16, 24, 24],
    qkv_bias=True,
    qk_norm=True,
    qk_norm_type="rms",
    guidance_embed=False,
    text_projection="single_refiner",
    use_attention_mask=True,
    text_states_dim=4096,
    attn_mode="torch",         # explicit torch, skips the flash fallback warning
    glyph_byT5_v2=False,
    vision_projection="none",
    use_cond_type_embedding=False,
)
