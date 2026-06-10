# HunyuanVideo-1.5 (HY15) Model Forward Pass Walkthrough

> Subject: `ARHunyuanVideo_1_5_DiffusionTransformer` (the backbone) in `hunyuan/modules/model.py`
> and `MMDoubleStreamBlock` (the dual-stream block). Line numbers refer to the current `model.py`.
> This document focuses on the **forward computation flow**, the core that biWM's bidirectional autoregressive world model training/distillation depends on.

---

## 0. One-line overview

HY15 is a **dual-stream MMDiT** (Multi-Modal Diffusion Transformer): it concatenates the "image latent token stream" and
the "text token stream" for **joint attention**, alternately updating the two streams block by block. This repo's variant is
**pure dual-stream** (`mm_single_blocks_depth == 0`, assert at line 903), and on top of the original adds **two zero-initialized camera-condition mechanisms**:

- **discrete per-frame camera** `action_in` (embeds the 81-class action per frame and adds it to the AdaLN modulation vector `vec`);
- **continuous PRoPE camera** `img_attn_prope_proj` (inside each double block, runs a camera relative-pose attention branch **in parallel**).

Both are **zero-init**, so they are a no-op initially and don't disturb the pretrained base; they are learned during training.

```
latent x_t ──img_in(patchify)──► img tokens ┐
                                            ├─ N× MMDoubleStreamBlock(joint attn + PRoPE) ─► final_layer ─► unpatchify ─► v (velocity)
caption ─►MLLM emb─►txt_in─►(+byt5+vision)──► txt tokens ┘
timestep(per-frame)+action(per-frame) ─► vec (AdaLN modulation)        viewmats/Ks ─► PRoPE branch
```

---

## 1. The four forward entrypoints and the dispatcher

The model-level `forward(...)` (line 1559) is a **dispatcher** that picks a concrete forward by flag:

| Entrypoint | flag | Attention | Use | KV-cache |
|---|---|---|---|---|
| `forward_bi` | `bi_inference=True` (default) | **bidirectional** (fully connected within the block; bidirectional within block / causal across blocks during teacher-forcing) | **biWM training / DMD teacher / bidirectional autoregressive generation** | no |
| `forward_txt` | `ar_txt_inference=True` | causal | in AR mode, first feed the **text** KV into the cache | yes (`cache_txt`) |
| `forward_vision` | `ar_vision_inference=True` | causal | in AR mode, generate block by block (causal KV-cache) | yes (`cache_vision`) |
| `forward_sr` | otherwise | — | super-resolution | — |

> ⭐ biWM's stage1 training and stage2 DMD (generator bidirectional autoregressive rollout + teacher scoring) **all go through `forward_bi`**.
> `forward_txt`/`forward_vision` are HunyuanVideo's original causal AR inference paths, which biWM currently does not use (we want "bidirectional" autoregression).

Below we detail **`forward_bi`** (lines 1372–1557), which is the main line.

---

## 2. The full flow of `ARHunyuanVideo_1_5_DiffusionTransformer.forward_bi`

### 2.1 Inputs
| Argument | Shape | Meaning |
|---|---|---|
| `hidden_states` | `[B, C_in, T, H, W]` | noisy latent (when `concat_condition=True`, `C_in=2*C+1`, containing noisy + condition latent + mask; see §2.2) |
| `timestep` | `[B*T_lat]` | the diffusion step **per latent frame** (sigma·1000) |
| `timestep_txt` | `[B]` | the text-stream timestep |
| `text_states` | `[B, L, 4096]` | the MLLM embedding of **one caption for the whole clip** (HY15's text encoder is an MLLM=Qwen2.5-VL, not T5) |
| `encoder_attention_mask` | `[B, L]` | caption mask |
| `extra_kwargs` | dict | `byt5_text_states [B,Lb,1472]` + `byt5_text_mask` (glyph, when `glyph_byT5_v2=True`) |
| `vision_states` | `[B, Lv, 3584]` or None | SigLIP vision condition (i2v; for t2v all zeros are passed → masked out by multiplying by 0) |
| `viewmats / Ks` | `[B, T_lat, 4,4] / [B,T_lat,3,3]` | PRoPE camera (w2c / intrinsics) |
| `action` | `[B, T_lat]` or None | **per-frame** 81-class discrete action labels |
| `clean_x / aug_timesteps` | same as hidden / `[B*T_lat]` | **teacher-forcing** (bidirectional autoregression): clean prefix + its per-frame noise schedule; None = plain bidirectional |
| `mask_type` | `"t2v"/"i2v"` | task type |

**Returns** `(img, None)`, where `img = velocity [B, C_out, T, H, W]` (flow-matching velocity field, `x0 = x_t - σ·v`).

### 2.2 Image-stream embedding + per-frame modulation vector (lines 1404–1431)
```python
bs, _, ot, oh, ow = x.shape
tt, th, tw = ot//ps0, oh//ps1, ow//ps2          # latent grid: tt frames, th*tw tokens per frame
img = self.img_in(img)                          # PatchEmbed: [B, tt*th*tw, hidden]  (takes 2C+1 channels when concat_condition)
vec = self.time_in(t)                            # [B*tt, hidden]  ← per-frame time embedding
if action is not None and self.action_in:        # ★ discrete per-frame camera
    vec = vec + self.action_in(action.reshape(-1))   # [B*tt, hidden]  per-frame action embedding (zero-init)
vec = repeat(vec, "(B T) C -> B (T H W) C", H=th, W=tw)   # broadcast each frame's vec to all spatial tokens of that frame
if viewmats is not None:                         # PRoPE camera is also expanded per-frame to tokens
    viewmats = repeat(viewmats, "B T M N -> B (T H W) M N", H=th, W=tw)
    Ks       = repeat(Ks,       "B T M N -> B (T H W) M N", H=th, W=tw)
```
> **Key to per-frame cameras**: `vec` is the per-token modulation vector; the tokens of the i-th frame share the modulation of "the i-th frame timestep + the i-th frame action".
> So **the discrete camera goes through the image-stream's per-frame AdaLN, not the text stream** (the text is one sequence for the whole clip, with no per-frame).

### 2.3 teacher-forcing (the core of bidirectional autoregression, lines 1426–1466)
When `clean_x is not None` (`is_tf=True`):
```python
clean_img = self.img_in(clean_x)                 # the clean prefix is also patchified
vec_clean = self.time_in(aug_timesteps); vec_clean = repeat(...)   # the clean half's modulation (noise schedule ≈ 0)
...
img = torch.cat([clean_img, img], dim=1)         # [B, 2*L, hidden]  clean first, noisy second
```
and it builds a **teacher-forcing BlockMask** (lines 1488–1509, `prepare_teacher_forcing_mask`,
`num_frame_per_block=4` is **hardcoded**): each frame's noisy tokens are **bidirectional within the block** and **causal** toward the preceding blocks.
→ this is **"bidirectional within block + autoregressive across blocks"**. At the end of the forward, only the noisy half's output is taken (lines 1545–1546).

> ⚠️ biWM DMD's generator rollout feeds the "already-generated blocks" via `clean_x` to do bidirectional autoregression; `--num_frame_per_block` must = 4 to align with this mask.

### 2.4 Text stream (lines 1473–1485, `get_text_and_mask`)
```python
txt = self.txt_in(text_states, timestep_txt, mask)   # single_refiner: caption MLLM embedding → hidden
if glyph_byT5_v2:                                    # concatenate ByT5 glyph tokens
    byt5_txt = self.byt5_in(byt5_text_states)        # 1472 → hidden
    txt, text_mask = reorder_txt_token(byt5_txt, txt, ...)   # [glyph valid][caption valid][pad...]
if vision_in and vision_states is not None:          # concatenate SigLIP vision tokens (t2v all 0 → masked)
    txt = cat([txt, vision_in(vision_states)], ...)
txt = txt[text_mask.bool()].unsqueeze(0)             # drop pad, get the final txt token stream
```
→ the text stream = **one sequence** made of **caption(MLLM) + glyph(ByT5) + vision(SigLIP)**, shared across the whole clip (not per-frame).

### 2.5 RoPE (lines 1413–1414)
`freqs_cos/sin = get_rotary_pos_embed((tt, th, tw))`: 3D rotary positional encoding (t,h,w three axes, `sum(rope_dim_list) = head_dim`).

### 2.6 N× dual-stream blocks (lines 1513–1542)
Loop over `self.double_blocks`, each `block.forward_bi(img, txt, vec_txt, vec, freqs_cis, vec_clean, tf_block_mask, viewmats, Ks, ...)`, see §3.

### 2.7 Output (lines 1544–1557)
```python
if is_tf: img = img[:, img.shape[1]//2:]    # take only the noisy half
img = self.final_layer(img, vec)            # AdaLN-out + Linear → patch pixels
img = self.unpatchify(img, tt, th, tw)      # [B, C_out, T, H, W]
return (img, None)
```

---

## 3. `MMDoubleStreamBlock.forward_bi` (lines 381–622)

A single block: the image stream and text stream each do **AdaLN modulation → QKV**, then **joint attention** (img+txt tokens are concatenated and attend together), the output is written back to each stream as a residual + each stream's MLP. HY15 adds a **parallel PRoPE camera attention branch** here.

### 3.1 Text-stream modulation (`modulate_txt`, line 191)
`vec_txt → 6 paths (shift/scale/gate)×2`; `txt → LayerNorm → modulate(shift,scale) → q/k/v → qk_norm`.

### 3.2 Image stream (two paths)

**(a) plain bidirectional** (`is_tf=False`, lines 536–611)
```python
img_q,img_k,img_v, gates... = self.modulate_img(vec, img)     # as above, image-stream AdaLN→QKV
img_q_prope,img_k_prope,img_v_prope, apply_fn_o = prope_qkv(   # ★ PRoPE: project q/k/v with camera pose
        q,k,v, viewmats=viewmats, Ks=Ks)
img_q,img_k = apply_rotary_emb(img_q,img_k, freqs_cis)        # the main branch then adds 3D RoPE
attn        = parallel_attention((img_q,txt_q),(img_k,txt_k),(img_v,txt_v), ...)  # joint: img+txt attend together
attn_prope  = parallel_attention((img_q_prope,txt_q), ...)    # parallel PRoPE branch (also joint)
img_attn, txt_attn         = attn[:img_len], attn[img_len:]
img_attn_prope = apply_fn_o(attn_prope[:img_len])             # PRoPE output then goes through camera inverse projection
img = img + gate1·( img_attn_proj(img_attn) + img_attn_prope_proj(img_attn_prope) )  # ★ main + PRoPE residual
img = img + gate2·( img_mlp(modulate(norm2(img))) )                                  # image MLP
```
> **PRoPE is a parallel branch**: it reuses the same q/k/v, attends separately after the camera relative-pose projection, and the output is added back to the main branch as a residual through the **zero-init**
> `img_attn_prope_proj` (initially = 0, doesn't perturb the base). See `prope_qkv` in `camera_rope.py`.

**(b) teacher-forcing** (`is_tf=True`, lines 409–535)
The clean / noisy halves are **separately** modulated with `vec_clean` / `vec` (different AdaLN parameters), each with its own RoPE, joint attention uses
`attn_mode='flex_tf' + tf_block_mask` (bidirectional within block / causal across blocks). PRoPE does `prope_qkv` separately for the clean/noisy halves.
Finally `img = cat([clean_img, noisy_img])`, then drop the clean half back at the model level.

### 3.3 Text-stream update (lines 613–621)
`txt = txt + gate·txt_attn_proj(txt_attn)`; `txt = txt + gate·txt_mlp(modulate(norm2(txt)))`.

> Note: in the dual stream, **txt also participates in attention and is updated** (not a read-only condition), which is the difference between MMDiT and single-stream cross-attn.

---

## 4. Condition injection summary (which path each takes, and whether per-frame)

| Condition | Injection point | Path | Per-frame? | Notes |
|---|---|---|---|---|
| **timestep** (diffusion step) | `vec = time_in(t)` | image-stream AdaLN | ✅ per-frame | `timestep [B*T_lat]` |
| **discrete camera action** (81 classes) | `vec += action_in(action)` | image-stream AdaLN | ✅ per-frame | `action_in` zero-init; §2.2 |
| **continuous camera PRoPE** (viewmats/Ks) | parallel attn within the block | image-stream attention | ✅ per-token | `img_attn_prope_proj` zero-init; §3.2 |
| **caption** (MLLM=Qwen2.5-VL) | `txt_in` (single_refiner) | text stream | ❌ one per clip | `text_states [B,L,4096]` |
| **glyph** (ByT5) | `byt5_in` + reorder | text stream (concatenated) | ❌ | `glyph_byT5_v2=True` |
| **vision** (SigLIP, i2v) | `vision_in` + concatenate | text stream (concatenated) | ❌ | t2v passes 0 to mask out |
| **image condition** (i2v first-frame latent) | `concat_condition` into `img_in` | image-stream channels | first frame | `C_in=2C+1` |
| **clean prefix** (bidirectional autoregression) | `clean_x` concat tokens + block mask | image-stream attention | per block | teacher-forcing; §2.3 |

> Core conclusion: **cameras (discrete + continuous) go per-frame through the image stream; text (caption/glyph/vision) goes through the text stream for the whole clip.** The two interact in joint attention.

---

## 5. Brief on the AR causal path (`forward_txt` / `forward_vision`, biWM not using for now)

HunyuanVideo's original causal autoregressive inference:
1. `forward_txt` (line 256 block / 1189 model): `attn_mode="torch_causal"`, writes the **text** KV into `kv_cache` (`cache_txt=True`), computed only once.
2. `forward_vision` (line 303 block / 1244 model): generates block by block, image tokens attend causally (`cache_vision`), PRoPE also runs in parallel.

> This is **causal KV-cache** autoregression (causal generator); biWM wants **bidirectional** autoregression (the generator still uses `forward_bi`, achieving bidirectional-within-block / causal-across-blocks via `clean_x`+block mask), so biWM's DMD generator does not take this path.

---

## 6. Shape quick reference (B=batch, single clip)

> ★ Shapes are **governed by the weights directory's `config.json`** (the model is fully built from it). Measured `transformer/480p_i2v`:
> `in_channels=out_channels=32`, `patch_size=[1,1,1]`, `hidden_size=2048`, `heads_num=16`,
> `mm_double_blocks_depth=54`, `text_states_dim=3584` (**Qwen2.5-VL MLLM, not 4096**),
> `vision_states_dim=1152` (SigLIP), `concat_condition=true`, `rope_theta=256`, `rope_dim_list=[16,56,56]`.

```
latent  x_t            : [B, C_in, T, H, W]      C_in = (2*C+1 if concat_condition else C); C=latent channels (480p=32)
patchify img_in        : [B, tt*th*tw, hidden]   tt=T/ps0, th=H/ps1, tw=W/ps2 (480p patch_size=[1,1,1]→tt=T,th=H,tw=W)
vec (AdaLN)            : [B, tt*th*tw, hidden]    per-frame time+action broadcast
txt (text stream)      : [B, L', hidden]          L' = caption(MLLM3584) + byt5 + vision valid tokens
joint attn sequence    : [B, tt*th*tw + L', hidden]
viewmats/Ks(expanded)  : [B, tt*th*tw, 4,4]/[..,3,3]
output velocity        : [B, C_out, T, H, W]      C_out = out_channels (480p=32)
```

---

## 7. Interface with training / DMD (biWM)

- **stage1** (`pipelines/hy15/train_hunyuan.py`): sample σ → add noise to `x_t` → `forward_bi(hidden=x_t(+cond), timestep=per-frame, text_states=MLLM, viewmats/Ks, action, byt5/vision via extra_kwargs)` → predict velocity → flow-matching loss.
