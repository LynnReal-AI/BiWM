"""Joystick (keycap-style) control overlay for the 81 discrete actions.

action_label = trans*9 + rot
  trans (WASD): 0 static 1 W 2 S 3 D 4 A 5 W+D 6 W+A 7 S+D 8 S+A
  rot (arrows): 0 static 1 up 2 down 3 right 4 left 5 up-right 6 up-left 7 down-right 8 down-left

Active key = orange rounded box + white glyph; inactive = dim glyph on dark box.
Run directly to render outputs/control_preview.png previewing several actions.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_AMBER = (235, 128, 32, 255)
_WHITE_ENABLE = (255, 255, 255, 255)
_WHITE_DISABLE = (245, 245, 245, 235)
_KEY_IDLE_BG = (0, 0, 0, 110)
_TYPEFACE_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# trans(0..8) -> active WASD keys
TRANS_KEY_LIST = {
    0: [], 1: ["W"], 2: ["S"], 3: ["D"], 4: ["A"],
    5: ["W", "D"], 6: ["W", "A"], 7: ["S", "D"], 8: ["S", "A"],
}
# rot(0..8) -> active arrow directions
ROT_DIRECTIONS = {
    0: [], 1: ["up"], 2: ["down"], 3: ["right"], 4: ["left"],
    5: ["up", "right"], 6: ["up", "left"], 7: ["down", "right"], 8: ["down", "left"],
}


def _typeface(px):
    try:
        return ImageFont.truetype(_TYPEFACE_PATH, px)
    except Exception:
        return ImageFont.load_default()


def _round_corner(draw, box, r, fill):
    try:
        draw.rounded_rectangle(box, radius=r, fill=fill)
    except Exception:
        draw.rectangle(box, fill=fill)


def _paint_letter(draw, cx, cy, ch, color, font):
    """Draw letter centered (WASD)."""
    l, t, r, b = draw.textbbox((0, 0), ch, font=font)
    draw.text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), ch, font=font, fill=color)


def _paint_arrow(draw, cx, cy, direction, color, a):
    """Draw a solid triangular arrow (a=half side length)."""
    if direction == "up":
        pts = [(cx, cy - a), (cx - a, cy + a * 0.7), (cx + a, cy + a * 0.7)]
    elif direction == "down":
        pts = [(cx, cy + a), (cx - a, cy - a * 0.7), (cx + a, cy - a * 0.7)]
    elif direction == "left":
        pts = [(cx - a, cy), (cx + a * 0.7, cy - a), (cx + a * 0.7, cy + a)]
    else:  # right
        pts = [(cx + a, cy), (cx - a * 0.7, cy - a), (cx - a * 0.7, cy + a)]
    draw.polygon(pts, fill=color)


def _paint_pad(img, ox, oy, cell, gap, kind, active, font):
    """Draw a T-shaped keypad. kind='wasd' or 'arrow'. active: set of active keys (letters or direction words)."""
    draw = ImageDraw.Draw(img)
    step = cell + gap
    # (col,row) of the 4 keys: top key centered, bottom row three keys
    if kind == "wasd":
        slots = [("W", 1, 0), ("A", 0, 1), ("S", 1, 1), ("D", 2, 1)]
    else:
        slots = [("up", 1, 0), ("left", 0, 1), ("down", 1, 1), ("right", 2, 1)]
    rad = max(4, cell // 5)
    for key, col, row in slots:
        x0 = ox + col * step
        y0 = oy + row * step
        cx, cy = x0 + cell / 2, y0 + cell / 2
        on = key in active
        _round_corner(draw, [x0, y0, x0 + cell, y0 + cell], rad, _AMBER if on else _KEY_IDLE_BG)
        col_glyph = _WHITE_ENABLE if on else _WHITE_DISABLE
        if kind == "wasd":
            _paint_letter(draw, cx, cy, key, col_glyph, font)
        else:
            _paint_arrow(draw, cx, cy, key, col_glyph, a=cell * 0.28)


def compose_control(action_label, cell=40, gap=6, pad_gap=26):
    """Render the RGBA image of (WASD keypad + arrow keypad). Returns PIL.Image(RGBA)."""
    trans, rot = divmod(int(action_label), 9)
    wasd_active = set(TRANS_KEY_LIST.get(trans, []))
    arrow_active = set(ROT_DIRECTIONS.get(rot, []))
    pad_w = 3 * cell + 2 * gap
    pad_h = 2 * cell + gap
    W = pad_w * 2 + pad_gap
    H = pad_h
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    font = _typeface(int(cell * 0.62))
    _paint_pad(img, 0, 0, cell, gap, "wasd", wasd_active, font)
    _paint_pad(img, pad_w + pad_gap, 0, cell, gap, "arrow", arrow_active, font)
    return img


def superimpose_control(frame_rgb, action_label, scale=1.0, margin=None, alpha=1.0):
    """Overlay the joystick onto the bottom-left of one RGB frame (np.uint8 [H,W,3])."""
    base = Image.fromarray(frame_rgb).convert("RGBA")
    Wf, Hf = base.size
    cell = max(16, int(round(Wf * 0.04 * scale)))
    js = compose_control(action_label, cell=cell, gap=max(3, cell // 7),
                         pad_gap=max(12, int(cell * 0.55)))
    if alpha < 1.0:
        a = js.split()[3].point(lambda p: int(p * alpha))
        js.putalpha(a)
    _jw, jh = js.size
    m = margin if margin is not None else max(10, int(Wf * 0.012))
    px = m
    py = Hf - jh - m
    base.alpha_composite(js, (max(0, px), max(0, py)))
    return np.array(base.convert("RGB"))


def superimpose_control_video(video_frames, action_labels, vae_temporal_stride=4,
                           pixel_to_latent=None, scale=1.0, margin=14, **_kw):
    """Overlay the joystick onto each pixel frame.
    video_frames: list of [H,W,3] uint8 RGB; action_labels: [T_lat] (tensor/list, 0~80).
    pixel->latent mapping: pixel_to_latent takes priority, otherwise (pf-1)//stride+1."""
    labs = action_labels
    try:
        import torch as _t
        if isinstance(labs, _t.Tensor):
            labs = labs.detach().cpu().tolist()
    except Exception:
        pass
    labs = list(labs)
    if labs and isinstance(labs[0], (list, tuple)):       # [B,T] -> take the 0th entry
        labs = list(labs[0])
    T_lat = len(labs)
    out = []
    for pf, frame in enumerate(video_frames):
        if pixel_to_latent is not None and pf < len(pixel_to_latent):
            li = int(pixel_to_latent[pf])
        else:
            li = 0 if pf == 0 else (pf - 1) // vae_temporal_stride + 1
        li = max(0, min(li, T_lat - 1))
        out.append(superimpose_control(np.asarray(frame), int(labs[li]), scale=scale, margin=margin))
    return out


# Preview: render a multi-action grid
def _action_caption(label):
    TRANS = ["static", "fwd", "bwd", "right", "left", "fwd-right", "fwd-left", "bwd-right", "bwd-left"]
    ROT = ["static", "pitchup", "pitchdown", "yawright", "yawleft",
           "up-right", "up-left", "down-right", "down-left"]
    t, r = divmod(label, 9)
    return f"L{label}  trans={TRANS[t]}  rot={ROT[r]}"


def _thumbnail(out="outputs/control_preview.png"):
    # representative actions: static / single / combos / translation+rotation mix
    cases = [0, 9, 36, 1, 3, 5 * 9, 4 * 9 + 2, 1 * 9 + 6, 8 * 9 + 8]
    cell = 46
    tile = compose_control(0, cell=cell)
    tw, th = tile.size
    cap_h = 26
    cols = 3
    rows = (len(cases) + cols - 1) // cols
    pad = 24
    cw = tw + pad * 2
    ch = th + cap_h + pad
    canvas = Image.new("RGB", (cw * cols, ch * rows), (24, 26, 30))
    d = ImageDraw.Draw(canvas)
    f = _typeface(15)
    for i, lab in enumerate(cases):
        r, c = divmod(i, cols)
        ox, oy = c * cw, r * ch
        js = compose_control(lab, cell=cell)
        canvas.paste(js, (ox + pad, oy + pad // 2), js)
        d.text((ox + pad, oy + th + pad // 2 + 4), _action_caption(lab), font=f, fill=(220, 220, 220))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)
    print(f"[control] preview saved: {out}  ({len(cases)} actions)")


if __name__ == "__main__":
    _thumbnail()
