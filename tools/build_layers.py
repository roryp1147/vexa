#!/usr/bin/env python3
"""
Vexa art-layer pipeline (offline build step, NOT needed at runtime).

Input : ../concept_v6.jpg  (1280x720 AI concept, Vexa bust on near-black)
Output: assets/layers/*.png + assets/layers/layout.json

What it does
------------
1. Paints out the baked-in playlist chip (viewer's left) by mirroring the
   clean right-hand side of the head across the face's symmetry axis, so both
   ear slots are identical clean housings (chips get drawn procedurally later).
2. Removes the baked choker O-ring (inpaint) - it is drawn at runtime as a
   pendulum.
3. Shifts red cables in the severed base / arm stumps to copper (keeps it
   reading as machinery: no blood/flesh, PG-13).
4. Keys the bust off the black background with a soft luminance key + a
   hole-filled solid silhouette (so black latex stays opaque), then
   un-premultiplies the edge pixels against black (no dark fringe).
5. Crops + upscales (Lanczos, premultiplied) to the fixed window size.
6. Splits the neon lighting out of the art:
     base.png   - "neutral" body: neon/rim lighting desaturated, skin veins
                  turned into dormant grooves
     tint.png   - where the neon/rim lighting was (L8) -> recoloured per mode
     veins.png  - thin vein lines on the skin (L8)   -> pulse with bass
     veins_bloom.png - blurred veins for the halo (L8)
     rim.png    - silhouette rim band (L8)           -> mode rim light
     under_visor.png - synthesized bare face under the visor (RGBA)
     visor_mask.png  - antialiased visor shape (L8, visor-rect sized)
7. Writes layout.json with every anchor in window (sprite) coordinates.

Requires (build only): numpy, pillow, scipy, opencv-python-headless
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(os.path.dirname(ROOT), "concept_v6.jpg")
OUT = os.path.join(ROOT, "assets", "layers")

# ---- geometry of the source art (1280x720 pixel coordinates) -------------
CROP = (398, 0, 874, 720)          # src box that becomes the window
SCALE = 1.06                       # -> ~505 x 763 window
AX = 636.0                         # facial symmetry axis (x)

# visor outline, right half (mirrored for the left half)
VISOR_RIGHT = [(636, 139), (660, 137), (684, 135), (712, 132), (720, 133),
               (723, 140), (723, 156), (721, 166), (712, 176), (692, 184),
               (664, 189), (648, 188), (636, 184)]
EYES = [(598, 158), (674, 158)]     # eye centres under the visor
EYE_W, EYE_H = 30, 12
LIPS = (635.5, 237.5, 31, 16)         # cx, cy, rx, ry (excluded from neon)
RING_PIVOT = (637.0, 312.5)         # bottom of the choker's bail
RING_R = 11.0                       # outer radius of baked O-ring
RING_OFFSET = 9.3                   # pivot -> ring centre
DIALS = {"volume": (472, 384), "balance": (797, 384)}
DIAL_R = 13.5
SLOT_RIGHT = (722, 95, 740, 124)    # viewer's right ear-slot housing
CHEST_HOVER = (498, 418, 774, 640)
PANEL_SLOT_Y = 553                  # seam under the corset band
PANEL_X = (500, 772)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rc = np.where(m, (mx - r) / np.where(m, d, 1), 0)
    gc = np.where(m, (mx - g) / np.where(m, d, 1), 0)
    bc = np.where(m, (mx - b) / np.where(m, d, 1), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = np.where(m, h, 0)
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return np.stack([h, s, mx], -1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


def poly_mask(shape, pts, ss=4):
    """Antialiased polygon mask (float 0..1) at the given HxW."""
    h, w = shape
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).polygon([(x * ss, y * ss) for x, y in pts], fill=255)
    im = im.resize((w, h), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def ellipse_mask(shape, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0


def visor_polygon():
    right = VISOR_RIGHT
    left = [(2 * AX - x, y) for x, y in reversed(right)]
    return right + left


# --------------------------------------------------------------------------
def paint_out_chip(img):
    """Mirror the clean right side of the head over the baked playlist chip."""
    h, w, _ = img.shape
    x0, x1, y0, y1 = 430, 606, 60, 178
    out = img.copy()
    for x in range(x0, x1):
        sx = int(round(2 * AX - x))
        wx = smoothstep(606, 590, x)           # 1 left of the seam, 0 at 606
        for_y = slice(y0, y1)
        wy = np.ones(y1 - y0, np.float32)
        wy *= smoothstep(y0, y0 + 10, np.arange(y0, y1))
        wy *= smoothstep(y1, y1 - 10, np.arange(y0, y1))
        wgt = (wx * wy)[:, None]
        out[for_y, x] = img[for_y, x] * (1 - wgt) + img[for_y, sx] * wgt
    return out


def copper_cables(img):
    """Red cables in the severed areas -> copper/orange."""
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    zone = (yy > 556) | ((yy > 430) & ((xx < 505) | (xx > 768)))
    hsv = rgb_to_hsv(img)
    hue = hsv[..., 0] * 360
    red = ((hue > 335) | (hue < 18)) & (hsv[..., 1] > 0.35) & zone
    amt = smoothstep(0.35, 0.6, hsv[..., 1]) * red
    hsv2 = hsv.copy()
    hsv2[..., 0] = 24 / 360.0
    hsv2[..., 1] = hsv[..., 1] * 0.85
    cop = hsv_to_rgb(hsv2)
    return img * (1 - amt[..., None]) + cop * amt[..., None]


def remove_ring(img):
    """Repaint the baked O-ring + its shadow with neck skin (ring is drawn at
    runtime as a pendulum). Row-wise interpolation between the clean skin on
    either side keeps the choker's lower edge and the neck shading intact."""
    h, w, _ = img.shape
    x0, x1, y0, y1 = 616, 657, 314, 343
    m = ellipse_mask((h, w), 636.5, 324, 17.5, 18)
    m[:y0] = False
    blur = ndi.gaussian_filter(img, (1.5, 1.5, 0))
    fill = img.copy()
    for y in range(y0, y1 + 1):
        cl, cr = blur[y, x0], blur[y, x1]
        t = np.linspace(0, 1, x1 - x0 + 1)[:, None]
        fill[y, x0:x1 + 1] = cl * (1 - t) + cr * t
    fill = ndi.gaussian_filter(fill, (1.2, 0.6, 0))
    rng = np.random.default_rng(3)
    fill = fill + rng.normal(0, 0.008, fill.shape).astype(np.float32)
    feather = ndi.gaussian_filter(m.astype(np.float32), 1.5)[..., None]
    out = img * (1 - feather) + fill * feather
    # the ring's top arc peeks out beside the bail stem just under the choker
    # edge: repaint those rows (keeping the stem) from the clean columns
    # either side so the arc can't 'ghost' while the real ring swings.
    xa, xb, ya, yb = 621, 652, 304, 315
    blur2 = ndi.gaussian_filter(out, (0.6, 0.6, 0))
    m2 = np.zeros((h, w), bool)
    for y in range(ya, yb):
        cl, cr = blur2[y, xa], blur2[y, xb]
        t = np.linspace(0, 1, xb - xa + 1)[:, None]
        row = cl * (1 - t) + cr * t
        for i, x in enumerate(range(xa, xb + 1)):
            if abs(x - 637) > 2:
                out[y, x] = row[i]
                m2[y, x] = True
    # soften the seam between the two repaints
    sm = ndi.gaussian_filter(out, (1.2, 0.8, 0))
    band = np.zeros((h, w), bool)
    band[311:319, 618:656] = True
    band &= np.abs(np.arange(w)[None, :] - 637) > 2
    out = np.where(band[..., None], sm, out)
    return out


# wire / arm-stump zones (src coords): here background shows BETWEEN strands,
# so enclosed dark pockets are opened up instead of being filled
WIRE_ZONE = [(398, 585), (520, 585), (560, 600), (600, 618), (636, 625), (672, 618),
             (712, 600), (752, 585), (874, 585), (874, 720), (398, 720)]
ARM_ZONES = [(398, 480, 500, 720), (772, 480, 874, 720)]


def background_field(img):
    """Smooth estimate of the (hazy, gradient) backdrop behind the bust.

    Seeds = dark pixels connected to the frame border; the field is a
    coarse-to-fine normalized convolution of those, so it is defined under
    the body too and includes the soft purple glow haze around it."""
    lum = img.max(-1)
    lab, _ = ndi.label(lum < 0.085)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    seed = np.isin(lab, border[border > 0])
    seed = ndi.binary_erosion(seed, iterations=2)
    vf = seed.astype(np.float32)
    field = None
    for sg in (60, 25, 10, 4):
        num = ndi.gaussian_filter(img * vf[..., None], (sg, sg, 0))
        den = ndi.gaussian_filter(vf, sg)
        est = num / np.maximum(den, 1e-6)[..., None]
        if field is None:
            field = est
        else:
            wgt = np.clip((den - 0.05) / 0.35, 0, 1)[..., None]
            field = field * (1 - wgt) + est * wgt
    return field, seed


def wire_zone_mask(shape):
    h, w = shape
    m = poly_mask(shape, WIRE_ZONE) > 0.5
    for x0, y0, x1, y1 in ARM_ZONES:
        m[y0:y1, x0:x1] = True
    return m


def make_alpha(img):
    """Matte against the estimated backdrop (not plain luminance).

    * d = colour distance to the backdrop field -> soft edge alpha; the glow
      haze around her is part of the backdrop, so it no longer turns into a
      grimy semi-opaque fringe.
    * solid body: d > t, closed, holes filled (black latex / visor interiors
      are as dark as the backdrop, so holes stay opaque) EXCEPT in the wire /
      arm-stump zones, where backdrop-like pockets >= 3 px wide are opened.
    * outer edge (not wires): contour smoothed (opening), alpha eroded a hair
      and feathered - only the outside boundary, never the interior.
    """
    field, seed = background_field(img)
    d = np.abs(img - field).max(-1)
    lum = img.max(-1)
    soft = smoothstep(0.022, 0.085, d)
    solid = d > 0.05
    # close small gaps so dark enclosed material (latex, ribbed armour) fills
    filled = ndi.binary_fill_holes(ndi.binary_closing(solid, iterations=3)) | solid
    # drop background specks: small blobs away from the dangling wires
    lab, n = ndi.label(filled)
    sizes = ndi.sum(filled, lab, range(1, n + 1))
    cys = ndi.center_of_mass(filled, lab, range(1, n + 1))
    good = [i + 1 for i in range(n) if sizes[i] >= 2500 or (sizes[i] >= 25 and cys[i][0] > 560)]
    filled = np.isin(lab, good)
    solid &= filled
    wz = wire_zone_mask(lum.shape)
    holes = filled & ~solid
    hl, hn = ndi.label(holes)
    if hn:
        idx = range(1, hn + 1)
        md = np.asarray(ndi.mean(d, hl, idx))
        inz = np.asarray(ndi.mean(wz.astype(np.float32), hl, idx))
        wide = ndi.binary_erosion(holes, iterations=1)
        wd = np.asarray(ndi.sum(wide, hl, idx))
        opn = [i + 1 for i in range(hn) if md[i] < 0.03 and inz[i] > 0.6 and wd[i] >= 12]
        pockets = np.isin(hl, opn)
    else:
        pockets = np.zeros_like(holes)
    body = filled & ~pockets
    # closing also bridges concave notches of the OUTER silhouette: there the
    # backdrop-like pixels keep their soft alpha instead of being filled
    band = filled & ~ndi.binary_erosion(filled, iterations=4)
    body &= ~(holes & band & (d < 0.04))
    interior = ndi.binary_erosion(body, iterations=2)
    alpha = np.maximum(soft, ndi.gaussian_filter(interior.astype(np.float32), 0.8))
    alpha *= ndi.binary_dilation(body, iterations=1)
    # wire zones: per-pixel alpha, no blanket fill between strands
    soft_w = smoothstep(0.016, 0.07, d)
    wza = np.where(pockets, 0.0, np.maximum(soft_w, ndi.gaussian_filter(
        ndi.binary_erosion(solid, iterations=1).astype(np.float32), 0.6)))
    wzs = ndi.gaussian_filter(wz.astype(np.float32), 3.0)
    alpha = alpha * (1 - wzs) + np.minimum(alpha, wza) * wzs
    # outer contour: smooth away fuzz bumps (outside the wire zones only)
    h, w = alpha.shape
    disk = ndi.generate_binary_structure(2, 1)
    core = ndi.binary_opening(alpha > 0.4, structure=disk, iterations=2)
    core = ndi.binary_dilation(core, iterations=1)
    shave = ndi.gaussian_filter(core.astype(np.float32), 0.7)
    smooth = np.clip(shave * 1.3, 0, 1)
    alpha = np.where(wzs < 0.5, np.minimum(alpha, smooth), alpha)
    # slight erosion + feather of the OUTER edge only
    edge = ~ndi.binary_erosion(alpha > 0.5, iterations=2)
    er = np.clip((alpha - 0.10) / 0.90, 0, 1)
    alpha = np.where(edge & (wzs < 0.5), er, alpha)
    alpha[alpha < 0.03] = 0
    return np.clip(alpha, 0, 1), interior, field


def unpremultiply(img, alpha, field=None):
    """Colour decontamination of edge pixels: remove the backdrop's share."""
    bg = field if field is not None else (np.array([5, 4, 7], np.float32) / 255)
    a = np.maximum(alpha, 1e-3)[..., None]
    col = (img - bg * (1 - a)) / a
    col = np.clip(col, 0, 1)
    w = smoothstep(0.98, 0.6, alpha)[..., None]  # only touch edge pixels
    return img * (1 - w) + col * w


def synth_under_visor(img):
    """Bare face where the visor sits.

    Normalized convolution of the clean skin around the visor (eyebrows,
    visor rim and hair are excluded by brightness), at growing radii, gives a
    smooth, correctly lit skin field; then a faint warm hollow where the eyes
    sit. Lids, liner, shadow and lashes are painted at runtime (eyes.py)."""
    out = img.copy()
    h, w, _ = img.shape
    poly = visor_polygon()
    vm = poly_mask((h, w), poly) > 0.02
    m = ndi.binary_dilation(vm, iterations=3)
    hsv = rgb_to_hsv(img)
    yy, xx = np.mgrid[0:h, 0:w]
    box = (xx > 520) & (xx < 752) & (yy > 95) & (yy < 230)
    valid = box & ~ndi.binary_dilation(vm, iterations=5) & (hsv[..., 2] > 0.42) & (hsv[..., 1] < 0.55)
    # skip lips/nostrils (dark-ish, saturated) and the bright eject buttons
    valid &= ~((hsv[..., 0] < 0.03) & (hsv[..., 1] > 0.5))
    vf = valid.astype(np.float32)
    fill = None
    for sg in (40, 20, 10, 5):          # coarse to fine, blended by support
        num = ndi.gaussian_filter(img * vf[..., None], (sg, sg, 0))
        den = ndi.gaussian_filter(vf, sg)
        est = num / np.maximum(den, 1e-6)[..., None]
        if fill is None:
            fill = est
        else:
            wgt = np.clip((den - 0.05) / 0.35, 0, 1)[..., None]
            fill = fill * (1 - wgt) + est * wgt
    fill = ndi.gaussian_filter(fill, (2.0, 2.0, 0))
    out = np.where(m[..., None], fill, out)
    # very faint warm hollows where the eyes sit
    shade = np.zeros((h, w), np.float32)
    for ex, ey in EYES:
        d = ((xx - ex) / (EYE_W * 0.85)) ** 2 + ((yy - ey) / (EYE_H * 1.25)) ** 2
        shade = np.maximum(shade, np.exp(-d * 1.6))
    out = out * (1 - 0.10 * shade[..., None]) + np.array([0.32, 0.17, 0.17]) * 0.10 * shade[..., None]
    rng = np.random.default_rng(7)
    out = out + rng.normal(0, 0.008, out.shape) * m[..., None]
    return np.clip(out, 0, 1), m


# --------------------------------------------------------------------------
def to_sprite(arr, size, resample=Image.LANCZOS):
    """Crop to CROP and scale. arr is HxW or HxWxC float32 in src coords."""
    x0, y0, x1, y1 = CROP
    a = arr[y0:y1, x0:x1]
    if a.ndim == 2:
        im = Image.fromarray(a.astype(np.float32), "F")
        return np.asarray(im.resize(size, resample), np.float32)
    chans = [np.asarray(Image.fromarray(a[..., c].astype(np.float32), "F").resize(size, resample), np.float32)
             for c in range(a.shape[2])]
    return np.stack(chans, -1)


def S(pt):
    """src point -> sprite point"""
    return [round((pt[0] - CROP[0]) * SCALE, 2), round((pt[1] - CROP[1]) * SCALE, 2)]


def SR(r):
    a = S((r[0], r[1]))
    b = S((r[2], r[3]))
    return [a[0], a[1], b[0] - a[0], b[1] - a[1]]


def save_l8(name, arr):
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(os.path.join(OUT, name), optimize=True)


def save_rgba(name, rgb, a):
    out = np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)])
    Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA").save(os.path.join(OUT, name), optimize=True)


def find_wire_tips(alpha, n_max=10):
    """Lowest points of the dangling strands = spark anchors (sprite coords)."""
    h, w = alpha.shape
    solid = alpha > 0.5
    low = np.full(w, -1)
    for x in range(w):
        ys = np.nonzero(solid[:, x])[0]
        if len(ys):
            low[x] = ys.max()
    tips = []
    win = 9
    for x in range(win, w - win):
        if low[x] < h * 0.70:
            continue
        seg = low[x - win:x + win + 1]
        if low[x] == seg.max() and (low[x] - seg.min()) > 6:
            if not tips or x - tips[-1][0] > 12:
                tips.append((x, int(low[x])))
            elif low[x] > tips[-1][1]:
                tips[-1] = (x, int(low[x]))
    # arm-stump strands (around y~0.74-0.82 of the height at the sides)
    for xr in [(0, int(w * 0.22)), (int(w * 0.78), w)]:
        sub = low[xr[0]:xr[1]]
        if (sub > 0).any():
            x = xr[0] + int(np.argmax(sub))
            if all(abs(x - t[0]) > 12 for t in tips):
                tips.append((x, int(low[x])))
    tips.sort(key=lambda t: -t[1])
    return [list(map(int, t)) for t in tips[:n_max]]


def main():
    os.makedirs(OUT, exist_ok=True)
    src = np.asarray(Image.open(SRC).convert("RGB"), np.float32) / 255.0
    assert src.shape[:2] == (720, 1280), src.shape
    img = paint_out_chip(src)
    img = remove_ring(img)
    img = copper_cables(img)

    alpha, interior, field = make_alpha(img)
    img = unpremultiply(img, alpha, field)
    face, face_m = synth_under_visor(img)

    W = int(round((CROP[2] - CROP[0]) * SCALE))
    H = int(round((CROP[3] - CROP[1]) * SCALE))
    size = (W, H)

    # premultiplied resize (no fringes)
    pm = to_sprite(img * alpha[..., None], size)
    a = np.clip(to_sprite(alpha, size), 0, 1)
    rgb = np.clip(pm / np.maximum(a, 1e-3)[..., None], 0, 1)
    rgb = np.where(a[..., None] > 1e-3, rgb, 0)
    # gentle unsharp to recover the upscale softness
    blur = ndi.gaussian_filter(rgb, (1.0, 1.0, 0))
    rgb = np.clip(rgb + 0.35 * (rgb - blur), 0, 1)
    a[a < 0.02] = 0

    # ---------------- neon / vein separation (sprite res) ---------------
    hsv = rgb_to_hsv(rgb)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bg = B - G
    val = hsv[..., 2]
    neon = smoothstep(0.035, 0.20, bg) * smoothstep(0.05, 0.25, val)
    yy, xx = np.mgrid[0:H, 0:W]

    def sp_ellipse(cx, cy, rx, ry):
        c = S((cx, cy))
        return ((xx - c[0]) / (rx * SCALE)) ** 2 + ((yy - c[1]) / (ry * SCALE)) ** 2 <= 1

    lips = ndi.gaussian_filter(sp_ellipse(*LIPS).astype(np.float32), 2)
    neon *= (1 - lips)
    vpoly = [S(p) for p in visor_polygon()]
    vmask = poly_mask((H, W), vpoly)
    neon *= (1 - ndi.gaussian_filter(vmask, 1.0))
    # eject buttons stay red: exclude the slot housings from the tint
    for r in (SLOT_RIGHT, (2 * AX - SLOT_RIGHT[2], SLOT_RIGHT[1], 2 * AX - SLOT_RIGHT[0], SLOT_RIGHT[3])):
        x, y, ww, hh = SR(r)
        neon[int(y):int(y + hh), int(x):int(x + ww)] *= 0.25

    # skin zone: pale, low-ish saturation, closed + hole filled, above corset
    skin = (val > 0.36) & (hsv[..., 1] < 0.55) & (R > G)
    lim_y = S((0, 500))[1]
    skin &= yy < lim_y
    skin = ndi.binary_opening(skin, iterations=1)
    skin = ndi.binary_closing(skin, iterations=5)
    skin = ndi.binary_fill_holes(skin)
    lab, n = ndi.label(skin)
    sizes = ndi.sum(skin, lab, range(1, n + 1))
    skin = np.isin(lab, 1 + np.nonzero(sizes >= 2000)[0])
    skin = ndi.binary_erosion(skin, iterations=1)
    skin &= vmask < 0.05
    skinf = ndi.gaussian_filter(skin.astype(np.float32), 1.0)

    # veins = thin magenta ridges on the skin
    ridge = bg - ndi.gaussian_filter(bg, 4.0)
    veins = smoothstep(0.012, 0.07, ridge) * smoothstep(0.02, 0.10, bg) * skinf
    veins = np.clip(ndi.gaussian_filter(veins, 0.5) * 1.25, 0, 1) * (1 - lips)
    bloom = ndi.gaussian_filter(veins, 3.5)
    bloom = bloom / max(bloom.max(), 1e-3)

    # neutral base: veins -> dormant grooves, other neon -> desaturated
    skin_col = ndi.gaussian_filter(rgb * (1 - veins[..., None]), (5, 5, 0)) / \
        np.maximum(ndi.gaussian_filter(1 - veins, 5), 1e-3)[..., None]
    groove = skin_col * 0.80
    base = rgb * (1 - veins[..., None]) + groove * veins[..., None]
    tint = neon * (1 - skinf * 0.85)       # skin keeps only a little tint (rim)
    lum = (0.30 * R + 0.55 * G + 0.15 * B)
    lum = np.maximum(lum, val * 0.75)      # neon pixels: keep their brightness
    gray = np.repeat(lum[..., None], 3, -1)
    base = base * (1 - tint[..., None]) + gray * tint[..., None]

    # rim light band: inner edge of the silhouette, strongest on the sides
    solid = a > 0.5
    dist = ndi.distance_transform_edt(solid)
    rim = smoothstep(7.0, 1.0, dist) * solid
    side = np.abs(xx - S((AX, 0))[0]) / (W * 0.5)
    rim = rim * np.clip(0.35 + side, 0, 1)
    rim *= (1 - vmask)
    rim[yy > S((0, 600))[1]] *= 0.3       # not on the dangling wires
    rim = ndi.gaussian_filter(rim, 0.8)

    save_rgba("base.png", base, a)
    save_l8("tint.png", tint)
    save_l8("veins.png", veins)
    save_l8("veins_bloom.png", bloom)
    save_l8("rim.png", rim)

    # ---------------- visor layers ----------------
    vys, vxs = np.nonzero(vmask > 0.01)
    vx0, vy0, vx1, vy1 = vxs.min() - 2, vys.min() - 2, vxs.max() + 3, vys.max() + 3
    save_l8("visor_mask.png", vmask[vy0:vy1, vx0:vx1])
    face_s = np.clip(to_sprite(face, size), 0, 1)
    fm = ndi.gaussian_filter(np.clip(to_sprite(face_m.astype(np.float32), size), 0, 1), 1.0)
    save_rgba("under_visor.png", face_s[vy0:vy1, vx0:vx1], fm[vy0:vy1, vx0:vx1])

    # ---------------- layout ----------------
    slot_r = SR(SLOT_RIGHT)
    slot_l = SR((2 * AX - SLOT_RIGHT[2], SLOT_RIGHT[1], 2 * AX - SLOT_RIGHT[0], SLOT_RIGHT[3]))
    layout = {
        "size": [W, H],
        "scale": SCALE,
        "axis_x": S((AX, 0))[0],
        "visor": {"rect": [int(vx0), int(vy0), int(vx1 - vx0), int(vy1 - vy0)],
                  "polygon": vpoly},
        "eyes": [S(e) + [round(EYE_W * SCALE, 1), round(EYE_H * SCALE, 1)] for e in EYES],
        "ring": {"pivot": S(RING_PIVOT), "radius": round(RING_R * SCALE, 2),
                 "offset": round(RING_OFFSET * SCALE, 2)},
        "dials": {k: S(v) + [round(DIAL_R * SCALE, 2)] for k, v in DIALS.items()},
        "chest_hover": SR(CHEST_HOVER),
        "panel": {"slot_y": S((0, PANEL_SLOT_Y))[1],
                  "x0": S((PANEL_X[0], 0))[0], "x1": S((PANEL_X[1], 0))[0]},
        # ear slots: 'left' = viewer's left (her right) = playlist chip (TODO stage 2)
        "ear_slots": {"left": slot_l, "right": slot_r},
        "wire_tips": find_wire_tips(a),
    }
    with open(os.path.join(OUT, "layout.json"), "w") as f:
        json.dump(layout, f, indent=1)

    # app icon: the cleaned head (original neon colours), square, multi-size
    x0, y0, side = 520, 0, 232
    head = np.dstack([img[y0:y0 + side, x0:x0 + side], alpha[y0:y0 + side, x0:x0 + side]])
    ic = Image.fromarray((np.clip(head, 0, 1) * 255).astype(np.uint8), "RGBA").resize((256, 256), Image.LANCZOS)
    ic.save(os.path.join(ROOT, "assets", "vexa.ico"),
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    ic.save(os.path.join(ROOT, "assets", "vexa_icon.png"))
    print("sprite", size, "visor", layout["visor"]["rect"], "tips", len(layout["wire_tips"]))


if __name__ == "__main__":
    main()
