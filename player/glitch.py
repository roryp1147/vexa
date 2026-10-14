"""
Easter-egg glitches, applied to the fully composed window frame (numpy on a
premultiplied ARGB32 QImage; only runs for the few frames a glitch lasts).

  * data tear  - forehead click: ~0.35 s of horizontal slice offsets and an
                 RGB split, with a stuttering envelope
  * corrupted  - very rare (about once per few hours of playback): her face
                 turns to static for a single frame or two
"""
from __future__ import annotations

import random

import numpy as np
from PySide6.QtGui import QImage

TEAR_DUR = 0.35
CORRUPT_DUR = 0.07           # ~2 frames at 30 fps
CORRUPT_MEAN_S = 3.0 * 3600  # mean seconds of playback between corrupted frames


def _view(img: QImage) -> np.ndarray:
    """Writable HxWx4 (B, G, R, A) view of a Format_ARGB32_Premultiplied image."""
    h, w = img.height(), img.width()
    buf = np.frombuffer(img.bits(), np.uint8, count=img.sizeInBytes())
    return buf.reshape(h, img.bytesPerLine() // 4, 4)[:, :w]


def _fix_alpha(a):
    """Keep premultiplied pixels valid after channels moved apart (A >= R,G,B).
    (element-wise maximum: ~10x faster than a reduce over the channel axis)"""
    np.maximum(a[..., 3], np.maximum(np.maximum(a[..., 0], a[..., 1]), a[..., 2]), out=a[..., 3])


def _shift_x(a, dx):
    if dx == 0:
        return a
    out = np.zeros_like(a)
    if dx > 0:
        out[:, dx:] = a[:, :-dx]
    else:
        out[:, :dx] = a[:, -dx:]
    return out


class Glitch:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.tear_t0 = None
        self.corrupt_t0 = None
        self.frames = 0

    def start_tear(self, now):
        self.tear_t0 = now

    def start_corrupt(self, now):
        self.corrupt_t0 = now

    def active(self, now) -> bool:
        if self.tear_t0 is not None and now - self.tear_t0 > TEAR_DUR:
            self.tear_t0 = None
        if self.corrupt_t0 is not None and now - self.corrupt_t0 > CORRUPT_DUR:
            self.corrupt_t0 = None
        return self.tear_t0 is not None or self.corrupt_t0 is not None

    def maybe_corrupt(self, now, dt) -> bool:
        if self.rng.random() < dt / CORRUPT_MEAN_S:
            self.start_corrupt(now)
            return True
        return False

    # ------------------------------------------------------------------ apply
    def process(self, img: QImage, now, face_rect, tint):
        self.frames += 1
        a = _view(img)
        if self.corrupt_t0 is not None:
            self._corrupt(a, face_rect, tint)
        if self.tear_t0 is not None:
            e = (now - self.tear_t0) / TEAR_DUR
            # stuttering envelope: hits hard, flickers, cuts out
            env = 1.0 if e < 0.35 else (0.35 if int(e * 14) % 2 else 0.9)
            self._tear(a, env)

    def _tear(self, a, env):
        rng = self.rng
        h, w = a.shape[:2]
        n = rng.randint(5, 10)
        for _ in range(n):
            y0 = rng.randrange(0, h - 4)
            bh = rng.choice((2, 3, 4, 6, 9, 14, 22, 30))
            y1 = min(h, y0 + bh)
            dx = int(rng.choice((-1, 1)) * rng.uniform(3, 22) * env)
            band = a[y0:y1].copy()
            band = _shift_x(band, dx)
            split = int(rng.uniform(2, 5) * env) or 1
            # premultiplied BGRA: R one way, B the other; alpha covers all channels
            band[..., 2] = _shift_x(band[..., 2:3], split)[..., 0]
            band[..., 0] = _shift_x(band[..., 0:1], -split)[..., 0]
            _fix_alpha(band)
            if rng.random() < 0.25:
                # smeared row (a "stuck" scanline dragged down the band)
                band[:] = band[rng.randrange(0, band.shape[0])][None]
            a[y0:y1] = band
        # thin full-width RGB split across the whole figure on the strong frames
        if env > 0.8:
            r = _shift_x(a[..., 2:3], 2)[..., 0]
            b = _shift_x(a[..., 0:1], -2)[..., 0]
            a[..., 2] = r
            a[..., 0] = b
            _fix_alpha(a)

    def _corrupt(self, a, r, tint):
        """Static-filled face for a frame: features still ghosting through,
        soft elliptical falloff, rows sliced sideways, mode-tinted snow."""
        rng = self.rng
        x0, y0 = max(0, r.x()), max(0, r.y())
        x1, y1 = min(a.shape[1], r.right() + 1), min(a.shape[0], r.bottom() + 1)
        reg = a[y0:y1, x0:x1].astype(np.float32)
        hh, ww = reg.shape[:2]
        # sliced rows first (the picture loses sync), then the snow on top
        for _ in range(rng.randint(4, 8)):
            yy = rng.randrange(0, hh - 3)
            bh = rng.randint(2, 9)
            reg[yy:yy + bh] = _shift_x(reg[yy:yy + bh], rng.randint(-12, 12))
        alpha = reg[..., 3:4] / 255.0
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        cx, cy = ww / 2, hh * 0.56
        d = ((xx - cx) / (ww * 0.40)) ** 2 + ((yy - cy) / (hh * 0.47)) ** 2
        mask = np.clip((1.15 - d) / 0.5, 0, 1)[..., None]
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        noise = nrng.random((hh, ww, 1)).astype(np.float32)
        rows = nrng.random((hh, 1, 1)).astype(np.float32)
        snow = np.clip(0.7 * noise + 0.3 * rows, 0, 1)
        col = np.array([tint[2], tint[1], tint[0]], np.float32) / 255.0
        static = snow * (0.45 + 0.55 * col) * 255.0
        k = 0.68 * mask
        reg[..., :3] = reg[..., :3] * (1 - k) + static * k * alpha
        out = np.clip(reg, 0, 255).astype(np.uint8)
        _fix_alpha(out)
        a[y0:y1, x0:x1] = out
