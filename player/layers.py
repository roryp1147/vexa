"""
Runtime art layers: loads the pre-built PNGs from assets/layers and turns them
into cached QPixmaps. All per-pixel work happens ONCE here (numpy, vectorised);
per frame we only blit pixmaps.

  body(mode)   - full sprite relit for a visor mode (neon tint + rim light)
  veins(mode)  - additive vein-glow sprite (cropped to its bbox), drawn with
                 CompositionMode_Plus at an opacity driven by the bass
"""
from __future__ import annotations

import json
import os

import numpy as np
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QImage, QPixmap

from . import config
from .qtutil import premul_to_pixmap, qimage_to_gray, qimage_to_rgba


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# Scanning eye: veins in ultraviolet / lavender. Additive light on pale skin
# washes toward white, so the colour is kept saturated (low green), the gain
# lower and the hot cores tinted (silvery lavender) instead of white.
VEIN_COLOUR = {"scan": (168, 92, 255)}
VEIN_SECONDARY_MIX = {"scan": 0.0}
VEIN_GAIN = {"scan": 0.95}
VEIN_CORE = {"scan": 0.22}
# a little brushed silver mixed into the neon relight (scan mode)
RELIGHT_SILVER = {"scan": 0.22}
SILVER = np.array([0.80, 0.80, 0.88], np.float32)


class Art:
    def __init__(self, layer_dir: str = config.LAYERS):
        self.dir = layer_dir
        with open(os.path.join(layer_dir, "layout.json")) as f:
            self.layout = json.load(f)
        self.W, self.H = self.layout["size"]
        base = qimage_to_rgba(QImage(self._p("base.png")))
        self.rgb = base[..., :3]
        self.alpha = base[..., 3]
        self.tint = qimage_to_gray(QImage(self._p("tint.png")))
        self.rim = qimage_to_gray(QImage(self._p("rim.png")))
        veins = qimage_to_gray(QImage(self._p("veins.png")))
        bloom = qimage_to_gray(QImage(self._p("veins_bloom.png")))
        glow = np.clip(veins + 0.55 * bloom, 0, 1)
        ys, xs = np.nonzero(glow > 0.004)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        self._vein_rect = QRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        self._veins = veins[y0:y1, x0:x1]
        self._glow = glow[y0:y1, x0:x1]
        # luminance of the neutral base (for colourising the neon parts)
        r, g, b = self.rgb[..., 0], self.rgb[..., 1], self.rgb[..., 2]
        self.lum = 0.30 * r + 0.55 * g + 0.15 * b
        self._body_cache: dict[int, QPixmap] = {}
        self._vein_cache: dict[int, QPixmap] = {}

        self.under_visor = QPixmap(self._p("under_visor.png"))
        m = QImage(self._p("visor_mask.png")).convertToFormat(QImage.Format.Format_Grayscale8)
        self.visor_mask = self._gray_to_alpha8(m)
        self.visor_mask_arr = qimage_to_gray(m)
        self.hit_alpha = (self.alpha * 255).astype(np.uint8)   # for hit tests

    def _p(self, name):
        return os.path.join(self.dir, name)

    @staticmethod
    def _gray_to_alpha8(g: QImage) -> QImage:
        w, h = g.width(), g.height()
        arr = np.frombuffer(g.constBits(), np.uint8).reshape(h, g.bytesPerLine())[:, :w]
        arr = np.ascontiguousarray(arr)
        return QImage(arr.data, w, h, w, QImage.Format.Format_Alpha8).copy()

    # ------------------------------------------------------------------
    def body(self, mi: int) -> QPixmap:
        pm = self._body_cache.get(mi)
        if pm is None:
            pm = self._compose_body(config.MODES[mi])
            self._body_cache[mi] = pm
        return pm

    def unlit(self) -> QPixmap:
        """The neutral, unpowered body (neon dormant) - used by the boot sequence."""
        pm = self._body_cache.get("unlit")
        if pm is None:
            a = self.alpha
            pm = premul_to_pixmap(self.rgb * 0.8 * a[..., None], a)
            self._body_cache["unlit"] = pm
        return pm

    def _compose_body(self, mode) -> QPixmap:
        out = self._relight(mode, (slice(None), slice(None)))
        a = self.alpha
        return premul_to_pixmap(out * a[..., None], a)

    def _relight(self, mode, sl):
        """Straight-alpha relit RGB for a region (sl = (rows, cols) slices)."""
        p = np.array(mode.primary, np.float32) / 255
        s = np.array(mode.secondary, np.float32) / 255
        lum = self.lum[sl][..., None]
        k = _smooth(0.06, 0.55, lum)
        col = (s * (1 - k) + p * k) * np.clip(lum * 2.1, 0, 1.25)
        hot = _smooth(0.55, 0.95, lum)                   # speculars go white-hot
        col = col * (1 - 0.55 * hot) + 0.55 * hot
        sv = RELIGHT_SILVER.get(mode.key, 0.0)
        if sv:                                           # silvery sheen on the neon parts
            col = col * (1 - sv * lum) + SILVER * lum * sv * 1.25
        t = self.tint[sl][..., None]
        out = self.rgb[sl] * (1 - t) + np.clip(col, 0, 1) * t
        out = out + self.rim[sl][..., None] * p * 0.42   # mode rim light
        out = out * (0.93 + 0.07 * p / max(p.max(), 1e-3))  # faint overall grade
        return np.clip(out, 0, 1)

    def body_region(self, mi: int, rect: QRect) -> np.ndarray:
        """Premultiplied float RGBA of the relit body inside rect (for face rigs)."""
        sl = (slice(rect.y(), rect.y() + rect.height()), slice(rect.x(), rect.x() + rect.width()))
        a = self.alpha[sl]
        return np.dstack([self._relight(config.MODES[mi], sl) * a[..., None], a]).astype(np.float32)

    def vein_region(self, rect: QRect):
        """(glow, veins) float arrays for rect in body coords (zero outside the vein bbox)."""
        h, w = rect.height(), rect.width()
        g = np.zeros((h, w), np.float32)
        v = np.zeros((h, w), np.float32)
        vr = self._vein_rect
        ix = rect.intersected(vr)
        if not ix.isEmpty():
            dy, dx = ix.y() - rect.y(), ix.x() - rect.x()
            sy, sx = ix.y() - vr.y(), ix.x() - vr.x()
            g[dy:dy + ix.height(), dx:dx + ix.width()] = self._glow[sy:sy + ix.height(), sx:sx + ix.width()]
            v[dy:dy + ix.height(), dx:dx + ix.width()] = self._veins[sy:sy + ix.height(), sx:sx + ix.width()]
        return g, v

    def vein_colour(self, mi: int, glow: np.ndarray, veins: np.ndarray):
        """-> (premul rgb, alpha) of the additive vein sprite for given glow/vein masks."""
        mode = config.MODES[mi]
        p = np.array(VEIN_COLOUR.get(mode.key, mode.primary), np.float32) / 255
        mx = VEIN_SECONDARY_MIX.get(mode.key, 0.0)
        if mx:
            p = p * (1 - mx) + np.array(mode.secondary, np.float32) / 255 * mx
        g = glow[..., None]
        core = (veins ** 2)[..., None] * VEIN_CORE.get(mode.key, 0.45)   # white-hot cores
        rgb = np.clip(g * p * VEIN_GAIN.get(mode.key, 1.15) + core * p ** 0.35, 0, 1) \
            if mode.key in VEIN_COLOUR else np.clip(g * p * 1.15 + core, 0, 1)
        # additive light: alpha must cover the brightest channel (valid premul)
        return rgb, np.clip(np.maximum(glow, rgb.max(-1)), 0, 1)

    def veins(self, mi: int) -> tuple[QPixmap, QPoint]:
        pm = self._vein_cache.get(mi)
        if pm is None:
            rgb, a = self.vein_colour(mi, self._glow, self._veins)
            # never glow outside her silhouette (no halo on light wallpapers)
            vr = self._vein_rect
            body = self.alpha[vr.y():vr.y() + vr.height(), vr.x():vr.x() + vr.width()]
            pm = premul_to_pixmap(rgb * body[..., None], a * body)
            self._vein_cache[mi] = pm
        return pm, self._vein_rect.topLeft()

    @property
    def vein_rect(self) -> QRect:
        return QRect(self._vein_rect)

    def warm(self):
        """Build every mode's pixmaps (call from an idle slot)."""
        for i in range(len(config.MODES)):
            self.body(i)
            self.veins(i)

    def opaque_at(self, x: int, y: int, thresh: int = 24) -> bool:
        if 0 <= x < self.W and 0 <= y < self.H:
            return self.hit_alpha[y, x] >= thresh
        return False
