"""
Visor visualizer: 4 modes rendered into a small offscreen image (visor-rect
sized, ~193x69), masked by the antialiased visor shape, then blitted.

  0 Spectrum  - mirrored segmented bar EQ (bass in the middle), peak caps
  1 Waveform  - oscilloscope with phosphor persistence
  2 Matrix    - falling procedural glyphs (no font needed)
  3 Scan eye  - visor goes see-through: glowing eyes, pulsing irises, sweep

Mode changes run a clunky shutter-wipe: two slatted blades slam shut,
the mode flips while closed, then they ratchet open.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (QColor, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF, QRadialGradient)

from . import config
from .eyes import EyeArtist, EyeLife
from .qtutil import qc, mix

SHUT_CLOSE = 0.16
SHUT_HOLD = 0.07
SHUT_OPEN = 0.17
SHUT_TOTAL = SHUT_CLOSE + SHUT_HOLD + SHUT_OPEN


class Visor:
    def __init__(self, art):
        L = art.layout["visor"]
        x, y, w, h = L["rect"]
        self.rect = QRect(x, y, w, h)
        self.w, self.h = w, h
        self.mask = art.visor_mask
        m = art.visor_mask_arr
        self.under = art.under_visor
        # per-column top/bottom of the visible visor (for bar baselines etc.)
        cols = m > 0.5
        self.top = np.array([np.argmax(cols[:, i]) if cols[:, i].any() else h // 2 for i in range(w)])
        self.bot = np.array([h - 1 - np.argmax(cols[::-1, i]) if cols[:, i].any() else h // 2 for i in range(w)])
        xs = np.nonzero(cols.any(0))[0]
        self.x0, self.x1 = int(xs.min()), int(xs.max())
        self.path = QPainterPath()
        poly = QPolygonF([QPointF(px, py) for px, py in L["polygon"]])
        self.path.addPolygon(poly)
        self.path.closeSubpath()
        self.eyes = [(ex - x, ey - y, ew, eh) for ex, ey, ew, eh in art.layout["eyes"]]
        self.img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        self.phosphor = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        self.phosphor.fill(0)
        self.mode = 0
        self.pending = None
        self.shut_t0 = None
        self.flipped = False
        self.scan_x = 0.0
        self.scan_dir = 1
        self.t = 0.0
        self._overlays: dict[int, QPixmap] = {}
        self._glyphs = self._build_glyphs()
        self.eye_artist = None          # built lazily on first scan-mode frame
        self.eye_life = EyeLife()
        self.heart_m = 0.0              # heart pupil morph: 0 outline .. 1 full
        self._bass_slow = 0.0
        self.next_flicker = 45.0        # visor clock: earliest random invert flicker
        self.flicker_cool = 0.0         # earliest flicker of any kind
        self._flick_rng = random.Random(23)
        self._init_matrix()

    # ------------------------------------------------------------------ state
    @property
    def transitioning(self) -> bool:
        return self.shut_t0 is not None

    def start_transition(self, new_mode: int, now: float) -> bool:
        if self.transitioning:
            return False
        self.pending = new_mode % len(config.MODES)
        self.shut_t0 = now
        self.flipped = False
        return True

    def step(self, now: float, dt: float, an) -> bool:
        """Advance animation state. Returns True when the mode flipped."""
        self.t += dt
        flipped = False
        if self.shut_t0 is not None:
            e = now - self.shut_t0
            if e >= SHUT_CLOSE and not self.flipped:
                self.mode = self.pending
                self.flipped = True
                flipped = True
                self.phosphor.fill(0)
            if e >= SHUT_TOTAL:
                self.shut_t0 = None
                self.pending = None
        # scan line ping-pong: slow and soft (a touch faster when loud)
        self.eye_life.step(dt)
        self._step_hearts(dt, an)
        speed = (22 + 30 * an.level) * dt
        self.scan_x += speed * self.scan_dir
        if self.scan_x > self.x1 - 3:
            self.scan_x, self.scan_dir = self.x1 - 3, -1
        elif self.scan_x < self.x0 + 3:
            self.scan_x, self.scan_dir = self.x0 + 3, 1
        if self.mode == 2:
            self._step_matrix(dt, an)
        return flipped

    def _step_hearts(self, dt, an):
        """Heart pupils follow the bass (fast attack, softer release); strong
        drops (and, rarely, just time) trigger the invert flicker."""
        target = max(0.0, min(1.0, (an.bass - 0.12) / 0.62))
        k = 1 - math.exp(-dt * (22.0 if target > self.heart_m else 6.0))
        self.heart_m += (target - self.heart_m) * k
        drop = an.bass - self._bass_slow
        self._bass_slow += (an.bass - self._bass_slow) * (1 - math.exp(-dt * 1.2))
        if self.mode != 3 or self.eye_life.flickering or self.t < self.flicker_cool:
            return
        loud = not getattr(an, "silent", False)
        if (an.bass > 0.82 and drop > 0.38 and self._flick_rng.random() < 0.35) or \
                (loud and self.t >= self.next_flicker):
            self.trigger_flicker()

    def trigger_flicker(self):
        self.eye_life.flicker()
        self.flicker_cool = self.t + 14.0
        self.next_flicker = self.t + self._flick_rng.uniform(40.0, 95.0)

    def closure(self, now: float) -> float:
        """0 = blades open, 1 = fully shut."""
        if self.shut_t0 is None:
            return 0.0
        e = now - self.shut_t0
        if e < SHUT_CLOSE:
            u = e / SHUT_CLOSE
            return u * u                                 # accelerate into the slam
        e -= SHUT_CLOSE
        if e < SHUT_HOLD:
            return 1.0 - 0.07 * abs(math.sin(e / SHUT_HOLD * math.pi * 2)) * (1 - e / SHUT_HOLD)
        e -= SHUT_HOLD
        u = min(1.0, e / SHUT_OPEN)
        # ratchet open in 3 hitches
        steps = 3
        k = math.floor(u * steps)
        f = u * steps - k
        f = min(1.0, f * 1.6)
        return 1.0 - (k + f * f * (3 - 2 * f)) / steps

    # ------------------------------------------------------------------ render
    def render(self, an, now: float, fx=None) -> QImage:
        mi = self.mode
        m = config.MODES[mi]
        img = self.img
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if mi == 3:
            self._draw_scan(p, an, m)
        else:
            self._draw_glass(p, m)
            if mi == 0:
                self._draw_spectrum(p, an, m)
            elif mi == 1:
                self._draw_wave(p, an, m)
            else:
                self._draw_matrix(p, an, m)
            p.drawPixmap(0, 0, self._overlay(mi))
        c = self.closure(now)
        if c > 0.001:
            self._draw_shutter(p, c, config.MODES[self.pending if (self.pending is not None and not self.flipped) else mi])
        if fx:
            self._apply_fx(p, fx, m, mi)
        # cut to the visor silhouette
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        p.drawImage(0, 0, self.mask)
        p.end()
        return img

    def paint_frame(self, p: QPainter, mi: int):
        """Antialiased rim around the visor (cached pixmap per mode)."""
        key = ("frame", mi)
        pm = self._overlays.get(key)
        if pm is None:
            r = self.rect.adjusted(-3, -3, 3, 3)
            pm = QPixmap(r.size())
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.translate(-r.x(), -r.y())
            self._paint_frame(q, mi)
            q.end()
            self._overlays[key] = pm
        p.drawPixmap(self.rect.x() - 3, self.rect.y() - 3, pm)

    def _paint_frame(self, p: QPainter, mi: int):
        m = config.MODES[mi]
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(8, 8, 10, 200), 2.2))
        p.drawPath(self.path)
        col = qc(mix((150, 156, 168), m.primary, 0.35), 210)
        p.setPen(QPen(col, 1.0))
        p.drawPath(self.path)
        # top-edge chrome glint
        r = self.rect
        g = QLinearGradient(r.left(), 0, r.right(), 0)
        g.setColorAt(0.0, QColor(255, 255, 255, 0))
        g.setColorAt(0.35, QColor(255, 255, 255, 150))
        g.setColorAt(0.5, QColor(255, 255, 255, 40))
        g.setColorAt(0.68, QColor(255, 255, 255, 120))
        g.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setPen(QPen(g, 0.9))
        p.setClipRect(QRectF(r.left(), r.top(), r.width(), 8))
        p.drawPath(self.path)
        p.restore()

    # -- glass background + static overlay (scanlines / segments / gloss)
    def _draw_glass(self, p, m):
        g = QLinearGradient(0, 0, 0, self.h)
        gl = m.glass
        g.setColorAt(0.0, qc((gl[0] * 1.8 + 6, gl[1] * 1.8 + 6, gl[2] * 1.8 + 8)))
        g.setColorAt(0.55, qc(gl))
        g.setColorAt(1.0, QColor(2, 2, 3))
        p.fillRect(0, 0, self.w, self.h, g)

    def _overlay(self, mi) -> QPixmap:
        pm = self._overlays.get(mi)
        if pm is not None:
            return pm
        m = config.MODES[mi]
        pm = QPixmap(self.w, self.h)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        if mi == 0:
            # LED segment gaps
            for y in range(0, self.h, 3):
                p.fillRect(0, y, self.w, 1, QColor(0, 0, 0, 150))
        else:
            for y in range(0, self.h, 2):
                p.fillRect(0, y, self.w, 1, QColor(0, 0, 0, 55))
        if mi == 1:
            # oscilloscope graticule
            pen = QPen(qc(m.primary, 30), 1)
            p.setPen(pen)
            for x in range(self.x0 + 8, self.x1, 16):
                p.drawLine(x, 0, x, self.h)
            mid = int(np.median((self.top + self.bot) / 2))
            p.setPen(QPen(qc(m.primary, 55), 1))
            p.drawLine(self.x0, mid, self.x1, mid)
        # gloss: soft reflection band across the top + a diagonal glint
        g = QLinearGradient(0, 0, 0, self.h)
        g.setColorAt(0.0, QColor(255, 255, 255, 46))
        g.setColorAt(0.28, QColor(255, 255, 255, 10))
        g.setColorAt(0.3, QColor(255, 255, 255, 0))
        p.fillRect(0, 0, self.w, self.h, g)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.moveTo(self.w * 0.18, 0)
        path.lineTo(self.w * 0.26, 0)
        path.lineTo(self.w * 0.16, self.h)
        path.lineTo(self.w * 0.10, self.h)
        path.closeSubpath()
        p.fillPath(path, QColor(255, 255, 255, 14))
        # edge vignette
        vg = QLinearGradient(0, 0, self.w, 0)
        vg.setColorAt(0.0, QColor(0, 0, 0, 160))
        vg.setColorAt(0.1, QColor(0, 0, 0, 0))
        vg.setColorAt(0.9, QColor(0, 0, 0, 0))
        vg.setColorAt(1.0, QColor(0, 0, 0, 160))
        p.fillRect(0, 0, self.w, self.h, vg)
        p.end()
        self._overlays[mi] = pm
        return pm

    # -- 0: spectrum
    def _draw_spectrum(self, p, an, m):
        bands = an.bands
        peaks = an.peaks
        n = len(bands)
        span = self.x1 - self.x0 - 4
        cx = (self.x0 + self.x1) / 2
        bw = span / (2 * n)
        g = QLinearGradient(0, self.h, 0, 0)
        g.setColorAt(0.0, qc(m.secondary))
        g.setColorAt(0.55, qc(m.primary))
        g.setColorAt(1.0, qc(mix(m.primary, (255, 255, 255), 0.55)))
        p.setPen(Qt.PenStyle.NoPen)
        cap = qc(mix(m.primary, (255, 255, 255), 0.7))
        breathe = 0.04 + 0.03 * math.sin(self.t * 2.0)       # idle shimmer
        for side in (-1, 1):
            for i in range(n):
                x = cx + side * (i * bw) + (0 if side > 0 else -bw)
                xi = int(min(self.w - 1, max(0, x + bw / 2)))
                top, bot = self.top[xi] + 2, self.bot[xi] - 1
                hh = bot - top
                v = max(float(bands[i]), breathe)
                bh = v * hh
                p.fillRect(QRectF(x + 0.6, bot - bh, bw - 1.4, bh), g)
                py = bot - float(peaks[i]) * hh
                if peaks[i] > 0.03:
                    p.fillRect(QRectF(x + 0.6, py - 1.5, bw - 1.4, 1.5), cap)
        # bass bloom from the bottom centre
        rg = QRadialGradient(QPointF(cx, self.h), self.w * 0.45)
        rg.setColorAt(0, qc(m.primary, int(90 * an.bass)))
        rg.setColorAt(1, qc(m.primary, 0))
        p.fillRect(0, 0, self.w, self.h, rg)

    # -- 1: waveform with phosphor persistence
    def _draw_wave(self, p, an, m):
        ph = self.phosphor
        q = QPainter(ph)
        q.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        q.fillRect(0, 0, self.w, self.h, QColor(0, 0, 0, 120))   # fade old trace
        q.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        wave = np.clip(an.wave, -1.0, 1.0)
        n = len(wave)
        if getattr(self, "_wave_x", None) is None or len(self._wave_x) != n:
            xs = np.linspace(self.x0 + 3, self.x1 - 3, n)
            xi = xs.astype(int)
            self._wave_x = xs
            self._wave_mid = (self.top[xi] + self.bot[xi]) / 2.0
            self._wave_amp = (self.bot[xi] - self.top[xi]) / 2.0 * 0.86
            self._wave_ph = np.arange(n) * 0.35
        v = wave + 0.03 * np.sin(self.t * 3 + self._wave_ph)      # idle: faint living line
        ys = self._wave_mid - v * self._wave_amp
        poly = QPolygonF([QPointF(x, y) for x, y in zip(self._wave_x.tolist(), ys.tolist())])
        # Cheap glow: the same polyline stroked with 1px antialiased
        # (cosmetic) pens at small vertical offsets. Wide stroked pens cost
        # ~10x more in the raster engine.
        q.setBrush(Qt.BrushStyle.NoBrush)
        layers = ((-2.0, qc(m.secondary, 60)), (2.0, qc(m.secondary, 60)),
                  (-1.0, qc(m.primary, 150)), (1.0, qc(m.primary, 150)),
                  (-0.4, qc(m.primary, 255)), (0.4, qc(m.primary, 255)),
                  (0.0, qc(mix(m.primary, (255, 255, 255), 0.7), 230)))
        for dy, col in layers:
            q.setPen(QPen(col, 0))
            q.translate(0, dy)
            q.drawPolyline(poly)
            q.translate(0, -dy)
        q.end()
        p.drawImage(0, 0, ph)

    # -- 2: matrix rain
    def _build_glyphs(self):
        rng = random.Random(1337)
        self.gw, self.gh = 4, 6
        n = 48
        glyphs = []
        while len(glyphs) < n:
            bits = [[1 if rng.random() < 0.45 else 0 for _ in range(self.gw)] for _ in range(self.gh)]
            # katakana-ish: give every glyph a stroke spine
            if rng.random() < 0.5:
                c = rng.randrange(self.gw)
                for r in range(self.gh):
                    bits[r][c] = 1
            else:
                r0 = rng.randrange(self.gh)
                for c in range(self.gw):
                    bits[r0][c] = 1
            if 7 <= sum(map(sum, bits)) <= 15:
                glyphs.append(bits)
        self.n_glyphs = n
        return glyphs

    def _glyph_atlas(self, m) -> QPixmap:
        key = ("atlas", m.key)
        pm = self._overlays.get(key)
        if pm is not None:
            return pm
        tiers = [mix(m.primary, (255, 255, 255), 0.75), m.primary,
                 mix(m.primary, m.secondary, 0.5), m.secondary, mix(m.secondary, (0, 0, 0), 0.5)]
        pm = QPixmap(self.n_glyphs * (self.gw + 1), len(tiers) * (self.gh + 1))
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        for ti, col in enumerate(tiers):
            c = qc(col)
            for gi, bits in enumerate(self._glyphs):
                for r in range(self.gh):
                    for cc in range(self.gw):
                        if bits[r][cc]:
                            p.fillRect(gi * (self.gw + 1) + cc, ti * (self.gh + 1) + r, 1, 1, c)
        p.end()
        self._overlays[key] = pm
        self.n_tiers = len(tiers)
        return pm

    def _init_matrix(self):
        rng = random.Random(7)
        self.cell_w, self.cell_h = 6, 7
        self.cols = []
        for x in range(self.x0 + 2, self.x1 - 3, self.cell_w):
            self.cols.append({
                "x": x, "y": rng.uniform(-40, self.h), "v": rng.uniform(18, 46),
                "len": rng.randint(3, 8),
                "g": [rng.randrange(48) for _ in range(16)],
            })
        self._mrng = rng

    def _step_matrix(self, dt, an):
        rng = self._mrng
        boost = 0.35 + 1.9 * an.level + 2.5 * an.bass
        for c in self.cols:
            c["y"] += c["v"] * boost * dt
            if c["y"] - c["len"] * self.cell_h > self.h:
                c["y"] = rng.uniform(-20, 0)
                c["v"] = rng.uniform(18, 46)
                c["len"] = rng.randint(3, 8)
            if rng.random() < 0.08 + 0.3 * an.level:
                c["g"][rng.randrange(16)] = rng.randrange(48)

    def _draw_matrix(self, p, an, m):
        atlas = self._glyph_atlas(m)
        gw, gh = self.gw + 1, self.gh + 1
        for c in self.cols:
            head = int(c["y"] // self.cell_h)
            for k in range(c["len"]):
                row = head - k
                y = row * self.cell_h
                if y < -self.cell_h or y > self.h:
                    continue
                tier = 0 if k == 0 else min(self.n_tiers - 1, 1 + (k * (self.n_tiers - 1)) // max(1, c["len"]))
                gi = c["g"][row % 16]
                p.drawPixmap(c["x"], y, atlas, gi * gw, tier * gh, self.gw, self.gh)
        if an.bass > 0.05:
            p.fillRect(0, 0, self.w, self.h, qc(m.primary, int(36 * an.bass)))

    # -- 3: scanning eye (see-through visor, alluring eyes - see eyes.py)
    def _ensure_eyes(self):
        if self.eye_artist is None:
            ex, ey, ew, eh = self.eyes[0]
            img = self.under.toImage()
            r = g = b = n = 0
            for (x, y, _, _) in self.eyes:
                for dx in range(-3, 4):
                    for dy in (-int(eh * 0.9), -int(eh * 0.8)):
                        c = img.pixelColor(int(x + dx), int(y + dy))
                        if c.alpha() > 200:
                            r += c.red(); g += c.green(); b += c.blue(); n += 1
            skin = QColor(r // n, g // n, b // n) if n else QColor(196, 150, 140)
            self.eye_artist = EyeArtist(ew * 1.32, eh * 1.40, skin)     # larger, alien almonds

    def _draw_scan(self, p, an, m):
        self._ensure_eyes()
        p.drawPixmap(0, 0, self.under)
        bass = an.bass
        gz = self.eye_life.gaze
        for i, (ex, ey, ew, eh) in enumerate(self.eyes):
            o = self.eye_life.open_for(i)
            # faint ultraviolet glow on the skin around each eye (dims as it closes)
            og = QRadialGradient(QPointF(ex, ey), ew * 1.0)
            og.setColorAt(0.0, qc(m.primary, int((22 + 70 * bass) * (0.25 + 0.75 * o))))
            og.setColorAt(1.0, qc(m.primary, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(og)
            p.drawEllipse(QPointF(ex, ey), ew * 1.0, ew * 0.62)
        for i, (ex, ey, ew, eh) in enumerate(self.eyes):
            side = -1 if i == 0 else 1          # viewer's left eye: outer corner left
            o = self.eye_life.open_for(i)
            self.eye_artist.draw(p, ex, ey, side, o, (gz[0], gz[1]), bass, m.primary, self.t,
                                 heart=self.heart_m, inv=self.eye_life.inverted)
        # clear glass, soft lavender tint (multiply keeps it see-through)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Multiply)
        p.fillRect(0, 0, self.w, self.h, QColor(214, 196, 238))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        gl = m.glass
        g = QLinearGradient(0, 0, 0, self.h)
        g.setColorAt(0.0, qc((gl[0] + 60, gl[1] + 26, gl[2] + 24), 30))
        g.setColorAt(0.55, qc(gl, 40))
        g.setColorAt(1.0, qc(gl, 90))
        p.fillRect(0, 0, self.w, self.h, g)
        for y in range(0, self.h, 2):
            p.fillRect(0, y, self.w, 1, QColor(0, 0, 0, 12))
        # thin, slow, soft scan line
        x = self.scan_x
        trail = 14
        sg = QLinearGradient(x - trail * self.scan_dir, 0, x, 0)
        sg.setColorAt(0.0, qc(m.primary, 0))
        sg.setColorAt(1.0, qc(m.primary, 34))
        p.fillRect(QRectF(min(x, x - trail * self.scan_dir), 0, trail, self.h), sg)
        p.fillRect(QRectF(x - 0.5, 0, 1.0, self.h), qc(mix(m.primary, (255, 255, 255), 0.5), 95))
        # glass reflections
        gg = QLinearGradient(0, 0, 0, self.h)
        gg.setColorAt(0.0, QColor(240, 236, 255, 30))
        gg.setColorAt(0.26, QColor(255, 255, 255, 4))
        gg.setColorAt(0.3, QColor(255, 255, 255, 0))
        p.fillRect(0, 0, self.w, self.h, gg)
        path = QPainterPath()
        path.moveTo(self.w * 0.62, 0)
        path.lineTo(self.w * 0.70, 0)
        path.lineTo(self.w * 0.58, self.h)
        path.lineTo(self.w * 0.52, self.h)
        path.closeSubpath()
        p.fillPath(path, QColor(255, 255, 255, 12))
        # the hearts' own light shines through the tinted glass
        for i in range(len(self.eyes)):
            self.eye_artist.draw_light(p, -1 if i == 0 else 1)

    # -- sleep dimming / boot CRT warm-up
    def _mid_y(self):
        return float(np.median((self.top + self.bot) / 2))

    def _apply_fx(self, p, fx, m, mi):
        w, h = self.w, self.h
        k = fx.get("sleep", 0.0)
        if k > 0.001:
            # dim to a faint standby: a breathing line + a standby LED
            p.fillRect(0, 0, w, h, QColor(2, 1, 3, int((150 if mi == 3 else 205) * k)))
            br = 0.5 + 0.5 * math.sin(self.t * 2 * math.pi / 7.0)
            mid = self._mid_y()
            g = QLinearGradient(self.x0 + 8, 0, self.x1 - 8, 0)
            g.setColorAt(0.0, qc(m.primary, 0))
            g.setColorAt(0.5, qc(m.primary, int(k * (48 + 52 * br))))
            g.setColorAt(1.0, qc(m.primary, 0))
            p.fillRect(QRectF(self.x0 + 8, mid - 0.5, self.x1 - self.x0 - 16, 1.0), g)
            led = QRadialGradient(QPointF(self.x1 - 12, mid + 8), 3.2)
            led.setColorAt(0.0, qc(mix(m.primary, (255, 255, 255), 0.4), int(k * (70 + 110 * br))))
            led.setColorAt(1.0, qc(m.primary, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(led)
            p.drawEllipse(QPointF(self.x1 - 12, mid + 8), 3.2, 3.2)
        boot = fx.get("boot")
        if boot is not None:
            line, opened, flash = boot
            mid = self._mid_y()
            if opened < 1.0:
                half = max(0.0, opened) * (h / 2 + 2)
                top, bot = mid - half, mid + half
                dark = QColor(5, 5, 7)
                p.fillRect(QRectF(0, 0, w, max(0.0, top)), dark)
                p.fillRect(QRectF(0, bot, w, max(0.0, h - bot)), dark)
                if line > 0:
                    cx = (self.x0 + self.x1) / 2
                    hw = line * (self.x1 - self.x0) / 2
                    edge = [top, bot] if opened > 0.02 else [mid]
                    for yy in edge:
                        gl = QLinearGradient(cx - hw, 0, cx + hw, 0)
                        hot = mix(m.primary, (255, 255, 255), 0.75)
                        gl.setColorAt(0.0, qc(hot, 0))
                        gl.setColorAt(0.15, qc(hot, 200))
                        gl.setColorAt(0.5, QColor(255, 255, 255, 255))
                        gl.setColorAt(0.85, qc(hot, 200))
                        gl.setColorAt(1.0, qc(hot, 0))
                        p.fillRect(QRectF(cx - hw, yy - 0.7, 2 * hw, 1.4), gl)
                        p.fillRect(QRectF(cx - hw, yy - 2.5, 2 * hw, 5.0), qc(m.primary, 40))
            if flash > 0.001:
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
                p.fillRect(0, 0, w, h, qc(mix(m.primary, (255, 255, 255), 0.6), int(120 * flash)))
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    # -- shutter blades
    def _draw_shutter(self, p, c, m):
        half = self.h / 2 + 2
        cov = c * half
        jitter = (random.random() - 0.5) * 1.2 if c < 0.98 else 0.0
        for side in (0, 1):
            if side == 0:
                r = QRectF(0, -half + cov + jitter, self.w, half)
                edge_y = r.bottom()
            else:
                r = QRectF(0, self.h - cov + jitter, self.w, half)
                edge_y = r.top()
            g = QLinearGradient(0, r.top(), 0, r.bottom())
            g.setColorAt(0.0, QColor(58, 61, 70))
            g.setColorAt(0.5, QColor(34, 36, 42))
            g.setColorAt(1.0, QColor(22, 23, 27))
            p.fillRect(r, g)
            # slats
            p.setPen(QPen(QColor(10, 10, 12, 220), 1))
            yy = r.top() + 4
            while yy < r.bottom() - 1:
                p.drawLine(QPointF(0, yy), QPointF(self.w, yy))
                p.setPen(QPen(QColor(110, 114, 124, 90), 1))
                p.drawLine(QPointF(0, yy + 1), QPointF(self.w, yy + 1))
                p.setPen(QPen(QColor(10, 10, 12, 220), 1))
                yy += 6
            # chrome leading edge + warning LED strip
            p.setPen(QPen(QColor(200, 204, 214, 230), 1.2))
            p.drawLine(QPointF(0, edge_y), QPointF(self.w, edge_y))
            p.setPen(QPen(qc(m.primary, 200), 1.0))
            off = -2 if side == 0 else 2
            p.drawLine(QPointF(self.w * 0.2, edge_y + off), QPointF(self.w * 0.8, edge_y + off))
