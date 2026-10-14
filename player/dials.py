"""Shoulder dials: volume (viewer's left) and balance (viewer's right).

Drag up/right to turn clockwise (wheel works too). Values snap to detents
and every detent crossing plays a ratchet tick. A knurled chrome knob pixmap
is cached per mode and only rotated at draw time."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (QColor, QConicalGradient, QPainter, QPen, QPixmap,
                           QRadialGradient)

from . import config
from .draw import pixel_text, pixel_text_width
from .qtutil import mix, qc

SWEEP = 270.0


class Dial:
    def __init__(self, name, cx, cy, r, lo, hi, value, steps, on_change, sounds, clock=time.monotonic):
        self.name = name
        self.c = QPointF(cx, cy)
        self.r = r + 1.5            # slightly larger than the baked disc
        self.lo, self.hi = lo, hi
        self.steps = steps
        self.value = value
        self.on_change = on_change
        self.sounds = sounds
        self.clock = clock
        self.dragging = False
        self.hover = False
        self._drag_start = None
        self._acc = 0.0
        self.last_touch = -1e9
        self.tag = "VOL"             # readout prefix: VOL (Vexa) / BRAVE / SPOTIF / SYS
        self.default = value         # double-click springs back here
        self._spring = None          # (t0, from_value) while springing back
        self._pm = {}

    # ----------------------------------------------------------------- model
    @property
    def norm(self) -> float:
        return (self.value - self.lo) / (self.hi - self.lo)

    SPRING_S = 0.45

    @property
    def shown_norm(self) -> float:
        """Norm used for drawing: during a spring-back the knob swings from
        where it was to the default with a small damped overshoot."""
        n = self.norm
        if self._spring is None:
            return n
        t0, v0 = self._spring
        e = self.clock() - t0
        if e >= self.SPRING_S or e < 0:
            self._spring = None
            return n
        u = e / self.SPRING_S
        k = math.exp(-5.5 * u) * math.cos(u * math.pi * 2.6)
        n0 = (v0 - self.lo) / (self.hi - self.lo)
        return n + (n0 - n) * k

    @property
    def springing(self) -> bool:
        return self._spring is not None and self.clock() - self._spring[0] < self.SPRING_S

    @property
    def angle(self) -> float:
        return -SWEEP / 2 + SWEEP * self.shown_norm

    def reset(self) -> bool:
        """Spring back to the default position (double-click). The target
        gets the default once, right away; the knob animates there."""
        self.dragging = False
        old = self.value
        self.value = self.default
        self.last_touch = self.clock()
        if abs(old - self.default) < 1e-9:
            self.sounds.play("detent")
            return False
        self._spring = (self.clock(), old)
        self.sounds.play("eject_spring")
        self.sounds.play("detent")
        self.on_change(self.value)
        return True

    def step_size(self):
        return (self.hi - self.lo) / self.steps

    def set_steps(self, k: int):
        """Move k detents; plays a tick per detent."""
        if k == 0:
            return
        old = self.value
        nv = round((self.value - self.lo) / self.step_size()) + k
        nv = max(0, min(self.steps, nv))
        self.value = self.lo + nv * self.step_size()
        if abs(self.value - old) > 1e-9:
            self.sounds.play("detent")
            self.on_change(self.value)
        self.last_touch = self.clock()

    def hit(self, x, y) -> bool:
        return math.hypot(x - self.c.x(), y - self.c.y()) <= self.r + 5

    def rect(self) -> QRect:
        R = int(self.r + 10)
        return QRect(int(self.c.x()) - R, int(self.c.y()) - R - 22, 2 * R, 2 * R + 24)

    def begin_drag(self, x, y):
        self.dragging = True
        self._drag_start = (x, y)
        self._acc = 0.0
        self.last_touch = self.clock()

    def drag_to(self, x, y):
        if not self.dragging:
            return
        sx, sy = self._drag_start
        d = (x - sx) - (y - sy)            # right or up = clockwise
        self._drag_start = (x, y)
        self._acc += d
        px_per_step = 7.0
        k = int(self._acc / px_per_step)
        if k:
            self._acc -= k * px_per_step
            # one tick per detent so fast drags still ratchet audibly
            for _ in range(abs(k)):
                self.set_steps(1 if k > 0 else -1)

    def end_drag(self):
        self.dragging = False
        self.last_touch = self.clock()

    def readout_visible(self) -> bool:
        return self.dragging or self.clock() - self.last_touch < 1.2

    def label(self) -> str:
        if self.name == "volume":
            return f"{self.tag} {int(round(self.value * 100))}"
        v = int(round(self.value * 100))
        return "BAL C" if v == 0 else (f"BAL L{-v}" if v < 0 else f"BAL R{v}")

    # ----------------------------------------------------------------- paint
    def _knob(self, mi) -> QPixmap:
        pm = self._pm.get(mi)
        if pm is not None:
            return pm
        m = config.MODES[mi]
        ss = 3
        R = self.r
        size = int(math.ceil(2 * R + 4))
        pm = QPixmap(size * ss, size * ss)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(ss, ss)
        c = QPointF(size / 2, size / 2)
        # knurled chrome skirt
        cg = QConicalGradient(c, 45)
        for i, v in enumerate([210, 90, 170, 60, 200, 80, 150, 70, 210]):
            cg.setColorAt(i / 8, QColor(v, v + 2, v + 8))
        p.setPen(QPen(QColor(5, 5, 7), 1.0))
        p.setBrush(cg)
        p.drawEllipse(c, R, R)
        p.setPen(QPen(QColor(20, 20, 24, 200), 0.7))
        for k in range(36):
            a = k * math.tau / 36
            p.drawLine(QPointF(c.x() + math.cos(a) * (R - 2.2), c.y() + math.sin(a) * (R - 2.2)),
                       QPointF(c.x() + math.cos(a) * (R - 0.5), c.y() + math.sin(a) * (R - 0.5)))
        # dark cap
        rg = QRadialGradient(QPointF(c.x() - R * 0.3, c.y() - R * 0.35), R)
        rg.setColorAt(0, QColor(78, 80, 90))
        rg.setColorAt(0.6, QColor(30, 31, 36))
        rg.setColorAt(1, QColor(12, 12, 14))
        p.setPen(QPen(QColor(150, 154, 164, 120), 0.6))
        p.setBrush(rg)
        p.drawEllipse(c, R - 3, R - 3)
        # pointer (mode neon)
        p.setPen(QPen(qc(m.primary, 90), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(c.x(), c.y() - R + 3.5), QPointF(c.x(), c.y() - R * 0.25))
        p.setPen(QPen(qc(mix(m.primary, (255, 255, 255), 0.4)), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(c.x(), c.y() - R + 3.5), QPointF(c.x(), c.y() - R * 0.25))
        p.end()
        pm.setDevicePixelRatio(ss)
        self._pm[mi] = pm
        return pm

    def paint(self, p: QPainter, mi: int):
        """Blit the cached composite; re-render only when the state changes."""
        key = (mi, round(self.shown_norm, 4), self.value, self.hover or self.dragging)
        if getattr(self, "_ckey", None) != key:
            R = int(self.r + 9)
            pm = QPixmap(2 * R, 2 * R)
            pm.fill(Qt.GlobalColor.transparent)
            q = QPainter(pm)
            q.translate(R - self.c.x(), R - self.c.y())
            self._paint(q, mi)
            q.end()
            self._cpm, self._ckey, self._cR = pm, key, R
        p.drawPixmap(int(self.c.x()) - self._cR, int(self.c.y()) - self._cR, self._cpm)

    def _paint(self, p: QPainter, mi: int):
        m = config.MODES[mi]
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        c = self.c
        # LED ring
        n = 11
        rr = self.r + 4.2
        for i in range(n):
            a = math.radians(-SWEEP / 2 + SWEEP * i / (n - 1) - 90)
            pos = QPointF(c.x() + math.cos(a) * rr, c.y() + math.sin(a) * rr)
            t = i / (n - 1)
            sn = max(0.0, min(1.0, self.shown_norm))
            if self.name == "balance":
                mid = 0.5
                on = (min(mid, sn) - 0.02 <= t <= max(mid, sn) + 0.02)
            else:
                on = t <= sn + 0.02 and sn > 0.001
            p.setPen(Qt.PenStyle.NoPen)
            if on:
                p.setBrush(qc(m.primary, 70))
                p.drawEllipse(pos, 2.4, 2.4)
                p.setBrush(qc(mix(m.primary, (255, 255, 255), 0.35)))
            else:
                p.setBrush(QColor(30, 14, 20, 220))
            p.drawEllipse(pos, 1.1, 1.1)
        if self.hover or self.dragging:
            p.setPen(QPen(qc(m.primary, 70), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, self.r + 0.8, self.r + 0.8)
        # shadow + knob
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 110))
        p.drawEllipse(QPointF(c.x() + 1, c.y() + 1.6), self.r, self.r)
        pm = self._knob(mi)
        s = pm.width() / pm.devicePixelRatio()
        p.translate(c)
        p.rotate(self.angle)
        p.drawPixmap(QPointF(-s / 2, -s / 2), pm)
        p.restore()

    def paint_readout(self, p: QPainter, mi: int):
        if not self.readout_visible():
            return
        m = config.MODES[mi]
        txt = self.label()
        w = pixel_text_width(txt) + 8
        x = self.c.x() - w / 2
        y = self.c.y() - self.r - 20
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(120, 124, 134), 0.8))
        p.setBrush(QColor(8, 8, 10, 235))
        p.drawRoundedRect(QRectF(x, y, w, 11), 2, 2)
        pixel_text(p, x + 4, y + 3, txt, qc(mix(m.primary, (255, 255, 255), 0.2)))
        p.restore()
