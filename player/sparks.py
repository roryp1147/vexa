"""
Rare little electrical sparks at the severed wire tips (layout.json: wire_tips).

A burst = a brief glow at the tip + a handful of short-lived particles
(white-hot, cooling to amber, falling with gravity) + 1-2 re-flashes, and a
faint crackle. Nothing is animated between bursts; while one is alive the
window only repaints the bursts' bounding box.
"""
from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient

from .qtutil import mix, qc

# glowing frayed copper ends of the three thick cables (measured on base.png);
# they look "live", so they arc more often than the thin wire tips
CABLE_ENDS = [(182, 733), (378, 685), (113, 706)]
BASE_INTERVAL = (28.0, 70.0)     # seconds between bursts (awake, quiet)
SLEEP_INTERVAL = (90.0, 200.0)


class _Burst:
    __slots__ = ("x", "y", "t", "parts", "flashes", "life")

    def __init__(self, x, y, rng, strength):
        self.x, self.y = x, y
        self.t = 0.0
        self.parts = []
        n = rng.randint(4, 7) + int(3 * strength)
        for _ in range(n):
            ang = rng.uniform(-math.pi * 0.95, -math.pi * 0.05) if rng.random() < 0.7 else \
                rng.uniform(0.1, math.pi - 0.1)
            sp = rng.uniform(60, 150) * (0.8 + 0.5 * strength)
            life = rng.uniform(0.12, 0.42)
            # [x, y, vx, vy, age, life, delay]
            self.parts.append([x, y, math.cos(ang) * sp, math.sin(ang) * sp - 20,
                               0.0, life, rng.choice((0.0, 0.0, 0.0, rng.uniform(0.05, 0.18)))])
        self.flashes = [0.0] + sorted(rng.uniform(0.06, 0.22) for _ in range(rng.randint(0, 2)))
        self.life = max(p[5] + p[6] for p in self.parts) + 0.05


class Sparks:
    def __init__(self, tips, seed=None):
        self.tips = [tuple(c) for c in CABLE_ENDS] + [tuple(t) for t in tips]
        self.weights = [3.0] * len(CABLE_ENDS) + [1.0] * len(tips)
        self.rng = random.Random(seed)
        self.bursts: list[_Burst] = []
        self.next_t = None
        self._prev = QRect()
        self._bass_hot = 0.0

    @property
    def active(self) -> bool:
        return bool(self.bursts)

    def schedule(self, now, sleeping=False):
        lo, hi = SLEEP_INTERVAL if sleeping else BASE_INTERVAL
        self.next_t = now + self.rng.uniform(lo, hi)

    def maybe_burst(self, now, dt, bass, audio, sleeping=False) -> bool:
        """Returns True when a new burst starts (caller plays the crackle)."""
        if self.next_t is None:
            self.schedule(now, sleeping)
        # heavy bass makes the loose wires a little more likely to arc
        hot = audio and bass > 0.85
        if hot and self.rng.random() < dt * 0.02:
            return self.burst(now, strength=1.0)
        if now >= self.next_t:
            return self.burst(now, strength=0.5 if audio else 0.2)
        return False

    def burst(self, now, strength=0.5, tip=None) -> bool:
        x, y = self.rng.choices(self.tips, self.weights)[0] if tip is None else self.tips[tip]
        self.bursts.append(_Burst(x + self.rng.uniform(-1, 1), y - 1, self.rng, strength))
        self.schedule(now)
        return True

    def step(self, dt) -> bool:
        if not self.bursts:
            return False
        g = 300.0
        for b in self.bursts:
            b.t += dt
            for p in b.parts:
                if b.t < p[6]:
                    continue
                p[4] += dt
                p[3] += g * dt
                p[2] *= (1 - 1.5 * dt)
                p[0] += p[2] * dt
                p[1] += p[3] * dt
        self.bursts = [b for b in self.bursts if b.t < b.life]
        return True

    def dirty_rect(self) -> QRect:
        r = QRect()
        for b in self.bursts:
            r = r.united(QRect(int(b.x) - 16, int(b.y) - 16, 32, 32))
            for p in b.parts:
                r = r.united(QRect(int(p[0]) - 5, int(p[1]) - 5, 10, 10))
                r = r.united(QRect(int(p[0] - p[2] * 0.04) - 4, int(p[1] - p[3] * 0.04) - 4, 8, 8))
        u = r.united(self._prev) if not self._prev.isNull() else r
        self._prev = r
        return u

    def paint(self, p: QPainter, mode):
        if not self.bursts:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        halo_c = mix(mode.primary, (255, 255, 255), 0.45)
        for b in self.bursts:
            # tip glow: bright on each (re)flash, decaying fast
            fl = max((math.exp(-(b.t - f) / 0.05) if b.t >= f else 0.0) for f in b.flashes)
            if fl > 0.02:
                r = 6 + 12 * fl
                g = QRadialGradient(QPointF(b.x, b.y), r)
                g.setColorAt(0.0, QColor(255, 250, 235, int(230 * fl)))
                g.setColorAt(0.25, qc(halo_c, int(150 * fl)))
                g.setColorAt(1.0, qc(halo_c, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(g)
                p.drawEllipse(QPointF(b.x, b.y), r, r)
            for x, y, vx, vy, age, life, delay in b.parts:
                if b.t < delay or age >= life:
                    continue
                k = age / life
                heat = 1 - k
                col = mix((255, 160, 60), (255, 250, 230), heat ** 1.5)
                a = int(255 * min(1.0, heat * 1.6))
                tail = 0.03 + 0.025 * heat
                p.setPen(QPen(qc(col, a), 1.3 if heat > 0.5 else 0.9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(x - vx * tail, y - vy * tail), QPointF(x, y))
                if heat > 0.6:
                    p.setPen(QPen(QColor(255, 255, 255, int(200 * heat)), 0.6))
                    p.drawPoint(QPointF(x, y))
        p.restore()

    def clear(self):
        self.bursts = []
