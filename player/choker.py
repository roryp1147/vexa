"""Choker O-ring pendulum: spring-damped swing driven by window acceleration.

theta'' = -w0^2 sin(theta) - k * a_x * cos(theta) - c * theta'
(a_x = horizontal window acceleration in px/s^2; vertical accel modulates g)
Integrated with semi-implicit Euler in fixed sub-steps for stability."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QConicalGradient, QPainter, QPen, QPixmap

from . import config
from .qtutil import mix, qc

W0 = 2 * math.pi * 1.45     # natural frequency (rad/s) -> ~1.45 Hz swing
DAMP = 1.1                  # 1/s
K_ACC = 0.0080              # rad/s^2 per px/s^2
MAX_THETA = math.radians(80)
FRICTION = 0.35             # rad/s^2 Coulomb friction


class Choker:
    def __init__(self, layout):
        R = layout["ring"]
        self.pivot = QPointF(*R["pivot"])
        self.radius = R["radius"] * 1.02
        self.offset = R["offset"]
        self.theta = 0.0
        self.omega = 0.0
        self._pm = {}
        self._last_pos = None
        self._last_vel = (0.0, 0.0)
        self.acc = (0.0, 0.0)

    @property
    def energy(self) -> float:
        return 0.5 * self.omega ** 2 + W0 * W0 * (1 - math.cos(self.theta))

    @property
    def moving(self) -> bool:
        return self.theta != 0.0 or self.omega != 0.0

    def feed_window_pos(self, x: float, y: float, dt: float):
        """Called every tick with the window position -> acceleration."""
        if self._last_pos is None or dt <= 0:
            self._last_pos = (x, y)
            return
        vx = (x - self._last_pos[0]) / dt
        vy = (y - self._last_pos[1]) / dt
        ax = (vx - self._last_vel[0]) / dt
        ay = (vy - self._last_vel[1]) / dt
        # clamp spikes (first frame of a drag, OS hiccups)
        lim = 60000.0
        self.acc = (max(-lim, min(lim, ax)), max(-lim, min(lim, ay)))
        self._last_pos = (x, y)
        self._last_vel = (vx, vy)

    def kick(self, omega: float):
        self.omega += omega

    def step(self, dt: float):
        ax, ay = self.acc
        n = max(1, int(dt / 0.004))
        h = dt / n
        g = W0 * W0 * max(0.2, 1.0 + ay * 0.00025)
        for _ in range(n):
            a = -g * math.sin(self.theta) - K_ACC * ax * math.cos(self.theta) - DAMP * self.omega
            a -= math.copysign(FRICTION, self.omega) if abs(self.omega) > 1e-3 else 0.0
            self.omega += a * h
            self.theta += self.omega * h
        if abs(self.theta) > MAX_THETA:
            self.theta = math.copysign(MAX_THETA, self.theta)
            self.omega *= -0.3
        # metal-on-leather friction: small swings just stop (and idle the timer)
        if abs(ax) < 1 and abs(ay) < 1 and self.energy < 0.004:
            self.theta = self.omega = 0.0
        self.acc = (0.0, 0.0)

    def rect(self) -> QRect:
        L = self.offset + self.radius + 6
        return QRect(int(self.pivot.x() - L), int(self.pivot.y() - 6), int(2 * L), int(L + 10))

    def _ring(self, mi) -> QPixmap:
        pm = self._pm.get(mi)
        if pm is not None:
            return pm
        m = config.MODES[mi]
        ss = 4
        R = self.radius
        size = int(math.ceil(2 * R + 6))
        pm = QPixmap(size * ss, size * ss)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(ss, ss)
        c = QPointF(size / 2, size / 2)
        rm = R - 1.4                         # centre line of the torus
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(6, 6, 8, 230), 3.6))
        p.drawEllipse(c, rm, rm)
        cg = QConicalGradient(c, 120)
        stops = [(0.0, (235, 238, 245)), (0.12, (120, 124, 134)), (0.3, (60, 62, 70)),
                 (0.45, mix((200, 204, 214), m.primary, 0.45)), (0.55, (90, 94, 104)),
                 (0.75, (210, 214, 222)), (0.88, (70, 72, 80)), (1.0, (235, 238, 245))]
        for t, col in stops:
            cg.setColorAt(t, qc(col))
        p.setPen(QPen(cg, 2.4))
        p.drawEllipse(c, rm, rm)
        p.setPen(QPen(QColor(255, 255, 255, 170), 0.6))
        p.drawArc(QRectF(c.x() - rm + 0.5, c.y() - rm + 0.5, 2 * rm - 1, 2 * rm - 1), 100 * 16, 50 * 16)
        p.end()
        pm.setDevicePixelRatio(ss)
        self._pm[mi] = pm
        return pm

    def paint(self, p: QPainter, mi: int):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pm = self._ring(mi)
        s = pm.width() / pm.devicePixelRatio()
        deg = -math.degrees(self.theta)
        # soft contact shadow on the neck (offset down-right)
        p.save()
        p.translate(self.pivot + QPointF(1.0, 2.2))
        p.rotate(deg)
        p.setPen(QPen(QColor(40, 10, 20, 70), 3.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(0, self.offset), self.radius - 1.4, self.radius - 1.4)
        p.restore()
        p.translate(self.pivot)
        p.rotate(deg)
        p.drawPixmap(QPointF(-s / 2, self.offset - s / 2), pm)
        p.restore()
        # bail drawn over the ring top so it reads as threaded through it
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bx, by = self.pivot.x(), self.pivot.y()
        p.setPen(QPen(QColor(8, 8, 10, 220), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(QRectF(bx - 3.2, by - 6.5, 6.4, 7.5), 200 * 16, 140 * 16)
        p.setPen(QPen(QColor(205, 208, 216), 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(QRectF(bx - 3.2, by - 6.5, 6.4, 7.5), 200 * 16, 140 * 16)
        p.restore()
