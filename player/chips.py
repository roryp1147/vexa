"""
Ear-slot chips (stage 2).

  * viewer's-left slot  -> PLAYLIST chip: album art, title/artist (marquee),
    source + time, and for local files a scrollable track list
  * viewer's-right slot -> MINI-PLAYER chip: art, title/artist, prev / play /
    next, a tiny bass LED bar, and a detach button that pops it off into its
    own floating window (MiniWindow) while Vexa hides

Each chip is a little cartridge: chrome/black housing with chamfered corners,
grip ridges and screws, a translucent glowing circuit card (procedural
traces, vias, SMD parts and an IC, tinted with the current mode) with a gold
contact tongue that plugs into the head, and a recessed glass display.

Motion is deliberately clunky (ChipMotion): button clack -> spring pop ->
pause -> servo slide with two gear-skip hitches -> latch overshoot. Back in:
servo slide with a hitch -> spring snap -> seat bounce. Sounds are scheduled
on the same timeline.

Efficiency: the housing+card art is rendered once per (kind, mode) at 3x and
scaled down; the static part of each display is cached by a content key;
only marquees / the LED bar / the list scroll are redrawn per frame, and only
while a chip is out (or moving).
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QFontMetrics, QImage,
                           QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient, QTransform)
from PySide6.QtWidgets import QMenu, QWidget

from . import config, tags
from .draw import icon_path, pixel_text, pixel_text_width, seven_seg, seven_seg_width
from .qtutil import mix, qc

SS = 3                      # supersampling for the static card art
SHADOW = 6                  # margin around the card for its drop shadow

# ------------------------------------------------------------------ fonts
_FONTS = {}


def font(px: float, bold=False) -> QFont:
    key = (px, bold)
    f = _FONTS.get(key)
    if f is None:
        fam = _FONTS.get("family")
        if fam is None:
            fam = "monospace"
            fdir = os.path.join(config.ASSETS, "fonts")
            for fn in ("IBMPlexMono-Medium.ttf", "IBMPlexMono-SemiBold.ttf"):
                fid = QFontDatabase.addApplicationFont(os.path.join(fdir, fn))
                fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                if fams:
                    fam = fams[0]
            _FONTS["family"] = fam
        f = QFont(fam)
        f.setPixelSize(int(round(px)))
        f.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Medium)
        f.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
        f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        _FONTS[key] = f
    return f


def fmt_time(ms: int) -> str:
    s = max(0, int(ms // 1000))
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


# ------------------------------------------------------------------ motion
def _hitch_table(hitches, n=256, depth=0.86, width=0.045, ramp=(0.08, 0.05)):
    """Normalised progress curve for a servo slide that stalls at `hitches`."""
    u = np.linspace(0, 1, n)
    s = np.ones(n)
    for h in hitches:
        s *= 1 - depth * np.exp(-((u - h) / width) ** 2)
    s *= np.clip(u / ramp[0], 0.15, 1) * np.clip((1 - u) / ramp[1], 0.25, 1)
    c = np.concatenate([[0.0], np.cumsum((s[1:] + s[:-1]) * 0.5)])
    return c / c[-1]


class ChipMotion:
    """Extension e in [0, 1] (1 = fully out) on a clunky timeline."""

    OUT_TABLE = _hitch_table((0.35, 0.70))
    IN_TABLE = _hitch_table((0.50,), ramp=(0.06, 0.03))
    POP = 0.14
    OUT_EVENTS = ((0.0, "eject_spring"), (0.26, "chip_slide_out"), (0.86, "chip_latch"))
    IN_EVENTS = ((0.0, "press"), (0.04, "chip_slide_in"), (0.565, "chip_seat"))
    OUT_DUR = 0.96
    IN_DUR = 0.67

    def __init__(self):
        self.state = "in"
        self.t0 = 0.0
        self.e = 0.0
        self._fired = 0

    @property
    def moving(self):
        return self.state in ("ejecting", "retracting")

    @property
    def visible(self):
        return self.state != "in"

    def eject(self, now):
        if self.state in ("in", "retracting"):
            self.state, self.t0, self._fired = "ejecting", now, 0
            return True
        return False

    def retract(self, now):
        if self.state in ("out", "ejecting"):
            self.state, self.t0, self._fired = "retracting", now, 0
            return True
        return False

    def toggle(self, now):
        return self.eject(now) if self.state in ("in", "retracting") else self.retract(now)

    def set_out(self):
        self.state, self.e = "out", 1.0

    def set_in(self):
        self.state, self.e = "in", 0.0

    @staticmethod
    def _lut(table, u):
        u = min(1.0, max(0.0, u)) * (len(table) - 1)
        i = int(u)
        if i >= len(table) - 1:
            return float(table[-1])
        f = u - i
        return float(table[i] * (1 - f) + table[i + 1] * f)

    def e_out(self, t):
        P = self.POP
        if t < 0.06:
            return 0.0
        if t < 0.16:                        # spring pop with a little overshoot
            u = (t - 0.06) / 0.10
            return P * (1 - (1 - u) ** 3) + 0.035 * math.sin(math.pi * u)
        if t < 0.26:                        # sits there a beat (servo engaging)
            return P + 0.004 * math.sin((t - 0.16) * 90) * (0.26 - t) / 0.1
        if t < 0.86:                        # servo slide with gear-skip hitches
            return P + (1 - P) * self._lut(self.OUT_TABLE, (t - 0.26) / 0.60)
        if t < 0.95:                        # latch overshoot
            u = (t - 0.86) / 0.09
            return 1.0 + 0.022 * math.sin(math.pi * u) * (1 - u * 0.3)
        return 1.0

    def e_in(self, t):
        if t < 0.04:
            return 1.0
        if t < 0.54:
            return 1.0 - 0.92 * self._lut(self.IN_TABLE, (t - 0.04) / 0.50)
        if t < 0.60:                        # spring pulls it home
            u = (t - 0.54) / 0.06
            return 0.08 * (1 - u * u)
        if t < 0.66:                        # seat bounce
            return 0.012 * math.sin(math.pi * (t - 0.60) / 0.06)
        return 0.0

    def update(self, now):
        """Advance; returns (changed, [sound names to play])."""
        sounds = []
        if not self.moving:
            return False, sounds
        t = now - self.t0
        events = self.OUT_EVENTS if self.state == "ejecting" else self.IN_EVENTS
        while self._fired < len(events) and t >= events[self._fired][0]:
            sounds.append(events[self._fired][1])
            self._fired += 1
        old = self.e
        if self.state == "ejecting":
            self.e = self.e_out(t)
            if t >= self.OUT_DUR:
                self.state, self.e = "out", 1.0
        else:
            self.e = self.e_in(t)
            if t >= self.IN_DUR:
                self.state, self.e = "in", 0.0
        return True if self.e != old or not self.moving else False, sounds


# ------------------------------------------------------------------ card art
def _housing_path(W, H, side_root_chamfer=10):
    """Canonical orientation: outer end at x=0, root (connector) at x=W."""
    c = 9.0          # outer chamfers
    r = 3.0
    p = QPainterPath()
    p.moveTo(c, 0)
    p.lineTo(W - side_root_chamfer, 0)
    p.lineTo(W, side_root_chamfer)
    p.lineTo(W, H - r)
    p.quadTo(W, H, W - r, H)
    p.lineTo(c, H)
    p.lineTo(0, H - c)
    p.lineTo(0, c)
    p.closeSubpath()
    return p


def _traces(p: QPainter, rect: QRectF, root_x: float, fingers, col: QColor, seed: int):
    """Procedural PCB traces: 45-degree routed lines from the gold fingers
    into the card, plus scattered short runs and vias."""
    rng = random.Random(seed)
    glow = QColor(col)
    glow.setAlpha(34)
    core = QColor(col)
    core.setAlpha(150)
    paths = []
    # bus from the contact fingers, fanning out across the card
    for i, fy in enumerate(fingers):
        x, y = root_x, fy
        path = QPainterPath(QPointF(x + 4, y))
        path.lineTo(x, y)
        steps = rng.randint(2, 4)
        for _ in range(steps):
            run = rng.uniform(8, 26)
            x -= run
            path.lineTo(x, y)
            dy = rng.choice((-1, 1)) * rng.uniform(4, 12)
            ny = min(rect.bottom() - 3, max(rect.top() + 3, y + dy))
            d = abs(ny - y)
            x -= d
            y = ny
            path.lineTo(x, y)
            if x < rect.left() + 10:
                break
        paths.append((path, QPointF(x, y)))
    # stray runs
    for _ in range(18):
        x = rng.uniform(rect.left() + 4, rect.right() - 6)
        y = rng.uniform(rect.top() + 3, rect.bottom() - 3)
        path = QPainterPath(QPointF(x, y))
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                x += rng.choice((-1, 1)) * rng.uniform(5, 16)
            else:
                d = rng.uniform(3, 9) * rng.choice((-1, 1))
                x += abs(d)
                y = min(rect.bottom() - 3, max(rect.top() + 3, y + d))
            x = min(rect.right() - 3, max(rect.left() + 3, x))
            path.lineTo(x, y)
        paths.append((path, QPointF(x, y)))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for path, _ in paths:
        p.setPen(QPen(glow, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path)
    for path, _ in paths:
        p.setPen(QPen(core, 0.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path)
    # vias
    for _, end in paths:
        p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 170), 0.5))
        p.setBrush(QColor(20, 14, 10, 220))
        p.drawEllipse(end, 1.25, 1.25)
    return rng


def render_card(kind: str, W: int, H: int, mode, flip: bool) -> tuple[QPixmap, dict]:
    """Housing + circuit card + gold tongue + display recess (no content).

    Returns (pixmap incl. SHADOW margin + tongue, geometry dict in pixmap
    coords). flip=True mirrors the hardware so the root faces left."""
    T = 22                       # tongue length (goes into the head)
    TH = int(H * 0.46)           # tongue height
    GRIP = 13
    PW, PH = W + T + 2 * SHADOW, H + 2 * SHADOW
    img = QImage(PW * SS, PH * SS, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(SS, SS)
    if flip:
        p.translate(PW, 0)
        p.scale(-1, 1)
    p.translate(SHADOW, SHADOW)
    prim, sec, gl = mode.primary, mode.secondary, mode.glass
    seed = 7 if kind == "playlist" else 13

    # drop shadow (soft, offset down/away from the head)
    for k in range(6, 0, -1):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 16))
        p.drawRoundedRect(QRectF(-k * 0.5 + 1.5, -k * 0.5 + 3.0, W + k, H + k), 6, 6)

    # --- tongue with gold fingers (drawn first: housing overlaps its root)
    ty = (H - TH) / 2
    tongue = QRectF(W - 2, ty, T + 2, TH)
    p.setPen(QPen(QColor(8, 8, 10), 0.8))
    tg = QLinearGradient(0, ty, 0, ty + TH)
    tg.setColorAt(0, qc(mix(gl, (40, 40, 40), 0.5)))
    tg.setColorAt(1, qc(mix(gl, (0, 0, 0), 0.4)))
    p.setBrush(tg)
    p.drawRoundedRect(tongue, 1.5, 1.5)
    n_f = 7
    fh = (TH - 6) / n_f
    fingers = []
    for i in range(n_f):
        fy = ty + 3 + i * fh
        r = QRectF(W + 1.5, fy + fh * 0.18, T - 3.0, fh * 0.64)
        gg = QLinearGradient(0, r.top(), 0, r.bottom())
        gg.setColorAt(0.0, QColor(255, 232, 150))
        gg.setColorAt(0.45, QColor(222, 170, 62))
        gg.setColorAt(1.0, QColor(140, 92, 26))
        p.setPen(QPen(QColor(90, 60, 14), 0.4))
        p.setBrush(gg)
        p.drawRoundedRect(r, 0.8, 0.8)
        p.setPen(QPen(QColor(255, 250, 220, 160), 0.4))
        p.drawLine(QPointF(r.left() + 1, r.top() + 0.4), QPointF(r.right() - 1, r.top() + 0.4))
        fingers.append(fy + fh * 0.5)

    # --- housing: dark brushed chrome
    hp = _housing_path(W, H)
    g = QLinearGradient(0, 0, 0, H)
    g.setColorAt(0.00, QColor(118, 122, 132))
    g.setColorAt(0.06, QColor(66, 69, 78))
    g.setColorAt(0.45, QColor(30, 31, 37))
    g.setColorAt(0.86, QColor(20, 21, 25))
    g.setColorAt(1.00, QColor(70, 72, 80))
    p.setPen(QPen(QColor(6, 6, 8), 1.1))
    p.setBrush(g)
    p.drawPath(hp)
    # brushed streaks
    rng = random.Random(seed + 1)
    p.save()
    p.setClipPath(hp)
    for _ in range(70):
        y = rng.uniform(0, H)
        a = rng.randint(6, 18)
        p.setPen(QPen(QColor(255, 255, 255, a) if rng.random() < 0.5 else QColor(0, 0, 0, a + 6), 0.4))
        x0 = rng.uniform(-10, W)
        p.drawLine(QPointF(x0, y), QPointF(x0 + rng.uniform(20, 90), y))
    # bevel highlight / shadow
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
    p.drawLine(QPointF(9, 1.0), QPointF(W - 10, 1.0))
    p.setPen(QPen(QColor(0, 0, 0, 120), 1.0))
    p.drawLine(QPointF(9, H - 1.0), QPointF(W - 3, H - 1.0))
    p.restore()

    # grip ridges at the outer end
    for i in range(5):
        x = 3.5 + i * 2.0
        p.setPen(QPen(QColor(0, 0, 0, 170), 0.8))
        p.drawLine(QPointF(x, H * 0.30), QPointF(x, H * 0.70))
        p.setPen(QPen(QColor(200, 204, 214, 70), 0.6))
        p.drawLine(QPointF(x + 0.8, H * 0.30), QPointF(x + 0.8, H * 0.70))

    # --- circuit card window
    card = QRectF(GRIP, 6, W - GRIP - 5, H - 12)
    cp = QPainterPath()
    cp.addRoundedRect(card, 3, 3)
    p.setPen(QPen(QColor(4, 4, 6), 1.0))
    cg = QLinearGradient(0, card.top(), 0, card.bottom())
    cg.setColorAt(0, qc(mix(gl, prim, 0.10), 245))
    cg.setColorAt(1, qc(mix(gl, (0, 0, 0), 0.3), 245))
    p.setBrush(cg)
    p.drawPath(cp)
    p.save()
    p.setClipPath(cp)
    # inner glow of the translucent card (light leaking through the resin)
    rg = QRadialGradient(QPointF(card.right() - 14, card.center().y()), card.width() * 0.7)
    rg.setColorAt(0, qc(prim, 70))
    rg.setColorAt(0.6, qc(sec, 24))
    rg.setColorAt(1, qc(sec, 0))
    p.fillRect(card, rg)
    rng = _traces(p, card, W - 4, fingers, qc(prim), seed)
    # SMD parts
    for _ in range(9):
        x = rng.uniform(card.left() + 3, card.right() - 10)
        y = rng.uniform(card.top() + 2, card.bottom() - 6)
        w, h = rng.choice(((4.5, 2.2), (2.2, 4.5), (3.4, 1.8)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(28, 22, 20))
        p.drawRect(QRectF(x, y, w, h))
        p.setBrush(QColor(196, 198, 204))
        if w > h:
            p.drawRect(QRectF(x, y, 0.9, h))
            p.drawRect(QRectF(x + w - 0.9, y, 0.9, h))
        else:
            p.drawRect(QRectF(x, y, w, 0.9))
            p.drawRect(QRectF(x, y + h - 0.9, w, 0.9))
    p.restore()

    # --- display recess (content area): leaves a PCB band at the root side
    band = 24 if W >= 150 else 20
    disp = QRectF(card.left() + 3, card.top() + 3, card.width() - band - 3, card.height() - 6)
    # the IC on the root band, legs and all
    ic = QRectF(disp.right() + 5, card.center().y() - 9, band - 9, 18)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(190, 192, 198))
    for k in range(5):
        yy = ic.top() + 2 + k * 3.4
        p.drawRect(QRectF(ic.left() - 1.6, yy, 1.8, 1.2))
        p.drawRect(QRectF(ic.right() - 0.2, yy, 1.8, 1.2))
    icg = QLinearGradient(0, ic.top(), 0, ic.bottom())
    icg.setColorAt(0, QColor(44, 44, 50))
    icg.setColorAt(1, QColor(14, 14, 17))
    p.setPen(QPen(QColor(0, 0, 0), 0.6))
    p.setBrush(icg)
    p.drawRect(ic)
    p.setBrush(QColor(70, 70, 78))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(ic.left() + 2.2, ic.top() + 2.2), 0.8, 0.8)
    # status LED above the IC
    lc = QPointF(ic.center().x(), card.top() + 5)
    lg = QRadialGradient(lc, 4.5)
    lg.setColorAt(0, qc(mix(prim, (255, 255, 255), 0.5), 230))
    lg.setColorAt(0.35, qc(prim, 160))
    lg.setColorAt(1, qc(prim, 0))
    p.setBrush(lg)
    p.drawEllipse(lc, 4.5, 4.5)
    # recess: dark smoked glass with a bevel
    p.setPen(QPen(QColor(0, 0, 0, 230), 1.2))
    dg = QLinearGradient(0, disp.top(), 0, disp.bottom())
    dg.setColorAt(0, qc(mix(gl, (0, 0, 0), 0.55), 250))
    dg.setColorAt(1, qc(mix(gl, (0, 0, 0), 0.75), 250))
    p.setBrush(dg)
    p.drawRoundedRect(disp, 2.5, 2.5)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 40), 0.7))
    p.drawLine(QPointF(disp.left() + 2, disp.bottom() + 0.8), QPointF(disp.right() - 2, disp.bottom() + 0.8))

    # screws in the housing corners
    for sx, sy in ((GRIP - 6.5, 3.6), (GRIP - 6.5, H - 3.6), (W - 8.0, H - 3.6)):
        sg = QRadialGradient(QPointF(sx - 0.5, sy - 0.5), 2.6)
        sg.setColorAt(0, QColor(214, 216, 224))
        sg.setColorAt(0.6, QColor(110, 112, 122))
        sg.setColorAt(1, QColor(36, 37, 42))
        p.setPen(QPen(QColor(0, 0, 0, 200), 0.5))
        p.setBrush(sg)
        p.drawEllipse(QPointF(sx, sy), 1.9, 1.9)
        p.setPen(QPen(QColor(20, 20, 24), 0.55))
        p.drawLine(QPointF(sx - 1.1, sy), QPointF(sx + 1.1, sy))
        p.drawLine(QPointF(sx, sy - 1.1), QPointF(sx, sy + 1.1))
    p.end()

    pm = QPixmap.fromImage(img.scaled(PW, PH, Qt.AspectRatioMode.IgnoreAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
    # geometry in pixmap coords (content is never mirrored)
    def fx(x):
        return PW - (SHADOW + x) if flip else SHADOW + x
    dx0, dx1 = fx(disp.left()), fx(disp.right())
    geo = {
        "size": (PW, PH),
        "display": QRect(int(round(min(dx0, dx1))), int(round(SHADOW + disp.top())),
                         int(round(disp.width())), int(round(disp.height()))),
        "card": QRect(int(round(min(fx(0), fx(W)))), SHADOW, W, H),
        "root_x": fx(W),            # where the card meets the head (pixmap x)
        # silkscreen strip on the PCB band below the IC (pixmap coords)
        "silk": QRectF(min(fx(ic.left()), fx(ic.right())), SHADOW + ic.bottom() + 4,
                       ic.width(), card.bottom() - ic.bottom() - 7),
    }
    return pm, geo


_PIX_CACHE: dict = {}


def ptext(p: QPainter, x, y, text: str, color: QColor) -> float:
    """pixel_text through a small pixmap cache (status/time strings repeat)."""
    key = ("p", text, color.rgba())
    pm = _PIX_CACHE.get(key)
    if pm is None:
        w = max(1, int(pixel_text_width(text)) + 1)
        img = QImage(w, 5, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        q = QPainter(img)
        pixel_text(q, 0, 0, text, color)
        q.end()
        pm = QPixmap.fromImage(img)
        _cache_put(key, pm)
    p.drawPixmap(QPointF(x, y), pm)
    return pm.width()


def seg_pixmap(text: str, h: int, on: QColor, off: QColor) -> QPixmap:
    key = ("s", text, h, on.rgba(), off.rgba())
    pm = _PIX_CACHE.get(key)
    if pm is None:
        w = int(seven_seg_width(h, text)) + 6
        img = QImage(w, h + 4, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        q = QPainter(img)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        seven_seg(q, 2, 2, h, text, on, off)
        q.end()
        pm = QPixmap.fromImage(img)
        _cache_put(key, pm)
    return pm


def scanlines(w: int, h: int) -> QPixmap:
    key = ("scan", w, h)
    pm = _PIX_CACHE.get(key)
    if pm is None:
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        q = QPainter(img)
        for yy in range(0, h, 2):
            q.fillRect(0, yy, w, 1, QColor(0, 0, 0, 22))
        g = QLinearGradient(0, 0, 0, h * 0.4)
        g.setColorAt(0, QColor(255, 255, 255, 22))
        g.setColorAt(1, QColor(255, 255, 255, 0))
        q.fillRect(0, 0, w, h, g)
        q.end()
        pm = QPixmap.fromImage(img)
        _PIX_CACHE[key] = pm
    return pm


def _cache_put(key, pm):
    _PIX_CACHE[key] = pm
    if len(_PIX_CACHE) > 160:
        for k in list(_PIX_CACHE)[:40]:
            if k[0] != "scan":
                _PIX_CACHE.pop(k, None)


def paint_silk(p: QPainter, r: QRectF, text: str, mode):
    """Board silkscreen: part number running up the PCB band."""
    w = pixel_text_width(text)
    if r.height() < w + 2:
        return
    p.save()
    p.translate(r.center().x() + 2.5, r.bottom() - (r.height() - w) / 2)
    p.rotate(-90)
    pixel_text(p, 0, 0, text, qc(mix(mode.primary, (255, 255, 255), 0.5), 150))
    p.restore()


# ------------------------------------------------------------------ text strips
class TextStrip:
    """Pre-rendered single-line text; draws static or as a marquee."""

    GAP = 28
    SPEED = 16.0        # px/s
    PAUSE = 1.8         # s at the start of each loop

    def __init__(self, text: str, f: QFont, color: QColor):
        self.text = text
        fm = QFontMetrics(f)
        self.w = max(1, fm.horizontalAdvance(text))
        self.h = fm.height()
        self.ascent = fm.ascent()
        img = QImage(self.w + 2, self.h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setFont(f)
        p.setPen(color)
        p.drawText(0, self.ascent, text)
        p.end()
        self.pm = QPixmap.fromImage(img)

    avail = 10 ** 6

    def overflows(self, avail: int | None = None) -> bool:
        return self.w > (self.avail if avail is None else avail)

    def offset(self, t: float, avail: int | None = None) -> int:
        avail = self.avail if avail is None else avail
        if self.w <= avail:
            return 0
        loop = (self.w + self.GAP) / self.SPEED + self.PAUSE
        u = t % loop
        return 0 if u < self.PAUSE else int((u - self.PAUSE) * self.SPEED)

    def draw(self, p: QPainter, x: int, y: int, avail: int, t: float):
        self.avail = avail
        if self.w <= avail:
            p.drawPixmap(x, y, self.pm)
            return
        off = self.offset(t, avail)
        p.save()
        p.setClipRect(QRect(x, y, avail, self.h))
        p.drawPixmap(x - off, y, self.pm)
        if off > self.w + self.GAP - avail:
            p.drawPixmap(x - off + self.w + self.GAP, y, self.pm)
        p.restore()


def _art_placeholder(w, h, mode, seed=3) -> QPixmap:
    """No album art: a little etched 'disc' in the mode colours."""
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(10, 8, 12))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QPointF(w / 2, h / 2)
    r = min(w, h) * 0.42
    g = QRadialGradient(c, r)
    g.setColorAt(0, qc(mode.glass))
    g.setColorAt(0.25, qc(mix(mode.secondary, (0, 0, 0), 0.6)))
    g.setColorAt(0.9, qc(mix(mode.primary, (0, 0, 0), 0.55)))
    g.setColorAt(1, qc(mode.primary, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    p.drawEllipse(c, r, r)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for k in range(3, 9):
        p.setPen(QPen(qc(mode.primary, 40), 0.5))
        p.drawEllipse(c, r * k / 9, r * k / 9)
    p.setPen(QPen(qc(mix(mode.primary, (255, 255, 255), 0.5), 200), 1.0))
    p.drawEllipse(c, r * 0.12, r * 0.12)
    p.end()
    return QPixmap.fromImage(img)


# ------------------------------------------------------------------ chips
class BaseChip:
    """Common slot/motion/geometry logic (body coordinates)."""

    kind = "base"
    W = 160
    H = 100
    CY = 132                    # vertical centre: where the side of her skull is straightest
    RAIL_H = 134                # slit length, same on both sides (fits the tallest chip)

    def __init__(self, side: str, slot_rect, sounds, axis_x: float, body_w: int, alpha=None):
        self.side = side                      # "L" (viewer's left) or "R"
        self.sounds = sounds
        self.motion = ChipMotion()
        sx, sy, sw, sh = slot_rect
        self.button = QRect(int(sx + 1), int(sy + 2), int(sw - 2), int(sh - 3))   # red eject button
        self.button_down = False
        self.button_hover = False
        self.y0 = int(round(self.CY - self.H / 2))
        # --- the SLIT: a straight vertical aperture in a slim rail on the side
        # of her head. The chip is only ever visible on the outside of the
        # slit (clip), so nothing shows inside/behind the head or poking out
        # of the far side at any frame of the motion.
        self.slit_top = int(round(self.CY - self.RAIL_H / 2))
        self.slit_bot = self.slit_top + self.RAIL_H
        rows = range(self.slit_top - 4, self.slit_bot + 5)
        ax = int(round(axis_x))
        edge = {}
        if alpha is not None:
            for y in rows:
                r = alpha[y, :ax] >= 128 if side == "L" else alpha[y, ax:] >= 128
                if r.any():
                    edge[y] = int(r.argmax()) if side == "L" else ax + len(r) - 1 - int(r[::-1].argmax())
        if not edge:                                # no art (tests): a plausible skull
            edge = {y: (139 if side == "L" else body_w - 139) for y in rows}
        for y in rows:                              # fill gaps from the nearest row
            if y not in edge:
                edge[y] = edge[min(edge, key=lambda k: abs(k - y))]
        self.edge = edge
        core = [edge[y] for y in range(self.slit_top, self.slit_bot + 1)]
        if side == "L":
            self.slit_x = min(core) - 1         # chip visible at body x <= slit_x
            self.root_out = self.slit_x - 3     # a few gold contacts show at the lip
            self.x_out = self.root_out - self.W
            self.x_in = self.slit_x + 1 + SHADOW    # whole pixmap inside the slit
        else:
            self.slit_x = max(core) + 1         # chip visible at body x >= slit_x
            self.root_out = self.slit_x + 3
            self.x_out = self.root_out
            self.x_in = self.slit_x - self.W - SHADOW
        self._rail = {}
        self._frames = {}
        self._static = None
        self._static_key = None
        self.hover_item = None
        self.pressed_item = None
        self.t_text = 0.0

    # -- geometry
    def frame(self, mi, mode):
        f = self._frames.get(mi)
        if f is None:
            f = render_card(self.kind, self.W, self.H, mode, flip=(self.side == "R"))
            self._frames[mi] = f
        return f

    def card_x(self) -> float:
        e = self.motion.e
        return self.x_in + (self.x_out - self.x_in) * e

    def pixmap_origin(self, geo) -> QPointF:
        """Top-left of the card pixmap in body coords."""
        x = self.card_x()
        cr = geo["card"]
        return QPointF(round(x) - cr.x(), self.y0 - cr.y())

    def extents_out(self, geo) -> tuple[int, int]:
        """(left, right) body-x extent of the pixmap when fully out (+overshoot)."""
        cr = geo["card"]
        PW = geo["size"][0]
        x = self.x_out + (self.x_out - self.x_in) * 0.03
        return int(math.floor(x - cr.x())), int(math.ceil(x - cr.x() + PW))

    def rect(self, geo) -> QRect:
        o = self.pixmap_origin(geo)
        PW, PH = geo["size"]
        return QRect(int(o.x()) - 1, int(o.y()) - 1, PW + 2, PH + 2)

    def display_rect(self, geo) -> QRect:
        o = self.pixmap_origin(geo)
        return geo["display"].translated(int(o.x()), int(o.y()))

    def card_rect(self, geo) -> QRect:
        o = self.pixmap_origin(geo)
        return geo["card"].translated(int(o.x()), int(o.y()))

    # -- input helpers
    def hit_button(self, x, y) -> bool:
        return self.button.adjusted(-2, -2, 2, 2).contains(x, y)

    def press_button(self, now) -> bool:
        if self.motion.toggle(now):
            return True
        return False

    # -- drawing of the red eject button state (on top of the body)
    def paint_button(self, p: QPainter):
        r = QRectF(self.button)
        if self.button_down:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 110))
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 2, 2)
            p.setPen(QPen(QColor(0, 0, 0, 160), 1.0))
            p.drawLine(QPointF(r.left() + 1, r.top() + 1), QPointF(r.right() - 1, r.top() + 1))
        elif self.button_hover or self.motion.visible:
            g = QRadialGradient(r.center(), r.height() * 0.8)
            a = 90 if self.button_hover else 50
            g.setColorAt(0, QColor(255, 90, 70, a))
            g.setColorAt(1, QColor(255, 40, 30, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(g)
            p.drawEllipse(r.center(), r.width() * 0.9, r.height() * 0.75)

    def update(self, now) -> bool:
        changed, snd = self.motion.update(now)
        for s in snd:
            self.sounds.play(s)
        return changed

    def emerged_rect(self, h=2000) -> QRect:
        """Body-coords region where chip pixels may appear (outside the slit)."""
        y0, y1 = self.slit_top - 1, self.slit_bot + 1      # only through the slit opening
        if self.side == "L":
            return QRect(-h, y0, self.slit_x + 1 + h, y1 - y0 + 1)
        return QRect(self.slit_x, y0, 2 * h, y1 - y0 + 1)

    def visible_x(self, x) -> bool:
        return x <= self.slit_x if self.side == "L" else x >= self.slit_x

    def paint(self, p: QPainter, mi: int, mode, np_, an, now):
        if not self.motion.visible:
            return
        pm, geo = self.frame(mi, mode)
        o = self.pixmap_origin(geo)
        p.save()
        p.setClipRect(self.emerged_rect(), Qt.ClipOperation.IntersectClip)
        p.drawPixmap(o, pm)
        d = geo["display"].translated(int(o.x()), int(o.y()))
        p.save()
        p.setClipRect(d, Qt.ClipOperation.IntersectClip)
        self.paint_display(p, d, mi, mode, np_, an, now)
        p.restore()
        paint_silk(p, geo["silk"].translated(o), self.LABEL, mode)
        p.restore()

    # -- the slit rail (always there; drawn after the chip, before the body)
    def rail_rect(self) -> QRect:
        ys = range(self.slit_top - 4, self.slit_bot + 5)
        if self.side == "L":
            x0, x1 = self.slit_x - 3, max(self.edge[y] for y in ys) + 6
        else:
            x0, x1 = min(self.edge[y] for y in ys) - 6, self.slit_x + 4
        return QRect(x0, self.slit_top - 4, x1 - x0, self.slit_bot - self.slit_top + 9)

    def paint_rail(self, p: QPainter, mi: int, mode):
        pm = self._rail.get(mi)
        if pm is None:
            pm = self._rail[mi] = self._render_rail(mode)
        p.drawPixmap(self.rail_rect().topLeft(), pm)

    def _render_rail(self, mode) -> QPixmap:
        """Slim gunmetal rail hugging the skull, straight outer face with the
        slit (dark groove + neon seam) the chip slides through."""
        rr = self.rail_rect()
        ss = 3
        img = QImage(rr.width() * ss, rr.height() * ss, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(ss, ss)
        p.translate(-rr.x(), -rr.y())
        sg = 1 if self.side == "L" else -1
        face = self.slit_x - 0.5 * sg          # outer face (the lip) of the rail
        top, bot = self.slit_top - 3.0, self.slit_bot + 3.0
        rad = 9.0

        def inner(y):                           # tucked under the skull outline
            e = self.edge[int(min(max(y, top), bot))]
            return e + 4.5 * sg
        path = QPainterPath(QPointF(face, top + rad))
        path.lineTo(face, bot - rad)
        path.cubicTo(face, bot - rad * 0.3, face + rad * 0.3 * sg, bot, face + rad * sg, bot)
        path.lineTo(inner(bot), bot)
        for y in range(int(bot) - 1, int(top), -1):
            path.lineTo(inner(y), y)
        path.lineTo(inner(top), top)
        path.lineTo(face + rad * sg, top)
        path.cubicTo(face + rad * 0.3 * sg, top, face, top + rad * 0.3, face, top + rad)
        path.closeSubpath()
        g = QLinearGradient(0, top, 0, bot)
        g.setColorAt(0.00, QColor(132, 136, 148))
        g.setColorAt(0.06, QColor(78, 81, 92))
        g.setColorAt(0.50, QColor(42, 44, 52))
        g.setColorAt(0.94, QColor(30, 31, 37))
        g.setColorAt(1.00, QColor(64, 66, 76))
        p.setPen(QPen(QColor(6, 6, 8, 230), 0.8))
        p.setBrush(g)
        p.drawPath(path)
        p.save()
        p.setClipPath(path)
        # rounded-metal sheen along the face
        hg = QLinearGradient(face, 0, face + 12 * sg, 0)
        hg.setColorAt(0.0, QColor(255, 255, 255, 0))
        hg.setColorAt(0.35, QColor(255, 255, 255, 34))
        hg.setColorAt(0.7, QColor(255, 255, 255, 0))
        p.fillRect(QRectF(rr), hg)
        # neon seam just inside the slit (the chip's light leaking out)
        gx = face + 1.4 * sg
        glow = qc(mode.primary, 70)
        p.setPen(QPen(glow, 2.6))
        p.drawLine(QPointF(gx + 2.6 * sg, top + 7), QPointF(gx + 2.6 * sg, bot - 7))
        p.setPen(QPen(qc(mix(mode.primary, (255, 255, 255), 0.35), 200), 0.7))
        p.drawLine(QPointF(gx + 2.6 * sg, top + 8), QPointF(gx + 2.6 * sg, bot - 8))
        # the aperture: dark groove right at the lip
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(2, 2, 3, 250))
        p.drawRoundedRect(QRectF(min(gx, gx + 1.9 * sg), top + 4.0, 1.9, bot - top - 8.0), 0.9, 0.9)
        # lip: chrome edge with a thin neon trim (reads as the slot's mouth)
        p.setPen(QPen(qc(mode.primary, 90), 1.6))
        p.drawLine(QPointF(face + 0.6 * sg, top + rad - 2), QPointF(face + 0.6 * sg, bot - rad + 2))
        p.setPen(QPen(qc(mix(mode.primary, (255, 255, 255), 0.55), 210), 0.55))
        p.drawLine(QPointF(face + 0.45 * sg, top + rad - 1), QPointF(face + 0.45 * sg, bot - rad + 1))
        p.restore()
        # screws on the wide ends of the rail
        for y in (top + 7.0, bot - 7.0):
            w = abs(inner(y) - face)
            if w < 13:
                continue
            sx = face + min(w * 0.55, 10.5) * sg
            sgr = QRadialGradient(QPointF(sx - 0.5, y - 0.5), 2.6)
            sgr.setColorAt(0, QColor(214, 216, 224))
            sgr.setColorAt(0.6, QColor(110, 112, 122))
            sgr.setColorAt(1, QColor(36, 37, 42))
            p.setPen(QPen(QColor(0, 0, 0, 200), 0.5))
            p.setBrush(sgr)
            p.drawEllipse(QPointF(sx, y), 1.7, 1.7)
            p.setPen(QPen(QColor(20, 20, 24), 0.55))
            p.drawLine(QPointF(sx - 1.0, y - 0.6), QPointF(sx + 1.0, y + 0.6))
        p.end()
        return QPixmap.fromImage(img.scaled(rr.width(), rr.height(), Qt.AspectRatioMode.IgnoreAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))


    def paint_display(self, p, d, mi, mode, np_, an, now):
        pass

    def animating(self, np_, an) -> bool:
        return self.motion.moving

    def _strip_offsets(self, now):
        t = now - self.t_text
        return tuple(s.offset(t) for k, s in self._strips.items()
                     if isinstance(s, TextStrip) and s.overflows())

    def frame_key(self, np_, an, now):
        """Everything visible on the chip; repaint only when this changes."""
        pos = np_.pos_at(now) // 1000 if np_ is not None else 0
        return (self.motion.e, self._strip_offsets(now), pos,
                np_.display_key() if np_ is not None else None,
                self.hover_item, self.pressed_item)


class PlaylistChip(BaseChip):
    kind = "playlist"
    LABEL = "VX-PL01"
    W = 178
    H = 128
    ROW = 11
    ART = 40

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.scroll = 0.0            # px
        self._strips = {}
        self._list_cache = None
        self._list_key = None
        self._drag = None
        self.files_provider = None   # callable -> (files, current_index)
        self.info_provider = tags.track_info

    # ---- layout inside the display (display-local coords)
    def _list_rect(self, d: QRect) -> QRect:
        top = self.ART + 12
        return QRect(d.x() + 2, d.y() + top, d.width() - 4, d.height() - top - 2)

    def _strip(self, key, text, f, color):
        s = self._strips.get(key)
        if s is None or s.text != text or s.pm is None:
            s = TextStrip(text, f, color)
            self._strips[key] = s
        return s

    def paint_display(self, p, d, mi, mode, np_, an, now):
        prim = mode.primary
        bright = qc(mix(prim, (255, 255, 255), 0.72))
        dim = qc(mix(prim, (150, 150, 160), 0.55), 210)
        # -- art
        art = None
        if np_ is not None and np_.art_key:
            img = tags.thumbnail(np_.art_key, self.ART, self.ART)
            if img is not None:
                art = img
        ax, ay = d.x() + 3, d.y() + 3
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.setBrush(QColor(0, 0, 0))
        p.drawRect(QRectF(ax - 1, ay - 1, self.ART + 1, self.ART + 1))
        if art is not None:
            p.drawImage(ax, ay, art)
        else:
            ph = self._strips.get(("ph", mi))
            if ph is None:
                ph = _art_placeholder(self.ART, self.ART, mode)
                self._strips[("ph", mi)] = ph
            p.drawPixmap(ax, ay, ph)
        # glass glare over the art
        g = QLinearGradient(ax, ay, ax + self.ART, ay + self.ART)
        g.setColorAt(0, QColor(255, 255, 255, 40))
        g.setColorAt(0.45, QColor(255, 255, 255, 0))
        p.fillRect(QRect(ax, ay, self.ART, self.ART), g)
        # -- text block
        tx = ax + self.ART + 5
        avail = d.right() - tx - 2
        title = (np_.title if np_ and np_.title else "NO MEDIA")
        artist = (np_.artist if np_ and np_.artist else ("—" if np_ and np_.title else "eject / open files"))
        st = self._strip(("t", mi), title, font(10, True), bright)
        sa = self._strip(("a", mi), artist, font(9), dim)
        st.draw(p, tx, ay - 1, avail, now - self.t_text)
        sa.draw(p, tx, ay + 13, avail, now - self.t_text)
        # source badge + state + time (pixel font)
        y3 = ay + 28
        src = "LOCAL" if (np_ is None or np_.source != "smtc") else (np_.app or "MEDIA").upper()[:10]
        bw = pixel_text_width(src) + 5
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(prim, 200))
        p.drawRoundedRect(QRectF(tx, y3 - 1, bw, 7), 1.2, 1.2)
        ptext(p, tx + 2.5, y3, src, QColor(10, 6, 8))
        ix = tx + bw + 4
        playing = bool(np_ and np_.playing)
        p.setBrush(bright)
        p.drawPath(icon_path("play", ix + 3, y3 + 2.5, 6, playing))
        if np_ is not None and (np_.duration_ms or np_.position_ms):
            tt = f"{fmt_time(np_.pos_at(now))}/{fmt_time(np_.duration_ms)}"
            ptext(p, ix + 9, y3, tt, dim)
        # progress bar
        py = ay + self.ART - 2
        pw = d.right() - tx - 3
        p.setBrush(QColor(255, 255, 255, 26))
        p.drawRect(QRectF(tx, py, pw, 2))
        if np_ is not None and np_.duration_ms:
            k = min(1.0, np_.pos_at(now) / max(1, np_.duration_ms))
            p.setBrush(qc(prim, 230))
            p.drawRect(QRectF(tx, py, pw * k, 2))
            p.setBrush(qc(mix(prim, (255, 255, 255), 0.6)))
            p.drawRect(QRectF(tx + pw * k - 1, py - 1, 2, 4))
        # separator
        sy = d.y() + self.ART + 8
        p.setPen(QPen(qc(prim, 60), 1))
        p.drawLine(d.x() + 2, sy, d.right() - 2, sy)
        p.setPen(QPen(QColor(0, 0, 0, 160), 1))
        p.drawLine(d.x() + 2, sy + 1, d.right() - 2, sy + 1)
        lr = self._list_rect(d)
        if np_ is not None and np_.source == "smtc":
            self._paint_smtc_info(p, lr, mode, np_, now)
        else:
            self._paint_list(p, lr, mi, mode, now)
        # display glass: faint scanlines + top glare (cached overlay)
        p.drawPixmap(d.x(), d.y(), scanlines(d.width(), d.height()))

    def _paint_smtc_info(self, p, r: QRect, mode, np_, now):
        prim = mode.primary
        dim = qc(mix(prim, (150, 150, 160), 0.55), 200)
        on = qc(mix(prim, (255, 255, 255), 0.35))
        off = qc(prim, 28)
        y = r.y() + 3
        if np_.album:
            sa = self._strip(("al", id(mode)), np_.album, font(9), dim)
            sa.draw(p, r.x() + 2, y - 1, r.width() - 4, now - self.t_text)
            y += 13
        # big 7-seg position
        t = fmt_time(np_.pos_at(now))
        h = 15
        seg = seg_pixmap(t, h, on, off)
        p.drawPixmap(r.x() + 1, y - 1, seg)
        ptext(p, r.x() + 4 + seg.width(), y + h - 5, "/" + fmt_time(np_.duration_ms), dim)
        y += h + 6
        note = "QUEUE N/A - WINDOWS MEDIA"
        ptext(p, r.x() + 2, min(y, r.bottom() - 6), note, qc(prim, 110))

    def _rows(self):
        if self.files_provider is None:
            return [], -1
        return self.files_provider()

    def _list_pixmap(self, width, mi, mode, files, cur):
        key = (width, mi, len(files), cur, tuple(files[:1]), self.hover_item)
        if self._list_key == key and self._list_cache is not None:
            return self._list_cache
        n = len(files)
        h = max(1, n * self.ROW)
        img = QImage(width, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        prim = mode.primary
        f = font(9)
        fm = QFontMetrics(f)
        for i, path in enumerate(files):
            y = i * self.ROW
            if i == cur:
                p.fillRect(QRect(0, y, width, self.ROW), qc(prim, 58))
                p.fillRect(QRect(0, y, 2, self.ROW), qc(prim, 255))
            elif self.hover_item == ("row", i):
                p.fillRect(QRect(0, y, width, self.ROW), QColor(255, 255, 255, 18))
            num = f"{i + 1:02d}"
            pixel_text(p, 4, y + 3, num, qc(prim, 200 if i == cur else 110))   # (list pixmap is cached)
            ti = self.info_provider(path)
            label = ti.title
            col = qc(mix(prim, (255, 255, 255), 0.8)) if i == cur else qc(mix(prim, (190, 190, 200), 0.6), 220)
            p.setFont(f)
            p.setPen(col)
            text = fm.elidedText(label, Qt.TextElideMode.ElideRight, width - 20)
            p.drawText(QPointF(14, y + fm.ascent() - 1), text)
        p.end()
        self._list_cache = QPixmap.fromImage(img)
        self._list_key = key
        return self._list_cache

    def _paint_list(self, p, r: QRect, mi, mode, now):
        files, cur = self._rows()
        prim = mode.primary
        if not files:
            pixel_text(p, r.x() + 4, r.y() + 6, "NO LOCAL TRACKS", qc(prim, 140))
            pixel_text(p, r.x() + 4, r.y() + 14, "RIGHT-CLICK > OPEN FILES", qc(prim, 90))
            return
        lw = r.width() - 4
        self._clamp_scroll(r, len(files))
        pm = self._list_pixmap(lw, mi, mode, files, cur)
        p.save()
        p.setClipRect(r)
        p.drawPixmap(r.x(), r.y() - int(self.scroll), pm)
        p.restore()
        # scrollbar
        total = len(files) * self.ROW
        if total > r.height():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 20))
            p.drawRect(QRectF(r.right() - 2, r.y(), 2, r.height()))
            th = max(6, r.height() * r.height() / total)
            ty = r.y() + (r.height() - th) * (self.scroll / max(1, total - r.height()))
            p.setBrush(qc(prim, 200))
            p.drawRect(QRectF(r.right() - 2, ty, 2, th))

    def _clamp_scroll(self, r: QRect, n: int):
        mx = max(0, n * self.ROW - r.height())
        self.scroll = min(max(0.0, self.scroll), mx)

    def ensure_visible(self, geo, idx):
        d = geo["display"]
        r = self._list_rect(d)
        y = idx * self.ROW
        if y < self.scroll:
            self.scroll = y
        elif y + self.ROW > self.scroll + r.height():
            self.scroll = y + self.ROW - r.height()

    # ---- input (body coords)
    def list_rect_body(self, geo) -> QRect:
        return self._list_rect(self.display_rect(geo))

    def row_at(self, geo, x, y):
        r = self.list_rect_body(geo)
        if not r.contains(x, y):
            return None
        files, _ = self._rows()
        i = int((y - r.y() + self.scroll) // self.ROW)
        return i if 0 <= i < len(files) else None

    def wheel(self, geo, steps):
        self.scroll -= steps * self.ROW * 2
        files, _ = self._rows()
        self._clamp_scroll(self.list_rect_body(geo), len(files))

    def animating(self, np_, an) -> bool:
        if self.motion.moving:
            return True
        if not self.motion.visible:
            return False
        # marquee or running clock need frames
        if any(isinstance(s, TextStrip) and s.overflows() for s in self._strips.values()):
            return True
        return bool(np_ and np_.playing)

    def frame_key(self, np_, an, now):
        files, cur = self._rows()
        return super().frame_key(np_, an, now) + (int(self.scroll), cur, len(files))


class MiniChip(BaseChip):
    kind = "mini"
    LABEL = "VX-MP02"
    W = 162
    H = 100
    ART = 34
    BTNS = ("prev", "play", "next")

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._strips = {}
        self.detached = False
        self.led = np.zeros(12)

    # ---- layout (display-local, origin at display top-left)
    def layout(self, d: QRect, dock_mode=False):
        bx = d.x() + 4
        by = d.y() + self.ART + 9
        btn = {}
        for i, n in enumerate(self.BTNS):
            btn[n] = QRect(bx + i * 26, by, 23, 17)
        btn["dock" if dock_mode else "detach"] = QRect(d.right() - 20, by, 17, 17)
        led = QRect(d.x() + 4, by + 22, d.width() - 8, 7)
        return btn, led

    def _strip(self, key, text, f, color):
        s = self._strips.get(key)
        if s is None or s.text != text:
            s = TextStrip(text, f, color)
            self._strips[key] = s
        return s

    def step_led(self, an, dt):
        """Bass-weighted LED bar with fast attack / slow release."""
        bands = getattr(an, "bands", None)
        target = np.zeros(len(self.led))
        if bands is not None and len(bands):
            lvl = min(1.0, 0.65 * float(an.bass) + 0.55 * float(np.mean(bands[: max(1, len(bands) // 3)])))
            target[:] = (np.arange(len(self.led)) < lvl * len(self.led) + 0.3).astype(float)
        k = np.where(target > self.led, 1.0, 1 - math.exp(-dt * 7))
        self.led = self.led + (target - self.led) * k

    def paint_display(self, p, d, mi, mode, np_, an, now, dock_mode=False):
        prim = mode.primary
        bright = qc(mix(prim, (255, 255, 255), 0.72))
        dim = qc(mix(prim, (150, 150, 160), 0.55), 210)
        ax, ay = d.x() + 3, d.y() + 3
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.setBrush(QColor(0, 0, 0))
        p.drawRect(QRectF(ax - 1, ay - 1, self.ART + 1, self.ART + 1))
        img = tags.thumbnail(np_.art_key, self.ART, self.ART) if (np_ and np_.art_key) else None
        if img is not None:
            p.drawImage(ax, ay, img)
        else:
            ph = self._strips.get(("ph", mi))
            if ph is None:
                ph = _art_placeholder(self.ART, self.ART, mode)
                self._strips[("ph", mi)] = ph
            p.drawPixmap(ax, ay, ph)
        g = QLinearGradient(ax, ay, ax + self.ART, ay + self.ART)
        g.setColorAt(0, QColor(255, 255, 255, 40))
        g.setColorAt(0.45, QColor(255, 255, 255, 0))
        p.fillRect(QRect(ax, ay, self.ART, self.ART), g)
        tx = ax + self.ART + 5
        avail = d.right() - tx - 2
        title = np_.title if np_ and np_.title else "NO MEDIA"
        artist = np_.artist if np_ and np_.artist else ""
        self._strip(("t", mi), title, font(10, True), bright).draw(p, tx, ay - 1, avail, now - self.t_text)
        if artist:
            self._strip(("a", mi), artist, font(9), dim).draw(p, tx, ay + 12, avail, now - self.t_text)
        # source + time
        src = "LOCAL" if (np_ is None or np_.source != "smtc") else (np_.app or "MEDIA").upper()[:10]
        ptext(p, tx, ay + 26, src, qc(prim, 190))
        if np_ is not None and np_.duration_ms:
            tt = fmt_time(np_.pos_at(now))
            ptext(p, d.right() - 3 - pixel_text_width(tt), ay + 26, tt, dim)
        btn, led = self.layout(d, dock_mode)
        playing = bool(np_ and np_.playing)
        for name, r in btn.items():
            self._paint_btn(p, QRectF(r), name, mode, playing)
        # LED bar
        n = len(self.led)
        sw = (led.width() - (n - 1) * 1.5) / n
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            v = float(self.led[i])
            c = mix(prim, (255, 255, 255), 0.55) if i >= n - 2 else (prim if i >= n * 0.5 else mode.secondary)
            p.setBrush(qc(c, int(30 + 225 * v)))
            p.drawRoundedRect(QRectF(led.x() + i * (sw + 1.5), led.y(), sw, led.height()), 1, 1)
        # progress hairline
        if np_ is not None and np_.duration_ms:
            k = min(1.0, np_.pos_at(now) / max(1, np_.duration_ms))
            p.setBrush(QColor(255, 255, 255, 24))
            p.drawRect(QRectF(led.x(), led.bottom() + 3, led.width(), 1.5))
            p.setBrush(qc(prim, 230))
            p.drawRect(QRectF(led.x(), led.bottom() + 3, led.width() * k, 1.5))
        p.drawPixmap(d.x(), d.y(), scanlines(d.width(), d.height()))

    def _paint_btn(self, p, r: QRectF, name, mode, playing):
        down = self.pressed_item == name
        hov = self.hover_item == name
        rr = r.translated(0, 1) if down else r
        p.setPen(QPen(QColor(0, 0, 0), 1.0))
        g = QLinearGradient(0, rr.top(), 0, rr.bottom())
        if down:
            g.setColorAt(0, QColor(30, 31, 36))
            g.setColorAt(1, QColor(62, 64, 72))
        else:
            g.setColorAt(0, QColor(150, 154, 164))
            g.setColorAt(0.12, QColor(84, 87, 96))
            g.setColorAt(1, QColor(26, 27, 31))
        p.setBrush(g)
        p.drawRoundedRect(rr, 2.5, 2.5)
        col = qc(mix(mode.primary, (255, 255, 255), 0.55 if hov else 0.25))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        c = rr.center()
        if name in ("detach", "dock"):
            # little arrow out of / into a box
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(col, 1.1))
            p.drawRect(QRectF(c.x() - 4.5, c.y() - 2.5, 7, 7))
            path = QPainterPath()
            if name == "detach":
                path.moveTo(c.x() - 1, c.y() + 1)
                path.lineTo(c.x() + 4.5, c.y() - 4.5)
                path.moveTo(c.x() + 1, c.y() - 4.5)
                path.lineTo(c.x() + 4.5, c.y() - 4.5)
                path.lineTo(c.x() + 4.5, c.y() - 1)
            else:
                path.moveTo(c.x() + 4.5, c.y() - 4.5)
                path.lineTo(c.x() - 1, c.y() + 1)
                path.moveTo(c.x() - 1, c.y() - 2.5)
                path.lineTo(c.x() - 1, c.y() + 1)
                path.lineTo(c.x() + 2.5, c.y() + 1)
            p.drawPath(path)
        else:
            p.drawPath(icon_path(name, c.x(), c.y(), 9, playing if name == "play" else None))
        if hov and not down:
            p.setPen(QPen(qc(mode.primary, 120), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rr.adjusted(-0.5, -0.5, 0.5, 0.5), 3, 3)

    def item_at(self, geo, x, y, origin=None, dock_mode=False):
        d = self.display_rect(geo) if origin is None else geo["display"].translated(origin)
        btn, _ = self.layout(d, dock_mode)
        for n, r in btn.items():
            if r.adjusted(-1, -1, 1, 1).contains(x, y):
                return n
        return None

    def animating(self, np_, an) -> bool:
        if self.motion.moving:
            return True
        if not self.motion.visible:
            return False
        return True        # LED bar / clock / marquee

    def frame_key(self, np_, an, now):
        return super().frame_key(np_, an, now) + (tuple((self.led * 6).astype(int)),)


# ------------------------------------------------------------------ floating mini
class MiniWindow(QWidget):
    """The detached mini-player: the same cartridge, floating on its own."""

    dockRequested = Signal()
    commandRequested = Signal(str)

    def __init__(self, chip: MiniChip, host):
        super().__init__(None)
        self.chip = chip
        self.host = host              # VexaWindow (mode, media hub, analyser)
        self.setWindowTitle("Vexa mini")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMouseTracking(True)
        mode = config.MODES[host.mode_i]
        pm, geo = chip.frame(host.mode_i, mode)
        self.setFixedSize(*geo["size"])
        self._drag = None
        self._press = None
        if host.windowIcon() is not None:
            self.setWindowIcon(host.windowIcon())

    def geo(self):
        mi = self.host.mode_i
        return self.chip.frame(mi, config.MODES[mi])

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        mi = self.host.mode_i
        mode = config.MODES[mi]
        pm, geo = self.geo()
        p.drawPixmap(0, 0, pm)
        d = geo["display"]
        p.save()
        p.setClipRect(d)
        np_ = self.host.media.now()
        self.chip.paint_display(p, d, mi, mode, np_, self.host.an, self.host._now(), dock_mode=True)
        p.restore()
        paint_silk(p, geo["silk"], self.chip.LABEL, mode)
        p.end()

    def refresh(self):
        pm, geo = self.geo()
        self.update(geo["display"].adjusted(-1, -1, 1, 1))

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = int(e.position().x()), int(e.position().y())
        pm, geo = self.geo()
        it = self.chip.item_at(geo, x, y, origin=QRect(0, 0, 0, 0).topLeft(), dock_mode=True)
        if it:
            self._press = it
            self.chip.pressed_item = it
            self.host.sounds.play("press")
            self.update()
            return
        self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)
            return
        x, y = int(e.position().x()), int(e.position().y())
        pm, geo = self.geo()
        it = self.chip.item_at(geo, x, y, origin=QRect(0, 0, 0, 0).topLeft(), dock_mode=True)
        if it != self.chip.hover_item:
            self.chip.hover_item = it
            self.update()

    def mouseReleaseEvent(self, e):
        x, y = int(e.position().x()), int(e.position().y())
        if self._press is not None:
            it = self._press
            self._press = None
            self.chip.pressed_item = None
            self.host.sounds.play("btn_release")
            pm, geo = self.geo()
            if self.chip.item_at(geo, x, y, origin=QRect(0, 0, 0, 0).topLeft(), dock_mode=True) == it:
                if it == "dock":
                    self.dockRequested.emit()
                else:
                    self.commandRequested.emit(it)
            self.update()
        self._drag = None

    def contextMenuEvent(self, e):
        m = QMenu(self)
        m.addAction("Dock Vexa", self.dockRequested.emit)
        m.addSeparator()
        m.addAction("Exit", self.host.close)
        m.exec(e.globalPos())

    def closeEvent(self, e):
        # closing the mini window (Alt+F4) brings Vexa back rather than orphaning her
        if not getattr(self, "_docking", False):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self, self.dockRequested.emit)   # dropped if we're gone
        super().closeEvent(e)
