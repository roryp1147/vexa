"""
Chest button row: a gunmetal tray that ratchets out of a slot under the
corset band on hover and retracts after ~3 s idle.

Buttons (procedural keycaps): shuffle, prev, play/pause, [LCD], stop, next,
repeat. AlienMonitor-style feel: hover glow, visible press travel (cap drops
2 px, bevel flips darker), chunky relay clack on press, tick on release.

The tray is rendered into a cached QPixmap that is rebuilt only when its
state changes (hover / press / toggles / LCD second / mode).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (QColor, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPixmap, QRadialGradient)

from . import config
from .draw import icon_path, pixel_text, pixel_text_width, rivet, seven_seg, seven_seg_width
from .qtutil import mix, qc

ITEMS = ["shuffle", "prev", "play", "lcd", "stop", "next", "repeat"]
BTN_W, BTN_H, LCD_W, GAP, MARGIN = 30, 25, 74, 4, 8
TRAY_H = 46
OUT_DUR = 0.46
IN_DUR = 0.36


def _clunk_out(u: float) -> float:
    """Extension curve: motor spin-up pause, 4 ratchet hitches, overshoot,
    latch drop. u in 0..1 -> position (can exceed 1 slightly)."""
    if u < 0.10:
        return 0.0
    if u < 0.80:
        v = (u - 0.10) / 0.70
        steps = 4
        k = math.floor(v * steps)
        f = min(1.0, (v * steps - k) * 1.35)
        return (k + f * f * (3 - 2 * f)) / steps * 1.06
    if u < 0.88:
        return 1.06
    v = (u - 0.88) / 0.12
    return 1.06 - 0.06 * v + 0.025 * math.sin(v * math.pi) * (1 - v)


def _clunk_in(u: float) -> float:
    if u < 0.08:
        return 1.0
    v = (u - 0.08) / 0.92
    steps = 3
    k = math.floor(v * steps)
    f = min(1.0, (v * steps - k) * 1.25)
    return 1.0 - (k + f * f) / steps


class ChestPanel:
    def __init__(self, layout, sounds):
        P = layout["panel"]
        self.sounds = sounds
        self.slot_y = int(round(P["slot_y"]))
        ax = layout["axis_x"]
        n_btn = len(ITEMS) - 1
        self.W = MARGIN * 2 + n_btn * BTN_W + LCD_W + (len(ITEMS) - 1) * GAP
        self.H = TRAY_H
        self.x = int(round(ax - self.W / 2))
        hx, hy, hw, hh = layout["chest_hover"]
        self.hover_zone = QRect(int(hx), int(hy), int(hw), int(hh))
        self.items = {}
        x = MARGIN
        for name in ITEMS:
            w = LCD_W if name == "lcd" else BTN_W
            h = 30 if name == "lcd" else BTN_H
            self.items[name] = QRect(x, 7 + (BTN_H - h) // 2 + 1, w, h)
            x += w + GAP
        self.pos = 0.0               # 0 hidden .. ~1 out
        self.state = "in"            # in | extending | out | retracting
        self.t0 = 0.0
        self.last_activity = -1e9
        self.hover = None
        self.pressed = None
        self.latched = False
        self.pinned = False          # "Controls always out": never auto-retracts
        # model state shown on the tray
        self.playing = False
        self.shuffle = False
        self.repeat = 0
        self.elapsed = 0
        self.total = 0
        self.mode = 0
        self.lcd_test = None         # boot self-test: "all" segments lit / "blank"
        self._cache = None
        self._cache_key = None

    # ---------------------------------------------------------------- geometry
    def tray_rect(self) -> QRect:
        """Current tray rect in window coords (part above slot is hidden)."""
        top = self.slot_y - self.H + int(round(self.pos * self.H))
        return QRect(self.x, top, self.W, self.H)

    def dirty_rect(self) -> QRect:
        return QRect(self.x - 8, self.slot_y - 4, self.W + 16, self.H + 14)

    @property
    def animating(self) -> bool:
        return self.state in ("extending", "retracting")

    @property
    def visible(self) -> bool:
        return self.pos > 0.001

    # ---------------------------------------------------------------- behaviour
    def poke(self, now: float):
        """Mouse activity over the chest / tray."""
        self.last_activity = now
        if self.state in ("in", "retracting"):
            self._extend(now)

    def set_pinned(self, on: bool, now: float):
        self.pinned = on
        self.last_activity = now
        if on and self.state in ("in", "retracting"):
            self._extend(now)

    def _extend(self, now):
        start = self.pos
        self.state = "extending"
        # resume from the current position if we were mid-retract
        self.t0 = now - (OUT_DUR * 0.1 + OUT_DUR * 0.7 * min(1.0, start / 1.06) if start > 0 else 0)
        self.latched = False
        self.sounds.play("servo_out")

    def _retract(self, now):
        self.state = "retracting"
        self.t0 = now
        self.hover = None
        self.pressed = None
        self.sounds.play("servo_in")

    def update(self, now: float, mouse_inside_tray: bool) -> bool:
        """Advance animation. Returns True if anything visible changed."""
        old = self.pos
        if self.state == "extending":
            u = (now - self.t0) / OUT_DUR
            if u >= 0.80 and not self.latched:
                self.latched = True
                self.sounds.play("latch")
            if u >= 1.0:
                self.state, self.pos = "out", 1.0
            else:
                self.pos = _clunk_out(u)
        elif self.state == "retracting":
            u = (now - self.t0) / IN_DUR
            if u >= 1.0:
                self.state, self.pos = "in", 0.0
                self.sounds.play("btn_release")
            else:
                self.pos = _clunk_in(u)
        elif self.state == "out":
            if self.pinned:
                pass
            elif mouse_inside_tray or self.pressed:
                self.last_activity = now
            elif now - self.last_activity > config.PANEL_IDLE_S:
                self._retract(now)
        return abs(self.pos - old) > 1e-4

    def item_at(self, x: int, y: int):
        if self.state != "out":
            return None
        r = self.tray_rect()
        lx, ly = x - r.x(), y - r.y()
        for name, ir in self.items.items():
            if name != "lcd" and ir.adjusted(-2, -2, 2, 2).contains(lx, ly):
                return name
        return None

    def contains(self, x: int, y: int) -> bool:
        return self.visible and self.tray_rect().contains(x, y) and y >= self.slot_y

    # ---------------------------------------------------------------- painting
    def paint_slot(self, p: QPainter, mi: int):
        """The seam under the corset band (always visible)."""
        m = config.MODES[mi]
        y = self.slot_y
        x0, x1 = self.x + 6, self.x + self.W - 6
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(4, 4, 6, 235))
        p.drawRoundedRect(QRectF(x0, y - 1.5, x1 - x0, 3.0), 1.5, 1.5)
        p.setPen(QPen(QColor(150, 154, 164, 140), 0.8))
        p.drawLine(QPointF(x0 + 2, y + 1.8), QPointF(x1 - 2, y + 1.8))
        # chunky end brackets with screws
        for bx in (x0 - 5, x1 - 1):
            g = QLinearGradient(0, y - 4, 0, y + 4)
            g.setColorAt(0, QColor(150, 154, 164))
            g.setColorAt(0.5, QColor(70, 73, 82))
            g.setColorAt(1, QColor(30, 31, 36))
            p.setPen(QPen(QColor(5, 5, 7, 230), 0.8))
            p.setBrush(g)
            p.drawRoundedRect(QRectF(bx, y - 4, 6, 8), 1.5, 1.5)
            rivet(p, bx + 3, y, 1.2)
        # tiny status LED at the slot's end
        p.setPen(Qt.PenStyle.NoPen)
        led = qc(m.primary, 230 if self.playing else 90)
        p.setBrush(led)
        p.drawEllipse(QPointF(x1 - 9, y), 1.2, 1.2)
        p.restore()

    def paint(self, p: QPainter, mi: int):
        if not self.visible:
            return
        pm = self._tray_pixmap(mi)
        r = self.tray_rect()
        p.save()
        p.setClipRect(QRect(self.x - 8, self.slot_y, self.W + 16, self.H + 14))
        # drop shadow on the chassis
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 90))
        p.drawRoundedRect(QRectF(r.x() + 2, r.y() + 3, r.width(), r.height()), 5, 5)
        p.drawPixmap(r.topLeft(), pm)
        # shadow of the slot lip on the tray
        g = QLinearGradient(0, self.slot_y, 0, self.slot_y + 6)
        g.setColorAt(0, QColor(0, 0, 0, 170))
        g.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(QRect(r.x(), self.slot_y, r.width(), 6), g)
        p.restore()

    def _key(self, mi):
        return (mi, self.hover, self.pressed, self.playing, self.shuffle, self.repeat,
                self.elapsed // 1000, self.total // 1000, self.lcd_test)

    def _tray_pixmap(self, mi) -> QPixmap:
        key = self._key(mi)
        if self._cache is not None and key == self._cache_key:
            return self._cache
        m = config.MODES[mi]
        pm = QPixmap(self.W, self.H)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.W, self.H
        body = QPainterPath()
        body.addRoundedRect(QRectF(0.5, -6, W - 1, H + 5.5), 6, 6)
        g = QLinearGradient(0, 0, 0, H)
        g.setColorAt(0.0, QColor(70, 73, 82))
        g.setColorAt(0.45, QColor(44, 46, 53))
        g.setColorAt(1.0, QColor(24, 25, 29))
        p.fillPath(body, g)
        # brushed-metal streaks
        p.setPen(QPen(QColor(255, 255, 255, 10), 1))
        for yy in range(2, H, 3):
            p.drawLine(4, yy, W - 4, yy)
        p.setPen(QPen(QColor(0, 0, 0, 200), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(body)
        p.setPen(QPen(QColor(160, 165, 176, 90), 0.8))
        p.drawLine(QPointF(4, H - 1.6), QPointF(W - 4, H - 1.6))
        # neon piping along the bottom lip
        p.setPen(QPen(qc(m.primary, 70), 4))
        p.drawLine(QPointF(14, H - 4), QPointF(W - 14, H - 4))
        p.setPen(QPen(qc(mix(m.primary, (255, 255, 255), 0.35), 230), 1.2))
        p.drawLine(QPointF(14, H - 4), QPointF(W - 14, H - 4))
        for rx in (4.5, W - 4.5):
            rivet(p, rx, 9, 1.7)
            rivet(p, rx, H - 9, 1.7)
        for name, r in self.items.items():
            if name == "lcd":
                self._paint_lcd(p, r, m)
            else:
                self._paint_key(p, name, r, m)
        p.end()
        self._cache, self._cache_key = pm, key
        return pm

    def _paint_key(self, p, name, r: QRect, m):
        hov = self.hover == name
        prs = self.pressed == name
        well = QRectF(r).adjusted(-1.5, -1.5, 1.5, 3.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 8, 10, 230))
        p.drawRoundedRect(well, 4, 4)
        travel = 2.0 if prs else 0.0
        cap = QRectF(r).translated(0, travel)
        if not prs:   # key side (gives it height)
            p.setBrush(QColor(18, 19, 22))
            p.drawRoundedRect(cap.translated(0, 2.0), 3.5, 3.5)
        if hov:
            for w_, a_ in ((6, 40), (3, 80)):
                p.setPen(QPen(qc(m.primary, a_), w_))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(cap, 3.5, 3.5)
        g = QLinearGradient(0, cap.top(), 0, cap.bottom())
        if prs:
            g.setColorAt(0, QColor(30, 31, 36))
            g.setColorAt(1, QColor(46, 48, 55))
        else:
            top = (104, 108, 118) if hov else (88, 92, 102)
            g.setColorAt(0, qc(top))
            g.setColorAt(0.5, QColor(56, 59, 67))
            g.setColorAt(1, QColor(36, 38, 44))
        p.setPen(QPen(QColor(6, 6, 8), 1))
        p.setBrush(g)
        p.drawRoundedRect(cap, 3.5, 3.5)
        # bevel
        inner = cap.adjusted(1.2, 1.2, -1.2, -1.2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if prs:
            p.setPen(QPen(QColor(0, 0, 0, 170), 1))
            p.drawLine(inner.topLeft(), inner.topRight())
        else:
            p.setPen(QPen(QColor(220, 224, 232, 110 if hov else 80), 1))
            p.drawLine(QPointF(inner.left() + 2, inner.top()), QPointF(inner.right() - 2, inner.top()))
            p.setPen(QPen(QColor(0, 0, 0, 120), 1))
            p.drawLine(QPointF(inner.left() + 2, inner.bottom()), QPointF(inner.right() - 2, inner.bottom()))
        # icon
        state = self.playing if name == "play" else None
        ip = icon_path(name, cap.center().x(), cap.center().y() + 0.5, 11.0, state)
        lit = (name == "shuffle" and self.shuffle) or (name == "repeat" and self.repeat) \
            or (name == "play" and self.playing)
        a = 255 if (hov or prs) else 200
        col = mix(m.primary, (255, 255, 255), 0.35 if hov else 0.1)
        if prs:
            col = mix(m.primary, (0, 0, 0), 0.25)
        if hov or lit:
            p.setPen(QPen(qc(m.primary, 90), 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(ip)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(col, a))
        p.drawPath(ip)
        # toggle LED
        if name in ("shuffle", "repeat", "play"):
            lx, ly = cap.right() - 4, cap.top() + 3.5
            on = lit
            p.setBrush(qc(m.primary, 255) if on else QColor(40, 20, 24))
            if on:
                rg = QRadialGradient(QPointF(lx, ly), 4)
                rg.setColorAt(0, qc(m.primary, 160))
                rg.setColorAt(1, qc(m.primary, 0))
                p.setBrush(rg)
                p.drawEllipse(QPointF(lx, ly), 4, 4)
                p.setBrush(qc(mix(m.primary, (255, 255, 255), 0.5)))
            p.drawEllipse(QPointF(lx, ly), 1.3, 1.3)
            if name == "repeat" and self.repeat == 2:
                pixel_text(p, cap.left() + 3, cap.top() + 2, "1", qc(m.primary))

    def _paint_lcd(self, p, r: QRect, m):
        rr = QRectF(r)
        p.setPen(QPen(QColor(4, 4, 5), 1))
        bez = QLinearGradient(0, rr.top(), 0, rr.bottom())
        bez.setColorAt(0, QColor(20, 21, 24))
        bez.setColorAt(1, QColor(80, 84, 94))
        p.setBrush(bez)
        p.drawRoundedRect(rr.adjusted(-1.5, -1.5, 1.5, 1.5), 3, 3)
        glass = rr.adjusted(1, 1, -1, -1)
        g = QLinearGradient(0, glass.top(), 0, glass.bottom())
        bg = mix(m.primary, (0, 0, 0), 0.9)
        g.setColorAt(0, qc(mix(bg, (0, 0, 0), 0.2)))
        g.setColorAt(1, qc(bg))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(glass, 2, 2)
        on = qc(mix(m.primary, (255, 255, 255), 0.25))
        off = qc(m.primary, 22)
        el = self._fmt(self.elapsed)
        if self.lcd_test:
            # segment test: every segment lit, then a blank frame
            test_on = on if self.lcd_test == "all" else off
            h = 13
            w = seven_seg_width(h, "88:88")
            seven_seg(p, glass.right() - w - 5, glass.top() + 3, h, "88:88", test_on, off)
            if self.lcd_test == "all":
                tw = pixel_text_width("88:88")
                pixel_text(p, glass.right() - tw - 5, glass.bottom() - 7, "88:88", qc(m.primary, 200))
                pixel_text(p, glass.left() + 4, glass.top() + 4, ">", qc(m.primary, 200))
            return
        h = 13
        w = seven_seg_width(h, el)
        seven_seg(p, glass.right() - w - 5, glass.top() + 3, h, el, on, off)
        tot = self._fmt(self.total) if self.total > 0 else "--:--"
        tw = pixel_text_width(tot)
        pixel_text(p, glass.right() - tw - 5, glass.bottom() - 7, tot, qc(m.primary, 200))
        pixel_text(p, glass.right() - tw - 13, glass.bottom() - 7, "/", qc(m.primary, 140))
        # play state glyph
        st = ">" if self.playing else ("=" if self.elapsed > 0 else "-")
        if st == "=":
            p.setBrush(qc(m.primary, 200))
            p.drawRect(QRectF(glass.left() + 4, glass.top() + 4, 1.5, 5))
            p.drawRect(QRectF(glass.left() + 6.5, glass.top() + 4, 1.5, 5))
        else:
            pixel_text(p, glass.left() + 4, glass.top() + 4, st, qc(m.primary, 200))
        # glass glare
        gl = QLinearGradient(0, glass.top(), 0, glass.top() + glass.height() * 0.45)
        gl.setColorAt(0, QColor(255, 255, 255, 34))
        gl.setColorAt(1, QColor(255, 255, 255, 0))
        p.setBrush(gl)
        p.drawRoundedRect(QRectF(glass.left(), glass.top(), glass.width(), glass.height() * 0.45), 2, 2)

    @staticmethod
    def _fmt(ms: int) -> str:
        s = max(0, int(ms // 1000))
        m, s = divmod(s, 60)
        if m >= 100:
            return f"{m // 60}:{m % 60:02d}"
        return f"{m}:{s:02d}"
