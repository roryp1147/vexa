"""
Scanning-eye mode: glossy all-black eyes with glowing violet HEART pupils.

Cat-tilted almond eyes, sclera and iris both glossy black (subtle catchlights
and a soft wet sheen), heavy smoky plum/black eyeshadow sweeping up towards
the brows and out to the temples, a sharp winged liner and thick dark lashes
(top + a few lower). The pupil is a small glowing violet heart that pulses
with the bass: an OUTLINE heart when quiet whose stroke thickens smoothly into
a FULL filled heart when loud. Now and then (strong drops / every so often)
the eyes flicker-INVERT: the whole eye glows violet with a black heart pupil.

Small-size crispness: the eyes are ~30 px wide on screen, so the heart is a
chunky ~9 px design. Every morph level is rendered at 8x and downscaled once,
and the heart is blitted at integer pixel positions (never resampled per
frame). Everything static (make-up, sclera, lids + lashes at 8 lid positions)
is rendered ONCE at 4x and cached; per frame we blit a few small pixmaps.
"""
from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QRadialGradient, QTransform)

SS = 4            # supersampling for the static layers
HSS = 8           # supersampling for the heart sprites
LEVELS = 8        # cached lid positions (0 = closed, LEVELS-1 = relaxed half-lid)
HEART_LEVELS = 12  # outline (0) -> full (HEART_LEVELS-1)


def _lerp(a, b, t):
    return a + (b - a) * t


def _cubic(p0, p1, p2, p3, t):
    u = 1 - t
    x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0]
    y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
    return x, y


def _quad(p0, p1, p2, t):
    u = 1 - t
    return (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])


def heart_path(cx: float, cy: float, w: float) -> QPainterPath:
    """A chunky heart (reads at ~7-10 px): full round lobes, a shallow notch
    and a short, slightly rounded point. Width w, height ~0.92 w, centred."""
    h = w * 0.92
    x0, y0 = cx - w / 2, cy - h / 2
    path = QPainterPath()

    def P(u, v):
        return QPointF(x0 + u * w, y0 + v * h)
    path.moveTo(P(0.50, 0.24))                       # notch
    path.cubicTo(P(0.40, 0.02), P(0.02, 0.00), P(0.02, 0.33))
    path.cubicTo(P(0.02, 0.58), P(0.30, 0.76), P(0.47, 0.97))
    path.quadTo(P(0.50, 1.01), P(0.53, 0.97))         # rounded point
    path.cubicTo(P(0.70, 0.76), P(0.98, 0.58), P(0.98, 0.33))
    path.cubicTo(P(0.98, 0.00), P(0.60, 0.02), P(0.50, 0.24))
    path.closeSubpath()
    return path


def _heart_sprite(size: int, m: float, fill: QColor, glow: QColor | None) -> QImage:
    """Heart at morph m (0 outline .. 1 full) on a size x size 1x canvas.
    The outline's stroke thickens inward until it closes into a solid fill."""
    S = size * HSS
    img = QImage(S, S, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = S / 2
    hw = (size - 2.0) * HSS                  # 1 px margin each side
    outer = heart_path(c, c + 0.03 * hw, hw)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(fill)
    p.drawPath(outer)
    # inner cut-out: a smaller heart (centred on the lobes' visual centre)
    # whose size shrinks to zero as m -> 1; min stroke ~1.15 px at 1x
    e = m * m * (3 - 2 * m)
    k = (1.0 - (2.3 * HSS) / hw) * (1.0 - e)
    if k > 0.02:
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        p.setBrush(QColor(0, 0, 0, 255))
        p.drawPath(heart_path(c, c - 0.02 * hw, hw * k))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    p.end()
    return img.scaled(size, size, Qt.AspectRatioMode.IgnoreAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


def _glow_sprite(size: int, pad: int, col: QColor, peak: float = 0.55) -> QImage:
    """Soft bloom around the heart (size + 2*pad canvas): the filled heart's
    coverage, box-blurred 3x (~gaussian) in numpy, tinted, premultiplied."""
    import numpy as np
    S = size + 2 * pad
    img = QImage(S * SS, S * SS, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(SS, SS)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255))
    p.drawPath(heart_path(S / 2, S / 2, size - 2))
    p.end()
    img = img.scaled(S, S, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(S, img.bytesPerLine() // 4, 4)[:, :S, 3]
    a = a.astype(np.float32) / 255.0
    r = max(1, pad // 3)
    for _ in range(3):
        for ax in (0, 1):
            c = np.cumsum(np.pad(a, [(r + 1, r) if i == ax else (0, 0) for i in range(2)], mode="constant"), axis=ax)
            a = (np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    a = np.clip(a / max(a.max(), 1e-6), 0, 1) * peak
    out = np.zeros((S, S, 4), np.uint8)
    out[..., 0] = (col.blue() / 255.0 * a * 255).astype(np.uint8)
    out[..., 1] = (col.green() / 255.0 * a * 255).astype(np.uint8)
    out[..., 2] = (col.red() / 255.0 * a * 255).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return QImage(np.ascontiguousarray(out).data, S, S, 4 * S, QImage.Format.Format_ARGB32_Premultiplied).copy()


class EyeArtist:
    """Draws one eye (outer corner towards +x); the other eye is mirrored."""

    HEART_FILL = QColor(214, 150, 255)      # glowing violet heart
    HEART_HOT = QColor(246, 224, 255)       # hot core of the full heart
    GLOW = QColor(176, 92, 255)

    def __init__(self, ew: float, eh: float, skin: QColor):
        self.w = ew
        self.h = eh
        self.skin = skin
        self.CW = int(math.ceil(ew * 2.1))
        self.CH = int(math.ceil(eh * 3.6))
        self.cx = self.CW / 2
        self.cy = self.CH * 0.60
        w, h = ew, eh
        self.inner = (-w / 2, h * 0.16)
        self.outer = (w / 2, -h * 0.26)          # upward cat-tilt
        self.up_c = ((-w * 0.22, -h * 1.02), (w * 0.26, -h * 1.00))
        self.lo_c = ((w * 0.26, h * 0.66), (-w * 0.20, h * 0.78))
        self.iris_r = h * 0.62
        self.iris_c = (w * 0.03, h * 0.10)
        # heart pupil: odd pixel size so it has a true centre column
        hs = int(round(h * 0.56))
        self.heart_size = hs + (1 - hs % 2) + 2     # + 1 px margin each side
        self._build()

    # ------------------------------------------------------------ geometry
    def _p(self, pt):
        return QPointF(self.cx + pt[0], self.cy + pt[1])

    def almond(self) -> QPainterPath:
        path = QPainterPath(self._p(self.inner))
        path.cubicTo(self._p(self.up_c[0]), self._p(self.up_c[1]), self._p(self.outer))
        path.cubicTo(self._p(self.lo_c[0]), self._p(self.lo_c[1]), self._p(self.inner))
        path.closeSubpath()
        return path

    def lid_ctrl(self, o: float):
        """Upper-lid curve control points for openness o (1 = relaxed half-lid)."""
        w, h = self.w, self.h
        open_c = ((-w * 0.22, -h * 0.34), (w * 0.26, -h * 0.50))
        closed_c = ((-w * 0.20, h * 0.78), (w * 0.26, h * 0.66))
        e = o * o * (3 - 2 * o)
        return ((_lerp(closed_c[0][0], open_c[0][0], e), _lerp(closed_c[0][1], open_c[0][1], e)),
                (_lerp(closed_c[1][0], open_c[1][0], e), _lerp(closed_c[1][1], open_c[1][1], e)))

    def lid_point(self, o, t):
        c1, c2 = self.lid_ctrl(o)
        return _cubic(self.inner, c1, c2, self.outer, t)

    def lower_point(self, t):
        return _cubic(self.inner, self.lo_c[1], self.lo_c[0], self.outer, t)

    # ------------------------------------------------------------ make-up
    def _blob(self, p, c, rx, ry, rot, stops):
        g = QRadialGradient(QPointF(0, 0), 1.0)
        for pos, col in stops:
            g.setColorAt(pos, col)
        p.save()
        p.translate(self._p(c))
        p.rotate(rot)
        p.scale(rx, ry)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawEllipse(QPointF(0, 0), 1.0, 1.0)
        p.restore()

    def _smoke(self, p: QPainter):
        """Heavy smoky eye: near-black plum at the lash line, sweeping up
        towards the brow and out to the temple, softly diffused (no hard
        edge, no panda: it fades to skin well below the brow)."""
        w, h = self.w, self.h
        # broad diffuse wash up to the brow
        self._blob(p, (w * 0.10, -h * 0.62), w * 0.66, h * 1.05, -8, [
            (0.0, QColor(30, 12, 40, 190)), (0.45, QColor(48, 20, 62, 140)),
            (0.72, QColor(84, 44, 104, 40)), (1.0, QColor(100, 60, 120, 0))])
        # deep lid colour, darkest along the lash line
        self._blob(p, (w * 0.12, -h * 0.30), w * 0.58, h * 0.70, -10, [
            (0.0, QColor(12, 4, 16, 235)), (0.50, QColor(20, 8, 28, 200)),
            (0.85, QColor(40, 16, 54, 70)), (1.0, QColor(40, 16, 54, 0))])
        # outer V: swept up and out towards the temple
        self._blob(p, (w * 0.56, -h * 0.62), w * 0.34, h * 0.62, -32, [
            (0.0, QColor(10, 3, 14, 220)), (0.55, QColor(26, 10, 36, 130)),
            (1.0, QColor(26, 10, 36, 0))])
        # crease definition
        crease = QPainterPath(self._p((-w * 0.46, -h * 0.30)))
        crease.cubicTo(self._p((-w * 0.18, -h * 1.40)), self._p((w * 0.34, -h * 1.36)), self._p((w * 0.66, -h * 0.70)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(24, 8, 32, 60), h * 0.42, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(crease)
        # lower lash line smudge (outer two-thirds, winged out)
        self._blob(p, (w * 0.18, h * 0.62), w * 0.44, h * 0.30, -6, [
            (0.0, QColor(16, 6, 22, 170)), (0.6, QColor(30, 12, 42, 80)), (1.0, QColor(30, 12, 42, 0))])
        # a whisper of violet shimmer on the centre of the lid (keeps it sexy, not sooty)
        self._blob(p, (w * 0.04, -h * 0.55), w * 0.20, h * 0.22, 0, [
            (0.0, QColor(150, 90, 200, 46)), (1.0, QColor(150, 90, 200, 0))])

    def _surround(self, p: QPainter):
        self._smoke(p)

    # ------------------------------------------------------------ layers
    def _paint_under(self, p: QPainter):
        w, h = self.w, self.h
        self._surround(p)
        # glossy black eyeball: deep plum-black, a touch lighter low in the eye
        a = self.almond()
        g = QLinearGradient(self._p((0, -h * 0.6)), self._p((0, h * 0.7)))
        g.setColorAt(0.0, QColor(2, 1, 4))
        g.setColorAt(0.55, QColor(10, 5, 16))
        g.setColorAt(1.0, QColor(26, 14, 38))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawPath(a)
        p.save()
        p.setClipPath(a)
        # wet sheen: a soft violet reflection along the lower rim
        lo = QPainterPath()
        for i in range(17):
            x, y = self.lower_point(0.12 + 0.76 * i / 16)
            pt = self._p((x, y - h * 0.10))
            lo.moveTo(pt) if i == 0 else lo.lineTo(pt)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(120, 70, 170, 70), h * 0.16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(lo)
        # iris: a barely-there darker disc with a faint violet limbal sheen
        ic = self._p(self.iris_c)
        ig = QRadialGradient(ic, self.iris_r)
        ig.setColorAt(0.0, QColor(0, 0, 0, 255))
        ig.setColorAt(0.80, QColor(4, 1, 8, 255))
        ig.setColorAt(0.93, QColor(60, 26, 96, 120))
        ig.setColorAt(1.0, QColor(60, 26, 96, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ig)
        p.drawEllipse(ic, self.iris_r, self.iris_r)
        p.restore()
        # tear duct (dusky)
        p.setBrush(QColor(120, 60, 90, 170))
        p.drawEllipse(self._p((self.inner[0] + w * 0.04, self.inner[1])), w * 0.03, h * 0.07)

    def _paint_over(self, p: QPainter, o: float):
        """Upper lid (skin + smoke), lid shadow, liner + wing, lashes."""
        c1, c2 = self.lid_ctrl(o)
        lid = QPainterPath(self._p(self.inner))
        lid.cubicTo(self._p(self.up_c[0]), self._p(self.up_c[1]), self._p(self.outer))
        lid.cubicTo(self._p(c2), self._p(c1), self._p(self.inner))
        lid.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.skin)
        p.drawPath(lid)
        p.save()                       # hairline overlap only inside the almond
        p.setClipPath(self.almond())
        p.setPen(QPen(self.skin, 0.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(lid)
        p.restore()
        p.save()
        p.setClipPath(lid)
        self._surround(p)
        # extra depth on the lid itself
        self._blob(p, (self.w * 0.10, (self.lid_point(o, 0.55)[1] - self.h * 0.20)), self.w * 0.55,
                   self.h * 0.45, -10, [(0.0, QColor(10, 3, 14, 150)), (1.0, QColor(10, 3, 14, 0))])
        p.restore()
        self._paint_lashes(p, o)

    def _paint_lashes(self, p: QPainter, o: float):
        w, h = self.w, self.h
        # shadow the lid casts onto the eyeball
        p.save()
        p.setClipPath(self.almond())
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k in range(3):
            off = 0.4 + k * 0.5
            pts = [self.lid_point(o, t / 16) for t in range(17)]
            sh = QPainterPath(self._p((pts[0][0], pts[0][1] + off)))
            for x, y in pts[1:]:
                sh.lineTo(self._p((x, y + off)))
            p.setPen(QPen(QColor(0, 0, 0, 120 - k * 35), 0.7))
            p.drawPath(sh)
        p.restore()
        # liner: thick, tapered along the lid into a long SHARP wing
        N = 24
        top, bot = [], []
        for i in range(N + 1):
            t = i / N
            x, y = self.lid_point(o, t)
            th = (0.45 + 1.55 * t ** 1.2) * h * 0.11 * (0.55 + 0.45 * o)
            top.append((x, y - th))
            bot.append((x, y + 0.2))
        ox, oy = self.outer
        wing_tip = (ox + w * 0.34, oy - h * (0.50 + 0.22 * o))
        liner = QPainterPath(self._p(bot[0]))
        for pt in top:
            liner.lineTo(self._p(pt))
        liner.lineTo(self._p(wing_tip))                                  # crisp straight wing
        liner.lineTo(self._p((ox + w * 0.05, oy + h * 0.10)))
        for pt in reversed(bot):
            liner.lineTo(self._p(pt))
        liner.closeSubpath()
        p.setPen(QPen(QColor(6, 2, 6, 230), 0.25))
        p.setBrush(QColor(8, 3, 8))
        p.drawPath(liner)
        # thick, dark, clumpy lashes: dense, longest at the outer corner
        n = 17
        rng = random.Random(3)
        for i in range(n):
            t = 0.16 + 0.84 * i / (n - 1)
            x, y = self.lid_point(o, t)
            L = h * (0.24 + 0.58 * t ** 1.35) * (0.62 + 0.38 * o) * rng.uniform(0.9, 1.08)
            up = (0.85 - 0.35 * t) * (2 * o - 1 if o < 0.5 else o)   # closed: lashes point down
            out = 0.22 + 0.85 * t
            base = (x, y - h * 0.06)
            ctrl = (x + L * out * 0.06, y - L * up * 0.72)
            tip = (x + L * out, y - L * up * 1.08)
            pts = [_quad(base, ctrl, tip, k / 6) for k in range(7)]
            p.setBrush(Qt.BrushStyle.NoBrush)
            for k in range(6):
                wdt = (h * 0.080 + 0.22) * (1 - k / 7.5)
                p.setPen(QPen(QColor(6, 2, 6, 240), wdt, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(self._p(pts[k]), self._p(pts[k + 1]))
        # lower liner: smudged, outer two-thirds, meeting the wing
        lo = QPainterPath()
        for i in range(17):
            t = 0.30 + 0.70 * i / 16
            x, y = self.lower_point(t)
            pt = self._p((x, y + 0.25))
            lo.moveTo(pt) if i == 0 else lo.lineTo(pt)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(14, 5, 18, 120), h * 0.20, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(lo)
        p.setPen(QPen(QColor(6, 2, 8, 200), h * 0.07, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(lo)
        # a few short lower lashes (outer half), flicking down and out
        for i in range(6):
            t = 0.50 + 0.46 * i / 5
            x, y = self.lower_point(t)
            L = h * (0.09 + 0.08 * (i / 5))
            base = (x, y + 0.3)
            tip = (x + L * (0.35 + 0.5 * i / 5), y + L * 0.95)
            p.setPen(QPen(QColor(8, 3, 10, 150), h * 0.04 + 0.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(self._p(base), self._p(tip))

    def _render(self, fn, *args) -> QImage:
        img = QImage(self.CW * SS, self.CH * SS, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(SS, SS)
        fn(p, *args)
        p.end()
        return img.scaled(self.CW, self.CH, Qt.AspectRatioMode.IgnoreAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    def _paint_inverted(self, p: QPainter):
        """Whole eye glowing violet (for the invert flicker)."""
        w, h = self.w, self.h
        ic = self._p(self.iris_c)
        g = QRadialGradient(ic, w * 0.55)
        g.setColorAt(0.0, QColor(246, 222, 255))
        g.setColorAt(0.35, QColor(208, 140, 255))
        g.setColorAt(0.75, QColor(150, 70, 240))
        g.setColorAt(1.0, QColor(84, 30, 170))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawPath(self.almond())

    def _build(self):
        mirror = QTransform().scale(-1, 1)
        under = self._render(self._paint_under)
        mask = self._render(lambda p: (p.setPen(Qt.PenStyle.NoPen), p.setBrush(QColor(0, 0, 0)),
                                       p.drawPath(self.almond())))
        inv = self._render(self._paint_inverted)
        self.under = {+1: QPixmap.fromImage(under), -1: QPixmap.fromImage(under.transformed(mirror))}
        self.inv = {+1: QPixmap.fromImage(inv), -1: QPixmap.fromImage(inv.transformed(mirror))}
        self.mask = {+1: mask, -1: mask.transformed(mirror)}
        self.over = {+1: [], -1: []}
        for k in range(LEVELS):
            o = k / (LEVELS - 1)
            img = self._render(self._paint_over, o)
            self.over[+1].append(QPixmap.fromImage(img))
            self.over[-1].append(QPixmap.fromImage(img.transformed(mirror)))
        # where the eyeball is visible for each lid level (almond minus lid)
        self.lid_alpha = {+1: [], -1: []}
        for k in range(LEVELS):
            o = k / (LEVELS - 1)

            def vis(p, o=o):
                c1, c2 = self.lid_ctrl(o)
                lid = QPainterPath(self._p(self.inner))
                lid.cubicTo(self._p(self.up_c[0]), self._p(self.up_c[1]), self._p(self.outer))
                lid.cubicTo(self._p(c2), self._p(c1), self._p(self.inner))
                lid.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0))
                p.drawPath(self.almond().subtracted(lid))
            img = self._render(vis)
            self.lid_alpha[+1].append(QPixmap.fromImage(img))
            self.lid_alpha[-1].append(QPixmap.fromImage(img.transformed(mirror)))
        hs = self.heart_size
        self.hearts = [QPixmap.fromImage(_heart_sprite(hs, k / (HEART_LEVELS - 1), self.HEART_FILL, None))
                       for k in range(HEART_LEVELS)]
        self.heart_hot = QPixmap.fromImage(_heart_sprite(hs, 1.0, self.HEART_HOT, None))
        self.heart_black = QPixmap.fromImage(_heart_sprite(hs, 1.0, QColor(6, 2, 10), None))
        self.glow_pad = max(3, int(round(hs * 0.45)))
        self.heart_glow = QPixmap.fromImage(_glow_sprite(hs, self.glow_pad, self.GLOW))
        self.scratch = QImage(self.CW, self.CH, QImage.Format.Format_ARGB32_Premultiplied)
        self._last = {}

    # ------------------------------------------------------------ per frame
    def heart_origin(self, side: int, gaze: tuple) -> tuple[int, int]:
        """Top-left of the heart sprite in eye-canvas pixels (integer: crisp)."""
        ix = self.cx + side * self.iris_c[0] + gaze[0]
        iy = self.cy + self.iris_c[1] + gaze[1]
        hs = self.heart_size
        return int(round(ix - hs / 2)), int(round(iy - hs / 2))

    def draw(self, p: QPainter, ex: float, ey: float, side: int, openness: float,
             gaze: tuple, bass: float, tint: tuple, t: float = 0.0,
             heart: float = 0.0, inv: float = 0.0):
        """side=+1: outer corner to the right (viewer's right eye).
        heart: 0 = outline heart .. 1 = full heart; inv: 0..1 inverted look."""
        x0 = int(round(ex - self.cx))
        y0 = int(round(ey - self.cy))
        p.drawPixmap(x0, y0, self.under[side])
        s = self.scratch
        s.fill(0)
        q = QPainter(s)
        hx, hy = self.heart_origin(side, gaze)
        hs = self.heart_size
        m = max(0.0, min(1.0, heart))
        if inv > 0.0:
            q.setOpacity(inv)
            q.drawPixmap(0, 0, self.inv[side])
            q.setOpacity(1.0)
        # heart pupil (blend of the two nearest morph levels -> smooth)
        f = m * (HEART_LEVELS - 1)
        k0 = int(f)
        k1 = min(HEART_LEVELS - 1, k0 + 1)
        fr = f - k0
        if inv < 1.0:
            # bloom first (additive), stronger when loud
            gp = self.glow_pad
            q.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            q.setOpacity((0.30 + 0.70 * m) * (1 - inv))
            q.drawPixmap(hx - gp, hy - gp, self.heart_glow)
            q.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            q.setOpacity(1 - inv)
            q.drawPixmap(hx, hy, self.hearts[k0])
            if fr > 0.01 and k1 != k0:
                q.setOpacity((1 - inv) * fr)
                q.drawPixmap(hx, hy, self.hearts[k1])
            hot = max(0.0, (m - 0.6) / 0.4)
            if hot > 0.01:                          # loud: the full heart runs white-hot
                q.setOpacity((1 - inv) * hot * 0.30)
                q.drawPixmap(hx, hy, self.heart_hot)
        if inv > 0.0:
            q.setOpacity(inv)
            q.drawPixmap(hx, hy, self.heart_black)
        q.setOpacity(1.0)
        # glossy catchlights (always: they make the black eyes look wet + alive)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.setPen(Qt.PenStyle.NoPen)
        r = self.iris_r
        cxl = self.cx + side * self.iris_c[0] + gaze[0] * 0.5
        cyl = self.cy + self.iris_c[1] + gaze[1] * 0.5
        q.setBrush(QColor(255, 250, 255, 200))
        q.drawEllipse(QPointF(cxl - side * r * 1.05, cyl - r * 0.18), r * 0.11, r * 0.09)
        q.setBrush(QColor(236, 220, 255, 70))
        q.drawEllipse(QPointF(cxl + side * r * 1.0, cyl + r * 0.30), r * 0.05, r * 0.045)
        sh = QLinearGradient(QPointF(0, cyl - r * 1.1), QPointF(0, cyl - r * 0.2))
        sh.setColorAt(0.0, QColor(255, 255, 255, 26))
        sh.setColorAt(1.0, QColor(255, 255, 255, 0))
        q.setBrush(sh)
        q.drawEllipse(QPointF(cxl - side * r * 0.15, cyl - r * 0.55), r * 1.05, r * 0.45)
        q.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        q.drawImage(0, 0, self.mask[side])
        q.end()
        p.drawImage(x0, y0, s)
        self._last[side] = (x0, y0, hx, hy, m, inv, openness)
        lvl = max(0, min(LEVELS - 1, int(round(openness * (LEVELS - 1)))))
        p.drawPixmap(x0, y0, self.over[side][lvl])


    def draw_light(self, p: QPainter, side: int):
        """Second, additive pass AFTER the visor glass: the heart's own light
        shines through the tinted glass (so it glows instead of looking
        printed on). Uses the geometry of the last draw() of this eye."""
        st = self._last.get(side)
        if st is None:
            return
        x0, y0, hx, hy, m, inv, o = st
        vis = max(0.0, min(1.0, (o - 0.15) / 0.5))     # hidden behind closed lids
        a = vis * (1 - inv)
        if a <= 0.01:
            return
        s = self.scratch
        s.fill(0)
        q = QPainter(s)
        gp = self.glow_pad
        q.setOpacity(a * (0.35 + 0.65 * m))
        q.drawPixmap(hx - gp, hy - gp, self.heart_glow)
        f = m * (HEART_LEVELS - 1)
        q.setOpacity(a * 0.38)
        q.drawPixmap(hx, hy, self.hearts[int(round(f))])
        q.setOpacity(1.0)
        q.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        q.drawImage(0, 0, self.mask[side])
        q.drawPixmap(0, 0, self.lid_alpha[side][max(0, min(LEVELS - 1, int(round(o * (LEVELS - 1)))))])
        q.end()
        p.save()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.drawImage(x0, y0, s)
        p.restore()


class EyeLife:
    """Slow blinks + small gaze drifts (cheap state machine), plus the eye
    reactions (slow blink, wink), boot flutter (force) and sleepy lids."""

    WINK_EYE = 1          # viewer's right eye (her left)
    REACT = {"slow_blink": (0.45, 0.32, 0.6), "wink": (0.2, 0.32, 0.45)}

    def __init__(self, seed=11):
        self.rng = random.Random(seed)
        self.next_blink = 3.0
        self.blink_t = None
        self.gaze = [0.0, 0.0]
        self.target = [0.0, 0.0]
        self.next_glance = 2.5
        self.t = 0.0
        self.react_kind = None
        self.react_t = 0.0
        self.force = None            # boot: explicit openness for both eyes
        self.sleep_k = 0.0           # 1 = asleep (lids nearly shut)
        self.flick_t = None          # invert flicker clock (None = not flickering)

    # invert flicker: (duration s, inverted?) - a nervous neon stutter, a
    # short hold fully inverted, then a stutter back
    FLICKER = ((0.05, 1), (0.06, 0), (0.04, 1), (0.09, 0), (0.05, 1), (0.04, 0),
               (0.50, 1), (0.05, 0), (0.04, 1), (0.07, 0), (0.03, 1))

    def flicker(self):
        """Start the invert flicker sequence (whole eye violet, black heart)."""
        self.flick_t = 0.0

    @property
    def flickering(self) -> bool:
        return self.flick_t is not None

    @property
    def inverted(self) -> float:
        """0..1: how inverted the eyes look right now."""
        if self.flick_t is None:
            return 0.0
        e = self.flick_t
        for d, v in self.FLICKER:
            if e < d:
                return float(v)
            e -= d
        return 0.0

    @classmethod
    def flicker_duration(cls) -> float:
        return sum(d for d, _ in cls.FLICKER)

    def react(self, kind: str):
        if kind in self.REACT:
            self.react_kind = kind
            self.react_t = 0.0
            self.blink_t = None

    @property
    def reacting(self) -> bool:
        return self.react_kind is not None

    def step(self, dt: float):
        self.t += dt
        if self.flick_t is not None:
            self.flick_t += dt
            if self.flick_t >= self.flicker_duration():
                self.flick_t = None
        if self.react_kind is not None:
            self.react_t += dt
            if self.react_t >= sum(self.REACT[self.react_kind]):
                self.react_kind = None
                self.next_blink = self.t + self.rng.uniform(2.5, 6.0)
        quiet = self.react_kind is not None or self.force is not None or self.sleep_k > 0.5
        if self.blink_t is None and self.t >= self.next_blink and not quiet:
            self.blink_t = 0.0
        if self.blink_t is not None:
            self.blink_t += dt
            if self.blink_t > 0.42:
                self.blink_t = None
                self.next_blink = self.t + self.rng.uniform(3.5, 8.0)
        if self.t >= self.next_glance:
            self.target = [self.rng.uniform(-1.6, 1.6), self.rng.uniform(-0.4, 0.5)]
            if self.rng.random() < 0.4 or self.sleep_k > 0.5:
                self.target = [0.0, 0.0]
            self.next_glance = self.t + self.rng.uniform(1.8, 5.0)
        k = 1 - math.exp(-dt * 7.0)
        self.gaze[0] += (self.target[0] - self.gaze[0]) * k
        self.gaze[1] += (self.target[1] - self.gaze[1]) * k

    @property
    def openness(self) -> float:
        """1 = relaxed sultry half-lid, 0 = closed. Slow, soft blink."""
        if self.blink_t is None:
            return 1.0
        b = self.blink_t
        if b < 0.13:
            return 1 - b / 0.13
        if b < 0.19:
            return 0.0
        return min(1.0, (b - 0.19) / 0.23)

    def _react_env(self) -> float:
        a, h, b = self.REACT[self.react_kind]
        e = self.react_t
        if e < a:
            u = e / a
        elif e < a + h:
            u = 1.0
        else:
            u = max(0.0, 1 - (e - a - h) / b)
        return u * u * (3 - 2 * u)

    def open_for(self, i: int) -> float:
        """Openness of eye i (0 = viewer's left)."""
        if self.force is not None:
            o = self.force
        else:
            o = self.openness
            if self.react_kind == "slow_blink":
                o = min(o, 1 - self._react_env())
            elif self.react_kind == "wink":
                env = self._react_env()
                o = min(o, 1 - env) if i == self.WINK_EYE else min(o, 1 - 0.22 * env)
        if self.sleep_k > 0:
            o = min(o, 1 - 0.9 * self.sleep_k)
        return max(0.0, min(1.0, o))

    @property
    def blinking(self) -> bool:
        return self.blink_t is not None
