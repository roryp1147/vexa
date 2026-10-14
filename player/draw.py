"""Procedural drawing helpers: 7-segment LCD digits, a 3x5 pixel font,
transport icons, rivets. No fonts or image assets needed."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF, QRadialGradient

# ---------------------------------------------------------------- 7-segment
#   a
# f   b
#   g
# e   c
#   d
SEG = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg",
    "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g", " ": "",
    "A": "abcefg", "b": "cdefg", "C": "adef", "d": "bcdeg", "E": "adefg", "F": "aefg",
    "L": "def", "r": "eg", "P": "abefg", "o": "cdeg", "n": "ceg", "U": "bcdef",
}


def _seg_poly(name, x, y, w, h, t):
    """Hexagonal segment polygon for a digit cell at (x,y) of size w x h."""
    m = h / 2
    s = t / 2
    def hseg(cx0, cx1, cy):
        return [QPointF(cx0 + s, cy), QPointF(cx0 + t, cy - s), QPointF(cx1 - t, cy - s),
                QPointF(cx1 - s, cy), QPointF(cx1 - t, cy + s), QPointF(cx0 + t, cy + s)]
    def vseg(cx, cy0, cy1):
        return [QPointF(cx, cy0 + s), QPointF(cx + s, cy0 + t), QPointF(cx + s, cy1 - t),
                QPointF(cx, cy1 - s), QPointF(cx - s, cy1 - t), QPointF(cx - s, cy0 + t)]
    sk = w * 0.12     # italic skew
    pts = {
        "a": hseg(x + sk, x + w + sk, y + s),
        "g": hseg(x + sk / 2, x + w + sk / 2, y + m),
        "d": hseg(x, x + w, y + h - s),
        "f": vseg(x + s + sk * 0.75, y, y + m),
        "b": vseg(x + w - s + sk * 0.75, y, y + m),
        "e": vseg(x + s + sk * 0.25, y + m, y + h),
        "c": vseg(x + w - s + sk * 0.25, y + m, y + h),
    }[name]
    return QPolygonF(pts)


def seven_seg(p, x, y, h, text, on: QColor, off: QColor | None = None, glow=True):
    """Draw text with 7-seg glyphs; returns the advance width."""
    w = h * 0.52
    t = max(1.2, h * 0.14)
    adv = w + h * 0.22
    cx = x
    p.setPen(Qt.PenStyle.NoPen)
    for ch in text:
        if ch in ":.":
            r = t * 0.55
            p.setBrush(on)
            if ch == ":":
                p.drawEllipse(QPointF(cx + r + 0.5, y + h * 0.3), r, r)
                p.drawEllipse(QPointF(cx + r - 0.2, y + h * 0.72), r, r)
            else:
                p.drawEllipse(QPointF(cx + r, y + h - r), r, r)
            cx += t * 1.6 + 1
            continue
        lit = SEG.get(ch, "")
        for sname in "abcdefg":
            poly = _seg_poly(sname, cx, y, w, h, t)
            if sname in lit:
                if glow:
                    gc = QColor(on)
                    gc.setAlpha(60)
                    p.setPen(QPen(gc, t * 1.3))
                    p.setBrush(on)
                    p.drawPolygon(poly)
                    p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(on)
                p.drawPolygon(poly)
            elif off is not None:
                p.setBrush(off)
                p.drawPolygon(poly)
        cx += adv
    return cx - x


def seven_seg_width(h, text):
    w = h * 0.52
    t = max(1.2, h * 0.14)
    adv = w + h * 0.22
    return sum((t * 1.6 + 1) if c in ":." else adv for c in text)


# ---------------------------------------------------------------- 3x5 pixel font
_F = {
    "A": "010101111101101", "B": "110101110101110", "C": "011100100100011", "D": "110101101101110",
    "E": "111100110100111", "F": "111100110100100", "G": "011100101101011", "H": "101101111101101",
    "I": "111010010010111", "J": "001001001101010", "K": "101101110101101", "L": "100100100100111",
    "M": "101111111101101", "N": "110101101101101", "O": "010101101101010", "P": "110101110100100",
    "Q": "010101101110011", "R": "110101110101101", "S": "011100010001110", "T": "111010010010010",
    "U": "101101101101111", "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
    "Y": "101101010010010", "Z": "111001010100111",
    "0": "111101101101111", "1": "010110010010111", "2": "110001010100111", "3": "110001010001110",
    "4": "101101111001001", "5": "111100110001110", "6": "011100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001110", "-": "000000111000000", "+": "000010111010000",
    ":": "000010000010000", ".": "000000000000010", "/": "001001010100100", "%": "101001010100101",
    " ": "000000000000000", ">": "100010001010100", "<": "001010100010001",
}


def pixel_text(p, x, y, text, color: QColor, px=1.0):
    """Tiny 3x5 pixel font (uppercase). Returns width."""
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    cx = x
    for ch in text.upper():
        bits = _F.get(ch, _F[" "])
        for i, b in enumerate(bits):
            if b == "1":
                p.drawRect(QRectF(cx + (i % 3) * px, y + (i // 3) * px, px, px))
        cx += 4 * px
    return cx - x


def pixel_text_width(text, px=1.0):
    return len(text) * 4 * px - px


# ---------------------------------------------------------------- icons
def icon_path(name: str, cx: float, cy: float, s: float, state=None) -> QPainterPath:
    """Transport icon centred at (cx,cy), nominal size s (px)."""
    path = QPainterPath()
    h = s / 2

    def tri(x0, dir_=1, w=None):
        w = w or s * 0.5
        pts = [QPointF(x0, cy - h * 0.8), QPointF(x0 + dir_ * w, cy), QPointF(x0, cy + h * 0.8)]
        path.addPolygon(QPolygonF(pts))
        path.closeSubpath()

    if name == "play":
        if state:   # playing -> show pause bars
            path.addRect(QRectF(cx - h * 0.75, cy - h * 0.8, s * 0.28, s * 0.8))
            path.addRect(QRectF(cx + h * 0.19, cy - h * 0.8, s * 0.28, s * 0.8))
        else:
            tri(cx - h * 0.55, 1, s * 0.75)
    elif name == "stop":
        path.addRect(QRectF(cx - h * 0.7, cy - h * 0.7, s * 0.7, s * 0.7))
    elif name == "next":
        tri(cx - h * 0.85, 1, s * 0.42)
        tri(cx - h * 0.05, 1, s * 0.42)
        path.addRect(QRectF(cx + h * 0.78, cy - h * 0.8, s * 0.12, s * 0.8))
    elif name == "prev":
        tri(cx + h * 0.85, -1, s * 0.42)
        tri(cx + h * 0.05, -1, s * 0.42)
        path.addRect(QRectF(cx - h * 0.9, cy - h * 0.8, s * 0.12, s * 0.8))
    elif name == "shuffle":
        # two crossing lanes with arrowheads
        t = s * 0.11
        a = QPainterPath()
        a.moveTo(cx - h, cy - h * 0.55)
        a.cubicTo(cx - h * 0.1, cy - h * 0.55, cx + h * 0.1, cy + h * 0.55, cx + h * 0.6, cy + h * 0.55)
        b = QPainterPath()
        b.moveTo(cx - h, cy + h * 0.55)
        b.cubicTo(cx - h * 0.1, cy + h * 0.55, cx + h * 0.1, cy - h * 0.55, cx + h * 0.6, cy - h * 0.55)
        from PySide6.QtGui import QPainterPathStroker
        st = QPainterPathStroker()
        st.setWidth(t * 1.6)
        st.setCapStyle(Qt.PenCapStyle.FlatCap)
        path = st.createStroke(a).united(st.createStroke(b))
        for yy in (-h * 0.55, h * 0.55):
            path.addPolygon(QPolygonF([QPointF(cx + h * 0.5, cy + yy - h * 0.38),
                                       QPointF(cx + h * 1.0, cy + yy),
                                       QPointF(cx + h * 0.5, cy + yy + h * 0.38)]))
            path.closeSubpath()
        path = path.simplified()
    elif name == "repeat":
        from PySide6.QtGui import QPainterPathStroker
        rr = QRectF(cx - h * 0.9, cy - h * 0.55, s * 0.9, s * 0.55)
        loop = QPainterPath()
        loop.addRoundedRect(rr, s * 0.22, s * 0.22)
        st = QPainterPathStroker()
        st.setWidth(s * 0.15)
        path = st.createStroke(loop)
        gap = QPainterPath()
        gap.addRect(QRectF(cx - s * 0.02, rr.top() - s * 0.2, s * 0.26, s * 0.4))
        gap.addRect(QRectF(cx - s * 0.24, rr.bottom() - s * 0.2, s * 0.26, s * 0.4))
        path = path.subtracted(gap)
        a = s * 0.21
        path.addPolygon(QPolygonF([QPointF(cx - s * 0.04, rr.top() - a), QPointF(cx + s * 0.24, rr.top()),
                                   QPointF(cx - s * 0.04, rr.top() + a)]))
        path.closeSubpath()
        path.addPolygon(QPolygonF([QPointF(cx + s * 0.04, rr.bottom() - a), QPointF(cx - s * 0.24, rr.bottom()),
                                   QPointF(cx + s * 0.04, rr.bottom() + a)]))
        path.closeSubpath()
        path = path.simplified()
    return path


def rivet(p, x, y, r=1.6):
    g = QRadialGradient(QPointF(x - r * 0.4, y - r * 0.4), r * 1.6)
    g.setColorAt(0, QColor(200, 204, 212))
    g.setColorAt(0.5, QColor(110, 114, 122))
    g.setColorAt(1, QColor(30, 31, 36))
    p.setPen(QPen(QColor(10, 10, 12, 200), 0.6))
    p.setBrush(QBrush(g))
    p.drawEllipse(QPointF(x, y), r, r)


def glow_color(c: QColor, a: int) -> QColor:
    g = QColor(c)
    g.setAlpha(a)
    return g
