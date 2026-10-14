"""Small QImage <-> numpy helpers and colour utilities."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QImage, QPixmap


def qimage_to_rgba(img: QImage) -> np.ndarray:
    """QImage -> float32 HxWx4 RGBA (straight alpha), 0..1."""
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())
    return buf[:, : w * 4].reshape(h, w, 4).astype(np.float32) / 255.0


def qimage_to_gray(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())
    return buf[:, :w].astype(np.float32) / 255.0


def rgba_to_qimage(rgb: np.ndarray, alpha: np.ndarray) -> QImage:
    """float rgb (straight) + alpha -> premultiplied QImage (deep copy)."""
    a = np.clip(alpha, 0, 1)[..., None]
    pm = np.clip(rgb, 0, 1) * a
    arr = (np.dstack([pm, a]) * 255 + 0.5).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888_Premultiplied)
    return img.copy()  # detach from numpy memory


def premul_to_pixmap(rgb_premul: np.ndarray, alpha: np.ndarray) -> QPixmap:
    # a valid premultiplied pixel never has colour > alpha: invalid ones wrap
    # around / glow on a layered window over light wallpapers
    a = np.clip(alpha, 0, 1)
    pm = np.minimum(np.clip(rgb_premul, 0, 1), a[..., None])
    arr = (np.dstack([pm, a]) * 255 + 0.5).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888_Premultiplied).copy()
    return QPixmap.fromImage(img)


def qc(rgb, a=255) -> QColor:
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))


def mix(c1, c2, t):
    return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))


def scale(c, k):
    return tuple(max(0, min(255, c[i] * k)) for i in range(3))
