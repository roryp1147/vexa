"""
Local-file metadata + album art (mutagen), cached.

  * title / artist / album from ID3, Vorbis comments (FLAC/OGG/Opus), MP4 atoms
  * embedded art: MP3 APIC, FLAC PICTURE blocks, M4A 'covr',
    OGG/Opus METADATA_BLOCK_PICTURE (base64 FLAC picture)
  * fallback: cover.jpg / folder.jpg / front.jpg / album.jpg (+png) next to the file

Everything is read lazily (only for the current track and for list rows that
are actually shown) and cached. Art is decoded once, then each requested
thumbnail size is scaled once and cached (QImage; the GUI turns it into a
QPixmap on first paint).
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

try:
    import mutagen
    HAVE_MUTAGEN = True
except Exception:          # pragma: no cover - optional dependency
    mutagen = None
    HAVE_MUTAGEN = False

COVER_NAMES = ("cover", "folder", "front", "album", "albumart", "albumartsmall")
COVER_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class TrackInfo:
    path: str
    title: str
    artist: str = ""
    album: str = ""
    duration_ms: int = 0
    art_key: str | None = None       # hash of the art bytes (None = no art)


_info_cache: dict[str, TrackInfo] = {}
_art_bytes: dict[str, bytes] = {}         # art_key -> encoded image bytes
_art_img: dict[str, QImage] = {}          # art_key -> decoded full image
_thumbs: dict[tuple, QImage] = {}         # (art_key, w, h) -> scaled


def _first(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else ""
    return str(v).strip()


def _key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _embedded_art(f) -> bytes | None:
    """Return the best embedded picture's bytes (front cover preferred)."""
    tags = getattr(f, "tags", None)
    # FLAC: f.pictures
    pics = list(getattr(f, "pictures", []) or [])
    if pics:
        pics.sort(key=lambda p: (getattr(p, "type", 0) != 3,))
        return bytes(pics[0].data)
    if tags is None:
        return None
    # ID3 (MP3, also AIFF/WAV with ID3)
    try:
        apics = tags.getall("APIC") if hasattr(tags, "getall") else []
    except Exception:
        apics = []
    if apics:
        apics.sort(key=lambda p: (getattr(p, "type", 0) != 3,))
        return bytes(apics[0].data)
    # MP4
    try:
        covr = tags.get("covr") if hasattr(tags, "get") else None
        if covr:
            return bytes(covr[0])
    except Exception:
        pass
    # Ogg Vorbis / Opus: base64 FLAC picture block
    try:
        blocks = tags.get("metadata_block_picture") or tags.get("METADATA_BLOCK_PICTURE")
        if blocks:
            from mutagen.flac import Picture
            best = None
            for b in blocks:
                pic = Picture(base64.b64decode(b))
                if best is None or pic.type == 3:
                    best = pic
            return bytes(best.data) if best else None
    except Exception:
        pass
    return None


def _folder_art(path: str) -> bytes | None:
    d = os.path.dirname(path)
    try:
        names = {n.lower(): n for n in os.listdir(d)}
    except Exception:
        return None
    for base in COVER_NAMES:
        for ext in COVER_EXTS:
            n = names.get(base + ext)
            if n:
                try:
                    with open(os.path.join(d, n), "rb") as fh:
                        return fh.read()
                except Exception:
                    pass
    return None


def _text_tags(f):
    """(title, artist, album) from any mutagen file (easy or raw tags)."""
    tags = getattr(f, "tags", None)
    if tags is None:
        return "", "", ""
    title = artist = album = ""
    try:  # ID3 frames
        if hasattr(tags, "getall"):
            title = _first(tags.get("TIT2"))
            artist = _first(tags.get("TPE1")) or _first(tags.get("TPE2"))
            album = _first(tags.get("TALB"))
    except Exception:
        pass
    if not title:
        try:  # MP4 atoms
            title = _first(tags.get("\xa9nam"))
            artist = artist or _first(tags.get("\xa9ART")) or _first(tags.get("aART"))
            album = album or _first(tags.get("\xa9alb"))
        except Exception:
            pass
    if not title:
        try:  # Vorbis comments (case-insensitive)
            title = _first(tags.get("title"))
            artist = artist or _first(tags.get("artist")) or _first(tags.get("albumartist"))
            album = album or _first(tags.get("album"))
        except Exception:
            pass
    return title, artist, album


def track_info(path: str) -> TrackInfo:
    """Tags + art key for a file (cached; never raises)."""
    ti = _info_cache.get(path)
    if ti is not None:
        return ti
    stem = os.path.splitext(os.path.basename(path))[0]
    title = artist = album = ""
    dur = 0
    art = None
    if HAVE_MUTAGEN:
        try:
            f = mutagen.File(path)
            if f is not None:
                title, artist, album = _text_tags(f)
                if getattr(f, "info", None) is not None and getattr(f.info, "length", None):
                    dur = int(f.info.length * 1000)
                art = _embedded_art(f)
        except Exception:
            pass
    if not title:
        stem = re.sub(r"^\d{1,3}[\s._-]+", "", stem) or stem     # "05 Neon Rain" -> "Neon Rain"
        if " - " in stem and not artist:
            artist, stem = [s.strip() for s in stem.split(" - ", 1)]
    if art is None:
        art = _folder_art(path)
    key = None
    if art:
        key = _key(art)
        _art_bytes.setdefault(key, art)
    ti = TrackInfo(path, title or stem, artist, album, dur, key)
    _info_cache[path] = ti
    return ti


def register_art(data: bytes | None) -> str | None:
    """Register external art bytes (e.g. an SMTC thumbnail); returns its key."""
    if not data:
        return None
    key = _key(data)
    _art_bytes.setdefault(key, data)
    return key


def art_image(key: str | None) -> QImage | None:
    if not key:
        return None
    img = _art_img.get(key)
    if img is None:
        data = _art_bytes.get(key)
        if not data:
            return None
        img = QImage.fromData(data)
        if img.isNull():
            return None
        _art_img[key] = img
        _art_bytes.pop(key, None)          # decoded once; drop the compressed copy
        while len(_art_img) > 12:          # keep memory bounded (thumbs stay cached)
            old = next(iter(_art_img))
            if old == key:
                break
            _art_img.pop(old)
    return img


def thumbnail(key: str | None, w: int, h: int) -> QImage | None:
    """Center-cropped, smooth-scaled thumbnail, scaled exactly once per size."""
    if not key:
        return None
    k = (key, w, h)
    t = _thumbs.get(k)
    if t is None:
        img = art_image(key)
        if img is None:
            return None
        s = img.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                       Qt.TransformationMode.SmoothTransformation)
        x = (s.width() - w) // 2
        y = (s.height() - h) // 2
        t = s.copy(x, y, w, h).convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        _thumbs[k] = t
        if len(_thumbs) > 64:
            _thumbs.pop(next(iter(_thumbs)))
    return t


def clear():
    _info_cache.clear()
    _art_bytes.clear()
    _art_img.clear()
    _thumbs.clear()
