#!/usr/bin/env python3
"""Windows-only sanity check for the SMTC bridge (no GUI, no sound).

    python tools/smtc_probe.py            # print what Vexa sees for 20 s
    python tools/smtc_probe.py --toggle   # also send play/pause once after 5 s

Start Spotify / YouTube Music in Chrome first. Every line printed is one
NowPlaying update exactly as the player receives it (source app, status,
title/artist/album, extrapolated position, art size, shuffle/repeat, controls).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QCoreApplication, QTimer  # noqa: E402

from player import tags  # noqa: E402
from player.platform_win import SmtcProvider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--toggle", action="store_true")
    a = ap.parse_args()
    app = QCoreApplication(sys.argv)
    prov = SmtcProvider()
    if prov.error:
        print("SMTC unavailable:", prov.error)
        return 1
    last = {"np": None}

    def on_update(np):
        last["np"] = np
        if np is None:
            print(f"{time.strftime('%H:%M:%S')}  (no media session)")
            return
        img = tags.art_image(np.art_key) if np.art_key else None
        art = f"{img.width()}x{img.height()}" if img is not None and not img.isNull() else "none"
        print(f"{time.strftime('%H:%M:%S')}  [{np.app}] {np.status:8s} {np.artist!r} - {np.title!r}"
              f"  album={np.album!r}  {np.position_ms/1000:.1f}/{np.duration_ms/1000:.1f}s"
              f"  art={art}  shuffle={np.shuffle} repeat={np.repeat}  controls={np.controls}")

    def tick():
        np = last["np"]
        if np is not None and np.playing:
            print(f"          extrapolated position {np.pos_at(time.monotonic())/1000:.1f}s")

    prov.updated.connect(on_update)
    t = QTimer()
    t.timeout.connect(tick)
    t.start(3000)
    if a.toggle:
        QTimer.singleShot(5000, lambda: print("-> play_pause:", prov.command("play_pause")))
    QTimer.singleShot(int(a.seconds * 1000), app.quit)
    rc = app.exec()
    prov.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
