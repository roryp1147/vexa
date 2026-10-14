#!/usr/bin/env python3
"""
Vexa - entry point.

    python main.py [audio files...] [--source auto|local|loopback|synth|none]
                   [--on-top] [--chips] [--no-boot]

Env:
    VEXA_MUTE=1          never produce sound (UI sounds + music), for testing
    VEXA_SOURCE=synth    default visual source (same values as --source)
    VEXA_NO_SETTINGS=1   don't read/write QSettings (window pos, mode, volume)
    VEXA_NO_SMTC=1       don't touch the Windows media session (SMTC)
    VEXA_NO_BOOT=1       skip the power-on boot sequence (same as --no-boot)
    VEXA_DPI_SCALE=1     let Qt scale the skin with Windows display scaling
                         (default: off -> the skin is drawn 1:1 in pixels,
                         crisp, like a classic Winamp skin)
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vexa retro android media player")
    ap.add_argument("files", nargs="*", help="audio files to queue")
    ap.add_argument("--source", default=None, help="auto | local | loopback | synth | none")
    ap.add_argument("--on-top", action="store_true", help="start always-on-top")
    ap.add_argument("--chips", action="store_true", help="eject both ear chips at startup")
    ap.add_argument("--no-boot", action="store_true", help="skip the power-on boot sequence")
    ap.add_argument("--quit-after", type=float, default=0, help="auto-exit after N seconds (testing)")
    args = ap.parse_args(argv)

    if os.environ.get("VEXA_DPI_SCALE", "0") != "1":
        # fixed-pixel skin: don't let 125%/150% display scaling blur/upsize it
        os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from player import config
    from player.platform_win import set_app_user_model_id

    set_app_user_model_id()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Vexa")
    app.setOrganizationName("Vexa")
    app.setQuitOnLastWindowClosed(True)
    icon = os.path.join(config.ASSETS, "vexa.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))

    from player.window import VexaWindow
    w = VexaWindow(source=args.source or config.SOURCE, files=args.files,
                   boot=False if args.no_boot else None)
    if args.on_top:
        w.set_on_top(True)
    w.show()
    if args.chips:
        QTimer.singleShot(600, lambda: (w.toggle_chip(w.chip_pl), w.toggle_chip(w.chip_mini)))
    if args.quit_after > 0:
        QTimer.singleShot(int(args.quit_after * 1000), w.close)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
