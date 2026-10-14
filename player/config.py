"""Paths, environment switches and the visor-mode palette."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
LAYERS = os.path.join(ASSETS, "layers")
SOUNDS = os.path.join(ASSETS, "sounds")

IS_WINDOWS = sys.platform == "win32"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# VEXA_MUTE=1 -> never create any sound output (UI sounds + music muted).
MUTE = env_flag("VEXA_MUTE")
# VEXA_SOURCE = auto | local | loopback | synth | none
SOURCE = os.environ.get("VEXA_SOURCE", "auto").strip().lower()

FPS_ACTIVE = 30      # max animation rate
FPS_IDLE = 8         # breathing veins only, nothing playing
IDLE_AFTER_S = 1.5   # silence this long -> idle rate
PANEL_IDLE_S = 3.0   # chest row retracts after this much idle
FPS_SLEEP = 4        # asleep: dim visor + slow breathing veins
SLEEP_AFTER_S = 120.0  # nothing played for this long -> sleep mode
# VEXA_NO_BOOT=1 -> skip the power-on sequence
BOOT = not env_flag("VEXA_NO_BOOT")


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    primary: tuple      # main neon colour (veins, LEDs, LCD)
    secondary: tuple    # shadow side of the neon / gradient partner
    glass: tuple        # dark tint of the visor glass


MODES = [
    Mode("spectrum", "Spectrum", (255, 46, 170), (140, 56, 255), (26, 4, 26)),
    Mode("waveform", "Waveform", (52, 150, 255), (150, 78, 255), (4, 10, 30)),
    Mode("matrix", "Matrix rain", (96, 255, 72), (0, 160, 70), (2, 18, 6)),
    Mode("scan", "Scanning eye", (178, 120, 255), (92, 46, 196), (30, 14, 54)),  # ultraviolet / lavender (+ silver, see layers)
]
MODE_INDEX = {m.key: i for i, m in enumerate(MODES)}
