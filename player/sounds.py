"""UI sound bank: pre-rendered procedural WAVs played through QSoundEffect.

With VEXA_MUTE=1 no QSoundEffect is ever created (no audio device touched)."""
from __future__ import annotations

import os
import random

from PySide6.QtCore import QUrl

from . import config

# per-sound playback gain (servo is long + dense, keep it under the clacks)
GAINS = {"servo_out": 0.35, "servo_in": 0.3, "latch": 0.7, "shutter": 0.6,
         "btn_release": 0.45, "detent_0": 0.5, "detent_1": 0.5,
         "chip_slide_out": 0.32, "chip_slide_in": 0.3, "eject_spring": 0.7,
         "chip_latch": 0.7, "chip_seat": 0.65, "detach": 0.75, "dock": 0.75,
         # stage 3 (sparks / glitches are deliberately quiet)
         "power_relay": 0.7, "relay_tick_0": 0.4, "relay_tick_1": 0.4, "boot_hum": 0.3,
         "crt_on": 0.5, "modem": 0.32, "crackle_0": 0.28, "crackle_1": 0.28, "crackle_2": 0.28,
         "glitch": 0.42, "static": 0.3, "sleep": 0.35, "wake": 0.45}
GROUPS = {"press": ["btn_press_0", "btn_press_1", "btn_press_2"],
          "detent": ["detent_0", "detent_1"],
          "crackle": ["crackle_0", "crackle_1", "crackle_2"],
          "relay_tick": ["relay_tick_0", "relay_tick_1"]}


class SoundBank:
    def __init__(self, parent=None, enabled: bool = True):
        self.enabled = enabled and not config.MUTE
        self.muted = False
        self.master = 0.8
        self.fx = {}
        self.log: list[str] = []      # what would have played (tests)
        self._alt = 0
        if not self.enabled:
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
            from .sfx_gen import SOUNDS, generate_all
            missing = any(not os.path.exists(os.path.join(config.SOUNDS, n + ".wav")) for n in SOUNDS)
            if missing:
                generate_all(config.SOUNDS, overwrite=False)
            for name in SOUNDS:
                # two instances each so rapid repeats overlap instead of cutting
                pair = []
                for _ in range(2):
                    e = QSoundEffect(parent)
                    e.setSource(QUrl.fromLocalFile(os.path.join(config.SOUNDS, name + ".wav")))
                    e.setVolume(GAINS.get(name, 0.8) * self.master)
                    pair.append(e)
                self.fx[name] = pair
        except Exception as e:  # no multimedia / no device: stay silent
            print(f"[vexa] UI sounds disabled: {e}")
            self.fx = {}

    def play(self, name: str):
        if name in GROUPS:
            name = random.choice(GROUPS[name]) if name in ("press", "crackle") else \
                GROUPS[name][self._next_alt()]
        self.log.append(name)
        if len(self.log) > 200:
            del self.log[:100]
        if not self.enabled or self.muted:
            return
        pair = self.fx.get(name)
        if not pair:
            return
        e = pair[0] if not pair[0].isPlaying() else pair[1]
        e.play()

    def stop(self, name: str):
        for e in self.fx.get(name, ()):
            if e.isPlaying():
                e.stop()

    def _next_alt(self):
        self._alt ^= 1
        return self._alt

    def set_muted(self, m: bool):
        self.muted = m
