"""
Power-on sequence (~2.9 s, click to skip): she boots like an old machine.

  0.00  main relay CLUNK, transformer hum; the body flickers like a
        fluorescent tube catching, then slowly comes up to full brightness
  0.45  vein banks light one after another, head -> chest (relay ticks)
  1.05  visor CRT warm-up: a dot stretches into a line, the line opens
        into the picture with a bright flash (degauss thoonk)
  1.30  chest tray clunks out for the LCD segment test (88:88, blank, live)
  1.55  crunchy modem handshake chirp
  2.75  final latch: ready

Pure timeline maths; VexaWindow asks it what to draw and which sounds to fire.
"""
from __future__ import annotations

import math

DUR = 2.9
N_BANDS = 7
BAND_T0 = 0.45
BAND_DT = 0.12
CRT_T0 = 1.05
LINE_DUR = 0.15
OPEN_DUR = 0.3
TRAY_T = 1.30
LCD_ALL = (1.40, 1.80)
LCD_BLANK = (1.80, 1.95)
MODEM_T = 1.55

EVENTS = [
    (0.00, "sound", "power_relay"),
    (0.05, "sound", "boot_hum"),
    (CRT_T0, "sound", "crt_on"),
    (TRAY_T, "tray", None),
    (MODEM_T, "sound", "modem"),
    (2.75, "sound", "latch"),
] + [(BAND_T0 + i * BAND_DT, "sound", "relay_tick") for i in range(0, N_BANDS, 2)]
EVENTS.sort(key=lambda e: e[0])


def _ss(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


# fluorescent-tube style catch: (time, level) keyframes, held (step) until 0.45
_FLICKER = [(0.0, 0.0), (0.06, 0.32), (0.10, 0.02), (0.16, 0.45), (0.21, 0.06),
            (0.27, 0.36), (0.31, 0.16), (0.36, 0.3)]


class Boot:
    def __init__(self, t0: float):
        self.t0 = t0
        self.done = False
        self._ev = 0

    def e(self, now) -> float:
        return now - self.t0

    def finish(self):
        self.done = True

    def step(self, now):
        """-> list of (kind, arg) events due now; marks done at the end."""
        if self.done:
            return []
        e = self.e(now)
        out = []
        while self._ev < len(EVENTS) and EVENTS[self._ev][0] <= e:
            out.append(EVENTS[self._ev][1:])
            self._ev += 1
        if e >= DUR:
            self.done = True
        return out

    # ---------------------------------------------------------------- looks
    def power(self, now) -> float:
        """Body brightness 0..1."""
        if self.done:
            return 1.0
        e = self.e(now)
        if e < 0.45:
            lvl = 0.0
            for t, v in _FLICKER:
                if e >= t:
                    lvl = v
            return lvl
        return 0.3 + 0.7 * _ss(0.45, 2.5, e)

    def band(self, i: int, now, base: float) -> float:
        """Opacity of vein bank i (top-down): flicker on, flash, settle."""
        if self.done:
            return base
        e = self.e(now) - (BAND_T0 + i * BAND_DT)
        if e < 0:
            return 0.0
        if e < 0.045:
            return 0.85
        if e < 0.085:
            return 0.05
        if e < 0.19:
            return 1.0
        return base + (1.0 - base) * math.exp(-(e - 0.19) / 0.3)

    def visor(self, now):
        """(line 0..1, open 0..1, flash 0..1) for the CRT warm-up."""
        if self.done:
            return None
        e = self.e(now) - CRT_T0
        if e < 0:
            return (0.0, 0.0, 0.0)
        line = _ss(0.0, LINE_DUR, e)
        opened = _ss(LINE_DUR, LINE_DUR + OPEN_DUR, e) if e > LINE_DUR else 0.0
        flash = 0.0
        if e > LINE_DUR:
            flash = math.exp(-(e - LINE_DUR) / 0.22) * 0.9
            flash *= 0.85 + 0.15 * math.sin(e * 90)                # tube flicker
        if opened >= 1.0 and flash < 0.01:
            return None
        return (line, opened, flash)

    def eyes(self, now):
        """Forced eye openness (scan mode) or None: closed, flutter, open."""
        if self.done:
            return None
        e = self.e(now)
        if e < 1.5:
            return 0.0
        if e < 1.62:
            return _ss(1.5, 1.62, e) * 0.45
        if e < 1.74:
            return 0.45 * (1 - _ss(1.62, 1.74, e))
        if e < 2.05:
            return _ss(1.74, 2.05, e)
        return None

    def lcd(self, now):
        if self.done:
            return None
        e = self.e(now)
        if LCD_ALL[0] <= e < LCD_ALL[1]:
            return "all" if int(e * 20) % 7 else "blank"         # a stutter in the test
        if LCD_BLANK[0] <= e < LCD_BLANK[1]:
            return "blank"
        return None
