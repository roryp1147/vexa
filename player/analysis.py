"""
Cheap audio analysis for the visor + veins.

One 1024-sample Hann-windowed rfft per frame (<= 30/s): ~20 us of numpy.
Produces:
  bands  - N log-spaced band levels 0..1 (spectrum bars), attack/decay smoothed
  wave   - 128-point oscilloscope trace (rising zero-crossing triggered)
  bass   - 30-160 Hz energy envelope 0..1 (veins), fast attack / slow decay
  level  - overall loudness 0..1
All levels pass through slow auto-gain so quiet YouTube tabs and loud local
files both drive the visuals.
"""
from __future__ import annotations

import math

import numpy as np

N_FFT = 1024
N_BANDS = 28
WAVE_PTS = 128


class Analyzer:
    def __init__(self, n_bands: int = N_BANDS):
        self.n_bands = n_bands
        self.window = np.hanning(N_FFT).astype(np.float32)
        self._sr = None
        self._band_idx = None
        self._bass_sl = None
        self.bands = np.zeros(n_bands, np.float32)
        self.peaks = np.zeros(n_bands, np.float32)
        self.wave = np.zeros(WAVE_PTS, np.float32)
        self.bass = 0.0
        self.level = 0.0
        self.rms = 0.0
        self._gain_spec = 1e-3
        self._gain_bass = 1e-3
        self._gain_wave = 1e-3
        self.silent_for = 10.0     # seconds of silence so far

    # ------------------------------------------------------------------
    def _setup(self, sr: int):
        self._sr = sr
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
        edges = np.geomspace(45.0, min(16000.0, sr / 2 - 1), self.n_bands + 1)
        idx = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            a = int(np.searchsorted(freqs, lo))
            b = max(a + 1, int(np.searchsorted(freqs, hi)))
            idx.append((a, b))
        self._band_idx = idx
        # vectorised band means via cumulative sums
        self._band_a = np.array([a for a, _ in idx])
        self._band_b = np.array([b for _, b in idx])
        b0 = int(np.searchsorted(freqs, 30.0))
        b1 = int(np.searchsorted(freqs, 160.0))
        self._bass_sl = slice(max(1, b0), max(b0 + 1, b1))
        # tilt: hi bands carry less energy in music; boost them a bit
        self._tilt = np.linspace(0.0, 1.0, self.n_bands).astype(np.float32) * 24.0

    @staticmethod
    def _coef(per_second: float, dt: float) -> float:
        """Frame-rate independent smoothing coefficient."""
        return 1.0 - math.exp(-per_second * dt)

    def process(self, samples: np.ndarray | None, sr: int, dt: float):
        """samples: latest mono float32 block (len >= N_FFT) or None (silence)."""
        dt = max(1e-3, min(dt, 0.25))
        if samples is None or len(samples) < N_FFT:
            self._decay(dt)
            return
        if sr != self._sr:
            self._setup(sr)
        x = samples[-N_FFT:].astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(x * x)))
        self.rms = rms
        if rms < 2e-4:
            self.silent_for += dt
        else:
            self.silent_for = 0.0

        spec = np.abs(np.fft.rfft(x * self.window))
        cs = np.concatenate(([0.0], np.cumsum(spec)))
        bandmag = (cs[self._band_b] - cs[self._band_a]) / (self._band_b - self._band_a)
        db = 20.0 * np.log10(bandmag + 1e-9) + self._tilt
        # auto gain on the loudest band (slow release, fast attack)
        top = float(db.max())
        g = self._gain_spec
        g = top if top > g else g + (top - g) * self._coef(0.25, dt)
        self._gain_spec = max(g, -40.0)
        target = np.clip((db - (self._gain_spec - 42.0)) / 42.0, 0.0, 1.0) ** 1.3
        if rms < 2e-4:
            target[:] = 0
        up = self._coef(28.0, dt)
        down = self._coef(7.0, dt)
        self.bands += (target - self.bands) * np.where(target > self.bands, up, down)
        # peak caps: hold then fall
        self.peaks = np.maximum(self.peaks - dt * 0.55, self.bands)

        # bass envelope
        bass_e = float(np.sqrt(np.mean(spec[self._bass_sl] ** 2)))
        gb = self._gain_bass
        gb = bass_e if bass_e > gb else gb + (bass_e - gb) * self._coef(0.2, dt)
        self._gain_bass = max(gb, 1e-4)
        b = min(1.0, bass_e / (self._gain_bass + 1e-9))
        b = max(0.0, (b - 0.25) / 0.75) ** 1.5      # gate + curve -> punchy kicks
        if rms < 2e-4:
            b = 0.0
        k = self._coef(30.0, dt) if b > self.bass else self._coef(5.0, dt)
        self.bass += (b - self.bass) * k


        # oscilloscope: trigger on a rising zero crossing for a stable trace
        seg = x[-N_FFT:]
        half = len(seg) // 2
        zc = np.nonzero((seg[:half - 1] <= 0) & (seg[1:half] > 0))[0]
        start = int(zc[0]) if len(zc) else 0
        w = seg[start:start + half]
        if len(w) < half:
            w = seg[-half:]
        pts = w[:: max(1, len(w) // WAVE_PTS)][:WAVE_PTS]
        peak = float(np.abs(x).max())
        gw = self._gain_wave
        gw = peak if peak > gw else gw + (peak - gw) * self._coef(0.3, dt)
        self._gain_wave = max(gw, 0.02)
        self.wave[: len(pts)] = pts / self._gain_wave
        lvl = min(1.0, rms / (self._gain_wave * 0.35))
        self.level += (lvl - self.level) * self._coef(10.0, dt)

    def _decay(self, dt: float):
        self.silent_for += dt
        down = self._coef(6.0, dt)
        self.bands -= self.bands * down
        self.peaks = np.maximum(self.peaks - dt * 0.55, 0)
        self.bass -= self.bass * self._coef(4.0, dt)
        self.level -= self.level * down
        self.wave *= 1.0 - down
        self.rms = 0.0

    @property
    def silent(self) -> bool:
        return self.silent_for > 0.25
