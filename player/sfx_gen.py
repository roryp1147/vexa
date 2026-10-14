"""
Procedural clunky-analog UI sounds (pure numpy -> 16-bit mono WAV).

Everything is built from physical-ish pieces instead of synth tones:
  * contact impulses  - very short, bright noise bursts (metal hitting metal)
  * modal ringing     - a few INHARMONIC damped sinusoids (stamped steel parts)
  * body thunk        - low damped sine (the plastic/metal housing)
  * contact bounce    - relays/latches hit 2-3 times within ~15 ms
  * motor whir        - pitch-ramping buzzy harmonic stack + gear whine + brush noise
then soft-saturated and low-passed so nothing sounds glassy.

Used by tools/build_sounds.py (pre-build into assets/sounds) and as a runtime
fallback if the WAVs are missing.
"""
from __future__ import annotations

import os
import wave

import numpy as np

SR = 44100


def _t(dur):
    return np.arange(int(SR * dur)) / SR


def _filter(x, lo=None, hi=None):
    """Zero-phase FFT band filter with soft (cosine) skirts. Pure numpy."""
    n = len(x)
    nf = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= np.clip((f - lo * 0.6) / (lo * 0.4), 0, 1)
    if hi:
        g *= np.clip((hi * 1.4 - f) / (hi * 0.4), 0, 1)
    return np.fft.irfft(X * g, nf)[:n]


def _impulse(rng, dur=0.004, tau=0.0007, lo=1800, hi=9000):
    t = _t(dur)
    n = rng.standard_normal(len(t)) * np.exp(-t / tau)
    return _filter(n, lo, hi)


def _modes(freqs, taus, amps, dur, rng, detune=0.01):
    t = _t(dur)
    out = np.zeros_like(t)
    for f, tau, a in zip(freqs, taus, amps):
        f = f * (1 + rng.uniform(-detune, detune))
        ph = rng.uniform(0, 2 * np.pi)
        out += a * np.sin(2 * np.pi * f * t + ph) * np.exp(-t / tau)
    return out


def _thunk(f0, tau, dur, drop=0.25):
    t = _t(dur)
    f = f0 * (1 - drop * (1 - np.exp(-t / 0.01)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / tau) * (1 - np.exp(-t / 0.0006))


def _place(buf, x, at, gain=1.0):
    i = int(at * SR)
    j = min(len(buf), i + len(x))
    if i < len(buf):
        buf[i:j] += gain * x[:j - i]


def _finish(x, drive=1.6, hi=7500, peak=0.85):
    x = _filter(x, 40, hi)
    x = np.tanh(drive * x / max(np.abs(x).max(), 1e-9)) / np.tanh(drive)
    # tiny fade in/out to avoid clicks at boundaries
    n = len(x)
    f = min(64, n // 8)
    x[:8] *= np.linspace(0, 1, 8)
    x[-f:] *= np.linspace(1, 0, f)
    return x / max(np.abs(x).max(), 1e-9) * peak


# --------------------------------------------------------------------------
def relay_clack(variant=0):
    """Button press: a stiff relay / tape-deck key clack with contact bounce."""
    rng = np.random.default_rng(100 + variant)
    dur = 0.13
    buf = np.zeros(int(SR * dur))
    base = [1180, 1370, 1010][variant % 3]
    ring = _modes([base, base * 2.31, base * 3.87, base * 5.42],
                  [0.016, 0.010, 0.006, 0.004], [1.0, 0.55, 0.35, 0.2], 0.09, rng)
    body = _thunk([190, 165, 215][variant % 3], 0.018, 0.08)
    hit = _impulse(rng)
    _place(buf, hit, 0.0, 1.0)
    _place(buf, ring, 0.0, 0.35)
    _place(buf, body, 0.0, 0.9)
    # contact bounce
    b1 = 0.006 + 0.002 * variant
    _place(buf, _impulse(rng, tau=0.0005), b1, 0.45)
    _place(buf, ring[: int(SR * 0.04)], b1, 0.12)
    _place(buf, _impulse(rng, tau=0.0004), b1 + 0.007, 0.15)
    return _finish(buf, drive=1.8, hi=6500)


def key_release():
    """Button release: smaller, duller tick as the cap springs back."""
    rng = np.random.default_rng(7)
    buf = np.zeros(int(SR * 0.06))
    _place(buf, _impulse(rng, tau=0.0005, lo=1200, hi=6000), 0, 0.6)
    _place(buf, _modes([900, 2100], [0.008, 0.004], [1, 0.4], 0.04, rng), 0, 0.25)
    _place(buf, _thunk(240, 0.01, 0.04), 0, 0.4)
    return _finish(buf, drive=1.3, hi=5500, peak=0.55)


def latch_clack():
    """Heavy latch locking home (end of the button-row slide)."""
    rng = np.random.default_rng(21)
    dur = 0.2
    buf = np.zeros(int(SR * dur))
    _place(buf, _impulse(rng, dur=0.006, tau=0.001, lo=900, hi=7000), 0, 1.0)
    _place(buf, _thunk(105, 0.035, 0.15, drop=0.3), 0, 1.1)
    _place(buf, _modes([620, 1490, 2630, 3910], [0.03, 0.018, 0.01, 0.006],
                       [0.8, 0.6, 0.35, 0.2], 0.15, rng), 0, 0.4)
    _place(buf, _impulse(rng, tau=0.0006), 0.011, 0.5)
    _place(buf, _thunk(140, 0.02, 0.06), 0.011, 0.35)
    _place(buf, _impulse(rng, tau=0.0004), 0.024, 0.18)
    return _finish(buf, drive=2.0, hi=6000)


def servo_whir(direction=1, dur=0.36):
    """Geared servo motor: buzzy harmonic stack ramping in pitch, gear whine,
    brush noise and a bit of mechanical rattle. direction=+1 out, -1 in."""
    rng = np.random.default_rng(31 if direction > 0 else 32)
    t = _t(dur)
    u = t / dur
    if direction > 0:
        f = 70 + 85 * np.clip(u * 2.2, 0, 1) - 12 * np.clip((u - 0.8) * 5, 0, 1)
    else:
        f = 150 - 30 * u
    ph = 2 * np.pi * np.cumsum(f) / SR
    motor = sum((0.9 ** k) / k * np.sin(k * ph + 0.3 * k) for k in range(1, 14))
    motor = np.sign(motor) * np.abs(motor) ** 0.7          # buzzier
    gear = 0.25 * np.sin(9 * ph) * (0.5 + 0.5 * np.sin(ph * 0.5))
    brush = _filter(rng.standard_normal(len(t)), 900, 3200) * (0.4 + 0.6 * np.abs(np.sin(ph)))
    env = np.clip(t / 0.025, 0, 1) * np.clip((dur - t) / 0.04, 0, 1)
    x = (0.55 * motor + gear + 0.35 * brush) * env
    # gear-tooth rattle: random tiny clicks
    for k in range(int(dur * 40)):
        at = rng.uniform(0.02, dur - 0.03)
        _place(x, _impulse(rng, dur=0.002, tau=0.0003), at, 0.25 * rng.uniform(0.3, 1))
    return _finish(x * 0.8, drive=1.4, hi=4800, peak=0.6)


def detent_tick(variant=0):
    """Dial detent: small ratchet tick."""
    rng = np.random.default_rng(50 + variant)
    buf = np.zeros(int(SR * 0.035))
    _place(buf, _impulse(rng, dur=0.002, tau=0.0003, lo=2000, hi=9000), 0, 1.0)
    _place(buf, _modes([2550 + 180 * variant, 4100, 930], [0.003, 0.002, 0.006],
                       [0.6, 0.3, 0.5], 0.03, rng), 0, 0.5)
    return _finish(buf, drive=1.5, hi=8000, peak=0.6)


def shutter_clack():
    """Visor shutter: spring-loaded blades snap shut (ka-) and reopen (-chunk)."""
    rng = np.random.default_rng(61)
    dur = 0.46
    buf = np.zeros(int(SR * dur))
    # trigger click
    _place(buf, _impulse(rng, tau=0.0005), 0.0, 0.35)
    # blades sliding: short metallic scrape
    n = int(SR * 0.14)
    sc = _filter(rng.standard_normal(n), 2500, 7000) * np.linspace(0.2, 1, n) ** 2 * 0.25
    _place(buf, sc, 0.02)
    # slam shut
    _place(buf, _impulse(rng, dur=0.006, tau=0.0009, lo=900), 0.16, 1.0)
    _place(buf, _thunk(130, 0.03, 0.12), 0.16, 0.9)
    _place(buf, _modes([780, 1830, 3050], [0.03, 0.015, 0.008], [1, 0.6, 0.3], 0.12, rng), 0.16, 0.45)
    # spring buzz while held
    tb = _t(0.07)
    _place(buf, 0.18 * np.sin(2 * np.pi * 310 * tb) * np.exp(-tb / 0.02) * (1 + np.sin(2 * np.pi * 45 * tb)), 0.168)
    # reopen + settle
    _place(buf, _impulse(rng, tau=0.0006), 0.24, 0.5)
    _place(buf, _thunk(170, 0.015, 0.06), 0.24, 0.4)
    _place(buf, _impulse(rng, tau=0.0005), 0.40, 0.55)
    _place(buf, _thunk(150, 0.02, 0.05), 0.40, 0.45)
    return _finish(buf, drive=1.7, hi=6500)


# ---------------------------------------------------------------- stage 2: chips
def _spring(f0, dur, tau=0.05, rng=None):
    """Coil-spring 'boing': inharmonic partials, falling pitch, wobble."""
    t = _t(dur)
    f = f0 * (1 + 0.35 * np.exp(-t / 0.018))
    ph = 2 * np.pi * np.cumsum(f) / SR
    wob = 1 + 0.35 * np.sin(2 * np.pi * 28 * t)
    x = (np.sin(ph) + 0.45 * np.sin(2.76 * ph + 1) + 0.2 * np.sin(5.4 * ph + 2)) * wob
    return x * np.exp(-t / tau) * (1 - np.exp(-t / 0.001))


def eject_spring():
    """Red eject button clack, then the chip springs loose (twang + pop)."""
    rng = np.random.default_rng(71)
    buf = np.zeros(int(SR * 0.32))
    _place(buf, relay_clack(1) * 0.8, 0.0)
    _place(buf, _impulse(rng, dur=0.005, tau=0.0008, lo=700, hi=6000), 0.06, 0.9)
    _place(buf, _thunk(150, 0.03, 0.12, drop=0.35), 0.06, 0.8)
    _place(buf, _spring(380, 0.22, 0.055), 0.065, 0.35)
    _place(buf, _modes([940, 2210, 3480], [0.02, 0.01, 0.006], [1, 0.5, 0.25], 0.1, rng), 0.06, 0.25)
    return _finish(buf, drive=1.7, hi=6500)


def chip_slide(direction=1, dur=0.6, hitches=(0.35, 0.7)):
    """Servo-driven chip carriage: geared whir that stalls and lurches at
    each hitch (with a gear-skip clunk), over a rail scrape."""
    rng = np.random.default_rng(81 if direction > 0 else 82)
    t = _t(dur)
    u = t / dur
    env = np.clip(t / 0.03, 0, 1) * np.clip((dur - t) / 0.03, 0, 1)
    slow = np.ones_like(t)
    for h in hitches:
        slow *= 1 - 0.8 * np.exp(-((u - h) / 0.045) ** 2)
    base = 120 if direction > 0 else 105
    f = (base + 35 * np.clip(u * 3, 0, 1)) * (0.55 + 0.45 * slow)
    ph = 2 * np.pi * np.cumsum(f) / SR
    motor = sum((0.88 ** k) / k * np.sin(k * ph + 0.4 * k) for k in range(1, 12))
    motor = np.sign(motor) * np.abs(motor) ** 0.7
    gear = 0.22 * np.sin(11 * ph) * (0.6 + 0.4 * np.sin(ph * 0.25))
    scrape = _filter(rng.standard_normal(len(t)), 1500, 5200) * 0.18 * (0.5 + 0.5 * slow)
    x = (0.5 * motor * (0.35 + 0.65 * slow) + gear * slow + scrape) * env
    for h in hitches:                                    # gear skip clunks
        at = h * dur + 0.03
        _place(x, _impulse(rng, dur=0.004, tau=0.0007, lo=800, hi=5000), at, 0.7)
        _place(x, _thunk(210, 0.012, 0.05), at, 0.5)
    for k in range(int(dur * 30)):
        _place(x, _impulse(rng, dur=0.002, tau=0.0003), rng.uniform(0.02, dur - 0.03),
               0.2 * rng.uniform(0.3, 1))
    return _finish(x * 0.8, drive=1.4, hi=5000, peak=0.6)


def chip_latch():
    """Chip carriage hits its stop and the detent latch drops in (click-CLACK)."""
    rng = np.random.default_rng(91)
    buf = np.zeros(int(SR * 0.2))
    _place(buf, _impulse(rng, dur=0.003, tau=0.0005, lo=2000, hi=8000), 0.0, 0.45)
    _place(buf, _impulse(rng, dur=0.006, tau=0.001, lo=900, hi=7000), 0.014, 1.0)
    _place(buf, _thunk(125, 0.03, 0.13, drop=0.3), 0.014, 1.0)
    _place(buf, _modes([700, 1720, 2890], [0.025, 0.014, 0.008], [0.9, 0.5, 0.3], 0.12, rng), 0.014, 0.4)
    _place(buf, _impulse(rng, tau=0.0005), 0.03, 0.3)
    return _finish(buf, drive=1.9, hi=6000)


def chip_seat():
    """Chip seated back in its slot: dull thunk with a contact click."""
    rng = np.random.default_rng(95)
    buf = np.zeros(int(SR * 0.16))
    _place(buf, _thunk(95, 0.04, 0.14, drop=0.3), 0.0, 1.1)
    _place(buf, _impulse(rng, dur=0.005, tau=0.0009, lo=600, hi=5000), 0.0, 0.8)
    _place(buf, _impulse(rng, dur=0.003, tau=0.0004, lo=2200, hi=8000), 0.018, 0.4)
    return _finish(buf, drive=1.8, hi=5500)


def detach_clunk():
    """Mini-player unplugs: magnetic catch lets go (CLUNK), contacts scrape,
    loose rattle."""
    rng = np.random.default_rng(101)
    buf = np.zeros(int(SR * 0.42))
    n = int(SR * 0.05)
    _place(buf, _filter(rng.standard_normal(n), 2500, 7500) * np.linspace(0.2, 1, n) * 0.3, 0.0)
    _place(buf, _impulse(rng, dur=0.007, tau=0.0012, lo=500, hi=6000), 0.05, 1.0)
    _place(buf, _thunk(78, 0.06, 0.25, drop=0.35), 0.05, 1.2)
    _place(buf, _modes([520, 1310, 2240, 3350], [0.04, 0.02, 0.012, 0.007],
                       [0.8, 0.5, 0.35, 0.2], 0.2, rng), 0.05, 0.35)
    for k, at in enumerate((0.13, 0.19, 0.235, 0.27)):    # rattle settling
        _place(buf, _impulse(rng, dur=0.003, tau=0.0005, lo=1200, hi=7000), at, 0.4 / (k + 1))
        _place(buf, _thunk(180, 0.01, 0.04), at, 0.2 / (k + 1))
    return _finish(buf, drive=1.8, hi=6000)


def dock_clunk():
    """Mini-player docks back: short slide, heavy clunk, latch click."""
    rng = np.random.default_rng(111)
    buf = np.zeros(int(SR * 0.34))
    n = int(SR * 0.1)
    _place(buf, _filter(rng.standard_normal(n), 1400, 5000) * np.linspace(0.3, 1, n) * 0.22, 0.0)
    _place(buf, _impulse(rng, dur=0.006, tau=0.001, lo=600, hi=6000), 0.1, 1.0)
    _place(buf, _thunk(90, 0.05, 0.2, drop=0.3), 0.1, 1.1)
    _place(buf, latch_clack() * 0.5, 0.135)
    return _finish(buf, drive=1.8, hi=6000)


# ---------------------------------------------------------------- stage 3: life
def _bitcrush(x, rate=8000, bits=6):
    """Sample-and-hold to `rate` + quantise: the crunchy telephone/modem grit."""
    hold = max(1, int(SR / rate))
    y = np.repeat(x[::hold], hold)[: len(x)]
    q = 2 ** (bits - 1)
    return np.round(y * q) / q


def power_relay():
    """Main power contactor slamming in + the transformer's first hum."""
    rng = np.random.default_rng(201)
    buf = np.zeros(int(SR * 0.45))
    _place(buf, _impulse(rng, dur=0.007, tau=0.0012, lo=500, hi=6500), 0.0, 1.0)
    _place(buf, _thunk(82, 0.05, 0.22, drop=0.3), 0.0, 1.2)
    _place(buf, _modes([560, 1340, 2410], [0.03, 0.016, 0.008], [0.8, 0.45, 0.25], 0.14, rng), 0.0, 0.35)
    _place(buf, _impulse(rng, tau=0.0006), 0.009, 0.45)
    _place(buf, _impulse(rng, tau=0.0004), 0.017, 0.2)
    t = _t(0.38)
    hum = sum(np.sin(2 * np.pi * 50 * k * t + k) / k for k in (1, 2, 3, 5))
    _place(buf, 0.22 * hum * np.clip(t / 0.03, 0, 1) * np.exp(-t / 0.16), 0.02)
    return _finish(buf, drive=1.8, hi=6000)


def relay_tick(variant=0):
    """Small signal relay ticking over (each vein bank coming online)."""
    rng = np.random.default_rng(211 + variant)
    buf = np.zeros(int(SR * 0.05))
    _place(buf, _impulse(rng, dur=0.002, tau=0.0003, lo=2500, hi=9000), 0, 1.0)
    _place(buf, _modes([3100 + 260 * variant, 5200], [0.004, 0.002], [0.5, 0.25], 0.03, rng), 0, 0.5)
    _place(buf, _impulse(rng, dur=0.002, tau=0.0003, lo=2500, hi=9000), 0.004, 0.3)
    return _finish(buf, drive=1.4, hi=8500, peak=0.5)


def boot_hum():
    """Old machine spinning up: mains hum swelling, a fan/whine rising in
    pitch and a little arcing crackle. Quiet bed under the boot sequence."""
    rng = np.random.default_rng(221)
    dur = 2.2
    t = _t(dur)
    u = t / dur
    env = np.clip(t / 0.25, 0, 1) * np.clip((dur - t) / 0.6, 0, 1)
    hum = sum(np.sin(2 * np.pi * 50 * k * t + 0.7 * k) * (0.8 ** k) for k in range(1, 8))
    f = 180 + 1400 * (1 - np.exp(-u * 2.2))
    whine = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.10 * np.clip(u * 3, 0, 1)
    fan = _filter(rng.standard_normal(len(t)), 300, 1600) * 0.25 * np.clip(u * 1.5, 0, 1)
    x = (0.45 * hum + whine + fan) * env
    for _ in range(14):
        _place(x, _impulse(rng, dur=0.002, tau=0.0003, lo=3000, hi=9000), rng.uniform(0.1, 1.4),
               0.15 * rng.uniform(0.3, 1))
    return _finish(x, drive=1.3, hi=5000, peak=0.5)


def crt_on():
    """CRT warming up: degauss THOONK with its wobbling 50 Hz buzz, the
    static crackle of the tube charging, a faint flyback tick."""
    rng = np.random.default_rng(231)
    dur = 0.75
    t = _t(dur)
    buf = np.zeros(len(t))
    _place(buf, _impulse(rng, dur=0.006, tau=0.001, lo=400, hi=5000), 0.0, 0.8)
    _place(buf, _thunk(70, 0.09, 0.4, drop=0.15), 0.0, 1.0)
    buzz = np.sign(np.sin(2 * np.pi * 50 * t)) * 0.5 + np.sin(2 * np.pi * 100 * t) * 0.3
    buf += _filter(buzz, 60, 1500) * np.exp(-t / 0.22) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t)) * 0.45
    st = _filter(rng.standard_normal(len(t)), 2000, 8000) * np.exp(-t / 0.18) * 0.22
    buf += st * (rng.random(len(t)) < 0.35)
    return _finish(buf, drive=1.6, hi=7000, peak=0.7)


def modem_chirp():
    """A short, crunchy dial-up style handshake: answer tone with phase
    flips, an FSK warble, then the scrambled 'kssshh', all band-limited to a
    phone line and bit-crushed. ~1.1 s."""
    rng = np.random.default_rng(241)
    parts = []
    t = _t(0.22)                                            # answer tone, 2 phase reversals
    ph = np.where((t > 0.07) & (t < 0.15), np.pi, 0.0)
    parts.append(np.sin(2 * np.pi * 2100 * t + ph) * 0.6)
    n = int(SR * 0.025)
    parts.append(np.zeros(n))
    seg = []                                                # FSK warble (V.21-ish tones)
    for _ in range(26):
        f = rng.choice([980, 1180, 1650, 1850, 2225])
        tt = _t(rng.uniform(0.008, 0.016))
        seg.append(np.sin(2 * np.pi * f * tt) * 0.55)
    parts.append(np.concatenate(seg))
    tt = _t(0.12)                                           # two-tone "bong"
    parts.append((np.sin(2 * np.pi * 1200 * tt) + np.sin(2 * np.pi * 2400 * tt)) * 0.3 * np.exp(-tt / 0.08))
    tt = _t(0.4)                                            # scrambled data hiss
    hiss = _filter(rng.standard_normal(len(tt)), 600, 3200) * 0.55
    hiss *= 0.75 + 0.25 * np.sign(np.sin(2 * np.pi * 37 * tt))
    hiss *= np.clip(tt / 0.02, 0, 1) * np.clip((0.4 - tt) / 0.12, 0, 1)
    parts.append(hiss)
    x = np.concatenate(parts)
    x = _filter(x, 320, 3300)
    x = _bitcrush(x / max(np.abs(x).max(), 1e-9), 7800, 6)
    return _finish(x, drive=1.5, hi=3600, peak=0.55)


def spark_crackle(variant=0):
    """Small electrical arc: a cluster of sharp snaps over a short sizzle."""
    rng = np.random.default_rng(251 + variant)
    dur = 0.22 + 0.06 * variant
    buf = np.zeros(int(SR * dur))
    t = _t(dur)
    siz = _filter(rng.standard_normal(len(t)), 3000, 9500) * np.exp(-t / (0.05 + 0.02 * variant)) * 0.35
    buf += siz * (rng.random(len(t)) < 0.5)
    at = 0.0
    for k in range(rng.integers(5, 9)):
        _place(buf, _impulse(rng, dur=0.003, tau=0.00025 + 0.0002 * rng.random(), lo=1800, hi=9500),
               at, rng.uniform(0.4, 1.0) * np.exp(-at / 0.08))
        at += rng.exponential(0.018)
        if at > dur - 0.01:
            break
    return _finish(buf, drive=2.0, hi=9000, peak=0.6)


def glitch_tear():
    """Data-tear glitch: a buzzy tone chopped into stuttering buffer repeats
    with pitch jumps, bit-crushed hard. ~0.38 s."""
    rng = np.random.default_rng(261)
    out = []
    for k in range(9):
        f = rng.choice([220, 330, 440, 660, 990, 1480])
        tt = _t(rng.uniform(0.015, 0.035))
        grain = np.sign(np.sin(2 * np.pi * f * tt)) * 0.6 + _filter(rng.standard_normal(len(tt)), 1500, 7000) * 0.3
        rep = int(rng.integers(1, 4))
        out.append(np.tile(grain, rep))
        if rng.random() < 0.3:
            out.append(np.zeros(int(SR * 0.008)))
    x = np.concatenate(out)[: int(SR * 0.38)]
    x = _bitcrush(x, 11025, 4)
    return _finish(x, drive=1.4, hi=7500, peak=0.5)


def static_tick():
    """One frame of TV static (the rare corrupted-face glitch)."""
    rng = np.random.default_rng(271)
    t = _t(0.09)
    x = _filter(rng.standard_normal(len(t)), 800, 8000) * np.clip((0.09 - t) / 0.03, 0, 1)
    return _finish(_bitcrush(x, 16000, 5), drive=1.2, hi=8000, peak=0.4)


def power_down():
    """Going to sleep: a soft relay click and a whine winding down."""
    rng = np.random.default_rng(281)
    dur = 1.3
    t = _t(dur)
    f = 60 + 700 * np.exp(-t / 0.35)
    wh = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5) * 0.35
    hum = np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.3) * 0.3
    buf = wh + hum
    _place(buf, relay_tick(1) * 0.8, 0.0)
    _place(buf, _thunk(90, 0.04, 0.15), 0.0, 0.4)
    return _finish(buf, drive=1.3, hi=5000, peak=0.45)


def wake_up():
    """Waking: a relay clicks in and a quick rising blip (the tube re-lights)."""
    rng = np.random.default_rng(291)
    dur = 0.32
    t = _t(dur)
    f = 300 + 1300 * (1 - np.exp(-t / 0.06))
    bl = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.09) * 0.4
    buf = bl
    _place(buf, relay_tick(0), 0.0)
    _place(buf, _thunk(120, 0.025, 0.1), 0.0, 0.5)
    return _finish(buf, drive=1.4, hi=6000, peak=0.5)


SOUNDS = {
    "btn_press_0": lambda: relay_clack(0),
    "btn_press_1": lambda: relay_clack(1),
    "btn_press_2": lambda: relay_clack(2),
    "btn_release": key_release,
    "latch": latch_clack,
    "servo_out": lambda: servo_whir(+1),
    "servo_in": lambda: servo_whir(-1, 0.30),
    "detent_0": lambda: detent_tick(0),
    "detent_1": lambda: detent_tick(1),
    "shutter": shutter_clack,
    # stage 2
    "eject_spring": eject_spring,
    "chip_slide_out": lambda: chip_slide(+1, 0.6, (0.35, 0.7)),
    "chip_slide_in": lambda: chip_slide(-1, 0.5, (0.5,)),
    "chip_latch": chip_latch,
    "chip_seat": chip_seat,
    "detach": detach_clunk,
    "dock": dock_clunk,
    # stage 3
    "power_relay": power_relay,
    "relay_tick_0": lambda: relay_tick(0),
    "relay_tick_1": lambda: relay_tick(1),
    "boot_hum": boot_hum,
    "crt_on": crt_on,
    "modem": modem_chirp,
    "crackle_0": lambda: spark_crackle(0),
    "crackle_1": lambda: spark_crackle(1),
    "crackle_2": lambda: spark_crackle(2),
    "glitch": glitch_tear,
    "static": static_tick,
    "sleep": power_down,
    "wake": wake_up,
}


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def generate_all(out_dir, overwrite=True):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, fn in SOUNDS.items():
        p = os.path.join(out_dir, name + ".wav")
        if overwrite or not os.path.exists(p):
            write_wav(p, fn())
        paths[name] = p
    return paths
