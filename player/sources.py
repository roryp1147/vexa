"""
Audio input abstraction.

  LocalPlayer     - QMediaPlayer playback of MP3/FLAC/WAV/M4A/OGG. Analysis taps
                    the *decoded* stream through QAudioBufferOutput (Qt >= 6.8),
                    time-aligned to what is currently audible.
  LoopbackSource  - Windows WASAPI loopback of the default output device, so the
                    visuals follow YouTube Music / Spotify / anything playing.
                    Uses PyAudioWPatch (PortAudio fork with loopback, cp314
                    wheels); falls back to `soundcard`; otherwise unavailable.
  SynthSource     - deterministic procedural test music (kick/snare/hats/bass/
                    pads) for development + screenshots on non-Windows boxes.
  AudioHub        - picks which source currently drives the visuals.

Every source answers latest(n) -> (mono float32 array | None, samplerate).
"""
from __future__ import annotations

import os
import random
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QUrl, Signal

from . import config

try:
    from PySide6.QtMultimedia import (QAudio, QAudioFormat, QAudioOutput,
                                      QMediaPlayer)
    try:
        from PySide6.QtMultimedia import QAudioBufferOutput  # Qt 6.8+
    except ImportError:  # pragma: no cover
        QAudioBufferOutput = None
    HAVE_MM = True
except ImportError:  # pragma: no cover
    HAVE_MM = False
    QAudioBufferOutput = None

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".aac", ".wma")


class RingBuffer:
    """Thread-safe mono float32 ring buffer with an absolute write counter."""

    def __init__(self, capacity: int = 1 << 15):
        self.buf = np.zeros(capacity, np.float32)
        self.cap = capacity
        self.total = 0                  # samples ever written
        self.lock = threading.Lock()

    def write(self, x: np.ndarray):
        x = np.asarray(x, np.float32).ravel()
        n = len(x)
        if n == 0:
            return
        with self.lock:
            if n >= self.cap:
                self.buf[:] = x[-self.cap:]
                self.total += n
                return
            i = self.total % self.cap
            j = min(self.cap, i + n)
            self.buf[i:j] = x[: j - i]
            if j - i < n:
                self.buf[: n - (j - i)] = x[j - i:]
            self.total += n

    def read_until(self, end: int, n: int) -> np.ndarray | None:
        """n samples ending at absolute index `end` (clamped to what exists)."""
        with self.lock:
            end = min(end, self.total)
            if end < n or self.total - end > self.cap - n:
                return None
            idx = (np.arange(end - n, end) % self.cap)
            return self.buf[idx]

    def latest(self, n: int) -> np.ndarray | None:
        return self.read_until(self.total, n)

    def clear(self):
        with self.lock:
            self.total = 0
            self.buf[:] = 0


# --------------------------------------------------------------------------
class SynthSource:
    """Procedural 118 bpm test track, a pure function of the sample index."""

    name = "synth"
    sr = 44100

    def __init__(self):
        self.t0 = time.monotonic()
        self.fixed_time: float | None = None   # tests pin the clock
        self.available = True

    def start(self):
        self.t0 = time.monotonic()

    def stop(self):
        pass

    def now(self) -> float:
        return self.fixed_time if self.fixed_time is not None else time.monotonic() - self.t0

    @staticmethod
    def _noise(i):
        return (np.modf(np.sin(i * 12.9898 + 78.233) * 43758.5453)[0]) * 2.0

    def render(self, i0: int, n: int) -> np.ndarray:
        sr = self.sr
        i = np.arange(i0, i0 + n, dtype=np.float64)
        t = i / sr
        beat = 60.0 / 118.0
        pb = t / beat
        bi = np.floor(pb)
        tb = (pb - bi) * beat
        bar = np.floor(bi / 4)
        phrase = bar % 16
        breakdown = (phrase >= 14)              # 2 bars without drums
        roots = np.array([55.0, 43.65, 65.41, 49.0])
        root = roots[(bar % 4).astype(int)]
        # kick (pitch-swept sine)
        kick_ph = 45 * tb + 90 * 0.03 * (1 - np.exp(-tb / 0.03))
        kick = np.sin(2 * np.pi * kick_ph) * np.exp(-tb / 0.26) * (~breakdown)
        # snare on beats 2 & 4
        sn_on = (bi % 2 == 1) & (~breakdown)
        snare = (self._noise(i) * 0.5 * np.exp(-tb / 0.09)
                 + 0.4 * np.sin(2 * np.pi * 186 * tb) * np.exp(-tb / 0.05)) * sn_on
        # hats on 8ths
        p8 = pb * 2
        t8 = (p8 - np.floor(p8)) * beat / 2
        hat = (self._noise(i) - self._noise(i - 1)) * np.exp(-t8 / 0.022) * 0.18
        # bass on off-8ths
        off = (np.floor(p8) % 2 == 1)
        bph = 2 * np.pi * root * t
        bass = sum(np.sin(k * bph) / k for k in range(1, 6)) * np.exp(-t8 / 0.16) * off * 0.5
        # pad chord
        pad = sum(np.sin(2 * np.pi * root * r * k * t) / k
                  for r in (4.0, 5.04, 6.0) for k in (1, 2, 3, 4, 5)) * 0.035 \
            * (0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
        # arpeggio 16ths in the 2nd half of each phrase
        p16 = pb * 4
        t16 = (p16 - np.floor(p16)) * beat / 4
        arp_n = np.array([4.0, 5.04, 6.0, 8.0])[(np.floor(p16) % 4).astype(int)]
        aph = 2 * np.pi * root * arp_n * 2 * t
        arp = sum(np.sin(k * aph) / k for k in (1, 3, 5, 7, 9)) * np.exp(-t16 / 0.07) \
            * (0.08 + 0.08 * (phrase >= 8))
        x = 0.9 * kick + 0.55 * snare + hat + bass + pad + arp
        return (np.tanh(x * 0.9) * 0.8).astype(np.float32)

    def latest(self, n: int):
        end = int(self.now() * self.sr)
        return self.render(max(0, end - n), n), self.sr

    def is_active(self) -> bool:
        return True


# --------------------------------------------------------------------------
class LoopbackSource:
    """WASAPI loopback capture of the default speaker (Windows only)."""

    name = "loopback"

    def __init__(self):
        self.ring = RingBuffer()
        self.sr = 48000
        self.backend = None
        self.available = False
        self.error = ""
        self._stop = threading.Event()
        self._thread = None
        self._last_data = 0.0
        self._pa = None
        self._stream = None
        if not config.IS_WINDOWS:
            self.error = "not Windows"
            return
        try:
            import pyaudiowpatch  # noqa: F401
            self.backend = "pyaudiowpatch"
            self.available = True
        except Exception:
            try:
                import soundcard  # noqa: F401
                self.backend = "soundcard"
                self.available = True
            except Exception as e:  # pragma: no cover - Windows only
                self.error = f"no loopback library ({e})"

    # -- PyAudioWPatch: callback stream on PortAudio's own thread ---------
    def _start_pyaudio(self):
        import pyaudiowpatch as pyaudio
        pa = pyaudio.PyAudio()
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        spk = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if not spk.get("isLoopbackDevice"):
            for lb in pa.get_loopback_device_info_generator():
                if spk["name"] in lb["name"]:
                    spk = lb
                    break
            else:
                raise RuntimeError("no loopback device for default speaker")
        ch = max(1, int(spk["maxInputChannels"]))
        self.sr = int(spk["defaultSampleRate"])

        def cb(data, frames, tinfo, status):
            a = np.frombuffer(data, np.float32)
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            self.ring.write(a)
            self._last_data = time.monotonic()
            return (None, pyaudio.paContinue)

        self._pa = pa
        self._stream = pa.open(format=pyaudio.paFloat32, channels=ch, rate=self.sr,
                               input=True, input_device_index=spk["index"],
                               frames_per_buffer=512, stream_callback=cb)
        self._stream.start_stream()

    # -- soundcard: blocking reads in our own thread ----------------------
    def _soundcard_loop(self):
        import soundcard as sc
        while not self._stop.is_set():
            try:
                spk = sc.default_speaker()
                mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
                with mic.recorder(samplerate=self.sr, blocksize=1024) as rec:
                    while not self._stop.is_set():
                        d = rec.record(numframes=1024)
                        self.ring.write(d.mean(axis=1) if d.ndim > 1 else d)
                        self._last_data = time.monotonic()
            except Exception as e:  # device change etc. -> retry
                self.error = str(e)
                self._stop.wait(2.0)

    def start(self):
        if not self.available or self._thread or self._stream:
            return
        try:
            if self.backend == "pyaudiowpatch":
                self._start_pyaudio()
            else:
                self._stop.clear()
                self._thread = threading.Thread(target=self._soundcard_loop,
                                                name="vexa-loopback", daemon=True)
                self._thread.start()
        except Exception as e:  # pragma: no cover - Windows only
            self.error = str(e)
            self.available = False

    def stop(self):
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass
        self._stream = self._pa = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self, n: int):
        if time.monotonic() - self._last_data > 0.5:   # WASAPI sends nothing when silent
            return None, self.sr
        return self.ring.latest(n), self.sr

    def is_active(self) -> bool:
        return self.available and time.monotonic() - self._last_data < 0.5


# --------------------------------------------------------------------------
class LocalPlayer(QObject):
    """Local file playback with a simple playlist (the chip UI comes in stage 2)."""

    name = "local"
    stateChanged = Signal()
    timeChanged = Signal()
    trackChanged = Signal()

    SR = 44100

    def __init__(self, parent=None, audible: bool = True):
        super().__init__(parent)
        self.available = HAVE_MM
        self.files: list[str] = []
        self.index = -1
        self.shuffle = False
        self.repeat = 0          # 0 off, 1 all, 2 one
        self._order: list[int] = []
        self.ring = RingBuffer()
        self._buf_end_time = 0.0      # µs media time of newest buffer end
        self._buf_arrival = 0.0
        self._buf_end_index = 0
        self.volume = 0.7
        self.muted = False
        if not HAVE_MM:
            return
        self.player = QMediaPlayer(self)
        # Under VEXA_MUTE we never even create an audio output -> guaranteed
        # silence; QMediaPlayer still runs its clock and feeds the buffer tap.
        self.audio_out = QAudioOutput(self) if audible else None
        if self.audio_out is not None:
            self.player.setAudioOutput(self.audio_out)
        self.buf_out = None
        if QAudioBufferOutput is not None:
            fmt = QAudioFormat()
            fmt.setSampleRate(self.SR)
            fmt.setChannelCount(2)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
            self.buf_out = QAudioBufferOutput(fmt, self)
            self.player.setAudioBufferOutput(self.buf_out)
            self.buf_out.audioBufferReceived.connect(self._on_buffer)
        self.player.playbackStateChanged.connect(lambda *_: self.stateChanged.emit())
        self.player.positionChanged.connect(lambda *_: self.timeChanged.emit())
        self.player.durationChanged.connect(lambda *_: self.timeChanged.emit())
        self.player.mediaStatusChanged.connect(self._on_status)
        self.player.errorOccurred.connect(self._on_error)
        self.set_volume(self.volume)

    # ---- analysis tap ---------------------------------------------------
    def _on_buffer(self, buf):
        try:
            fmt = buf.format()
            ch = max(1, fmt.channelCount())
            a = np.frombuffer(buf.constData(), np.float32, count=buf.frameCount() * ch)
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            self.ring.write(a)
            self._buf_arrival = time.monotonic()
            self._buf_start_index = self.ring.total - len(a)
        except Exception:
            pass

    def latest(self, n: int):
        if not self.is_playing():
            return None, self.SR
        # Buffers arrive as they start playing; advance through the newest
        # buffer with the wall clock so the visuals stay in sync and smooth.
        el = time.monotonic() - self._buf_arrival
        end = getattr(self, "_buf_start_index", 0) + int(el * self.SR)
        return self.ring.read_until(end, n), self.SR

    def is_active(self) -> bool:
        return self.is_playing()

    # ---- transport ------------------------------------------------------
    def is_playing(self) -> bool:
        return HAVE_MM and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def has_media(self) -> bool:
        return bool(self.files)

    def set_files(self, paths, play=True):
        files = [p for p in paths if os.path.splitext(p)[1].lower() in AUDIO_EXTS]
        if not files:
            return
        self.files = files
        self._reshuffle()
        self._load(self._order[0] if self.shuffle else 0)
        if play:
            self.play()

    def add_files(self, paths):
        if not self.files:
            self.set_files(paths)
            return
        self.files += [p for p in paths if os.path.splitext(p)[1].lower() in AUDIO_EXTS]
        self._reshuffle()

    def _reshuffle(self):
        self._order = list(range(len(self.files)))
        if self.shuffle:
            random.shuffle(self._order)

    def _load(self, idx):
        if not HAVE_MM or not self.files:
            return
        self.index = idx % len(self.files)
        self.ring.clear()
        self.player.setSource(QUrl.fromLocalFile(self.files[self.index]))
        self.trackChanged.emit()

    def play(self):
        if HAVE_MM and self.files:
            self.player.play()

    def pause(self):
        if HAVE_MM:
            self.player.pause()

    def toggle(self):
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def stop(self):
        if HAVE_MM:
            self.player.stop()

    def _step(self, d):
        if not self.files:
            return
        was = self.is_playing() or d != 0
        if self.shuffle and self._order:
            pos = self._order.index(self.index) if self.index in self._order else 0
            self._load(self._order[(pos + d) % len(self._order)])
        else:
            self._load(self.index + d)
        if was:
            self.play()

    def next(self):
        self._step(+1)

    def previous(self):
        if HAVE_MM and self.player.position() > 3000:
            self.player.setPosition(0)
            return
        self._step(-1)

    def set_shuffle(self, on: bool):
        self.shuffle = on
        self._reshuffle()

    def cycle_repeat(self):
        self.repeat = (self.repeat + 1) % 3
        return self.repeat

    def _on_status(self, st):
        if st == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.repeat == 2:
                self.player.setPosition(0)
                self.play()
            elif self.repeat == 1 or self.index < len(self.files) - 1 or self.shuffle:
                last = (not self.shuffle) and self.index == len(self.files) - 1
                if last and self.repeat == 0:
                    return
                self._step(+1)

    def _on_error(self, err, msg):
        print(f"[vexa] media error: {msg}")

    def set_volume(self, v: float):
        self.volume = max(0.0, min(1.0, v))
        if HAVE_MM and self.audio_out is not None:
            lin = QAudio.convertVolume(self.volume, QAudio.VolumeScale.LogarithmicVolumeScale,
                                       QAudio.VolumeScale.LinearVolumeScale)
            self.audio_out.setVolume(lin)

    def set_muted(self, m: bool):
        self.muted = m
        if HAVE_MM and self.audio_out is not None:
            self.audio_out.setMuted(m)

    def position_ms(self) -> int:
        return self.player.position() if HAVE_MM else 0

    def duration_ms(self) -> int:
        return self.player.duration() if HAVE_MM else 0

    def track_name(self) -> str:
        if 0 <= self.index < len(self.files):
            return os.path.splitext(os.path.basename(self.files[self.index]))[0]
        return ""


# --------------------------------------------------------------------------
class AudioHub:
    """Chooses the source that drives the visuals.

    auto     : local playback when playing, else system loopback (Windows)
    local    : local playback only
    loopback : system loopback only
    synth    : procedural test track (dev / screenshots)
    none     : silence
    """

    def __init__(self, local: LocalPlayer, mode: str = "auto"):
        self.local = local
        self.loopback = LoopbackSource()
        self.synth = SynthSource()
        self.mode = None
        self.set_mode(mode)

    def set_mode(self, mode: str):
        if mode not in ("auto", "local", "loopback", "synth", "none"):
            mode = "auto"
        self.mode = mode
        if mode in ("auto", "loopback"):
            self.loopback.start()
        else:
            self.loopback.stop()
        if mode == "synth":
            self.synth.start()

    def latest(self, n: int):
        m = self.mode
        if m == "synth":
            return self.synth.latest(n)
        if m in ("auto", "local") and self.local.is_playing():
            return self.local.latest(n)
        if m in ("auto", "loopback") and self.loopback.available:
            return self.loopback.latest(n)
        return None, 44100

    def describe(self) -> str:
        lb = self.loopback
        lbs = f"loopback={lb.backend}" if lb.available else f"loopback=off ({lb.error})"
        return f"source={self.mode} {lbs}"

    def shutdown(self):
        self.loopback.stop()
        if self.local.available:
            self.local.stop()
