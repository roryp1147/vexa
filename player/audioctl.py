"""
Volume / balance routing for the shoulder dials.

Which audio the knobs act on depends on what is playing:

  * local file playing (or a paused local track)  -> Vexa itself:
      volume  = Vexa's own player volume (QAudioOutput)
      balance = Vexa's own Windows audio session, L/R channel volumes
                (IChannelAudioVolume of the session whose PID is ours)
  * SMTC app (Spotify, YouTube Music in Brave/Chrome/Edge, ...) -> that app:
      the SMTC SourceAppUserModelId ("brave.exe", "Spotify.exe",
      "Chrome", "MSEdge", "SpotifyAB.SpotifyMusic_...!Spotify", ...) is
      matched to the audio session(s) of that process (ISimpleAudioVolume +
      IChannelAudioVolume); fallback: the loudest active session; fallback:
      the system master volume (IAudioEndpointVolume) and its channel balance
  * system audio (loopback, no SMTC session)      -> loudest active session,
      else the system master volume

Everything Windows-specific lives in WinAudioBackend (pycaw). The routing
and the targets only use a handful of COM-style methods, so the tests drive
them with mock sessions on any OS.
"""
from __future__ import annotations

import os
import time

from . import config

# ------------------------------------------------------------------ helpers


def levels_for(balance: float, peak: float = 1.0) -> tuple[float, float]:
    """balance -1 (left) .. +1 (right) -> (left, right) channel levels."""
    b = max(-1.0, min(1.0, balance))
    return peak * min(1.0, 1.0 - b), peak * min(1.0, 1.0 + b)


def balance_of(left: float, right: float) -> float:
    m = max(left, right)
    if m <= 1e-6:
        return 0.0
    return max(-1.0, min(1.0, (right - left) / m))


def _side(i: int, n: int):
    """Channel index -> 'L' / 'R' / None (centre/LFE untouched).
    WASAPI order: FL FR FC LFE BL BR SL SR."""
    if n < 2:
        return None
    if i < 2:
        return "L" if i == 0 else "R"
    if i in (2, 3):
        return None
    return "L" if i % 2 == 0 else "R"


KNOWN_APPS = {
    # SMTC id fragment -> process name (without .exe)
    "brave": "brave", "chrome": "chrome", "msedge": "msedge", "microsoftedge": "msedge",
    "firefox": "firefox", "308046b0af4a39cb": "firefox", "opera": "opera", "vivaldi": "vivaldi",
    "spotify": "spotify", "tidal": "tidal", "deezer": "deezer", "foobar2000": "foobar2000",
    "vlc": "vlc", "zunemusic": "microsoft.media.player", "zunevideo": "video.ui",
    "musicbee": "musicbee", "itunes": "itunes", "applemusic": "applemusic", "discord": "discord",
}


def app_tokens(aid: str) -> list[str]:
    """SMTC SourceAppUserModelId -> candidate process names (lower, no .exe)."""
    a = (aid or "").strip().lower()
    if not a:
        return []
    a = a.split("!")[0]                       # UWP: Family_hash!App
    a = a.split("._crx_")[0]                  # Chromium PWA: Chrome._crx_<id>
    base = a.replace("\\", "/").rsplit("/", 1)[-1]
    toks = []
    if base.endswith(".exe"):
        toks.append(base[:-4])
    for k, v in KNOWN_APPS.items():
        if k in a and v not in toks:
            toks.append(v)
    if not toks:
        for part in base.replace("_", ".").split("."):
            if len(part) >= 4 and part not in ("exe", "com", "microsoft", "app", "apps"):
                toks.append(part)
    return toks


def _matches(name: str, tokens) -> bool:
    n = (name or "").lower()
    if n.endswith(".exe"):
        n = n[:-4]
    if not n:
        return False
    for t in tokens:
        if n == t or (len(t) >= 4 and (t in n or n in t)):
            return True
    return False


# ------------------------------------------------------------------ targets
class Target:
    kind = "none"
    tag = "VOL"
    fallback = False

    def volume(self):
        return None

    def set_volume(self, v: float) -> bool:
        return False

    def balance(self):
        return None

    def set_balance(self, b: float) -> bool:
        return False

    def describe(self) -> str:
        return self.kind


class OwnTarget(Target):
    """Vexa's own playback: player volume + own session channel balance."""
    kind = "own"
    tag = "VOL"

    def __init__(self, backend, get_vol, set_vol):
        self.backend = backend
        self.get_vol, self.set_vol = get_vol, set_vol
        self.bal = 0.0
        self._ch = None
        self.applied = False

    def volume(self):
        return self.get_vol()

    def set_volume(self, v):
        self.set_vol(v)
        return True

    def balance(self):
        return self.bal

    def set_balance(self, b):
        self.bal = max(-1.0, min(1.0, b))
        return self.apply()

    def apply(self) -> bool:
        """Push the balance to our session (it only exists once audio plays)."""
        if self.backend is None or not self.backend.ok:
            self.applied = False
            return False
        try:
            if self._ch is None:
                s = self.backend.own_session()
                if s is None:
                    self.applied = False
                    return False
                self._ch = s.channel
            _set_channels(self._ch, self.bal, session=True)
            self.applied = True
            return True
        except Exception:
            self._ch = None
            self.applied = False
            return False

    def describe(self):
        return "own (Vexa player volume + own session balance)"


class SessionTarget(Target):
    """Another app's audio session(s), e.g. every brave.exe session."""
    kind = "app"

    def __init__(self, sessions, name: str, fallback=False):
        self.sessions = list(sessions)
        self.name = name
        self.fallback = fallback
        base = name.lower()[:-4] if name.lower().endswith(".exe") else name
        self.tag = base.upper()[:6] or "APP"

    def volume(self):
        try:
            return float(self.sessions[0].simple.GetMasterVolume())
        except Exception:
            return None

    def set_volume(self, v):
        ok = False
        for s in self.sessions:
            try:
                s.simple.SetMasterVolume(max(0.0, min(1.0, v)), None)
                ok = True
            except Exception:
                pass
        return ok

    def balance(self):
        try:
            ch = self.sessions[0].channel
            n = ch.GetChannelCount()
            if n < 2:
                return 0.0
            return balance_of(ch.GetChannelVolume(0), ch.GetChannelVolume(1))
        except Exception:
            return None

    def set_balance(self, b):
        ok = False
        for s in self.sessions:
            try:
                _set_channels(s.channel, b, session=True)
                ok = True
            except Exception:
                pass
        return ok

    def describe(self):
        pids = ",".join(str(s.pid) for s in self.sessions)
        return f"app session {self.name} pid {pids}" + (" (fallback: loudest)" if self.fallback else "")


class MasterTarget(Target):
    """System master volume of the default output device."""
    kind = "master"
    tag = "SYS"

    def __init__(self, endpoint, fallback=True):
        self.ep = endpoint
        self.fallback = fallback

    def volume(self):
        try:
            return float(self.ep.GetMasterVolumeLevelScalar())
        except Exception:
            return None

    def set_volume(self, v):
        try:
            self.ep.SetMasterVolumeLevelScalar(max(0.0, min(1.0, v)), None)
            return True
        except Exception:
            return False

    def balance(self):
        try:
            n = self.ep.GetChannelCount()
            if n < 2:
                return 0.0
            return balance_of(self.ep.GetChannelVolumeLevelScalar(0), self.ep.GetChannelVolumeLevelScalar(1))
        except Exception:
            return None

    def set_balance(self, b):
        try:
            m = float(self.ep.GetMasterVolumeLevelScalar())
            _set_channels(self.ep, b, session=False, peak=m)
            return True
        except Exception:
            return False

    def describe(self):
        return "system master volume (default output device)"


def _set_channels(iface, b, session=True, peak=1.0):
    n = iface.GetChannelCount()
    left, right = levels_for(b, peak)
    for i in range(n):
        side = _side(i, n)
        lv = left if side == "L" else right if side == "R" else peak
        if n == 1:
            lv = peak
        if session:
            iface.SetChannelVolume(i, lv, None)
        else:
            iface.SetChannelVolumeLevelScalar(i, lv, None)


# ------------------------------------------------------------------ router
class AudioRouter:
    """Resolves the dial target for a route and remembers it."""

    REFRESH_FALLBACK_S = 4.0

    def __init__(self, backend, own: OwnTarget):
        self.backend = backend
        self.own = own
        self.route = ("local",)
        self.target: Target = own
        self.t_resolved = -1e9

    @staticmethod
    def _external(sessions, own_pid):
        return [s for s in sessions if s.pid not in (0, own_pid)]

    def resolve(self, route, now=None) -> Target:
        now = time.monotonic() if now is None else now
        self.route = route
        self.t_resolved = now
        self.target = self._resolve(route)
        return self.target

    def _resolve(self, route) -> Target:
        b = self.backend
        if route[0] == "local" or b is None or not b.ok:
            return self.own
        try:
            sessions = self._external(b.sessions(), b.own_pid)
        except Exception:
            sessions = []
        if route[0] == "smtc":
            toks = app_tokens(route[1] if len(route) > 1 else "")
            hits = [s for s in sessions if _matches(s.name, toks)]
            if hits:
                hits.sort(key=lambda s: -s.peak)
                return SessionTarget(hits, hits[0].name)
        loud = [s for s in sessions if s.peak > 0.0005]
        if loud:
            loud.sort(key=lambda s: -s.peak)
            top = loud[0]
            same = [s for s in loud if s.name.lower() == top.name.lower()]
            return SessionTarget(same, top.name, fallback=True)
        try:
            ep = b.endpoint()
        except Exception:
            ep = None
        if ep is not None:
            return MasterTarget(ep)
        return Target()

    def needs_refresh(self, now) -> bool:
        return (self.route[0] != "local" and self.target.fallback
                and now - self.t_resolved > self.REFRESH_FALLBACK_S)


# ------------------------------------------------------------------ Windows backend
class SessionRef:
    __slots__ = ("pid", "name", "peak", "simple", "channel")

    def __init__(self, pid, name, peak, simple, channel):
        self.pid, self.name, self.peak = pid, name, peak
        self.simple, self.channel = simple, channel


class WinAudioBackend:
    """pycaw access to the default render device's sessions + endpoint.
    All calls happen on the GUI thread (COM is initialised there)."""

    def __init__(self):
        self.ok = False
        self.error = ""
        self.own_pid = os.getpid()
        self._ep = None
        if not config.IS_WINDOWS:
            self.error = "not on Windows"
            return
        try:
            from pycaw.pycaw import AudioUtilities  # noqa: F401
            self.ok = True
        except Exception as e:
            self.error = f"pycaw missing ({e.__class__.__name__})"

    def _ref(self, s):
        from pycaw.pycaw import IAudioMeterInformation
        pid = s.ProcessId
        try:
            name = s.Process.name() if s.Process is not None else ""
        except Exception:
            name = ""
        try:
            peak = float(s._ctl.QueryInterface(IAudioMeterInformation).GetPeakValue())
        except Exception:
            peak = 0.0
        try:
            ch = s.channelAudioVolume()
        except AttributeError:
            from pycaw.api.audioclient import IChannelAudioVolume
            ch = s._ctl.QueryInterface(IChannelAudioVolume)
        return SessionRef(pid, name, peak, s.SimpleAudioVolume, ch)

    def sessions(self) -> list[SessionRef]:
        from pycaw.pycaw import AudioUtilities
        out = []
        for s in AudioUtilities.GetAllSessions():
            try:
                out.append(self._ref(s))
            except Exception:
                pass
        return out

    def own_session(self):
        from pycaw.pycaw import AudioUtilities
        for s in AudioUtilities.GetAllSessions():
            try:
                if s.ProcessId == self.own_pid:
                    return self._ref(s)
            except Exception:
                pass
        return None

    def endpoint(self):
        if self._ep is not None:
            try:
                self._ep.GetMasterVolumeLevelScalar()
                return self._ep
            except Exception:
                self._ep = None
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        spk = AudioUtilities.GetSpeakers()
        if spk is None:
            return None
        ev = getattr(spk, "EndpointVolume", None)
        if ev is None:                         # older pycaw: raw IMMDevice
            import comtypes
            ev = spk.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None).QueryInterface(
                IAudioEndpointVolume)
        self._ep = ev
        return ev
