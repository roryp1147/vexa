"""
Now-playing model + routing between local playback and the Windows media
session (SMTC: Spotify, YouTube Music in a browser, ...).

  NowPlaying     plain data for the chips / LCD (title, artist, art key,
                 status, position extrapolated from a timestamp)
  NullProvider   no SMTC (Linux / missing winrt / failure)
  MockProvider   scripted "Spotify" session for the offscreen tests
  MediaHub       decides who the chest buttons + chips talk to:
                   * local playback while it plays
                   * else an SMTC session that is playing
                   * else paused local media (position > 0)
                   * else any SMTC session, else local, else nothing
The real SMTC provider lives in platform_win.SmtcProvider (background asyncio
thread, results delivered by a queued Qt signal).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from PySide6.QtCore import QObject, Signal

from . import tags


@dataclass
class NowPlaying:
    source: str = "none"            # "local" | "smtc" | "none"
    app: str = ""                   # "Spotify", "Chrome", "Local files", ...
    app_id: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    art_key: str | None = None      # key into tags' art cache
    playing: bool = False
    status: str = "stopped"         # playing | paused | stopped | changing | closed
    position_ms: int = 0
    duration_ms: int = 0
    stamp: float = 0.0              # clock time when position_ms was sampled
    shuffle: bool = False
    repeat: int = 0                 # local scheme: 0 off, 1 all, 2 one
    controls: dict = field(default_factory=dict)   # capability flags

    def pos_at(self, now: float) -> int:
        p = self.position_ms
        if self.playing and self.stamp:
            p += int(max(0.0, now - self.stamp) * 1000)
        if self.duration_ms:
            p = min(p, self.duration_ms)
        return max(0, p)

    def display_key(self):
        return (self.source, self.title, self.artist, self.art_key, self.playing, self.app)


def app_label(app_id: str) -> str:
    """Friendly name for an SMTC SourceAppUserModelId."""
    a = (app_id or "").lower()
    for k, v in (("spotify", "Spotify"), ("chrome", "Chrome"), ("msedge", "Edge"),
                 ("firefox", "Firefox"), ("opera", "Opera"), ("brave", "Brave"),
                 ("vivaldi", "Vivaldi"), ("zunemusic", "Media Player"), ("vlc", "VLC"),
                 ("foobar", "foobar2000"), ("aimp", "AIMP"), ("musicbee", "MusicBee"),
                 ("itunes", "iTunes"), ("applemusic", "Apple Music"), ("tidal", "TIDAL"),
                 ("deezer", "Deezer"), ("youtube", "YouTube Music")):
        if k in a:
            return v
    base = app_id.split("!")[0].split("\\")[-1]
    return base.removesuffix(".exe")[:16] or "Media"


# ----------------------------------------------------------------- providers
class NullProvider(QObject):
    updated = Signal(object)        # NowPlaying | None
    available = False
    error = "not on Windows"

    def command(self, name: str, *args):
        return False

    def shutdown(self):
        pass


class MockProvider(QObject):
    """Scripted SMTC session for tests ("Spotify"-style)."""
    updated = Signal(object)
    available = True
    error = ""

    def __init__(self, clock, np: NowPlaying | None = None):
        super().__init__()
        self.clock = clock
        self.np = np
        self.commands: list[str] = []

    def set(self, np: NowPlaying | None):
        self.np = np
        if np is not None and not np.stamp:
            np.stamp = self.clock()
        self.updated.emit(None if np is None else replace(np))

    def command(self, name: str, *args):
        self.commands.append(name)
        np = self.np
        if np is None:
            return False
        now = self.clock()
        np.position_ms = np.pos_at(now)
        np.stamp = now
        if name == "toggle":
            np.playing = not np.playing
        elif name == "pause":
            np.playing = False
        elif name == "play":
            np.playing = True
        elif name in ("next", "prev"):
            np.position_ms = 0
            np.title = ("Next" if name == "next" else "Previous") + " track"
        elif name == "shuffle":
            np.shuffle = bool(args[0]) if args else not np.shuffle
        elif name == "repeat":
            np.repeat = int(args[0]) if args else (np.repeat + 1) % 3
        np.status = "playing" if np.playing else "paused"
        self.updated.emit(replace(np))
        return True

    def shutdown(self):
        pass


def make_provider():
    """Real SMTC provider on Windows (if winrt imports), else NullProvider."""
    try:
        from .platform_win import SmtcProvider
        p = SmtcProvider()
        if p.available:
            return p
        return p                    # carries .error for the log line
    except Exception as e:          # never let SMTC take the app down
        n = NullProvider()
        n.error = f"smtc init failed: {e}"
        return n


# ----------------------------------------------------------------- the hub
class MediaHub(QObject):
    changed = Signal()              # display-relevant change (track/art/status/route)

    def __init__(self, local, provider, clock=time.monotonic):
        super().__init__()
        self.local = local
        self.provider = provider
        self.clock = clock
        self.smtc: NowPlaying | None = None
        self._key = None
        provider.updated.connect(self._on_smtc)
        local.stateChanged.connect(self._poke)
        local.trackChanged.connect(self._poke)

    # -- SMTC updates arrive here on the GUI thread (queued signal)
    def _on_smtc(self, np):
        if np is not None and not np.stamp:
            np.stamp = self.clock()
        self.smtc = np
        self._poke()

    def _poke(self, *_):
        k = self.now().display_key()
        if k != self._key:
            self._key = k
            self.changed.emit()

    # -- routing
    def route(self) -> str:
        lp, s = self.local, self.smtc
        if lp.is_playing():
            return "local"
        if s is not None and s.playing:
            return "smtc"
        if lp.has_media() and lp.position_ms() > 0:
            return "local"
        if s is not None:
            return "smtc"
        if lp.has_media():
            return "local"
        return "none"

    def local_np(self) -> NowPlaying:
        lp = self.local
        if not lp.has_media() or lp.index < 0:
            return NowPlaying(source="local", app="Local files")
        ti = tags.track_info(lp.files[lp.index])
        dur = lp.duration_ms() or ti.duration_ms
        playing = lp.is_playing()
        return NowPlaying(source="local", app="Local files", title=ti.title, artist=ti.artist,
                          album=ti.album, art_key=ti.art_key, playing=playing,
                          status="playing" if playing else ("paused" if lp.position_ms() else "stopped"),
                          position_ms=lp.position_ms(), duration_ms=dur, stamp=0.0,
                          shuffle=lp.shuffle, repeat=lp.repeat)

    def now(self) -> NowPlaying:
        r = self.route()
        if r == "local":
            return self.local_np()
        if r == "smtc":
            return self.smtc
        return NowPlaying()

    # -- transport (chest buttons, mini player)
    def control(self, name: str, *args) -> str:
        """Returns the route used ('local' / 'smtc' / 'none')."""
        r = self.route()
        if r == "smtc":
            s = self.smtc
            if name == "stop":
                name = "pause"                       # SMTC 'stop' kills some apps' sessions
            elif name == "shuffle":
                args = (not s.shuffle,)
            elif name == "repeat":
                args = ((s.repeat + 1) % 3,)
            try:
                self.provider.command(name, *args)
            except Exception as e:
                print(f"[vexa] smtc command failed: {e}")
            return r
        lp = self.local
        if name in ("toggle", "play"):
            lp.toggle() if name == "toggle" else lp.play()
        elif name == "pause":
            lp.pause()
        elif name == "stop":
            lp.stop()
        elif name == "next":
            lp.next()
        elif name == "prev":
            lp.previous()
        elif name == "shuffle":
            lp.set_shuffle(not lp.shuffle)
        elif name == "repeat":
            lp.cycle_repeat()
        self._poke()
        return r

    def shutdown(self):
        try:
            self.provider.shutdown()
        except Exception:
            pass
