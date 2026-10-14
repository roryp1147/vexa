"""
Windows-only integrations, all guarded so the app runs anywhere.

  * set_app_user_model_id()  - own taskbar identity/icon grouping
  * SessionBalance           - per-app L/R balance via the Windows audio
                               session (IChannelAudioVolume, through pycaw).
                               Only affects Vexa's own local playback.
  * SmtcProvider             - SMTC now-playing + transport for YouTube
                               Music / Spotify via winrt-Windows.Media.Control
                               (background asyncio thread -> Qt signal).
"""
from __future__ import annotations

import asyncio
import datetime
import os
import threading
import time

from PySide6.QtCore import QObject, Signal

from . import config, tags
from .media import NowPlaying, app_label


def set_app_user_model_id(app_id: str = "Vexa.MediaPlayer"):
    if not config.IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


class SessionBalance:
    """Left/right balance for this process' audio session.

    The session only exists once Vexa has started playing audio, so apply()
    retries lazily. On non-Windows or without pycaw it is a no-op and the
    dial is visual only."""

    def __init__(self):
        self.ok = False
        self.balance = 0.0
        self._vol = None
        if not config.IS_WINDOWS:
            return
        try:
            from pycaw.pycaw import AudioUtilities  # noqa: F401
            self.ok = True
        except Exception:
            self.ok = False

    def _iface(self):
        if self._vol is not None:
            return self._vol
        from pycaw.pycaw import AudioUtilities
        s = AudioUtilities.GetProcessSession(os.getpid())
        if s is None:
            return None
        self._vol = s.channelAudioVolume()
        return self._vol

    def apply(self, balance: float | None = None):
        if balance is not None:
            self.balance = max(-1.0, min(1.0, balance))
        if not self.ok:
            return False
        try:
            iv = self._iface()
            if iv is None:
                return False
            n = iv.GetChannelCount()
            left = min(1.0, 1.0 - self.balance)
            right = min(1.0, 1.0 + self.balance)
            for i in range(n):
                iv.SetChannelVolume(i, left if i % 2 == 0 else right, None)
            return True
        except Exception:
            self._vol = None   # session recreated (device change) -> retry later
            return False


class SmtcProvider(QObject):
    """Windows System Media Transport Controls (Spotify, YouTube Music in a
    browser, ...) through pywinrt (winrt-Windows.Media.Control, cp314 wheels).

    * a daemon thread runs its own asyncio loop; the GUI thread never awaits
      anything WinRT
    * light polling: playback status + timeline every 1 s while a session
      exists (2.5 s otherwise), media properties every ~2 s or when the
      status/session changes, thumbnail only when the track changes
    * results go to the GUI as a NowPlaying via a queued Qt signal
    * commands are scheduled on the loop with run_coroutine_threadsafe and
      wake the poller for an immediate refresh
    * every WinRT call is wrapped; any failure just means "no session"
    """
    updated = Signal(object)

    OWN_IDS = ("python", "vexa")          # never show / control ourselves

    def __init__(self):
        super().__init__()
        self.available = False
        self.error = ""
        self._loop = None
        self._thread = None
        self._stop = False
        self._wake = None
        self._session = None
        self._last_props = 0.0
        self._props = None                 # (title, artist, album, art_key)
        self._thumb_for = None
        self._last_emit = None
        self._errors = 0
        if not config.IS_WINDOWS or config.env_flag("VEXA_NO_SMTC"):
            self.error = "not on Windows" if not config.IS_WINDOWS else "disabled (VEXA_NO_SMTC)"
            return
        try:
            from winrt.windows.media.control import (  # noqa: F401
                GlobalSystemMediaTransportControlsSessionManager)
            from winrt.windows.storage.streams import Buffer, InputStreamOptions  # noqa: F401
        except Exception as e:
            self.error = f"winrt missing ({e.__class__.__name__}: {e})"
            return
        self.available = True
        self._thread = threading.Thread(target=self._run, name="vexa-smtc", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ thread
    def _run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._wake = asyncio.Event()
            self._loop.run_until_complete(self._main())
        except Exception as e:
            self.error = f"smtc thread died: {e}"
            print(f"[vexa] {self.error}")
            self.available = False

    async def _main(self):
        from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as Mgr
        mgr = None
        while not self._stop:
            delay = 2.5
            try:
                if mgr is None:
                    mgr = await Mgr.request_async()
                found = await self._poll(mgr)
                delay = 1.0 if found else 2.5
                self._errors = 0
            except Exception as e:
                self._errors += 1
                if self._errors in (1, 10, 100):
                    print(f"[vexa] smtc poll error ({self._errors}): {e!r}")
                self._session = None
                if self._errors % 5 == 0:
                    mgr = None              # re-acquire the manager now and then
                self._emit(None)
                delay = min(10.0, 2.0 + self._errors)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    def _pick(self, mgr):
        def ok(sess):
            try:
                aid = (sess.source_app_user_model_id or "").lower()
            except Exception:
                return False
            return not any(k in aid for k in self.OWN_IDS)
        cur = mgr.get_current_session()
        if cur is not None and ok(cur):
            return cur
        best = None
        try:
            for sess in mgr.get_sessions():
                if not ok(sess):
                    continue
                if best is None:
                    best = sess
                try:
                    if int(sess.get_playback_info().playback_status) == 4:
                        return sess
                except Exception:
                    pass
        except Exception:
            pass
        return best

    async def _poll(self, mgr) -> bool:
        sess = self._pick(mgr)
        if sess is None:
            self._session = None
            self._props = None
            self._emit(None)
            return False
        aid = sess.source_app_user_model_id or ""
        changed_session = self._session is None or getattr(self, "_aid", None) != aid
        self._session = sess
        self._aid = aid
        info = sess.get_playback_info()
        status = int(info.playback_status)           # 0 closed 1 opened 2 changing 3 stopped 4 playing 5 paused
        playing = status == 4
        shuffle = False
        repeat = 0
        try:
            shuffle = bool(info.is_shuffle_active) if info.is_shuffle_active is not None else False
        except Exception:
            pass
        try:
            rm = info.auto_repeat_mode
            rm = int(rm) if rm is not None else 0      # 0 none 1 track 2 list
            repeat = {0: 0, 1: 2, 2: 1}.get(rm, 0)      # -> local scheme (1 all, 2 one)
        except Exception:
            pass
        ctl = {}
        try:
            c = info.controls
            ctl = {"play_pause": bool(c.is_play_pause_toggle_enabled), "next": bool(c.is_next_enabled),
                   "prev": bool(c.is_previous_enabled), "shuffle": bool(c.is_shuffle_enabled),
                   "repeat": bool(c.is_repeat_enabled)}
        except Exception:
            pass
        # timeline (some apps only update it on seek/state change -> extrapolate)
        pos_ms = dur_ms = 0
        try:
            tl = sess.get_timeline_properties()
            pos_ms = int(tl.position.total_seconds() * 1000)
            dur_ms = int((tl.end_time - tl.start_time).total_seconds() * 1000)
            if playing:
                lu = tl.last_updated_time
                if lu is not None and lu.year > 2000:
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    if lu.tzinfo is None:
                        lu = lu.replace(tzinfo=datetime.timezone.utc)
                    pos_ms += int(max(0.0, (now_utc - lu).total_seconds()) * 1000)
            if dur_ms > 0:
                pos_ms = min(pos_ms, dur_ms)
        except Exception:
            pass
        # media properties (+ thumbnail when the track changes)
        now = time.monotonic()
        last = self._last_emit
        if (changed_session or self._props is None or now - self._last_props > 2.0
                or (last is not None and last.playing != playing)):
            self._last_props = now
            props = await sess.try_get_media_properties_async()
            title = (props.title or "").strip()
            artist = (props.artist or "").strip() or (props.album_artist or "").strip()
            album = (props.album_title or "").strip()
            art_key = self._props[3] if self._props else None
            ident = (aid, title, artist, album)
            if ident != self._thumb_for:
                self._thumb_for = ident
                art_key = None
                try:
                    if props.thumbnail is not None:
                        data = await self._read_thumb(props.thumbnail)
                        art_key = tags.register_art(data)
                except Exception as e:
                    print(f"[vexa] smtc thumbnail failed: {e!r}")
            self._props = (title, artist, album, art_key)
        title, artist, album, art_key = self._props
        np = NowPlaying(source="smtc", app=app_label(aid), app_id=aid, title=title, artist=artist,
                        album=album, art_key=art_key, playing=playing,
                        status={4: "playing", 5: "paused", 3: "stopped", 2: "changing"}.get(status, "closed"),
                        position_ms=pos_ms, duration_ms=max(0, dur_ms), stamp=time.monotonic(),
                        shuffle=shuffle, repeat=repeat, controls=ctl)
        self._emit(np)
        return True

    async def _read_thumb(self, ref) -> bytes | None:
        from winrt.windows.storage.streams import Buffer, InputStreamOptions
        stream = await ref.open_read_async()
        try:
            size = int(stream.size)
            if size <= 0 or size > 16_000_000:
                return None
            buf = Buffer(size)
            await stream.read_async(buf, buf.capacity, InputStreamOptions.READ_AHEAD)
            n = int(buf.length)
            return bytes(memoryview(buf)[:n])
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _emit(self, np):
        # only push when something visible changed (position is extrapolated in the GUI)
        last = self._last_emit
        if np is None and last is None:
            return
        if np is not None and last is not None:
            same = (np.display_key() == last.display_key() and np.shuffle == last.shuffle
                    and np.repeat == last.repeat and np.duration_ms == last.duration_ms
                    and abs(np.pos_at(np.stamp) - last.pos_at(np.stamp)) < 1500)
            if same:
                return
        self._last_emit = np
        self.updated.emit(np)

    # ------------------------------------------------------------ commands
    def command(self, name: str, *args) -> bool:
        if not self.available or self._loop is None:
            return False
        try:
            asyncio.run_coroutine_threadsafe(self._do(name, *args), self._loop)
            return True
        except Exception as e:
            print(f"[vexa] smtc command {name} failed: {e!r}")
            return False

    async def _do(self, name, *args):
        sess = self._session
        if sess is None:
            return
        try:
            if name == "toggle":
                await sess.try_toggle_play_pause_async()
            elif name == "play":
                await sess.try_play_async()
            elif name == "pause":
                await sess.try_pause_async()
            elif name == "next":
                await sess.try_skip_next_async()
            elif name == "prev":
                await sess.try_skip_previous_async()
            elif name == "shuffle":
                await sess.try_change_shuffle_active_async(bool(args[0]))
            elif name == "repeat":
                from winrt.windows.media import MediaPlaybackAutoRepeatMode as RM
                m = {0: RM.NONE, 1: RM.LIST, 2: RM.TRACK}[int(args[0]) % 3]
                await sess.try_change_auto_repeat_mode_async(m)
        except Exception as e:
            print(f"[vexa] smtc {name} failed: {e!r}")
        await asyncio.sleep(0.15)            # let the app update its state
        self._last_props = 0.0
        self._wake.set()

    def shutdown(self):
        self._stop = True
        if self._loop is not None and self._wake is not None:
            try:
                self._loop.call_soon_threadsafe(self._wake.set)
            except Exception:
                pass


class MediaSession:
    """Back-compat shim (stage 1 name): availability only."""

    def __init__(self):
        self.available = False
        if not config.IS_WINDOWS:
            return
        try:
            import winrt.windows.media.control  # noqa: F401
            self.available = True
        except Exception:
            self.available = False
