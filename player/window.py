"""
VexaWindow: the shaped, frameless, per-pixel-alpha player window.

Frame loop (QTimer, coarse):
  * 30 fps while audio is present or something animates (shutter, tray,
    ring swing, drag, dial readout); 8 fps "breathing" idle otherwise; the
    timer stops completely while minimised/hidden.
  * Only dirty regions are repainted: the visor every frame, the ring /
    tray / dials when they move, the full body only when the vein glow
    level actually changes.
Per frame work is: 1 rfft (1024), a handful of pixmap blits, one small
offscreen visor image. No per-pixel Python.
"""
from __future__ import annotations

import math
import os
import random
import time

from PySide6.QtCore import QPoint, QRect, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QCursor, QGuiApplication, QImage, QPainter, QRegion
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QWidget

from . import config
from .analysis import Analyzer
from .boot import N_BANDS, Boot
from .chips import MiniChip, MiniWindow, PlaylistChip
from .choker import Choker
from .dials import Dial
from .face import LEVELS as FACE_LEVELS
from .face import FaceDirector, FaceRig
from .glitch import Glitch
from .layers import Art
from .media import MediaHub, make_provider
from .panel import ChestPanel
from .audioctl import AudioRouter, OwnTarget, WinAudioBackend
from .qtutil import mix
from .sounds import SoundBank
from .sources import AudioHub, LocalPlayer
from .sparks import Sparks
from .visor import Visor

FADE_DUR = 0.24     # body relight cross-fade (runs while the shutter reopens)


class VexaWindow(QWidget):
    def __init__(self, source: str = config.SOURCE, files=None, use_settings: bool = True,
                 media_provider=None, boot: bool | None = None, audio_backend=None):
        super().__init__(None)
        self.setWindowTitle("Vexa")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.art = Art()
        self.setFixedSize(self.art.W, self.art.H)
        L = self.art.layout
        self.sounds = SoundBank(self)
        self.visor = Visor(self.art)
        self.panel = ChestPanel(L, self.sounds)
        self.ring = Choker(L)
        self.local = LocalPlayer(self, audible=not config.MUTE)
        self.hub = AudioHub(self.local, source)
        # dial routing: Vexa's own output / the SMTC app's session / master volume
        self.audio_backend = audio_backend if audio_backend is not None else WinAudioBackend()
        self.own_target = OwnTarget(self.audio_backend, lambda: self.local.volume, self.local.set_volume)
        self.router = AudioRouter(self.audio_backend, self.own_target)
        self._route_check_t = -1e9
        self._virtual_clock = False
        self._last_t = time.monotonic()
        # now-playing: local playback + Windows media session (SMTC)
        self.provider = media_provider if media_provider is not None else make_provider()
        self.media = MediaHub(self.local, self.provider, clock=self._now)
        self.an = Analyzer(20)
        # ear-slot chips (content offset ox/rx grows the window while they're out)
        self.ox = 0
        self.rx = 0
        es = L["ear_slots"]
        self.chip_pl = PlaylistChip("L", es["left"], self.sounds, L["axis_x"], self.art.W, self.art.hit_alpha)
        self.chip_pl.files_provider = lambda: (self.local.files, self.local.index)
        self.chip_mini = MiniChip("R", es["right"], self.sounds, L["axis_x"], self.art.W, self.art.hit_alpha)
        self.chips = (self.chip_pl, self.chip_mini)
        self._chip_rects = {}
        self._chip_press = None        # (chip, item) being pressed
        self._list_drag = None         # (start_y, start_scroll, moved)
        self.detached = False
        self.mini_win = None
        vx, vy, vr = L["dials"]["volume"]
        bx, by, br = L["dials"]["balance"]
        self.dials = {
            "volume": Dial("volume", vx, vy, vr, 0.0, 1.0, 0.7, 20, self._on_volume, self.sounds, self._now),
            "balance": Dial("balance", bx, by, br, -1.0, 1.0, 0.0, 20, self._on_balance, self.sounds, self._now),
        }

        self.mode_i = 0
        self.fade_from = None
        self.fade_t0 = 0.0
        self.glow = 0.2
        self._drawn_glow = -1.0
        self.muted = False
        self.on_top = False
        self._drag_off = None
        self._active_dial = None
        self._press_item = None
        self._lcd_key = None
        self.frame_count = 0
        self._vein_dirty = self.art.vein_rect.adjusted(-4, -4, 4, 4)

        # ---- stage 3: life (reactions, sparks, boot, sleep, glitches)
        self.face = FaceRig(self.art)
        self.director = FaceDirector()
        self._face_requested = False
        self._mouth_lvl = 0.0
        self._face_drawn = False
        self._pending_trigger = None
        self._last_ident = None
        self._hover_since = None
        self._hover_fired = False
        self._press_global = None
        self._press_body = None
        self.sparks = Sparks(L["wire_tips"])
        self._sparks_was = False
        self.glitch = Glitch()
        self._glitch_was = False
        self._gl_img = None
        self.boot = None
        self._boot_pending = config.BOOT if boot is None else bool(boot)
        self.sleeping = False
        self.sleep_k = 0.0
        self.last_active_t = None
        ax = int(round(L["axis_x"]))
        col = self.art.hit_alpha[:, ax]
        head_top = int((col >= 128).argmax())
        vr = self.visor.rect
        self.forehead_zone = QRect(ax - 50, head_top + 8, 100, max(10, vr.y() - head_top - 14))
        self.face_zone = QRect(ax - 44, vr.bottom() + 3, 88, 296 - vr.bottom())
        self.head_rect = QRect(ax - 82, max(0, head_top - 4), 164, 300 - head_top)

        self.settings = QSettings("Vexa", "Vexa") if use_settings and not config.env_flag("VEXA_NO_SETTINGS") else None
        self._load_settings()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.timeout.connect(self.tick)
        self._interval = 0
        self._set_interval(1000 // config.FPS_ACTIVE)

        self.local.stateChanged.connect(self._sync_panel)
        # the Windows audio session only exists once playback starts
        self.local.stateChanged.connect(self._on_local_state)
        self.local.trackChanged.connect(self._on_track)
        self.local.stateChanged.connect(lambda: self.local.is_playing() and self.wake())
        self.media.changed.connect(self._on_media_changed)
        QTimer.singleShot(400, self.art.warm)   # pre-build other modes when idle
        if files:
            self.local.set_files(files)
        print(f"[vexa] {self.hub.describe()} mute={config.MUTE} "
              f"balance={'session' if self.audio_backend.ok else 'visual-only'} "
              f"smtc={'yes' if getattr(self.provider, 'available', False) else 'no'}"
              f"{'' if getattr(self.provider, 'available', False) else ' (' + str(getattr(self.provider, 'error', '')) + ')'}")

    # ================================================================ settings
    def _load_settings(self):
        s = self.settings
        if s is None:
            return
        try:
            self.mode_i = self.visor.mode = int(s.value("mode", 0)) % len(config.MODES)
            self.dials["volume"].value = float(s.value("volume", 0.7))
            self.dials["balance"].value = float(s.value("balance", 0.0))
            self.local.set_volume(self.dials["volume"].value)
            self.own_target.set_balance(self.dials["balance"].value)
            if s.value("panel_pinned", "false") in (True, "true"):
                self.set_panel_pinned(True, save=False)
            if s.value("on_top", "false") in (True, "true"):
                self.set_on_top(True)
            pos = s.value("pos")
            # only restore if that spot is still on a connected monitor
            if isinstance(pos, QPoint) and QGuiApplication.screenAt(
                    pos + QPoint(self.width() // 2, self.height() // 3)) is not None:
                self.move(pos)
        except Exception:
            pass

    def _save_settings(self):
        s = self.settings
        if s is None:
            return
        s.setValue("mode", self.mode_i)
        # only Vexa's own levels are ours to remember (other apps keep theirs)
        s.setValue("volume", self.local.volume)
        s.setValue("balance", self.own_target.bal)
        s.setValue("panel_pinned", self.panel.pinned)
        s.setValue("on_top", self.on_top)
        s.setValue("pos", self.pos() + QPoint(self.ox, 0))   # body position, margins excluded

    # ================================================================ timing
    def _set_interval(self, ms):
        if ms != self._interval:
            self._interval = ms
            self.timer.setInterval(ms)

    def showEvent(self, e):
        self._last_t = time.monotonic()
        self.timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        if not self.detached:          # detached mini player still needs the clock + analysis
            self.timer.stop()
        super().hideEvent(e)

    def changeEvent(self, e):
        if e.type() == e.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMinimized and not self.detached:
                self.timer.stop()
            elif self.isVisible():
                self.timer.start()
        super().changeEvent(e)

    def tick(self, now: float | None = None, dt: float | None = None):
        """One animation step. `now`/`dt` may be forced by tests."""
        t = time.monotonic() if now is None else now
        self._virtual_clock = now is not None
        if dt is None:
            dt = t - self._last_t
        self._last_t = t
        dt = max(0.001, min(dt, 0.25))
        self.frame_count += 1

        samples, sr = self.hub.latest(1024)
        self.an.process(samples, sr, dt)
        audio = not self.an.silent

        # ---- boot sequence
        if self._boot_pending:
            self._boot_pending = False
            self.boot = Boot(t)
        booting = False
        if self.boot is not None:
            for kind, arg in self.boot.step(t):
                if kind == "sound":
                    self.sounds.play(arg)
                elif kind == "tray":
                    self.panel.poke(t)
            self.panel.lcd_test = self.boot.lcd(t)
            self.visor.eye_life.force = self.boot.eyes(t)
            if self.boot.done:
                self._end_boot(t)
            else:
                booting = True
        if not booting and not self._face_requested:
            self._face_requested = True
            order = [self.mode_i] + [i for i in range(len(config.MODES)) if i != self.mode_i]
            for i in order:
                self.face.request(i)
        if self.face.pending:
            self.face.poll()

        # ---- sleep: nothing has played for a while
        np_ = self.media.now()
        playing = audio or np_.playing
        if self.last_active_t is None:
            self.last_active_t = t
        if playing:
            self.last_active_t = t
            if self.sleeping:
                self.wake(t)
        elif (not self.sleeping and not booting and not self.detached
              and t - self.last_active_t > config.SLEEP_AFTER_S):
            self.go_sleep(t)
        old_k = self.sleep_k
        if self.sleeping:
            self.sleep_k = min(1.0, self.sleep_k + dt / 2.0)      # slow power-down
        else:
            self.sleep_k = max(0.0, self.sleep_k - dt / 0.3)      # wakes fast
        self.visor.eye_life.sleep_k = self.sleep_k
        sleep_moving = self.sleep_k != old_k

        # veins: bass pulse while playing, slow breathing when idle, slower asleep
        if audio:
            target = 0.16 + 0.74 * self.an.bass
        else:
            target = 0.16 + 0.07 * (0.5 + 0.5 * math.sin(t * 2 * math.pi * 0.2))
        if self.sleep_k > 0:
            asleep = 0.05 + 0.15 * (0.5 + 0.5 * math.sin(t * 2 * math.pi * 0.085))
            target = target * (1 - self.sleep_k) + asleep * self.sleep_k
        self.glow = target

        # ---- facial reactions
        ident = (np_.title, np_.artist) if np_.title else None
        if ident != self._last_ident:
            if self._last_ident is not None and ident is not None:
                self._pending_trigger = "track"
            self._last_ident = ident
        if not booting and self.sleep_k < 0.01:
            trig = self.director.feed(t, dt, self.an.bass, audio)
            trig = trig or self._pending_trigger
            if trig:
                self._try_react(trig, t)
        self._pending_trigger = None
        self._mouth_lvl = self.director.mouth_level(t) if self.director.active else 0.0

        # ---- sparks + rare glitch
        if not booting:
            if self.sparks.maybe_burst(t, dt, self.an.bass, audio, self.sleeping):
                self.sounds.play("crackle")
            if audio and self.sleep_k == 0 and self.glitch.maybe_corrupt(t, dt):
                self.sounds.play("static")
        self.sparks.step(dt)
        glitching = self.glitch.active(t)

        # choker physics from the window's motion
        self.ring.feed_window_pos(self.x() + self.ox, self.y(), dt)
        self.ring.step(dt)

        # visor (shutter may flip the mode)
        old_mode = self.mode_i
        if self.visor.step(t, dt, self.an):
            self.fade_from = old_mode
            self.fade_t0 = t
            self.mode_i = self.visor.mode
        fading = self.fade_from is not None
        if fading and t - self.fade_t0 >= FADE_DUR:
            self.fade_from = None

        # chest tray
        cur = self.mapFromGlobal(QCursor.pos()) - QPoint(self.ox, 0)
        # the tray stays out while the cursor rests on it; moving over the
        # chest re-pokes it (mouseMoveEvent); otherwise it retracts after idle
        inside = self.underMouse() and self.panel.contains(cur.x(), cur.y())
        tray_moved = self.panel.update(t, inside)
        if abs(t - self._route_check_t) > 0.5:        # dial target follows the source
            self._route_check_t = t
            self._retarget(t)
        lcd_changed = self._sync_panel()

        # ear chips
        self.chip_mini.step_led(self.an, dt)
        chip_dirty = []
        for c in self.chips:
            was_vis = c.motion.visible
            moved = c.update(t)
            if moved or c.motion.visible != was_vis:
                self._update_margins()
            if c.motion.visible:
                k = c.frame_key(np_, self.an, t)     # repaint only on visible change
                if moved or k != getattr(c, "_fkey", None):
                    c._fkey = k
                    chip_dirty.append(c)
            elif was_vis:
                chip_dirty.append(c)
        if self.detached and self.mini_win is not None:
            k = self.chip_mini.frame_key(np_, self.an, t)
            if k != getattr(self.mini_win, "_fkey", None):
                self.mini_win._fkey = k
                self.mini_win.refresh()

        # ---- dirty regions (Windows' layered-window update cost scales with area)
        reg = QRegion()
        if fading or booting or glitching or self._glitch_was:
            self.update()                       # whole body is being relit / booting / glitch
            for c in chip_dirty:
                self._chip_dirty(c)
        else:
            if abs(self.glow - self._drawn_glow) > 0.006:
                reg += self._vein_dirty         # veins live on head/neck/chest only
            reg += self.visor.rect.adjusted(-3, -3, 3, 3)
            if self.ring.moving:
                reg += self.ring.rect()
            if tray_moved or (lcd_changed and self.panel.visible):
                reg += self.panel.dirty_rect()
            ending = self._dial_readout_ending()
            for d in self.dials.values():
                if d.readout_visible() or d.dragging or d.springing or ending:
                    reg += d.rect()
            for c in chip_dirty:
                reg += self._chip_dirty(c)
            if self._mouth_lvl > 0 or self._face_drawn:
                reg += self.face.rect
            if self.sparks.active or self._sparks_was:
                reg += self.sparks.dirty_rect()
            self._update_body(reg)
        self._sparks_was = self.sparks.active
        self._glitch_was = glitching

        busy = (audio or self.visor.transitioning or fading or self.panel.animating
                or any(c.animating(np_, self.an) for c in self.chips) or self.detached
                or self.ring.moving or self._drag_off is not None or self._active_dial is not None
                or any(d.springing for d in self.dials.values())
                or any(d.readout_visible() for d in self.dials.values())
                or (self.panel.state == "out" and not self.panel.pinned) or booting or glitching or self.sparks.active
                or self.director.active or self.visor.eye_life.reacting or self.visor.eye_life.flickering
                or sleep_moving)
        if busy:
            fps = config.FPS_ACTIVE
        elif self.sleep_k >= 0.999:
            fps = config.FPS_SLEEP
        else:
            fps = config.FPS_IDLE
        self._set_interval(1000 // fps)

    def _dial_readout_ending(self):
        vis = tuple(d.readout_visible() for d in self.dials.values())
        changed = getattr(self, "_ro_vis", None) != vis
        self._ro_vis = vis
        return changed

    # ================================================================ life
    def _end_boot(self, t=None):
        self.boot = None
        self._boot_pending = False
        self.panel.lcd_test = None
        self.visor.eye_life.force = None
        self.last_active_t = self._now() if t is None else t
        self.update()

    def skip_boot(self):
        """Click during the boot sequence: jump straight to 'ready'."""
        if self.boot is None and not self._boot_pending:
            return
        for name in ("boot_hum", "modem", "crt_on"):
            self.sounds.stop(name)
        self.sounds.play("latch")
        if self.boot is not None:
            self.boot.finish()
        self._end_boot()

    @property
    def booting(self) -> bool:
        return self.boot is not None or self._boot_pending

    def go_sleep(self, t=None):
        if self.sleeping:
            return
        self.sleeping = True
        self.director.cancel()
        self.sounds.play("sleep")
        if self.panel.state in ("out", "extending") and not self.panel.pressed and not self.panel.pinned:
            self.panel._retract(self._now() if t is None else t)     # tuck the tray away

    def wake(self, t=None):
        if not self.sleeping:
            return
        t = self._now() if t is None else t
        self.sleeping = False
        self.last_active_t = t
        self.sounds.play("wake")
        self._set_interval(1000 // config.FPS_ACTIVE)
        if not self.timer.isActive() and self.isVisible() and not self._virtual_clock:
            self.timer.start()

    def _eyes_visible(self) -> bool:
        return self.mode_i == 3 and not self.visor.transitioning and self.fade_from is None

    def _try_react(self, trigger: str, t: float) -> bool:
        d = self.director
        if not d.can_react(t) or not d.roll(trigger):
            return False
        kind = d.choose(trigger, self._eyes_visible(), lambda m: self.face.ready(self.mode_i, m))
        if kind is None:
            return False
        self._start_reaction(kind, t)
        return True

    def _start_reaction(self, kind, t):
        self.director.start(kind, t)
        ek = self.director.eye_kind()
        if ek:
            self.visor.eye_life.react(ek)
        self._set_interval(1000 // config.FPS_ACTIVE)

    def react(self, kind: str, now: float | None = None):
        """Force a reaction now (tests / debugging); builds frames if needed."""
        from .face import PLAN
        now = self._now() if now is None else now
        m = PLAN[kind][0]
        if m:
            self.face.frames(self.mode_i, m)
        self.director.cancel()
        self._start_reaction(kind, now)

    def data_tear(self):
        """Forehead click easter egg."""
        now = self._now()
        self.glitch.start_tear(now)
        self.sounds.play("glitch")
        self._set_interval(1000 // config.FPS_ACTIVE)
        self.update()

    def _body_click(self, x, y):
        """A click (press + release without dragging) on her body."""
        if self.forehead_zone.contains(x, y) and self.art.opaque_at(x, y):
            self.data_tear()
        elif self.face_zone.contains(x, y) and self.art.opaque_at(x, y):
            self._pending_trigger = "click"

    def _sync_panel(self, *_) -> bool:
        pnl = self.panel
        np_ = self.media.now()
        if np_.source == "smtc":
            pnl.playing = np_.playing
            pnl.shuffle = np_.shuffle
            pnl.repeat = np_.repeat
            pnl.elapsed = np_.pos_at(self._now())
            pnl.total = np_.duration_ms
        else:
            pnl.playing = self.local.is_playing()
            pnl.shuffle = self.local.shuffle
            pnl.repeat = self.local.repeat
            pnl.elapsed = self.local.position_ms()
            pnl.total = self.local.duration_ms()
        key = (pnl.playing, pnl.shuffle, pnl.repeat, pnl.elapsed // 1000, pnl.total // 1000)
        changed = key != self._lcd_key
        self._lcd_key = key
        return changed

    def _on_track(self):
        self._sync_panel()
        if self.local.index >= 0:
            pm, geo = self.chip_pl.frame(self.mode_i, config.MODES[self.mode_i])
            self.chip_pl.ensure_visible(geo, self.local.index)

    def _on_media_changed(self):
        # new track / art / state: restart marquees and repaint visible chips
        now = self._now()
        for c in self.chips:
            c.t_text = now
            if c.motion.visible:
                self._update_body(QRegion(self._chip_dirty(c)))
        self._sync_panel()
        self.update(self._to_win(self.panel.dirty_rect()))
        if self.mini_win is not None:
            self.mini_win.update()

    # ================================================================ chips + margins
    def _to_win(self, r: QRect) -> QRect:
        return r.translated(self.ox, 0)

    def _update_body(self, reg: QRegion):
        if self.ox:
            reg = reg.translated(self.ox, 0)
        self.update(reg)

    def _chip_dirty(self, c) -> QRect:
        """Current + previous chip rect (body coords) to repaint."""
        pm, geo = c.frame(self.mode_i, config.MODES[self.mode_i])
        r = c.rect(geo)
        old = self._chip_rects.get(id(c))
        self._chip_rects[id(c)] = r
        u = r.united(old) if old is not None else r
        return u

    def _needed_margins(self):
        left = right = 0
        mode = config.MODES[self.mode_i]
        if self.chip_pl.motion.visible:
            pm, geo = self.chip_pl.frame(self.mode_i, mode)
            l, _ = self.chip_pl.extents_out(geo)
            left = max(0, -l)
        if self.chip_mini.motion.visible:
            pm, geo = self.chip_mini.frame(self.mode_i, mode)
            _, r = self.chip_mini.extents_out(geo)
            right = max(0, r - self.art.W)
        return left, right

    def _update_margins(self):
        self._set_margins(*self._needed_margins())

    def _set_margins(self, left: int, right: int):
        """Grow/shrink the window around the body without moving her on screen."""
        if (left, right) == (self.ox, self.rx):
            return
        dx = left - self.ox
        g = self.geometry()
        w = self.art.W + left + right
        self.ox, self.rx = left, right
        self.setMinimumSize(1, 1)
        self.setMaximumSize(16777215, 16777215)
        self.setGeometry(g.x() - dx, g.y(), w, self.art.H)
        self.setFixedSize(w, self.art.H)
        if self._drag_off is not None:
            self._drag_off += QPoint(dx, 0)
        self._chip_rects.clear()
        self.update()

    def toggle_chip(self, c, by_button=True):
        if c is self.chip_mini and self.detached:
            return
        now = self._now()
        if c.motion.toggle(now):
            if c.motion.state == "ejecting":
                self._update_margins()        # grow before it pokes out
                c.t_text = now
                if c is self.chip_pl and self.local.index >= 0:
                    pm, geo = c.frame(self.mode_i, config.MODES[self.mode_i])
                    c.ensure_visible(geo, self.local.index)
            self._set_interval(1000 // config.FPS_ACTIVE)
            if not self.timer.isActive() and self.isVisible() and not self._virtual_clock:
                self.timer.start()

    def detach_mini(self):
        """Pop the mini-player chip off into its own floating window; Vexa hides."""
        c = self.chip_mini
        if self.detached or c.motion.state != "out":
            return
        mode = config.MODES[self.mode_i]
        pm, geo = c.frame(self.mode_i, mode)
        o = c.pixmap_origin(geo)
        gpos = self.mapToGlobal(QPoint(int(o.x()) + self.ox, int(o.y())))
        self.sounds.play("detach")
        self.detached = True
        c.detached = True
        c.hover_item = c.pressed_item = None
        self.mini_win = MiniWindow(c, self)
        self.mini_win.dockRequested.connect(self.dock_mini)
        self.mini_win.commandRequested.connect(self._mini_command)
        if self.on_top:
            self.mini_win.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.mini_win.move(gpos + QPoint(14 if c.side == "R" else -14, -8))   # pops off the slot
        c.motion.set_in()
        self._update_margins()
        self.mini_win.show()
        self.hide()

    def dock_mini(self):
        if not self.detached:
            return
        self.sounds.play("dock")
        self.detached = False
        c = self.chip_mini
        c.detached = False
        c.hover_item = c.pressed_item = None
        c.motion.set_out()                     # plugs back in, seated out
        c.t_text = self._now()
        self._update_margins()
        w = self.mini_win
        self.mini_win = None
        if w is not None:
            w._docking = True
            w.close()
            w.deleteLater()
        self.show()
        self.raise_()
        self.activateWindow()

    def _mini_command(self, name):
        self.button_action({"play": "play", "prev": "prev", "next": "next"}.get(name, name))
        if self.mini_win is not None:
            self.mini_win.update()

    # ================================================================ painting
    def paintEvent(self, e):
        if self.glitch.active(self._last_t):
            # glitch frames: compose offscreen, tear it up, then blit the lot
            if self._gl_img is None or self._gl_img.size() != self.size():
                self._gl_img = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
            self._gl_img.fill(0)
            q = QPainter(self._gl_img)
            self._paint_content(q)
            q.end()
            self.glitch.process(self._gl_img, self._last_t, self.head_rect.translated(self.ox, 0),
                                config.MODES[self.mode_i].primary)
            p = QPainter(self)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            p.drawImage(0, 0, self._gl_img)
            p.end()
            return
        p = QPainter(self)
        self._paint_content(p)
        p.end()

    def _face_layers(self, mi):
        """-> (body [(pm, opacity)], veins [(pm, weight)], base_vein_weight) for the
        mouth patch, or None when no reaction is showing."""
        lvl = self._mouth_lvl
        kind = self.director.mouth
        if lvl <= 0.001 or kind is None or self.fade_from is not None or self.booting:
            return None
        if not self.face.ready(mi, kind):
            return None
        body, veins = self.face.frames(mi, kind)
        f = min(1.0, lvl) * FACE_LEVELS
        i = int(f)
        if i == 0:
            return [(body[0], f)], [(veins[0], f)], 1.0 - f
        if i >= FACE_LEVELS:
            return [(body[-1], 1.0)], [(veins[-1], 1.0)], 0.0
        fr = f - i
        return [(body[i - 1], 1.0), (body[i], fr)], [(veins[i - 1], 1 - fr), (veins[i], fr)], 0.0

    def _paint_content(self, p: QPainter):
        mi = self.mode_i
        t = self._last_t
        if self.ox:
            p.translate(self.ox, 0)
        # chips slide out of the slits in the rails on the sides of her head
        # (clipped to the outside of the slit), rails on top, body on top of that
        np_ = None
        for c in self.chips:
            if c.motion.visible:
                if np_ is None:
                    np_ = self.media.now()
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                c.paint(p, mi, config.MODES[mi], np_, self.an, self._last_t)
                p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for c in self.chips:
            c.paint_rail(p, mi, config.MODES[mi])
        if self.booting:
            # unpowered body; the neon relight comes up with the power
            power = self.boot.power(t) if self.boot is not None else 0.0
            p.drawPixmap(0, 0, self.art.unlit())
            p.setOpacity(power ** 1.5)
            p.drawPixmap(0, 0, self.art.body(mi))
            p.setOpacity(1.0)
        else:
            p.drawPixmap(0, 0, self.art.body(mi))
        face = self._face_layers(mi)
        fr = self.face.rect
        if face is not None:
            for pm, op in face[0]:
                p.setOpacity(op)
                p.drawPixmap(fr.topLeft(), pm)
            p.setOpacity(1.0)
        self._face_drawn = face is not None
        vein_from = None
        if self.fade_from is not None:
            k = min(1.0, (t - self.fade_t0) / FADE_DUR)
            # old relight fades out on top of the new one
            p.setOpacity(1.0 - k)
            p.drawPixmap(0, 0, self.art.body(self.fade_from))
            p.setOpacity(1.0)
            vein_from = (self.fade_from, 1.0 - k)
        mode = config.MODES[mi]
        if self.booting:
            power = self.boot.power(t) if self.boot is not None else 0.0
            self.panel.paint_slot(p, mi)
            self.ring.paint(p, mi)
            for d in self.dials.values():
                d.paint(p, mi)
            # unpowered: everything so far sinks into the dark (shape kept)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceAtop)
            p.fillRect(QRect(-self.ox, 0, self.width(), self.height()),
                       QColor(0, 0, 0, int(225 * (1 - power) ** 0.8)))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            if self.boot is not None:
                pm, off = self.art.veins(mi)
                bh = (pm.height() + N_BANDS - 1) // N_BANDS
                for i in range(N_BANDS):
                    op = self.boot.band(i, t, self.glow)
                    if op > 0.003:
                        p.setOpacity(min(1.0, op))
                        p.drawPixmap(QPoint(off.x(), off.y() + i * bh), pm, QRect(0, i * bh, pm.width(), bh))
            p.setOpacity(1.0)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            self._drawn_glow = self.glow
            self.panel.paint(p, mi)
        else:
            power = 1.0
            self._paint_veins(p, mi, vein_from, face)
            self.panel.paint_slot(p, mi)
            self.panel.paint(p, mi)
            self.ring.paint(p, mi)
            for d in self.dials.values():
                d.paint(p, mi)
        self.sparks.paint(p, mode)
        fx = {}
        if self.sleep_k > 0:
            fx["sleep"] = self.sleep_k
        if self.booting:
            v = self.boot.visor(t) if self.boot is not None else (0.0, 0.0, 0.0)
            if v is not None:
                fx["boot"] = v
        img = self.visor.render(self.an, t, fx or None)
        p.drawImage(self.visor.rect.topLeft(), img)
        if power < 1.0:
            p.setOpacity(0.35 + 0.65 * power)
        self.visor.paint_frame(p, mi)
        p.setOpacity(1.0)
        for d in self.dials.values():
            d.paint_readout(p, mi)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for c in self.chips:
            c.paint_button(p)

    def _paint_veins(self, p, mi, vein_from, face):
        """Bass-driven vein glow (additive). With a mouth reaction showing, the
        veins inside the mouth rect come from the warped vein frames."""
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        pm, off = self.art.veins(mi)
        fr = self.face.rect
        if face is not None:
            full = QRegion(QRect(-self.ox, 0, self.width(), self.height()))
            p.setClipRegion(full.subtracted(QRegion(fr)))
        if vein_from:
            pmf, offf = self.art.veins(vein_from[0])
            p.setOpacity(self.glow * vein_from[1])
            p.drawPixmap(offf, pmf)
            p.setOpacity(self.glow * (1 - vein_from[1]))
        else:
            p.setOpacity(self.glow)
        p.drawPixmap(off, pm)
        if face is not None:
            p.setClipRect(fr)
            base_w = face[2]
            if base_w > 0.001:
                p.setOpacity(self.glow * base_w)
                p.drawPixmap(off, pm)
            p.setClipping(False)
            for vpm, wgt in face[1]:
                if wgt > 0.001:
                    p.setOpacity(self.glow * wgt)
                    p.drawPixmap(fr.topLeft(), vpm)
        p.setOpacity(1.0)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        self._drawn_glow = self.glow

    # ================================================================ actions
    def cycle_mode(self, step: int = 1):
        target = (self.visor.mode + step) % len(config.MODES)
        self.set_mode(target)

    def set_mode(self, i: int):
        if i == self.visor.mode or self.visor.transitioning:
            return
        if self.visor.start_transition(i, self._now()):
            self.sounds.play("shutter")
            self._set_interval(1000 // config.FPS_ACTIVE)

    def _now(self):
        # tests drive a virtual clock through tick(now=...)
        return self._last_t if self._virtual_clock else time.monotonic()

    def set_mode_immediate(self, i: int):
        self.visor.mode = self.mode_i = i % len(config.MODES)
        self.update()

    # ---- dials -> whatever is producing the sound (see audioctl.py)
    def _vol_route(self):
        r = self.media.route()
        if r == "local":
            return ("local",)
        if r == "smtc" and self.media.smtc is not None:
            return ("smtc", self.media.smtc.app_id or self.media.smtc.app or "")
        if self.hub.mode in ("auto", "loopback") and getattr(self.hub.loopback, "available", False):
            return ("system",)
        return ("local",)

    def _retarget(self, now=None, refresh=False):
        """Re-resolve the dial target when the source changes (or a fallback
        target is stale); the knobs then jump to the target's actual current
        volume / balance."""
        now = self._now() if now is None else now
        key = self._vol_route()
        if key == self.router.route and not refresh and not self.router.needs_refresh(now):
            return False
        self.router.resolve(key, now)
        self._sync_dials_from_target()
        return True

    def _sync_dials_from_target(self):
        tg = self.router.target
        vd, bd = self.dials["volume"], self.dials["balance"]
        vd.tag = tg.tag
        if self._active_dial is None:
            v = tg.volume()
            if v is not None:
                vd.value = max(0.0, min(1.0, float(v)))
            b = tg.balance()
            if b is not None:
                bd.value = max(-1.0, min(1.0, float(b)))
        for d in (vd, bd):
            self.update(self._to_win(d.rect()))

    def _dial_target(self):
        """Target for a dial gesture: refresh a stale fallback first (the app's
        session may have appeared since, e.g. playback resumed)."""
        now = self._now()
        r = self.router
        if r.route[0] != "local" and r.target.fallback and now - r.t_resolved > 1.0 \
                and self._active_dial is None:
            self._retarget(now, refresh=True)
        return r.target

    def _on_volume(self, v):
        tg = self.router.target
        if not tg.set_volume(v) and tg.kind != "own":
            tg = self.router.resolve(self._vol_route(), self._now())
            self.dials["volume"].tag = tg.tag
            tg.set_volume(v)

    def _on_balance(self, v):
        tg = self.router.target
        if tg.kind == "own":
            tg.set_balance(v)            # may be pending until our session exists
        elif not tg.set_balance(v):
            tg = self.router.resolve(self._vol_route(), self._now())
            self.dials["volume"].tag = tg.tag
            tg.set_balance(v)

    def _on_local_state(self, *_):
        if self.local.is_playing():
            # the Windows session appears a moment after playback starts
            self.own_target.apply()
            for ms in (300, 1000, 2500):
                QTimer.singleShot(ms, lambda: self.own_target.applied or self.own_target.apply())
        self._retarget()

    # ---- chest tray pinned out
    def set_panel_pinned(self, on: bool, save=True):
        self.panel.set_pinned(bool(on), self._now())
        self._set_interval(1000 // config.FPS_ACTIVE)
        if not self.timer.isActive() and self.isVisible() and not self._virtual_clock:
            self.timer.start()
        if save:
            self._save_settings()

    def button_action(self, name: str):
        """Chest row / mini player / keys. Local playback while it plays,
        otherwise the Windows media session (Spotify, YouTube Music...)."""
        route = self.media.route()
        if route == "none" and name == "play":
            self.open_files()
        else:
            # (with nothing loaded, shuffle/repeat still set the local preferences)
            self.media.control("toggle" if name == "play" else name)
        self._sync_panel()
        self.update(self._to_win(self.panel.dirty_rect()))

    def open_files(self):
        start = os.path.expanduser("~/Music")
        files, _ = QFileDialog.getOpenFileNames(
            self, "Vexa - open audio files", start,
            "Audio (*.mp3 *.flac *.wav *.m4a *.ogg *.opus *.aac);;All files (*)")
        if files:
            self.local.set_files(files)

    def play_index(self, i: int):
        lp = self.local
        if 0 <= i < len(lp.files):
            lp._load(i)
            lp.play()

    def set_muted(self, m: bool):
        self.muted = m
        self.local.set_muted(m)
        self.sounds.set_muted(m)

    def set_on_top(self, on: bool):
        self.on_top = on
        vis = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        if vis:
            self.show()

    # ================================================================ input
    def _visor_hit(self, x, y) -> bool:
        r = self.visor.rect
        if not r.contains(x, y):
            return False
        return self.art.visor_mask_arr[y - r.y(), x - r.x()] > 0.5

    def _dial_at(self, x, y):
        for d in self.dials.values():
            if d.hit(x, y):
                return d
        return None

    def _pos(self, e):
        """Event position in body coordinates."""
        return int(e.position().x()) - self.ox, int(e.position().y())

    def _chip_hit(self, x, y):
        """-> (chip, item) for a point over a visible chip's card, else None.
        item: 'row:<i>' / 'list' (playlist), a button name (mini), or 'body'."""
        mode = config.MODES[self.mode_i]
        for c in self.chips:
            if not c.motion.visible:
                continue
            pm, geo = c.frame(self.mode_i, mode)
            if not c.card_rect(geo).contains(x, y) or not c.visible_x(x):
                continue
            if c is self.chip_pl and c.motion.state == "out":
                i = c.row_at(geo, x, y)
                if i is not None:
                    return c, f"row:{i}"
                if c.list_rect_body(geo).contains(x, y):
                    return c, "list"
            if c is self.chip_mini and c.motion.state == "out":
                it = c.item_at(geo, x, y)
                if it:
                    return c, it
            return c, "body"
        return None

    def mouseDoubleClickEvent(self, e):
        """Double-click a shoulder knob: spring it back to its default
        (volume -> startup default, balance -> centre). The first click of
        the pair only pressed the knob (a press never moves it), so nothing
        jumps; the second press is handled here instead of starting a drag."""
        x, y = self._pos(e)
        if e.button() == Qt.MouseButton.LeftButton and not self.booting:
            d = self._dial_at(x, y)
            if d is not None:
                if self._active_dial is not None:
                    self._active_dial.end_drag()
                    self._active_dial = None
                self._dial_target()
                d.reset()
                self.update(self._to_win(d.rect()))
                return
        self.mousePressEvent(e)

    def mousePressEvent(self, e):
        x, y = self._pos(e)
        if e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        if self.booting:
            self.skip_boot()
            return
        if self.sleeping or self.sleep_k > 0.3:
            self.wake()                    # a click only wakes her
            return
        self.last_active_t = self._now()
        # red eject buttons on the ear slots
        for c in self.chips:
            if c.hit_button(x, y) and not (c is self.chip_mini and self.detached):
                c.button_down = True
                self._chip_press = (c, "eject")
                self.sounds.play("press") if c.motion.state in ("out", "ejecting") else None
                self.update(self._to_win(c.button.adjusted(-6, -6, 6, 6)))
                return
        hit = self._chip_hit(x, y)
        if hit is not None and hit[1] != "body":
            c, it = hit
            if it.startswith("row:") or it == "list":
                self._list_drag = (y, c.scroll, False, it)
                return
            c.pressed_item = it
            self._chip_press = (c, it)
            self.sounds.play("press")
            self._update_body(QRegion(self._chip_dirty(c)))
            return
        item = self.panel.item_at(x, y)
        if item:
            self.panel.pressed = item
            self._press_item = item
            self.panel.poke(self._now())
            self.sounds.play("press")
            self.update(self._to_win(self.panel.dirty_rect()))
            return
        if self._visor_hit(x, y):
            self.cycle_mode()
            return
        d = self._dial_at(x, y)
        if d:
            self._dial_target()
            self._active_dial = d
            d.begin_drag(x, y)
            self.update(self._to_win(d.rect()))
            return
        if self.panel.contains(x, y):
            return   # tray body: not draggable, avoids accidental moves
        # anywhere else on her (or on a chip's housing) drags the window;
        # a press+release without moving is a "click" on her (face / forehead)
        self._drag_off = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        self._press_global = e.globalPosition().toPoint()
        self._press_body = (x, y)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        x, y = self._pos(e)
        now = self._now()
        if self._drag_off is not None:
            self.move(e.globalPosition().toPoint() - self._drag_off)
            return
        if self._list_drag is not None:
            y0, s0, moved, it = self._list_drag
            if moved or abs(y - y0) > 3:
                c = self.chip_pl
                pm, geo = c.frame(self.mode_i, config.MODES[self.mode_i])
                c.scroll = s0 - (y - y0)
                c.wheel(geo, 0)                       # clamp
                self._list_drag = (y0, s0, True, it)
                self._update_body(QRegion(self._chip_dirty(c)))
            return
        if self._active_dial is not None:
            self._active_dial.drag_to(x, y)
            self.update(self._to_win(self._active_dial.rect()))
            return
        if self.panel.hover_zone.contains(x, y) or self.panel.contains(x, y):
            self.panel.poke(now)
        # lingering on her face now and then earns a look
        if self.face_zone.contains(x, y):
            if self._hover_since is None:
                self._hover_since = now
            elif not self._hover_fired and now - self._hover_since > 1.2:
                self._hover_fired = True
                self._pending_trigger = self._pending_trigger or "hover"
        else:
            self._hover_since = None
            self._hover_fired = False
        hov = self.panel.item_at(x, y)
        if hov != self.panel.hover:
            self.panel.hover = hov
            self.update(self._to_win(self.panel.dirty_rect()))
        for d in self.dials.values():
            h = d.hit(x, y)
            if h != d.hover:
                d.hover = h
                self.update(self._to_win(d.rect()))
        over_btn = False
        for c in self.chips:
            h = c.hit_button(x, y) and not (c is self.chip_mini and self.detached)
            over_btn = over_btn or h
            if h != c.button_hover:
                c.button_hover = h
                self.update(self._to_win(c.button.adjusted(-6, -6, 6, 6)))
        hit = self._chip_hit(x, y)
        for c in self.chips:
            want = None
            if hit is not None and hit[0] is c and hit[1] != "body":
                want = ("row", int(hit[1][4:])) if hit[1].startswith("row:") else hit[1]
            if c is self.chip_pl and want == "list":
                want = None
            if want != c.hover_item:
                c.hover_item = want
                if c.motion.visible:
                    self._update_body(QRegion(self._chip_dirty(c)))
        clickable = hov or over_btn or self._visor_hit(x, y) or self._dial_at(x, y) or \
            (hit is not None and hit[1] not in ("body", "list"))
        if clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, e):
        x, y = self._pos(e)
        if self._chip_press is not None:
            c, it = self._chip_press
            self._chip_press = None
            if it == "eject":
                c.button_down = False
                self.update(self._to_win(c.button.adjusted(-6, -6, 6, 6)))
                if c.hit_button(x, y):
                    self.toggle_chip(c)
            else:
                c.pressed_item = None
                self.sounds.play("btn_release")
                hit = self._chip_hit(x, y)
                if hit is not None and hit[0] is c and hit[1] == it:
                    if it == "detach":
                        self.detach_mini()
                    else:
                        self.button_action(it)
                if c.motion.visible:
                    self._update_body(QRegion(self._chip_dirty(c)))
            return
        if self._list_drag is not None:
            y0, s0, moved, it = self._list_drag
            self._list_drag = None
            if not moved and it.startswith("row:"):
                hit = self._chip_hit(x, y)
                if hit is not None and hit[1] == it:
                    self.sounds.play("press")
                    self.play_index(int(it[4:]))
                    self._update_body(QRegion(self._chip_dirty(self.chip_pl)))
            return
        if self._press_item is not None:
            name = self._press_item
            self._press_item = None
            self.panel.pressed = None
            self.sounds.play("btn_release")
            if self.panel.item_at(x, y) == name:
                self.button_action(name)
            self.update(self._to_win(self.panel.dirty_rect()))
        if self._active_dial is not None:
            self._active_dial.end_drag()
            self.update(self._to_win(self._active_dial.rect()))
            self._active_dial = None
        if self._drag_off is not None:
            self._drag_off = None
            self.unsetCursor()
            pg = self._press_global
            self._press_global = None
            if pg is not None and (e.globalPosition().toPoint() - pg).manhattanLength() <= 3:
                self._body_click(*self._press_body)
            else:
                self._save_settings()

    def leaveEvent(self, e):
        self.panel.hover = None
        for d in self.dials.values():
            d.hover = False
        for c in self.chips:
            c.button_hover = False
            c.hover_item = None
        self.update()
        super().leaveEvent(e)

    def wheelEvent(self, e):
        x, y = self._pos(e)
        k = 1 if e.angleDelta().y() > 0 else -1
        c = self.chip_pl
        if c.motion.state == "out":
            pm, geo = c.frame(self.mode_i, config.MODES[self.mode_i])
            if c.card_rect(geo).contains(x, y):
                c.wheel(geo, k)
                self._update_body(QRegion(self._chip_dirty(c)))
                return
        d = self._dial_at(x, y) or self.dials["volume"]
        self._dial_target()
        d.set_steps(k)
        self.update(self._to_win(d.rect()))

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_V:
            self.cycle_mode(-1 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1)
        elif k == Qt.Key.Key_Space:
            self.button_action("play")
        elif k == Qt.Key.Key_M:
            self.set_muted(not self.muted)
        elif k == Qt.Key.Key_P:
            self.toggle_chip(self.chip_pl)
        elif k == Qt.Key.Key_I:
            self.toggle_chip(self.chip_mini)
        elif k == Qt.Key.Key_D:
            self.detach_mini()
        elif k == Qt.Key.Key_Right:
            self.button_action("next")
        elif k == Qt.Key.Key_Left:
            self.button_action("prev")
        elif k == Qt.Key.Key_Up:
            self.dials["volume"].set_steps(1)
            self.update(self._to_win(self.dials["volume"].rect()))
        elif k == Qt.Key.Key_Down:
            self.dials["volume"].set_steps(-1)
            self.update(self._to_win(self.dials["volume"].rect()))
        else:
            super().keyPressEvent(e)

    # ================================================================ menu
    def build_menu(self) -> QMenu:
        m = config.MODES[self.mode_i]
        pc = "rgb(%d,%d,%d)" % m.primary
        dark = "rgb(%d,%d,%d)" % tuple(int(v) for v in mix(m.primary, (0, 0, 0), 0.75))
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: #16171b; color: #d8dae0; border: 1px solid #55585f;
                     font-family: 'Lucida Console','Consolas',monospace; font-size: 11px; padding: 3px; }}
            QMenu::item {{ padding: 4px 18px 4px 22px; }}
            QMenu::item:selected {{ background: {dark}; color: white; border-left: 2px solid {pc}; }}
            QMenu::separator {{ height: 1px; background: #3a3c42; margin: 3px 6px; }}
            QMenu::indicator {{ width: 9px; height: 9px; left: 6px; }}
            QMenu::indicator:checked {{ background: {pc}; border: 1px solid #000; }}
            QMenu::indicator:unchecked {{ background: #2a2b30; border: 1px solid #000; }}
        """)
        menu.addAction("Open files...", self.open_files)
        vm = menu.addMenu("Visor mode")
        grp = QActionGroup(vm)
        for i, mode in enumerate(config.MODES):
            a = QAction(f"{mode.label}", vm, checkable=True)
            a.setChecked(i == self.visor.mode)
            a.triggered.connect(lambda _=False, i=i: self.set_mode(i))
            grp.addAction(a)
            vm.addAction(a)
        sm = menu.addMenu("Audio source")
        sg = QActionGroup(sm)
        for key, label in (("auto", "Auto (local, else system audio)"), ("local", "Local files only"),
                           ("loopback", "System audio (WASAPI loopback)"), ("synth", "Synthetic test signal")):
            a = QAction(label, sm, checkable=True)
            a.setChecked(self.hub.mode == key)
            if key == "loopback" and not self.hub.loopback.available:
                a.setEnabled(False)
            a.triggered.connect(lambda _=False, k=key: self.hub.set_mode(k))
            sg.addAction(a)
            sm.addAction(a)
        pl = QAction("Playlist chip", menu, checkable=True)
        pl.setChecked(self.chip_pl.motion.state in ("out", "ejecting"))
        pl.triggered.connect(lambda _=False: self.toggle_chip(self.chip_pl))
        menu.addAction(pl)
        mp = QAction("Mini-player chip", menu, checkable=True)
        mp.setChecked(self.chip_mini.motion.state in ("out", "ejecting"))
        mp.triggered.connect(lambda _=False: self.toggle_chip(self.chip_mini))
        menu.addAction(mp)
        po = QAction("Controls always out", menu, checkable=True)
        po.setChecked(self.panel.pinned)
        po.toggled.connect(self.set_panel_pinned)
        dm = QAction("Detach mini player", menu)
        dm.setEnabled(self.chip_mini.motion.state == "out")
        dm.triggered.connect(self.detach_mini)
        menu.addAction(dm)
        menu.addAction(po)
        menu.addSeparator()
        mu = QAction("Mute", menu, checkable=True)
        mu.setChecked(self.muted)
        mu.toggled.connect(self.set_muted)
        menu.addAction(mu)
        ot = QAction("Always on top", menu, checkable=True)
        ot.setChecked(self.on_top)
        ot.toggled.connect(self.set_on_top)
        menu.addAction(ot)
        menu.addSeparator()
        menu.addAction("Exit", self.close)
        return menu

    def contextMenuEvent(self, e):
        self.build_menu().exec(e.globalPos())

    def closeEvent(self, e):
        self._save_settings()
        self.timer.stop()
        self.face.shutdown()
        self.hub.shutdown()
        self.media.shutdown()
        if self.mini_win is not None:
            self.mini_win._docking = True
            self.mini_win.close()
        super().closeEvent(e)
        QApplication.instance().quit()
