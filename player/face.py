"""
Facial reactions (smirk, gasp, lip bite) as layered face art.

Every reaction is a smooth, supersampled *warp of her own mouth* (taken from
the relit body of the current visor mode, so it lines up exactly), plus a few
procedurally painted details (dark mouth interior, a hint of teeth, a dimple).
For each (mode, reaction) we pre-render LEVELS intensity frames once (numpy,
~10-20 ms) and at runtime only blit/cross-fade two small pixmaps over the
mouth rect. The vein glow inside that rect is warped identically, so the
chin vein follows the lower lip.

Eye reactions (slow blink, wink) live in eyes.EyeLife; they only show in
Scanning-eye mode where her eyes are visible.
"""
from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import QRect

from .qtutil import premul_to_pixmap

# mouth landmarks in body coordinates (measured on assets/layers/base.png)
CX = 252.5            # mouth centre x
HW = 28.5             # half width (corners at ~224 / ~281)
REGION = QRect(206, 224, 94, 70)
SS = 3                # supersampling of the warp
LEVELS = 8            # pre-rendered intensity steps (cross-faded at runtime)

MOUTH_REACTIONS = ("smirk", "gasp", "bite")
EYE_REACTIONS = ("slow_blink", "wink")

# timeline (seconds): ease in, hold, ease out
TIMING = {
    "smirk": (0.45, 1.6, 0.7),
    "gasp": (0.28, 0.9, 0.65),
    "bite": (0.4, 1.3, 0.6),
    "slow_blink": (0.0, 1.35, 0.0),
    "wink": (0.0, 1.0, 0.0),
}


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _mouth_line(u):
    return 248.6 + 1.4 * np.minimum(u * u, 1.44)


def _upper_top(u):
    return 237.6 + 4.2 * np.minimum(u * u, 1.44)


def _lower_bot(u):
    return 264.0 - 13.0 * np.minimum(u * u, 1.0)


def _lens(u, p=0.75):
    return np.clip(1 - u * u, 0, 1) ** p


def _bilinear(img, sx, sy):
    """Sample HxWxC image at float pixel coords (edge-clamped)."""
    h, w = img.shape[:2]
    sx = np.clip(sx, 0, w - 1.001)
    sy = np.clip(sy, 0, h - 1.001)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _cubic_matrix(n, k):
    """(n*k x n) Catmull-Rom upsampling matrix (sample centres aligned)."""
    m = np.zeros((n * k, n), np.float32)
    for i in range(n * k):
        x = (i + 0.5) / k - 0.5
        x0 = math.floor(x)
        t = x - x0
        ws = (-0.5 * t ** 3 + t * t - 0.5 * t, 1.5 * t ** 3 - 2.5 * t * t + 1,
              -1.5 * t ** 3 + 2 * t * t + 0.5 * t, 0.5 * t ** 3 - 0.5 * t * t)
        for j, wgt in zip(range(x0 - 1, x0 + 3), ws):
            m[i, min(n - 1, max(0, j))] += wgt
    return m


def _upsample(img, k):
    h, w = img.shape[:2]
    my, mx = _cubic_matrix(h, k), _cubic_matrix(w, k)
    out = np.einsum("yh,hwc->ywc", my, img)
    out = np.einsum("xw,ywc->yxc", mx, out)
    return out


def _down(img, k):
    h, w = img.shape[0] // k, img.shape[1] // k
    return img.reshape(h, k, w, k, -1).mean(axis=(1, 3))


def _teeth(xt, Y, top, h, half):
    """Upper teeth (two incisors + laterals) hanging from y=top.
    -> (coverage, rgb HxWx3). xt: x relative to the dental midline."""
    w4 = half / 2.0
    ax = np.abs(xt)
    lat = ax > w4
    hh = h * np.where(lat, 0.78, 1.0)
    k = np.floor((xt + half) / w4)
    q = np.abs(xt - (-half + (k + 0.5) * w4)) / (w4 / 2)       # 0 centre .. 1 edge of a tooth
    bottom = top + hh - 0.55 * np.clip(q - 0.45, 0, 1) ** 2 * 2.4
    cov = _ss(top - 0.3, top + 0.15, Y) * (1 - _ss(bottom - 0.3, bottom + 0.2, Y))
    cov = cov * _ss(half + 0.3, half - 1.3, ax)
    arch = 0.94 - 0.24 * np.clip(ax / half, 0, 1) ** 1.5            # sides curve away into shadow
    vert = 0.66 + 0.34 * _ss(top, top + np.maximum(hh, 0.5), Y)      # shaded under the lip
    sep = 1 - 0.32 * (np.exp(-(xt ** 2) / (2 * 0.28 ** 2)) + np.exp(-((ax - w4) ** 2) / (2 * 0.3 ** 2)))
    rgb = np.array([0.90, 0.85, 0.84], np.float32) * (arch * vert * sep)[..., None]
    return cov, rgb


class FaceRig:
    """Pre-rendered mouth reaction frames per (mode, reaction)."""

    def __init__(self, art):
        self.art = art
        self.rect = QRect(REGION)
        r = self.rect
        k = SS
        # destination sample grid in body coordinates (supersampled)
        xs = r.x() + (np.arange(r.width() * k) + 0.5) / k
        ys = r.y() + (np.arange(r.height() * k) + 0.5) / k
        self.X, self.Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        self.U = (self.X - CX) / HW
        h, w = r.height(), r.width()
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
        self._edge = _ss(1.0, 6.0, dist)[..., None]
        self._bases = {}               # id(src) -> (1x body premul, 1x vein glow/core)
        self._src_cache: dict[int, np.ndarray] = {}
        self._vein_src = None
        self._frames: dict[tuple, tuple] = {}
        # background pre-rendering (numpy releases the GIL for the heavy maths;
        # only the tiny QPixmap conversions happen on the GUI thread in poll())
        self._pool = None
        self._jobs = {}               # (mi, kind) -> [future per level]

    # ------------------------------------------------------------------ data
    def _src(self, mi):
        s = self._src_cache.get(mi)
        if s is None:
            s = _upsample(self.art.body_region(mi, self.rect), SS)
            s[..., 3] = np.clip(s[..., 3], 0, 1)
            s[..., :3] = np.clip(s[..., :3], 0, s[..., 3:4])
            g, v = self.art.vein_region(self.rect)
            self._bases[id(s)] = (self.art.body_region(mi, self.rect), np.dstack([g, v]))
            self._src_cache[mi] = s
        return s

    def _base_for(self, src):
        return self._bases[id(src)]

    def _veins_src(self):
        if self._vein_src is None:
            g, v = self.art.vein_region(self.rect)
            self._vein_src = np.clip(_upsample(np.dstack([g, v]), SS), 0, 1)
        return self._vein_src

    def frames(self, mi: int, kind: str):
        """-> (body pixmaps[LEVELS], vein pixmaps[LEVELS]); level i = (i+1)/LEVELS."""
        key = (mi, kind)
        f = self._frames.get(key)
        if f is None:
            f = self._build(mi, kind)
            self._frames[key] = f
        return f

    def ready(self, mi: int, kind: str) -> bool:
        return (mi, kind) in self._frames

    def request(self, mi: int):
        """Start pre-rendering this mode's reactions in the background."""
        for kind in MOUTH_REACTIONS:
            key = (mi, kind)
            if key in self._frames or key in self._jobs:
                continue
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vexa-face")
            src = self._src(mi)
            vsrc = self._veins_src()
            self._jobs[key] = [self._pool.submit(self._job, mi, src, vsrc, kind, (i + 1) / LEVELS)
                               for i in range(LEVELS)]

    def _job(self, mi, src, vsrc, kind, s):
        img, vg = self.render(src, vsrc, kind, s)
        rgb, a = self.art.vein_colour(mi, vg[..., 0], vg[..., 1])
        return img, rgb, a

    @property
    def pending(self) -> bool:
        return bool(self._jobs)

    def poll(self) -> bool:
        """Collect finished background frames (GUI thread). True if any landed."""
        done = []
        for key, futs in self._jobs.items():
            if all(f.done() for f in futs):
                done.append(key)
        for key in done:
            futs = self._jobs.pop(key)
            try:
                body, veins = [], []
                for f in futs:
                    img, rgb, a = f.result()
                    body.append(premul_to_pixmap(img[..., :3], img[..., 3]))
                    veins.append(premul_to_pixmap(rgb, a))
                self._frames[key] = (body, veins)
            except Exception as e:          # never let a face frame take the app down
                print(f"[vexa] face rig failed for {key}: {e}")
        return bool(done)

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def prebuild(self, mi: int):
        for kind in MOUTH_REACTIONS:
            self.frames(mi, kind)

    def drop_mode(self, keep: int):
        for k in [k for k in self._frames if k[0] != keep]:
            del self._frames[k]
        for k in [k for k in self._src_cache if k != keep]:
            del self._src_cache[k]

    def _build(self, mi, kind):
        body, veins = [], []
        src = self._src(mi)
        vsrc = self._veins_src()
        for i in range(LEVELS):
            s = (i + 1) / LEVELS
            img, vg = self.render(src, vsrc, kind, s)
            body.append(premul_to_pixmap(img[..., :3], img[..., 3]))
            rgb, a = self.art.vein_colour(mi, vg[..., 0], vg[..., 1])
            veins.append(premul_to_pixmap(rgb, a))
        return body, veins

    # ------------------------------------------------------------------ warp
    def _to_src_px(self, sx, sy):
        r = self.rect
        return (sx - r.x()) * SS - 0.5, (sy - r.y()) * SS - 0.5

    def render(self, src, vsrc, kind, s):
        """One reaction frame at intensity s (0..1) -> (premul RGBA 1x, vein (glow, core) 1x)."""
        X, Y, U = self.X, self.Y, self.U
        ml = _mouth_line(U)
        sx, sy = X.copy(), Y.copy()
        paint = []            # (coverage, rgb, alpha) layers composited over the warp
        if kind == "smirk":
            self._smirk(X, Y, U, ml, s, sx, sy, paint, side=1, amt=1.0)
        elif kind == "gasp":
            self._gasp(X, Y, U, ml, s, sx, sy, paint)
        elif kind == "bite":
            self._smirk(X, Y, U, ml, s, sx, sy, paint, side=1, amt=0.35)
            self._bite(X, Y, U, ml, s, sx, sy, paint)
        px, py = self._to_src_px(sx, sy)
        out = _bilinear(src, px, py)
        vout = _bilinear(vsrc, px, py)
        touched = np.hypot(sx - X, sy - Y) / 0.3          # where anything actually moved...
        for cov, rgb, alpha in paint:
            a = (np.clip(cov, 0, 1) * alpha)[..., None]
            out[..., :3] = out[..., :3] * (1 - a) + np.asarray(rgb, np.float32) * a * out[..., 3:4]
            vout *= (1 - a)              # no vein light inside the mouth / on teeth
            touched = touched + a[..., 0] * 4                  # ...or was painted
        out, vout = _down(out, SS), _down(vout, SS)
        # Everywhere else keep her *exact* original pixels (the resampling would
        # soften them a hair), feathered, and forced at the rect edge -> seamless.
        m = _down(np.clip(touched, 0, 1)[..., None], SS)
        m = np.minimum(m, self._edge)
        base, vbase = self._base_for(src)
        out = base + (out - base) * m
        vout = vbase + (vout - vbase) * m
        return out, vout

    @staticmethod
    def _smirk(X, Y, U, ml, s, sx, sy, paint, side=1, amt=1.0):
        """One corner (her left = viewer's right) lifts and pulls out."""
        xc = CX + side * HW
        yc = 249.6
        w = np.exp(-((X - xc) ** 2) / (2 * 9.0 ** 2) - ((Y - yc) ** 2) / (2 * 7.0 ** 2))
        # the near half of both lips curls up a little with it
        half = _ss(-0.15, 1.0, U * side) ** 1.6
        lipw = np.exp(-((Y - ml) ** 2) / (2 * 8.0 ** 2))
        dy = -(4.0 * w + 1.4 * half * lipw) * s * amt
        dx = side * (1.1 * w) * s * amt
        # the other corner tightens a hair (asymmetry reads as a smirk, not a smile)
        wo = np.exp(-((X - (CX - side * HW)) ** 2) / (2 * 7.0 ** 2) - ((Y - yc) ** 2) / (2 * 6.0 ** 2))
        dx += side * 0.5 * wo * s * amt
        win = _ss(0.0, 9.0, np.minimum(X - REGION.x(), REGION.right() - X)) * \
            _ss(0.0, 9.0, np.minimum(Y - REGION.y(), REGION.bottom() - Y))
        sx -= dx * win
        sy -= dy * win
        if amt >= 0.99:
            # soft dimple / cheek crease just outside the lifted corner
            cx, cy = xc + side * 4.2, yc - 2.6
            ang = np.arctan2(Y - cy, (X - cx) * side)
            rr = np.hypot((X - cx) / 1.0, (Y - cy) / 1.25)
            arc = np.exp(-((rr - 3.6) ** 2) / (2 * 0.75 ** 2)) * _ss(-1.9, -0.2, ang) * (1 - _ss(1.0, 1.9, ang))
            paint.append((arc * s, (0.36, 0.17, 0.17), 0.30))
            hl = np.exp(-((X - (xc + side * 1.0)) ** 2) / (2 * 4.5 ** 2) - ((Y - (yc - 6.5)) ** 2) / (2 * 2.6 ** 2))
            paint.append((hl * s, (1.0, 0.86, 0.80), 0.10))

    @staticmethod
    def _gasp(X, Y, U, ml, s, sx, sy, paint):
        """Lips part softly: upper lip lifts a touch, lower lip drops."""
        L = _lens(U, 0.6)
        Up = 1.5 * s * L
        Dn = 4.4 * s * L
        tu = ml - _upper_top(U)            # upper lip thickness
        tl = _lower_bot(U) - ml            # lower lip thickness
        above = Y < ml
        d = ml - Y
        fu = 1 - _ss(tu + Up, tu + Up + 6.5, d)
        sy_up = Y + Up * fu
        e = Y - ml
        fl = 1 - _ss(tl + Dn, tl + Dn + 16.0, e)
        sy_dn = Y - Dn * fl
        sy[:] = np.where(above, sy_up, sy_dn)
        # mouth narrows slightly as it opens (an "o", not a grin)
        lipw = np.exp(-((Y - ml) ** 2) / (2 * 9.0 ** 2))
        sx += 1.0 * s * U * lipw * _ss(1.45, 1.0, np.abs(U))
        # interior: between the displaced lip edges (antialiased by the supersampling)
        top = ml - Up
        bot = ml + Dn
        cov = _ss(top - 0.25, top + 0.25, Y) * (1 - _ss(bot - 0.25, bot + 0.25, Y))
        paint.append((cov, (0.06, 0.012, 0.035), 1.0))
        span = np.maximum(bot - top, 1e-3)
        dd = np.clip((Y - top) / span, 0, 1)
        # tongue: soft plum glow low in the mouth
        tg = cov * _ss(0.45, 0.95, dd) * _ss(0.75, 0.2, np.abs(U))
        paint.append((tg, (0.36, 0.11, 0.17), 0.6))
        # upper teeth: an ivory hint right under the upper lip
        th = np.minimum(1.9, span * 0.4) * np.clip((s - 0.3) / 0.45, 0, 1)
        tc, trgb = _teeth(X - (CX - 0.5), Y, top - 0.2, th, 10.5)
        paint.append((cov * tc, trgb, 0.9))
        # wet inner rim of the lower lip
        rim = _ss(bot - 0.2, bot + 0.3, Y) * (1 - _ss(bot + 0.5, bot + 1.3, Y)) * _ss(0.85, 0.3, np.abs(U))
        paint.append((rim * s, (0.95, 0.72, 0.86), 0.22))

    @staticmethod
    def _bite(X, Y, U, ml, s, sx, sy, paint):
        """Upper teeth resting on a gently tucked lower lip (a little off-centre)."""
        ub, wb = -0.14, 0.40
        bump = _ss(1.0, 0.3, np.abs(U - ub) / wb)          # 1 in the middle, 0 at the sides
        lift = 0.9 * s * bump
        tu = ml - _upper_top(U)
        d = ml - Y
        fu = 1 - _ss(tu + lift, tu + lift + 6.0, d)
        above = Y < ml
        sy_up = Y + lift * fu
        # lower lip rolls in under the teeth: the visible lip starts further down
        tl = _lower_bot(U) - ml
        e = Y - ml
        tuck = 2.3 * s * bump * (1 - _ss(tl * 0.5, tl + 1.5, e))
        sy_dn = Y + tuck
        sy[:] = np.where(above, sy_up, sy_dn)
        top = ml - lift
        xt = X - (CX + ub * HW)
        half = 6.4
        th = 2.7 * s
        # the crease the teeth press into the lip + the lip bulging a little below
        crease = np.exp(-((Y - (top + th + 0.45)) ** 2) / (2 * 0.55 ** 2)) * _ss(half + 1.5, half - 2.5, np.abs(xt))
        paint.append((crease * s, (0.08, 0.015, 0.05), 0.65))
        bul = np.exp(-((Y - (top + th + 2.3)) ** 2) / (2 * 0.9 ** 2)) * _ss(half + 2.5, half - 3.0, np.abs(xt))
        paint.append((bul * s, (0.62, 0.38, 0.58), 0.25))
        tc, trgb = _teeth(xt, Y, top - 0.3, th + 0.3, half)
        paint.append((tc * np.clip((s - 0.15) / 0.45, 0, 1), trgb, 0.93))


TRIGGER_CHANCE = {"drop": 0.75, "track": 0.5, "idle": 0.9, "click": 0.8, "hover": 0.4}
TRIGGER_PREFS = {
    "drop": ("gasp", "smirk", "bite"),
    "track": ("smirk", "wink", "slow_blink", "bite"),
    "idle": ("slow_blink", "bite", "smirk", "wink"),
    "click": ("wink", "smirk", "bite", "gasp"),
    "hover": ("smirk", "wink", "slow_blink"),
}
# reaction -> (mouth kind, mouth gain, eye kind)
PLAN = {
    "smirk": ("smirk", 1.0, None),
    "gasp": ("gasp", 1.0, None),
    "bite": ("bite", 1.0, None),
    "slow_blink": (None, 0.0, "slow_blink"),
    "wink": ("smirk", 0.55, "wink"),          # a wink comes with half a smirk
}


class FaceDirector:
    """Decides *when* and *how* she reacts: bass drops, track changes, long
    idle and the occasional face hover/click; 20-40 s cooldown between them."""

    COOLDOWN = (20.0, 40.0)

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.kind = None             # current reaction name
        self.mouth = None            # mouth kind being shown
        self.gain = 1.0
        self.t0 = 0.0
        self.next_ok = 8.0           # nothing during/just after boot
        self.quiet_t = 0.0           # how long the bass has been low (drop detection)
        self.idle_since = None
        self.last_idle_react = -1e9
        self.last = None

    # ---------------------------------------------------------------- state
    @property
    def active(self) -> bool:
        return self.kind is not None

    def can_react(self, now) -> bool:
        return self.kind is None and now >= self.next_ok

    def choose(self, trigger, eyes_visible, ready) -> str | None:
        """Pick a reaction for a trigger (None if nothing suitable is ready)."""
        pool = [k for k in TRIGGER_PREFS.get(trigger, tuple(PLAN))
                if (PLAN[k][2] is None or eyes_visible) and (PLAN[k][0] is None or ready(PLAN[k][0]))]
        if not pool:
            return None
        if self.last in pool and len(pool) > 1:
            pool.remove(self.last)
        # earlier entries in the preference list are a bit more likely
        weights = [1.0 / (1 + 0.45 * i) for i in range(len(pool))]
        return self.rng.choices(pool, weights)[0]

    def start(self, kind, now):
        self.kind = kind
        self.last = kind
        self.mouth, self.gain, _ = PLAN[kind]
        self.t0 = now
        self.next_ok = now + self.duration(kind) + self.rng.uniform(*self.COOLDOWN)

    def eye_kind(self, kind=None):
        plan = PLAN.get(kind or self.kind)
        return plan[2] if plan else None

    def duration(self, kind=None) -> float:
        """Length of a reaction; 0 when nothing is (or `kind` isn't) known."""
        plan = PLAN.get(kind or self.kind)
        if plan is None:
            return 0.0
        m, _, e = plan
        d = sum(TIMING[m]) if m else 0.0
        return max(d, sum(TIMING[e]) if e else 0.0)

    def mouth_level(self, now) -> float:
        """Eased mouth intensity (already scaled by gain); ends the reaction."""
        if self.kind is None:
            return 0.0
        if now - self.t0 >= self.duration():
            self.kind = self.mouth = None
            return 0.0
        if self.mouth is None:
            return 0.0
        a, h, b = TIMING[self.mouth]
        e = now - self.t0
        if e >= a + h + b:
            return 0.0
        if e < a:
            u = e / a
            v = u * u * (3 - 2 * u)
        elif e < a + h:
            v = 1.0
        else:
            u = (e - a - h) / b
            v = 1 - u * u * (3 - 2 * u)
        return v * self.gain

    def cancel(self):
        self.kind = self.mouth = None

    # ---------------------------------------------------------------- triggers
    def feed(self, now, dt, bass, audio: bool) -> str | None:
        """Watch the signal. Returns 'drop' / 'idle' when one fires."""
        if audio:
            self.idle_since = None
            if bass < 0.35:
                self.quiet_t += dt
            else:
                if bass > 0.82 and self.quiet_t > 4.0:
                    self.quiet_t = 0.0
                    return "drop"
                self.quiet_t = max(0.0, self.quiet_t - dt * 3)
        else:
            self.quiet_t = 0.0
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since > 55.0 and now - self.last_idle_react > 90.0:
                self.last_idle_react = now
                return "idle"
        return None

    def roll(self, trigger) -> bool:
        return self.rng.random() < TRIGGER_CHANCE.get(trigger, 0.5)
