"""The YouTube video: one frozen TabPFN table swinging up pendulums it has never seen.

About a minute and a half at 1920x1080, 60 fps. Every frame of the pendulum is a real run of the
frozen policy on CPU, recorded at the 500 Hz physics rate; nothing is keyframed. The script
simulates the story, renders it with MuJoCo in software, draws the explanations and the live
panels over it, and pipes the frames straight into ffmpeg.

  python scripts/make_video.py                                   # -> out/tabpfn_furuta_youtube.mp4
  python scripts/make_video.py --stills 12.3,40.0                # a few frames as PNGs, for checking

The video itself is not committed: it is ~100 MB and regenerates from this script and the policy
table in about fifteen minutes on a twelve-core CPU.

Software rendering (osmesa) is the default because it needs no GPU and gave identical results on
the machine this was made on; set MUJOCO_GL=egl to use one instead.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from tabpfn_control.envs import wrap_angle  # noqa: E402
from tabpfn_control.history import HistoryEncoder  # noqa: E402
from tabpfn_control.policy import FrozenPolicy  # noqa: E402
from tabpfn_control.teacher import LP_RANGE, MTIP_RANGE, TEST_PLANTS, build_plant  # noqa: E402

W, H, FPS = 1920, 1080, 60
FONTS = Path(__file__).parent / "assets" / "fonts"
REPO = "github.com/simon-bachhuber/tabpfn-furuta-in-context-adaptation"

# palette -----------------------------------------------------------------------------------------
RED = (255, 59, 59)             # explanations, and nothing else
WHITE = (245, 247, 250)
GREY = (169, 180, 198)
DIM = (110, 122, 142)
PANEL = (9, 13, 22)
NAVY = (11, 16, 32)
BLUE = (76, 141, 255)           # training pendulums
AMBER = (255, 176, 32)          # test pendulums, and the bob of the one on screen
SIG_ALPHA, SIG_PHI, SIG_U = (255, 176, 32), (124, 183, 255), (61, 220, 151)
GREEN = (61, 220, 151)

# the story: which pendulums, which take -----------------------------------------------------------
HERO = (0, 7, 7)                # start on test plant 1, swap to test plant 8, seed 7
PUSH = 0.25                     # N m on the pendulum joint for one 50 ms control step
TOUR = [(1, 1), (4, 2), (2, 3), (3, 2)]   # (test plant index, seed): 2, 5, 3, 4 in the README's numbering


def font(weight: str, size: float) -> ImageFont.FreeTypeFont:
    name = {"display": "InterDisplay-Bold", "display-semi": "InterDisplay-SemiBold", "bold": "Inter-Bold",
            "semi": "Inter-SemiBold", "medium": "Inter-Medium", "regular": "Inter-Regular"}[weight]
    return ImageFont.truetype(str(FONTS / f"{name}.ttf"), int(round(size)))


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t: float, t0: float, t1: float) -> float:
    return ease((t - t0) / max(t1 - t0, 1e-9))


# =================================================================================================
# simulation
# =================================================================================================
@dataclass
class Episode:
    """One continuous run of the frozen policy, at physics resolution."""
    name: str
    qpos: np.ndarray                 # (N, 2): arm, pendulum, at every 2 ms physics substep
    sub_dt: float
    t_ctl: np.ndarray                # (K,) start time of each 50 ms control step
    u_tab: np.ndarray                # (K,) what TabPFN chose
    rig: list                        # (K,) (length, tip mass, motor sign) during that step
    events: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return (len(self.qpos) - 1) * self.sub_dt

    def pose(self, t: float) -> np.ndarray:
        return self.qpos[int(np.clip(round(t / self.sub_dt), 0, len(self.qpos) - 1))]

    def step_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t_ctl, t, side="right") - 1, 0, len(self.t_ctl) - 1))

    def rig_at(self, t: float) -> tuple:
        return self.rig[self.step_at(t)]


def _policy():
    X, Y, extra, kind, k, w, ref = FrozenPolicy.load_table("data/policy_table.npz")
    env = build_plant(TEST_PLANTS[0])
    enc = HistoryEncoder(env, kind, k=k, window=w)
    return FrozenPolicy(env, enc, X, Y, device="cpu", extra=extra, ref=ref), enc


def _rig(env) -> tuple:
    return (round(env.Lp, 6), round(env.m_tip, 6), -1.0 if env.kt < 0 else 1.0)


def _run(pol, enc, xi, seed, script, max_s=40.0, dither=0.1) -> Episode:
    """The evaluation protocol: one second of excitation off camera, then control from hanging.
    `script(env, t, ev, phase)` may apply events and returns the new phase, or None to stop."""
    env = build_plant(xi)
    rng = np.random.default_rng(seed)
    past: deque = deque(maxlen=max(enc.window, enc.k) + 2)
    s = env.reset(rng, hanging=False)
    for _ in range(20):
        u = float(rng.uniform(-6, 6)); s2 = env.step(u); past.append((s.copy(), u, s2.copy())); s = s2
        if abs(wrap_angle(s[2])) > 1.2:
            s = env.reset(rng, hanging=False)
    s = env.reset(rng, hanging=True)
    qpos, t_ctl, u_tab, rig = [env.data.qpos.copy()], [], [], []
    ev, streak, phase = {}, 0, "swing"
    for k in range(int(max_s / env.dt)):
        t = k * env.dt
        phase = script(env, t, ev, phase)
        if phase is None:
            break
        h = enc.encode(list(past))
        a = float(pol.act(s, h))
        u = float(np.clip(a + rng.normal(0, dither), -env.max_voltage, env.max_voltage))
        t_ctl.append(t); u_tab.append(a); rig.append(_rig(env))
        sp = s.copy()
        s = env.step(u, capture=qpos.append, capture_every=1)
        past.append((sp, u, s.copy()))
        streak = streak + 1 if abs(wrap_angle(s[2])) < 0.15 else 0
        if streak == 20:                              # a second upright: note when it arrived
            ev.setdefault("caught", []).append(t - 19 * env.dt)
    return Episode(xi_name(xi), np.array(qpos), env.model.opt.timestep, np.array(t_ctl),
                   np.array(u_tab), rig, ev)


def xi_name(xi) -> str:
    return f"{xi['Lp'] * 100:.0f} cm, {xi['m_tip'] * 1e3:.0f} g, {'reversed' if xi['motor_sign'] < 0 else 'forward'}"


def simulate() -> dict:
    pol, enc = _policy()
    a, b, seed = HERO
    xb = TEST_PLANTS[b]

    def hero(env, t, ev, phase):
        caught = ev.get("caught", [])
        if phase == "swing" and caught:
            ev["up"] = caught[0]; phase = "bal1"
        if phase == "bal1" and t >= ev["up"] + 2.0:
            env.push(PUSH); ev["push"] = t; phase = "pushed"
        if phase == "pushed" and len(caught) >= 2:
            ev["rec"] = caught[1]; phase = "bal2"
        if phase == "bal2" and t >= ev["rec"] + 1.5:
            ev["rig_before"] = _rig(env)
            env.set_geometry(Lp=xb["Lp"], m_tip=xb["m_tip"])
            if (xb["motor_sign"] < 0) != (env.kt < 0):
                env.set_plant(motor_scale=-1.0)
            env.restyle(); ev["change"] = t; phase = "changed"
        if phase == "changed" and len(caught) >= 3:
            ev["rec2"] = caught[2]; phase = "bal3"
        if phase == "bal3" and t >= ev["rec2"] + 3.0:
            return None
        return phase

    def tour(env, t, ev, phase):
        caught = ev.get("caught", [])
        if caught:
            ev.setdefault("up", caught[0])
            if t >= ev["up"] + 2.2:
                return None
        return phase

    t0 = time.perf_counter()
    out = {"hero": _run(pol, enc, TEST_PLANTS[a], seed, hero)}
    for i, sd in TOUR:
        out[f"tour{i}"] = _run(pol, enc, TEST_PLANTS[i], sd, tour, max_s=14.0)
    for k, e in out.items():
        print(f"  {k:6s} {e.name:24s} {e.duration:5.1f} s  "
              + ", ".join(f"{n} {v:.1f}" for n, v in e.events.items() if isinstance(v, float)), flush=True)
    print(f"simulated in {time.perf_counter() - t0:.0f} s", flush=True)
    return out


# =================================================================================================
# 3D
# =================================================================================================
GROUPS = {
    "rig": ("plinth", "base", "motorcase", "motorcap", "polarity", "armrod", "armhub", "pendrod", "hinge", "bob"),
    "pendulum": ("pendrod", "hinge", "bob"),
    "arm": ("armrod", "armhub", "motorcase", "motorcap", "polarity"),
    "motor": ("motorcase", "motorcap", "polarity"),
}


class Scene:
    """One MuJoCo model redressed per frame to whatever pendulum the recording says was there."""

    def __init__(self, w=W, h=H):
        self.env = build_plant(TEST_PLANTS[0])
        m = self.env.model
        m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 1920), max(h, 1080)
        self.r = mujoco.Renderer(m, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.0, 0.0, 0.47]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.62, 150.0, -4.0
        self.rig = None
        self.ids = {g: [m.geom(n).id for n in names] for g, names in GROUPS.items()}
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * 0.5, h * 0.40
        r = np.sqrt(((xx - cx) / (w * 0.55)) ** 2 + ((yy - cy) / (h * 0.75)) ** 2)
        self.glow = (np.clip(1 - r, 0, 1) ** 2)[..., None] * np.array([34, 48, 78], np.float32)
        rv = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h / 2) / (h * 0.62)) ** 2)
        self.vignette = np.clip(1 - 0.38 * np.clip(rv - 0.35, 0, None) ** 1.6, 0.45, 1)[..., None]

    def set_rig(self, rig) -> None:
        if rig == self.rig:
            return
        Lp, mt, sign = rig
        self.env.set_geometry(Lp=Lp, m_tip=mt)
        if (sign < 0) != (self.env.kt < 0):
            self.env.set_plant(motor_scale=-1.0)
        self.env.restyle()
        self.rig = rig

    def _draw(self, qpos) -> None:
        self.env.data.qpos[:] = qpos
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.r.update_scene(self.env.data, camera=self.cam)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1

    def render(self, qpos, rig) -> np.ndarray:
        self.set_rig(rig)
        self._draw(qpos)
        img = self.r.render().astype(np.float32)
        lum = img.mean(axis=2, keepdims=True) / 255.0
        img = img + self.glow * (1 - lum) ** 2                # light the backdrop, not the rig
        return np.clip(img * self.vignette, 0, 255).astype(np.uint8)

    def bbox(self, qpos, rig, group: str, pad=26) -> tuple:
        self.set_rig(rig)
        self.r.enable_segmentation_rendering()
        self._draw(qpos)
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], self.ids[group])
        ys, xs = np.nonzero(mask)
        return (int(xs.min()) - pad, int(ys.min()) - pad, int(xs.max()) + pad, int(ys.max()) + pad)


# =================================================================================================
# drawing
# =================================================================================================
SS = 2          # every vector overlay is drawn at twice the size and scaled down, for clean edges


def blank(w=W, h=H) -> Image.Image:
    return Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))


def down(layer: Image.Image, w=W, h=H) -> Image.Image:
    return layer.resize((w, h), Image.LANCZOS)


def text_size(fnt, s: str) -> tuple:
    x0, y0, x1, y1 = fnt.getbbox(s)
    return x1 - x0, y1 - y0


def wrap(fnt, s: str, width: int) -> list:
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if text_size(fnt, trial)[0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w_
    return lines + ([cur] if cur else [])


_SHADOWS: dict = {}


def shadowed_text(base: Image.Image, xy, s, fnt, fill, alpha=1.0, shadow=0.75, radius=10) -> None:
    """Text with a soft dark halo, so it reads over any part of the scene. Coordinates in SS space.

    The halo is blurred once per string on a tile just big enough for it and cached: blurring a
    full-frame layer per text per frame made the title alone take ten minutes to render."""
    if alpha <= 0 or not s:
        return
    key = (s, fnt.path, fnt.size, int(radius))
    if key not in _SHADOWS:
        x0, y0, x1, y1 = fnt.getbbox(s)
        pad = int(radius * 3)
        tile = Image.new("L", (int(x1) + 2 * pad, int(y1) + 2 * pad), 0)
        ImageDraw.Draw(tile).text((pad, pad), s, font=fnt, fill=255)
        _SHADOWS[key] = (tile.filter(ImageFilter.GaussianBlur(radius)), pad)
    halo, pad = _SHADOWS[key]
    a = shadow * alpha
    layer = Image.new("RGBA", halo.size, (0, 0, 0, 0))
    layer.putalpha(halo.point(lambda v: int(v * a)))
    ox, oy = int(round(xy[0])) - pad, int(round(xy[1])) - pad
    cx0, cy0 = max(ox, 0), max(oy, 0)
    cx1, cy1 = min(ox + layer.width, base.width), min(oy + layer.height, base.height)
    if cx1 > cx0 and cy1 > cy0:
        base.alpha_composite(layer.crop((cx0 - ox, cy0 - oy, cx1 - ox, cy1 - oy)), (cx0, cy0))
    ImageDraw.Draw(base).text(xy, s, font=fnt, fill=(*fill, int(255 * alpha)))


def arrow(d: ImageDraw.ImageDraw, p0, p1, bend: float, width: float, col, progress: float) -> None:
    """A quadratic Bézier from p0 to p1, drawn up to `progress`, with a head once it arrives."""
    if progress <= 0:
        return
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = -(y1 - y0), (x1 - x0)
    nrm = math.hypot(nx, ny) or 1.0
    cx, cy = mx + bend * nx / nrm, my + bend * ny / nrm
    n = 60
    pts = []
    for i in range(int(n * progress) + 1):
        s = i / n
        pts.append(((1 - s) ** 2 * x0 + 2 * (1 - s) * s * cx + s * s * x1,
                    (1 - s) ** 2 * y0 + 2 * (1 - s) * s * cy + s * s * y1))
    if len(pts) >= 2:
        d.line(pts, fill=col, width=int(width), joint="curve")
        for p in (pts[0], pts[-1]):
            r = width / 2
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=col)
    if progress >= 0.98:
        tx, ty = 2 * (x1 - cx), 2 * (y1 - cy)
        L = math.hypot(tx, ty) or 1.0
        ux, uy = tx / L, ty / L
        hl, hw = width * 4.2, width * 2.6
        bx, by = x1 - ux * hl, y1 - uy * hl
        d.polygon([(x1 + ux * width * 0.4, y1 + uy * width * 0.4),
                   (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)], fill=col)


@dataclass
class Note:
    """One explanation: a red box around something, a red arrow, red text."""
    title: str
    sub: str = ""
    side: str = "left"          # left | right | above
    y: int | None = None        # vertical position of the text block (1080p), default: box centre
    x: int | None = None        # left edge of the text block, if it must be pinned
    maxw: int | None = None     # wrap the title to this width (1080p px)


def note_layer(box, note: Note, t: float, dur: float) -> tuple:
    """(overlay RGBA at W×H, dim factor) for an explanation `t` seconds into its `dur`."""
    fade_out = 1 - ramp(t, dur - 0.22, dur)
    k_box = ramp(t, 0.05, 0.40) * fade_out
    k_text = ramp(t, 0.20, 0.55) * fade_out
    k_arrow = ramp(t, 0.42, 0.78)                  # the arrow grows out of text already there
    L = blank(); d = ImageDraw.Draw(L)
    x0, y0, x1, y1 = box
    grow = 1 + 0.06 * (1 - ramp(t, 0.05, 0.40))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0, bx1 = cx + (x0 - cx) * grow, cx + (x1 - cx) * grow
    by0, by1 = cy + (y0 - cy) * grow, cy + (y1 - cy) * grow
    if k_box > 0:
        a = int(255 * k_box)
        d.rounded_rectangle((bx0 * SS - 6, by0 * SS - 6, bx1 * SS + 6, by1 * SS + 6), radius=18 * SS,
                            outline=(0, 0, 0, int(a * 0.45)), width=10 * SS // 2)
        d.rounded_rectangle((bx0 * SS, by0 * SS, bx1 * SS, by1 * SS), radius=16 * SS,
                            outline=(*RED, a), width=5 * SS)

    ft, fs = font("semi", 46 * SS), font("regular", 28 * SS)
    title_lines = wrap(ft, note.title, note.maxw * SS) if note.maxw else [note.title]
    tl_h = 56 * SS
    tw = max(text_size(ft, s_)[0] for s_ in title_lines)
    th = text_size(ft, title_lines[0])[1] + tl_h * (len(title_lines) - 1)
    sub_lines = wrap(fs, note.sub, (note.maxw * SS) if note.maxw else max(tw, 470 * SS)) if note.sub else []
    sw = max([text_size(fs, s_)[0] for s_ in sub_lines] + [0])
    block_w, line_h = max(tw, sw), 38 * SS
    block_h = th + (16 * SS + line_h * len(sub_lines) if sub_lines else 0)
    lo_y, hi_y = 96 * SS, (TEL[1] - 40) * SS - block_h       # clear of the chip and the bottom panels
    margin, gap = 64 * SS, 120 * SS

    def fits(side):
        if side == "left":
            x = (note.x * SS) if note.x is not None else bx0 * SS - gap - block_w
            return x, x >= margin and x + block_w <= bx0 * SS - 40 * SS
        x = (note.x * SS) if note.x is not None else bx1 * SS + gap
        return x, x >= bx1 * SS + 40 * SS and x + block_w <= W * SS - margin

    if note.side == "above":
        tx = float(np.clip((note.x if note.x is not None else x0) * SS, margin, W * SS - margin - block_w))
        ty = (note.y if note.y is not None else y0 - 230) * SS
        anchor = (tx + tw * 0.5, ty + block_h + 26 * SS)
        target = ((x0 + x1) / 2 * SS, y0 * SS - 10 * SS)
        bend = -70 * SS
    else:
        side = note.side
        tx, ok = fits(side)
        if not ok:                                         # never let the text sit on the object
            other = "right" if side == "left" else "left"
            tx2, ok2 = fits(other)
            if ok2:
                side, tx = other, tx2
            else:
                tx = margin if side == "left" else W * SS - margin - block_w
        ty = float(np.clip((note.y if note.y is not None else cy - 60) * SS, lo_y, max(lo_y, hi_y)))
        first_w = text_size(ft, title_lines[0])[0]
        mid = ty + text_size(ft, title_lines[0])[1] * 0.58
        if side == "left":
            anchor, ax_, bend = (tx + first_w + 28 * SS, mid), bx0 * SS - 12 * SS, -55 * SS
        else:
            anchor, ax_, bend = (tx - 28 * SS, mid), bx1 * SS + 12 * SS, 55 * SS
        target = (ax_, float(np.clip(mid + 70 * SS, by0 * SS + 36 * SS, by1 * SS - 36 * SS)))
    if k_arrow > 0 and fade_out > 0:
        arrow(d, anchor, target, bend, 5 * SS, (*RED, int(255 * fade_out)), k_arrow)
    lift = (1 - k_text) * 18 * SS
    for i, s_ in enumerate(title_lines):
        shadowed_text(L, (tx, ty + i * tl_h + lift), s_, ft, RED, alpha=k_text, radius=12 * SS)
    for i, s_ in enumerate(sub_lines):
        shadowed_text(L, (tx, ty + th + 16 * SS + i * line_h + lift), s_, fs, (226, 232, 240),
                      alpha=k_text, radius=10 * SS)
    return down(L), 0.50 * ramp(t, 0.0, 0.35) * fade_out


def spotlight(img: Image.Image, box, dim: float) -> Image.Image:
    """Darken everything but the box, softly."""
    if dim <= 0:
        return img
    m = Image.new("L", (W // 4, H // 4), 255)
    x0, y0, x1, y1 = [v / 4 for v in box]
    ImageDraw.Draw(m).rounded_rectangle((x0, y0, x1, y1), radius=5, fill=0)
    m = m.filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR)
    a = np.asarray(m, np.float32)[..., None] / 255.0 * dim
    return Image.fromarray((np.asarray(img, np.float32) * (1 - a)).astype(np.uint8))


def dim_all(img: Image.Image, k: float) -> Image.Image:
    return Image.fromarray((np.asarray(img, np.float32) * (1 - k)).astype(np.uint8)) if k > 0 else img


def panel(w, h, radius=22) -> Image.Image:
    L = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    ImageDraw.Draw(L).rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), radius=radius * SS,
                                        fill=(*PANEL, 178), outline=(255, 255, 255, 26), width=2 * SS)
    return L


# ------------------------------------------------------------------------------------ telemetry
TEL = (36, H - 36 - 408, 540, 408)          # x, y, w, h: bottom left, clear of the swing


class Telemetry:
    """Both joint angles and TabPFN's output, over the last few seconds."""

    WINDOW = 6.0

    def __init__(self):
        self.base = None

    def draw(self, ep: Episode, t: float, paused: bool) -> Image.Image:
        x, y, w, h = TEL
        L = panel(w, h); d = ImageDraw.Draw(L)
        f_head, f_lab, f_val, f_tick = font("semi", 19 * SS), font("medium", 20 * SS), font("semi", 22 * SS), font("regular", 15 * SS)
        d.text((24 * SS, 20 * SS), "LIVE TELEMETRY", font=f_head, fill=(*GREY, 255))
        if paused:
            tag = "❚❚ PAUSED"
            tw_, _ = text_size(f_head, tag)
            d.rounded_rectangle(((w - 30) * SS - tw_ - 24 * SS, 14 * SS, (w - 22) * SS, 46 * SS),
                                radius=14 * SS, fill=(255, 255, 255, 30))
            d.text(((w - 30) * SS - tw_ - 12 * SS, 20 * SS), tag, font=f_head, fill=(*WHITE, 235))
        else:
            d.ellipse(((w - 40) * SS, 26 * SS, (w - 28) * SS, 38 * SS), fill=(*GREEN, 255))
            tl = f"t = {t:4.1f} s"
            d.text(((w - 50) * SS - text_size(f_head, tl)[0], 20 * SS), tl, font=f_head, fill=(*DIM, 255))

        t0 = max(0.0, t - self.WINDOW)
        n = max(2, int((t - t0) / ep.sub_dt))
        ts = np.linspace(t0, t, min(n, 360))
        q = np.array([ep.pose(s) for s in ts]) if len(ts) else np.zeros((0, 2))
        elev = 180 - np.degrees(np.abs(wrap_angle(q[:, 1]))) if len(q) else np.zeros(0)
        phi = np.degrees(wrap_angle(q[:, 0])) if len(q) else np.zeros(0)     # a dial: it wraps
        k0, k1 = ep.step_at(t0), ep.step_at(t)
        tu, uu = ep.t_ctl[k0:k1 + 1], ep.u_tab[k0:k1 + 1]
        now = ep.pose(t)
        rows = [("PENDULUM", "angle from hanging", elev, (0, 180), ("up", "down"), SIG_ALPHA,
                 f"{180 - abs(float(np.degrees(wrap_angle(now[1])))):5.1f}°"),
                ("ARM", "twist angle", phi, (-180, 180), ("+180", "−180"), SIG_PHI,
                 f"{float(np.degrees(wrap_angle(now[0]))):+6.1f}°"),
                ("TABPFN OUTPUT", "motor voltage", None, (-10, 10), ("+10 V", "−10 V"), SIG_U,
                 f"{float(ep.u_tab[k1]):+5.1f} V")]
        top, row_h, plot_h = 64, 112, 62
        px0, gut, px1 = 24, 62, w - 24                    # labels live in the gutter, not on the data
        for i, (name, what, series, (lo, hi), ticks, col, val) in enumerate(rows):
            ry = top + i * row_h
            d.text((px0 * SS, ry * SS), name, font=f_lab, fill=(*col, 255))
            nw, _ = text_size(f_lab, name)
            d.text((px0 * SS + nw + 10 * SS, ry * SS + 2 * SS), what, font=f_tick, fill=(*DIM, 255))
            vw, _ = text_size(f_val, val)
            d.text((px1 * SS - vw, (ry - 2) * SS), val, font=f_val, fill=(*WHITE, 255))
            gy0, gy1 = (ry + 30) * SS, (ry + 30 + plot_h) * SS
            ga = (px0 + gut) * SS
            d.rounded_rectangle((ga, gy0, px1 * SS, gy1), radius=8 * SS, fill=(255, 255, 255, 10))
            for lab, yy in ((ticks[0], gy0 + 2 * SS), (ticks[1], gy1 - 18 * SS)):
                lw, _ = text_size(f_tick, lab)
                d.text((ga - 10 * SS - lw, yy), lab, font=f_tick, fill=(*DIM, 210))
            ref = 0 if lo < 0 < hi else hi
            zy = gy1 - (ref - lo) / (hi - lo) * (gy1 - gy0)
            for xs in range(ga, px1 * SS, 14 * SS):
                d.line((xs, zy, xs + 6 * SS, zy), fill=(255, 255, 255, 38), width=SS)

            def xmap(tt, ga=ga):
                return ga + (np.asarray(tt) - (t - self.WINDOW)) / self.WINDOW * (px1 * SS - ga)

            def ymap(v, gy0=gy0, gy1=gy1, lo=lo, hi=hi):
                return gy1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (gy1 - gy0)
            if series is not None and len(series) >= 2:
                xs_, ys_ = xmap(ts), ymap(series)
                cut = np.nonzero(np.abs(np.diff(series)) > 180)[0] + 1      # a wrap is not a jump
                for a_, b_ in zip(np.r_[0, cut], np.r_[cut, len(series)]):
                    if b_ - a_ >= 2:
                        d.line(list(zip(xs_[a_:b_], ys_[a_:b_])), fill=(*col, 255), width=int(2.6 * SS), joint="curve")
                ex, ey = xs_[-1], ys_[-1]
            elif series is None and len(tu):
                pts = []
                for j in range(len(tu)):
                    xa = float(xmap(max(tu[j], t - self.WINDOW)))
                    xb_ = float(xmap(tu[j + 1])) if j + 1 < len(tu) else float(xmap(t))
                    yv = float(ymap(uu[j])); pts += [(xa, yv), (xb_, yv)]
                d.line(pts, fill=(*col, 255), width=int(2.6 * SS))
                ex, ey = pts[-1]
            else:
                continue
            d.ellipse((ex - 5 * SS, ey - 5 * SS, ex + 5 * SS, ey + 5 * SS), fill=(*col, 255))
        return down(L, w, h)


# ------------------------------------------------------------------------------------ plant map
def plant_map(w: int, h: int, big: bool, show_test: float = 1.0):
    """The training and test pendulums in length and tip mass, drawn with matplotlib.
    Returns the image and the pixel position of every test pendulum and of the training region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    fp = lambda wgt, sz: FontProperties(fname=str(FONTS / f"Inter-{wgt}.ttf"), size=sz)  # noqa: E731
    xi = np.load("data/demos.npz")["xi"]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    if big:
        ax = fig.add_axes([0.10, 0.13, 0.86, 0.80])
    else:
        ax = fig.add_axes([0.14, 0.17, 0.80, 0.66])
    ax.set_facecolor((1, 1, 1, 0.0))
    col = lambda c, a=1.0: (c[0] / 255, c[1] / 255, c[2] / 255, a)  # noqa: E731
    s_tr = 150 if big else 26
    for sign, mk in [(1.0, "o"), (-1.0, "s")]:
        m = xi[:, 2] == sign
        ax.scatter(xi[m, 0] * 100, xi[m, 1] * 1e3, s=s_tr, marker=mk, color=col(BLUE, 0.85),
                   edgecolors=col(NAVY), linewidths=1.2 if big else 0.6, zorder=3)
    if show_test > 0:
        for i, p in enumerate(TEST_PLANTS):
            mk = "o" if p["motor_sign"] > 0 else "s"
            ax.scatter([p["Lp"] * 100], [p["m_tip"] * 1e3], s=(620 if big else 120) * show_test,
                       marker=mk, color=col(AMBER, show_test), edgecolors=col(NAVY, show_test),
                       linewidths=2.0 if big else 1.0, zorder=5)
            if big:
                ax.text(p["Lp"] * 100, p["m_tip"] * 1e3, str(i + 1), ha="center", va="center",
                        fontproperties=fp("Bold", 15), color=col(NAVY, show_test), zorder=6)
    ax.set_xlim(LP_RANGE[0] * 100 - 1.5, LP_RANGE[1] * 100 + 1.5)
    ax.set_ylim(MTIP_RANGE[0] * 1e3 - 3.5, MTIP_RANGE[1] * 1e3 + 3.5)
    ax.grid(True, color=(1, 1, 1, 0.07), lw=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=col(GREY, 0.85), length=0, labelsize=15 if big else 9)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(fp("Regular", 15 if big else 9))
    if not big:
        ax.set_xticks([20, 30, 40]); ax.set_yticks([10, 30, 50])
    ax.set_xlabel("pendulum length  [cm]", fontproperties=fp("Medium", 18 if big else 10), color=col(GREY), labelpad=10 if big else 4)
    ax.set_ylabel("mass at the tip  [g]", fontproperties=fp("Medium", 18 if big else 10), color=col(GREY), labelpad=10 if big else 4)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba()))
    to_px = lambda x_, y_: tuple(ax.transData.transform((x_, y_)) * [1, -1] + [0, h])  # noqa: E731
    pts = [to_px(p["Lp"] * 100, p["m_tip"] * 1e3) for p in TEST_PLANTS]
    (ax0, ay1), (ax1, ay0) = to_px(xi[:, 0].min() * 100, xi[:, 1].min() * 1e3), to_px(xi[:, 0].max() * 100, xi[:, 1].max() * 1e3)
    plt.close(fig)
    return img.copy(), pts, (ax0, ay0, ax1, ay1)


MAP = (W - 36 - 540, H - 36 - 408, 540, 408)


class Minimap:
    def __init__(self):
        x, y, w, h = MAP
        self.img, self.pts, _ = plant_map(w - 20, h - 64, big=False)

    def draw(self, current: int | None, visited: set, t: float) -> Image.Image:
        x, y, w, h = MAP
        base = down(panel(w, h), w, h)
        base.alpha_composite(self.img, (10, 52))
        L = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
        f_head, f_small = font("semi", 19 * SS), font("regular", 16 * SS)
        d.text((24 * SS, 20 * SS), "PLANT MAP", font=f_head, fill=(*GREY, 255))
        leg = [("training", BLUE), ("validation", AMBER)]
        lx = (w - 24) * SS
        for name, c in reversed(leg):
            tw_, _ = text_size(f_small, name)
            lx -= tw_; d.text((lx, 22 * SS), name, font=f_small, fill=(*GREY, 255))
            lx -= 22 * SS; d.ellipse((lx, 26 * SS, lx + 13 * SS, 39 * SS), fill=(*c, 255)); lx -= 22 * SS
        for i in visited:
            px, py = self.pts[i]
            cx_, cy_ = (px + 10) * SS, (py + 52) * SS
            d.ellipse((cx_ - 11 * SS, cy_ - 11 * SS, cx_ + 11 * SS, cy_ + 11 * SS), outline=(*WHITE, 150), width=2 * SS)
        if current is not None:
            px, py = self.pts[current]
            cx_, cy_ = (px + 10) * SS, (py + 52) * SS
            pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.1)
            rr = (15 + 7 * pulse) * SS
            d.ellipse((cx_ - rr, cy_ - rr, cx_ + rr, cy_ + rr), outline=(*WHITE, int(120 + 110 * (1 - pulse))), width=3 * SS)
            d.ellipse((cx_ - 15 * SS, cy_ - 15 * SS, cx_ + 15 * SS, cy_ + 15 * SS), outline=(*WHITE, 255), width=3 * SS)
            lab = f"#{current + 1}"
            d.text((cx_ + 20 * SS, cy_ - 30 * SS), lab, font=font("bold", 20 * SS), fill=(*WHITE, 255))
        base.alpha_composite(down(L, w, h))
        return base


# ------------------------------------------------------------------------------------ chips
def chip(text: str, status: str = "", alpha: float = 1.0, y: int = 36) -> tuple:
    f, fs = font("semi", 26 * SS), font("semi", 22 * SS)
    tw, th = text_size(f, text)
    sw = text_size(fs, status)[0] + 50 * SS if status else 0
    w_, h_ = tw + sw + 56 * SS, 56 * SS
    L = Image.new("RGBA", (w_, h_), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    d.rounded_rectangle((0, 0, w_ - 1, h_ - 1), radius=28 * SS, fill=(*PANEL, int(190 * alpha)),
                        outline=(255, 255, 255, int(30 * alpha)), width=2 * SS)
    d.text((28 * SS, 13 * SS), text, font=f, fill=(*WHITE, int(255 * alpha)))
    if status:
        sx = 28 * SS + tw + 26 * SS
        d.ellipse((sx, 22 * SS, sx + 12 * SS, 34 * SS), fill=(*GREEN, int(255 * alpha)))
        d.text((sx + 22 * SS, 15 * SS), status, font=fs, fill=(*GREEN, int(255 * alpha)))
    img = L.resize((w_ // SS, h_ // SS), Image.LANCZOS)
    return img, ((W - img.width) // 2, y)


# =================================================================================================
# the edit
# =================================================================================================
class Video:
    def __init__(self, story: dict):
        self.story = story
        self.scene = Scene()
        self.tel = Telemetry()
        self.minimap = None
        self.shots: list = []           # (duration, fn(t_local) -> PIL RGB)
        self.names: list = []
        self.cache: dict = {}

    # ---- building blocks
    def frame3d(self, ep: Episode, t: float, rig=None) -> Image.Image:
        key = (id(ep), round(t, 4), rig)
        if key not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            self.cache[key] = Image.fromarray(self.scene.render(ep.pose(t), rig or ep.rig_at(t)))
        return self.cache[key]

    def hud(self, img: Image.Image, ep: Episode, t: float, paused: bool, extra=None, alpha=1.0) -> Image.Image:
        img = img.convert("RGBA")
        tel = self.tel.draw(ep, t, paused)
        if alpha < 1:
            tel.putalpha(Image.eval(tel.getchannel("A"), lambda v: int(v * alpha)))
        img.alpha_composite(tel, TEL[:2])
        for layer, xy in (extra or []):
            img.alpha_composite(layer, xy)
        return img.convert("RGB")

    def add(self, dur: float, fn, name: str = "") -> None:
        self.shots.append((dur, fn))
        self.names.append(name or getattr(fn, "__name__", "shot"))

    def timeline(self) -> str:
        t, out = 0.0, []
        for (dur, _), n in zip(self.shots, self.names):
            out.append(f"  {t:6.2f}  {dur:5.2f}s  {n}"); t += dur
        return "\n".join(out)

    # ---- shot types
    def play(self, ep, t0, t1, speed=1.0, extra=None, badge=None):
        def fn(tl):
            t = min(t0 + tl * speed, t1)
            ex = list(extra(t) if callable(extra) else (extra or []))
            if badge:
                ex.append(self._badge(badge))
            return self.hud(self.frame3d(ep, t), ep, t, False, ex)
        self.add((t1 - t0) / speed, fn, f"play {t0:.2f}->{t1:.2f} x{speed}")

    def _badge(self, text):
        f = font("semi", 30 * SS)
        tw, th = text_size(f, text)
        L = Image.new("RGBA", (tw + 64 * SS, 66 * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
        d.rounded_rectangle((0, 0, L.width - 1, L.height - 1), radius=33 * SS, fill=(*PANEL, 205),
                            outline=(*AMBER, 150), width=3 * SS)
        d.text((32 * SS, 15 * SS), text, font=f, fill=(*AMBER, 255))
        img = L.resize((L.width // SS, L.height // SS), Image.LANCZOS)
        return img, (W - 40 - img.width, 40)

    def explain(self, ep, t, group, note: Note, dur=2.9, rig=None, extra=None, box=None):
        base = self.frame3d(ep, t, rig)
        if box is None:
            box = self.scene.bbox(ep.pose(t), rig or ep.rig_at(t), group)
        cache = {}

        def fn(tl):
            k = "hold" if 0.82 < tl < dur - 0.25 else tl          # nothing moves in the hold
            if k in cache:
                return cache[k]
            ov, dim = note_layer(box, note, tl, dur)
            img = spotlight(self.hud(base, ep, t, True, extra), box, dim)
            img = img.convert("RGBA"); img.alpha_composite(ov)
            out = img.convert("RGB")
            if k == "hold":
                cache[k] = out
            return out
        self.add(dur, fn, f"explain: {note.title}")

    def caption(self, ep, t, text, sub="", dur=2.6, rig=None, extra=None):
        base = self.frame3d(ep, t, rig)

        def fn(tl):
            k = ramp(tl, 0.0, 0.4) * (1 - ramp(tl, dur - 0.3, dur))
            img = self.hud(dim_all(base, 0.5 * k), ep, t, True, extra).convert("RGBA")
            L = blank(); ft, fs = font("display", 72 * SS), font("regular", 32 * SS)
            tw, th = text_size(ft, text)
            y = 250 * SS + (1 - k) * 20 * SS
            d = ImageDraw.Draw(L)
            d.rounded_rectangle(((W * SS - 90 * SS) / 2, y - 40 * SS, (W * SS + 90 * SS) / 2, y - 32 * SS),
                                radius=4 * SS, fill=(*RED, int(255 * k)))
            shadowed_text(L, ((W * SS - tw) / 2, y), text, ft, WHITE, alpha=k, radius=16 * SS)
            if sub:
                sw, _ = text_size(fs, sub)
                shadowed_text(L, ((W * SS - sw) / 2, y + th + 34 * SS), sub, fs, GREY, alpha=k, radius=10 * SS)
            img.alpha_composite(down(L))
            return img.convert("RGB")
        self.add(dur, fn, f"caption: {text}")

    # ---- the whole story
    def build(self):
        hero = self.story["hero"]; ev = hero.events
        tour = [(i, self.story[f"tour{i}"]) for i, _ in TOUR]
        a, b, _ = HERO
        rig_a, rig_b = ev["rig_before"], hero.rig_at(ev["change"] + 0.01)

        # title
        bg = self.frame3d(hero, 0.0)
        blurred = dim_all(Image.fromarray(np.asarray(bg)).filter(ImageFilter.GaussianBlur(22)), 0.55)

        def title(tl):
            reveal = ramp(tl, 4.3, 5.0)
            img = Image.blend(blurred, self.hud(bg, hero, 0.0, False), reveal) if reveal > 0 else blurred.copy()
            k = ramp(tl, 0.25, 1.1) * (1 - ramp(tl, 4.2, 4.8))
            L = blank(); d = ImageDraw.Draw(L)
            f_over, f_title, f_sub = font("semi", 24 * SS), font("display", 96 * SS), font("regular", 36 * SS)
            y = 330 * SS + (1 - ramp(tl, 0.25, 1.1)) * 30 * SS
            x = 170 * SS
            d.rounded_rectangle((x, y - 64 * SS, x + 110 * SS, y - 56 * SS), radius=4 * SS, fill=(*RED, int(255 * k)))
            shadowed_text(L, (x, y - 34 * SS), "PRIOR LABS  ·  TABPFN-3.5 HACKATHON", f_over, GREY, alpha=k, radius=8 * SS)
            for i, line in enumerate(["A frozen TabPFN", "as a feedback controller"]):
                shadowed_text(L, (x, y + 20 * SS + i * 112 * SS), line, f_title, WHITE, alpha=k, radius=18 * SS)
            k2 = ramp(tl, 0.9, 1.7) * (1 - ramp(tl, 4.2, 4.8))
            shadowed_text(L, (x, y + 280 * SS), "It swings up and balances pendulums it was never trained on.",
                          f_sub, GREY, alpha=k2, radius=10 * SS)
            shadowed_text(L, (x, y + 334 * SS), "No retraining. One forward pass every 50 ms.",
                          f_sub, GREY, alpha=k2, radius=10 * SS)
            img = img.convert("RGBA"); img.alpha_composite(down(L))
            return img.convert("RGB")
        self.add(5.0, title)

        # part 1: what it is
        t_up, t_push, t_rec, t_chg, t_rec2 = ev["up"], ev["push"], ev["rec"], ev["change"], ev["rec2"]
        self.play(hero, 0.0, 2.0)
        self.explain(hero, 2.0, "rig", Note("Furuta pendulum", "A motor twists the arm. The pendulum on its end swings freely.", "left"))
        t_b = t_up + 0.7
        self.play(hero, 2.0, t_b)
        self.explain(hero, t_b, "pendulum", Note("Goal: swing it up and balance it", "", "right", y=250))
        self.explain(hero, t_b, "arm", Note("TabPFN controls the twist motion", "It picks the motor voltage 20 times a second. That is all it does.", "left", y=330))
        tx, ty, tw_, th_ = TEL
        self.explain(hero, t_b, None, Note("Live telemetry", "Both joint angles and TabPFN's output, as they happen, for the rest of the video.", "above", y=400, x=80),
                     box=(tx - 4, ty - 4, tx + tw_ + 4, ty + th_ + 4))

        # part 2: a push
        t_c = t_push + 0.12
        self.play(hero, t_b, t_c)
        self.explain(hero, t_c, "pendulum", Note("We just pushed it", "A sudden external kick. Nothing is retrained.", "right", y=260))
        self.play(hero, t_c, t_rec + 0.8, speed=0.5, badge="½×  slow motion")
        self.play(hero, t_rec + 0.8, t_chg - 0.02)

        # part 3: a different pendulum
        t_d = t_chg - 0.02
        self.caption(hero, t_d, "What if we change the system?", "Same frozen table. A pendulum it has never seen.")

        def morph(tl, dur=1.1):
            s = ramp(tl, 0.05, dur - 0.1)
            Lp = rig_a[0] + (rig_b[0] - rig_a[0]) * s
            mt = rig_a[1] + (rig_b[1] - rig_a[1]) * s
            sign = rig_b[2] if s > 0.55 else rig_a[2]
            img = Image.fromarray(self.scene.render(hero.pose(t_chg), (Lp, mt, sign)))
            return self.hud(img, hero, t_chg, True)
        self.add(1.1, morph)
        self.explain(hero, t_chg, "pendulum",
                     Note("Longer pendulum, heavier tip",
                          f"{rig_a[0] * 100:.0f} → {rig_b[0] * 100:.0f} cm,   {rig_a[1] * 1e3:.0f} → {rig_b[1] * 1e3:.0f} g at the tip",
                          "left", y=250), rig=rig_b)
        self.explain(hero, t_chg, "motor",
                     Note("Motor wired backwards", "Every correction it has made so far now pushes the wrong way.",
                          "right", y=300), rig=rig_b)
        self.play(hero, t_chg, t_rec2 + 0.9, speed=0.5, badge="½×  slow motion")
        t_e = t_rec2 + 0.9
        self.play(hero, t_e, t_e + 1.0)
        t_e += 1.0
        self.explain(hero, t_e, "pendulum", Note("TabPFN adapts online", "Nothing is retrained: the last second of motion tells it which pendulum it is driving.", "left", y=260))
        self.explain(hero, t_e, "rig", Note("It has never seen this pendulum", "It interpolates between the pendulums it was trained on.", "right", y=280))

        # part 4: the plant map
        end3d = self.hud(self.frame3d(hero, t_e), hero, t_e, True)
        big_rect = (110, 210, 1180, 790)
        big_tr, pts_big, tr_box = plant_map(big_rect[2], big_rect[3], big=True, show_test=0.0)
        big_all, _, _ = plant_map(big_rect[2], big_rect[3], big=True, show_test=1.0)
        heading = self._heading()

        def slide(show_test=0.0, extra_note=None, tl=0.0, dur=1.0, img_override=None):
            img = Image.new("RGB", (W, H), NAVY)
            img = Image.fromarray(np.clip(np.asarray(img, np.float32) + self.scene.glow[:, :, :] * 0.8, 0, 255).astype(np.uint8)).convert("RGBA")
            img.alpha_composite(heading)
            chart = img_override if img_override is not None else (big_all if show_test >= 1 else big_tr)
            img.alpha_composite(chart, big_rect[:2])
            if extra_note is not None:
                box, note = extra_note
                ov, dim = note_layer(box, note, tl, dur)
                img = spotlight(img.convert("RGB"), box, dim * 0.8).convert("RGBA")
                img.alpha_composite(ov)
            return img.convert("RGB")

        def to_slide(tl, dur=0.9):
            k = ramp(tl, 0, dur)
            return Image.blend(end3d, slide(), k)
        self.add(0.9, to_slide)
        self.add(0.5, lambda tl: slide())
        bx = (big_rect[0] + tr_box[0] - 26, big_rect[1] + tr_box[1] - 26, big_rect[0] + tr_box[2] + 26, big_rect[1] + tr_box[3] + 26)
        self.add(2.9, lambda tl: slide(0.0, (bx, Note("60 training pendulums", "One planner demonstration each. Together they are the table TabPFN reads.", "right", y=330, x=1340, maxw=500)), tl, 2.9), "note: training")

        def pop(tl, dur=0.9):
            k = ramp(tl, 0, dur)
            img, _, _ = plant_map(big_rect[2], big_rect[3], big=True, show_test=k)
            return slide(img_override=img)
        self.add(0.9, pop)
        for i, title_, sub_ in [(a, "The first pendulum in this video", "Never in the training set."),
                                (b, "The one it was swapped for", "Also never seen. There are eight validation pendulums like these.")]:
            px, py = pts_big[i]
            box = (big_rect[0] + px - 34, big_rect[1] + py - 34, big_rect[0] + px + 34, big_rect[1] + py + 34)
            self.add(2.9, lambda tl, box=box, t_=title_, s_=sub_: slide(1.0, (box, Note(t_, s_, "right", y=int(box[1]) - 20, x=1340, maxw=500)), tl, 2.9),
                     f"note: {title_}")

        def result(tl, dur=3.6):
            img = slide(1.0).convert("RGBA")
            k = ramp(tl, 0.1, 0.5) * (1 - ramp(tl, dur - 0.25, dur))
            L = blank(); x = 1370 * SS
            shadowed_text(L, (x, 330 * SS), "32 / 32", font("display", 120 * SS), WHITE, alpha=k, radius=16 * SS)
            shadowed_text(L, (x, 490 * SS), "validation runs swing up", font("medium", 32 * SS), GREY, alpha=k, radius=8 * SS)
            shadowed_text(L, (x, 530 * SS), "and balance", font("medium", 32 * SS), GREY, alpha=k, radius=8 * SS)
            k2 = ramp(tl, 0.6, 1.0) * (1 - ramp(tl, dur - 0.25, dur))
            shadowed_text(L, (x, 620 * SS), "12 / 12", font("display", 72 * SS), WHITE, alpha=k2, radius=12 * SS)
            shadowed_text(L, (x, 716 * SS), "on pendulums drawn fresh,", font("medium", 28 * SS), GREY, alpha=k2, radius=8 * SS)
            shadowed_text(L, (x, 752 * SS), "never used for any choice", font("medium", 28 * SS), GREY, alpha=k2, radius=8 * SS)
            img.alpha_composite(down(L))
            return img.convert("RGB")
        self.add(3.6, result)

        # part 5: the tour, with the map in the corner
        self.minimap = Minimap()
        visited = {a, b}
        first_i, first_ep = tour[0]
        mx, my, mw, mh = MAP

        def slide_bg():
            img = Image.new("RGB", (W, H), NAVY)
            img = Image.fromarray(np.clip(np.asarray(img, np.float32) + self.scene.glow * 0.8, 0, 255).astype(np.uint8)).convert("RGBA")
            img.alpha_composite(heading)
            return img.convert("RGB")
        bg_only = slide_bg()

        def to_tour(tl, dur=1.2):
            s = ramp(tl, 0.0, dur)
            scene = self.hud(self.frame3d(first_ep, 0.0), first_ep, 0.0, True, alpha=ramp(tl, 0.4, dur))
            img = Image.blend(bg_only, scene, ramp(tl, 0.15, dur * 0.85)).convert("RGBA")
            x = big_rect[0] + (mx + 10 - big_rect[0]) * s
            y = big_rect[1] + (my + 52 - big_rect[1]) * s
            w_ = int(big_rect[2] + (mw - 20 - big_rect[2]) * s)
            h_ = int(big_rect[3] + (mh - 64 - big_rect[3]) * s)
            mm = self.minimap.draw(first_i, visited, tl)
            mm.putalpha(Image.eval(mm.getchannel("A"), lambda v: int(v * ramp(tl, 0.75, dur))))
            img.alpha_composite(mm, (mx, my))
            travel = big_all.resize((max(w_, 2), max(h_, 2)), Image.LANCZOS)
            fade = 1 - ramp(tl, 0.55, dur * 0.95)
            if fade > 0:
                travel.putalpha(Image.eval(travel.getchannel("A"), lambda v: int(v * fade)))
                img.alpha_composite(travel, (int(x), int(y)))
            return img.convert("RGB")
        self.add(1.2, to_tour)

        for n, (i, ep) in enumerate(tour):
            p = TEST_PLANTS[i]
            label = (f"Validation pendulum {i + 1} of 8   ·   {p['Lp'] * 100:.0f} cm   ·   {p['m_tip'] * 1e3:.0f} g at the tip"
                     f"   ·   motor {'reversed' if p['motor_sign'] < 0 else 'forward'}")
            up = ep.events.get("up", ep.duration)
            vis = set(visited)

            def extra(t, i=i, up=up, vis=vis, label=label):
                c, xy = chip(label, "balanced" if t >= up + 0.4 else "")
                return [(c, xy), (self.minimap.draw(i, vis, t), (mx, my))]
            if n > 0:
                prev_i, prev_ep = tour[n - 1]

                def cross(tl, prev=prev_ep, pi=prev_i, ep=ep, i=i, extra=extra, pvis=set(visited) - {i}):
                    k = ramp(tl, 0, 0.45)
                    A = self.hud(self.frame3d(prev, prev.duration), prev, prev.duration, False,
                                 [(self.minimap.draw(pi, pvis, tl), (mx, my))])
                    B = self.hud(self.frame3d(ep, 0.0), ep, 0.0, False, extra(0.0))
                    return Image.blend(A, B, k)
                self.add(0.45, cross)
            self.play(ep, 0.0, ep.duration, extra=extra)
            visited.add(i)

        # the end
        last_i, last_ep = tour[-1]
        fin = self.hud(self.frame3d(last_ep, last_ep.duration), last_ep, last_ep.duration, False,
                       [(self.minimap.draw(last_i, visited, 0.0), (mx, my))])
        fin_blur = dim_all(fin.filter(ImageFilter.GaussianBlur(24)), 0.6)

        def end(tl, dur=7.2):
            img = Image.blend(fin, fin_blur, ramp(tl, 0.0, 0.8))
            img = img.convert("RGBA"); L = blank()
            k1 = ramp(tl, 0.5, 1.2) * (1 - ramp(tl, 3.9, 4.4))
            f1, f2 = font("medium", 36 * SS), font("display-semi", 52 * SS)
            s1, s2 = "Additional details at", REPO
            w1, _ = text_size(f1, s1); w2, _ = text_size(f2, s2)
            shadowed_text(L, ((W * SS - w1) / 2, 440 * SS), s1, f1, GREY, alpha=k1, radius=8 * SS)
            shadowed_text(L, ((W * SS - w2) / 2, 500 * SS), s2, f2, WHITE, alpha=k1, radius=12 * SS)
            ImageDraw.Draw(L).rounded_rectangle(((W * SS - 90 * SS) / 2, 410 * SS, (W * SS + 90 * SS) / 2, 418 * SS),
                                                 radius=4 * SS, fill=(*RED, int(255 * k1)))
            k2 = ramp(tl, 4.5, 5.2) * (1 - ramp(tl, dur - 0.7, dur))
            f3 = font("display", 120 * SS)
            w3, h3 = text_size(f3, "Thank you")
            shadowed_text(L, ((W * SS - w3) / 2, (H * SS - h3) / 2 - 30 * SS), "Thank you", f3, WHITE, alpha=k2, radius=18 * SS)
            img.alpha_composite(down(L))
            return dim_all(img.convert("RGB"), ramp(tl, dur - 0.5, dur))
        self.add(7.2, end)

    def _heading(self) -> Image.Image:
        L = blank()
        shadowed_text(L, (110 * SS, 70 * SS), "Trained on 60 pendulums, validated on 8 it never saw", font("display", 56 * SS), WHITE, radius=12 * SS)
        shadowed_text(L, (110 * SS, 150 * SS), "Every dot is one pendulum: its length and the mass on its tip. Circles have the motor forward, squares reversed.",
                      font("regular", 27 * SS), GREY, radius=8 * SS)
        return down(L)

    # ---- output
    @property
    def duration(self) -> float:
        return sum(d for d, _ in self.shots)

    def frame_at(self, t: float) -> Image.Image:
        for dur, fn in self.shots:
            if t < dur:
                return fn(t)
            t -= dur
        return self.shots[-1][1](self.shots[-1][0] - 1e-6)

    def encode(self, out: Path, fps: int, crf: int, preset: str) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2", "-g", str(fps // 2), "-bf", "2",
               "-movflags", "+faststart", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n = int(round(self.duration * fps)); t0 = time.perf_counter()
        for f in range(n):
            img = self.frame_at(f / fps)
            proc.stdin.write(np.asarray(img, np.uint8).tobytes())
            if f % (fps * 5) == 0:
                el = time.perf_counter() - t0
                print(f"  frame {f:5d}/{n}  {f / fps:5.1f} s of video  ({el:4.0f} s elapsed, "
                      f"~{el / max(f, 1) * (n - f) / 60:4.1f} min left)", flush=True)
        proc.stdin.close(); proc.wait()
        print(f"saved {out}  ({n} frames, {self.duration:.1f} s, {out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/tabpfn_furuta_youtube.mp4")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--crf", type=int, default=16, help="x264 quality; lower is better and bigger")
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--stills", default=None, help="comma-separated video times to write as PNGs instead")
    ap.add_argument("--cache", default=None, help="reuse a simulated story from this file, or write it there")
    args = ap.parse_args()

    if args.cache and Path(args.cache).exists():
        import pickle
        story = pickle.loads(Path(args.cache).read_bytes())
        print(f"story loaded from {args.cache}")
    else:
        story = simulate()
        if args.cache:
            import pickle
            Path(args.cache).write_bytes(pickle.dumps(story))
    video = Video(story)
    video.build()
    print(f"edit: {len(video.shots)} shots, {video.duration:.1f} s", flush=True)
    print(video.timeline(), flush=True)
    out = Path(os.path.expanduser(args.out))
    if args.stills:
        for s in args.stills.split(","):
            f = out.with_name(f"{out.stem}_{float(s):05.1f}.png")
            video.frame_at(float(s)).save(f); print("wrote", f)
        return
    video.encode(out, args.fps, args.crf, args.preset)


if __name__ == "__main__":
    main()
