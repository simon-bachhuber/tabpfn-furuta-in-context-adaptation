"""Interactive demo: one frozen TabPFN table controls the pendulum, and you change the pendulum.

Nothing is re-fit, ever. A single table maps [state, 10 numbers summarising the last second] ->
voltage in one forward pass per 50 ms step. The plant is never named to the model, so when you
reverse the motor or randomise the pendulum the same table has to read the change out of its own
recent history — which takes about a second, the length of the window.

  python scripts/demo.py --viewer              # native MuJoCo window, drag the pendulum with the mouse
  python scripts/demo.py --web                 # browser instead, for a machine with no display
  python scripts/demo.py --record media/demo.mp4 --seconds 40    # scripted: push, reverse, randomise

--viewer opens MuJoCo's own interactive window, where ctrl + left-drag on the pendulum applies a
force and ctrl + right-drag applies a torque, so you can fight the controller by hand. Keys change
the plant underneath it: f reverses the motor, r randomises everything, 0 drops it back to hanging,
d toggles the exploration dither. On macOS the window has to own the main thread, so launch it with
mjpython (which ships with the mujoco wheel) rather than python:

  mjpython scripts/demo.py --viewer

--web renders through mjviser/Viser client-side and needs no local display or GL, which is the only
reason it exists; mjviser cannot drag bodies with the mouse, so there the disturbances are buttons.

Three clocks run in --viewer and none waits on another. Physics advances in real time on the last
action it was handed, which is a zero-order hold and what a real rig does between updates. TabPFN
runs on its own thread as fast as it can, so a slow forward pass costs control rate rather than
stalling the window. Rendering keeps to --render-fps. On macOS the controller is put on the CPU,
because the viewer is already working the GPU hard and a stuttering window is more obvious than a
slightly slower control loop.
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

if ("--record" in sys.argv and "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
    os.environ["MUJOCO_GL"] = "osmesa"   # headless: software rendering is the only thing that works

import numpy as np

from tabpfn_control.envs import wrap_angle
from tabpfn_control.history import HistoryEncoder
from tabpfn_control.policy import POLARITY_COL, FrozenPolicy
from tabpfn_control.teacher import (LP_RANGE, MTIP_RANGE, TEST_PLANTS, build_plant,
                                    sample_plant)


def describe(xi: dict) -> str:
    return (f"{xi['Lp'] * 100:.0f} cm rod, {xi['m_tip'] * 1e3:.0f} g at the tip, motor "
            f"{'REVERSED' if xi['motor_sign'] < 0 else 'forward'}")


def retune(env, xi: dict) -> None:
    """Swap in a different pendulum without rebuilding the model the viewer is already drawing."""
    env.set_geometry(Lp=xi["Lp"], m_tip=xi["m_tip"])
    if (xi["motor_sign"] < 0) != (env.kt < 0):
        env.set_plant(motor_scale=-1.0)
    env.restyle()


def make_renderer(env, width=1200, height=800):
    """A MuJoCo offscreen renderer, or None when this machine has no working GL backend.

    Headless rendering needs EGL or OSMesa (`libgl1`, `libgl1-mesa-dri`). Where neither loads, the
    recording falls back to a schematic drawn from the same angles, so the command always produces
    a file. The browser demo is unaffected either way: mjviser renders client-side.
    """
    import warnings

    import mujoco
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # a missing GL backend warns before it raises
            r = mujoco.Renderer(env.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        # three-quarter view: the pendulum swings in a plane that turns with the arm, and from here
        # that plane is seen close to face-on, so the swing reads as a swing and not as a nod
        cam.lookat[:] = [0.0, 0.0, 0.42]
        cam.distance, cam.azimuth, cam.elevation = 1.30, 150.0, -4.0
        r.update_scene(env.data, camera=cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1        # the studio backdrop: no horizon line
        r.render()
        return r, cam
    except Exception as e:  # noqa: BLE001
        print(f"MuJoCo offscreen rendering unavailable ({type(e).__name__}); drawing a schematic instead")
        return None, None


def load_policy(env, path, device):
    X, Y, extra, kind, k, window, ref = FrozenPolicy.load_table(path)
    enc = HistoryEncoder(env, kind, k=k, window=window)
    t0 = time.perf_counter()
    pol = FrozenPolicy(env, enc, X, Y, device=device, extra=extra, ref=ref)
    print(f"frozen table: {X.shape[1]} cols ({enc.n_features} history), {len(X)} rows, fit {time.perf_counter() - t0:.0f}s")
    print("no planner, no LQR, no re-fitting — one TabPFN call per control step")
    return pol, enc


def viewer_blocker() -> str | None:
    """Why the native viewer cannot open here, or None if it should. Checked before the 20 s fit."""
    import mujoco.viewer
    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        return ("On macOS the MuJoCo window must own the main thread, so this mode has to be started\n"
                "with mjpython, which the mujoco wheel installs next to python:\n\n"
                f"    mjpython {' '.join(sys.argv)}\n\n"
                "Use --web instead if you would rather drive it from a browser.")
    return None


def apply_mouse(env, mujoco, pert, gain: float = 60.0) -> np.ndarray | None:
    """Turn the viewer's current drag into a force on the pendulum, or None if nothing is dragging.

    MuJoCo's own ``mjv_applyPerturbForce`` handles the twist (ctrl + right-drag) but declines the
    translation, because it only pushes *free* bodies and this pendulum hangs off a hinge. A force
    on a hinged body is perfectly well defined — MuJoCo projects it onto the joint — so the drag is
    reproduced here as a spring pulling the grabbed point towards the pointer.

    It has to be the grabbed point. ``xfrc_applied`` acts at the centre of mass, and a force there
    exerts no moment about a hinge it happens to point along, so dragging would go dead in one
    direction. Offsetting to where the pointer actually took hold adds the missing couple.
    """
    env.data.xfrc_applied[:] = 0.0
    sel = int(pert.select)
    if not pert.active or sel <= 0 or sel >= env.model.nbody:
        return None
    mujoco.mjv_applyPerturbForce(env.model, env.data, pert)       # the twist, when that is the drag
    if pert.active & int(mujoco.mjtPertBit.mjPERT_TRANSLATE):
        f = gain * float(env.model.body_mass[sel]) * (np.asarray(pert.refpos) - env.data.xpos[sel])
        n = float(np.linalg.norm(f))
        if n > 5.0:                                               # keep one flick from launching it
            f *= 5.0 / n
        grab = env.data.xpos[sel] + env.data.xmat[sel].reshape(3, 3) @ np.asarray(pert.localpos)
        env.data.xfrc_applied[sel, :3] += f
        env.data.xfrc_applied[sel, 3:] += np.cross(grab - env.data.xipos[sel], f)
    return np.asarray(env.data.xfrc_applied[sel][:3]).copy()


def draw(scn, mujoco, kind, rgba, label="", **kw) -> None:
    """Append one geom to the viewer's user scene, silently skipping a full buffer."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, float))
    if "connector" in kw:
        a, b, w = kw["connector"]
        mujoco.mjv_connector(g, kind, w, np.asarray(a, float), np.asarray(b, float))
    else:
        g.pos[:] = kw["pos"]
        g.size[:] = kw.get("size", [0.02, 0.02, 0.02])
    g.label = label[:99]
    scn.ngeom += 1


def render(v, mujoco, env, ids, labels, u, force) -> None:
    """Captions on the plates, an arrow for the motor torque, an arrow for whatever you are pulling."""
    with v.lock():
        sc = v.user_scn
        sc.ngeom = 0
        for bid, name in ids.items():
            draw(sc, mujoco, mujoco.mjtGeom.mjGEOM_LABEL, [1, 1, 1, 1], labels[name],
                 pos=np.asarray(env.data.xpos[bid]) + [0, 0, 0.045])
        hub = np.asarray(env.data.xpos[env.model.body("arm").id])
        tq = float(np.clip(u / env.max_voltage, -1, 1)) * 0.22
        draw(sc, mujoco, mujoco.mjtGeom.mjGEOM_ARROW,
             [0.25, 0.75, 1.0, 0.95] if tq >= 0 else [1.0, 0.55, 0.15, 0.95],
             f"motor {u:+.1f} V", connector=(hub, hub + [0, tq, 0], 0.012))
        if force is not None and (n := float(np.linalg.norm(force))) > 1e-6:
            pen = np.asarray(env.data.xipos[env.model.body("pendulum").id])
            draw(sc, mujoco, mujoco.mjtGeom.mjGEOM_ARROW, [0.95, 0.45, 0.10, 0.95],
                 f"your pull {n:.1f} N", connector=(pen, pen + 0.09 * force / n, 0.012))


def overlay(v, mujoco, rows) -> bool:
    """Draw `rows` of (label, value) in the viewer's corner. False if this build cannot, so the
    caller stops trying rather than raising once per frame."""
    try:
        v.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                      "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows))])
        return True
    except Exception:  # noqa: BLE001 -- an overlay is not worth losing the demo over
        return False


def run_viewer(env, pol, enc, args) -> None:
    """MuJoCo's own window. Click the plates on the table, or grab the pendulum with the mouse.

    The viewer runs passively, so this loop owns the physics and therefore also owns applying
    whatever the mouse is asking for. The plates are ordinary bodies with no joints, which means
    the viewer's picker reports them like any other body and a click becomes a command.
    """
    import mujoco
    import mujoco.viewer
    from tabpfn_control.mujoco_env import BUTTONS

    rng = np.random.default_rng(1)
    past: deque = deque(maxlen=max(enc.window, enc.k) + 2)
    ids = {env.model.body(f"btn_{n}").id: n for n, _, _ in BUTTONS}
    labels = {n: t for n, t, _ in BUTTONS}
    st = {"plant": describe({"Lp": env.Lp, "m_tip": env.m_tip, "motor_sign": np.sign(env.kt)}),
          "dither": args.dither, "ms": 0.0, "steps": 0,
          "overlay": True, "mode": "", "drag": False, "which": -1}
    queued: list = []

    def say(msg: str) -> None:
        print(f"[{st['steps'] * env.dt:7.1f}s] {msg}", flush=True)

    def on_key(keycode: int) -> None:                 # the plates are the interface; keys still work
        c = chr(keycode).lower() if 32 <= keycode < 127 else ""
        keys = {"s": "switch", "f": "flip", "r": "randomise", "0": "reset", "d": "dither",
                "[": "pushl", "]": "pushr"}
        if c in keys:
            queued.append(keys[c])

    def act_on(name: str) -> None:
        if name == "flip":
            info = env.set_plant(motor_scale=-1.0)
            st["plant"] = f"motor REVERSED (kt={info['kt']:+.2f})"
            say(f"motor polarity reversed -> {st['plant']}; nothing re-fit, the window has to notice")
        elif name == "switch":
            st["which"] = (st["which"] + 1) % len(TEST_PLANTS)
            xi = TEST_PLANTS[st["which"]]; retune(env, xi)
            st["plant"] = f"test plant {st['which'] + 1}/{len(TEST_PLANTS)}: " + describe(xi)
            say(f"switched to {st['plant']} — nothing re-fit, the window has to work it out")
        elif name == "randomise":
            xi = sample_plant(rng); retune(env, xi)
            st["plant"] = describe(xi); st["which"] = -1
            say(f"random pendulum -> {st['plant']}")
        elif name == "reset":
            env.reset(rng, hanging=True); past.clear(); say("reset to hanging, history cleared")
        elif name == "dither":
            st["dither"] = 0.0 if st["dither"] else args.dither
            say(f"exploration dither {'OFF — the plant is now unidentifiable from its own trace' if not st['dither'] else 'back on'}")
        elif name in ("pushl", "pushr"):
            env.push((-1 if name == "pushl" else 1) * args.push_torque)
            say(f"pushed {'left' if name == 'pushl' else 'right'} with {args.push_torque:.2f} N m")

    # The controller runs on its own thread so a slow forward pass cannot stall the window.
    # Physics advances in real time on the last action it was given — a zero-order hold, which is
    # exactly what a real rig does between controller updates — and rendering keeps its own clock.
    lock = threading.Lock()
    shared = {"u": 0.0, "state": env.state.copy(), "h": np.zeros(enc.n_features),
              "ctrl_ms": 0.0, "ctrl_n": 0}
    fresh = threading.Event()
    stop = threading.Event()

    def controller() -> None:
        while not stop.is_set():
            if not fresh.wait(0.5):          # nothing new to act on yet
                continue
            fresh.clear()
            with lock:
                s_now, hist = shared["state"].copy(), list(past)
            t = time.perf_counter()
            h = enc.encode(hist)
            u = float(pol.act(s_now, h))
            dt_ms = (time.perf_counter() - t) * 1000
            with lock:
                shared["u"], shared["h"] = u, h
                shared["ctrl_ms"], shared["ctrl_n"] = dt_ms, shared["ctrl_n"] + 1

    print("click a plate on the table, or double-click the pendulum then ctrl-drag it")
    print(f"'switch plants' cycles the {len(TEST_PLANTS)} test pendulums the policy was never fitted on")
    print("plates: " + "  |  ".join(t for _, t, _ in BUTTONS))
    print(f"physics {1 / env.dt:.0f} Hz in real time, rendering {args.render_fps:.0f} fps, "
          f"TabPFN on its own thread at whatever rate it manages")
    env.reset(rng, hanging=True); env.restyle()
    worker = threading.Thread(target=controller, daemon=True)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        v.cam.lookat[:] = [0.0, -0.06, 0.26]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.62, 96.0, -14.0
        with v.lock():
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True      # show what the mouse is doing
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
        worker.start()
        wall0, sim_t, frame_dt = time.perf_counter(), 0.0, 1.0 / args.render_fps
        next_frame, next_report, force, frames = wall0, wall0 + 2.0, None, 0
        try:
            while v.is_running():
                now = time.perf_counter()
                for _ in range(4):                    # catch up, but never at the window's expense
                    if sim_t >= now - wall0:
                        break
                    with v.lock():
                        if (sel := int(v.perturb.select)) in ids and not v.perturb.active:
                            queued.append(ids[sel]); v.perturb.select = 0
                    while queued:
                        act_on(queued.pop(0))
                    with lock:
                        u = shared["u"]
                    u = float(np.clip(u + rng.normal(0, st["dither"]), -env.max_voltage, env.max_voltage))
                    with v.lock():
                        force = apply_mouse(env, mujoco, v.perturb, args.drag_gain)
                    s_prev = env.state.copy()
                    s2 = env.step(u)
                    with lock:
                        past.append((s_prev, u, s2.copy()))
                        shared["state"] = s2.copy()
                    fresh.set()
                    sim_t += env.dt
                    st["steps"] += 1
                    now = time.perf_counter()
                if sim_t < now - wall0 - 1.0:         # fell badly behind: resynchronise, do not spiral
                    wall0, sim_t = now - sim_t - env.dt, sim_t

                if now >= next_frame:
                    frames += 1
                    next_frame = max(now, next_frame + frame_dt)
                    with lock:
                        h, ctrl_ms, ctrl_n = shared["h"].copy(), shared["ctrl_ms"], shared["ctrl_n"]
                    a = abs(wrap_angle(env.state[2]))
                    mode = "balancing" if a < 0.3 else "swinging up"
                    if mode != st["mode"]:
                        say(mode + (f" — caught it at {a:.3f} rad" if mode == "balancing" else ""))
                        st["mode"] = mode
                    if (force is not None) != st["drag"]:
                        st["drag"] = force is not None
                        say("you are pulling on the pendulum" if st["drag"] else "let go")
                    render(v, mujoco, env, ids, labels, shared["u"], force)
                    if st["overlay"]:
                        g = float(h[POLARITY_COL]) if enc.n_features > POLARITY_COL else 0.0
                        st["overlay"] = overlay(v, mujoco, [
                            ("plant", st["plant"]),
                            ("identified gain", "(window still filling)" if g == 0 else
                             f"{g:+.3f}, so the motor pushes {'forward' if g > 0 else 'BACKWARD'}"),
                            ("acting", f"{mode}, |alpha| {a:.3f} rad"),
                            ("TabPFN", f"{ctrl_ms:.0f} ms per call, {ctrl_n} calls, own thread"),
                            ("rates", f"physics {1 / env.dt:.0f} Hz, render {args.render_fps:.0f} fps"),
                            ("dither", f"{st['dither']:.2f} V"),
                            ("mouse", "pulling" if st["drag"] else "double-click the pendulum, then ctrl-drag"),
                        ])
                    v.sync()
                    if now >= next_report:
                        say(f"{mode}, |alpha| {a:.3f} rad, gain {h[POLARITY_COL]:+.3f} | "
                            f"TabPFN {ctrl_ms:.0f} ms ({ctrl_n / max(now - wall0, 1e-9):.1f} Hz), "
                            f"render {frames / max(now - wall0, 1e-9):.0f} fps, "
                            f"sim {'on' if abs(sim_t - (now - wall0)) < 0.2 else 'behind'} real time")
                        next_report = now + 5.0
                if (idle := min(next_frame, wall0 + sim_t) - time.perf_counter()) > 0:
                    time.sleep(min(idle, frame_dt))
        finally:
            stop.set(); fresh.set(); worker.join(timeout=1.0)
    el = time.perf_counter() - wall0
    print(f"{st['steps']} control steps in {el:.0f}s | TabPFN {shared['ctrl_n']} calls "
          f"({shared['ctrl_n'] / max(el, 1e-9):.1f} Hz) | {frames / max(el, 1e-9):.0f} fps rendered")


def run_web(env, pol, enc, args) -> None:
    import mujoco
    import viser
    from mjviser import Viewer

    rng = np.random.default_rng(1)
    past: deque = deque(maxlen=max(enc.window, enc.k) + 2)
    lock = threading.Lock()
    pending = {"push": 0.0, "reset": False, "flip": False, "randomise": False}
    state = {"ms": 0.0, "calls": 0,
             "plant": describe({"Lp": env.Lp, "m_tip": env.m_tip, "motor_sign": 1.0})}

    server = viser.ViserServer(host="0.0.0.0", port=args.port, label="One frozen TabPFN table controls this pendulum")
    with server.gui.add_folder("Disturb it"):
        b_l = server.gui.add_button("push  <--")
        b_r = server.gui.add_button("push  -->")
        s_tq = server.gui.add_slider("torque [N m]", min=0.05, max=0.6, step=0.05, initial_value=0.3)
    with server.gui.add_folder("Change the pendulum (nothing is re-fit)"):
        b_flip = server.gui.add_button("reverse motor polarity", color="red")
        b_rand = server.gui.add_button("randomise mass / damping / gain", color="orange")
        b_reset = server.gui.add_button("reset (hanging)")
        t_plant = server.gui.add_text("plant", initial_value=state["plant"], disabled=True)
    with server.gui.add_folder("What the table sees"):
        t_gain = server.gui.add_text("identified control gain", initial_value="(filling)", disabled=True)
        t_mode = server.gui.add_text("acting", initial_value="-", disabled=True)
        t_lat = server.gui.add_text("step time", initial_value="-", disabled=True)
        t_ang = server.gui.add_text("|angle| from upright", initial_value="-", disabled=True)
        s_dither = server.gui.add_slider("exploration dither [V]", min=0.0, max=1.0, step=0.05, initial_value=args.dither)
    server.gui.add_markdown(
        "**One frozen table. One forward pass per step.** Rows are `[state, 10 numbers summarising "
        "the last second] -> voltage`, cloned from planner demonstrations on 30 randomised pendulums. "
        "This one is not among them and is never named.\n\n"
        "Reverse the motor or randomise the plant: **nothing is re-fit**. The window refills with "
        "post-change data and the same table reads a different pendulum out of it. Watch the identified "
        "gain flip sign, then the pendulum come back.\n\n"
        "*Dither matters: under a deterministic controller a plant is unidentifiable from its own trace.*")

    @b_l.on_click
    def _(_): pending.update(push=-float(s_tq.value))

    @b_r.on_click
    def _(_): pending.update(push=float(s_tq.value))

    @b_flip.on_click
    def _(_): pending.update(flip=True)

    @b_rand.on_click
    def _(_): pending.update(randomise=True)

    @b_reset.on_click
    def _(_): pending.update(reset=True)

    n_sub, counter = env.n_substeps, {"k": 0}

    def step_fn(m, d) -> None:
        k = counter["k"]; counter["k"] = k + 1
        if k % n_sub:
            mujoco.mj_step(m, d); return
        if pending["flip"]:
            info = env.set_plant(motor_scale=-1.0); pending["flip"] = False
            state["plant"] = f"motor REVERSED (kt={info['kt']:+.2f})"; t_plant.value = state["plant"]
        if pending["randomise"]:
            xi = sample_plant(rng); retune(env, xi); pending["randomise"] = False
            state["plant"] = describe(xi)
            t_plant.value = state["plant"]
        if pending["reset"]:
            env.reset(rng, hanging=True); pending["reset"] = False; state.pop("prev", None)

        s = env._read_state()
        t = time.perf_counter()
        with lock:
            h = enc.encode(list(past)); u = pol.act(s, h)
        state["ms"] = (time.perf_counter() - t) * 1000; state["calls"] += 1
        u = float(np.clip(u + rng.normal(0, float(s_dither.value)), -env.max_voltage, env.max_voltage))
        d.ctrl[0] = u
        d.qfrc_applied[env._v_pend] = pending["push"]; pending["push"] = 0.0
        if "prev" in state:
            past.append((state["prev"], state["prev_u"], s.copy()))
        state["prev"], state["prev_u"] = s.copy(), u
        g = float(h[POLARITY_COL]) if enc.n_features > POLARITY_COL else 0.0
        a = abs(wrap_angle(s[2]))
        t_gain.value = "(filling)" if g == 0 else f"{g:+.3f}  ->  motor pushes {'FORWARD' if g > 0 else 'BACKWARD'}"
        t_mode.value = "balancing" if a < 0.3 else "swinging up"
        t_lat.value = f"{state['ms']:.0f} ms of {env.dt * 1000:.0f} ms | {state['calls']} calls | 1 per step"
        t_ang.value = f"{a:.3f} rad"
        if (k // n_sub) % 100 == 0:
            print(f"t={k * m.opt.timestep:6.1f}s |alpha|={a:.3f} u={u:+5.1f}V gain={g:+.3f} "
                  f"{state['ms']:.0f}ms plant={state['plant']}", flush=True)
        mujoco.mj_step(m, d)

    def reset_fn(m, d) -> None:
        env.reset(rng, hanging=True); state.pop("prev", None)

    print(f"open the printed viser URL (port {args.port})")
    Viewer(env.model, env.data, step_fn=step_fn, reset_fn=reset_fn, server=server).run()


def run_record(env, pol, enc, args) -> None:
    """Scripted run -> mp4 + gif: swing up, get pushed, motor reversed, plant randomised."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(3)
    past: deque = deque(maxlen=max(enc.window, enc.k) + 2)
    # one second of random voltages before control starts: a plant is not identifiable from a trace
    # in which the input never varies, so the window has to be filled with something informative
    s = env.reset(rng, hanging=False)
    for _ in range(20):
        u = float(rng.uniform(-6, 6)); sp = s.copy(); s = env.step(u); past.append((sp, u, s.copy()))
        if abs(wrap_angle(s[2])) > 1.2:
            s = env.reset(rng, hanging=False)
    s = env.reset(rng, hanging=True)
    n = int(args.seconds / env.dt)
    events = {int(t / env.dt): e for t, e in
              [(8.0, "push"), (14.0, "flip"), (24.0, "push"), (30.0, "randomise")] if t < args.seconds}
    log = {"t": [], "alpha": [], "gain": [], "event": [], "ms": [], "plant": []}
    # The controller decides 20 times a second; the integrator runs at 500 Hz. Sampling poses
    # between decisions is what separates a smooth video from a slideshow of control steps.
    sub = max(1, env.n_substeps // args.capture_per_step)
    poses: list = []                         # (qpos, control step) at ~100 Hz, replayed afterwards
    # ...and with it the shape of the pendulum at that moment. The rig changes mid-run, so replaying
    # joint angles alone would draw the whole clip with whatever pendulum it happened to end on.
    rig = []
    plant = describe({"Lp": env.Lp, "m_tip": env.m_tip, "motor_sign": 1.0})
    for k in range(n):
        ev = events.get(k, "")
        if ev == "push":
            env.push(rng.choice([-1, 1]) * args.push_torque)
        elif ev == "flip":
            env.set_plant(motor_scale=-1.0); plant = "MOTOR REVERSED"   # set_plant restyles the rig
        elif ev == "randomise":
            xi = sample_plant(rng); retune(env, xi)
            plant = "randomised: " + describe(xi)
        t0 = time.perf_counter()
        h = enc.encode(list(past)); u = pol.act(s, h)
        ms = (time.perf_counter() - t0) * 1000
        u = float(np.clip(u + rng.normal(0, args.dither), -env.max_voltage, env.max_voltage))
        sp = s.copy()
        s = env.step(u, capture=lambda q, k=k: poses.append((q, k)), capture_every=sub)
        past.append((sp, u, s.copy()))
        log["t"].append(k * env.dt); log["alpha"].append(float(wrap_angle(s[2])))
        log["gain"].append(float(h[POLARITY_COL]) if enc.n_features > POLARITY_COL else 0.0)
        log["event"].append(ev); log["ms"].append(ms); log["plant"].append(plant)
        rig.append((round(env.Lp, 6), round(env.m_tip, 6), float(np.sign(env.kt))))
    print(f"controller step: median {np.median(log['ms']):.0f} ms, p95 {np.percentile(log['ms'], 95):.0f} ms "
          f"(budget {env.dt * 1000:.0f} ms)")
    for k, ev in events.items():
        after = np.abs(np.array(log["alpha"])[k:])
        back = next((i for i in range(len(after) - 20) if (after[i:i + 20] < 0.15).all()), None)
        print(f"  {ev:10s} at {k * env.dt:4.1f}s -> back upright after "
              f"{'never' if back is None else f'{back * env.dt:.1f}s'}")

    import mujoco
    renderer, cam = make_renderer(env)
    shown = None                                      # the rig currently loaded into the renderer
    al = np.array(log["alpha"]); frames = []
    fps = args.capture_per_step / env.dt
    fig = plt.figure(figsize=(12.8, 5.6), dpi=args.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.25], hspace=0.5, wspace=0.12)
    ax_p, ax_g, ax_t = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[0, 1])
    for q, k in poses:
        ax_p.clear(); ax_p.axis("off")
        if renderer is not None:
            if rig[k] != shown:                       # put the rig back the way it was at frame k
                Lp_k, mt_k, sign_k = rig[k]
                env.set_geometry(Lp=Lp_k, m_tip=mt_k)
                if (sign_k < 0) != (env.kt < 0):
                    env.set_plant(motor_scale=-1.0)
                env.restyle(); shown = rig[k]
            env.data.qpos[:] = q; env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=cam)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            ax_p.imshow(renderer.render())
        else:
            ax_p.set_aspect("equal"); ax_p.set_xlim(-1.3, 1.3); ax_p.set_ylim(-1.3, 1.3)
            ax_p.plot([0, np.sin(al[k])], [0, np.cos(al[k])], lw=6, color="#1f77b4")
            ax_p.plot([np.sin(al[k])], [np.cos(al[k])], "o", ms=13, color="#d62728")
        recent = [log["event"][j] for j in range(max(0, k - 8), k + 1) if log["event"][j]]
        ax_p.set_title(("PUSHED" if "push" in recent else "MOTOR REVERSED" if "flip" in recent
                        else "NEW PENDULUM" if "randomise" in recent else log["plant"][k]),
                       fontsize=12.5, color=("#e5a000" if recent else "#555"),
                       weight=("bold" if recent else "normal"), pad=6)
        ax_t.clear(); ax_t.plot(log["t"][:k + 1], al[:k + 1], color="#333", lw=1.2)
        ax_t.axhline(0, ls=":", c="#888"); ax_t.set_xlim(0, args.seconds); ax_t.set_ylim(-3.4, 3.4)
        ax_t.set_ylabel("angle from upright [rad]")
        ax_g.clear(); ax_g.plot(log["t"][:k + 1], log["gain"][:k + 1], color="#2d6cdf", lw=1.4)
        ax_g.axhline(0, ls=":", c="#888"); ax_g.set_xlim(0, args.seconds)
        ax_g.set_ylabel("identified gain"); ax_g.set_xlabel("time [s]")
        for j, e in events.items():
            if j <= k:
                for ax in (ax_t, ax_g):
                    ax.axvline(log["t"][j], color="#e5a000", alpha=0.7, lw=1.2)
        fig.suptitle(f"One frozen TabPFN table  |  t = {log['t'][k]:5.1f} s  |  "
                     f"one forward pass per 50 ms step, nothing re-fit", fontsize=12)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    print(f"{len(frames)} frames at {fps:.0f} fps, {frames[0].width}x{frames[0].height}")

    out = Path(args.record); out.parent.mkdir(parents=True, exist_ok=True)
    # the gif is the README's inline preview, so keep it light: every other frame, 60 % scale
    step = max(1, round(fps / args.gif_fps))
    gif = [f.resize((int(f.width * args.gif_scale), int(f.height * args.gif_scale)), Image.LANCZOS)
           .convert("P", palette=Image.ADAPTIVE, colors=args.gif_colors) for f in frames[::step]]
    gif[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif[1:],
                duration=round(1000 * step / fps), loop=0, optimize=True)
    print(f"saved {out.with_suffix('.gif')}  {len(gif)} frames, {gif[0].width}x{gif[0].height}, "
          f"{fps / step:.0f} fps, {out.with_suffix('.gif').stat().st_size / 1e6:.1f} MB")
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(out.with_suffix(".mp4")), fps=fps, codec="libx264",
                                quality=9, macro_block_size=2, ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
            for f in frames:
                w.append_data(np.asarray(f))
        print(f"saved {out.with_suffix('.mp4')}  {fps:.0f} fps, "
              f"{out.with_suffix('.mp4').stat().st_size / 1e6:.1f} MB")
    except Exception as e:  # noqa: BLE001
        print(f"mp4 not written ({e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/policy_table.npz")
    ap.add_argument("--web", action="store_true", help="browser via mjviser, for a machine with no display")
    ap.add_argument("--viewer", action="store_true", help="native MuJoCo window with mouse disturbances")
    ap.add_argument("--record", default=None, metavar="PATH", help="scripted run -> mp4 + gif")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--push-torque", type=float, default=0.3)
    ap.add_argument("--dither", type=float, default=0.15)
    ap.add_argument("--drag-gain", type=float, default=60.0, help="[viewer] how hard a mouse drag pulls")
    ap.add_argument("--render-fps", type=float, default=30.0, help="[viewer] frames per second to aim for")
    ap.add_argument("--capture-per-step", type=int, default=3,
                    help="[record] poses sampled per 50 ms control step; 3 gives a 60 fps clip")
    ap.add_argument("--dpi", type=int, default=150, help="[record] figure resolution")
    ap.add_argument("--gif-fps", type=float, default=20.0,
                    help="[record] frame rate of the gif, which pays for every frame in file size")
    ap.add_argument("--gif-scale", type=float, default=0.45, help="[record] gif size, relative to the mp4")
    ap.add_argument("--gif-colors", type=int, default=80, help="[record] gif palette size")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.viewer and (why := viewer_blocker()):
        print(why)                                  # before the fit, not after twenty seconds of it
        raise SystemExit(1)
    if args.viewer and args.device == "auto" and sys.platform == "darwin":
        # On a Mac the viewer already leans on the GPU every frame, and TabPFN's automatic choice
        # there is Metal. Both on one device makes each contend with the other, and the window is
        # the part a person notices. The context table is small enough that the CPU keeps up.
        args.device = "cpu"
        print("macOS: running TabPFN on the CPU so the viewer keeps the GPU (--device mps to override)")
    # Start on a pendulum from the middle of the family the policy was taught, not on the bare-rod
    # default, so what you are looking at is a plant the table has any business driving.
    env = build_plant({"Lp": float(np.mean(LP_RANGE)), "m_tip": float(np.mean(MTIP_RANGE)),
                       "motor_sign": 1.0}, buttons=args.viewer)   # plates only in the clickable window
    pol, enc = load_policy(env, args.table, args.device)
    if args.record:
        run_record(env, pol, enc, args)
    elif args.viewer:
        run_viewer(env, pol, enc, args)
    else:
        run_web(env, pol, enc, args)


if __name__ == "__main__":
    main()
