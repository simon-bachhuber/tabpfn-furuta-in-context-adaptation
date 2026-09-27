"""The demonstration teacher: MPPI over the simulator, then that plant's own LQR to hold.

Privileged at collection time only — it reads the true dynamics and the true linearisation. Neither
exists at test time, and the plants it teaches on are disjoint from the ones the policy is evaluated
on. Being closed-loop is the point: unlike an open-loop pumping law it also demonstrates recoveries
from states off the nominal path, which is the coverage a cloned policy needs.
"""

from __future__ import annotations

import numpy as np

from tabpfn_control.envs import wrap_angle
from tabpfn_control.lqr import LQRController, dlqr
from tabpfn_control.planners import MPPI, HybridController, OracleModel


def planner_episode(env, rng, steps: int = 300, dither: float = 0.5, stop_after: int | None = 30,
                    push_prob: float = 0.015, push_torque: float = 0.3, horizon: int = 30,
                    n_samples: int = 500, catch_vel_tol: float = 20.0, warmup: int = 20):
    """One demonstration. Returns (states, voltages, teacher labels, next states, trainable) or None.

    The first `warmup` steps are random voltages, and `trainable` is False for them. They exist so
    the history window is already full when the planner starts, which is what the controller sees at
    run time and, more to the point, is what makes the opening of the swing-up usable at all. The
    ridge statistic needs sixteen transitions before it means anything, so without a preamble every
    demonstration silently threw away its first sixteen rows — the most valuable ones it had, the
    moment a pendulum starts moving from rest.
    """
    Ad, Bd = env.linearize_upright_discrete()
    try:
        K = dlqr(Ad, Bd, np.diag([1, 0.1, 10, 0.1]), np.array([[0.1]]))
    except Exception:  # noqa: BLE001
        return None
    ctrl = HybridController(
        MPPI(env, horizon=horizon, n_samples=n_samples, lam=0.5, sigma=5.0, replan_every=4,
             rng=np.random.default_rng(int(rng.integers(1 << 30)))),
        LQRController(K, np.zeros(4), (0, 2), env.max_voltage),
        lambda x: wrap_angle(x[2]), vel_fn=lambda x: x[1], catch_vel_tol=catch_vel_tol)
    ctrl.reset()
    model = OracleModel(env)
    S, U, L, S2, train = [], [], [], [], []
    s = env.reset(rng, hanging=True)
    for _ in range(warmup):                       # fill the window; not demonstrations of anything
        u = float(rng.uniform(-env.max_voltage, env.max_voltage))
        s2 = env.step(u)
        S.append(s.copy()); U.append(u); L.append(u); S2.append(s2.copy()); train.append(False)
        s = s2
    s = env.reset(rng, hanging=True)              # back to hanging, window kept
    caught = None
    for t in range(steps):
        lab = float(np.clip(ctrl.act(model, s), -env.max_voltage, env.max_voltage))
        u = float(np.clip(lab + rng.normal(0, dither), -env.max_voltage, env.max_voltage))
        if abs(wrap_angle(s[2])) < 0.3 and rng.random() < push_prob:
            env.push(rng.choice([-1, 1]) * push_torque)     # make the planner show recoveries too
            caught = None
        s2 = env.step(u)
        S.append(s.copy()); U.append(u); L.append(lab); S2.append(s2.copy()); train.append(True)
        s = s2
        if caught is None and abs(wrap_angle(s[2])) < 0.15:
            caught = t
        if stop_after is not None and caught is not None and t - caught >= stop_after:
            break
    return np.array(S), np.array(U), np.array(L), np.array(S2), np.array(train)


# The plant family the frozen policy has to cover. Three quantities vary and nothing else:
#
#   Lp       how long the pendulum is
#   m_tip    how much mass is bolted to its far end
#   sign     which way the motor turns
#
# The first two set the inertia and the energy a swing-up has to inject; the third inverts the sign
# of every control decision.
#
# The bounds are measured, not guessed. A planner with full knowledge of each plant was run over a
# grid of candidates, and these are the limits inside which it swings up and holds on every seed and
# both polarities (16/16 at each corner). Outside them the rig runs out of motor: at a 65 g tip the
# long end drops to 8/12, at 80 g it fails at every length, and 110 g fails outright. That ceiling
# is the plant and not the teacher — re-running the 80 g edge with a much stronger planner (horizon
# 45, 1200 samples, a tighter catch window) did not improve it. A family the teacher cannot
# demonstrate is not a fair thing to ask a frozen policy to interpolate over.
#
# Reproduce with:  python scripts/evaluate.py --feasibility
LP_RANGE = (0.20, 0.40)          # metres
MTIP_RANGE = (0.01, 0.05)        # kilograms bolted to the tip
ROD_MASS = 0.02                  # the rod itself stays light, so the tip mass is what matters


def sample_plant(rng, allow_flip: bool = True) -> dict:
    """Hidden parameters of one pendulum, drawn at random. The model is never told these."""
    return {"Lp": float(rng.uniform(*LP_RANGE)),
            "m_tip": float(rng.uniform(*MTIP_RANGE)),
            "motor_sign": float(rng.choice([1.0, -1.0])) if allow_flip else 1.0}


def sample_plants(rng, n: int, allow_flip: bool = True) -> list:
    """`n` pendulums that actually cover the family, rather than n independent draws.

    Uniform sampling leaves holes. With thirty independent draws the long, heavy end of the box
    routinely gets two or three examples and the policy then fails there, which is exactly what
    happened the first time this was trained. A Latin hypercube puts one plant in every length band
    and every mass band, so coverage is guaranteed rather than hoped for, and the motor directions
    are dealt out evenly instead of flipped coin by coin.
    """
    edges = lambda lo, hi: lo + (np.arange(n) + rng.random(n)) * (hi - lo) / n  # noqa: E731
    lp = rng.permutation(edges(*LP_RANGE))
    mt = rng.permutation(edges(*MTIP_RANGE))
    sign = rng.permutation(np.where(np.arange(n) < n // 2, 1.0, -1.0)) if allow_flip else np.ones(n)
    return [{"Lp": float(lp[i]), "m_tip": float(mt[i]), "motor_sign": float(sign[i])} for i in range(n)]


# Eight pendulums the policy has never been fitted on, fixed so the figure, the demo's plant cycle
# and the verification all talk about the same machines.
#
# They sit *inside* the region the training plants cover, not on its boundary, because the claim is
# interpolation: each of these is surrounded by machines the table has seen, and none of them is one
# of those machines. Four motors forward, four reversed.
TEST_PLANTS = [
    {"Lp": 0.23, "m_tip": 0.018, "motor_sign": +1.0},
    {"Lp": 0.23, "m_tip": 0.042, "motor_sign": -1.0},
    {"Lp": 0.37, "m_tip": 0.018, "motor_sign": -1.0},
    {"Lp": 0.37, "m_tip": 0.042, "motor_sign": +1.0},
    {"Lp": 0.30, "m_tip": 0.030, "motor_sign": +1.0},
    {"Lp": 0.26, "m_tip": 0.036, "motor_sign": -1.0},
    {"Lp": 0.34, "m_tip": 0.022, "motor_sign": +1.0},
    {"Lp": 0.31, "m_tip": 0.044, "motor_sign": -1.0},
]


def build_plant(xi: dict, **kw):
    """One pendulum from the family, as a fresh environment."""
    from tabpfn_control.mujoco_env import MujocoFuruta  # noqa: PLC0415

    env = MujocoFuruta(Lp=xi["Lp"], mp=ROD_MASS, m_tip=xi["m_tip"], **kw)
    if xi.get("motor_sign", 1.0) < 0:
        env.set_plant(motor_scale=-1.0)
    env.restyle()
    return env
