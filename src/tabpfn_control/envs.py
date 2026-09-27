"""The Furuta (rotary inverted) pendulum, as a hand-derived Lagrangian model.

State ``(phi, phi_dot, alpha, alpha_dot)``: ``phi`` is the driven arm angle, ``alpha`` the pendulum
angle with ``alpha = 0`` upright and ``pi`` hanging. The input is motor voltage. Integration is RK4
at ``dt = 0.05`` s, the control period.

:class:`~tabpfn_control.mujoco_env.MujocoFuruta` is an independent MuJoCo implementation of the same
machine; the two agree to 5 % on accelerations (``tests/test_envs.py``), which is what makes the
MuJoCo plant trustworthy as the thing being controlled. ``make_env("furuta-mujoco")`` returns it and
is what every script uses; the model here is the reference twin.

Both expose ``reset()``, ``step(u)``, ``state``, ``set_state()``, ``cost()`` and ``features()``,
the TabPFN input encoding of a state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap to [-pi, pi)."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


# ---------------------------------------------------------------------------
# Furuta pendulum (Qube-Servo 2)
# ---------------------------------------------------------------------------


@dataclass
class Furuta:
    """Rotary inverted pendulum, alpha = 0 upright.

    Equations of motion (derived from the Lagrangian; see PLAN.md §3.1 and
    tests/test_envs.py for the checks):

        J0 = Jr + mp*Lr^2, J1 = mp*Lr*lp, J2 = mp*lp^2 + Jp

        (J0 + J2 sin^2 a) phi'' + J1 cos a  a'' + 2 J2 sin a cos a phi' a' - J1 sin a a'^2 = tau - Dr phi'
         J1 cos a          phi'' + J2       a'' - J2 sin a cos a phi'^2 - mp g lp sin a       = -Dp a'

    Motor: tau = kt (V - km phi') / Rm.
    """

    # Default: a DIY-scale rig (30 cm pendulum, 15 cm arm, hobby motor). `Furuta.qube()`
    # returns Quanser Qube-Servo 2 parameters instead; that rig's faster unstable pole
    # (10.6 rad/s) and weaker motor make it a poor match for a 20 Hz control loop.
    Rm: float = 4.0
    kt: float = 0.1
    km: float = 0.1
    mr: float = 0.15
    Lr: float = 0.15
    Dr: float = 0.003
    mp: float = 0.05
    Lp: float = 0.30
    Dp: float = 0.001
    g: float = 9.81
    dt: float = 0.05
    n_substeps: int = 10
    max_voltage: float = 10.0
    max_speed: float = 30.0  # rad/s, safety clip on both joints
    horizon: int = 160
    delta_mode: str = "full"  # "full": [d alpha, d alpha_dot, d phi_dot]; "accel": [d alpha_dot, d phi_dot], angles integrated
    state: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, np.pi, 0.0]))

    n_state = 4
    n_action = 1
    action_low = np.array([-10.0])
    action_high = np.array([10.0])
    name = "furuta"

    @classmethod
    def qube(cls, **kw) -> "Furuta":
        """Quanser Qube-Servo 2 nominal parameters."""
        return cls(Rm=8.4, kt=0.042, km=0.042, mr=0.095, Lr=0.085, Dr=0.0015, mp=0.024, Lp=0.129, Dp=0.0005, **kw)

    def __post_init__(self) -> None:
        self.Jr = self.mr * self.Lr**2 / 12.0
        self.Jp = self.mp * self.Lp**2 / 12.0
        self.lp = self.Lp / 2.0
        self.J0 = self.Jr + self.mp * self.Lr**2
        self.J1 = self.mp * self.Lr * self.lp
        self.J2 = self.mp * self.lp**2 + self.Jp

    def reset(self, rng: np.random.Generator | None = None, hanging: bool = True) -> np.ndarray:
        rng = rng or np.random.default_rng()
        a0 = np.pi if hanging else 0.0
        self.state = np.array([0.0, 0.0, a0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])
        self.state[2] = wrap_angle(self.state[2])
        return self.state.copy()

    def set_state(self, s: np.ndarray) -> None:
        self.state = np.array(s, dtype=float)

    def accelerations(self, s: np.ndarray, V: np.ndarray, damping: bool = True) -> tuple[np.ndarray, np.ndarray]:
        phi_d, a, a_d = s[..., 1], s[..., 2], s[..., 3]
        sa, ca = np.sin(a), np.cos(a)
        tau = self.kt * (V - self.km * phi_d) / self.Rm
        Dr, Dp = (self.Dr, self.Dp) if damping else (0.0, 0.0)
        # M [phi'', a'']^T = rhs
        m11 = self.J0 + self.J2 * sa**2
        m12 = self.J1 * ca
        m22 = self.J2
        r1 = tau - Dr * phi_d - 2 * self.J2 * sa * ca * phi_d * a_d + self.J1 * sa * a_d**2
        r2 = -Dp * a_d + self.J2 * sa * ca * phi_d**2 + self.mp * self.g * self.lp * sa
        det = m11 * m22 - m12 * m12
        phi_dd = (m22 * r1 - m12 * r2) / det
        a_dd = (m11 * r2 - m12 * r1) / det
        return phi_dd, a_dd

    def _deriv(self, s: np.ndarray, V: np.ndarray, damping: bool = True) -> np.ndarray:
        phi_dd, a_dd = self.accelerations(s, V, damping)
        return np.stack([s[..., 1], phi_dd, s[..., 3], a_dd], axis=-1)

    def dynamics(self, s: np.ndarray, V: np.ndarray, damping: bool = True) -> np.ndarray:
        """Vectorised RK4 step: states (N,4), V (N,) -> next states (N,4)."""
        V = np.clip(V, -self.max_voltage, self.max_voltage)
        h = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            k1 = self._deriv(s, V, damping)
            k2 = self._deriv(s + 0.5 * h * k1, V, damping)
            k3 = self._deriv(s + 0.5 * h * k2, V, damping)
            k4 = self._deriv(s + h * k3, V, damping)
            s = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        s = s.copy()
        s[..., 1] = np.clip(s[..., 1], -self.max_speed, self.max_speed)
        s[..., 3] = np.clip(s[..., 3], -self.max_speed, self.max_speed)
        s[..., 0] = wrap_angle(s[..., 0])
        s[..., 2] = wrap_angle(s[..., 2])
        return s

    def step(self, u: float | np.ndarray) -> np.ndarray:
        self.state = self.dynamics(self.state, np.asarray(u, dtype=float).reshape(()))
        return self.state.copy()

    def energy(self, s: np.ndarray) -> np.ndarray:
        """Total mechanical energy (for the conservation test)."""
        phi_d, a, a_d = s[..., 1], s[..., 2], s[..., 3]
        T = 0.5 * (self.J0 + self.J2 * np.sin(a) ** 2) * phi_d**2 + self.J1 * np.cos(a) * phi_d * a_d + 0.5 * self.J2 * a_d**2
        Vp = self.mp * self.g * self.lp * np.cos(a)
        return T + Vp

    def cost(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi, phi_d, a, a_d = wrap_angle(s[..., 0]), s[..., 1], wrap_angle(s[..., 2]), s[..., 3]
        u = np.clip(np.asarray(u), -self.max_voltage, self.max_voltage).reshape(a.shape)
        return a**2 + 0.01 * a_d**2 + 0.1 * phi**2 + 0.005 * phi_d**2 + 0.001 * u**2

    def upright(self, s: np.ndarray, tol: float = 0.15) -> np.ndarray:
        return np.abs(wrap_angle(s[..., 2])) < tol

    @staticmethod
    def features(s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """TabPFN input row(s): cos a, sin a, a_dot, phi_dot, V (phi omitted: dynamics are invariant to it)."""
        phi_d, a, a_d = s[..., 1], s[..., 2], s[..., 3]
        u = np.asarray(u, dtype=float).reshape(a.shape)
        return np.stack([np.cos(a), np.sin(a), a_d, phi_d, u], axis=-1)

    def delta(self, s: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Targets. full: [d alpha (wrapped), d alpha_dot, d phi_dot]; accel: [d alpha_dot, d phi_dot]."""
        dad, dpd = s2[..., 3] - s[..., 3], s2[..., 1] - s[..., 1]
        if self.delta_mode == "accel":
            return np.stack([dad, dpd], axis=-1)
        return np.stack([wrap_angle(s2[..., 2] - s[..., 2]), dad, dpd], axis=-1)

    def apply_delta(self, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self.delta_mode == "accel":
            a_d2, phi_d2 = s[..., 3] + d[..., 0], s[..., 1] + d[..., 1]
            a2 = wrap_angle(s[..., 2] + 0.5 * (s[..., 3] + a_d2) * self.dt)
        else:
            a_d2, phi_d2 = s[..., 3] + d[..., 1], s[..., 1] + d[..., 2]
            a2 = wrap_angle(s[..., 2] + d[..., 0])
        phi2 = wrap_angle(s[..., 0] + 0.5 * (s[..., 1] + phi_d2) * self.dt)
        return np.stack([phi2, phi_d2, a2, a_d2], axis=-1)

    def linearize_upright(self) -> tuple[np.ndarray, np.ndarray]:
        """Continuous-time (A, B) at the upright equilibrium, from the analytic EOM."""
        J0, J1, J2, k = self.J0, self.J1, self.J2, self.mp * self.g * self.lp
        det = J0 * J2 - J1**2
        kt_R = self.kt / self.Rm
        # states x = [phi, phi_d, alpha, alpha_d]; alpha small, sin a ~ a, cos a ~ 1
        # M [phi'', a''] = [tau - Dr phi_d, -Dp a_d + k a]  with M = [[J0, J1],[J1, J2]]
        A = np.zeros((4, 4))
        B = np.zeros((4, 1))
        A[0, 1] = 1.0
        A[2, 3] = 1.0
        # phi'' = (J2 r1 - J1 r2)/det ; a'' = (J0 r2 - J1 r1)/det
        # r1 = kt_R (V - km phi_d) - Dr phi_d ; r2 = -Dp a_d + k a
        c1 = -(kt_R * self.km + self.Dr)
        A[1, 1] = J2 * c1 / det
        A[1, 2] = -J1 * k / det
        A[1, 3] = J1 * self.Dp / det
        A[3, 1] = -J1 * c1 / det
        A[3, 2] = J0 * k / det
        A[3, 3] = -J0 * self.Dp / det
        B[1, 0] = J2 * kt_R / det
        B[3, 0] = -J1 * kt_R / det
        return A, B


def make_env(name: str, delta_mode: str = "full", **kw):
    if name == "furuta-mujoco":
        from tabpfn_control.mujoco_env import MujocoFuruta  # noqa: PLC0415

        return MujocoFuruta(delta_mode=delta_mode, **kw)
    if name == "furuta-qube":
        env = Furuta.qube(max_voltage=5.0, delta_mode=delta_mode, **kw)
        env.action_low, env.action_high, env.name = np.array([-5.0]), np.array([5.0]), "furuta-qube"
        return env
    return Furuta(delta_mode=delta_mode)
