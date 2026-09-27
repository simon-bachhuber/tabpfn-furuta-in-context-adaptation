"""Sampling-based MPC (MPPI) over a one-step model.

The model interface is a callable ``model(states, actions) -> (next_states, std)``
where ``states`` is (N, n_state), ``actions`` (N,), ``next_states`` (N, n_state)
and ``std`` (N, n_state) or ``None``. The true simulator and the TabPFN model
both implement it (see :class:`OracleModel` and ``model.TabPFNDynamics``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class OracleModel:
    """The true simulator as a model (upper bound for every learned model)."""

    def __init__(self, env):
        self.env = env
        self.n_calls = 0

    def __call__(self, s: np.ndarray, u: np.ndarray):
        self.n_calls += 1
        return self.env.dynamics(s, u), None


@dataclass
class MPPI:
    """Model predictive path integral control with k-step replanning.

    Parameters
    ----------
    horizon: planning steps H
    n_samples: candidate sequences N
    lam: MPPI temperature
    sigma: exploration std of the action perturbations (in action units)
    replan_every: execute this many actions of the plan before replanning
    unc_weight: beta for the uncertainty term (positive = pessimistic)
    terminal_value: optional callable states -> cost-to-go added at the horizon end
    """

    env: object
    horizon: int = 25
    n_samples: int = 1000
    lam: float = 1.0
    sigma: float = 1.0
    replan_every: int = 1
    unc_weight: float = 0.0
    gamma: float = 1.0
    terminal_value: object | None = None
    noise: str = "white"  # "white" or "spline" (smooth perturbations, better for rhythmic tasks)
    n_knots: int = 8
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    plan: np.ndarray | None = None
    _queue: list = field(default_factory=list)

    def reset(self) -> None:
        self.plan = np.zeros(self.horizon)
        self._queue = []

    def rollout_cost(self, model, s0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """U: (N, H) action sequences -> total cost (N,)."""
        N, H = U.shape
        s = np.repeat(s0[None, :], N, axis=0)
        total = np.zeros(N)
        disc = 1.0
        for t in range(H):
            u = np.clip(U[:, t], self.env.action_low[0], self.env.action_high[0])
            total += disc * self.env.cost(s, u)
            s, std = model(s, u)
            if std is not None and self.unc_weight != 0.0:
                total += disc * self.unc_weight * std.sum(axis=-1)
            disc *= self.gamma
        if self.terminal_value is not None:
            total += disc * self.terminal_value(s)
        return total

    def sample_noise(self) -> np.ndarray:
        if self.noise == "spline":
            k = max(2, self.n_knots)
            knots = self.rng.normal(0.0, self.sigma, size=(self.n_samples, k))
            xk = np.linspace(0, self.horizon - 1, k)
            xs = np.arange(self.horizon)
            return np.stack([np.interp(xs, xk, row) for row in knots])
        return self.rng.normal(0.0, self.sigma, size=(self.n_samples, self.horizon))

    def act(self, model, s: np.ndarray) -> float:
        if self.plan is None:
            self.reset()
        if self._queue:
            return self._queue.pop(0)
        lo, hi = self.env.action_low[0], self.env.action_high[0]
        noise = self.sample_noise()
        U = np.clip(self.plan[None, :] + noise, lo, hi)
        cost = self.rollout_cost(model, s, U)
        w = np.exp(-(cost - cost.min()) / self.lam)
        w /= w.sum()
        self.plan = (w[:, None] * U).sum(axis=0)
        k = max(1, self.replan_every)
        actions = list(self.plan[:k])
        self.plan = np.concatenate([self.plan[k:], np.repeat(self.plan[-1:], k)])
        self._queue = actions[1:]
        return float(actions[0])


class HybridController:
    """MPPI for swing-up, LQR catch near the upright, back to MPPI if it falls."""

    def __init__(self, planner: MPPI, lqr, angle_fn, catch_tol: float = 0.3, release_tol: float = 0.6,
                 vel_fn=None, catch_vel_tol: float = float("inf")):
        """vel_fn(s) -> speed measure that must be below catch_vel_tol to engage the LQR (a pendulum arriving
        at the top too fast saturates a linear controller)."""
        self.planner, self.lqr, self.angle_fn = planner, lqr, angle_fn
        self.catch_tol, self.release_tol = catch_tol, release_tol
        self.vel_fn, self.catch_vel_tol = vel_fn, catch_vel_tol
        self.in_lqr = False
        self.n_lqr_steps = 0

    def reset(self):
        self.planner.reset()
        self.in_lqr = False
        self.n_lqr_steps = 0

    def act(self, model, s: np.ndarray) -> float:
        a = abs(self.angle_fn(s))
        if self.in_lqr and a > self.release_tol:
            self.in_lqr = False
            self.planner.reset()
        elif not self.in_lqr and a < self.catch_tol and (self.vel_fn is None or abs(self.vel_fn(s)) < self.catch_vel_tol):
            self.in_lqr = True
        if self.in_lqr:
            self.n_lqr_steps += 1
            return self.lqr(s)
        return self.planner.act(model, s)
