"""Discrete-time LQR from a continuous (A, B) or a discrete (Ad, Bd) linearization."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = sla.expm(M * dt)
    return E[:n, :n], E[:n, n:]


def dlqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Gain K such that u = -K x minimises sum x'Qx + u'Ru for x+ = Ad x + Bd u."""
    P = sla.solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(Bd.T @ P @ Bd + R, Bd.T @ P @ Ad)


class LQRController:
    """u = -K (x - x_ref), with the angle components wrapped."""

    def __init__(self, K: np.ndarray, x_ref: np.ndarray, angle_idx: tuple[int, ...], u_lim: float):
        self.K, self.x_ref, self.angle_idx, self.u_lim = np.asarray(K), np.asarray(x_ref), angle_idx, u_lim

    def __call__(self, x: np.ndarray) -> float:
        e = np.asarray(x, dtype=float) - self.x_ref
        for i in self.angle_idx:
            e[i] = ((e[i] + np.pi) % (2 * np.pi)) - np.pi
        return float(np.clip((-self.K @ e).item(), -self.u_lim, self.u_lim))
