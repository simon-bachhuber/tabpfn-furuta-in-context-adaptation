"""History encodings that let a frozen in-context model identify the plant it is driving.

The question these answer: instead of re-fitting TabPFN when the plant changes, can each query row
carry enough of its own recent experience that one fixed table covers a *family* of plants? The
encoder must produce identical features offline (building the table) and online (in the control
loop), so both paths call :meth:`HistoryEncoder.encode`.

Encodings, cheapest to richest in structure:

``none``      no history — the Markovian baseline.
``lags``      raw recent rows: [s_{t-i}, u_{t-i}] for i = 1..k. Most general, highest dimension,
              and the model has to discover for itself that these imply a plant.
``resp``      action–response pairs: [u_{t-i}, Δs_{t-i}]. The map u → Δs *is* the plant, so this
              hands the model the relationship rather than the raw trace.
``ols``       the sufficient statistic: ridge-regression coefficients of the velocity deltas on
              [1, u, α̇, φ̇, sin α] over a window. This is a local linear model of the plant,
              recomputed every step in microseconds; TabPFN then supplies the nonlinear map
              conditioned on it. Classical recursive system identification as a feature vector.
``gain``      just the control gains cov(u, Δv) / var(u) — two numbers, robust, and enough to
              carry the actuator's sign and scale.
"""

from __future__ import annotations

import numpy as np


class HistoryEncoder:
    def __init__(self, env, kind: str = "ols", k: int = 10, window: int = 50, ridge: float = 1e-3):
        self.env, self.kind, self.k, self.window, self.ridge = env, kind, k, window, ridge
        self.n_state = env.n_state
        # statistics need enough rows for a stable regression, not the whole window: online, the
        # window fills gradually and the controller must cope with a partial one
        self.needed = 0 if kind == "none" else (k if kind in ("lags", "resp") else min(window, 16))

    # ---------------------------------------------------------------- sizing
    @property
    def n_features(self) -> int:
        if self.kind == "none":
            return 0
        if self.kind == "lags":
            return self.k * (self.n_state + 1)
        if self.kind == "resp":
            return self.k * (1 + 3)          # u plus the three velocity/angle deltas
        if self.kind == "ols":
            return 2 * 5                     # two targets x [1, u, alpha_dot, phi_dot, sin alpha]
        if self.kind == "gain":
            return 2
        raise ValueError(self.kind)

    # ---------------------------------------------------------------- encode
    def encode(self, past: list) -> np.ndarray:
        """``past``: list of (s, u, s2) in time order, most recent last. Short history is zero-padded,
        which is also what the model sees in the first moments after a reset."""
        if self.kind == "none":
            return np.zeros(0)
        out = np.zeros(self.n_features)
        if not past:
            return out
        if self.kind == "lags":
            for i in range(min(self.k, len(past))):
                s, u, _ = past[-1 - i]
                out[i * (self.n_state + 1):(i + 1) * (self.n_state + 1)] = np.concatenate([s, [u]])
            return out
        if self.kind == "resp":
            for i in range(min(self.k, len(past))):
                s, u, s2 = past[-1 - i]
                d = self.env.delta(s[None], s2[None])[0]
                out[i * 4:(i + 1) * 4] = np.concatenate([[u], d[:3]]) if len(d) >= 3 else np.concatenate([[u], d, np.zeros(3 - len(d))])
            return out
        w = past[-self.window:]
        S = np.array([p[0] for p in w]); U = np.array([p[1] for p in w]); S2 = np.array([p[2] for p in w])
        if len(w) < 8:
            return out
        dv = S2[:, [3, 1]] - S[:, [3, 1]] if self.n_state == 4 else (S2[:, [1]] - S[:, [1]])
        if self.kind == "gain":
            varu = U.var() + 1e-9
            out[:dv.shape[1]] = [float(np.cov(U, dv[:, j])[0, 1] / varu) for j in range(dv.shape[1])]
            return out
        # ols: ridge solve of dv on [1, u, alpha_dot, phi_dot, sin alpha]
        a = S[:, 2] if self.n_state == 4 else S[:, 0]
        A = np.stack([np.ones(len(w)), U, S[:, 3] if self.n_state == 4 else S[:, 1],
                      S[:, 1] if self.n_state == 4 else np.zeros(len(w)), np.sin(a)], axis=1)
        sc = A.std(axis=0) + 1e-9
        An = A / sc
        G = An.T @ An + self.ridge * len(w) * np.eye(A.shape[1])
        for j in range(min(2, dv.shape[1])):
            rhs = An.T @ dv[:, j]
            try:
                coef = np.linalg.solve(G, rhs) / sc
            except np.linalg.LinAlgError:
                # A column with no variance collapses onto the intercept once both are divided by
                # their (zero) spread, and the normal equations go singular. That is not a numerical
                # accident, it is the identifiability failure itself: hold the voltage constant and
                # the data cannot separate the input's effect from a constant offset. Least squares
                # still returns the minimum-norm answer, so the controller degrades instead of dying.
                coef = np.linalg.lstsq(G, rhs, rcond=None)[0] / sc
            out[j * 5:(j + 1) * 5] = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # ------------------------------------------------------- offline builder
    def encode_burst(self, S: np.ndarray, U: np.ndarray, S2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One contiguous burst from one plant -> (history features per row, usable mask)."""
        n = len(U)
        H = np.zeros((n, self.n_features))
        past: list = []
        usable = np.zeros(n, dtype=bool)
        for i in range(n):
            H[i] = self.encode(past)
            usable[i] = len(past) >= self.needed
            past.append((S[i], U[i], S2[i]))
            if len(past) > max(self.window, self.k) + 2:
                past.pop(0)
        return H, usable


def build_table(env, enc: HistoryEncoder, bursts: list) -> tuple[np.ndarray, np.ndarray]:
    """bursts: list of (S, U, S2) from possibly different plants -> (X with history, Y deltas)."""
    Xs, Ys = [], []
    for S, U, S2 in bursts:
        H, ok = enc.encode_burst(S, U, S2)
        if not ok.any():
            continue
        X = np.hstack([env.features(S, U), H]) if enc.n_features else env.features(S, U)
        Xs.append(X[ok]); Ys.append(env.delta(S, S2)[ok])
    return np.concatenate(Xs), np.concatenate(Ys)
