"""TabPFN-3.5 as a probabilistic one-step dynamics model.

One cached regressor per state-delta dimension. ``predict`` returns the mean and a
std estimate from the 16/84 % quantiles of the bar distribution. ``__call__``
implements the planner model interface: (states, actions) -> (next_states, std).
"""

from __future__ import annotations

import time

import numpy as np


class TabPFNDynamics:
    def __init__(self, env, version: str = "v3.5-fast", n_estimators: int = 1, device: str = "auto",
                 backend: str = "local", chunk: int = 10_000, seed: int = 0, stacked: bool = False):
        """stacked=True: ONE regressor over all targets, rows tagged with a task-id column and
        targets standardized per task (one TabPFN call per planning step instead of n_out)."""
        self.env, self.version, self.n_estimators, self.device = env, version, n_estimators, device
        self.backend, self.chunk, self.seed, self.stacked = backend, chunk, seed, stacked
        self.models: list = []
        self.n_calls = 0
        self.n_rows = 0
        self.fit_time = 0.0
        self.predict_time = 0.0
        self.X_ctx: np.ndarray | None = None
        self.Y_ctx: np.ndarray | None = None

    # ------------------------------------------------------------------ backend
    def _new_regressor(self):
        if self.backend == "client":
            from tabpfn_client import TabPFNRegressor

            return TabPFNRegressor.create_default_for_version(self.version, fit_mode="fit_with_cache")
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        ver = {"v3.5-fast": ModelVersion.V3_5_FAST, "v3.5": ModelVersion.V3_5}[self.version]
        return TabPFNRegressor.create_default_for_version(
            ver, fit_mode="fit_with_cache", n_estimators=self.n_estimators, device=self.device,
            random_state=self.seed,
        )

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "TabPFNDynamics":
        t0 = time.perf_counter()
        X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.X_ctx, self.Y_ctx = X, Y
        self.n_out = Y.shape[1]
        self.models = []
        if self.stacked:
            self.y_mu, self.y_sd = Y.mean(0), Y.std(0) + 1e-9
            Xs, ys = self._stack(X, (Y - self.y_mu) / self.y_sd)
            m = self._new_regressor()
            m.fit(Xs, ys)
            self.models.append(m)
        else:
            for j in range(Y.shape[1]):
                m = self._new_regressor()
                m.fit(X, Y[:, j])
                self.models.append(m)
        self.fit_time += time.perf_counter() - t0
        return self

    def _stack(self, X: np.ndarray, Y: np.ndarray | None = None):
        """Rows for every (row, task) pair; task id as an extra column."""
        n, k = len(X), self.n_out
        Xs = np.concatenate([np.hstack([X, np.full((n, 1), j, dtype=np.float64)]) for j in range(k)])
        ys = None if Y is None else np.concatenate([Y[:, j] for j in range(k)])
        return Xs, ys

    # ------------------------------------------------------------------ predict
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std), each (N, n_out)."""
        t0 = time.perf_counter()
        X = np.asarray(X, dtype=np.float64)
        if self.stacked:
            Xs, _ = self._stack(X)
            m = self.models[0]
            mu, sd = [], []
            for i in range(0, len(Xs), self.chunk):
                out = m.predict(Xs[i:i + self.chunk], output_type="main", quantiles=[0.16, 0.84])
                q = out["quantiles"]
                mu.append(np.asarray(out["mean"])); sd.append(0.5 * (np.asarray(q[1]) - np.asarray(q[0])))
                self.n_calls += 1; self.n_rows += min(self.chunk, len(Xs) - i)
            mu, sd = np.concatenate(mu).reshape(self.n_out, len(X)).T, np.concatenate(sd).reshape(self.n_out, len(X)).T
            self.predict_time += time.perf_counter() - t0
            return mu * self.y_sd + self.y_mu, sd * self.y_sd
        means, stds = [], []
        for m in self.models:
            mu, sd = [], []
            for i in range(0, len(X), self.chunk):
                out = m.predict(X[i:i + self.chunk], output_type="main", quantiles=[0.16, 0.84])
                q = out["quantiles"]
                mu.append(np.asarray(out["mean"]))
                sd.append(0.5 * (np.asarray(q[1]) - np.asarray(q[0])))
                self.n_calls += 1
                self.n_rows += min(self.chunk, len(X) - i)
            means.append(np.concatenate(mu))
            stds.append(np.concatenate(sd))
        self.predict_time += time.perf_counter() - t0
        return np.stack(means, -1), np.stack(stds, -1)

    def __call__(self, s: np.ndarray, u: np.ndarray):
        X = self.env.features(s, u)
        d, sd = self.predict(X)
        return self.env.apply_delta(s, d), sd

    def stats(self) -> dict:
        return {"n_calls": self.n_calls, "n_rows": self.n_rows, "fit_time_s": self.fit_time,
                "predict_time_s": self.predict_time, "context_rows": 0 if self.X_ctx is None else len(self.X_ctx)}
