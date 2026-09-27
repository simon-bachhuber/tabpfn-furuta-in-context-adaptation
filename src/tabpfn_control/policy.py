"""The frozen context-adaptive policy: one table, `[state, history] -> voltage`, one call per step.

Nothing is re-fit at run time. The table is built once from demonstrations on many plants; each
query row carries a summary of its own recent experience, and those columns are what tell the model
which pendulum it is driving. See `history.py` for the encodings.
"""

from __future__ import annotations

import os

import numpy as np

# The context table is deliberately larger than TabPFN's default CPU guard (5,000 rows): coverage of
# swing-up *and* balancing across a family of plants is the whole point. A GPU fits it in seconds and
# answers a query in ~24 ms; CPU works but is several times slower than the 50 ms control period.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

# Voltage grid the continuous teacher actions are quantised onto: dense near zero, because
# balancing needs resolution while swing-up needs authority.
LEVELS = np.array([-10, -6, -4, -2.5, -1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5, 2.5, 4, 6, 10])

# Column of the ridge statistic holding d(delta phi_dot)/du: correlates r=0.98 with the true motor
# gain, so its sign is the identified motor polarity.
POLARITY_COL = 6

# Constants for the energy feature below. They are deliberately NOT the plant's own values. The
# policy is supposed to work out what pendulum it is driving from its recent history; handing it the
# true length or mass in a feature column would answer the question it is being asked. One fixed set
# is chosen when the table is built and used unchanged forever after, so the feature means the same
# thing on every plant. These defaults describe the original bare-rod rig.
PUMP_REF = {"J2": 0.05 * 0.15 ** 2 + 0.05 * 0.30 ** 2 / 12, "mgl": 0.05 * 9.81 * 0.15}


def pump_ref_for(mp: float, Lp: float, m_tip: float, g: float = 9.81) -> dict:
    """The energy constants of one representative pendulum: a rod of mass `mp` and length `Lp` with
    `m_tip` bolted to its end. Evaluated once at the centre of a plant family, never per plant."""
    return {"J2": m_tip * Lp ** 2 + mp * Lp ** 2 / 3.0,
            "mgl": g * (m_tip * Lp + mp * Lp / 2.0)}


def pump_feats(S: np.ndarray, ref: dict | None = None) -> np.ndarray:
    """Two columns that spare the model an XOR.

    The teacher pumps with sign(alpha_dot * cos alpha) times the motor polarity — a product of three
    signs, which a smooth learner would have to reconstruct as a parity function from raw columns.
    Pendulum energy changes at a rate proportional to the first product, so handing over that
    product and the energy error turns the label into a monotone function of a feature.

    `ref` fixes the constants; see PUMP_REF. It is a reference pendulum, not this one, so these
    columns are a coordinate the model reads angles in, not a measurement of the plant.

    The energy column is divided through by the reference's own `mgl`, which costs nothing and
    matters: the energy needed to invert a pendulum varies about sixfold across this family, so an
    absolute figure computed from one reference is badly wrong at both ends and tells a long, heavy
    pendulum it is closer to upright than it is. The dimensionless form still carries a reference
    constant, in the ratio J2/mgl, but that ratio varies only about twofold.
    """
    ref = PUMP_REF if ref is None else ref
    a, ad = S[:, 2], S[:, 3]
    return np.stack([ad * np.cos(a),
                     0.5 * (ref["J2"] / ref["mgl"]) * ad ** 2 + np.cos(a) - 1.0], axis=1)


def policy_features(env, S: np.ndarray, H: np.ndarray | None = None, extra: bool = True,
                    ref: dict | None = None) -> np.ndarray:
    """State columns (no action: this is a policy), optional pumping columns, then history columns."""
    base = env.features(S, np.zeros(len(S)))[:, :-1]
    if extra:
        base = np.hstack([base, pump_feats(S, ref)])
    return np.hstack([base, H]) if (H is not None and H.shape[1]) else base


def build_table(env, enc, z, extra: bool = True, ref: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Demonstrations (one npz from generate_training_data.py) -> (X, y) for the policy table."""
    S, U, L, S2, b = z["S"], z["U"], z["L"], z["S2"], z["burst"]
    # rows the teacher was not actually demonstrating on (the excitation preamble) fill the history
    # window but never become training rows
    train = z["train"] if "train" in z.files else np.ones(len(S), bool)
    Xs, Ys = [], []
    for bid in np.unique(b):
        m = b == bid
        s, u, lab, s2 = S[m], U[m], L[m], S2[m]
        H, ok = enc.encode_burst(s, u, s2)
        keep = ok & train[m]
        X = policy_features(env, s, H, extra, ref)
        Xs.append(X[keep]); Ys.append(lab[keep])
    return np.concatenate(Xs), np.concatenate(Ys)


def stratify(X: np.ndarray, Y: np.ndarray, max_rows: int, rng,
             swing_share: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Decide how much of the context goes to swinging up rather than to holding.

    Demonstrations are mostly balancing, and a plain subsample teaches a policy that only holds. An
    even split fixes that, but even is not obviously right: holding is easy everywhere in this
    family, while swinging up a long pendulum with a heavy tip is where the rig runs out of motor.
    Context rows are the scarce resource, since every one of them is paid for in latency, so the
    share is a knob and `--swing-share` is how it was set.
    """
    swing = np.abs(X[:, 1]) > np.sin(0.3)          # column 1 is sin(alpha)
    n_sw = min(int(swing.sum()), int(max_rows * swing_share))
    n_up = min(int((~swing).sum()), max_rows - n_sw)
    idx = np.concatenate([rng.choice(np.where(swing)[0], n_sw, replace=False),
                          rng.choice(np.where(~swing)[0], n_up, replace=False)])
    return X[idx], Y[idx]


class FrozenPolicy:
    """A fitted TabPFN classifier over `LEVELS`. `fit` happens once; `act` is one forward pass."""

    def __init__(self, env, enc, X: np.ndarray, Y: np.ndarray, device: str = "auto", seed: int = 0,
                 extra: bool = True, version: str = "v3.5-fast", ref: dict | None = None):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        self.env, self.enc, self.extra = env, enc, extra
        self.ref = PUMP_REF if ref is None else ref
        ver = {"v3.5-fast": ModelVersion.V3_5_FAST, "v3.5": ModelVersion.V3_5}[version]
        y = np.abs(np.asarray(Y)[:, None] - LEVELS[None, :]).argmin(axis=1)
        self.model = TabPFNClassifier.create_default_for_version(
            ver, fit_mode="fit_with_cache", n_estimators=1, device=device, random_state=seed).fit(X, y)
        self.classes_ = np.asarray(self.model.classes_)
        self.X, self.Y = X, Y

    def act(self, s: np.ndarray, h: np.ndarray) -> float:
        x = policy_features(self.env, np.asarray(s)[None], np.asarray(h)[None], self.extra, self.ref)
        p = np.asarray(self.model.predict_proba(x))[0]
        return float(LEVELS[self.classes_[int(p.argmax())]])

    def save_table(self, path: str) -> None:
        np.savez(path, X=self.X, Y=self.Y, extra=self.extra, kind=self.enc.kind,
                 k=self.enc.k, window=self.enc.window,
                 ref_J2=self.ref["J2"], ref_mgl=self.ref["mgl"])

    @staticmethod
    def load_table(path: str):
        """(X, Y, extra, kind, k, window, ref). Tables written before the energy constants were
        pinned carry none, and get the bare-rod defaults they were in fact built with."""
        z = np.load(path)
        ref = ({"J2": float(z["ref_J2"]), "mgl": float(z["ref_mgl"])}
               if "ref_J2" in z.files else dict(PUMP_REF))
        return z["X"], z["Y"], bool(z["extra"]), str(z["kind"]), int(z["k"]), int(z["window"]), ref
