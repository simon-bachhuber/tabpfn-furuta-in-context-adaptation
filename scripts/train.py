"""Build the frozen policy table and save it.

TabPFN has no weights to update, so "training" here is assembling the context: encode each
demonstration row as [state, pumping features, history statistic] -> teacher voltage, balance
swing-up against balancing rows, and fit once. The saved npz is the whole model.

  python scripts/train.py --demos data/demos.npz --out data/policy_table.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np

from tabpfn_control.envs import make_env
from tabpfn_control.history import HistoryEncoder
from tabpfn_control.policy import FrozenPolicy, build_table, pump_ref_for, stratify
from tabpfn_control.teacher import LP_RANGE, MTIP_RANGE, ROD_MASS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/demos.npz")
    ap.add_argument("--encoding", default="ols", choices=["ols", "gain", "resp", "lags", "none"])
    ap.add_argument("--window", type=int, default=20, help="history window (steps) for ols/gain, lags for resp/lags")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--swing-share", type=float, default=0.5,
                    help="fraction of the context spent on swing-up rather than balancing rows")
    ap.add_argument("--no-pump-feats", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/policy_table.npz")
    args = ap.parse_args()

    env = make_env("furuta-mujoco")
    enc = HistoryEncoder(env, args.encoding, k=args.window, window=args.window)
    # One reference pendulum, the middle of the family, fixes the energy feature's constants for
    # good. Using each plant's own numbers here would hand the policy the length and mass it is
    # meant to infer, so the constants are pinned once and saved with the table.
    ref = pump_ref_for(ROD_MASS, float(np.mean(LP_RANGE)), float(np.mean(MTIP_RANGE)))
    print(f"energy feature pinned to a {np.mean(LP_RANGE):.2f} m rod with "
          f"{np.mean(MTIP_RANGE) * 1e3:.0f} g at the tip: J2={ref['J2']:.5f}, mgl={ref['mgl']:.4f}")
    X, Y = build_table(env, enc, np.load(args.demos), extra=not args.no_pump_feats, ref=ref)
    rng = np.random.default_rng(args.seed)
    X, Y = stratify(X, Y, args.max_rows, rng, swing_share=args.swing_share)
    sw = int((np.abs(X[:, 1]) > np.sin(0.3)).sum())
    print(f"table: {len(X)} rows x {X.shape[1]} cols ({enc.n_features} history, '{args.encoding}'), "
          f"{sw} swinging up and {len(X) - sw} balancing")
    t0 = time.perf_counter()
    pol = FrozenPolicy(env, enc, X, Y, device=args.device, extra=not args.no_pump_feats,
                       seed=args.seed, ref=ref)
    print(f"fit (one forward pass over the context): {time.perf_counter() - t0:.0f}s")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pol.save_table(args.out)
    print(f"saved {args.out} — this file is the model; nothing is re-fit after it")


if __name__ == "__main__":
    main()
