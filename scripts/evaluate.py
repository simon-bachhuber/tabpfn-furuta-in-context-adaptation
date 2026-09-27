"""Does the history actually identify the plant? Three experiments, all on plants never trained on.

closed-loop (default)
    Run the frozen policy on unseen pendulums. The only difference between conditions is whether
    the history columns carry live values or zeros — same weights, same table, same everything else.

    Each episode opens with one second of random voltages, because a plant is not identifiable
    without excitation; --cold-start removes it and starts the window empty instead.

      python scripts/evaluate.py --table data/policy_table.npz --episodes 6

sweep (--sweep)
    How much history, and shaped how? Fits one dynamics model per encoding on a family of plants and
    scores one-step error on held-out plants, bracketed by a Markovian model and a per-plant oracle.

      python scripts/evaluate.py --sweep --demos data/family_train.npz --test-demos data/family_test.npz

feasibility (--feasibility)
    Can the teacher solve every plant the policy is asked to cover? Runs the privileged planner at
    the corners of the family and reports swing-up and hold. This is what fixes the ranges in
    teacher.py; a family the teacher cannot demonstrate is not a fair thing to ask a policy for.

      python scripts/evaluate.py --feasibility

identify (--identify)
    Is the history statistic really doing system identification? Correlates what the encoder reads
    off each held-out burst against that plant's true hidden parameters. Pure NumPy, no TabPFN.

      python scripts/evaluate.py --identify --test-demos data/family_test.npz
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from tabpfn_control.envs import make_env, wrap_angle
from tabpfn_control.history import HistoryEncoder, build_table as build_dyn_table
from tabpfn_control.policy import POLARITY_COL, FrozenPolicy
from tabpfn_control.teacher import TEST_PLANTS, build_plant, sample_plant


def load_policy(env, path, device):
    X, Y, extra, kind, k, window, ref = FrozenPolicy.load_table(path)
    enc = HistoryEncoder(env, kind, k=k, window=window)
    return FrozenPolicy(env, enc, X, Y, device=device, extra=extra, ref=ref), enc


def identify_burst(env, rng, n=20, amp=6.0):
    """A one-second excitation burst, so the window is informative before balancing must start."""
    past, s = [], env.reset(rng, hanging=False)
    for _ in range(n):
        u = float(rng.uniform(-amp, amp))
        s2 = env.step(u)
        past.append((s.copy(), u, s2.copy()))
        s = s2
        if abs(wrap_angle(s[2])) > 1.2:
            s = env.reset(rng, hanging=False)
    return past


def episode(env, pol, enc, steps, rng, condition="rolling", dither=0.1, flip_at=None, warmup=True):
    past: deque = deque(maxlen=max(enc.window, enc.k) + 2)
    if warmup:
        past.extend(identify_burst(env, rng))
    s = env.reset(rng, hanging=True)
    al, ms, gains, flipped = [], [], [], None
    for k in range(steps):
        if flip_at is not None and k == flip_at:
            env.set_plant(motor_scale=-1.0); flipped = k
        t0 = time.perf_counter()
        h = np.zeros(enc.n_features) if condition == "zero" else enc.encode(list(past))
        u = pol.act(s, h)
        ms.append((time.perf_counter() - t0) * 1000)
        gains.append(float(h[POLARITY_COL]) if enc.n_features > POLARITY_COL else 0.0)
        u = float(np.clip(u + rng.normal(0, dither), -env.max_voltage, env.max_voltage))
        sp = s.copy(); s = env.step(u); past.append((sp, u, s.copy()))
        al.append(abs(float(wrap_angle(s[2]))))
    al = np.array(al)
    swing = next((i for i in range(len(al) - 20) if (al[i:i + 20] < 0.15).all()), None)
    # Two different questions, kept apart. Whether it ever gets the pendulum up is about the
    # swing-up; whether it keeps it there afterwards is about the balance. Scoring only the second
    # half of the episode confuses the two, because a slow swing-up then reads as a balance failure.
    after = al[swing:] if swing is not None else al[:0]
    out = {"swingup_s": None if swing is None else swing * env.dt,
           "held_after_catch": float((after < 0.15).mean()) if len(after) else 0.0,
           "rms_after_catch": float(np.sqrt(np.mean(after ** 2))) if len(after) else float("nan"),
           "upright_frac_last_half": float((al[len(al) // 2:] < 0.15).mean()),
           "rms_last_half": float(np.sqrt(np.mean(al[len(al) // 2:] ** 2))),
           "ms_median": float(np.median(ms)), "alpha": al.tolist(), "gain": gains}
    if flipped is not None:
        after = al[flipped:]
        back = next((i for i in range(len(after) - 20) if (after[i:i + 20] < 0.15).all()), None)
        out["recovery_s"] = None if back is None else back * env.dt
    return out


def closed_loop(args) -> None:
    env = make_env("furuta-mujoco")
    pol, enc = load_policy(env, args.table, args.device)
    print(f"frozen table: {pol.X.shape[1]} cols ({enc.n_features} history), {len(pol.X)} rows")
    steps = int(args.seconds / env.dt)
    flip_at = int(args.flip_at / env.dt) if args.flip_at else None
    results = {"config": vars(args), "runs": {}}
    for cond in args.conditions.split(","):
        runs = []
        for ep in range(args.episodes):
            r = np.random.default_rng(900 + ep)
            xi = sample_plant(r)
            xi["motor_sign"] = 1.0 if ep % 2 else -1.0        # half the test plants wired backwards
            e = build_plant(xi)
            out = episode(e, pol, enc, steps, np.random.default_rng(ep), condition=cond,
                          flip_at=flip_at, warmup=not args.cold_start)
            out["xi"] = xi
            runs.append(out)
            su = "none" if out["swingup_s"] is None else f"{out['swingup_s']:.1f}s"
            motor = "reversed" if xi["motor_sign"] < 0 else "normal  "
            print(f"[{cond:7s}] ep{ep} ({xi['Lp']:.2f} m, {xi['m_tip'] * 1e3:.0f} g tip, motor {motor}): "
                  f"swing-up {su:>5} | upright(2nd half) {out['upright_frac_last_half']:.2f} | "
                  f"rms {out['rms_last_half']:.3f} | {out['ms_median']:.0f} ms")
        ok = sum(r["swingup_s"] is not None for r in runs)
        print(f"[{cond:7s}] SUMMARY swung up {ok}/{len(runs)} | upright(2nd half) "
              f"{np.mean([r['upright_frac_last_half'] for r in runs]):.2f}")
        results["runs"][cond] = runs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("saved", args.out)


def test_plants(args) -> None:
    """The eight fixed pendulums, several seeds each, with the rolling history and with it blanked.

    One episode per plant is not a measurement: the dither and the starting state move the result
    enough that two configurations can trade places on noise alone. Every plant is run `--seeds`
    times and a plant counts as solved only if it swings up and holds on every one of them.
    """
    env = build_plant(TEST_PLANTS[0])
    pol, enc = load_policy(env, args.table, args.device)
    steps = int(args.seconds / env.dt)
    results = {"config": {k: v for k, v in vars(args).items() if k != "out"}, "plants": []}
    solved = 0
    for i, xi in enumerate(TEST_PLANTS):
        row = {"i": i, **xi}
        for cond in args.conditions.split(","):
            runs = []
            for sd in range(args.seeds):
                e = build_plant(xi)
                out = episode(e, pol, enc, steps, np.random.default_rng(100 + 37 * i + sd),
                              condition=cond, warmup=not args.cold_start)
                runs.append({k: out[k] for k in ("swingup_s", "held_after_catch", "rms_after_catch",
                                                 "upright_frac_last_half", "ms_median")})
            row[cond] = runs
        good = lambda x: x["swingup_s"] is not None and x["held_after_catch"] > 0.95  # noqa: E731
        r = row["rolling"]
        held = [x for x in r if good(x)]
        solved += len(held) == len(r)
        su = [x["swingup_s"] for x in r if x["swingup_s"] is not None]
        z = row.get("zero")
        print(f"plant {i + 1} ({xi['Lp']:.2f} m, {xi['m_tip'] * 1e3:4.0f} g, motor "
              f"{'reversed' if xi['motor_sign'] < 0 else 'forward '}): "
              f"{len(held)}/{len(r)} | up in "
              f"{(f'{min(su):.1f}-{max(su):.1f}s' if su else 'never'):>11} | "
              f"rms {(max(x['rms_after_catch'] for x in held) if held else float('nan')):.3f}"
              + (f"   [history zeroed: {sum(good(x) for x in z)}/{len(z)}]" if z else ""))
        results["plants"].append(row)
    print(f"SUMMARY: {solved}/{len(TEST_PLANTS)} plants solved on every seed")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("saved", args.out)


def _corner(job):
    from tabpfn_control.teacher import build_plant, planner_episode
    Lp, mt, sign, seed, steps = job
    env = build_plant({"Lp": Lp, "m_tip": mt, "motor_sign": sign})
    out = planner_episode(env, np.random.default_rng(seed), steps=steps, stop_after=None, push_prob=0.0)
    if out is None:
        return {"Lp": Lp, "m_tip": mt, "sign": sign, "ok": False, "swing_s": None, "held": 0.0}
    a = np.abs(wrap_angle(out[3][out[4]][:, 2]))          # the demonstrated part, not the preamble
    caught = next((i for i in range(len(a) - 20) if (a[i:i + 20] < 0.15).all()), None)
    held = float((a[-60:] < 0.15).mean())
    return {"Lp": Lp, "m_tip": mt, "sign": sign, "ok": bool(caught is not None and held > 0.9),
            "swing_s": None if caught is None else caught * env.dt, "held": held}


def feasibility(args) -> None:
    """The planner, which knows everything about the plant, at the corners of the family."""
    import itertools
    from concurrent.futures import ProcessPoolExecutor

    from tabpfn_control.teacher import LP_RANGE, MTIP_RANGE

    lps = [float(x) for x in args.lengths.split(",")] if args.lengths else list(LP_RANGE)
    mts = [float(x) for x in args.tips.split(",")] if args.tips else list(MTIP_RANGE)
    jobs = [(Lp, mt, sg, sd, args.steps) for Lp, mt in itertools.product(lps, mts)
            for sg in (1.0, -1.0) for sd in range(args.seeds)]
    print(f"{len(jobs)} planner runs over {len(lps)}x{len(mts)} plants, both polarities, "
          f"{args.seeds} seeds, {args.steps * 0.05:.0f} s each")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_corner, jobs))
    by: dict = {}
    for r in res:
        by.setdefault((r["Lp"], r["m_tip"]), []).append(r)
    print(f"{'length':>7} {'tip':>7}  {'solved':>7}  swing-up      held")
    for (Lp, mt), rs in sorted(by.items()):
        sw = [r["swing_s"] for r in rs if r["swing_s"] is not None]
        print(f"{Lp:6.2f}m {mt * 1e3:5.0f}g  {sum(r['ok'] for r in rs):3d}/{len(rs):<3d}  "
              f"{(f'{min(sw):.1f}-{max(sw):.1f}s' if sw else 'never'):>12}  "
              f"{np.mean([r['held'] for r in rs]):.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("saved", args.out)


def identify(args) -> None:
    """What the ten history columns recover about a plant they have never been told about."""
    env = make_env("furuta-mujoco")
    z = np.load(args.test_demos)
    S, U, S2, b, plant, xi = z["S"], z["U"], z["S2"], z["burst"], z["plant"], z["xi"]
    names = ["length", "tip mass", "motor direction"]         # the columns of xi, in order
    rows = {}
    for w in [int(x) for x in args.windows.split(",")]:
        enc = HistoryEncoder(env, "ols", k=w, window=w)
        stats, truth = [], []
        for i in np.unique(b):
            m = b == i
            if m.sum() <= w:
                continue
            s, u, s2 = S[m][:w], U[m][:w], S2[m][:w]
            stats.append(enc.encode([(s[j], u[j], s2[j]) for j in range(w)]))
            truth.append(xi[plant[m][0]])
        A, T = np.array(stats), np.array(truth)
        gain = A[:, POLARITY_COL]
        n_par = T.shape[1]
        # a parameter is "recovered" if some single statistic column correlates with it; the motor
        # direction is a sign, so it is scored by agreement rather than by correlation
        best = [max(abs(np.corrcoef(A[:, c], T[:, j])[0, 1]) for c in range(A.shape[1]))
                for j in range(n_par)]
        pol = float((np.sign(gain) == np.sign(T[:, 2])).mean())
        rows[w] = {"bursts": len(A), "polarity_acc": pol, "r_best_col": best}
        print(f"window {w:3d} steps ({w * env.dt:.1f} s), {len(A)} bursts | motor direction recovered on "
              f"{pol:.0%} | " + " ".join(f"{n} r={bb:.2f}" for n, bb in zip(names, best)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print("saved", args.out)


def sweep(args) -> None:
    from tabpfn_control.model import TabPFNDynamics

    env = make_env("furuta-mujoco")
    ztr, zte = np.load(args.demos), np.load(args.test_demos)

    def bursts(z):
        S, U, S2, b = z["S"], z["U"], z["S2"], z["burst"]
        return [(S[b == i], U[b == i], S2[b == i]) for i in np.unique(b)]

    tr, te = bursts(ztr), bursts(zte)
    rng = np.random.default_rng(0)
    # a full sweep is ~45 min of forward passes, so --resume lets an interrupted one continue
    results = json.loads(Path(args.out).read_text()) if (args.resume and Path(args.out).exists()) else {}
    for name in args.encodings.split(","):
        if name in results:
            continue
        kind = "".join(c for c in name if c.isalpha())
        n = int("".join(c for c in name if c.isdigit()) or 0)
        enc = HistoryEncoder(env, kind, k=n, window=n)
        Xtr, Ytr = build_dyn_table(env, enc, tr)
        Xte, Yte = build_dyn_table(env, enc, te)
        if len(Xtr) > args.max_rows:
            i = rng.choice(len(Xtr), args.max_rows, replace=False); Xtr, Ytr = Xtr[i], Ytr[i]
        m = TabPFNDynamics(env, device=args.device, stacked=True, backend=args.backend).fit(Xtr, Ytr)
        mu, _ = m.predict(Xte[:2000])
        rel = np.sqrt(np.mean((mu - Yte[:2000]) ** 2, axis=0)) / Yte[:2000].std(axis=0)
        results[name] = {"cols": int(Xtr.shape[1]), "rel": rel.tolist(), "mean_rel": float(rel.mean())}
        print(f"[{name:8s}] {Xtr.shape[1]:3d} cols | one-step RMSE/std {np.round(rel, 3)} | mean {rel.mean():.3f}")
        Path(args.out).write_text(json.dumps(results, indent=1))
    if "oracle_per_plant" not in results:                 # the bound: one model per test plant
        rels = []
        for p in np.unique(zte["plant"]):
            m_ = zte["plant"] == p
            X, Y = env.features(zte["S"][m_], zte["U"][m_]), env.delta(zte["S"][m_], zte["S2"][m_])
            cut = len(X) // 2
            mod = TabPFNDynamics(env, device=args.device, stacked=True, backend=args.backend).fit(X[:cut], Y[:cut])
            mu, _ = mod.predict(X[cut:cut + 800])
            rels.append(np.sqrt(np.mean((mu - Y[cut:cut + 800]) ** 2, 0)) / Y[cut:cut + 800].std(0))
        R = np.array(rels).mean(0)
        results["oracle_per_plant"] = {"rel": R.tolist(), "mean_rel": float(R.mean())}
        print(f"[oracle  ] one model per test plant | mean {R.mean():.3f}")
        Path(args.out).write_text(json.dumps(results, indent=1))
    print("saved", args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/policy_table.npz")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--flip-at", type=float, default=0.0, help="reverse the motor at this time (0 = never)")
    ap.add_argument("--conditions", default="rolling,zero")
    ap.add_argument("--cold-start", action="store_true",
                    help="skip the one-second excitation burst and start the window empty")
    ap.add_argument("--sweep", action="store_true", help="run the history-encoding comparison instead")
    ap.add_argument("--identify", action="store_true", help="correlate the history statistic with the true plant")
    ap.add_argument("--feasibility", action="store_true", help="can the teacher solve the family's corners?")
    ap.add_argument("--test-plants", action="store_true", help="the eight fixed pendulums in teacher.py")
    ap.add_argument("--lengths", default=None, help="[feasibility] comma-separated, default the family's ends")
    ap.add_argument("--tips", default=None, help="[feasibility] comma-separated, default the family's ends")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per plant (feasibility and test-plants)")
    ap.add_argument("--steps", type=int, default=300, help="[feasibility] control steps per run")
    ap.add_argument("--workers", type=int, default=10, help="[feasibility] parallel planner processes")
    ap.add_argument("--windows", default="20,50,100", help="[identify] window lengths to score")
    ap.add_argument("--demos", default="data/family_train.npz")
    ap.add_argument("--test-demos", default="data/family_test.npz")
    ap.add_argument("--encodings", default="none,lags2,lags5,lags10,lags20,resp2,resp5,resp10,resp20,ols20,ols50,ols100,gain50")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--resume", action="store_true", help="[sweep] keep encodings already in the output file")
    ap.add_argument("--backend", default="local", choices=["local", "client"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or ("results/history_sweep.json" if args.sweep else
                            "results/identification.json" if args.identify else
                            "results/feasibility.json" if args.feasibility else
                            "results/test_plants.json" if args.test_plants else "results/evaluation.json")
    (sweep if args.sweep else identify if args.identify else feasibility if args.feasibility else
     test_plants if args.test_plants else closed_loop)(args)


if __name__ == "__main__":
    main()
