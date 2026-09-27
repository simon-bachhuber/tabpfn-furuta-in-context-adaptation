"""Generate the demonstrations the frozen policy is built from.

Each episode draws a pendulum from the family in `teacher.py` — a length, a mass bolted to its far
end, and a motor direction — and records a planner teacher solving it. The parameters are stored
only so the README's identification check can be reproduced; they never enter the model.

  python scripts/generate_training_data.py --plants 30 --episodes 4 --out data/demos.npz

Roughly 15 minutes for the default 30x4 across ten workers, and reproducible whatever
`--workers` is, because each plant is seeded from its own index.

`--mode bursts` instead records short bursts of *random* voltages from random states: no controller
runs, so it is fast, and it is what a dynamics model needs (state-action coverage rather than good
control). Those sets feed the history-encoding sweep in evaluate.py, not the policy.

  python scripts/generate_training_data.py --mode bursts --plants 30 --out data/family_train.npz
  python scripts/generate_training_data.py --mode bursts --plants 8 --seed 7 --out data/family_test.npz
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tabpfn_control.envs import wrap_angle
from tabpfn_control.teacher import build_plant, planner_episode, sample_plants


def random_bursts(args) -> None:
    """Excitation data for the dynamics sweep: random states, random voltages, no controller."""
    rng = np.random.default_rng(args.seed)
    S, U, S2, plant, burst, xis = [], [], [], [], [], []
    for p, xi in enumerate(sample_plants(rng, args.plants, allow_flip=not args.no_flip)):
        env = build_plant(xi)                            # a fresh pendulum, built to its own geometry
        xis.append([xi["Lp"], xi["m_tip"], xi["motor_sign"]])
        for b in range(args.bursts):
            near = rng.random() < args.near_upright_frac  # most bursts near the top, where control happens
            s = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-6, 6),
                          rng.normal(0, 0.25) if near else rng.uniform(-np.pi, np.pi),
                          rng.normal(0, 1.0) if near else rng.uniform(-8, 8)])
            env.set_state(s)
            for _ in range(args.burst_len):
                u = float(rng.uniform(-env.max_voltage, env.max_voltage))
                s2 = env.step(u)
                S.append(s.copy()); U.append(u); S2.append(s2.copy())
                plant.append(p); burst.append(p * args.bursts + b)
                s = s2
    out = {"S": np.array(S), "U": np.array(U), "S2": np.array(S2),
           "plant": np.array(plant), "burst": np.array(burst), "xi": np.array(xis)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"{len(S)} transitions | {args.plants} plants | {len(np.unique(out['burst']))} bursts | "
          f"{int((out['xi'][:, 2] < 0).sum())} with reversed motors -> {args.out}")


def _demos_for_plant(job):
    """One plant's demonstrations. Top level so a process pool can reach it.

    Plants are not equally hard, and a fixed budget per plant spends most of it on the easy ones. A
    long pendulum with a heavy tip takes the teacher several seconds and several pump cycles to
    invert, and that is exactly the behaviour the policy has least of. So after the first round,
    any plant the teacher found slow gets a second round. Difficulty is read off the teacher's own
    time to catch, which is available at collection time and owes nothing to how the policy later
    performs.
    """
    p, xi, seed, steps, stop_after, push_prob, episodes, hard_after = job
    env = build_plant(xi)
    rng = np.random.default_rng([seed, p])
    demos, catch = [], []

    def collect(n):
        for _ in range(n):
            out = planner_episode(env, rng, steps=steps, stop_after=stop_after, push_prob=push_prob)
            if out is None:
                continue
            s, u, lab, s2, tr = out
            up = np.abs(wrap_angle(s[tr][:, 2])) < 0.15
            if up.mean() < 0.05:                               # the teacher never got it up
                continue
            catch.append(int(np.argmax(up)))
            demos.append((s, u, lab, s2, tr))

    collect(episodes)
    if catch and float(np.median(catch)) > hard_after:
        collect(episodes)
    return p, demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="planner", choices=["planner", "bursts"])
    ap.add_argument("--plants", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=4, help="demonstrations attempted per plant")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--stop-after-catch", type=int, default=30,
                    help="end a demo this many steps after the catch, so swing-up is not swamped by balancing")
    ap.add_argument("--push-prob", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--bursts", type=int, default=12, help="[bursts mode] bursts per plant")
    ap.add_argument("--burst-len", type=int, default=130, help="[bursts mode] steps per burst")
    ap.add_argument("--near-upright-frac", type=float, default=0.6, help="[bursts mode]")
    ap.add_argument("--no-flip", action="store_true", help="exclude reversed-polarity plants")
    ap.add_argument("--hard-after", type=int, default=50,
                    help="a plant the teacher takes this many steps to catch gets a second round")
    ap.add_argument("--workers", type=int, default=10, help="plants demonstrated in parallel")
    ap.add_argument("--out", default="data/demos.npz")
    args = ap.parse_args()
    if args.mode == "bursts":
        return random_bursts(args)

    # Each plant is seeded from its own index rather than from a shared stream, so the result does
    # not depend on how many workers ran it. --workers 1 reproduces the same data serially.
    rng = np.random.default_rng(args.seed)
    plan = [(p, xi, args.seed, args.steps, args.stop_after_catch, args.push_prob, args.episodes,
             args.hard_after)
            for p, xi in enumerate(sample_plants(rng, args.plants, allow_flip=not args.no_flip))]
    xis = [[xi["Lp"], xi["m_tip"], xi["motor_sign"]] for _, xi, *_ in plan]

    S, U, L, S2, burst, plant, train, good = [], [], [], [], [], [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for p, demos in ex.map(_demos_for_plant, plan):
            for s, u, lab, s2, tr in demos:
                good += 1
                S.append(s); U.append(u); L.append(lab); S2.append(s2); train.append(tr)
                burst.append(np.full(len(u), len(burst))); plant.append(np.full(len(u), p))
            print(f"  plant {p + 1}/{args.plants}: {len(demos)} demos ({good} so far)", flush=True)
    out = {"S": np.concatenate(S), "U": np.concatenate(U), "L": np.concatenate(L), "S2": np.concatenate(S2),
           "burst": np.concatenate(burst), "plant": np.concatenate(plant),
           "train": np.concatenate(train), "xi": np.array(xis)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file := args.out, **out)
    sw = float((np.abs(wrap_angle(out["S"][out["train"]][:, 2])) > 0.3).mean())
    x = np.array(xis)
    print(f"{len(out['S'])} rows | {good} demos over {args.plants} plants | {sw:.0%} swing-up rows | "
          f"length {x[:, 0].min():.2f}-{x[:, 0].max():.2f} m, tip {x[:, 1].min():.3f}-{x[:, 1].max():.3f} kg, "
          f"{int((x[:, 2] < 0).sum())} reversed -> {out_file}")


if __name__ == "__main__":
    main()
