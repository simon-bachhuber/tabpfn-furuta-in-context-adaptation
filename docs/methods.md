# Methods

How a frozen tabular model ends up acting as a feedback controller, and why each piece is shaped the
way it is. For the numbers see the [README](../README.md); for how to regenerate them see
[reproduction.md](reproduction.md).

## The plant

A Furuta pendulum: a driven horizontal arm with a free pendulum hanging off its end. The upright is
unstable and underactuated — the motor turns the arm, never the pendulum directly — so reaching it
means pumping energy in over several swings and then catching it.

State is `(φ, φ̇, α, α̇)`: arm angle, arm rate, pendulum angle with `α = 0` upright, pendulum rate.
The input is motor voltage, applied through a DC-motor model `τ = kt (V − km φ̇) / Rm`, so the motor
loses authority as the arm speeds up. The control period is 50 ms.

Two independent implementations exist and are checked against each other. `envs.py` holds a
hand-derived Lagrangian model; `mujoco_env.py` holds a MuJoCo model built from MJCF. A test asserts
they agree on accelerations to 5 % at matched parameters. Everything that runs is the MuJoCo one;
the analytic twin exists so that agreement can be asserted rather than assumed.

The table, floor, lighting and the clickable plates in the viewer are scenery. They carry no mass,
no contacts and no degrees of freedom, and a test checks the dynamics are bit-identical with and
without them.

## The plant family

Three things vary and nothing else:

| | range | what it changes |
|---|---|---|
| pendulum length | 0.20–0.40 m | inertia, and the energy a swing-up must inject |
| mass at the tip | 10–50 g | the same, on a light 20 g rod, so the tip dominates |
| motor direction | ±1 | the sign of every control decision |

Length and tip mass are compile-time quantities in MJCF, so each plant is a separately compiled
model rather than one rig with numbers scaled. `MujocoFuruta.set_geometry` exists for the case where
that is not possible — the viewer holds one model for a whole session — and it works by compiling
the geometry it wants and copying the resulting arrays across, so the live model holds exactly what
the compiler would have produced, inertia included. A test checks it against a freshly built model.

**The bounds were measured before anything was trained.** A planner with full knowledge of each
plant was run across a grid of candidates. Inside the published range it swings up and holds 16/16
at every corner and both polarities. Outside it the rig runs out of motor: at a 65 g tip the long
end drops to 8/12, at 80 g it fails at every length, and 110 g fails outright. That ceiling is the
plant and not the teacher — re-running the 80 g edge with horizon 45, 1200 samples and a tighter
catch window did not improve it. A family the teacher cannot demonstrate is not a fair thing to ask
a frozen policy to interpolate over.

One process note worth repeating: test at the horizon the real collection uses. The longest, heaviest
corner reads 14/16 at a 10 s budget and 16/16 at the 15 s budget collection actually uses.

## The policy

TabPFN has no weights to update, so the model *is* the table. One control step, every 50 ms:

```text
┌─▶ Furuta pendulum (MuJoCo). Its length, tip mass and motor direction are hidden.
│      │
│      ├── state now ───────────▶  6 columns  cos α, sin α, α̇, φ̇, α̇·cos α, Ẽ
│      │
│      └── last 20 steps (1 s) ─▶ 10 columns  ridge fit of Δα̇, Δφ̇ on [1, u, α̇, φ̇, sin α]
│                                      │
│                                      ▼  together, one 16-column query row
│   ┌────────────────────────┬────────────────────┬─────────┐
│   │ state + pumping (6)    │ history (10)       │ voltage │
│   ├────────────────────────┼────────────────────┼─────────┤
│   │ ...                    │ ...                │  +4 V   │  context: 8,000 rows
│   │ ...                    │ ...                │  −0.3 V │  of demonstrations on
│   │ ...                    │ ...                │  +10 V  │  60 pendulums, frozen
│   ├────────────────────────┼────────────────────┼─────────┤
│   │ now                    │ last second        │    ?    │  ◀── query row
│   └────────────────────────┴────────────────────┴─────────┘
│                                      │
│                                      ▼
│               TabPFNClassifier: one forward pass, nothing re-fit
│                                      │
│                                      ▼  a probability for each of 15 voltage classes
│     −10  −6  −4  −2.5  −1.5  −0.8  −0.3  0  +0.3  +0.8  +1.5  +2.5  +4  +6  +10 V
│      ·    ·   ·    ·     ·     ▂     █   ▄    ▁     ·     ·     ·    ·   ·   ·    e.g.
│                                      │
│                                      ▼  argmax, plus a small exploration dither
└──────────────────────────── u, held on the motor for the next 50 ms
```

**State columns.** `cos α, sin α, α̇, φ̇`. The arm angle φ is omitted because the plant is
rotationally symmetric about it.

**Two pumping columns.** The teacher pumps energy with `sign(α̇ · cos α) × motor direction`, a
product of three signs that a smooth learner would have to reconstruct as a parity function from raw
columns. Handing over the product `α̇·cos α` and the energy error turns the label into a monotone
function of a feature instead.

These two are computed from **one fixed reference pendulum, never the one running**. That
distinction is the entire experiment: reading the live length and mass into a feature would hand the
policy the answer it is supposed to infer from history. The constants are pinned when the table is
built and saved inside it, and a test asserts the features are identical on the shortest and longest
plants in the family.

The energy column is dimensionless, divided through by the reference's own `mgl`. The energy needed
to invert a pendulum varies about sixfold across this family, so an absolute figure computed from
one reference is badly wrong at both ends and tells a long, heavy pendulum it is closer to upright
than it is. The dimensionless form still carries a reference constant, in the ratio `J₂/mgl`, but
that ratio varies only about twofold.

**Ten history columns.** Ridge-regression coefficients of the recent velocity deltas on
`[1, u, α̇, φ̇, sin α]` over a 20-step window: a local linear model of whatever pendulum is currently
attached, recomputed every step in microseconds, with TabPFN supplying the nonlinear feedback law
conditioned on it. Classical recursive system identification, as a feature vector.

**Actions are a class, not a number.** A regressor head cannot clone a bang-bang teacher: averaging
±10 V labels yields 0 V, the one action that does nothing. The action set is quantised onto 15
voltage levels, dense near zero because balancing needs resolution while swing-up needs authority,
and predicted as a class.

## Why that history encoding

Thirteen encodings were compared, each fitted as a one-step dynamics model on 30 plants and scored
on 8 never seen, bracketed by a Markovian model that can only average over the family and an oracle
fitted on the test plant itself.

| encoding | columns | one-step error | gap to the oracle closed |
|---|---|---|---|
| ridge coefficients, 100-step window | 15 | 0.180 | 89 % |
| ridge coefficients, 50-step window | 15 | 0.197 | 85 % |
| **ridge coefficients, 20-step window** (used) | **15** | **0.208** | 82 % |
| **control gain only, cov(u, Δv)/var(u)** | **7** | **0.234** | 75 % |
| raw lags, 2 | 15 | 0.336 | 50 % |
| action–response pairs, 2 lags | 13 | 0.359 | 44 % |
| action–response pairs, 5 lags | 25 | 0.456 | 20 % |
| no history, Markovian | 5 | 0.537 | 0 % |
| raw lags, 5 | 30 | 0.582 | **−11 %**, worse than no history |
| action–response pairs, 20 lags | 85 | 0.583 | **−12 %** |
| raw lags, 10 | 55 | 0.631 | **−23 %** |
| raw lags, 20 | 105 | 0.637 | **−25 %** |
| action–response pairs, 10 lags | 45 | 0.665 | **−32 %** |
| oracle, one model per test plant | — | 0.136 | 100 % |

Seven columns of sufficient statistic beat 105 raw lag columns, and **every raw-lag encoding past
two lags is worse than no history at all**. Handing more of the trace over makes things
monotonically worse in both raw families. Give a tabular model a sufficient statistic and it works;
make it rediscover one from a trace and it does not. This is the single most useful thing learned
here about feeding a sequence to a tabular model.

The policy uses a 20-step window rather than the better 100-step one, because in closed loop the
window has to refill after every change to the plant, and one second of staleness after a motor
reversal beats five.

```bash
python scripts/evaluate.py --sweep --demos data/family_train.npz --test-demos data/family_test.npz
```

## The statistic really is system identification

Checkable without TabPFN at all: correlate what the encoder reads off a held-out excitation burst
against that plant's true hidden parameters.

| window | motor direction | length | tip mass |
|---|---|---|---|
| 20 steps, 1.0 s | 99 % | r = 0.60 | r = 0.37 |
| 50 steps, 2.5 s | 100 % | r = 0.72 | r = 0.37 |
| 100 steps, 5.0 s | 100 % | r = 0.84 | r = 0.47 |

The direction is recovered almost immediately, length follows, and the tip mass barely comes through
at all. That ordering explains the closed-loop result exactly: blanking the history destroys the
policy on reversed motors, where the information lost was certain, and leaves it mostly working on
forward ones, where what it lost was a weak estimate of how heavy the bob is. A statistic that
recovers the thing the controller most needs is worth more than one that recovers everything
equally badly.

```bash
python scripts/evaluate.py --identify --test-demos data/family_test.npz
```

**Without excitation the problem is singular, not merely hard.** Under a deterministic controller
`u` is a function of the state, so the voltage column collapses onto the intercept and the normal
equations genuinely have no unique solution. Both demos inject a small exploration dither and both
give you a way to switch it off and watch identification fail. The encoder falls back to least
squares when the solve fails, because a controller that dies on a singular matrix is worse than one
that degrades.

## The teacher

Demonstrations come from MPPI sampling-based planning over the true simulator, with that plant's own
LQR catching the pendulum near the upright, plus random pushes. The teacher is privileged at
*collection* time only: it reads the true dynamics and the true linearisation, neither of which
exists at run time, and the plants it teaches on are disjoint from the ones the policy is evaluated
on.

Being closed-loop is the point. An open-loop bang-bang pumping law with a per-plant LQR catch
produced 0–2 successes out of 6 however many rows it generated; the closed-loop planner produced a
working policy from *fewer* swing-up rows, because it also demonstrates recoveries from states off
the nominal path, which is the coverage a cloned policy needs.

Three collection details earned their place by being measured, after a first attempt that failed on
long pendulums:

1. **Latin hypercube, not independent draws.** Thirty uniform draws leave the long, heavy end of the
   box with two or three examples and the policy fails exactly there. Every length band and every
   mass band now holds the same number of plants.
2. **An excitation preamble on every demonstration**, excluded from the training rows but kept in
   the history. The ridge statistic needs sixteen transitions before it means anything, so without a
   preamble every demonstration silently discarded its first sixteen rows — the moment the pendulum
   starts moving from rest, which is the most valuable data it had.
3. **Difficulty-weighted collection.** A plant the teacher is slow to catch gets a second round, so
   the budget follows the difficulty. Difficulty is read off the teacher's own time to catch, which
   is available at collection time and owes nothing to how the policy later performs.

## Latency

Latency binds, not accuracy. Sampling-based planning over a TabPFN dynamics model needs 20–40
*dependent* forward passes per step and cannot close a 50 ms loop; a single-call feedback law can.
That constraint is what pushed this project from model-based planning to a policy table.

One call over the 8,000-row context takes a median of 50 ms on an idle twelve-core CPU, right at the
control period, and about 25 ms on the integrated GPU. A 12,000-row table was tried and rejected: no
better on the eight validation pendulums, and 63 ms puts it outside real time.

In the interactive viewer three clocks run independently. Physics advances in real time on the last
action it was handed, which is a zero-order hold and what a real rig does between controller updates.
TabPFN runs on its own thread as fast as it manages, so a slow forward pass costs control rate rather
than stalling the window. Rendering keeps its own cadence, 30 fps by default.

| TabPFN cost per call | control rate | physics | rendering |
|---|---|---|---|
| 200 ms, forced | 5.0 Hz | 20 Hz, real time | 29.9 fps |
| 5 ms, forced | 20.1 Hz | 20 Hz, real time | 30.0 fps |
| the real policy, on CPU | 19.7 Hz | 20 Hz, real time | 29.9 fps |

The control rate cannot exceed the physics rate, because there is no new state to act on until the
next step lands.

## How the measurements are scored

A plant counts as solved only if, on **every** seed, it swings up and then holds to the end of the
episode. Two details behind that:

- **Swinging up and staying up are separate numbers.** Scoring the second half of an episode
  conflates them, because a slow swing-up then reads as a balance failure.
- **Several seeds per plant, always.** One episode per plant moves enough on the dither alone that
  two configurations trade places on noise. Every headline number is four seeds per plant; the
  feasibility sweep is eight.

## What is deliberately not claimed

The teacher is privileged, so this is behaviour cloning across a plant family rather than
reinforcement learning from scratch. The policy is frozen at run time, but it was assembled from
demonstrations a planner produced with full knowledge of each plant.

The family is bounded and the bounds are stated. Outside them the rig itself cannot do the task.

Everything is simulation. The MuJoCo model agrees with an independent analytic model to 5 %, which
is a statement about two models agreeing, not about either matching hardware.
