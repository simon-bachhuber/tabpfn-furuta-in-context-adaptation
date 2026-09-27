# Reproduction

Everything expensive is committed, so a fresh clone can reproduce every figure and every number in
the README without re-running the slow parts. This page covers both routes: checking the committed
results, and regenerating them from nothing.

## Install

```bash
git clone <this repository> && cd tabpfn-furuta-in-context-adaptation
uv venv --python 3.14
uv pip install -r requirements.lock      # exact versions the committed numbers came from
uv pip install -e . --no-deps
```

`requirements.lock` is what to use when the numbers have to match. `pyproject.toml` carries loose
bounds so the package stays installable as its dependencies move; installing from it instead is
fine for playing with the demo and not fine for comparing against the committed results.

Optional extras, if installing from `pyproject.toml` rather than the lock file:

```bash
uv pip install -e ".[web,video,dev]"     # browser demo, mp4 recording, pytest
```

## The one thing that is not committed

TabPFN's weights. They are about 1.2 GB, they are licensed, and they are not ours to redistribute,
so they are fetched on first use into `~/.cache/tabpfn` and cached there afterwards.

You need a Prior Labs token, and the licence has to be accepted once in a browser before the
download will succeed:

```bash
export TABPFN_TOKEN=...       # https://platform.priorlabs.ai
```

The first run that touches TabPFN pays for the download. Every run after that is offline. Nothing
else in this repository needs a network.

If the download is refused, it is almost always the licence rather than the token; the error names
the URL to open.

## Check the committed results

None of this regenerates anything. It reads what is in `data/` and `results/` and confirms the code
still produces it.

```bash
pytest -q                                    # 14 tests, ~2 s, no TabPFN needed
python scripts/make_figure.py                # rewrites media/training_data.png from data/demos.npz
python scripts/evaluate.py --identify        # ~5 s, no TabPFN needed
```

The headline table, which does need TabPFN and the committed `data/policy_table.npz`:

```bash
python scripts/evaluate.py --test-plants --seeds 4     # ~25 min on CPU
```

Expect 8/8 plants and 32/32 episodes with the rolling history, 10/32 with it blanked, and 0/16 on
the reversed motors with it blanked.

Run it as written. Shortening the episode with `--seconds` makes pendulum 4 look like a failure: it
is the longest and heaviest of the eight and has taken as long as 18.8 s to get up, so a 10 s
episode cuts it off mid-swing. Cutting `--seeds` down to 1 is worse than useless — a single episode
per plant moves enough on the dither alone that configurations trade places on noise, which is why
the default is 8 and the headline number is 4.

## Regenerate from nothing

In order, with the cost of each step on an idle twelve-core CPU:

```bash
python scripts/evaluate.py --feasibility                                  # ~12 min
python scripts/generate_training_data.py --plants 60 --episodes 4         # ~25 min  -> data/demos.npz
python scripts/train.py                                                   # ~10 s    -> data/policy_table.npz
python scripts/evaluate.py --test-plants --seeds 4                        # ~25 min  -> results/test_plants.json
python scripts/evaluate.py --episodes 12                                  # ~20 min  -> results/evaluation.json
python scripts/make_figure.py                                             # ~2 s     -> media/training_data.png
python scripts/demo.py --record media/demo.mp4 --seconds 40               # ~13 min  -> media/demo.{mp4,gif}
```

The encoding sweep needs excitation data rather than demonstrations, from the same script:

```bash
python scripts/generate_training_data.py --mode bursts --plants 30 --seed 0 --out data/family_train.npz
python scripts/generate_training_data.py --mode bursts --plants 8  --seed 7 --out data/family_test.npz
python scripts/evaluate.py --sweep                                        # ~50 min  -> results/history_sweep.json
python scripts/evaluate.py --identify                                     # ~5 s     -> results/identification.json
```

Recording is slower than it looks because it renders 2,400 frames: the clip is 60 fps, sampled
between control decisions rather than at them. `--capture-per-step`, `--dpi`, `--gif-fps`,
`--gif-scale` and `--gif-colors` trade quality against time and file size.

About three hours end to end. Data generation and the feasibility sweep parallelise over
plants with `--workers`; the TabPFN evaluations are sequential because each control step depends on
the last.

### Determinism

Data generation seeds each plant from its own index rather than from a shared stream, so
`--workers` does not change the result: one worker and eleven produce the same file.

The TabPFN evaluations are seeded per episode and are reproducible on the same package versions and
the same device. They are *not* bit-identical across CPU and GPU, or across TabPFN versions, which
is why `requirements.lock` exists. Conclusions do not move; the third decimal place does.

## Run the demo

The demo needs only `data/policy_table.npz`, which is committed.

```bash
python scripts/demo.py --viewer       # native MuJoCo window, clickable plates
mjpython scripts/demo.py --viewer     # on macOS, where the window must own the main thread
python scripts/demo.py --web          # browser instead; needs no local display or GL
```

`--web` is the fallback for a machine with no display or no OpenGL. It renders the same MuJoCo scene
client-side through [mjviser](https://github.com/mujocolab/mjviser).

## The YouTube video

A self-contained walkthrough of about ninety seconds, 1920x1080 at 60 fps, with the explanations
drawn on screen and live telemetry of both joints and TabPFN's output:

```bash
python scripts/make_video.py                  # -> out/tabpfn_furuta_youtube.mp4, ~15 min on CPU
python scripts/make_video.py --stills 8.5,40  # a few frames as PNGs, to check a change quickly
```

Every frame of the pendulum is a real run of the frozen policy, simulated at the start of the script
and recorded at the 500 Hz physics rate; nothing is keyframed. The takes are fixed by seed in the
script, and the mid-run pendulum swap is a genuine swap: in about half of the seeds tried the
swapped pendulum does not even fall, and the take used is one where it does, because the recovery is
the thing worth showing. The rendered file is not committed, since it is about 100 MB and
regenerates from the script. It renders in software and needs no GPU; the Inter typeface it uses is
bundled under `scripts/assets/fonts` with its licence.

## Troubleshooting

**`mujoco.FatalError: an OpenGL platform library has not been loaded`** — a headless Linux box with
no GL stack. `--record` selects software rendering automatically when there is no display; for
anything else install `libgl1 libglx-mesa0 libgl1-mesa-dri`, or use `--web`, which needs no GL at
all. Recording falls back to a schematic plot if no context can be made, so the command always
produces a file.

**macOS: the viewer refuses to start** — it must own the main thread, so run it under `mjpython`,
which the mujoco wheel installs next to `python`. The script checks for this up front and says so
rather than failing after the table has been fitted.

**macOS: ctrl-drag does nothing** — double-click the pendulum first. Selection is a separate step in
MuJoCo. Also, macOS delivers ctrl with left-click as a right-click, so that gesture arrives as the
twist rather than the pull; use a two-finger drag, a real mouse, or the push plates.

**`RuntimeError: Running on CPU with more than 5000 samples`** — the context table is deliberately
larger than TabPFN's default CPU guard. The package sets `TABPFN_ALLOW_CPU_LARGE_DATASET=1` on
import; you only see this if you are calling TabPFN directly.

**The GPU faults with `unspecified launch failure`** — see the note in the README. On the AMD
integrated GPU used here this happened twice, unprompted, partway through long runs of inference,
recovering on its own each time. Every committed number was produced on CPU. Pass `--device cpu` to
avoid the GPU entirely.
