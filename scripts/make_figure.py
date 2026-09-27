"""The plant family as a picture: what the table was taught, and what it is asked to drive.

The point of the figure is that the eight test pendulums sit *inside* the cloud of training
pendulums. That is what makes the claim interpolation rather than extrapolation, and it is much
easier to see than to describe.

  python scripts/make_figure.py --out media/training_data.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tabpfn_control.teacher import LP_RANGE, MTIP_RANGE, TEST_PLANTS

# Chart surface and ink, and two categorical slots that validate as a pair for scatter.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
TRAIN, TEST = "#2a78d6", "#eb6834"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/demos.npz")
    ap.add_argument("--results", default="results/test_plants.json",
                    help="if present, rings any test plant the policy failed")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="media/training_data.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    xi = np.load(args.demos)["xi"]                       # (n_plants, 3): length, tip mass, direction
    ok = {}
    if Path(args.results).exists():
        for p in json.loads(Path(args.results).read_text())["plants"]:
            runs = p["rolling"]                              # one entry per seed
            ok[p["i"]] = all(r["swingup_s"] is not None and r.get("held_after_catch", 1.0) > 0.95
                             for r in runs)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    ax.add_patch(Rectangle((LP_RANGE[0] * 100, MTIP_RANGE[0] * 1e3),
                           (LP_RANGE[1] - LP_RANGE[0]) * 100, (MTIP_RANGE[1] - MTIP_RANGE[0]) * 1e3,
                           facecolor="#2a78d6", alpha=0.045, edgecolor=GRID, lw=1.0, zorder=0))

    for sign, marker, lbl in [(1.0, "o", "motor forward"), (-1.0, "s", "motor reversed")]:
        m = xi[:, 2] == sign
        ax.scatter(xi[m, 0] * 100, xi[m, 1] * 1e3, s=42, marker=marker, c=TRAIN,
                   alpha=0.8, linewidths=1.4, edgecolors=SURFACE, zorder=2)

    for i, p in enumerate(TEST_PLANTS):
        marker = "o" if p["motor_sign"] > 0 else "s"
        failed = ok.get(i, True) is False
        ax.scatter([p["Lp"] * 100], [p["m_tip"] * 1e3], s=210, marker=marker, c=TEST,
                   linewidths=2.0, edgecolors=SURFACE, zorder=4)
        if failed:                                        # never quietly hide one that did not work
            ax.scatter([p["Lp"] * 100], [p["m_tip"] * 1e3], s=520, marker=marker,
                       facecolors="none", edgecolors=TEST, linewidths=1.6, linestyle=":", zorder=3)
        ax.annotate(str(i + 1), (p["Lp"] * 100, p["m_tip"] * 1e3), color=SURFACE, zorder=5,
                    ha="center", va="center", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("pendulum length  [cm]", color=INK_2, fontsize=10.5, labelpad=8)
    ax.set_ylabel("mass at the tip  [g]", color=INK_2, fontsize=10.5, labelpad=8)
    ax.set_title("One frozen table, taught on the blue pendulums, asked to drive the orange ones",
                 color=INK, fontsize=12, pad=14, loc="left")
    ax.grid(True, color=GRID, lw=0.7, alpha=0.7, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.set_xlim(LP_RANGE[0] * 100 - 1.6, LP_RANGE[1] * 100 + 1.6)
    ax.set_ylim(MTIP_RANGE[0] * 1e3 - 3.2, MTIP_RANGE[1] * 1e3 + 3.2)

    key = [Line2D([], [], marker="o", ls="", ms=6.5, mfc=TRAIN, mec=SURFACE,
                  label=f"{len(xi)} training pendulums"),
           Line2D([], [], marker="o", ls="", ms=10, mfc=TEST, mec=SURFACE,
                  label=f"{len(TEST_PLANTS)} validation pendulums, never fitted on"),
           Line2D([], [], marker="o", ls="", ms=6.5, mfc=INK_2, mec=SURFACE, label="motor forward"),
           Line2D([], [], marker="s", ls="", ms=6.5, mfc=INK_2, mec=SURFACE, label="motor reversed")]
    if any(v is False for v in ok.values()):
        key.append(Line2D([], [], marker="o", ls=":", ms=12, mfc="none", mec=TEST,
                          label="did not swing up"))
    leg = ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(key),
                    frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK_2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor=SURFACE)
    print(f"{len(xi)} training plants, {len(TEST_PLANTS)} validation plants -> {args.out} "
          f"({args.dpi} dpi)" + (f", {sum(v is False for v in ok.values())} marked as failing" if ok else ""))


if __name__ == "__main__":
    main()
