"""
Figure 3: does a gate respect the failure budget it states?

x = the budget delta the policy was run with; y = the failure rate actually
observed among committed candidates. The diagonal is the promise. Anything
in the shaded region above it is a gate that broke its own guarantee.

Reads sensitivity_honest.csv (delta sweep) only.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
ENVS = [("gb1", "GB1"), ("developability", "Developability (Jain)"), ("trpb", "TrpB")]
POLICIES = [("unified", "Satisficing, one-sided", BLUE, "o"),
            ("unified_eps", "Satisficing, two-sided", ORANGE, "s"),
            ("unified_exact", "Proposed", AQUA, "*")]

rows = defaultdict(list)
with open(HERE / "sensitivity_honest.csv") as f:
    for r in csv.DictReader(f):
        if r["sweep"] == "delta":
            rows[(r["environment"], r["policy"], float(r["value"]))].append(
                (int(r["committed"]), int(r["good"])))


def fail_rate(key):
    v = rows[key]
    com = sum(c for c, g in v)
    return (com - sum(c and g for c, g in v)) / com if com else float("nan")


plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (env, title) in zip(axes, ENVS):
    deltas = sorted({k[2] for k in rows if k[0] == env})
    ax.fill_between([0, 0.32], [0, 0.32], 0.32, color="#fdf1ec", lw=0)  # = #eb6834 at 7% over white; solid so EPS matches
    ax.plot([0, 0.32], [0, 0.32], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    for pol, label, colour, marker in POLICIES:
        ys = [fail_rate((env, pol, d)) for d in deltas]
        ax.plot(deltas, ys, marker=marker, color=colour, lw=1.8,
                ms=11 if marker == "*" else 5.5, mec="white", mew=0.8, zorder=4, label=label)
    ax.set_xlim(0, 0.32)
    ax.set_ylim(0, 0.32)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("Stated failure budget δ")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Observed failure rate\namong committed candidates")
axes[0].annotate("budget exceeded", (0.055, 0.245), color="#b8471f", fontsize=8)
axes[0].annotate("promise", (0.20, 0.165), color=MUTED, fontsize=8, rotation=32, ha="center")
axes[2].legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.6)
fig.tight_layout()
for ext in ("png", "pdf", "eps", "tiff"):
    extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
    fig.savefig(HERE / f"calibration_figure.{ext}", dpi=(600 if ext == "tiff" else 200), bbox_inches="tight", **extra)
print("wrote calibration_figure.png/.pdf")
