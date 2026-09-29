"""
Figure 2 of c_protein_design_4.md: assay cost vs good-candidate retention on
the corrected environments, one panel per dataset, reading
campaigns_honest.csv only.

Every point carries a direct label, so identity never rests on colour (and
the aqua slot's sub-3:1 contrast is relieved, per the palette rule).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
ENVS = [("gb1", "GB1"), ("developability", "Developability (Jain)"), ("trpb", "TrpB")]
BASE = {"greedy": ("Greedy", "s"), "dapp": ("DAPP", "^"), "rdm": ("RDM", "D")}

rows = defaultdict(list)
with open(HERE / "campaigns_honest.csv") as f:
    for r in csv.DictReader(f):
        rows[(r["environment"], r["method"], r["eps"])].append(
            (int(r["committed"]), int(r["good"]), int(r["rounds"])))


def point(key):
    rs = rows[key]
    good = sum(g for _, g, _ in rs)
    return sum(r for *_, r in rs) / len(rs), sum(c and g for c, g, _ in rs) / good


plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
for ax, (env, title) in zip(axes, ENVS):
    eps = sorted({k[2] for k in rows if k[0] == env and k[1] == "unified" and k[2]}, key=float)
    curve = [point((env, "unified", e)) for e in eps]
    ax.plot([x for x, _ in curve], [y for _, y in curve], "-o", color=BLUE, lw=2, ms=4.5, zorder=3)
    two_sided_off = (2, -16) if env == "trpb" else (7, -3)
    ax.annotate("Unified, two-sided", curve[-1], textcoords="offset points", xytext=two_sided_off,
                color=INK, fontsize=8, ha="left")
    x, y = point((env, "unified", ""))
    ax.plot(x, y, "o", mfc="white", mec=BLUE, mew=2, ms=7, zorder=3)
    ax.annotate("Unified, one-sided", (x, y), textcoords="offset points", xytext=(7, -3),
                color=INK, fontsize=8)
    x, y = point((env, "gpucb", ""))
    ax.plot(x, y, "P", color=ORANGE, ms=9, mec="white", mew=1, zorder=4)
    ax.annotate("GP-UCB", (x, y), textcoords="offset points", xytext=((6, 6) if env == "trpb" else (2, 8)), color=INK, fontsize=8)
    x, y = point((env, "unified_exact", ""))
    ax.plot(x, y, "*", color=AQUA, ms=15, mec="white", mew=1, zorder=5)
    exact_off = (-10, -4) if env == "trpb" else (-8, 8)
    ax.annotate("Unified-exact", (x, y), textcoords="offset points", xytext=exact_off,
                color=INK, fontsize=8, ha="right", fontweight="bold")
    for m, (lab, mk) in BASE.items():
        x, y = point((env, m, ""))
        ax.plot(x, y, mk, color=GREY, ms=6, zorder=2)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(-7, -3), color=MUTED,
                    fontsize=7, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.set_xlim(1, 24)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("Mean assay rounds per campaign (log)")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Good-candidate retention\nP(commit | truly above bar)")
axes[0].set_ylim(0.75, 1.01)
fig.tight_layout()
for ext in ("png", "pdf", "eps", "tiff"):
    extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
    fig.savefig(HERE / f"honest_figure.{ext}", dpi=(600 if ext == "tiff" else 200), bbox_inches="tight", **extra)
print("wrote honest_figure.png/.pdf")
