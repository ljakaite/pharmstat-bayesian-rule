"""
Tables for c_protein_design_4.md from campaigns_honest.csv (every policy on
the corrected environments). No simulation -- per-campaign rows only.

Writes results_honest.md:
  T1 operating characteristics, all policies, per environment
  T2 paired comparisons: unified-exact vs GP-UCB and vs published unified
  T3 expected-utility grid over cost ratios
  T4 ablations (no look-ahead, flat prior) and the misspecification sweep
"""
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).parent
ENVS = ["gb1", "developability", "trpb"]
ENV_LABEL = {"gb1": "GB1", "developability": "Developability (Jain)", "trpb": "TrpB"}
LABEL = {"greedy": "Greedy", "dapp": "DAPP", "rdm": "RDM", "gpucb": "GP-UCB",
         "unified_exact": "Unified-exact", "exact_no_lookahead": "Exact, no look-ahead",
         "exact_flat_prior": "Exact, flat prior"}
C_BAD = [1, 3, 10]
C_ROUND = [0.005, 0.02, 0.05]


def load():
    d = defaultdict(list)
    with open(HERE / "campaigns_honest.csv") as f:
        for r in csv.DictReader(f):
            d[(r["environment"], r["method"], r["eps"])].append(
                (int(r["committed"]), int(r["good"]), int(r["rounds"])))
    return d


def name(method, eps):
    if method == "unified":
        return "Unified (one-sided)" if eps == "" else f"Unified, ε={float(eps):g}"
    if method.startswith("unified_exact_mis"):
        return f"Unified-exact, bias −{method.split('mis')[1]}σ"
    return LABEL.get(method, method)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summ(rows):
    n = len(rows)
    commit = sum(c for c, g, r in rows)
    good = sum(g for c, g, r in rows)
    gc = sum(c and g for c, g, r in rows)
    bc = commit - gc
    rs = sorted(r for c, g, r in rows)
    lo, hi = wilson(bc, commit)
    return dict(n=n, commit=commit / n, retention=gc / good if good else float("nan"),
                bad=bc, n_commit=commit, fail=bc / commit if commit else float("nan"),
                lo=lo, hi=hi, yld=gc / n, mean=sum(rs) / n, p95=rs[int(0.95 * n) - 1])


def mcnemar(a, b):
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def boot(a, b, n_boot=2000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(d) / n, ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


def utility(rows, c_bad, c_round):
    v = [(1.0 if (c and g) else 0.0) - (c_bad if (c and not g) else 0.0) - c_round * r for c, g, r in rows]
    m = sum(v) / len(v)
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))


def main():
    data = load()
    eps_values = sorted({k[2] for k in data if k[1] == "unified" and k[2]}, key=float)
    mis = sorted({k[1] for k in data if k[1].startswith("unified_exact_mis")})
    out = []

    out.append("## T1. Operating characteristics on the corrected environments "
               "(n = 1,000 paired campaigns, δ = 0.10)\n")
    order = ["greedy", "dapp", "rdm", "gpucb"]
    for env in ENVS:
        out.append(f"\n### {ENV_LABEL[env]}\n")
        out.append("| Policy | Commit | Retention | Fail \\| commit (k/n) [95% CI] | Yield | Rounds mean | p95 |")
        out.append("|---|---|---|---|---|---|---|")
        keys = [(env, m, "") for m in order] + [(env, "unified", "")] + \
               [(env, "unified", e) for e in eps_values] + [(env, "unified_exact", "")]
        for k in keys:
            s = summ(data[k])
            out.append(f"| {name(k[1], k[2])} | {s['commit']:.3f} | {s['retention']:.3f} | "
                       f"{s['fail']:.3f} ({s['bad']}/{s['n_commit']}) [{s['lo']:.3f}, {s['hi']:.3f}] | "
                       f"{s['yld']:.3f} | {s['mean']:.2f} | {s['p95']} |")
        base = sum(g for c, g, r in data[(env, "gpucb", "")]) / 1000
        out.append(f"\nBase rate of truly above-bar candidates: {base:.3f}.")

    out.append("\n\n## T2. Unified-exact vs GP-UCB and vs the published unified gate (paired, n = 1,000)\n")
    out.append("| Environment | Comparator | Retention (exact vs comp.) | Good commits only-E / only-C (p) | "
               "Bad commits only-E / only-C (p) | Rounds exact vs comp. | Δ mean rounds [95% CI] | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|---|---|")
    for env in ENVS:
        e_rows = data[(env, "unified_exact", "")]
        se = summ(e_rows)
        for comp_key, comp_label in [((env, "gpucb", ""), "GP-UCB"),
                                     ((env, "unified", ""), "Unified (one-sided)"),
                                     ((env, "unified", "0.02"), "Unified, ε=0.02")]:
            c_rows = data[comp_key]
            sc = summ(c_rows)
            gx, gy, gp = mcnemar([c and g for c, g, r in e_rows], [c and g for c, g, r in c_rows])
            bx, by, bp = mcnemar([c and not g for c, g, r in e_rows], [c and not g for c, g, r in c_rows])
            er, cr = [r for c, g, r in e_rows], [r for c, g, r in c_rows]
            d, lo, hi = boot(er, cr)
            try:
                wp = f"{wilcoxon(er, cr).pvalue:.2g}"
            except ValueError:
                wp = "n/a"
            out.append(f"| {ENV_LABEL[env]} | {comp_label} | {se['retention']:.3f} vs {sc['retention']:.3f} | "
                       f"{gx} / {gy} ({gp:.2g}) | {bx} / {by} ({bp:.2g}) | {se['mean']:.2f} vs {sc['mean']:.2f} | "
                       f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {wp} |")

    out.append("\n\n## T3. Expected utility per campaign, U = 1·[good commit] − C_bad·[bad commit] − c·rounds (± s.e.)\n")
    for env in ENVS:
        out.append(f"\n### {ENV_LABEL[env]}\n")
        pols = [(env, m, "") for m in order] + [(env, "unified", ""), (env, "unified", "0.02"),
                                                (env, "unified_exact", "")]
        out.append("| C_bad | c/round | " + " | ".join(name(p[1], p[2]) for p in pols) + " |")
        out.append("|---|---|" + "---|" * len(pols))
        for cb in C_BAD:
            for cr in C_ROUND:
                vals = [utility(data[p], cb, cr) for p in pols]
                best = max(range(len(vals)), key=lambda i: vals[i][0])
                cells = [(f"**{m:.3f}**" if i == best else f"{m:.3f}") + f" ±{s:.3f}"
                         for i, (m, s) in enumerate(vals)]
                out.append(f"| {cb} | {cr} | " + " | ".join(cells) + " |")

    out.append("\n\n## T4. Ablations and model misspecification (unified-exact)\n")
    out.append("| Environment | Variant | Commit | Retention | Fail \\| commit [95% CI] | Rounds |")
    out.append("|---|---|---|---|---|---|")
    for env in ENVS:
        for m in ["unified_exact", "exact_no_lookahead", "exact_flat_prior"] + mis:
            s = summ(data[(env, m, "")])
            out.append(f"| {ENV_LABEL[env]} | {name(m, '')} | {s['commit']:.3f} | {s['retention']:.3f} | "
                       f"{s['fail']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}] | {s['mean']:.2f} |")

    (HERE / "results_honest.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
