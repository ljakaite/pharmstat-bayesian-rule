"""
Tables from sensitivity_honest.csv for c_protein_design_5.md.

For each environment x sweep x policy: commit rate, retention, failure given
commit with a Wilson 95% interval, mean rounds, and -- for the delta sweep --
an explicit verdict on whether the satisficing guarantee (failure | commit
<= delta) held, judged against the interval as well as the point estimate.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ENVS = ["gb1", "developability", "trpb"]
ENV_LABEL = {"gb1": "GB1", "developability": "Developability (Jain)", "trpb": "TrpB"}
POL_LABEL = {"unified": "Unified, one-sided", "unified_eps": "Unified, two-sided (ε=0.02)",
             "unified_exact": "Unified-exact"}
SWEEPS = [("delta", "δ"), ("noise", "Noise ×"), ("threshold", "Bar")]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    rows = defaultdict(list)
    deltas = {}
    with open(HERE / "sensitivity_honest.csv") as f:
        for r in csv.DictReader(f):
            key = (r["environment"], r["sweep"], float(r["value"]), r["policy"])
            rows[key].append((int(r["committed"]), int(r["good"]), int(r["rounds"])))
            deltas[key] = float(r["delta"])

    out = ["# Sensitivity sweeps on the corrected environments (n = 500 per cell)\n"]
    violations = []
    for sweep, sweep_label in SWEEPS:
        out.append(f"\n## {sweep_label} sweep\n")
        for env in ENVS:
            vals = sorted({k[2] for k in rows if k[0] == env and k[1] == sweep})
            if not vals:
                continue
            out.append(f"\n### {ENV_LABEL[env]}\n")
            out.append(f"| {sweep_label} | Policy | Commit | Retention | Failure \\| commit (k/n) [95% CI] | Rounds | Guarantee |")
            out.append("|---|---|---|---|---|---|---|")
            for v in vals:
                for pol in ["unified", "unified_eps", "unified_exact"]:
                    key = (env, sweep, v, pol)
                    rs = rows.get(key)
                    if not rs:
                        continue
                    n = len(rs)
                    commit = sum(c for c, g, r in rs)
                    good = sum(g for c, g, r in rs)
                    gc = sum(c and g for c, g, r in rs)
                    bad = commit - gc
                    fail = bad / commit if commit else float("nan")
                    lo, hi = wilson(bad, commit)
                    d = deltas[key]
                    if not commit:
                        verdict = "no commits"
                    elif fail <= d:
                        verdict = "held"
                    elif lo <= d:
                        verdict = f"**point {fail:.3f} > δ** (CI covers δ)"
                    else:
                        verdict = f"**VIOLATED** ({fail:.3f} > {d:.2f})"
                        violations.append((ENV_LABEL[env], sweep_label, v, POL_LABEL[pol], fail, d))
                    out.append(f"| {v:g} | {POL_LABEL[pol]} | {commit/n:.3f} | "
                               f"{gc/good if good else float('nan'):.3f} | "
                               f"{fail:.3f} ({bad}/{commit}) [{lo:.3f}, {hi:.3f}] | "
                               f"{sum(r for c, g, r in rs)/n:.2f} | {verdict} |")

    out.append("\n\n## Guarantee violations (failure | commit exceeds δ, interval excludes δ)\n")
    if violations:
        out.append("| Environment | Sweep | Value | Policy | Failure | δ |")
        out.append("|---|---|---|---|---|---|")
        for e, s, v, p, f, d in violations:
            out.append(f"| {e} | {s} | {v:g} | {p} | {f:.3f} | {d:.2f} |")
    else:
        out.append("None.")

    (HERE / "results_sensitivity.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[-40:]))
    print(f"\nWrote {HERE/'results_sensitivity.md'}; {len(violations)} clear violation(s).")


if __name__ == "__main__":
    main()
