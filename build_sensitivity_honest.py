"""
Sensitivity sweeps on the CORRECTED environments (c_protein_design_5.md).

c_SREP_21/22.md swept delta, assay noise and bar height for the published
one-sided gate, on the environments later found to leak (c_protein_design_4.md
§I.1). This re-runs all three sweeps on the corrected environments and adds
the two policies those sweeps never covered: the two-sided gate and the
exact-belief gate.

Grids are exactly c_SREP_21/22.md's, so the corrected numbers are directly
comparable with the superseded ones:
  delta      0.01, 0.05, 0.10, 0.20, 0.30                (BASE_SEED 5000)
  noise      x0.5, x1, x1.5, x2, x3 on both sigmas       (BASE_SEED 5500)
  threshold  per environment, five bar heights            (BASE_SEED 5500)

n = 500 campaigns per cell, paired across policies within a cell. When the
bar moves, "good" is recomputed at the swept bar, so retention and failure
always refer to the same specification the policy was given.

Writes sensitivity_honest.csv (one row per campaign) and, via
analyse_sensitivity.py, the tables.
"""
import csv
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
for sub in ("", "gb1", "developability", "trpb"):
    sys.path.insert(0, str(HERE / sub))

import gb1_stagegate as g
import trpb_stagegate as t
g.use_honest_models()
t.use_honest_models()

import build_exact_results as B

OUT = HERE / "sensitivity_honest.csv"
N = 500
SEED_DELTA, SEED_OTHER = 5000, 5500
DELTAS = [0.01, 0.05, 0.10, 0.20, 0.30]
NOISE = [0.5, 1.0, 1.5, 2.0, 3.0]
THRESHOLDS = {"gb1": [0.7, 0.85, 1.0, 1.3, 1.6],
              "developability": [-0.2, -0.1, 0.0, 0.1, 0.2],
              "trpb": [0.05, 0.08, 0.1, 0.15, 0.2]}
POLICIES = ["unified", "unified_eps", "unified_exact"]


def main():
    envs = {e.name: e for e in B.build_envs()}

    import gb1_env, dev_env, dev_pomdp as d, trpb_env
    fit = gb1_env.load_gb1(); gm = g.build_models(fit, random.Random(1))
    data = dev_env.load_assays(); comp, _ = dev_env.compute_composite(data)
    dm, _ = dev_env.fit_panel_models(data, comp); names = list(data.keys())
    tf = trpb_env.load_trpb(); wt = trpb_env.reference_variant(tf)
    tm = t.build_models(tf, random.Random(1), wt)

    published = {
        "gb1": (g, lambda i, kw: g.run_campaign_unified(fit, gm, random.Random(i), **kw)),
        "developability": (d, lambda i, kw: d.run_campaign_unified(names, comp, dm, data, random.Random(i), **kw)),
        "trpb": (t, lambda i, kw: t.run_campaign_unified(tf, tm, wt, random.Random(i), **kw)),
    }

    def true_of(r):
        return r.get("true_fitness", r.get("true_val"))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["environment", "policy", "sweep", "value", "delta",
                                          "campaign", "committed", "good", "rounds"])
        w.writeheader()
        for env_name, env in envs.items():
            t0 = time.time()
            base_thr = env.c["threshold"]
            base_sc, base_sm = env.c["sigma_cheap"], env.c["sigma_monitor"]
            cells = ([("delta", dl, SEED_DELTA, dict(delta=dl), base_thr) for dl in DELTAS] +
                     [("noise", m, SEED_OTHER,
                       dict(sigma_cheap=base_sc * m, sigma_monitor=base_sm * m), base_thr) for m in NOISE] +
                     [("threshold", th, SEED_OTHER, dict(threshold=th), th) for th in THRESHOLDS[env_name]])
            for sweep, value, seed, kw, thr in cells:
                delta = kw.get("delta", 0.10)
                for policy in POLICIES:
                    if policy == "unified_exact":
                        res = [env.campaign(seed + i, {}, overrides=kw) for i in range(N)]
                    else:
                        kw2 = dict(kw)
                        if policy == "unified_eps":
                            kw2["eps_abandon"] = 0.02
                        res = [published[env_name][1](seed + i, kw2) for i in range(N)]
                    for i, r in enumerate(res):
                        w.writerow(dict(environment=env_name, policy=policy, sweep=sweep, value=value,
                                        delta=delta, campaign=i, committed=int(r["committed"]),
                                        good=int(true_of(r) >= thr), rounds=r["rounds"]))
                    com = [r for r in res if r["committed"]]
                    good = [r for r in res if true_of(r) >= thr]
                    fail = sum(true_of(r) < thr for r in com) / max(1, len(com))
                    flag = "" if fail <= delta or not com else "  <-- EXCEEDS delta"
                    print(f"  [{env_name}] {policy:14s} {sweep:9s}={value:<6} commit={len(com)/N:.3f} "
                          f"retention={sum(r['committed'] for r in good)/max(1,len(good)):.3f} "
                          f"fail={fail:.3f} rounds={sum(r['rounds'] for r in res)/N:.2f}{flag}", flush=True)
            print(f"[{env_name}] done in {time.time()-t0:.0f}s", flush=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
