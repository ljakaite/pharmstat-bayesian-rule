"""
Re-runs EVERY policy on the corrected ("honest") environments and writes
campaigns_honest.csv -- the single results file behind c_protein_design_4.md.

Why a re-run was needed (c_protein_design_4.md §2): on GB1 and TrpB the
published model set contains a component that reproduces the candidate's own
measurement exactly (pairwise, for any lead within two substitutions;
additive, for single mutants). Every policy that consumes `model_preds`
-- unified, greedy, and the informed-prior ablation -- was therefore given
the answer by one of its "competing" models. `use_honest_models()` drops that
component and restricts campaigns to the two-substitution pool, where the two
surviving models have genuine error. Developability has no such leakage
(panel errors 0.20-0.38) and is unchanged; it is re-run here only so that one
file holds every number.

Policies: the four published baselines, the published unified gate (one-sided
and over the eps grid), and the exact-belief gate with its ablations and a
model-misspecification sweep.
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

OUT = HERE / "campaigns_honest.csv"
N = 1000
EPS_GRID = [None, 0.02, 0.05, 0.10, 0.20, 0.30]
BASELINES = ["greedy", "dapp", "rdm", "gpucb"]


def emit(w, env_name, method, eps, results, thr, tv):
    for i, r in enumerate(results):
        w.writerow(dict(environment=env_name, method=method, eps="" if eps is None else eps,
                        campaign=i, committed=int(r["committed"]), good=int(tv(r) >= thr),
                        rounds=r["rounds"]))
    good = [r for r in results if tv(r) >= thr]
    com = [r for r in results if r["committed"]]
    print(f"  [{env_name}] {method:22s}{'' if eps is None else f' eps={eps}':9s} "
          f"commit={len(com)/len(results):.3f} retention={sum(r['committed'] for r in good)/len(good):.3f} "
          f"fail|commit={sum(tv(r) < thr for r in com)/max(1, len(com)):.4f} "
          f"rounds={sum(r['rounds'] for r in results)/len(results):.2f}", flush=True)


def main():
    envs = B.build_envs()
    published = {}   # env name -> (module, call(fn, i, kwargs))

    import gb1_env, dev_env, dev_pomdp as d, trpb_env
    fit = gb1_env.load_gb1(); gm = g.build_models(fit, random.Random(1))
    published["gb1"] = (g, lambda fn, i, kw: fn(fit, gm, random.Random(2000 + i), **kw))
    data = dev_env.load_assays(); comp, _ = dev_env.compute_composite(data)
    dm, _ = dev_env.fit_panel_models(data, comp); names = list(data.keys())
    published["developability"] = (d, lambda fn, i, kw: fn(names, comp, dm, data, random.Random(3000 + i), **kw))
    tf = trpb_env.load_trpb(); wt = trpb_env.reference_variant(tf); tm = t.build_models(tf, random.Random(1), wt)
    published["trpb"] = (t, lambda fn, i, kw: fn(tf, tm, wt, random.Random(4000 + i), **kw))

    exact_variants = [("unified_exact", {}),
                      ("exact_no_lookahead", {"lookahead": False}),
                      ("exact_flat_prior", {"flat_prior": True})]
    exact_variants += [(f"unified_exact_mis{b:g}", {"bias": -b}) for b in B.MIS_BIAS]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["environment", "method", "eps", "campaign",
                                          "committed", "good", "rounds"])
        w.writeheader()
        for env in envs:
            t0 = time.time()
            thr = env.c["threshold"]
            mod, call = published[env.name]
            tv = B.true_of
            pub_tv = lambda r: r.get("true_fitness", r.get("true_val"))
            for m in BASELINES:
                emit(w, env.name, m, None, [call(getattr(mod, f"run_campaign_{m}"), i, {}) for i in range(N)], thr, pub_tv)
            for eps in EPS_GRID:
                emit(w, env.name, "unified", eps,
                     [call(mod.run_campaign_unified, i, {"eps_abandon": eps}) for i in range(N)], thr, pub_tv)
            for label, variant in exact_variants:
                emit(w, env.name, label, None,
                     [env.campaign(B.BASE_SEED[env.name] + i, variant) for i in range(N)], thr, tv)
            print(f"[{env.name}] done in {time.time()-t0:.0f}s", flush=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
