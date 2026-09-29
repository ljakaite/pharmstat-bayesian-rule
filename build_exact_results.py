"""
Runs the exact-belief gate (exact_gate.py) on all three real environments,
on exactly the campaigns every published policy already ran (same BASE_SEEDs,
same lead-selection rule, same noise and costs), and appends the results to
campaigns.csv's schema in campaigns_exact.csv.

Variants run per environment (c_protein_design_4.md):
  unified_exact          full policy: exact mixture belief + one-step
                         look-ahead stopping/precision, delta = 0.10
  exact_no_lookahead     ablation: same belief, no look-ahead (cheap assay
                         every round until the gate resolves) -- isolates
                         what the computed option value contributes
  exact_flat_prior       ablation: same policy, priors replaced by a single
                         vague component centred at the population mean --
                         isolates what the domain models contribute
  unified_exact_mis*     misspecification sweep: every model prediction
                         shifted by a bias of k * (model's own error sd),
                         k in {0.5, 1, 2}, testing the delta guarantee when
                         M is wrong (the one axis never tested,
                         c_SREP_23.md step 2)

Model prior widths are each model's measured RMSE against the real data on a
sample of the same candidate pool the campaigns draw from (fixed separate
seed, never a campaign seed).
"""
import csv
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
for sub in ("", "gb1", "developability", "trpb"):
    sys.path.insert(0, str(HERE / sub))

from exact_gate import model_error_sds, run_campaign_exact

OUT = HERE / "campaigns_exact.csv"
N = 1000
CAL_SAMPLE = 2000
MIS_BIAS = [0.5, 1.0, 2.0]


def true_of(r):
    return r["true_val"]


# ── environment adapters ────────────────────────────────────────────────────

class Env:
    """Wraps one environment: how to pick a lead on a given rng (identically
    to the published policies), what the models predict for it, the assay
    noise/costs, and the model prior widths measured from real data."""

    def __init__(self, name, mod, pick, preds_for, sample_for, constants, pool, truth):
        self.name, self.mod, self.pick, self.preds_for = name, mod, pick, preds_for
        self.sample_for, self.c, self.pool, self.truth = sample_for, constants, pool, truth
        self.model_sds = self._calibrate()
        self.pop_mean = statistics.mean(truth(v) for v in pool)
        self.pop_sd = statistics.pstdev(truth(v) for v in pool)
        print(f"[{name}] model prior sds = "
              f"{', '.join(f'{m}={s:.3f}' for m, s in zip(mod.MODELS, self.model_sds))}; "
              f"pop mean={self.pop_mean:.3f} sd={self.pop_sd:.3f}", flush=True)

    def _calibrate(self):
        rng = random.Random(99991)          # never a campaign seed
        sample = [self.pool[rng.randrange(len(self.pool))] for _ in range(CAL_SAMPLE)]
        series = [[] for _ in self.mod.MODELS]
        for lead in sample:
            preds = self.preds_for(lead)
            if preds is None:
                continue
            y = self.truth(lead)
            for i, p in enumerate(preds):
                series[i].append((p, y))
        return model_error_sds(series)

    def campaign(self, seed, variant, overrides=None):
        rng = random.Random(seed)
        lead = self.pick(rng)
        preds = self.preds_for(lead)
        while preds is None:                 # same retry as run_campaign_unified
            lead = self.pick(rng)
            preds = self.preds_for(lead)
        sds = list(self.model_sds)
        if variant.get("flat_prior"):
            preds, sds = [self.pop_mean], [self.pop_sd]
        elif variant.get("bias"):
            preds = [p + variant["bias"] * s for p, s in zip(preds, sds)]
        c = dict(self.c)
        # Sensitivity sweeps (c_protein_design_5.md) override delta, the two
        # sigmas, or the bar height; everything else stays the environment's own.
        kw = dict(threshold=c["threshold"], sigma_cheap=c["sigma_cheap"],
                  sigma_monitor=c["sigma_monitor"], delta=0.10)
        kw.update(overrides or {})
        return run_campaign_exact(
            preds, sds,
            sample_fn=lambda sigma: self.sample_for(lead, sigma, rng),
            true_val_fn=lambda: self.truth(lead),
            cost_cheap=c["cost_cheap"], cost_monitor=c["cost_monitor"],
            cost_freeze=c["cost_freeze"], cost_failure=c["cost_failure"], gamma=c["gamma"],
            t_max=c["t_max"], lookahead=variant.get("lookahead", True), **kw)


def build_envs():
    import gb1_env, gb1_stagegate as g
    import dev_env, dev_pomdp as d
    import trpb_env, trpb_stagegate as t

    def consts(m, thr):
        return dict(threshold=thr, cost_cheap=m.COST_CHEAP, cost_monitor=m.COST_MONITOR,
                    cost_freeze=m.COST_FREEZE, cost_failure=m.COST_FAILURE, gamma=m.GAMMA,
                    t_max=m.T_MAX_ROUNDS, sigma_cheap=m.SIGMA_CHEAP, sigma_monitor=m.SIGMA_MONITOR)

    fit = gb1_env.load_gb1()
    gm = g.build_models(fit, random.Random(1))
    gb1_pool = [v for v in fit if g.MIN_HD <= gb1_env.hamming(v, gb1_env.WT) <= 2]

    def gb1_preds(lead):
        p = [gm[m](lead) for m in g.MODELS]
        return None if any(x is None for x in p) else p

    envs = [Env("gb1", g, lambda rng: g.pick_lead(fit, rng), gb1_preds,
                lambda lead, sigma, rng: rng.gauss(fit[lead], sigma),
                consts(g, g.FITNESS_THRESHOLD), gb1_pool, lambda v: fit[v])]

    data = dev_env.load_assays()
    comp, _ = dev_env.compute_composite(data)
    dm, _ = dev_env.fit_panel_models(data, comp)
    names = list(data.keys())
    envs.append(Env("developability", d, lambda rng: rng.choice(names),
                    lambda lead: list(dev_env.panel_predict(dm, lead, data).values()),
                    lambda lead, sigma, rng: rng.gauss(comp[lead], sigma),
                    consts(d, d.THRESHOLD), names, lambda v: comp[v]))

    tf = trpb_env.load_trpb()
    wt = trpb_env.reference_variant(tf)
    tm = t.build_models(tf, random.Random(1), wt)
    trpb_pool = [v for v in tf if t.MIN_HD <= sum(a != b for a, b in zip(v, wt)) <= 2]

    def trpb_preds(lead):
        p = [tm[m](lead) for m in t.MODELS]
        return None if any(x is None for x in p) else p

    envs.append(Env("trpb", t, lambda rng: t.pick_lead(tf, wt, rng), trpb_preds,
                    lambda lead, sigma, rng: rng.gauss(tf[lead], sigma),
                    consts(t, t.FITNESS_THRESHOLD), trpb_pool, lambda v: tf[v]))
    return envs


BASE_SEED = {"gb1": 2000, "developability": 3000, "trpb": 4000}


def main():
    envs = build_envs()
    variants = [("unified_exact", {}),
                ("exact_no_lookahead", {"lookahead": False}),
                ("exact_flat_prior", {"flat_prior": True})]
    variants += [(f"unified_exact_mis{b:g}", {"bias": -b}) for b in MIS_BIAS]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["environment", "method", "eps", "campaign",
                                          "committed", "good", "rounds"])
        w.writeheader()
        for env in envs:
            thr = env.c["threshold"]
            for label, variant in variants:
                t0 = time.time()
                res = [env.campaign(BASE_SEED[env.name] + i, variant) for i in range(N)]
                for i, r in enumerate(res):
                    w.writerow(dict(environment=env.name, method=label, eps="", campaign=i,
                                    committed=int(r["committed"]), good=int(true_of(r) >= thr),
                                    rounds=r["rounds"]))
                good = [r for r in res if true_of(r) >= thr]
                com = [r for r in res if r["committed"]]
                print(f"  [{env.name}] {label:22s} commit={len(com)/N:.3f} "
                      f"retention={sum(r['committed'] for r in good)/len(good):.3f} "
                      f"fail|commit={sum(true_of(r) < thr for r in com)/max(1,len(com)):.4f} "
                      f"rounds={sum(r['rounds'] for r in res)/N:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
