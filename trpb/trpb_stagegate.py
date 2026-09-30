"""
TrpB stage-gate -- fourth benchmark (c_SREP_16.md), structurally identical
to gb1_stagegate.py (same single-lead optimal-stopping design, same three
competing models: additive/pairwise/global-nonlinear). Kept as a separate
module rather than refactored into a shared one, to avoid any risk of
regressing the already-validated, already-reported GB1/developability
numbers -- consistent with how dev_pomdp.py already duplicates rather than
shares this same logic.

**Recalibrated constants, not reused blindly from GB1** -- TrpB's fitness
scale is much smaller and more skewed (population sd 0.060 vs. GB1's 0.395,
mean 0.021 vs. a landscape centred near its own reference point) per
c_SREP_16.md's calibration check:
  - threshold = 0.1, single-mutant pre-filter bar = 0.05 -> ~50% base rate
    among promising leads (checked empirically, not assumed -- the same
    discipline that caught GB1's 10% base-rate problem in c_SREP_11.md)
  - noise (SIGMA_CHEAP/MONITOR) and cost constants scaled down by roughly
    the same ratio as the population sd ratio to GB1's (~1/6.6), so the
    option-value dimensional-consistency fix from c_SREP_13.md doesn't
    silently break on a landscape with a very different scale
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stagegate_baselines import (run_campaign_dapp as _shared_dapp, run_campaign_rdm as _shared_rdm,
                                  run_campaign_gpucb as _shared_gpucb)

from trpb_env import load_trpb, reference_variant, hamming, additive_prediction, AA20

MODELS = ["additive", "pairwise", "global_nonlinear"]
GAMMA = 0.95
DELTA = 0.10
LAMBDA_VOI = 1.0
COST_CHEAP = 0.003
COST_MONITOR = 0.008
COST_FREEZE = 0.015
COST_FAILURE = 0.45
FITNESS_THRESHOLD = 0.1
SIGMA_CHEAP = 0.04
SIGMA_MONITOR = 0.012
T_MAX_ROUNDS = 30
TRPB_FITNESS_PRIOR_SD = 0.05988734106908349  # population sd, computed in c_SREP_16.md


def build_pairwise_table(fitness: dict, wt: str):
    pair_correction = {}
    for i in range(4):
        for j in range(i + 1, 4):
            for ai in AA20:
                for aj in AA20:
                    if ai == wt[i] and aj == wt[j]:
                        continue
                    v = list(wt)
                    v[i], v[j] = ai, aj
                    v = "".join(v)
                    if v not in fitness:
                        continue
                    s_i = wt[:i] + ai + wt[i + 1:]
                    s_j = wt[:j] + aj + wt[j + 1:]
                    if s_i not in fitness or s_j not in fitness:
                        continue
                    additive2 = fitness[wt] + (fitness[s_i] - fitness[wt]) + (fitness[s_j] - fitness[wt])
                    pair_correction[(i, ai, j, aj)] = fitness[v] - additive2
    return pair_correction


def pairwise_prediction(variant: str, fitness: dict, pair_table: dict, wt: str):
    pred = additive_prediction(variant, fitness, wt)
    if pred is None:
        return None
    mutated = [(i, variant[i]) for i in range(4) if variant[i] != wt[i]]
    for a in range(len(mutated)):
        for b in range(a + 1, len(mutated)):
            i, ai = mutated[a]
            j, aj = mutated[b]
            key = (i, ai, j, aj) if i < j else (j, aj, i, ai)
            pred += pair_table.get(key, 0.0)
    return pred


def build_global_nonlinear_link(fitness: dict, wt: str, rng: random.Random = None, n_fit: int = 4000):
    from sklearn.isotonic import IsotonicRegression
    rng = rng or random.Random(7)
    variants = [v for v in fitness if hamming(v, wt) >= 2]
    sample = rng.sample(variants, min(n_fit, len(variants)))
    xs, ys = [], []
    for v in sample:
        pred = additive_prediction(v, fitness, wt)
        if pred is not None:
            xs.append(pred)
            ys.append(fitness[v])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(xs, ys)
    return iso


def build_models(fitness: dict, rng: random.Random, wt: str):
    pair_table = build_pairwise_table(fitness, wt)
    iso = build_global_nonlinear_link(fitness, wt, rng)

    def f_additive(v):
        return additive_prediction(v, fitness, wt)

    def f_pairwise(v):
        return pairwise_prediction(v, fitness, pair_table, wt)

    def f_global(v):
        a = additive_prediction(v, fitness, wt)
        return None if a is None else float(iso.predict([a])[0])

    return {"additive": f_additive, "pairwise": f_pairwise, "global_nonlinear": f_global}


def _logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))


def update_belief(pi_m, model_preds, obs, sigma, floor=0.01):
    log_m = [_logpdf(obs, model_preds[m], sigma) for m in MODELS]
    mx = max(log_m)
    pi_m = [p * math.exp(l - mx) for p, l in zip(pi_m, log_m)]
    s = sum(pi_m)
    pi_m = [p / s for p in pi_m]
    return [(1 - floor) * p + floor / len(pi_m) for p in pi_m]


def posterior_mean_h(pi_m, model_preds, obs_history):
    model_mean = sum(p * model_preds[m] for p, m in zip(pi_m, MODELS))
    if not obs_history:
        return model_mean
    obs_mean = sum(obs_history) / len(obs_history)
    n = len(obs_history)
    w = n / (n + 2.0)
    return w * obs_mean + (1 - w) * model_mean


def _shrink_toward_evidence(prior_val, obs_history):
    """Bug fix, c_SREP_19.md (found via c_SREP_18.md's ablation) -- see
    gb1_stagegate.py's version for the full rationale."""
    if not obs_history:
        return prior_val
    obs_mean = sum(obs_history) / len(obs_history)
    n = len(obs_history)
    w = n / (n + 2.0)
    return w * obs_mean + (1 - w) * prior_val


def satisficing_check(pi_m, model_preds, h, obs_history, rng, delta=DELTA, n_samples=200,
                       sigma_cheap=SIGMA_CHEAP, threshold=FITNESS_THRESHOLD):
    n = len(obs_history)
    sd = sigma_cheap / math.sqrt(n + 1)
    hits = 0
    for _ in range(n_samples):
        m = MODELS[rng.choices(range(len(MODELS)), weights=pi_m)[0]]
        sample_mean = _shrink_toward_evidence(model_preds[m], obs_history)
        sample = rng.gauss(sample_mean, sd)
        if sample >= threshold:
            hits += 1
    return (hits / n_samples) >= (1.0 - delta)


def p_good_estimate(pi_m, model_preds, obs_history, rng, n_samples=200,
                    sigma_cheap=SIGMA_CHEAP, threshold=FITNESS_THRESHOLD):
    """P(true value >= threshold) under the same sampling scheme as
    satisficing_check, returned as a probability instead of a pass/fail
    verdict. Used only by the two-boundary (eps_abandon) variant of
    run_campaign_unified (c_protein_design_3.md)."""
    n = len(obs_history)
    sd = sigma_cheap / math.sqrt(n + 1)
    hits = 0
    for _ in range(n_samples):
        m = MODELS[rng.choices(range(len(MODELS)), weights=pi_m)[0]]
        sample = rng.gauss(_shrink_toward_evidence(model_preds[m], obs_history), sd)
        if sample >= threshold:
            hits += 1
    return hits / n_samples


def voi_estimate(pi_m, model_preds, h, obs_history, rng, n_samples=20,
                  sigma_cheap=SIGMA_CHEAP, threshold=FITNESS_THRESHOLD):
    n = len(obs_history)
    sd_now = sigma_cheap / math.sqrt(n + 1)
    sd_after = sigma_cheap / math.sqrt(n + 2)
    closeness = math.exp(-((h - threshold) ** 2) / (2 * sd_now ** 2 + 1e-9))
    return max(0.0, (sd_now - sd_after) * closeness * 5.0)


def option_value_raw(pi_m, model_preds, h, obs_history, rng, T_horizon=6, assay_cost_rate=COST_CHEAP,
                      sigma_cheap=SIGMA_CHEAP, threshold=FITNESS_THRESHOLD):
    v_freeze = h - COST_FREEZE
    n = len(obs_history)
    sd_future = sigma_cheap / math.sqrt(n + T_horizon + 1)
    sd_now = sigma_cheap / math.sqrt(n + 1)
    closeness = math.exp(-((h - threshold) ** 2) / (2 * sd_now ** 2 + 1e-9))
    reliability_gain = (sd_now - sd_future) * closeness * 2.0
    v_wait = h + reliability_gain - assay_cost_rate * T_horizon - COST_FREEZE
    return v_wait - v_freeze




# ── Honest model set (c_protein_design_4.md) ────────────────────────────────
# The pairwise model's correction table is built FROM the measured double
# mutants, so for any lead at Hamming distance <= 2 from the reference it
# reproduces that lead's own measurement exactly (verified: mean |error| =
# 0.00000 over the whole candidate pool). The additive model is likewise
# exact by construction on single mutants. A model that returns the answer
# is not a competing hypothesis about it, so results on the published model
# set are inflated for every policy that consumes `model_preds`.
# `use_honest_models()` restricts campaigns to the two-substitution pool and
# to the models that genuinely generalise there (additive: mean |error|
# 0.56; global_nonlinear: 0.07).
MIN_HD = 1


def use_honest_models():
    global MODELS, MIN_HD
    MODELS = ["additive", "global_nonlinear"]
    MIN_HD = 2


_LEAD_POOL_CACHE = {}


def pick_lead(fitness, wt, rng, min_hd=None, max_hd=2, single_mutant_bar=0.05):
    min_hd = MIN_HD if min_hd is None else min_hd
    key = (id(fitness), wt, min_hd, max_hd, single_mutant_bar)
    cached = _LEAD_POOL_CACHE.get(key)
    if cached is None:
        # The stored tuple keeps a reference to `fitness`, so its id() cannot be
        # recycled by the garbage collector while this entry is live -- without
        # that, a freed dict's address could be reused and collide on the key.
        cached = (*_build_lead_pool(fitness, wt, min_hd, max_hd, single_mutant_bar), fitness)
        _LEAD_POOL_CACHE[key] = cached
    promising, candidates, _ = cached
    return rng.choice(promising) if promising else rng.choice(candidates)


def _build_lead_pool(fitness, wt, min_hd, max_hd, single_mutant_bar):
    """The candidate pool depends only on (fitness, wt, min_hd, max_hd, bar), so
    it is identical for every campaign, but was rebuilt on every call --
    rescanning the whole landscape each time (c_protein_design_7.md). Memoised;
    `rng.choice` still draws from the same list, so leads are unchanged."""
    candidates = [v for v in fitness if min_hd <= hamming(v, wt) <= max_hd]
    promising = []
    for v in candidates:
        ok = True
        for i in range(4):
            if v[i] == wt[i]:
                continue
            single = wt[:i] + v[i] + wt[i + 1:]
            if fitness.get(single, 0.0) < single_mutant_bar:
                ok = False
                break
        if ok:
            promising.append(v)
    return promising, candidates


def run_campaign_unified(fitness, models, wt, rng, lam=LAMBDA_VOI, delta=DELTA, t_max=T_MAX_ROUNDS,
                          sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
                          threshold=FITNESS_THRESHOLD, eps_abandon=None):
    """sigma_cheap/sigma_monitor/threshold default to the module constants
    (unchanged behaviour for every existing caller) but are now overridable
    -- added in c_SREP_22.md, same as gb1_stagegate.py."""
    lead = pick_lead(fitness, wt, rng)
    model_preds = {m: models[m](lead) for m in MODELS}
    if any(p is None for p in model_preds.values()):
        return run_campaign_unified(fitness, models, wt, rng, lam, delta, t_max,
                                     sigma_cheap, sigma_monitor, threshold, eps_abandon)
    pi_m = [1.0 / len(MODELS)] * len(MODELS)
    obs_history = []
    cum_r = 0.0
    for t in range(t_max):
        h = posterior_mean_h(pi_m, model_preds, obs_history)
        sat = satisficing_check(pi_m, model_preds, h, obs_history, rng, delta=delta,
                                 sigma_cheap=sigma_cheap, threshold=threshold)
        voi = voi_estimate(pi_m, model_preds, h, obs_history, rng,
                            sigma_cheap=sigma_cheap, threshold=threshold) if sat else 0.0
        ov = option_value_raw(pi_m, model_preds, h, obs_history, rng,
                               sigma_cheap=sigma_cheap, threshold=threshold)
        use_monitor = (lam * voi) > COST_MONITOR - COST_CHEAP
        sigma = sigma_monitor if use_monitor else sigma_cheap
        obs = rng.gauss(fitness[lead], sigma)
        obs_history.append(obs)
        pi_m = update_belief(pi_m, model_preds, obs, sigma)
        cum_r += (GAMMA ** t) * (-(COST_MONITOR if use_monitor else COST_CHEAP))
        if ov < 0.0:
            if eps_abandon is None:
                break
            # Two-boundary stop (c_protein_design_3.md): option value says
            # "ready to decide", but only stop if the decision is actually
            # clear -- commit side P(good) >= 1-delta, abandon side
            # P(good) <= eps_abandon. In between, keep assaying (t_max caps it).
            p = p_good_estimate(pi_m, model_preds, obs_history, rng,
                                sigma_cheap=sigma_cheap, threshold=threshold)
            if p >= 1.0 - delta or p <= eps_abandon:
                break

    h_final = posterior_mean_h(pi_m, model_preds, obs_history)
    true_fitness = fitness[lead]
    sat_final = satisficing_check(pi_m, model_preds, h_final, obs_history, rng, delta=delta,
                                   sigma_cheap=sigma_cheap, threshold=threshold)
    if sat_final:
        committed = True
        failed = true_fitness < threshold
        cum_r += (-COST_FREEZE) + (-COST_FAILURE if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_fitness=true_fitness,
                h_final=h_final, rounds=t + 1, lead=lead, pi_m=pi_m)


def run_campaign_unified_vague_prior(fitness, models, wt, rng, lam=LAMBDA_VOI, delta=DELTA, t_max=T_MAX_ROUNDS):
    """Prior-matched ablation (c_SREP_18.md) -- see gb1_stagegate.py's
    version for the rationale. model_preds forced to flat 0.0."""
    lead = pick_lead(fitness, wt, rng)
    model_preds = {m: 0.0 for m in MODELS}
    pi_m = [1.0 / len(MODELS)] * len(MODELS)
    obs_history = []
    cum_r = 0.0
    for t in range(t_max):
        h = posterior_mean_h(pi_m, model_preds, obs_history)
        sat = satisficing_check(pi_m, model_preds, h, obs_history, rng, delta=delta)
        voi = voi_estimate(pi_m, model_preds, h, obs_history, rng) if sat else 0.0
        ov = option_value_raw(pi_m, model_preds, h, obs_history, rng)
        use_monitor = (lam * voi) > COST_MONITOR - COST_CHEAP
        sigma = SIGMA_MONITOR if use_monitor else SIGMA_CHEAP
        obs = rng.gauss(fitness[lead], sigma)
        obs_history.append(obs)
        pi_m = update_belief(pi_m, model_preds, obs, sigma)
        cum_r += (GAMMA ** t) * (-(COST_MONITOR if use_monitor else COST_CHEAP))
        if ov < 0.0:
            break
    h_final = posterior_mean_h(pi_m, model_preds, obs_history)
    true_fitness = fitness[lead]
    sat_final = satisficing_check(pi_m, model_preds, h_final, obs_history, rng, delta=delta)
    if sat_final:
        committed = True
        failed = true_fitness < FITNESS_THRESHOLD
        cum_r += (-COST_FREEZE) + (-COST_FAILURE if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_fitness=true_fitness,
                h_final=h_final, rounds=t + 1, lead=lead, pi_m=pi_m)


def run_campaign_gpucb_informed_prior(fitness, models, wt, rng, t_max=T_MAX_ROUNDS):
    """Prior-matched ablation (c_SREP_18.md): GP-UCB's prior mean set to the
    average of the three domain models' predictions for the lead."""
    lead = pick_lead(fitness, wt, rng)
    model_preds = {m: models[m](lead) for m in MODELS}
    if any(p is None for p in model_preds.values()):
        return run_campaign_gpucb_informed_prior(fitness, models, wt, rng, t_max)
    informed_mean = sum(model_preds.values()) / len(model_preds)
    result = _shared_gpucb(
        sample_fn=lambda sigma: rng.gauss(fitness[lead], sigma),
        true_val_fn=lambda: fitness[lead],
        threshold=FITNESS_THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
        prior_sd=TRPB_FITNESS_PRIOR_SD, prior_mean=informed_mean,
    )
    result["lead"] = lead
    result["true_fitness"] = result["true_val"]
    return result


def run_campaign_greedy(fitness, models, wt, rng, t_max=T_MAX_ROUNDS):
    lead = pick_lead(fitness, wt, rng)
    model_preds = {m: models[m](lead) for m in MODELS}
    if any(p is None for p in model_preds.values()):
        return run_campaign_greedy(fitness, models, wt, rng, t_max)
    pi_m = [1.0 / len(MODELS)] * len(MODELS)
    obs_history = []
    cum_r = 0.0
    for t in range(t_max):
        obs = rng.gauss(fitness[lead], SIGMA_CHEAP)
        obs_history.append(obs)
        pi_m = update_belief(pi_m, model_preds, obs, SIGMA_CHEAP)
        h = posterior_mean_h(pi_m, model_preds, obs_history)
        cum_r += (GAMMA ** t) * (-COST_CHEAP)
        if h >= FITNESS_THRESHOLD:
            break
    true_fitness = fitness[lead]
    if h >= FITNESS_THRESHOLD:
        committed = True
        failed = true_fitness < FITNESS_THRESHOLD
        cum_r += (-COST_FREEZE) + (-COST_FAILURE if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_fitness=true_fitness,
                h_final=h, rounds=t + 1, lead=lead, pi_m=pi_m)


def run_campaign_dapp(fitness, models, wt, rng, t_max=T_MAX_ROUNDS):
    lead = pick_lead(fitness, wt, rng)
    result = _shared_dapp(
        sample_fn=lambda sigma: rng.gauss(fitness[lead], sigma),
        true_val_fn=lambda: fitness[lead],
        threshold=FITNESS_THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
    )
    result["lead"] = lead
    result["true_fitness"] = result["true_val"]
    return result


def run_campaign_rdm(fitness, models, wt, rng, t_max=T_MAX_ROUNDS):
    lead = pick_lead(fitness, wt, rng)
    result = _shared_rdm(
        sample_fn=lambda sigma: rng.gauss(fitness[lead], sigma),
        true_val_fn=lambda: fitness[lead],
        threshold=FITNESS_THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
    )
    result["lead"] = lead
    result["true_fitness"] = result["true_val"]
    return result


def run_campaign_gpucb(fitness, models, wt, rng, t_max=T_MAX_ROUNDS):
    lead = pick_lead(fitness, wt, rng)
    result = _shared_gpucb(
        sample_fn=lambda sigma: rng.gauss(fitness[lead], sigma),
        true_val_fn=lambda: fitness[lead],
        threshold=FITNESS_THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
        prior_sd=TRPB_FITNESS_PRIOR_SD,
    )
    result["lead"] = lead
    result["true_fitness"] = result["true_val"]
    return result
