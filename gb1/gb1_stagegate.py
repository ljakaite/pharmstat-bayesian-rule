"""
GB1 restructured as a small-action stage-gate decision (c_SREP_10.md §4,
option 4), replacing the discredited GB1-as-160k-action-space design from
c_SREP_6/8/9.

Grok's critique (c_SREP_10.md), corroborated by c_SREP_9.md's own findings:
Algorithm 1 is a SMALL-ACTION, single-candidate, stopping-time-style
controller (satisficing/VoI/option-value scored once per step against a
reference policy) -- not a per-candidate combinatorial search tool, and the
"epistatic regime" label carried almost no real predictive signal (it's a
static-map property, not a genuine switching latent).

This version fixes both:
  - ONE fixed lead candidate per campaign (chosen once, exogenously -- no
    per-round "propose k candidates and rank them" search loop). The
    landscape is used only to generate noisy assay observations of THAT
    lead and to reveal its true fitness at commit -- the "hidden simulator
    of nature" role Grok's option 4 describes, never the action space.
  - belief pi_m over THREE REAL, PUBLISHED competing models of the same
    static landscape (additive / pairwise / global-nonlinear -- see
    `build_models` below), replacing the discredited additive/epistatic
    regime. No regime (R) dimension: GB1 genuinely has no switching latent,
    so none is invented (c_SREP_10.md's explicit recommendation).

Structurally this is now a Bayesian optimal-stopping problem: assay the same
fixed lead repeatedly (at a chosen precision), maintain a proper posterior
mean/uncertainty over its true fitness, and decide each round whether to
keep assaying or freeze (irreversibly commit) -- which is arguably an even
cleaner match to Algorithm 1's actual character than Biscayne's own richer
regime-switching setup, precisely because GB1 has no action-driven state
dynamics to force in artificially.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stagegate_baselines import (run_campaign_dapp as _shared_dapp, run_campaign_rdm as _shared_rdm,
                                  run_campaign_gpucb as _shared_gpucb)

from gb1_env import load_gb1, hamming, additive_prediction, WT, AA20

MODELS = ["additive", "pairwise", "global_nonlinear"]
GAMMA = 0.95
DELTA = 0.10
LAMBDA_VOI = 1.0
COST_CHEAP = 0.02
COST_MONITOR = 0.05
COST_FREEZE = 0.10
COST_FAILURE = 3.0
FITNESS_THRESHOLD = 1.0
SIGMA_CHEAP = 0.30   # cheap/low-N assay: noisier
SIGMA_MONITOR = 0.10  # confirmatory assay: more precise, costs more
T_MAX_ROUNDS = 30


# ── Build the three competing models (fit once, globally, from real data) ──

def build_pairwise_table(fitness: dict, wt: str = WT):
    """Pairwise-epistasis correction term per position-pair, estimated from
    the real measured double mutants (all HD=2 combinations are present in
    GB1's near-complete data) -- Olson et al.'s "pairwise epistasis
    throughout an entire domain" approach, not a synthetic model."""
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


def pairwise_prediction(variant: str, fitness: dict, pair_table: dict, wt: str = WT):
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


def build_global_nonlinear_link(fitness: dict, wt: str = WT, rng: random.Random = None, n_fit: int = 4000):
    """Otwinowski (2018, PNAS)-style global-epistasis model: fitness as a
    monotonic nonlinear function of the additive latent trait, fit globally
    across the landscape. Uses isotonic regression (a correctly-monotonic,
    non-parametric fit) as a fast, honest stand-in for Otwinowski's own
    spline/sigmoid link -- fit once, from real data, not hand-picked."""
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


def build_models(fitness: dict, rng: random.Random, wt: str = WT):
    """Returns a dict of model_name -> function(variant) -> predicted fitness."""
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


# ── Belief update (pi_m only -- no regime dimension, per c_SREP_10.md) ─────

def _logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))


def update_belief(pi_m, model_preds, obs, sigma, floor=0.01):
    """floor: small pseudo-count so a single unlucky observation can't pin a
    model's weight at exactly 0.0 and make it permanently unrecoverable by
    later evidence -- a fragility noticed in the n=10 pilot (c_SREP_11.md),
    not fixed at the time; fixed here."""
    log_m = [_logpdf(obs, model_preds[m], sigma) for m in MODELS]
    mx = max(log_m)
    pi_m = [p * math.exp(l - mx) for p, l in zip(pi_m, log_m)]
    s = sum(pi_m)
    pi_m = [p / s for p in pi_m]
    pi_m = [(1 - floor) * p + floor / len(pi_m) for p in pi_m]
    return pi_m


def posterior_mean_h(pi_m, model_preds, obs_history):
    """Posterior mean estimate of the lead's true fitness: model-averaged
    prediction, shrunk toward the observed data as evidence accumulates --
    a simple, defensible Bayes-linear combination rather than a running max
    (the bug class in the old GB1-search design, c_SREP_8.md §4)."""
    model_mean = sum(p * model_preds[m] for p, m in zip(pi_m, MODELS))
    if not obs_history:
        return model_mean
    obs_mean = sum(obs_history) / len(obs_history)
    n = len(obs_history)
    w = n / (n + 2.0)  # more observations -> trust the data more than the model prior
    return w * obs_mean + (1 - w) * model_mean


# ── Satisficing / VoI / option value ────────────────────────────────────────

def _shrink_toward_evidence(prior_val, obs_history):
    """Same w=n/(n+2) shrinkage already used by posterior_mean_h, applied to
    a single model's prior prediction instead of the pi_m-weighted average.
    Fix for the bug found in c_SREP_18.md's prior-matched ablation: without
    this, satisficing_check sampled around the static `model_preds[m]`
    forever -- observations only narrowed the sampling WIDTH (via `sd`
    shrinking with n), never corrected the sampling LOCATION, so a flat or
    wrong prior could never be overridden by real evidence
    (`unified_vague_prior` got exactly 0/1,000 commits on all three
    environments before this fix, c_SREP_18.md §1)."""
    if not obs_history:
        return prior_val
    obs_mean = sum(obs_history) / len(obs_history)
    n = len(obs_history)
    w = n / (n + 2.0)
    return w * obs_mean + (1 - w) * prior_val


def satisficing_check(pi_m, model_preds, h, obs_history, rng, delta=DELTA, n_samples=200,
                       sigma_cheap=SIGMA_CHEAP, threshold=FITNESS_THRESHOLD):
    """P(true fitness >= threshold), approximated by treating each model's
    own EVIDENCE-SHRUNK prediction (weighted by pi_m) as a plausible
    true-fitness value, perturbed by current posterior uncertainty -- shrinks
    as obs_history grows, exactly like Biscayne's satisficing rollout uses
    the assumed physical model rather than real future data. Prior to
    c_SREP_19.md's fix, this sampled around the raw (never-updated)
    model_preds[m] instead of `_shrink_toward_evidence(model_preds[m], ...)`
    -- see that function's docstring for the bug this corrects.

    `sigma_cheap`/`threshold` default to the module constants (unchanged
    behaviour for every existing caller) but are now overridable -- added in
    c_SREP_22.md to sweep noise and threshold height the same way c_SREP_21.md
    already swept delta."""
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
    """Non-myopic VoI: expected reduction in posterior spread (a proxy for
    decision-relevant value) from one more cheap assay."""
    n = len(obs_history)
    sd_now = sigma_cheap / math.sqrt(n + 1)
    sd_after = sigma_cheap / math.sqrt(n + 2)
    # Value = reduction in the probability mass that would flip the
    # satisficing decision -- approximated as reduction in posterior sd,
    # scaled by how close h currently sits to the threshold (VoI is highest
    # right at the decision boundary, ~0 far from it).
    closeness = math.exp(-((h - threshold) ** 2) / (2 * sd_now ** 2 + 1e-9))
    return max(0.0, (sd_now - sd_after) * closeness * 5.0)


def option_value_raw(pi_m, model_preds, h, obs_history, rng, T_horizon=6,
                      assay_cost_rate=COST_CHEAP, sigma_cheap=SIGMA_CHEAP,
                      threshold=FITNESS_THRESHOLD):
    """Positive -> keep assaying preferred; negative -> freeze now.
    v_freeze: value of locking in the current posterior mean now.
    v_wait: expected value after T_horizon more cheap assays -- posterior
    mean barely moves once n is large (diminishing returns, correctly
    captured since sd shrinks as 1/sqrt(n)), net of the ongoing assay cost."""
    v_freeze = h - COST_FREEZE
    n = len(obs_history)
    sd_future = sigma_cheap / math.sqrt(n + T_horizon + 1)
    sd_now = sigma_cheap / math.sqrt(n + 1)
    # Expected posterior mean doesn't drift (unbiased), but its RELIABILITY
    # improves -- value that improvement only insofar as it could still flip
    # a near-threshold decision; approximate via the same closeness weight.
    closeness = math.exp(-((h - threshold) ** 2) / (2 * sd_now ** 2 + 1e-9))
    reliability_gain = (sd_now - sd_future) * closeness * 2.0
    v_wait = h + reliability_gain - assay_cost_rate * T_horizon - COST_FREEZE
    return v_wait - v_freeze


# ── Campaign runners ─────────────────────────────────────────────────────────



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
# 0.69; global_nonlinear: 0.22).
MIN_HD = 1


def use_honest_models():
    global MODELS, MIN_HD
    MODELS = ["additive", "global_nonlinear"]
    MIN_HD = 2


def pick_lead(fitness, rng, min_hd=None, max_hd=2, single_mutant_bar=0.5):
    """A real stage-gate campaign doesn't start from a uniformly random
    mutant -- it starts from a lead that already survived some upstream
    screen. First pilot run (c_SREP_11.md) picked leads uniformly at random
    and found only ~10% clear the WT threshold at all, regardless of any
    downstream decision quality -- a lead-SELECTION problem masquerading as
    a lead-EVALUATION one. Fixed here with a legitimate, non-circular
    pre-filter: require each of the lead's OWN constituent single mutants
    (already-known data, not the combined fitness being decided) to
    individually clear a modest bar -- exactly the kind of prior information
    a real campaign would have on hand before proposing a combined lead."""
    min_hd = MIN_HD if min_hd is None else min_hd
    key = (id(fitness), min_hd, max_hd, single_mutant_bar)
    cached = _LEAD_POOL_CACHE.get(key)
    if cached is None:
        # The stored tuple keeps a reference to `fitness`, so its id() cannot be
        # recycled by the garbage collector while this entry is live -- without
        # that, a freed dict's address could be reused and collide on the key.
        cached = (*_build_lead_pool(fitness, min_hd, max_hd, single_mutant_bar), fitness)
        _LEAD_POOL_CACHE[key] = cached
    promising, candidates, _ = cached
    return rng.choice(promising) if promising else rng.choice(candidates)


_LEAD_POOL_CACHE = {}


def _build_lead_pool(fitness, min_hd, max_hd, single_mutant_bar):
    """The candidate pool depends only on (fitness, min_hd, max_hd, bar) -- it
    is identical for every campaign, but was rebuilt on every call, rescanning
    the whole landscape each time (c_protein_design_7.md: 99.6% of GB1 runtime).
    Memoised here. `rng.choice` still draws from the same list in the same
    order, so leads, and therefore every reported result, are unchanged."""

    candidates = [v for v in fitness if min_hd <= hamming(v, WT) <= max_hd]
    promising = []
    for v in candidates:
        ok = True
        for i in range(4):
            if v[i] == WT[i]:
                continue
            single = WT[:i] + v[i] + WT[i + 1:]
            if fitness.get(single, 0.0) < single_mutant_bar:
                ok = False
                break
        if ok:
            promising.append(v)
    return promising, candidates


def run_campaign_unified(fitness, models, rng, lam=LAMBDA_VOI, delta=DELTA, t_max=T_MAX_ROUNDS,
                          sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
                          threshold=FITNESS_THRESHOLD, eps_abandon=None):
    """sigma_cheap/sigma_monitor/threshold default to the module constants
    (unchanged behaviour for every existing caller) but are now overridable
    -- added in c_SREP_22.md to sweep noise and threshold height the same
    way c_SREP_21.md already swept delta."""
    lead = pick_lead(fitness, rng)
    model_preds = {m: models[m](lead) for m in MODELS}
    if any(p is None for p in model_preds.values()):
        return run_campaign_unified(fitness, models, rng, lam, delta, t_max,
                                     sigma_cheap, sigma_monitor, threshold, eps_abandon)  # retry on edge case
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
    # ABANDON vs FREEZE (c_SREP_13.md): stopping (ov<0 / t_max) only means
    # "ready to decide" -- it must not force a commit. A prior version
    # unconditionally froze here regardless of what the belief said, which
    # made failure rate structurally identical across every method (the
    # n=500 pilot's 0-discordant-pairs McNemar result caught this). Gate the
    # actual commit on the FINAL satisficing check: if it still passes,
    # freeze (pay the commit cost, take the failure penalty if wrong); if
    # not, abandon (walk away, only the assay costs already spent are sunk).
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


def run_campaign_greedy(fitness, models, rng, t_max=T_MAX_ROUNDS):
    """BDT-analogue: myopic -- freeze as soon as the model-averaged posterior
    mean first clears the threshold, no satisficing/VoI/option-value check."""
    lead = pick_lead(fitness, rng)
    model_preds = {m: models[m](lead) for m in MODELS}
    if any(p is None for p in model_preds.values()):
        return run_campaign_greedy(fitness, models, rng, t_max)
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
    # Same abandon-vs-freeze fix as run_campaign_unified (c_SREP_13.md): a
    # myopic rule that never saw its own posterior mean clear the bar
    # (t_max exhausted) shouldn't be forced to commit anyway.
    if h >= FITNESS_THRESHOLD:
        committed = True
        failed = true_fitness < FITNESS_THRESHOLD
        cum_r += (-COST_FREEZE) + (-COST_FAILURE if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_fitness=true_fitness,
                h_final=h, rounds=t + 1, lead=lead, pi_m=pi_m)


def run_campaign_unified_vague_prior(fitness, models, rng, lam=LAMBDA_VOI, delta=DELTA, t_max=T_MAX_ROUNDS):
    """Prior-matched ablation (c_SREP_18.md): same satisficing/VoI/option-
    value decision rule as run_campaign_unified, but model_preds is
    overridden to a flat 0.0 for all three models -- the SAME uninformative
    prior GP-UCB's default uses -- instead of the fitted domain models.
    Isolates whether unified's advantage is the decision rule itself or just
    a richer starting belief. Duplicates run_campaign_unified's loop rather
    than parametrising it, to avoid any risk of changing the already-
    validated, already-reported original function's behaviour."""
    lead = pick_lead(fitness, rng)
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


def run_campaign_gpucb_informed_prior(fitness, models, rng, t_max=T_MAX_ROUNDS):
    """Prior-matched ablation (c_SREP_18.md): same GP-UCB confidence-bound
    stopping rule, but its prior mean is set to the average of the three
    domain models' predictions for the lead -- the SAME information
    unified's belief starts from -- instead of the flat 0.0 default."""
    lead = pick_lead(fitness, rng)
    model_preds = {m: models[m](lead) for m in MODELS}
    if any(p is None for p in model_preds.values()):
        return run_campaign_gpucb_informed_prior(fitness, models, rng, t_max)
    informed_mean = sum(model_preds.values()) / len(model_preds)
    result = _shared_gpucb(
        sample_fn=lambda sigma: rng.gauss(fitness[lead], sigma),
        true_val_fn=lambda: fitness[lead],
        threshold=FITNESS_THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
        prior_sd=GB1_FITNESS_PRIOR_SD, prior_mean=informed_mean,
    )
    result["lead"] = lead
    result["true_fitness"] = result["true_val"]
    return result


def run_campaign_dapp(fitness, models, rng, t_max=T_MAX_ROUNDS):
    """DAPP-analogue, ported from run_strategy_dapp via the shared
    stagegate_baselines module (c_SREP_12.md) -- raw noisy observation,
    consecutive-trigger, no belief updating."""
    lead = pick_lead(fitness, rng)
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


GB1_FITNESS_PRIOR_SD = 0.3953899564571686  # population sd, computed once (c_SREP_14.md)


def run_campaign_gpucb(fitness, models, rng, t_max=T_MAX_ROUNDS):
    """GP-UCB-analogue -- see stagegate_baselines.run_campaign_gpucb's
    docstring for the honest adaptation this required (candidate-selection
    -> confidence-bound stopping, once GB1 stopped being a search problem)."""
    lead = pick_lead(fitness, rng)
    result = _shared_gpucb(
        sample_fn=lambda sigma: rng.gauss(fitness[lead], sigma),
        true_val_fn=lambda: fitness[lead],
        threshold=FITNESS_THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
        prior_sd=GB1_FITNESS_PRIOR_SD,
    )
    result["lead"] = lead
    result["true_fitness"] = result["true_val"]
    return result


def run_campaign_rdm(fitness, models, rng, t_max=T_MAX_ROUNDS):
    """RDM-analogue, ported via the shared stagegate_baselines module --
    running-mean state trigger, no belief updating."""
    lead = pick_lead(fitness, rng)
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
