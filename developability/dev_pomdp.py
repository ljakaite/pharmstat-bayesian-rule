"""
POMDP layer for the developability stage-gate (c_SREP_11.md §3/§4), structured
identically to gb1_stagegate.py's proven design (one fixed lead per campaign,
belief over competing models, no invented regime, posterior-mean state,
satisficing/VoI/option-value stopping decision) -- see that module's
docstring for the full rationale. Here the lead is one of 137 REAL
clinical-stage antibodies (Jain et al. 2017) and the "models" are three real
fitted assay-subset predictors (`dev_env.PANELS`) of a real, checkable
composite developability score, instead of GB1's additive/pairwise/
global-nonlinear fitness predictors.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stagegate_baselines import (run_campaign_dapp as _shared_dapp, run_campaign_rdm as _shared_rdm,
                                  run_campaign_gpucb as _shared_gpucb)

from dev_env import PANELS, load_assays, load_clinical_status, compute_composite, fit_panel_models, panel_predict

MODELS = list(PANELS.keys())
GAMMA = 0.95
DELTA = 0.10
LAMBDA_VOI = 1.0
COST_CHEAP = 0.02
COST_MONITOR = 0.05
COST_FREEZE = 0.10
COST_FAILURE = 3.0
THRESHOLD = 0.0          # "success" = at least population-average developability composite
SIGMA_CHEAP = 0.30
SIGMA_MONITOR = 0.10
T_MAX_ROUNDS = 30


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
                       sigma_cheap=SIGMA_CHEAP, threshold=THRESHOLD):
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
                    sigma_cheap=SIGMA_CHEAP, threshold=THRESHOLD):
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
                  sigma_cheap=SIGMA_CHEAP, threshold=THRESHOLD):
    n = len(obs_history)
    sd_now = sigma_cheap / math.sqrt(n + 1)
    sd_after = sigma_cheap / math.sqrt(n + 2)
    closeness = math.exp(-((h - threshold) ** 2) / (2 * sd_now ** 2 + 1e-9))
    return max(0.0, (sd_now - sd_after) * closeness * 5.0)


def option_value_raw(pi_m, model_preds, h, obs_history, rng, T_horizon=6, assay_cost_rate=COST_CHEAP,
                      sigma_cheap=SIGMA_CHEAP, threshold=THRESHOLD):
    v_freeze = h - COST_FREEZE
    n = len(obs_history)
    sd_now = sigma_cheap / math.sqrt(n + 1)
    sd_future = sigma_cheap / math.sqrt(n + T_horizon + 1)
    closeness = math.exp(-((h - threshold) ** 2) / (2 * sd_now ** 2 + 1e-9))
    reliability_gain = (sd_now - sd_future) * closeness * 2.0
    v_wait = h + reliability_gain - assay_cost_rate * T_horizon - COST_FREEZE
    return v_wait - v_freeze


def run_campaign_unified(names, composite, models, data, rng, lam=LAMBDA_VOI, delta=DELTA, t_max=T_MAX_ROUNDS,
                          sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR, threshold=THRESHOLD, eps_abandon=None):
    """sigma_cheap/sigma_monitor/threshold default to the module constants
    (unchanged behaviour for every existing caller) but are now overridable
    -- added in c_SREP_22.md, same as gb1_stagegate.py."""
    lead = rng.choice(names)
    model_preds = panel_predict(models, lead, data)
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
        obs = rng.gauss(composite[lead], sigma)
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
    true_val = composite[lead]
    # ABANDON vs FREEZE (c_SREP_13.md) -- same fix as gb1_stagegate.py: don't
    # force a commit just because the stopping condition (ov<0/t_max) fired;
    # gate the actual freeze on the final satisficing check.
    sat_final = satisficing_check(pi_m, model_preds, h_final, obs_history, rng, delta=delta,
                                   sigma_cheap=sigma_cheap, threshold=threshold)
    if sat_final:
        committed = True
        failed = true_val < threshold
        cum_r += (-COST_FREEZE) + (-COST_FAILURE if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_val=true_val,
                h_final=h_final, rounds=t + 1, lead=lead, pi_m=pi_m)


def run_campaign_unified_vague_prior(names, composite, models, data, rng, lam=LAMBDA_VOI,
                                      delta=DELTA, t_max=T_MAX_ROUNDS):
    """Prior-matched ablation (c_SREP_18.md) -- see gb1_stagegate.py's
    version for the full rationale. model_preds forced to flat 0.0 for all
    three panels instead of the fitted panel predictions."""
    lead = rng.choice(names)
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
        obs = rng.gauss(composite[lead], sigma)
        obs_history.append(obs)
        pi_m = update_belief(pi_m, model_preds, obs, sigma)
        cum_r += (GAMMA ** t) * (-(COST_MONITOR if use_monitor else COST_CHEAP))
        if ov < 0.0:
            break
    h_final = posterior_mean_h(pi_m, model_preds, obs_history)
    true_val = composite[lead]
    sat_final = satisficing_check(pi_m, model_preds, h_final, obs_history, rng, delta=delta)
    if sat_final:
        committed = True
        failed = true_val < THRESHOLD
        cum_r += (-COST_FREEZE) + (-COST_FAILURE if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_val=true_val,
                h_final=h_final, rounds=t + 1, lead=lead, pi_m=pi_m)


def run_campaign_gpucb_informed_prior(names, composite, models, data, rng, t_max=T_MAX_ROUNDS):
    """Prior-matched ablation (c_SREP_18.md): GP-UCB's prior mean set to the
    average of the three panel models' predictions for the lead."""
    lead = rng.choice(names)
    model_preds = panel_predict(models, lead, data)
    informed_mean = sum(model_preds.values()) / len(model_preds)
    result = _shared_gpucb(
        sample_fn=lambda sigma: rng.gauss(composite[lead], sigma),
        true_val_fn=lambda: composite[lead],
        threshold=THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
        prior_sd=DEV_COMPOSITE_PRIOR_SD, prior_mean=informed_mean,
    )
    result["lead"] = lead
    return result


def run_campaign_greedy(names, composite, models, data, rng, t_max=T_MAX_ROUNDS):
    lead = rng.choice(names)
    model_preds = panel_predict(models, lead, data)
    pi_m = [1.0 / len(MODELS)] * len(MODELS)
    obs_history = []
    cum_r = 0.0
    for t in range(t_max):
        obs = rng.gauss(composite[lead], SIGMA_CHEAP)
        obs_history.append(obs)
        pi_m = update_belief(pi_m, model_preds, obs, SIGMA_CHEAP)
        h = posterior_mean_h(pi_m, model_preds, obs_history)
        cum_r += (GAMMA ** t) * (-COST_CHEAP)
        if h >= THRESHOLD:
            break
    true_val = composite[lead]
    if h >= THRESHOLD:
        committed = True
        failed = true_val < THRESHOLD
        cum_r += (-COST_FREEZE) + (-COST_FAILURE if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_val=true_val,
                h_final=h, rounds=t + 1, lead=lead, pi_m=pi_m)


def run_campaign_dapp(names, composite, models, data, rng, t_max=T_MAX_ROUNDS):
    lead = rng.choice(names)
    result = _shared_dapp(
        sample_fn=lambda sigma: rng.gauss(composite[lead], sigma),
        true_val_fn=lambda: composite[lead],
        threshold=THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
    )
    result["lead"] = lead
    return result


def run_campaign_rdm(names, composite, models, data, rng, t_max=T_MAX_ROUNDS):
    lead = rng.choice(names)
    result = _shared_rdm(
        sample_fn=lambda sigma: rng.gauss(composite[lead], sigma),
        true_val_fn=lambda: composite[lead],
        threshold=THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
    )
    result["lead"] = lead
    return result


DEV_COMPOSITE_PRIOR_SD = 0.5319984386485853  # population sd, computed once (c_SREP_14.md)


def run_campaign_gpucb(names, composite, models, data, rng, t_max=T_MAX_ROUNDS):
    """GP-UCB-analogue -- see stagegate_baselines.run_campaign_gpucb's
    docstring for the honest adaptation this required."""
    lead = rng.choice(names)
    result = _shared_gpucb(
        sample_fn=lambda sigma: rng.gauss(composite[lead], sigma),
        true_val_fn=lambda: composite[lead],
        threshold=THRESHOLD, cost_cheap=COST_CHEAP, cost_monitor=COST_MONITOR,
        cost_freeze=COST_FREEZE, cost_failure=COST_FAILURE, gamma=GAMMA, rng=rng,
        t_max=t_max, sigma_cheap=SIGMA_CHEAP, sigma_monitor=SIGMA_MONITOR,
        prior_sd=DEV_COMPOSITE_PRIOR_SD,
    )
    result["lead"] = lead
    return result
