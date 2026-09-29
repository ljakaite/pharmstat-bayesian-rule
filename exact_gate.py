"""
Exact-belief stage-gate policy ("unified-exact"), c_protein_design_4.md.

Motivation (c_protein_design_3.md §6.6): the published unified policy keeps a
heuristic belief -- satisficing_check samples around each model's prediction
shrunk toward the data by a fixed w = n/(n+2) weight, and option_value_raw
stops on a closeness heuristic. Diagnostics showed most of its remaining
good-lead losses are leads whose fitted-model prior sits on the wrong side of
the bar: the update is too slow to overturn a confidently wrong model. The
same informed prior handed to an exactly-updated confidence-bound gate gave
much higher retention, so the defect is the update/stopping rule, not the
domain models.

This module keeps the architecture (belief over competing models M, an
explicit failure budget, value-of-information precision choice, option-value
timing) and replaces the two heuristics with their exact counterparts:

  BELIEF   a proper mixture-of-Gaussians posterior over the latent quality h.
           Component m is N(mu_m, s_m^2), started at the model's own
           prediction with width tau_m = that model's measured held-out error
           (so a model known to be inaccurate starts vague, and evidence can
           overturn it). Each observation updates every component in closed
           form and reweights the mixture by its marginal likelihood.

  p        P(h >= threshold) = sum_m pi_m * Phi((mu_m - threshold)/s_m)
           -- exact under the mixture, replacing the Monte-Carlo satisficing
           check. The delta guarantee is unchanged in meaning: commit only if
           p >= 1 - delta.

  STOPPING one-step look-ahead on the actual decision. Deciding now is worth
           max(commit, kill); one more assay is worth the expectation of that
           same quantity over the predictive distribution of the next
           observation, minus its cost. Stop when the look-ahead gain is not
           worth the assay. This is the option value the heuristic was
           approximating, computed rather than approximated.

  PRECISION the same look-ahead evaluated for the cheap and the confirmatory
           assay; buy precision only when its extra information pays for its
           extra cost. This is the VoI step, likewise computed.

Gate outcome: commit iff p >= 1 - delta at the stopping point, else abandon.
An optional eps_abandon adds the c_protein_design_3.md kill boundary (stop
early once p <= eps); the look-ahead already stops on economics, so it is off
by default.

Costs, noise, threshold and the lead-selection rule are supplied by the
caller, so this runs on the same campaigns (same seeds -> same leads) as
every policy already reported.
"""
import math

SQRT2 = math.sqrt(2.0)
# Gauss-Hermite nodes/weights (7-point) for the predictive expectation.
_GH_X = [-2.651961356835233, -1.673551628767471, -0.8162878828589647, 0.0,
         0.8162878828589647, 1.673551628767471, 2.651961356835233]
_GH_W = [0.0009717812450995192, 0.05451558281912703, 0.4256072526101278,
         0.8102646175568073, 0.4256072526101278, 0.05451558281912703,
         0.0009717812450995192]
_GH_NORM = math.sqrt(math.pi)


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def p_above(weights, means, sds, threshold):
    """Exact P(h >= threshold) under the Gaussian mixture."""
    return sum(w * (1.0 - _phi((threshold - mu) / sd))
               for w, mu, sd in zip(weights, means, sds))


def _update(weights, means, sds, obs, sigma):
    """Closed-form conjugate update of every mixture component, plus the
    mixture reweighting by each component's marginal likelihood of `obs`."""
    ov = sigma * sigma
    new_w, new_mu, new_sd, logs = [], [], [], []
    for w, mu, sd in zip(weights, means, sds):
        pv = sd * sd
        post_v = 1.0 / (1.0 / pv + 1.0 / ov)
        post_m = post_v * (mu / pv + obs / ov)
        marg_v = pv + ov  # predictive variance of obs under this component
        logs.append(math.log(max(w, 1e-300)) - 0.5 * math.log(2 * math.pi * marg_v)
                    - 0.5 * (obs - mu) ** 2 / marg_v)
        new_mu.append(post_m)
        new_sd.append(math.sqrt(post_v))
    mx = max(logs)
    raw = [math.exp(l - mx) for l in logs]
    s = sum(raw)
    new_w = [r / s for r in raw]
    # Same small floor as the published update_belief: no component's weight
    # may be pinned at exactly 0 and become unrecoverable (c_SREP_11.md).
    k = len(new_w)
    floor = 0.01
    new_w = [(1 - floor) * w + floor / k for w in new_w]
    return new_w, new_mu, new_sd


def _decide_value(p, value_commit, cost_failure, cost_freeze):
    """Value of deciding now: commit (expected reward net of failure risk and
    the commit cost) or abandon (0). Abandoning is always available, so the
    value of a decision is the better of the two."""
    commit = p * value_commit - (1.0 - p) * cost_failure - cost_freeze
    return max(commit, 0.0)


def _lookahead_value(weights, means, sds, sigma, threshold, value_commit,
                     cost_failure, cost_freeze):
    """E_y[ max(commit, abandon) ] after one more assay of precision sigma,
    integrated over the predictive distribution of y with Gauss-Hermite
    quadrature per mixture component."""
    total = 0.0
    for w, mu, sd in zip(weights, means, sds):
        pred_sd = math.sqrt(sd * sd + sigma * sigma)
        for x, gw in zip(_GH_X, _GH_W):
            y = mu + SQRT2 * pred_sd * x
            w2, m2, s2 = _update(weights, means, sds, y, sigma)
            p2 = p_above(w2, m2, s2, threshold)
            total += w * (gw / _GH_NORM) * _decide_value(p2, value_commit, cost_failure, cost_freeze)
    return total


def run_campaign_exact(model_preds, model_sds, sample_fn, true_val_fn, threshold,
                       cost_cheap, cost_monitor, cost_freeze, cost_failure, gamma,
                       t_max=30, sigma_cheap=None, sigma_monitor=None, delta=0.10,
                       value_commit=1.0, eps_abandon=None, lookahead=True):
    """model_preds/model_sds: per-model prior mean and prior sd (the model's
    measured held-out error) for THIS lead. sample_fn(sigma) -> one noisy
    assay of the lead; true_val_fn() -> its true value, called once at the
    end. Returns the same result dict shape as every other policy."""
    k = len(model_preds)
    weights = [1.0 / k] * k
    means = list(model_preds)
    sds = list(model_sds)
    cum_r = 0.0
    t = 0

    for t in range(t_max):
        p = p_above(weights, means, sds, threshold)
        if p >= 1.0 - delta:
            break                      # confident enough to commit
        if eps_abandon is not None and p <= eps_abandon:
            break                      # confident enough to kill
        if lookahead:
            v_now = _decide_value(p, value_commit, cost_failure, cost_freeze)
            best_sigma, best_gain = None, 0.0
            for sigma, cost in ((sigma_cheap, cost_cheap), (sigma_monitor, cost_monitor)):
                gain = _lookahead_value(weights, means, sds, sigma, threshold, value_commit,
                                        cost_failure, cost_freeze) - v_now - cost
                if gain > best_gain:
                    best_sigma, best_gain = sigma, gain
            if best_sigma is None:
                break                  # no assay pays for itself: decide now
            sigma = best_sigma
        else:
            # Ablation: no look-ahead, always the cheap assay until the gate
            # resolves or the horizon runs out.
            sigma = sigma_cheap
        obs = sample_fn(sigma)
        weights, means, sds = _update(weights, means, sds, obs, sigma)
        cum_r += (gamma ** t) * (-(cost_monitor if sigma == sigma_monitor else cost_cheap))

    p_final = p_above(weights, means, sds, threshold)
    true_val = true_val_fn()
    if p_final >= 1.0 - delta:
        committed = True
        failed = true_val < threshold
        cum_r += (-cost_freeze) + (-cost_failure if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_val=true_val,
                p_final=p_final, rounds=t + 1, pi_m=weights)


def model_error_sds(preds_and_truths, floor=1e-3):
    """Per-model prior width: the RMSE of that model's predictions against
    measured values, on a sample of the same data the models were fitted to
    describe. Returns one sd per model, in model order."""
    out = []
    for series in preds_and_truths:
        n = len(series)
        if n == 0:
            out.append(floor)
            continue
        mse = sum((p - y) ** 2 for p, y in series) / n
        out.append(max(math.sqrt(mse), floor))
    return out
