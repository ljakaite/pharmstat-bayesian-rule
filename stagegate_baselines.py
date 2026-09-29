"""
Shared DAPP-analogue / RDM-analogue baselines for the single-lead
optimal-stopping stage-gate structure now used by BOTH
`gb1/gb1_stagegate.py` and `developability/dev_pomdp.py` (c_SREP_10/11.md).

Written once, generically, and wired into both environments via thin
adapters -- the two environments share the exact same problem shape (fix one
lead, repeatedly assay it, decide when to freeze), so there is no reason to
duplicate this logic per-environment the way the old, now-superseded
`gb1_pomdp.py` did.

Ported from the real `run_strategy_dapp` / `run_strategy_rdm_conservative`
in `biscayne-pomdp-ems/biscayne/biscayne_main.py` (read directly, not
reconstructed from memory -- see c_SREP_9.md §2 for the first porting pass,
onto the now-discredited combinatorial-search design). Same defining
features preserved here:

  DAPP  -- triggers on the RAW noisy per-round observation (not a Bayesian
           posterior) crossing thresholds for `consec_needed` consecutive
           rounds. No belief updating over models at all.
  RDM   -- triggers on a plain running MEAN of observations (denoised by
           averaging, but NOT model-informed the way the unified/greedy
           posterior mean is) against static pre-committed thresholds. Also
           no belief updating over models.

One asymmetry from Biscayne, carried over unchanged from the c_SREP_9.md
port and still correct here: Biscayne's barrier is a DEFENSIVE response to
BAD news (risk of failure). Here, freezing is a commitment to a GOOD
candidate. So both baselines' "freeze" trigger fires on sustained HIGH
signal (as in the original c_SREP_9.md port), not low -- the domain's
success/failure polarity is inverted relative to Biscayne's, not the
baselines' logic.

A third baseline, `run_campaign_gpucb`, was added later (c_SREP_14.md) for
the ALDE/MLDE-field comparison flagged as missing back in c_SREP_7.md §3 --
see its own docstring for why it had to be genuinely adapted, not directly
ported, once GB1 stopped being a per-candidate search problem (c_SREP_10.md).
"""


def run_campaign_dapp(sample_fn, true_val_fn, threshold, cost_cheap, cost_monitor,
                       cost_freeze, cost_failure, gamma, rng, t_max=30,
                       sigma_cheap=None, sigma_monitor=None,
                       consec_needed=2, signal_margin=0.5):
    """sample_fn(sigma) -> noisy observation of the fixed lead's true value.
    true_val_fn() -> the lead's true value (called once, at freeze).
    signal_margin: how far above/below `threshold` (in units of sigma_cheap)
    counts as a confidently-high / confidently-low raw reading."""
    signal_high = threshold + signal_margin * sigma_cheap
    signal_low = threshold - signal_margin * sigma_cheap
    use_monitor = True  # DAPP's default "MEDIUM"-equivalent: resourced until told otherwise
    consec_high = consec_low = 0
    cum_r = 0.0
    triggered = False
    for t in range(t_max):
        sigma = sigma_monitor if use_monitor else sigma_cheap
        obs = sample_fn(sigma)
        consec_high = consec_high + 1 if obs > signal_high else 0
        consec_low = consec_low + 1 if obs < signal_low else 0
        if use_monitor and consec_low >= consec_needed:
            use_monitor = False  # de-escalate precision spend once it looks like a loser
        cum_r += (gamma ** t) * (-(cost_monitor if use_monitor else cost_cheap))
        if consec_high >= consec_needed:
            triggered = True
            break
    # ABANDON vs FREEZE (c_SREP_13.md, same fix as gb1_stagegate.py): only a
    # genuine consec_high trigger commits. Running out of rounds without ever
    # seeing sustained high signal means DAPP never got confident -- it
    # should walk away, not be forced to freeze on a signal it never trusted.
    true_val = true_val_fn()
    if triggered:
        committed = True
        failed = true_val < threshold
        cum_r += (-cost_freeze) + (-cost_failure if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_val=true_val, rounds=t + 1)


def run_campaign_rdm(sample_fn, true_val_fn, threshold, cost_cheap, cost_monitor,
                      cost_freeze, cost_failure, gamma, rng, t_max=30,
                      sigma_cheap=None, sigma_monitor=None, signal_margin=0.5):
    """Same shape as DAPP but triggers on a running MEAN of observations
    (denoised, non-Bayesian) rather than the single latest raw reading --
    the real distinction Biscayne's RDM has from its DAPP."""
    signal_high = threshold + signal_margin * sigma_cheap
    signal_low = threshold - signal_margin * sigma_cheap
    use_monitor = True
    obs_history = []
    cum_r = 0.0
    triggered = False
    for t in range(t_max):
        sigma = sigma_monitor if use_monitor else sigma_cheap
        obs = sample_fn(sigma)
        obs_history.append(obs)
        running_mean = sum(obs_history) / len(obs_history)
        if use_monitor and running_mean < signal_low:
            use_monitor = False
        cum_r += (gamma ** t) * (-(cost_monitor if use_monitor else cost_cheap))
        if running_mean > signal_high:
            triggered = True
            break
    # Same abandon-vs-freeze fix as DAPP above.
    true_val = true_val_fn()
    if triggered:
        committed = True
        failed = true_val < threshold
        cum_r += (-cost_freeze) + (-cost_failure if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_val=true_val, rounds=t + 1)


def run_campaign_gpucb(sample_fn, true_val_fn, threshold, cost_cheap, cost_monitor,
                        cost_freeze, cost_failure, gamma, rng, t_max=30,
                        sigma_cheap=None, sigma_monitor=None, prior_sd=1.0, beta=1.5,
                        prior_mean=0.0):
    """GP-UCB-analogue -- the field's dominant tool per c_SREP_4.md §1b /
    c_SREP_7.md §3 (ALDE and the wider MLDE literature use GP-UCB/Thompson
    sampling as their standard acquisition function), added here for the
    comparison c_SREP_7.md §3 flagged as needed and never built.

    Honest adaptation, not a direct port -- flagged plainly: GP-UCB's
    textbook role is CANDIDATE SELECTION over a large search space (which of
    many sequences to test next). Since c_SREP_10.md's correction, this
    design has no candidate space to search -- one lead is fixed per
    campaign, matching Algorithm 1's actual small-action character. So
    GP-UCB's role here is adapted from candidate-selection to a genuine
    CONFIDENCE-BOUND STOPPING rule -- the same posterior-mean-plus-width
    machinery, applied to "is it worth continuing to assay this one lead"
    rather than "which lead should I assay next". This is a real, standard
    use of the same bandit-theory tool (the lower-confidence-bound safety
    criterion used in safe/constrained Bayesian optimisation), not a
    different algorithm wearing GP-UCB's name.

    Maintains a genuine Bayesian posterior over the lead's true value via
    exact Gaussian conjugate updating (the single-point degenerate case of a
    GP posterior) -- UNLIKE unified/greedy, it uses an UNINFORMATIVE prior
    (mean 0, width `prior_sd`), matching how GP-UCB is actually used in the
    field: it does not have access to the competing-model domain knowledge
    unified's panel/model beliefs are built on. That is a real, fair point
    of difference from unified, not a handicap introduced to make GP-UCB
    look worse -- it is what makes GP-UCB "field-standard" rather than
    "informed by this specific paper's domain models".

    Commits once the LOWER confidence bound clears the threshold (confident
    it's good); abandons early once the UPPER bound falls below threshold
    (confident further assay can't save it, no point spending more).

    `prior_mean` defaults to 0.0 (the genuinely uninformative field-standard
    setting used everywhere above). c_SREP_18.md's prior-matched ablation
    passes a non-zero value here (the three-model mixture average) to give
    GP-UCB the SAME domain information unified's belief starts from, testing
    whether unified's advantage is the decision rule or just a richer
    prior -- see that doc for the actual ablation and its result."""
    post_mean = prior_mean
    post_var = prior_sd ** 2
    cum_r = 0.0
    triggered_commit = triggered_abandon = False
    for t in range(t_max):
        sd = post_var ** 0.5
        use_monitor = sd > sigma_monitor  # spend on precision while still uncertain
        sigma = sigma_monitor if use_monitor else sigma_cheap
        obs = sample_fn(sigma)
        obs_var = sigma ** 2
        post_var_new = 1.0 / (1.0 / post_var + 1.0 / obs_var)
        post_mean = post_var_new * (post_mean / post_var + obs / obs_var)
        post_var = post_var_new
        cum_r += (gamma ** t) * (-(cost_monitor if use_monitor else cost_cheap))
        sd = post_var ** 0.5
        lcb, ucb = post_mean - beta * sd, post_mean + beta * sd
        if lcb > threshold:
            triggered_commit = True
            break
        if ucb < threshold:
            triggered_abandon = True
            break
    true_val = true_val_fn()
    if triggered_commit:
        committed = True
        failed = true_val < threshold
        cum_r += (-cost_freeze) + (-cost_failure if failed else 0.0)
    else:
        committed = False
        failed = None
    return dict(cum_reward=cum_r, failed=failed, committed=committed, true_val=true_val, rounds=t + 1)
