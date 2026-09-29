"""
Paired statistical comparison for the stage-gate pilots (c_SREP_13.md).

Campaigns are properly paired across methods by construction: every campaign
runner's first RNG draw is the lead selection, and each seed index i gets a
freshly-seeded `random.Random(base_seed + i)` per method call -- so method A
and method B at the same index i see the identical lead (verified directly,
not assumed, in c_SREP_13.md before this module was written). That licenses
a PAIRED test, which has much more power than an unpaired one for this
sample size.

Since the abandon-vs-freeze fix (c_SREP_13.md), `failed` is `None` for
abandoned campaigns (no commitment was made, so "failure" doesn't apply) --
this module reports three things, not one, matching what a real stage-gate
process actually tracks separately: commit rate, failure-rate-among-commits,
and cost (rounds). Conflating these into a single "failure rate" the way the
pre-fix pilots did is exactly the mistake that produced the vacuous
0-discordant-pairs result this fix was written to correct.

Failure/commit rate: exact paired McNemar's test (via a binomial test on the
discordant pairs) -- the right test for paired binary outcomes, plus a
Wilson-score 95% CI on each rate (better-behaved than a normal-approximation
CI near 0, which matters for the near-zero failure rates seen here).
Rounds / cost: Wilcoxon signed-rank test -- paired, non-parametric (round
counts and rewards are visibly skewed, not roughly normal) -- plus a normal-
approximation 95% CI on the paired mean difference (c_SREP_15.md: added when
scaling to n=1,000 specifically to tighten the unified-vs-GPUCB margin, per
c_SREP_14.md §4 point 3 -- a p-value alone doesn't show HOW tight).
"""
import csv
import math
from pathlib import Path

from scipy.stats import wilcoxon, binomtest


def summarize(results):
    """Reduces a list of campaign-result dicts to the three summary numbers
    every pilot/sensitivity script already prints by hand (c_SREP_13.md's
    three-KPI discipline: commit rate, failure|committed, cost). Factored
    out here (c_SREP_24.md) so the Pareto-figure data pass and the printed
    summaries can't silently drift apart."""
    n = len(results)
    committed = [x for x in results if x["committed"]]
    commit_rate = len(committed) / n
    fails = [x["failed"] for x in committed]
    fail_rate = sum(fails) / len(fails) if fails else float("nan")
    mean_rounds = sum(x["rounds"] for x in results) / n
    return dict(n=n, n_committed=len(committed), commit_rate=commit_rate,
                fail_rate=fail_rate, mean_rounds=mean_rounds)


def append_pareto_row(csv_path, environment, method, setting_type, setting_value, results):
    """Appends one (environment, method, setting) row to the shared Pareto
    data file -- c_SREP_24.md's fix for c_SREP_23.md's finding that none of
    the sweep scripts saved structured output. Call once per (environment,
    method, setting) combination; the header is written once, on first use."""
    row = dict(environment=environment, method=method, setting_type=setting_type,
               setting_value=setting_value, **summarize(results))
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (centre - half, centre + half)


def paired_mean_diff_ci(a, b, z=1.96):
    """Normal-approximation 95% CI for the paired mean difference a-b, plus
    the paired MEDIAN difference alongside it. Added after c_SREP_16.md
    found the two can disagree starkly for skewed, small-integer data (round
    counts, mostly tied or off-by-one, occasionally blown out by a rare
    worst-case run to t_max) -- the normal-approximation CI on the mean is
    not trustworthy when that happens (it's an outlier-driven statistic, not
    a "typical case" one); reporting the median alongside makes that visible
    instead of silently trusting a CI whose own assumptions are violated."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    median_d = sorted(diffs)[n // 2] if n % 2 else (sorted(diffs)[n // 2 - 1] + sorted(diffs)[n // 2]) / 2
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var_d / n)
    return mean_d, median_d, (mean_d - z * se, mean_d + z * se)


def mcnemar_paired(a, b):
    """Exact paired test via binomial(min(b,c), b+c, 0.5) on discordant
    pairs, where b = a-True/b-False, c = a-False/b-True."""
    b_count = sum(1 for x, y in zip(a, b) if x and not y)
    c_count = sum(1 for x, y in zip(a, b) if not x and y)
    n_discordant = b_count + c_count
    if n_discordant == 0:
        return dict(b=b_count, c=c_count, n_discordant=0, p_value=1.0)
    p = binomtest(min(b_count, c_count), n_discordant, 0.5, alternative="two-sided").pvalue
    return dict(b=b_count, c=c_count, n_discordant=n_discordant, p_value=p)


def compare(results_a, results_b, name_a, name_b):
    n = len(results_a)
    committed_a = [r["committed"] for r in results_a]
    committed_b = [r["committed"] for r in results_b]
    rounds_a = [r["rounds"] for r in results_a]
    rounds_b = [r["rounds"] for r in results_b]

    commit_rate_a, commit_rate_b = sum(committed_a) / n, sum(committed_b) / n
    mc_commit = mcnemar_paired(committed_a, committed_b)

    # Failure rate, among each method's OWN committed campaigns (not forced
    # into the same pairs, since one method may commit where the other
    # abandoned on the very same lead).
    fails_a = [r["failed"] for r in results_a if r["committed"]]
    fails_b = [r["failed"] for r in results_b if r["committed"]]
    fail_rate_a = sum(fails_a) / len(fails_a) if fails_a else float("nan")
    fail_rate_b = sum(fails_b) / len(fails_b) if fails_b else float("nan")

    # Failure rate McNemar restricted to campaigns where BOTH methods
    # committed (the only subset where a paired failure comparison is
    # actually meaningful).
    both_committed = [(r_a["failed"], r_b["failed"]) for r_a, r_b in zip(results_a, results_b)
                       if r_a["committed"] and r_b["committed"]]
    if both_committed:
        fa, fb = zip(*both_committed)
        mc_fail = mcnemar_paired(list(fa), list(fb))
    else:
        mc_fail = dict(b=0, c=0, n_discordant=0, p_value=float("nan"))

    try:
        w = wilcoxon(rounds_a, rounds_b)
        w_p = float(w.pvalue)
    except ValueError:
        w_p = 1.0

    mean_r_a, mean_r_b = sum(rounds_a) / n, sum(rounds_b) / n
    ci_commit_a = wilson_ci(sum(committed_a), n)
    ci_commit_b = wilson_ci(sum(committed_b), n)
    ci_fail_a = wilson_ci(sum(fails_a), len(fails_a)) if fails_a else (float("nan"),) * 2
    ci_fail_b = wilson_ci(sum(fails_b), len(fails_b)) if fails_b else (float("nan"),) * 2
    mean_diff, median_diff, ci_diff = paired_mean_diff_ci(rounds_a, rounds_b)

    print(f"  {name_a} vs {name_b} (n={n} paired campaigns):")
    print(f"    commit rate:  {commit_rate_a:.3f} [{ci_commit_a[0]:.3f},{ci_commit_a[1]:.3f}] vs "
          f"{commit_rate_b:.3f} [{ci_commit_b[0]:.3f},{ci_commit_b[1]:.3f}]  "
          f"(McNemar exact: {mc_commit['n_discordant']} discordant, p={mc_commit['p_value']:.4f})")
    print(f"    failure rate | committed:  {fail_rate_a:.3f} [{ci_fail_a[0]:.3f},{ci_fail_a[1]:.3f}] "
          f"(n={len(fails_a)}) vs {fail_rate_b:.3f} [{ci_fail_b[0]:.3f},{ci_fail_b[1]:.3f}] (n={len(fails_b)})  "
          f"(McNemar on {len(both_committed)} jointly-committed pairs: "
          f"{mc_fail['n_discordant']} discordant, p={mc_fail['p_value']:.4f})")
    print(f"    mean rounds:  {mean_r_a:.2f} vs {mean_r_b:.2f}  paired mean diff={mean_diff:.3f} "
          f"95% CI=[{ci_diff[0]:.3f},{ci_diff[1]:.3f}]  paired MEDIAN diff={median_diff:.3f}  "
          f"(Wilcoxon signed-rank: p={w_p:.6f})")
    if abs(median_diff) < 1e-9 < abs(mean_diff) and w_p > 0.05:
        print(f"    NOTE: mean CI excludes 0 but median diff is ~0 and Wilcoxon is non-significant -- "
              f"this is outlier-driven, not a typical-case difference. Trust Wilcoxon here, not the mean CI.")
    return dict(commit_rate_a=commit_rate_a, commit_rate_b=commit_rate_b,
                ci_commit_a=ci_commit_a, ci_commit_b=ci_commit_b,
                mcnemar_commit_p=mc_commit["p_value"],
                fail_rate_a=fail_rate_a, fail_rate_b=fail_rate_b,
                ci_fail_a=ci_fail_a, ci_fail_b=ci_fail_b,
                mcnemar_fail_p=mc_fail["p_value"], n_both_committed=len(both_committed),
                mean_rounds_a=mean_r_a, mean_rounds_b=mean_r_b,
                mean_diff=mean_diff, median_diff=median_diff, ci_diff=ci_diff, wilcoxon_p=w_p)
