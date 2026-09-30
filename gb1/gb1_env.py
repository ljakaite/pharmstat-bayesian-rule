"""
GB1 environment for the protein-design POMDP pilot.

Loads the real Wu et al. (2016) / Olson et al. (2014) GB1 combinatorial
landscape (four sites V39/D40/G41/V54, ~149,361 variants), sourced via the
FLIP benchmark repository (J-SNACKKB/FLIP), and builds:

  - a ground-truth fitness oracle (hidden from the agent during a campaign,
    revealed only for a queried variant's noisy assay, or fully at commit)
  - a per-variant hidden "regime" label (additive vs epistatic), derived
    directly from the data itself rather than assumed: a variant is labelled
    "epistatic" if its true fitness deviates materially from the fitness
    predicted by summing its constituent single-mutant effects (the
    reciprocal-sign-epistasis structure documented in Wu et al. 2016 and the
    Wu/Olson et al. 2016 eLife "indirect paths" paper -- see c_SREP_7.md §2).

This mirrors biscayne_main.py's role: a real, small, fast environment that
the ported Algorithm 1 (satisficing + VoI + real-options) can run against.
"""
import csv
import random
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "four_mutations_full_data.csv"
WT = "VDGV"  # wild-type residues at the four mutated positions

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def load_gb1(path: Path = DATA_PATH):
    """Returns dict: 4-letter variant code -> true fitness (float)."""
    fitness = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row["Variants"]
            if len(v) != 4:
                continue
            try:
                fitness[v] = float(row["Fitness"])
            except (ValueError, KeyError):
                continue
    return fitness


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def additive_prediction(variant: str, fitness: dict, wt: str = WT) -> float:
    """Predicted fitness under a purely additive (no-epistasis) model:
    wild-type fitness + sum of each single-mutant's individual effect.
    Requires all four single-mutant fitnesses to be present (true for GB1's
    near-complete landscape)."""
    pred = fitness[wt]
    for i in range(4):
        if variant[i] == wt[i]:
            continue
        single = wt[:i] + variant[i] + wt[i + 1:]
        if single not in fitness:
            return None
        pred += fitness[single] - fitness[wt]
    return pred


def build_regime_labels(fitness: dict, wt: str = WT, epistasis_threshold: float = None):
    """Hidden ground-truth regime per variant: 'epistatic' if |true - additive
    prediction| exceeds `epistasis_threshold`, else 'additive'. HD<=1 variants
    are additive by construction (a single mutation cannot show combinatorial
    epistasis).

    `epistasis_threshold=None` (default) uses the MEDIAN absolute deviation
    across all HD>=2 variants, giving a balanced ~50/50 split -- chosen after
    checking the actual deviation distribution (c_SREP_8.md): a fixed
    absolute threshold like 0.15 labelled 93% of variants "epistatic",
    because GB1's median deviation from additivity is itself ~1.0 (this is a
    real, well-known property of GB1 -- it's why Wu et al. chose this
    landscape to study epistasis -- not a bug), which makes a small fixed
    threshold uninformative as a regime split. The median-based cut is a
    data-driven, non-degenerate alternative; still a first-pass choice, not a
    literature-derived one -- flagged as an open item."""
    devs = {}
    for v in fitness:
        if hamming(v, wt) < 2:
            continue
        pred = additive_prediction(v, fitness, wt)
        if pred is not None:
            devs[v] = abs(fitness[v] - pred)

    if epistasis_threshold is None:
        sorted_devs = sorted(devs.values())
        epistasis_threshold = sorted_devs[len(sorted_devs) // 2]

    regime = {}
    for v in fitness:
        if hamming(v, wt) <= 1:
            regime[v] = "additive"
        elif v in devs:
            regime[v] = "epistatic" if devs[v] > epistasis_threshold else "additive"
    return regime


class GB1Environment:
    """One self-contained GB1 campaign environment instance.

    Ground truth (fitness, regime) is hidden from the agent. The agent only
    sees noisy assay observations of variants it chooses to query, exactly as
    biscayne_main.py hides the true regime_seq/model from pi_r/pi_m.
    """

    def __init__(self, fitness: dict, regime: dict, rng: random.Random,
                 sigma_low: float = 0.18, sigma_high: float = 0.06):
        self.fitness = fitness
        self.regime = regime
        self.rng = rng
        self.variants = list(fitness.keys())
        self.sigma_low = sigma_low   # noise for a single cheap assay
        self.sigma_high = sigma_high  # noise for a confirmatory ("monitor") assay

    def random_neighbor(self, variant: str, max_hd: int = 2) -> str:
        """A candidate reachable by <=max_hd point mutations from `variant`
        -- the discrete analogue of biscayne's regime-driven state transition,
        standing in for 'propose the next variant to test'."""
        v = list(variant)
        n_mut = self.rng.randint(1, max_hd)
        positions = self.rng.sample(range(4), n_mut)
        for p in positions:
            v[p] = self.rng.choice(AA20)
        cand = "".join(v)
        return cand if cand in self.fitness else variant  # fall back if not in the measured set

    def true_fitness(self, variant: str) -> float:
        return self.fitness.get(variant, 0.0)

    def true_regime(self, variant: str) -> str:
        return self.regime.get(variant, "additive")

    def assay(self, variant: str, monitor: bool = False) -> float:
        """Noisy observation of a variant's fitness."""
        sigma = self.sigma_high if monitor else self.sigma_low
        true = self.true_fitness(variant)
        return self.rng.gauss(true, sigma)

    def empirical_gain_pools(self, rng: random.Random, n_samples: int = 200_000):
        """Precompute the REAL empirical distribution of positive-part
        true-fitness gain from one random-neighbour step, bucketed by the
        current variant's regime label. Replaces the hand-picked parametric
        rollout constants in gb1_pomdp.py's original version with bootstrap
        pools drawn from the actual data (c_SREP_9.md §1).

        Finding worth carrying forward, not hidden: the two regimes' pooled
        distributions come out close to each other (~0.44 vs ~0.44 P(improve)
        per c_SREP_9.md's measurement) -- because `random_neighbor` can jump
        1-2 mutations away regardless of the *current* variant's own regime
        label, so a single-variant regime tag is a weak predictor of the
        *next* step's outcome. Kept as the honest empirical pools rather than
        smoothing this over; flagged in c_SREP_9.md as the next thing to fix
        (regime should probably be a property of a local neighbourhood/basin,
        not a single variant).
        """
        pools = {"additive": [], "epistatic": []}
        for _ in range(n_samples):
            v = rng.choice(self.variants)
            r = self.regime.get(v, "additive")
            cand = self.random_neighbor(v)
            gain = max(0.0, self.true_fitness(cand) - self.true_fitness(v))
            pools[r].append(gain)
        return pools
