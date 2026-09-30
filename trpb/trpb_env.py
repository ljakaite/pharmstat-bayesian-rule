"""
TrpB environment: the fourth benchmark (c_SREP_14.md §4 point 4 / c_SREP_16.md),
added specifically to enable a same-landscape comparison against ALDE's own
published numbers, per c_SREP_7.md §3.

Real, public data: Johnston, Almhjell, Watkins-Dulaney, Liu, Porter, Yang,
Arnold, "A combinatorially complete epistatic fitness landscape in an enzyme
active site," PNAS 121(31):e2400439121 (2024) -- the exact TrpB dataset ALDE
itself uses. Sourced via the HuggingFace dataset
`SaProtHub/Dataset-TrpB_fitness_landsacpe` (MIT license) rather than ALDE's
own Zenodo bundle (5.8GB, packaged with ML encodings this project doesn't
need) or PNAS's supplementary files directly (not checked for automated-
fetch blocking, given the HuggingFace copy worked immediately).

Four mutated positions, found empirically (not assumed from the paper's own
1-indexed residue numbering, to avoid an off-by-one error against this
dataset's own 0-indexed sequence strings): 0-indexed [182, 183, 226, 227],
matching the paper's stated 183/184 and 227/228 position pairs exactly.

**One honest substitution, flagged rather than glossed over:** the dataset
does not label which row is the literal parent (Tm9D8*) sequence, and this
session did not track down Tm9D8*'s exact residues at these four positions
from a primary source in the time available. Used the single
highest-fitness variant in the data (fitness exactly 1.0, the landscape's
own stated maximum) as the reference point for the additive/pairwise/
global-nonlinear models instead -- a real, data-grounded anchor (arguably a
more relevant reference for a directed-evolution *lead-selection* scenario
than an arbitrary historical wild-type would be), not a biological parent
sequence. If exact fidelity to Tm9D8* matters later, this is the one thing
to revisit.
"""
import csv
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "dataset.csv"
POSITIONS = (182, 183, 226, 227)  # 0-indexed into the full sequence string
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _extract_code(seq: str) -> str:
    return "".join(seq[p] for p in POSITIONS)


def load_trpb(path: Path = DATA_PATH):
    """Returns dict: 4-letter variant code -> fitness (float)."""
    fitness = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = _extract_code(row["protein"])
            try:
                fitness[code] = float(row["label"])
            except (ValueError, KeyError):
                continue
    return fitness


WT = None  # set by `reference_variant` below -- there is no biological WT
           # in this dataset (see module docstring); computed once, not
           # hardcoded, so it's traceable to the actual data.


def reference_variant(fitness: dict) -> str:
    """The best-known variant (fitness == the landscape's own max) -- see
    module docstring for why this replaces a literal WT/parent."""
    return max(fitness, key=fitness.get)


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def additive_prediction(variant: str, fitness: dict, wt: str):
    pred = fitness[wt]
    for i in range(4):
        if variant[i] == wt[i]:
            continue
        single = wt[:i] + variant[i] + wt[i + 1:]
        if single not in fitness:
            return None
        pred += fitness[single] - fitness[wt]
    return pred
