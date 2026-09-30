# Two-sided Bayesian go/kill rules for preclinical candidate confirmation

Code and results for:

> Jakaite, L. *A two-sided Bayesian go/kill rule for preclinical candidate confirmation: a calibrated failure risk at fewer assays.* Manuscript submitted, 2026.

The paper formulates confirmation of one nominated candidate as a Bayesian optimal-stopping
problem and compares seven decision rules on three public datasets. This repository reproduces
every number, table and figure in it.

**The whole result set regenerates in about 21 seconds and 280 MB of memory on a desktop CPU.**
No GPU, no cluster, no long run.

---

## Quick start

```bash
git clone https://github.com/ljakaite/pharmstat-bayesian-rule.git
cd pharmstat-bayesian-rule
pip install -r requirements.txt
# add the three datasets first -- see "Data" below
python3 build_honest_results.py       # all rules, 3 datasets, n=1000  -> campaigns_honest.csv   (~21 s)
python3 analyse_honest.py             # -> results_honest.md      (Tables 3-5)
python3 build_sensitivity_honest.py   # 45 cells x 500 campaigns   -> sensitivity_honest.csv  (~45 s)
python3 analyse_sensitivity.py        # -> results_sensitivity.md (Tables 1-2)
python3 plot_calibration.py           # -> Figure 1
python3 plot_honest.py                # -> Figure 2
```

## What is here

| File | Purpose |
|---|---|
| `exact_gate.py` | **The proposed rule.** Exact mixture-of-Gaussians belief over competing models, closed-form update, one-step look-ahead stopping and assay-precision choice, two-sided (δ, ε) gate. |
| `stagegate_baselines.py` | Comparators shared across datasets: confidence-bound (GP-UCB-type) gate, DAPP-type and RDM-type signpost rules. |
| `stagegate_stats.py` | Summary statistics and CSV output shared by the run scripts. |
| `gb1/`, `trpb/`, `developability/` | One environment each: data loader (`*_env.py`) and the dataset-specific policy wiring (`*_stagegate.py`, `dev_pomdp.py`), including the robust-satisficing comparator specified in Section 4.4 of the paper. |
| `build_honest_results.py` | Main comparison: every rule, three datasets, 1,000 paired campaigns. |
| `build_exact_results.py` | The proposed rule plus its ablations and the misspecification sweep. |
| `build_sensitivity_honest.py` | Sweeps the failure budget δ, assay noise and specification height — 45 cells per rule. |
| `analyse_honest.py`, `analyse_sensitivity.py` | Turn the per-campaign CSVs into the paper's tables, with Wilson intervals, exact McNemar tests and Wilcoxon signed-rank tests. |
| `plot_calibration.py`, `plot_honest.py` | Figures 1 and 2, written as `.png`, `.pdf`, `.eps` and 600-dpi `.tiff`. |

### Reproducibility design

Campaign seeds are shared across rules, so campaign *i* presents **the same candidate and the
same noise stream** to every rule. All comparisons in the paper are therefore paired, and
re-running any script reproduces the published CSVs byte for byte.

## Data

The three datasets are public but are **not redistributed here**. Download them and place them as
follows:

| Dataset | Expected path | Source |
|---|---|---|
| GB1 four-site landscape (149,361 variants) | `gb1/data/four_mutations_full_data.csv` | Wu et al., *eLife* 2016; via the FLIP benchmark repository |
| TrpB four-site landscape (160,000 variants) | `trpb/data/dataset.csv` | Johnston et al., *PNAS* 2024 |
| Clinical-stage antibody panel (137 antibodies, 12 assays) | `developability/data/pnas_xlsx/pnas.1616408114.sd01–sd03.xlsx` | Jain et al., *PNAS* 2017, Supplementary Datasets S1–S3 |

`gb1/gb1_env.py` and `trpb/trpb_env.py` accept either the full published files or column-trimmed
copies; each loader documents the schema it expects.

## Requirements

Python 3.10 or later, plus `scipy`, `scikit-learn`, `matplotlib` and `openpyxl`
(see `requirements.txt`). Everything else is the standard library.

## A note on the comments

Source comments cite internal development notes by filename (`c_SREP_*.md`,
`c_protein_design_*.md`). Those notes are the authors' working record — they say *why* a
particular constant, fix or design choice was made, including several bugs found and corrected
during development — and they are not distributed. The comments are readable without them; the
citations simply mark where a decision was recorded.

Two of those corrections are reported in the paper itself and are worth knowing when reading the
code:

- **The candidate-pool leakage correction** (paper, Section 5.2). An earlier competing-model set
  for the two landscapes contained a pairwise-epistasis model that reproduced each candidate's
  own measurement exactly, because its interaction table was fitted on the very variants later
  used as candidates. `use_honest_models()` in `gb1/gb1_stagegate.py` and
  `trpb/trpb_stagegate.py` removes it and restricts campaigns to the two-substitution pool.
  Calling it is what reproduces the paper; not calling it reproduces the inflated earlier
  numbers.
- **Lead-pool memoisation.** `pick_lead` caches the candidate pool, which is invariant across
  campaigns. This is a pure speed optimisation — verified to leave every reported result byte
  identical — and it is the reason the full run takes seconds rather than twenty minutes.

## Licence

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper; if you use the archived snapshot, cite its DOI as
well: [doi:10.5281/zenodo.23048060](https://doi.org/10.5281/zenodo.23048060).
