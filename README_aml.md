# Expansion or Contraction? Oversampling vs Edge Pruning for Imbalanced Anti-Money-Laundering Detection with Graph Attention Networks

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.x-3C2179)
![Task](https://img.shields.io/badge/task-edge%20classification-green)

Code for the MSc dissertation *Expansion or Contraction: Comparing Oversampling and Edge Pruning for Imbalanced Anti-Money-Laundering Detection with Graph Attention Network* (University of Surrey, 2026).

**Author:** Nguyen Long Hoang · **Supervisor:** Dr Alaa Marshan · 📄 [Full dissertation](docs/dissertation.pdf)

---

## Overview

Money laundering happens through chains of transactions, which makes graph neural networks a natural fit for detecting it. Two problems get in the way:

- **Extreme class imbalance.** Illicit transactions are well under 0.1% of activity.
- **Over-smoothing.** The deep networks needed to follow multi-hop laundering chains make account representations converge until they can no longer be told apart.

The literature offers two opposite fixes, based on opposite diagnoses of the same failure:

| Strategy | Method | Idea |
|---|---|---|
| **Signal expansion** | GraphSMOTE-Edge | Synthesise new illicit transactions by interpolating between real ones. An adaptation of GraphSMOTE from node classification to edge classification, developed for this study. |
| **Graph contraction** | DropEdge | Randomly remove message-passing edges each epoch to slow over-smoothing. |

This project is a controlled comparison of the two on the IBM AML **HI-Small** benchmark. Everything except the intervention is held fixed: data, temporal split, training budget and evaluation protocol. The study covers ten conditions, three network depths (2, 4, 8) and five seeds per condition, and all analysis rules were fixed before any result was seen.

---

## Key findings

1. **The loss function matters more than either intervention.** Class-weighted cross-entropy reaches about 17× chance. Plain cross-entropy and focal loss (α = 0.25, γ = 2) stay at chance level.
2. **Depth helps only on the directed-multigraph backbone.** Multi-GAT with DropEdge reaches about 5× the baseline at depth 8, the study's one statistically supported result. However, Multi-GAT *without* any intervention already gets most of that gain (3.2× the baseline), and the extra contribution of DropEdge cannot be separated from run-to-run noise.
3. **Over-smoothing is severe and measurable.** Final-layer Dirichlet energy falls roughly 600-fold from depth 2 to depth 8. The backbone that resists it best is the one that detects best at depth.
4. **Interpolation adds nothing over duplication.** GraphSMOTE-Edge performs no better than simply duplicating the same illicit transactions, so the absence of benefit lies in the interpolation step itself.

**Practical takeaway:** at this level of imbalance, choose the loss function first and the topological intervention second, and evaluate both at the depth you plan to deploy. A shallow-only evaluation would have found no depth effect at all.

<p align="center">
  <img src="assets/auprc_vs_depth.png" width="70%" alt="AUPRC against network depth for the six depth-sweep conditions">
</p>

---

## Results

AUPRC is the primary metric. A random ranking scores an AUPRC equal to the positive prevalence, which is about 0.15% in the test period, so every result is also reported as **lift**: AUPRC divided by that chance level. A lift of 1× is random. Values are mean ± standard deviation over five seeds.

### All ten conditions at the reference depth (2 layers)

| ID | Backbone | Intervention | Loss | AUPRC | Lift |
|---|---|---|---|:-:|:-:|
| E0 | MLP (no graph) | none | weighted CE | 0.0086 ± 0.0076 | 5.7× |
| E1 | GAT | none | cross-entropy | 0.0021 ± 0.0003 | 1.4× |
| E2 | GAT | none | weighted CE | 0.0248 ± 0.0062 | 16.6× |
| E3 | GAT | none | focal | 0.0018 ± 0.0003 | 1.2× |
| E4 | GAT | DropEdge | weighted CE | 0.0239 ± 0.0129 | 16.0× |
| E5 | GAT | GraphSMOTE-Edge | weighted CE | 0.0370 ± 0.0144 | 24.8× |
| E6 | Multi-GAT | none | weighted CE | 0.0295 ± 0.0083 | 19.7× |
| E7 | Multi-GAT | DropEdge | weighted CE | 0.0274 ± 0.0133 | 18.4× |
| E8 | Multi-GAT | GraphSMOTE-Edge | weighted CE | 0.0165 ± 0.0097 | 11.1× |
| E9 | GAT | duplication (control) | weighted CE | **0.0407 ± 0.0180** | **27.3×** |

Most standard deviations overlap, so differences at depth 2 are reported in the dissertation as directional only.

### Depth sweep

| Condition | Depth 2 | Depth 4 | Depth 8 |
|---|:-:|:-:|:-:|
| E2 GAT, none | 0.0273 (18×) | 0.0199 (13×) | 0.0077 (5×) |
| E4 GAT, DropEdge | 0.0229 (15×) | 0.0117 (8×) | 0.0098 (7×) |
| E5 GAT, GraphSMOTE-Edge | 0.0390 (26×) | 0.0245 (16×) | 0.0105 (7×) |
| E6 Multi-GAT, none | 0.0311 (21×) | 0.0304 (20×) | 0.0247 (17×) |
| E7 Multi-GAT, DropEdge | 0.0235 (16×) | 0.0246 (16×) | **0.0367 (25×)** |
| E8 Multi-GAT, GraphSMOTE-Edge | 0.0175 (12×) | 0.0150 (10×) | 0.0153 (10×) |

The sweep is a separate execution with its own seeds, so its depth-2 values differ slightly from the main table. Comparisons are only made within one table.

**Principal result (depth 8, paired across seeds):**

| Comparison | Mean difference | 95% CI | Paired t-test | Seeds favouring the first |
|---|:-:|:-:|:-:|:-:|
| E7 vs E2 (Multi-GAT + DropEdge vs GAT baseline) | +0.0290 | +0.0147 to +0.0433 | t(4) = 5.62, p = 0.005 | 5 / 5 |
| E7 vs E6 (does DropEdge add anything to Multi-GAT?) | — | spans zero | t(4) = 1.26, p = 0.28 | 3 / 5 |
| E7 vs E8 (contraction vs expansion on Multi-GAT) | +0.0214 | +0.0055 to +0.0372 | t(4) = 3.74, p = 0.020 | 5 / 5 |

### Intervention strength (depth 2)

DropEdge declines steadily as the drop rate rises (16× at p = 0.1, down to 10× at p = 0.5). GraphSMOTE-Edge is stable between oversampling ratios 0.5 and 1.0 (25–26×) and then collapses at r = 2.0 (7×). Both degrade at high strength, but contraction erodes gradually while expansion holds and then fails.

<p align="center">
  <img src="assets/dirichlet_energy.png" width="85%" alt="Final-layer Dirichlet energy against depth">
</p>

---

## Method

**Data.** IBM Transactions for Anti-Money-Laundering, HI-Small variant: 5,078,345 transactions among about 515,000 accounts over 18 days, 0.1019% illicit. A 30% subset is used (the first 30% of each day, so every day stays represented): 1,523,495 transactions, 1,101 of them illicit. The split is chronological 60:20:20 on day boundaries. Illicit prevalence rises over time, from about 0.05% in training to about 0.15% in test, leaving roughly 390 illicit transactions in the test period.

**Task.** Binary classification of edges (transactions) in a directed multigraph of accounts (nodes), using 11 aggregated account features and 7 transaction features.

**Backbones.**

- **GAT**: an edge-aware Graph Attention Network.
- **Multi-GAT**: the same GAT with two of the three directed-multigraph adaptations from Egressy et al. (2024), reverse message passing and port numbering.

Per-layer edge updates are disabled for both, so the backbones differ only in those adaptations.

**Interventions.**

- **DropEdge** removes 20% of message-passing edges each epoch. The supervised edge set is never reduced, so any change in recall cannot come from accidentally undersampling positive examples.
- **GraphSMOTE-Edge** interpolates new illicit transactions in the continuous feature space between each anchor and one of its K = 5 nearest illicit neighbours, at an oversampling ratio of 1.0. Categorical attributes are inherited from the anchor, and cyclical time features are recomputed from the inherited timestamp.
- **Duplication control (E9)** copies minority edges at the same ratio without interpolating, isolating what interpolation contributes.

**Training.** Hidden dimension 64, four attention heads, Adam, gradient clipping at 1.0, dropout 0.1, up to 100 epochs with early stopping after 20 epochs without validation improvement. The learning rate (0.006) was tuned at both the shallowest and deepest depths and reconciled to one value, so it is not confounded with depth.

**Evaluation.**

- AUPRC with chance level and lift.
- Recall and precision at a threshold chosen on validation for 10% precision.
- Precision at k = 100 and 500 alerts.
- Two over-smoothing diagnostics: Dirichlet energy and mean average distance, computed on a fixed sample of 1,000 accounts.

**Leakage controls.** Four leakage controls are applied, described in Section 3.4.4 of the dissertation.

---

## Getting started

### Requirements

| Component | Version used |
|---|---|
| Python | 3.10 or later |
| PyTorch | 2.x with CUDA |
| PyTorch Geometric | 2.x |
| pandas, NumPy, scikit-learn, SciPy, Matplotlib | current at time of writing |
| GPU | NVIDIA L4, 22 GiB |

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**A GPU is required in practice.** Full-batch training at depth 8 on the subset needs about 12 GiB. The code runs on CPU, but the full programme would take days rather than hours.

### Data

The dataset is public but too large to include. Download `HI-Small_Trans.csv` from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and convert it to the schema the pipeline expects:

```bash
python scripts/format_kaggle_pandas.py /path/to/HI-Small_Trans.csv
```

This writes `formatted_transactions.csv`.

### Configuration

All settings are read from environment variables, with defaults in `experiments/common.py`.

```bash
export AML_DATA=/path/to/formatted_transactions.csv
export AML_RESULTS=./results
export AML_FIGURES=./figures
```

| Variable | Default | Meaning |
|---|---|---|
| `AML_SUBSET_FRACTION` | `0.30` | proportion of each day retained |
| `AML_DEPTHS` | `2 4 8` | depths in the sweep |
| `AML_SEEDS` | `0 1 2 3 4` | five seeds per condition |
| `AML_MAX_EPOCHS` | `100` | with early stopping at patience 20 |
| `AML_EDGE_UPDATES` | `0` | per-layer edge MLP, disabled throughout |
| `AML_REQUIRE_GPU` | `L4` | refuses to start on other hardware |

Results from different hardware are not strictly comparable, so the device is recorded in every result file. A run on different hardware invalidates the cache rather than silently mixing results. Set `AML_REQUIRE_GPU` to your device name if it is not an L4.

### Run the full programme

```bash
cd experiments
python run_all.py --dry-run        # print the plan without executing
python run_all.py                  # execute in order
python run_all.py --from 5         # resume from step 5
```

Each step runs as a separate process, because process termination is the only guaranteed way to release GPU memory. Running everything in one long-lived process lets allocations from earlier steps fragment the memory pool. The full programme takes roughly **11 to 13 hours** on an L4.

| # | Script | Runs | Approx. time |
|---|---|:-:|:-:|
| 1 | `exp1_data_understanding.py` | — | 5 min |
| 2–4 | `exp2_tuning.py --depth 2`, `--depth 8`, `--combine` | 12 | 30 min |
| 5 | `exp3_main_matrix.py` | 50 | 65 min |
| 6 | `exp4_depth_sweep.py` | 90 | 6 h |
| 7 | `exp5_intervention_strength.py` | 30 | 50 min |
| 8 | `exp6_figures.py` | — | seconds |
| 9 | `exp7_statistics.py` | — | 2 min |

Results are written after every condition, so an interrupted programme resumes without repeating finished work. Each result file records the configuration that produced it (seeds, epochs, depth, edge-update setting, device, and whether prediction scores were kept). A run with different settings discards the cached file instead of combining results from different configurations.

### Reproduce individual analyses

Each script runs on its own, as long as the results it depends on exist.

```bash
cd experiments
python exp6_figures.py                  # every figure and table
python exp6_figures.py --export-only    # the CSV exports alone
python exp7_statistics.py --depth 8     # statistical analysis
python exp9_subset_validation.py        # subset representativeness
```

For formatted output suitable for direct quotation:

```python
import sys; sys.path.insert(0, "src")
import report

report.split_prevalence_block("results")       # Section 4.3.2
report.paired_block("results", 8, "E7_vs_E2")  # Section 5.4.2
report.max_f1_block("results", 8)
report.inductive_block("results", 8)
report.environment_block()                     # Section 4.4.6
```

Run the unit tests with:

```bash
python -m pytest tests/ -q
```

---

## Outputs

**Result files (`AML_RESULTS`)**

| File | Contents |
|---|---|
| `exp1_data_understanding.json` | data profile, split prevalences, leakage checks |
| `exp2_tuning*.json` | per-depth tuning results and the reconciled values |
| `exp3_main_depth2.json` | all ten conditions at the reference depth |
| `exp4_depth{2,4,8}.json` | the depth sweep with over-smoothing diagnostics |
| `exp4_depth*_scores.json` | raw per-transaction predictions |
| `exp5_intervention_strength.json` | DropEdge rate and oversampling ratio sweeps |
| `exp7_statistics_depth8.json` | paired tests, bootstrap, maximum F1, inductive split |

**Figures and tables (`AML_FIGURES`)** include two master exports that support further analysis without retraining:

- `results_long.csv`: one row per experiment, depth, seed and metric.
- `results_wide.csv`: one row per condition, with mean and standard deviation.

`CUSTOM_FIGURES.md` explains how to build additional figures from these.

---

## Repository structure

```
src/                        library code, unchanged across experiments
  data_understanding.py     profiling, schema and missingness checks
  features.py               node and edge feature construction
  data_pipeline.py          graph construction, temporal split, leakage controls
  models.py                 GATe backbone (shared by both arms) and MLP baseline
  multigat.py               reverse message passing and port numbering
  graph_cache.py            one-time graph preparation on the target device
  interventions.py          DropEdge, GraphSMOTE-Edge, duplication, loss functions
  trainer.py                training loop with message/supervision separation
  diagnostics.py            Dirichlet energy and mean average distance
  evaluation.py             AUPRC, recall, precision, top-k, threshold selection
  tuning.py                 two-depth grid search and reconciliation
  run_experiments.py        condition matrix, orchestration, checkpointing
  figures.py                all figures and tables
  results_export.py         tidy CSV exports
  stats_analysis.py         paired tests, bootstrap, maximum F1
  inductive.py              performance split by account history
  subset_validation.py      subset against full dataset
  report.py                 formatted output for the written report
experiments/                one script per experiment
scripts/                    data formatting and environment setup
configs/                    default configuration
tests/                      unit tests
docs/dissertation.pdf       the full dissertation
assets/                     figures used in this README
```

---

## Verifying the methodological claims

Four claims in the dissertation can be checked directly against the code.

**The subset preserves the structure of the data.** `exp9_subset_validation.py` compares the subset with the full dataset across transaction attributes, graph topology and temporal structure. It reports effect sizes (Kolmogorov–Smirnov statistic and total variation distance) rather than p-values, because at these sample sizes a two-sample test flags differences far too small to affect any model. Degree is compared after normalising by the mean, since a subset necessarily has lower absolute degree.

**Message passing and supervision are separated.** DropEdge removes edges from the message-passing graph only; the supervised edge set is never reduced. See `_run_model` in `src/trainer.py` and the `sup_edge_index` parameter of `GATe.forward` in `src/models.py`.

**The tuned learning rate is not confounded with depth.** Tuning runs at both the shallowest and deepest depths and is reconciled to a single value. The reconciliation rule is stated in `src/tuning.py` and was fixed before results were examined.

**The interventions are isolated from one another.** Each condition applies exactly one intervention, and per-layer edge updates are disabled uniformly so the two backbones differ only in the directed-multigraph adaptations. The condition matrix is defined in `ABLATION` in `src/run_experiments.py`.

---

## Limitations

- **Statistical power is the binding constraint.** With about 390 illicit test transactions and five seeds, standard deviations are often as large as the means. Only one comparison produces non-overlapping ranges; other differences are reported as directional.
- **A p-value over five seeds measures training variability on one subset and one temporal split.** It is not evidence of generalisation to other graphs or to the full benchmark.
- **Results are not comparable with published full-dataset numbers.** The study uses a 30% subset, implements two of the three Multi-GNN adaptations, and disables edge updates.
- **Single temporal split, and depth limited to 8** by GPU memory.
- **Bitwise reproducibility is not claimed.** GPU floating-point reduction order is non-deterministic, so identical runs vary in the third or fourth decimal place. Seeds are fixed for initialisation and intervention sampling.

---

## Troubleshooting

**`CUDA out of memory`**: reduce `AML_SUBSET_FRACTION`, or cap `AML_DEPTHS` at a lower value. The heaviest configuration is E8 at the greatest depth, so probing it first shows what will fit.

**Results look stale or inconsistent**: each file records its configuration and re-runs when settings differ. To force a clean start, move the contents of `AML_RESULTS` elsewhere and re-run.

**Figures are produced but do not display**: the code uses a headless Matplotlib backend when run as a script and an interactive one inside a notebook. Figures are always written to `AML_FIGURES`.

**Maximum F1 or the inductive analysis reports missing scores**: those analyses need the saved per-transaction predictions. Run `python exp8_add_scores.py --depth 8` to regenerate that depth with scores kept.

---

## Acknowledgements

The edge-aware GAT layer and temporal-split logic are adapted from the [Multi-GNN implementation](https://github.com/IBM/Multi-GNN) of Egressy et al. (2024), released under the Apache 2.0 licence. Everything else is original to this study: feature construction, graph caching, message/supervision separation, GraphSMOTE-Edge and its duplication control, over-smoothing diagnostics, evaluation protocol, tuning procedure and experiment runner.

## References

- Altman, E., et al. (2023). Realistic synthetic financial transactions for anti-money laundering models. *NeurIPS Datasets and Benchmarks*.
- Egressy, B., et al. (2024). Provably powerful graph neural networks for directed multigraphs. *AAAI*.
- Rong, Y., et al. (2020). DropEdge: Towards deep graph convolutional networks on node classification. *ICLR*.
- Zhao, T., Zhang, X., & Wang, S. (2021). GraphSMOTE: Imbalanced node classification on graphs with graph neural networks. *WSDM*.
- Veličković, P., et al. (2018). Graph attention networks. *ICLR*.

## Citation

```bibtex
@mastersthesis{hoang2026expansion,
  author = {Hoang, Nguyen Long},
  title  = {Expansion or Contraction: Comparing Oversampling and Edge Pruning for
            Imbalanced Anti-Money-Laundering Detection with Graph Attention Network},
  school = {University of Surrey},
  year   = {2026}
}
```
