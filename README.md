# KG-MSPF

Knowledge-Guided Multi-Scale Prognosis with Subtype-Aware Heterogeneous Graphs.

This repository provides the evaluated KG-MSPF implementation, matched survival baselines, component controls and data preparation utilities. Each patient is represented by clinical, gene, pathway, subtype and patient nodes. Relational attention supplies graph features; clinical, subtype, pathway and graph scores are combined with Cox anchors.

## Installation

Python 3.12 is the tested runtime. From the repository root:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.tested.txt
python -m pytest tests -q
```

The default device in `reproduce.py` is CPU. CUDA can be selected with `--device cuda` when supported by the PyTorch installation. Windows AMD users may install `requirements.directml.txt` and select `--device directml`. RSF and Cox baselines use CPU. Hardware and library differences can affect numerical results and execution time.

## Data

Download `METABRIC_RNA_Mutation.csv` separately and place it under `data/`. The exact source, input fingerprint, endpoint coding and expected cohort size are described in [data/README.md](data/README.md) and [data_sources.json](data_sources.json).

Prepare TCGA-BRCA from the pinned cBioPortal DataHub revision:

```bash
python fetch_tcga.py --revision 7a4b4ec142f76ac04db22834232b23142d25bd2a --cache data/tcga_source --out external_data/tcga_brca_harmonized.csv
```

Third-party patient-level data are obtained from their original providers and are not included in this code distribution. The MIT software license does not relicense the source datasets.

## Reproduce the evaluation

The fixed paper settings are in `paper_config.json`. To inspect the plan without fitting models:

```bash
python reproduce.py --stages main,holdout,ablation,pam50,prior,external --dry-run
```

Run stages explicitly:

```bash
python reproduce.py --stages main --out outputs/reproduction --device cpu
python reproduce.py --stages holdout,ablation,pam50,prior --out outputs/reproduction --device cpu
python reproduce.py --stages external --out outputs/reproduction --device cpu
python postprocess.py --root outputs/reproduction
```

The main comparison contains 11 models across 5 repetitions of 5 folds (275 model fits). Component, PAM50 and prior controls use the first five-fold repetition. The public reproduction plan covers the configurations reported with the manuscript: 437 model/configuration/split records across all six stages; reference configurations can reuse compatible main fits. Derived blends share fitted components and are not additional independent training experiments. `run_experiments.py --help` provides lower-level selectors.

Each result is saved separately and compatible completed results are reused. Use a new output directory after changing code, settings, data or dependencies. Full runs are computationally expensive. The software tests use synthetic arrays and do not establish study performance.

## Evaluation protocol

- Overall survival uses `death_from_cancer`: any recorded death is an event; living patients are censored. In this input table, `overall_survival` is a living-status indicator and is not used as the event label.
- Gene screening, imputation and scaling are fitted on training patients. Inner validation has independently fitted preprocessing.
- Repetition split seeds are `42+r`; training seeds are `42+1000*r`. Folds are stratified jointly by PAM50 and event status. Preserve the input row order to reproduce splits.
- The cohort hold-out trains on METABRIC cohorts 1–3 and evaluates cohorts 4–5.
- TCGA uses shared-feature METABRIC refitting and unlabeled within-cohort expression standardization. Unavailable mutation calls remain missing. It is not unchanged deployment of the original full-input model.
- Repeated-split comparisons use corrected resampled t-tests and Holm adjustment across ten baseline comparisons. Conditional patient bootstraps do not refit models.

The evaluation is retrospective; the cohorts had been inspected during development and are not new blind validation datasets.

## Implementation scope

This is a code-only distribution. It contains no precomputed performance tables or study-result figures. Running the supplied scripts generates results locally.

DeepSurv and single-event DeepHit are local reimplementations. SurvTRACE-style and Cox-PASNet-style are approximations; the latter does not reproduce the original sparse-coding step. See `model_implementations.json` for the implemented model definitions. Identifiers inside result files retain their short model names; these identifiers do not imply complete original-author implementations.

## Prediction

The experiment runner saves portable CPU checkpoints. For example:

```bash
python inference_heterogat.py --checkpoint outputs/reproduction/holdout/checkpoints/KG-MSPF_cohort_holdout.pt --input harmonized.csv --output predictions.csv --device cpu
```

Load only checkpoints from a trusted source. A new input must use the harmonized schema expected by `CohortTable.from_harmonized_csv`. Individual predictions use the training reference scale; within-cohort scaling requires an explicit reference batch. No fitted patient-level model checkpoint is included in this distribution.

## Citation and license

Creator: Feiran Wang. Software version: 1.0.0. Citation metadata are provided in `CITATION.cff`. Source code is distributed under the existing MIT license in `LICENSE`; dataset terms remain those of the original providers.

## Repository and archived version

Source repository: https://github.com/salty-kk/KG-MSPF

Version 1.0.0 archive: https://doi.org/10.5281/zenodo.23667027
