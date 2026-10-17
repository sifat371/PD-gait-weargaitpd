# When Participant Disjoint Validation Is Not Enough to Establish Generalization in Parkinson's Gait Classification

**Code and aggregate results for the research manuscript**

**Authors:** Md. Sifat, Sania Akter, Akif Islam, and Md. Ekramul Hamid  
**Paper:** [arXiv:2610.16220](https://arxiv.org/abs/2610.16220)

## Overview

This study examines the relationship between acquisition site, participant-disjoint evaluation, and the generalization of pressure-walkway gait classification for Parkinson's disease. The repository contains executable analysis scripts, versioned dependencies, source-data integrity checks, manuscript tables, and non-identifying model output CSVs.

## Primary results

| Model | Mean ROC-AUC ± SD |
| --- | ---: |
| Site only | 0.955 ± 0.003 |
| Age and sex | 0.755 ± 0.009 |
| Site, age and sex | 0.953 ± 0.005 |
| Gait features only | 0.882 ± 0.014 |
| Site, age, sex and gait features | **0.950 ± 0.011** |

Additional results include within-VA-Seattle performance, JHOC + Bayview to VA-Seattle transport, within-diagnosis site prediction, and hurried-pace sensitivity analyses. See [RESULTS.md](RESULTS.md).

## Files

- [`analysis/run-code/`](analysis/run-code/) — main experimental implementations.
- [`analysis/verify_models.py`](analysis/verify_models.py) — participant-disjoint model evaluation.
- [`results/analysis_outputs/`](results/analysis_outputs/) — actual aggregate CSV outputs from the executed analysis.
- [`results/manuscript/`](results/manuscript/) — tables of numerical values reported in the manuscript.
- [`scripts/run_analysis.sh`](scripts/run_analysis.sh) — reproduce the analysis with access to the original dataset.
- [`METHODS.md`](METHODS.md) — experimental settings and analysis coverage.

## Reproduction

**Dataset:** [WearGait-PD on Synapse, syn52540892](https://doi.org/10.7303/syn52540892). Obtain access according to the data provider's terms. Do not place the input CSVs inside the repository.

The expected input folder contains:

```text
PD - Demographic+Clinical - datasetV1.csv
CONTROLS - Demographic+Clinical - datasetV1.csv
PKMAS Walkway Gait Metrics - HP+SP.csv
```

With Python 3.13 and `uv` available, execute:

```bash
bash scripts/run_analysis.sh /absolute/path/to/data-folder
```

The script verifies the input checksums, creates a private working directory and runs the complete analysis. The checked primary classification suite evaluates five pooled models and six within-diagnosis site classifiers with 20 × 5 participant-disjoint outer folds and three-fold inner model selection.

**Data privacy:** Original clinical CSVs, participant identifiers, individual predictions and clinical intermediate records are not included. Complete runtime folders may contain restricted participant-level outputs and must not be uploaded to GitHub.

## Citation

Please cite the associated paper, [arXiv:2610.16220](https://arxiv.org/abs/2610.16220). Citation metadata are provided in [CITATION.cff](CITATION.cff).

This repository is maintained as the authors' publication code archive. See [CONTRIBUTING.md](CONTRIBUTING.md) for its maintainer-only contribution policy.
