# Result tables

This repository provides two kinds of result files:

**Analysis outputs** in [`results/analysis_outputs/`](results/analysis_outputs/) are aggregate CSVs written by the executable analysis on the study data. These include the pooled five-model decomposition, six within-diagnosis site models, gait-only analyses, VA participant-averaged uncertainty, source-to-target evaluation and site composition.

**Manuscript tables** in [`results/manuscript/`](results/manuscript/) reproduce the values reported in the paper, with the published rounding and interval conventions. These are convenient reviewer references and are not additional experiments.

| Comparison | Manuscript ROC-AUC |
| --- | ---: |
| Pooled, site only | 0.955 ± 0.003 |
| Pooled, gait only | 0.882 ± 0.014 |
| Pooled, site + demographics + gait | 0.950 ± 0.011 |
| Within-VA participant-averaged OOF | 0.668 (0.477–0.836) |
| Source JHOC + Bayview to VA-Seattle | 0.576 (0.375–0.770) |

The participant-averaged VA estimate differs by design from the repeat-level AUC mean displayed in `nested_cv_results.csv`. The fixed-C permutation table and robustness sensitivity table in `results/manuscript/` report the article's reference statistics; they are not counted among the 11 models checked by the verification harness.

All shared outputs are aggregate summaries only. No participant-level records, model prediction arrays or source dataset files are included.
