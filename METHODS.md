# Experimental methods

This repository accompanies [arXiv:2610.16220](https://arxiv.org/abs/2610.16220) and supplies the analysis scripts for diagnosis classification, site prediction, and site-aware evaluation.

## Participants and features

The primary SelfPace analysis includes 133 participants (65 with Parkinson's disease and 68 controls) and 159 pressure-walkway gait features. Three physical acquisition sites were represented: JHOC, Bayview, and VA-Seattle. VA-Seattle was the only site containing both diagnostic groups.

## Models and validation

- Classifier: L1-regularized logistic regression (`liblinear`).
- Outer validation: participant-disjoint stratified 5-fold CV repeated 20 times.
- Inner validation: 3 stratified folds optimizing ROC-AUC.
- C candidates: 0.03, 0.1, 0.3, 1.0.
- Seed: 42; inner split seed increases with outer split index.
- Preprocessing: median imputation in training folds, followed by standardization of model predictors.
- Feature screening: exclude unusable `#Samples` columns and high-missingness gait features according to the study protocol before cross-validation.

The pooled diagnosis comparison uses site alone; age and sex; site, age, and sex; 159 gait features; and site, age, sex, and gait features. The within-diagnosis site analyses also incorporate measured covariates, including anthropometric data; the PD analysis requests additional clinical covariates.

The primary site-aware analyses evaluate VA-Seattle using participant-averaged repeated out-of-fold probabilities and test JHOC + Bayview to VA-Seattle transport. HurriedPace provides a separate walking-condition analysis.

## Experiment-to-result mapping

| Analysis | Script | Public aggregate results |
| --- | --- | --- |
| Five pooled diagnosis models | `analysis/verify_models.py` | `results/analysis_outputs/pooled_baseline_decomposition.csv` |
| Six within-diagnosis site models | `analysis/verify_models.py` | `results/analysis_outputs/adjusted_site_signature_results.csv` |
| Site composition | `analysis/run-code/weargait_site_audit_primary.py` | `results/analysis_outputs/site_diagnosis_composition.csv` |
| Pooled/VA/HurriedPace gait CV | `analysis/run-code/weargait_site_audit_primary.py` | `results/analysis_outputs/nested_cv_results.csv` |
| Transport to VA-Seattle | `analysis/run-code/weargait_site_audit_primary.py` | `results/analysis_outputs/jhu_to_va_transport.csv` |
| VA participant-level bootstrap | `analysis/run-code/weargait_site_audit_additional.py` | `results/analysis_outputs/va_participant_uncertainty.csv` |
| Gait site prediction | `analysis/run-code/weargait_site_audit_primary.py` | `results/analysis_outputs/site_signature_results.csv` |

The programmatic 11-model comparison covers pooled and within-diagnosis classification results. **Tables 6 and 7** in `results/manuscript/` contain the numerical statistics reported in the article and are provided as reference tables rather than asserted outputs from the 11-model verification script.

**Important AUC distinction:** VA-Seattle participant-averaged repeated OOF AUC (0.668) is a different estimator from the mean of repeat-wise AUCs (0.598). The manuscript's 0.668 is listed in the participant-bootstrap output file.

## Data management

Input CSVs remain outside the repository and are obtained from the independent Synapse dataset provider. Model runs write to a private temporary working folder, including potential participant-level products. Only non-identifying aggregate tables belong on GitHub.
