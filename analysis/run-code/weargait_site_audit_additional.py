#!/usr/bin/env python3
"""Additional analyses for the WearGait-PD site-confounding validation audit.

This script adds the analyses requested during the final methodological review:
1. diagnosis baselines based on site and demographics;
2. diagnosis-stratified site prediction adjusted for measured covariates;
3. participant-bootstrap uncertainty for VA-only averaged out-of-fold predictions;
4. a label-permutation sensitivity analysis.

Run weargait_site_audit_primary.py first or run both scripts from the accompanying notebook.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import weargait_site_audit_primary as primary

SEED = primary.SEED
C_GRID = primary.C_GRID
OUT = primary.OUT
OUT.mkdir(parents=True, exist_ok=True)


def height_of(rec: dict[str, str]) -> float:
    return primary.to_float(rec.get("Height (in)", ""))


def weight_of(rec: dict[str, str]) -> float:
    return primary.to_float(rec.get("Weight (kg)", ""))


def years_since_diagnosis(rec: dict[str, str]) -> float:
    return primary.to_float(rec.get("Years since PD diagnosis", ""))


def dbs_status(rec: dict[str, str]) -> float:
    text = primary.clean_text(rec.get("DBS?", "")).lower()
    if text.startswith("y"):
        return 1.0
    if text.startswith("n"):
        return 0.0
    return np.nan


def build_covariates(frame: pd.DataFrame, clinical: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rec = clinical.get(row.participant_id, {})
        hy, axial, total = primary.clinical_outcomes(rec) if rec else (np.nan, np.nan, np.nan)
        rows.append({
            "participant_id": row.participant_id,
            "diagnosis": int(row.diagnosis),
            "site": row.site,
            "age": primary.age_of(rec) if rec else np.nan,
            "sex": primary.sex_of(rec) if rec else np.nan,
            "height": height_of(rec),
            "weight": weight_of(rec),
            "years_since_pd_diagnosis": years_since_diagnosis(rec),
            "hy": hy,
            "axial": axial,
            "total": total,
            "dbs": dbs_status(rec),
        })
    return pd.DataFrame(rows)


def site_design(sites: np.ndarray) -> np.ndarray:
    """Two dummy columns with Bayview as the reference category."""
    return np.column_stack([
        (sites == "JHOC").astype(float),
        (sites == "VA-Seattle").astype(float),
    ])


def make_pipeline(c: float | None = None) -> Pipeline:
    estimator = LogisticRegression(
        penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED,
        C=1.0 if c is None else c,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])


def repeated_nested_cv(X: np.ndarray, y: np.ndarray, repeats: int = 20, seed: int = SEED) -> dict[str, Any]:
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    oof = np.full((repeats, len(y)), np.nan)
    aucs: list[float] = []
    for split_idx, (train, test) in enumerate(outer.split(X, y)):
        repeat = split_idx // 5
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + split_idx)
        search = GridSearchCV(
            make_pipeline(),
            {"classifier__C": C_GRID},
            scoring="roc_auc",
            cv=inner,
            refit=True,
            n_jobs=1,
        )
        search.fit(X[train], y[train])
        oof[repeat, test] = search.predict_proba(X[test])[:, 1]
    for repeat in range(repeats):
        aucs.append(float(roc_auc_score(y, oof[repeat])))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs, ddof=1)),
        "oof": oof,
    }


def bootstrap_auc(y: np.ndarray, scores: np.ndarray, n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    values: list[float] = []
    for _ in range(n_boot):
        index = np.r_[
            rng.choice(idx0, len(idx0), replace=True),
            rng.choice(idx1, len(idx1), replace=True),
        ]
        values.append(float(roc_auc_score(y[index], scores[index])))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_auc_difference(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = SEED + 1,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    values: list[float] = []
    for _ in range(n_boot):
        index = np.r_[
            rng.choice(idx0, len(idx0), replace=True),
            rng.choice(idx1, len(idx1), replace=True),
        ]
        values.append(float(
            roc_auc_score(y[index], scores_a[index])
            - roc_auc_score(y[index], scores_b[index])
        ))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def fixed_fivefold_auc(X: np.ndarray, y: np.ndarray, seed: int, c: float = 1.0) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.full(len(y), np.nan)
    model = make_pipeline(c)
    for train, test in cv.split(X, y):
        fitted = clone(model)
        fitted.fit(X[train], y[train])
        scores[test] = fitted.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y, scores))


def main() -> None:
    dataset = primary.load_walkway()
    frame = dataset.frame
    clinical = {
        **primary.load_clinical(primary.CONTROL_CSV, "Control"),
        **primary.load_clinical(primary.PD_CSV, "PD"),
    }
    self_paced = frame[frame.task == "SelfPace"].reset_index(drop=True)
    gait = self_paced[dataset.feature_names].to_numpy(float)
    diagnosis = self_paced.diagnosis.to_numpy(int)
    sites = self_paced.site.to_numpy(str)
    covariates = build_covariates(self_paced, clinical)

    site_matrix = site_design(sites)
    age_sex = covariates[["age", "sex"]].to_numpy(float)
    site_age_sex = np.column_stack([site_matrix, age_sex])

    diagnosis_models = {
        "Site only": site_matrix,
        "Age and sex only": age_sex,
        "Site, age, and sex": site_age_sex,
        "159 gait features": gait,
        "Site, age, sex, and gait features": np.column_stack([site_age_sex, gait]),
    }
    diagnosis_rows = []
    for name, X in diagnosis_models.items():
        result = repeated_nested_cv(X, diagnosis, repeats=20)
        diagnosis_rows.append({
            "model": name,
            "n": len(diagnosis),
            "auc_mean": result["auc_mean"],
            "auc_sd": result["auc_sd"],
        })
    diagnosis_df = pd.DataFrame(diagnosis_rows)
    diagnosis_df.to_csv(OUT / "pooled_baseline_decomposition.csv", index=False)

    adjusted_rows = []
    definitions = [
        (0, "Controls", "Bayview vs VA-Seattle", ["age", "sex", "height", "weight"]),
        (1, "PD", "JHOC vs VA-Seattle", [
            "age", "sex", "height", "weight", "years_since_pd_diagnosis", "hy", "total", "dbs"
        ]),
    ]
    for diagnosis_value, label, comparison, columns in definitions:
        mask = diagnosis == diagnosis_value
        target_site = (sites[mask] == "VA-Seattle").astype(int)
        measured = covariates.loc[mask, columns].to_numpy(float)
        model_matrices = {
            "Measured covariates only": measured,
            "Gait features only": gait[mask],
            "Measured covariates plus gait": np.column_stack([measured, gait[mask]]),
        }
        for name, X in model_matrices.items():
            result = repeated_nested_cv(X, target_site, repeats=20)
            adjusted_rows.append({
                "diagnosis_group": label,
                "site_comparison": comparison,
                "n": int(mask.sum()),
                "model": name,
                "auc_mean": result["auc_mean"],
                "auc_sd": result["auc_sd"],
            })
    adjusted_df = pd.DataFrame(adjusted_rows)
    adjusted_df.to_csv(OUT / "adjusted_site_signature_results.csv", index=False)

    va = sites == "VA-Seattle"
    va_y = diagnosis[va]
    full = repeated_nested_cv(gait[va], va_y, repeats=20)
    candidate_index = dataset.feature_names.index(primary.CANDIDATE)
    candidate = repeated_nested_cv(gait[va, candidate_index].reshape(-1, 1), va_y, repeats=20)
    full_scores = np.nanmean(full["oof"], axis=0)
    candidate_scores = np.nanmean(candidate["oof"], axis=0)
    full_ci = bootstrap_auc(va_y, full_scores)
    candidate_ci = bootstrap_auc(va_y, candidate_scores)
    difference_ci = bootstrap_auc_difference(va_y, candidate_scores, full_scores)
    uncertainty = pd.DataFrame([
        {
            "model": "159-feature model",
            "mean_repeated_nested_auc": full["auc_mean"],
            "repeated_auc_sd": full["auc_sd"],
            "averaged_oof_auc": roc_auc_score(va_y, full_scores),
            "bootstrap_ci_low": full_ci[0],
            "bootstrap_ci_high": full_ci[1],
        },
        {
            "model": "Integrated Pressure %CV",
            "mean_repeated_nested_auc": candidate["auc_mean"],
            "repeated_auc_sd": candidate["auc_sd"],
            "averaged_oof_auc": roc_auc_score(va_y, candidate_scores),
            "bootstrap_ci_low": candidate_ci[0],
            "bootstrap_ci_high": candidate_ci[1],
        },
    ])
    uncertainty.to_csv(OUT / "va_participant_uncertainty.csv", index=False)
    pd.DataFrame([{
        "comparison": "Integrated Pressure %CV minus 159-feature model",
        "averaged_oof_auc_difference": roc_auc_score(va_y, candidate_scores) - roc_auc_score(va_y, full_scores),
        "bootstrap_ci_low": difference_ci[0],
        "bootstrap_ci_high": difference_ci[1],
    }]).to_csv(OUT / "va_paired_auc_difference.csv", index=False)

    rng = np.random.default_rng(SEED)
    global_values = []
    conditional_values = []
    for iteration in range(200):
        globally_permuted = rng.permutation(diagnosis)
        conditional = diagnosis.copy()
        for site in np.unique(sites):
            index = np.where(sites == site)[0]
            conditional[index] = rng.permutation(conditional[index])
        global_values.append(fixed_fivefold_auc(gait, globally_permuted, 1000 + iteration, c=1.0))
        conditional_values.append(fixed_fivefold_auc(gait, conditional, 2000 + iteration, c=1.0))

    permutation_df = pd.DataFrame([
        {
            "scheme": "Global label permutation",
            "n_permutations": 200,
            "mean_auc": np.mean(global_values),
            "sd_auc": np.std(global_values, ddof=1),
            "percentile_2_5": np.percentile(global_values, 2.5),
            "percentile_97_5": np.percentile(global_values, 97.5),
            "protocol": "Fixed C=1.0; five-fold cross-validation",
        },
        {
            "scheme": "Within-site label permutation",
            "n_permutations": 200,
            "mean_auc": np.mean(conditional_values),
            "sd_auc": np.std(conditional_values, ddof=1),
            "percentile_2_5": np.percentile(conditional_values, 2.5),
            "percentile_97_5": np.percentile(conditional_values, 97.5),
            "protocol": "Fixed C=1.0; labels permuted within physical site",
        },
    ])
    permutation_df.to_csv(OUT / "label_permutation_sensitivity.csv", index=False)

    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Site\nonly", "Age + sex", "Site +\nage + sex", "Gait\nfeatures", "Site + covariates\n+ gait"]
    axes[0].bar(labels, diagnosis_df.auc_mean, yerr=diagnosis_df.auc_sd, capsize=3)
    axes[0].axhline(0.5, linestyle="--", linewidth=1)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title("Pooled nested cross-validation")
    axes[0].tick_params(axis="x", rotation=12)

    transport = pd.read_csv(OUT / "jhu_to_va_transport.csv")
    self_paced_transport = transport[transport.task == "SelfPace"].iloc[:2]
    va_values = [full["auc_mean"], candidate["auc_mean"]] + self_paced_transport.auc.tolist()
    va_labels = ["VA-only\nfull model", "VA-only\nsingle feature", "JHU→VA\nfull model", "JHU→VA\nsingle feature"]
    low = [0, 0] + (self_paced_transport.auc - self_paced_transport.ci_low).tolist()
    high = [0, 0] + (self_paced_transport.ci_high - self_paced_transport.auc).tolist()
    axes[1].bar(va_labels, va_values, yerr=np.array([low, high]), capsize=3)
    axes[1].axhline(0.5, linestyle="--", linewidth=1)
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("Within-site and transport analyses")
    axes[1].tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(figdir / "Figure2_performance_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, (group, title) in zip(axes, [
        ("Controls", "Controls: Bayview vs VA-Seattle"),
        ("PD", "PD: JHOC vs VA-Seattle"),
    ]):
        subset = adjusted_df[adjusted_df.diagnosis_group == group]
        labels = ["Covariates\nonly", "Gait\nonly", "Covariates\n+ gait"]
        ax.bar(labels, subset.auc_mean, yerr=subset.auc_sd, capsize=3)
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel("ROC AUC")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figdir / "Figure3_adjusted_site_predictability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(permutation_df))
    means = permutation_df.mean_auc.to_numpy(float)
    lower = means - permutation_df.percentile_2_5.to_numpy(float)
    upper = permutation_df.percentile_97_5.to_numpy(float) - means
    ax.errorbar(means, positions, xerr=np.vstack([lower, upper]), fmt="o", capsize=4)
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_yticks(positions, permutation_df.scheme)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Cross-validated ROC AUC")
    ax.set_title("Label-permutation sensitivity analysis")
    fig.tight_layout()
    fig.savefig(figdir / "FigureS2_label_permutation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "diagnosis_baselines": diagnosis_df.to_dict(orient="records"),
        "adjusted_site_signatures": adjusted_df.to_dict(orient="records"),
        "va_uncertainty": uncertainty.to_dict(orient="records"),
        "permutation": permutation_df.to_dict(orient="records"),
    }
    (OUT / "revision2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote additional revision analyses to", OUT)


if __name__ == "__main__":
    main()
