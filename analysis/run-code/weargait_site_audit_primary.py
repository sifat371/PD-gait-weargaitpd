#!/usr/bin/env python3
"""Corrected three-site analysis for the WearGait-PD pressure-walkway study.

The original WearGait-PD data descriptor identifies three acquisition sites:
- NLS: Johns Hopkins Outpatient Center (JHOC)
- HC: Johns Hopkins Bayview Campus (Bayview)
- WPD / WHC: VA Puget Sound (VA-Seattle)

In the pressure-walkway subset used here, JHOC contributes PD participants only,
Bayview contributes controls only, and VA-Seattle contributes both groups. This
script therefore distinguishes optimistic pooled analyses from the only
within-site case-control analysis (VA-Seattle) and from Johns Hopkins-to-VA transport.
"""
from __future__ import annotations

import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

def locate_input(name: str) -> Path:
    candidates = [SCRIPT_DIR.parent / "private-inputs"]
    for folder in candidates:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Required input file not found: {name}. Place it beside the script or in /mnt/data.")

OUT = SCRIPT_DIR.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
WALKWAY_CSV = locate_input("PKMAS Walkway Gait Metrics - HP+SP(10).csv")
CONTROL_CSV = locate_input("CONTROLS - Demographic+Clinical - datasetV1(8).csv")
PD_CSV = locate_input("PD - Demographic+Clinical - datasetV1(8).csv")
SEED = 42
C_GRID = [0.03, 0.1, 0.3, 1.0]
CANDIDATE = "Integ. Pressure (p x sec.)__%CV"


def clean_text(v: Any) -> str:
    return str(v or "").strip()


def to_float(v: Any) -> float:
    s = clean_text(v)
    if not s or s.lower() in {"nan", "na", "n/a", "none", "-", "--"}:
        return np.nan
    try:
        return float(s.replace(",", ""))
    except Exception:
        return np.nan


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    temp = np.empty(len(ranked))
    temp[order] = adjusted
    q[valid] = temp
    return q


def site_from_id(pid: str) -> str:
    if pid.startswith("NLS"):
        return "JHOC"
    if pid.startswith("HC"):
        return "Bayview"
    if pid.startswith(("WPD", "WHC")):
        return "VA-Seattle"
    return "Unknown"


@dataclass
class DataSet:
    frame: pd.DataFrame
    feature_names: list[str]


def load_walkway() -> DataSet:
    with WALKWAY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    h0, h1 = rows[0], rows[1]
    names = []
    for i, (a, b) in enumerate(zip(h0, h1)):
        a, b = clean_text(a), clean_text(b)
        if i < 3:
            names.append(a or b)
        elif a and b:
            names.append(f"{a}__{b}")
        else:
            names.append(a or b or f"column_{i}")

    records = []
    seen = set()
    for source_row, row in enumerate(rows[2:], start=3):
        row = row + [""] * max(0, len(names) - len(row))
        task = clean_text(row[0])
        original_id = clean_text(row[1])
        label_text = clean_text(row[2])
        if task not in {"SelfPace", "HurriedPace"}:
            continue
        pid = re.sub(r"_\d+$", "", original_id)
        key = (pid, task)
        if key in seen:
            continue
        seen.add(key)
        rec = {
            "participant_id": pid,
            "original_id": original_id,
            "task": task,
            "diagnosis": 1 if label_text == "PD" else 0,
            "site": site_from_id(pid),
            "source_row": source_row,
        }
        for name, raw in zip(names[3:], row[3:]):
            rec[name] = to_float(raw)
        records.append(rec)
    df = pd.DataFrame(records)
    raw_features = names[3:]
    sample_cols = [c for c in raw_features if "#Samples" in c]
    candidates = [c for c in raw_features if c not in sample_cols]
    missing = df[candidates].isna().mean()
    high_missing = missing[missing > 0.30].index.tolist()
    keep = [c for c in candidates if c not in high_missing]
    audit = {
        "raw_features": len(raw_features),
        "sample_count_removed": len(sample_cols),
        "high_missing_removed": high_missing,
        "retained_features": len(keep),
        "records": len(df),
    }
    (OUT / "feature_cleaning_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return DataSet(df, keep)


def load_clinical(path: Path, group: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[1]
    out = {}
    for row in rows[2:]:
        row = row + [""] * max(0, len(header) - len(row))
        pid = clean_text(row[0])
        if pid:
            rec = {header[i]: clean_text(row[i]) for i in range(len(header))}
            rec["group"] = group
            out[pid] = rec
    return out


def age_of(rec: dict[str, str]) -> float:
    return to_float(rec.get("Age", rec.get("Age (years)", "")))


def sex_of(rec: dict[str, str]) -> float:
    s = clean_text(rec.get("Sex", "") or rec.get("Gender", "")).lower()
    if s.startswith("f"):
        return 1.0
    if s.startswith("m"):
        return 0.0
    return np.nan


def clinical_outcomes(rec: dict[str, str]) -> tuple[float, float, float]:
    hy = np.nan
    for k in rec:
        if "hoehn" in k.lower() and "yahr" in k.lower():
            hy = to_float(rec[k])
            if np.isfinite(hy):
                break
    total = np.nan
    for k in rec:
        kl = k.lower().replace(" ", "")
        if "mdsupdrs" in kl and ("partiii" in kl or "part3" in kl or "iii" in kl) and "total" in kl:
            total = to_float(rec[k])
            if np.isfinite(total):
                break
    axial_items = []
    for item in ["3-9", "3-10", "3-11", "3-12", "3-13"]:
        found = np.nan
        for k in rec:
            kl = k.lower().replace("_", "-").replace(" ", "")
            if item in kl:
                found = to_float(rec[k])
                if np.isfinite(found):
                    break
        axial_items.append(found)
    axial = float(np.nansum(axial_items)) if np.all(np.isfinite(axial_items)) else np.nan
    return hy, axial, total


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED)),
    ])


def threshold_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return (
        balanced_accuracy_score(y, pred),
        tp / (tp + fn) if tp + fn else np.nan,
        tn / (tn + fp) if tn + fp else np.nan,
    )


def repeated_nested_cv(X: np.ndarray, y: np.ndarray, repeats: int = 20, seed: int = SEED) -> dict[str, Any]:
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    oof = np.full((repeats, len(y)), np.nan)
    best_cs = []
    for split_idx, (tr, te) in enumerate(outer.split(X, y)):
        rep = split_idx // 5
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + split_idx)
        gs = GridSearchCV(make_pipeline(), {"clf__C": C_GRID}, scoring="roc_auc", cv=inner, refit=True)
        gs.fit(X[tr], y[tr])
        oof[rep, te] = gs.predict_proba(X[te])[:, 1]
        best_cs.append(float(gs.best_params_["clf__C"]))
    aucs, bals, sens, specs = [], [], [], []
    for rep in range(repeats):
        p = oof[rep]
        aucs.append(roc_auc_score(y, p))
        b, se, sp = threshold_metrics(y, p)
        bals.append(b); sens.append(se); specs.append(sp)
    return {
        "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)),
        "balanced_accuracy_mean": float(np.mean(bals)),
        "sensitivity_mean": float(np.mean(sens)), "specificity_mean": float(np.mean(specs)),
        "best_C_median": float(np.median(best_cs)), "oof": oof,
    }


def fit_source(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> GridSearchCV:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(make_pipeline(), {"clf__C": C_GRID}, scoring="roc_auc", cv=inner, refit=True)
    gs.fit(X, y)
    return gs


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    for _ in range(n_boot):
        idx = np.r_[rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)]
        vals.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]).tolist())


def directionless_auc(y: np.ndarray, v: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(v)
    a = roc_auc_score(y[mask], v[mask])
    return (a, 1) if a >= 0.5 else (1 - a, -1)


def partial_spearman(df: pd.DataFrame, x: str, y: str, covars: list[str]) -> tuple[int, float, float]:
    d = df[[x, y] + covars].dropna().copy()
    # The source PD clinical CSV contains no matched MDS-UPDRS Part III total
    # values. Preserve n=0 as an unavailable result, not a regression on 0 rows.
    if len(d) < 3:
        return len(d), float("nan"), float("nan")
    rx = d[x].rank(method="average").to_numpy(float)
    ry = d[y].rank(method="average").to_numpy(float)
    C = pd.get_dummies(d[covars], drop_first=True).astype(float).to_numpy()
    if C.shape[1]:
        rx = rx - LinearRegression().fit(C, rx).predict(C)
        ry = ry - LinearRegression().fit(C, ry).predict(C)
    r, p = pearsonr(rx, ry)
    return len(d), float(r), float(p)


def main() -> None:
    data = load_walkway()
    df = data.frame
    clinical = {**load_clinical(CONTROL_CSV, "Control"), **load_clinical(PD_CSV, "PD")}

    site_table = pd.crosstab([df["task"], df["site"]], df["diagnosis"]).rename(columns={0: "Control", 1: "PD"})
    site_table.to_csv(OUT / "site_diagnosis_composition.csv")

    results = []
    transfer_rows = []
    candidate_rows = []
    site_signature_rows = []
    severity_rows = []

    for task in ["SelfPace", "HurriedPace"]:
        t = df[df.task == task].copy().reset_index(drop=True)
        X = t[data.feature_names].to_numpy(float)
        y = t.diagnosis.to_numpy(int)
        sites = t.site.to_numpy(str)
        ids = t.participant_id.to_numpy(str)

        pooled = repeated_nested_cv(X, y, repeats=20)
        results.append({"analysis": "Pooled nested CV", "task": task, "n": len(y), **{k:v for k,v in pooled.items() if k != "oof"}})

        va = sites == "VA-Seattle"
        va_full = repeated_nested_cv(X[va], y[va], repeats=20)
        results.append({"analysis": "VA-only nested CV full model", "task": task, "n": int(va.sum()), **{k:v for k,v in va_full.items() if k != "oof"}})

        # Source is the two Johns Hopkins sites; target is the only mixed site.
        jhu = ~va
        source_model = fit_source(X[jhu], y[jhu])
        p_va = source_model.predict_proba(X[va])[:, 1]
        transfer_rows.append({
            "task": task, "model": "Full 159-feature model", "source": "JHOC PD + Bayview controls",
            "target": "VA-Seattle mixed site", "n_target": int(va.sum()),
            "auc": float(roc_auc_score(y[va], p_va)),
            "ci_low": bootstrap_auc_ci(y[va], p_va)[0], "ci_high": bootstrap_auc_ci(y[va], p_va)[1],
            "best_C": float(source_model.best_params_["clf__C"]),
        })

        # Source-only feature selection. This identifies the candidate without target labels.
        source_feature_stats = []
        for j, name in enumerate(data.feature_names):
            auc, sign = directionless_auc(y[jhu], X[jhu, j])
            source_feature_stats.append((auc, j, sign, name))
        source_feature_stats.sort(reverse=True, key=lambda z: z[0])
        _, selected_j, selected_sign, selected_name = source_feature_stats[0]
        score_va = selected_sign * X[va, selected_j]
        transfer_rows.append({
            "task": task, "model": f"Source-selected single feature: {selected_name}",
            "source": "JHOC PD + Bayview controls", "target": "VA-Seattle mixed site",
            "n_target": int(va.sum()), "auc": float(roc_auc_score(y[va], score_va)),
            "ci_low": bootstrap_auc_ci(y[va], score_va)[0], "ci_high": bootstrap_auc_ci(y[va], score_va)[1],
            "best_C": np.nan,
        })

        # Prespecified candidate details.
        j = data.feature_names.index(CANDIDATE)
        auc_source, sign = directionless_auc(y[jhu], X[jhu, j])
        score_target = sign * X[va, j]
        auc_target = float(roc_auc_score(y[va], score_target))
        p_target = float(mannwhitneyu(X[va & (y == 1), j], X[va & (y == 0), j], alternative="two-sided").pvalue)
        candidate_cv = repeated_nested_cv(X[va, j].reshape(-1, 1), y[va], repeats=20)
        candidate_rows.append({
            "task": task, "feature": CANDIDATE, "source_auc": auc_source,
            "target_direct_auc": auc_target, "target_ci_low": bootstrap_auc_ci(y[va], score_target)[0],
            "target_ci_high": bootstrap_auc_ci(y[va], score_target)[1], "target_mannwhitney_p": p_target,
            "target_nested_auc_mean": candidate_cv["auc_mean"], "target_nested_auc_sd": candidate_cv["auc_sd"],
            "target_pd_minus_control_mean": float(np.nanmean(X[va & (y == 1), j]) - np.nanmean(X[va & (y == 0), j])),
            "selected_from_source_only": selected_name == CANDIDATE,
        })

        # Site predictability independent of diagnosis.
        for diagnosis, label, site0 in [(0, "Controls", "Bayview"), (1, "PD", "JHOC")]:
            mask = y == diagnosis
            site_y = (sites[mask] == "VA-Seattle").astype(int)
            sig = repeated_nested_cv(X[mask], site_y, repeats=20)
            site_signature_rows.append({
                "task": task, "diagnosis_group": label, "comparison": f"{site0} vs VA-Seattle",
                "n": int(mask.sum()), "auc_mean": sig["auc_mean"], "auc_sd": sig["auc_sd"],
                "balanced_accuracy_mean": sig["balanced_accuracy_mean"],
            })

        # Candidate site effects within diagnosis.
        for diagnosis, label, site0 in [(0, "Controls", "Bayview"), (1, "PD", "JHOC")]:
            m0 = (y == diagnosis) & (sites == site0)
            m1 = (y == diagnosis) & (sites == "VA-Seattle")
            site_auc, _ = directionless_auc(np.r_[np.zeros(m0.sum(), dtype=int), np.ones(m1.sum(), dtype=int)], np.r_[X[m0,j], X[m1,j]])
            candidate_rows.append({
                "task": task, "feature": CANDIDATE, "source_auc": np.nan,
                "target_direct_auc": np.nan, "target_ci_low": np.nan, "target_ci_high": np.nan,
                "target_mannwhitney_p": float(mannwhitneyu(X[m0,j], X[m1,j], alternative="two-sided").pvalue),
                "target_nested_auc_mean": np.nan, "target_nested_auc_sd": np.nan,
                "target_pd_minus_control_mean": float(np.nanmean(X[m1,j]) - np.nanmean(X[m0,j])),
                "selected_from_source_only": False,
                "site_effect_group": label, "site_effect_auc": site_auc,
                "site_effect_comparison": f"{site0} vs VA-Seattle",
            })

        # Save feature-level source and target screen.
        screen = []
        for j, name in enumerate(data.feature_names):
            a_src, s_src = directionless_auc(y[jhu], X[jhu,j])
            a_tgt, s_tgt = directionless_auc(y[va], X[va,j])
            p_tgt = mannwhitneyu(X[va & (y == 1),j], X[va & (y == 0),j], alternative="two-sided").pvalue
            screen.append({"feature": name, "source_auc": a_src, "source_direction": s_src,
                           "va_auc": a_tgt, "va_direction": s_tgt, "va_p": p_tgt})
        screen_df = pd.DataFrame(screen)
        screen_df["va_q"] = bh_fdr(screen_df.va_p.to_numpy(float))
        screen_df.to_csv(OUT / f"feature_screen_{task}.csv", index=False)

        # Restore the prespecified candidate index: the feature-screening loop
        # above reuses j for each feature and otherwise leaves j at Right eGVI.
        j = data.feature_names.index(CANDIDATE)

        # Clinical analysis is only meaningful for self-paced candidate in the main paper.
        if task == "SelfPace":
            pd_rows = []
            for i, pid in enumerate(ids):
                if y[i] != 1 or pid not in clinical:
                    continue
                hy, axial, total = clinical_outcomes(clinical[pid])
                pd_rows.append({
                    "participant_id": pid, "site": sites[i], "site_va": int(sites[i] == "VA-Seattle"),
                    "candidate": X[i,j], "age": age_of(clinical[pid]), "sex": sex_of(clinical[pid]),
                    "hy": hy, "axial": axial, "total": total,
                })
            pd_df = pd.DataFrame(pd_rows)
            pd_df.to_csv(OUT / "candidate_clinical_data.csv", index=False)
            for outcome in ["hy", "axial", "total"]:
                pooled_n = pd_df[["candidate", outcome]].dropna().shape[0]
                pooled_rho, pooled_p = spearmanr(pd_df.candidate, pd_df[outcome], nan_policy="omit")
                n_adj, r_adj, p_adj = partial_spearman(pd_df, "candidate", outcome, ["site_va", "age", "sex"])
                severity_rows.append({
                    "outcome": outcome, "n": pooled_n, "pooled_spearman_rho": float(pooled_rho),
                    "pooled_p": float(pooled_p), "adjusted_partial_rho": r_adj, "adjusted_p": p_adj,
                })

    results_df = pd.DataFrame(results)
    transfer_df = pd.DataFrame(transfer_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    site_sig_df = pd.DataFrame(site_signature_rows)
    severity_df = pd.DataFrame(severity_rows)
    severity_df["adjusted_q"] = bh_fdr(severity_df.adjusted_p.to_numpy(float))

    results_df.to_csv(OUT / "nested_cv_results.csv", index=False)
    transfer_df.to_csv(OUT / "jhu_to_va_transport.csv", index=False)
    candidate_df.to_csv(OUT / "candidate_results.csv", index=False)
    site_sig_df.to_csv(OUT / "site_signature_results.csv", index=False)
    severity_df.to_csv(OUT / "candidate_severity_results.csv", index=False)

    # Main figures.
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    # Figure 1: site-by-diagnosis composition.
    comp = df[df.task == "SelfPace"].groupby(["site", "diagnosis"]).size().unstack(fill_value=0).reindex(["JHOC", "Bayview", "VA-Seattle"])
    ax = comp.rename(columns={0:"Controls",1:"PD"}).plot(kind="bar", stacked=True, figsize=(7.2,4.5))
    ax.set_ylabel("Participants")
    ax.set_xlabel("Acquisition site")
    ax.set_title("Pressure-walkway subset composition")
    ax.legend(title="Group", frameon=False)
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(figdir / "Figure1_site_diagnosis_composition.png", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)

    # Figure 2: performance panels.
    sp_nested = results_df[results_df.task == "SelfPace"]
    labels_a = ["Pooled\nfull model", "VA-only\nfull model", "VA-only\nsingle feature"]
    vals_a = [
        float(sp_nested[sp_nested.analysis == "Pooled nested CV"].auc_mean.iloc[0]),
        float(sp_nested[sp_nested.analysis == "VA-only nested CV full model"].auc_mean.iloc[0]),
        float(candidate_df[(candidate_df.task == "SelfPace") & candidate_df.target_nested_auc_mean.notna()].target_nested_auc_mean.iloc[0]),
    ]
    err_a = [
        float(sp_nested[sp_nested.analysis == "Pooled nested CV"].auc_sd.iloc[0]),
        float(sp_nested[sp_nested.analysis == "VA-only nested CV full model"].auc_sd.iloc[0]),
        float(candidate_df[(candidate_df.task == "SelfPace") & candidate_df.target_nested_auc_mean.notna()].target_nested_auc_sd.iloc[0]),
    ]
    tr = transfer_df[transfer_df.task == "SelfPace"].copy()
    labels_b = ["Full model", "Source-selected\nsingle feature"]
    vals_b = tr.auc.to_list()[:2]
    low_b = [v-l for v,l in zip(vals_b,tr.ci_low.to_list()[:2])]
    high_b = [h-v for v,h in zip(vals_b,tr.ci_high.to_list()[:2])]
    fig, axes = plt.subplots(1,2,figsize=(10.2,4.4))
    axes[0].bar(labels_a, vals_a, yerr=err_a, capsize=4)
    axes[0].axhline(0.5, linestyle="--", linewidth=1)
    axes[0].set_ylim(0,1)
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title("Repeated nested cross-validation\n(mean ± SD)")
    axes[1].bar(labels_b, vals_b, yerr=np.array([low_b,high_b]), capsize=4)
    axes[1].axhline(0.5, linestyle="--", linewidth=1)
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_title("JHU discovery → VA-Seattle testing\n(95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(figdir / "Figure2_model_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: candidate distributions by actual site and diagnosis.
    sp = df[df.task == "SelfPace"].copy()
    groups = [
        ("JHOC PD", sp[(sp.site == "JHOC") & (sp.diagnosis == 1)][CANDIDATE].dropna().to_numpy()),
        ("Bayview controls", sp[(sp.site == "Bayview") & (sp.diagnosis == 0)][CANDIDATE].dropna().to_numpy()),
        ("VA PD", sp[(sp.site == "VA-Seattle") & (sp.diagnosis == 1)][CANDIDATE].dropna().to_numpy()),
        ("VA controls", sp[(sp.site == "VA-Seattle") & (sp.diagnosis == 0)][CANDIDATE].dropna().to_numpy()),
    ]
    fig, ax = plt.subplots(figsize=(8.2,4.8))
    ax.boxplot([g[1] for g in groups], labels=[g[0] for g in groups], showfliers=False)
    rng = np.random.default_rng(SEED)
    for i, (_, vals) in enumerate(groups, start=1):
        xj = i + rng.normal(0,0.045,len(vals))
        ax.scatter(xj, vals, s=16, alpha=0.65)
    ax.set_ylabel("Integrated Pressure %CV")
    ax.set_title("Candidate feature varies by both acquisition site and diagnosis")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(figdir / "Figure3_candidate_by_site_and_diagnosis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "site_composition": site_table.reset_index().to_dict(orient="records"),
        "nested_cv": results_df.to_dict(orient="records"),
        "transport": transfer_df.to_dict(orient="records"),
        "site_signature": site_sig_df.to_dict(orient="records"),
        "severity": severity_df.to_dict(orient="records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote corrected analysis outputs to", OUT)


if __name__ == "__main__":
    main()
