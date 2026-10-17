"""Reproduce the manuscript primary and site-prediction CV models.

This harness records aggregate diagnostics without changing the estimator.
No participant records, individual predictions or fitted models are serialized.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterator
import warnings


BASE = Path(__file__).resolve().parent
os.umask(0o077)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "run-code"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


EXPECTED = {
    "PKMAS Walkway Gait Metrics - HP+SP(10).csv":
        "e39be6a358a78649c7e0174a41da6399c07f1d19c90c650c4b251d091f6eac51",
    "PD - Demographic+Clinical - datasetV1(8).csv":
        "b9a96bcde7e637f98fe505b387dc35541c1a1ff007d0a5c7cbfd6bea5d82a5da",
    "CONTROLS - Demographic+Clinical - datasetV1(8).csv":
        "7cded38b5acce9ca86a2a276af8c83eca5de8891d5b38e55849a49e6ac6fc8c5",
}
LOG = logging.getLogger("model_verification")


def write_json(path: Path, value: Any) -> None:
    """Write only the explicitly supplied aggregate result values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_hash(value: np.ndarray) -> str:
    """Fingerprint shape, dtype and array bytes without exposing values."""
    value = np.ascontiguousarray(value)
    header = json.dumps({"shape": value.shape, "dtype": value.dtype.str}).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def environment() -> dict[str, Any]:
    """Record the executing interpreter and complete installed distribution set."""
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(sorted(versions.items())),
        "thread_environment": {k: os.environ.get(k) for k in
            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        "scope": "Five SelfPace pooled matrices and six within-diagnosis site matrices",
    }


def source_statement_setup(additional: Any) -> dict[str, Any]:
    """Execute the analysis matrix setup statements, excluding main loops."""
    path = Path(additional.__file__)
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    pooled_setup = []
    definitions = None
    site_setup = None
    for statement in main.body:
        if isinstance(statement, ast.Assign):
            names = [x.id for x in statement.targets if isinstance(x, ast.Name)]
            if "definitions" in names:
                definitions = statement
        if isinstance(statement, ast.For) and isinstance(statement.iter, ast.Name):
            if statement.iter.id == "definitions":
                site_setup = []
                for inner in statement.body:
                    site_setup.append(inner)
                    if isinstance(inner, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "model_matrices"
                        for t in inner.targets
                    ):
                        break
    for statement in main.body:
        pooled_setup.append(statement)
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "diagnosis_models" for t in statement.targets
        ):
            break
    assert definitions is not None and site_setup is not None
    setup_module = ast.Module(body=pooled_setup + [definitions], type_ignores=[])
    scope = dict(vars(additional))
    exec(compile(setup_module, str(path), "exec"), scope)
    site_module = ast.Module(body=site_setup, type_ignores=[])
    site_jobs = []
    for diagnosis_value, label, comparison, columns in scope["definitions"]:
        scope.update(diagnosis_value=diagnosis_value, label=label,
                     comparison=comparison, columns=columns)
        exec(compile(site_module, str(path), "exec"), scope)
        for name, matrix in scope["model_matrices"].items():
            site_jobs.append((label, comparison, name, matrix.copy(),
                              scope["target_site"].copy(), list(columns)))
    scope["site_jobs"] = site_jobs
    (BASE / "metadata/exact_setup_statements.txt").write_text(
        ast.unparse(setup_module) + "\n\n# Site-loop matrix statements\n"
        + ast.unparse(site_module) + "\n"
    )
    return scope


def split_fingerprints(X: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Record deterministic outer-partition fingerprints and aggregate sizes."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    return [{"outer_split": i, "repeat": i // 5, "fold": i % 5,
             "train_hash": array_hash(tr), "test_hash": array_hash(te),
             "n_train": len(tr), "n_test": len(te),
             "n_train_positive": int(y[tr].sum()), "n_test_positive": int(y[te].sum())}
            for i, (tr, te) in enumerate(cv.split(X, y))]


def matrix_audit(X: np.ndarray, y: np.ndarray, module: Any) -> dict[str, Any]:
    """Check inputs and first-training-fold preprocessing without fitting an estimator."""
    assert not np.isinf(X).any()
    assert set(np.unique(y)) == {0, 1}
    tr, te = next(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42).split(X, y))
    preprocessing = module.make_pipeline()[:-1]
    transformed = preprocessing.fit_transform(X[tr], y[tr])
    held_out = preprocessing.transform(X[te])
    assert transformed.shape[1] == X.shape[1]
    assert np.isfinite(transformed).all() and np.isfinite(held_out).all()
    active = np.ptp(transformed, axis=0) > 1e-12
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(transformed.std(axis=0)[active], 1, atol=1e-10)
    all_empty = np.flatnonzero(np.isnan(X).all(axis=0)).tolist()
    for column in all_empty:
        assert np.all(transformed[:, column] == 0)
    return {
        "shape": list(X.shape), "matrix_sha256": array_hash(X),
        "target_sha256": array_hash(y), "positive": int(y.sum()),
        "negative": int((y == 0).sum()), "missing_cells": int(np.isnan(X).sum()),
        "missing_by_column": np.isnan(X).sum(axis=0).tolist(),
        "all_empty_columns": all_empty,
        "preprocessing_first_training_fold": {
            "train_rows": len(tr), "held_out_rows": len(te),
            "output_features": int(transformed.shape[1]),
            "all_finite": True, "centered": True, "nonconstant_columns_unit_sd": True,
            "constant_columns": int((~active).sum()),
            "all_empty_columns_retained_as_zero": True,
        },
        "outer_partitions": split_fingerprints(X, y),
    }


def preflight(scope: dict[str, Any], additional: Any) -> dict[str, Any]:
    """Verify the manuscript cohort and model input matrices."""
    data, frame, cov = scope["dataset"], scope["self_paced"], scope["covariates"]
    counts = pd.crosstab(frame.site, frame.diagnosis)
    assert len(frame) == 133 and frame.participant_id.is_unique
    assert int(frame.diagnosis.sum()) == 65
    assert len(data.feature_names) == 159
    assert counts.loc["JHOC"].tolist() == [0, 49]
    assert counts.loc["Bayview"].tolist() == [49, 0]
    assert counts.loc["VA-Seattle"].tolist() == [19, 16]
    expected = {"Site only": 2, "Age and sex only": 2, "Site, age, and sex": 4,
                "159 gait features": 159, "Site, age, sex, and gait features": 163}
    audits = {}
    for name, X in scope["diagnosis_models"].items():
        assert X.shape == (133, expected[name])
        audits[name] = matrix_audit(X, scope["diagnosis"], additional)
    for label, comparison, name, X, y, columns in scope["site_jobs"]:
        audits[f"{label}: {name}"] = matrix_audit(X, y, additional)
    pid_hash = hashlib.sha256("\n".join(frame.participant_id).encode()).hexdigest()
    pipeline = additional.make_pipeline()
    clf = pipeline.named_steps["classifier"]
    assert clf.penalty == "l1" and clf.solver == "liblinear"
    assert clf.random_state == 42 and clf.max_iter == 5000
    assert additional.C_GRID == [0.03, 0.1, 0.3, 1.0]
    assert additional.SEED == 42
    report = {
        "n_selfpace": len(frame), "pd": 65, "controls": 68,
        "n_hurriedpace": int(data.frame.task.eq("HurriedPace").sum()),
        "sites": counts.rename(columns={0: "Control", 1: "PD"}).to_dict(orient="index"),
        "retained_gait_features": data.feature_names,
        "complete_gait_features": int(frame[data.feature_names].notna().all().sum()),
        "participant_order_sha256": pid_hash,
        "demographic_complete_cases_diagnostic_only": int(cov[["age", "sex"]].notna().all(axis=1).sum()),
        "covariate_missingness": {c: int(cov[c].isna().sum()) for c in
            ["age", "sex", "height", "weight", "years_since_pd_diagnosis", "hy", "total", "dbs"]},
        "site_covariate_definitions": [list(x) for x in scope["definitions"]],
        "pd_total_field_finite_count": int(cov.loc[frame.diagnosis.eq(1), "total"].notna().sum()),
        "all_pooled_models_retain_133": True,
        "pipeline_repr": repr(pipeline), "classifier_params": clf.get_params(),
        "model_inputs": audits,
    }
    assert report["pd_total_field_finite_count"] == 0
    write_json(BASE / "outputs/preflight.json", report)
    LOG.info("PREFLIGHT PASSED n=133 PD=65 control=68 gait=159; demographics imputed; all pooled matrices retain 133")
    LOG.info("Missingness %s", report["covariate_missingness"])
    return report


@contextmanager
def recording_search(module: Any, name: str, n_rows: int) -> Iterator[list[dict[str, Any]]]:
    """Observe completed original GridSearchCV fits, preserving the original fit call."""
    original_class = module.GridSearchCV
    records: list[dict[str, Any]] = []

    class RecordingGridSearchCV(original_class):
        def fit(self, X: np.ndarray, y: np.ndarray | None = None,
                **params: Any) -> Any:
            split = len(records)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = super().fit(X, y, **params)
            assert self.scoring == "roc_auc"
            assert self.cv.n_splits == 3 and self.cv.shuffle
            assert self.cv.random_state == 42 + split
            assert self.param_grid == {"classifier__C": [0.03, 0.1, 0.3, 1.0]}
            warning_counts = Counter(type(w.message).__name__ for w in captured)
            messages = Counter(str(w.message) for w in captured)
            record = {
                "model": name, "outer_split": split, "repeat": split // 5,
                "fold": split % 5, "n_total": n_rows, "n_train": len(X),
                "inner_seed": int(self.cv.random_state),
                "selected_C": float(self.best_params_["classifier__C"]),
                "inner_mean_auc_by_C": {
                    str(p["classifier__C"]): float(a)
                    for p, a in zip(self.cv_results_["params"], self.cv_results_["mean_test_score"])
                },
                "best_inner_auc": float(self.best_score_),
                "fit_seconds": time.perf_counter() - start,
                "warning_counts": dict(warning_counts),
                "warning_messages": dict(messages),
                "best_model_nonzero_coefficients": int(np.count_nonzero(
                    self.best_estimator_.named_steps["classifier"].coef_)),
            }
            records.append(record)
            with (BASE / "logs/outer_fit_events.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if split % 10 == 9:
                LOG.info("FIT %s %d/100 C=%s inner_seed=%d", name, split + 1,
                         record["selected_C"], record["inner_seed"])
            return result

    module.GridSearchCV = RecordingGridSearchCV
    try:
        yield records
    finally:
        module.GridSearchCV = original_class


def run_one(module: Any, name: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Run model cross-validation and export aggregate results and C frequencies."""
    start = time.perf_counter()
    LOG.info("START %s rows=%d predictors=%d", name, len(y), X.shape[1])
    with recording_search(module, name, len(y)) as fits:
        result = module.repeated_nested_cv(X, y, repeats=20, seed=42)
    assert len(fits) == 100
    oof = result["oof"]
    assert oof.shape == (20, len(y)) and np.isfinite(oof).all()
    repeat_aucs = [float(roc_auc_score(y, p)) for p in oof]
    assert np.mean(repeat_aucs) == result["auc_mean"]
    assert np.std(repeat_aucs, ddof=1) == result["auc_sd"]
    warnings_total: Counter[str] = Counter()
    for fit in fits:
        warnings_total.update(fit["warning_counts"])
    summary = {
        "model": name, "n": len(y), "n_positive": int(y.sum()),
        "n_predictors": X.shape[1], "auc_mean": float(result["auc_mean"]),
        "auc_sd": float(result["auc_sd"]), "repeat_aucs": repeat_aucs,
        "participant_average_oof_auc_diagnostic": float(roc_auc_score(y, oof.mean(axis=0))),
        "selected_C_frequencies": {str(c): sum(f["selected_C"] == c for f in fits)
                                   for c in module.C_GRID},
        "matrix_sha256": array_hash(X), "target_sha256": array_hash(y),
        "oof_array_sha256_only": array_hash(oof),
        "warning_counts": dict(warnings_total), "elapsed_seconds": time.perf_counter() - start,
    }
    LOG.info("DONE %s AUC=%.15f SD=%.15f Cs=%s seconds=%.2f", name,
             summary["auc_mean"], summary["auc_sd"], summary["selected_C_frequencies"],
             summary["elapsed_seconds"])
    return summary


def main() -> None:
    """Verify inputs first, then run only original pooled and site CV protocols."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(BASE / "logs/execution.log", mode="a")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
    for filename, expected in EXPECTED.items():
        actual = hashlib.sha256((BASE / "private-inputs" / filename).read_bytes()).hexdigest()
        assert actual == expected, filename
    LOG.info("All three copied CSV checksums verified before analysis module import")
    import weargait_site_audit_additional as additional
    import weargait_site_audit_primary as primary
    assert Path(additional.__file__).parent == BASE / "run-code"
    assert Path(primary.__file__).parent == BASE / "run-code"
    assert primary.OUT.resolve() == (BASE / "outputs").resolve()
    assert all(p.parent.resolve() == (BASE / "private-inputs").resolve()
               for p in [primary.WALKWAY_CSV, primary.CONTROL_CSV, primary.PD_CSV])
    write_json(BASE / "metadata/environment.json", environment())
    scope = source_statement_setup(additional)
    preflight(scope, additional)
    if args.preflight_only:
        return
    assert not (BASE / "outputs/verification_results.json").exists(), "Do not overwrite a completed run"
    assert not (BASE / "logs/outer_fit_events.jsonl").exists(), "Do not mix execution attempts"
    results: dict[str, Any] = {"pooled": [], "richer_covariate_site_prediction": []}
    for name, X in scope["diagnosis_models"].items():
        result = run_one(additional, name, X, scope["diagnosis"])
        results["pooled"].append(result)
        write_json(BASE / "outputs/verification_results.json", results)
    for label, comparison, name, X, y, columns in scope["site_jobs"]:
        result = run_one(additional, f"{label}: {name}", X, y)
        result.update(diagnosis_group=label, comparison=comparison,
                      requested_covariates=columns)
        results["richer_covariate_site_prediction"].append(result)
        write_json(BASE / "outputs/verification_results.json", results)
    for key, filename in [("pooled", "pooled_baseline_decomposition.csv"),
                          ("richer_covariate_site_prediction", "adjusted_site_signature_results.csv")]:
        pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                      for r in results[key]]).to_csv(BASE / "outputs" / filename, index=False)
    LOG.info("COMPLETE 11 original-protocol models; 1100 outer refits; raw participant predictions not written")


if __name__ == "__main__":
    main()
