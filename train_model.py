"""Train, tune, evaluate and save the predictive-maintenance models.

Run:  python train_model.py
Outputs (all real, computed here - nothing is hard-coded):
  models/predictive_maintenance_model.joblib   selected pipeline + threshold
  outputs/model_comparison.csv                 4-model comparison table
  outputs/evaluation_metrics.json              full metrics + confusion matrices
  outputs/feature_importance.csv               permutation importance
  outputs/dataset_stats.json                   historical dataset statistics
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, fbeta_score,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import core

SEED = 42
TEST_SIZE = 0.20
F_BETA = 2          # recall counts 4x more than precision (missed failure is costlier than a false alert)
DATA_PATH = core.BASE_DIR / "data" / "ai4i2020.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pm.train")


def build_pipeline(estimator) -> Pipeline:
    """Feature engineering -> scaling/encoding -> classifier. Fitted on training data only."""
    prep = ColumnTransformer(
        [("num", StandardScaler(), core.NUMERIC_MODEL),
         ("cat", OneHotEncoder(categories=[core.TYPE_VALUES]), [core.CATEGORICAL])],
        verbose_feature_names_out=False)
    return Pipeline([("features", FunctionTransformer(core.add_features)),
                     ("prep", prep),
                     ("clf", estimator)])


def model_spaces():
    """The four required models and their hyper-parameter grids."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                                {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                          {"clf__max_depth": [3, 4, 5, 6, 8, 10], "clf__min_samples_leaf": [1, 5, 20, 50]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                 n_jobs=1, random_state=SEED),
                          {"clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [1, 5]}),
        "ANN (MLP)": (MLPClassifier(early_stopping=True, max_iter=500, random_state=SEED),
                      {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4, 1e-2],
                       "clf__learning_rate_init": [0.001, 0.01]}),
    }


def metrics_at(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=F_BETA, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, prob):
    """Threshold that maximises F-beta on out-of-fold training predictions."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    prec, rec = prec[:-1], rec[:-1]
    b2 = F_BETA ** 2
    f = (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)
    return float(thr[int(np.argmax(f))])


def dataset_stats(df: pd.DataFrame) -> dict:
    y = df[core.TARGET]
    bins = [-1, 50, 100, 150, 200, 260]
    labels = ["0-50", "51-100", "101-150", "151-200", "201+"]
    wear = pd.cut(df["Tool wear [min]"], bins=bins, labels=labels)
    by_wear = df.groupby(wear, observed=True)[core.TARGET].agg(["count", "sum", "mean"])
    by_type = df.groupby("Type")[core.TARGET].agg(["count", "sum", "mean"])
    return {
        "total_records": int(len(df)),
        "failures": int(y.sum()),
        "non_failures": int((y == 0).sum()),
        "failure_rate": float(y.mean()),
        "failure_modes": {m: int(df[m].sum()) for m in core.LEAKAGE_COLUMNS},
        "by_type": {t: {"records": int(r["count"]), "failures": int(r["sum"]), "rate": float(r["mean"])}
                    for t, r in by_type.iterrows()},
        "by_tool_wear": {str(k): {"records": int(r["count"]), "failures": int(r["sum"]), "rate": float(r["mean"])}
                         for k, r in by_wear.iterrows()},
        "operating_stats": {c: {k: float(v) for k, v in df[c].describe()[["mean", "std", "min", "50%", "max"]].items()}
                            for c in core.RAW_NUMERIC},
        "failed_vs_ok_means": {c: {"ok": float(df.loc[y == 0, c].mean()), "failed": float(df.loc[y == 1, c].mean())}
                               for c in core.RAW_NUMERIC},
    }


def main():
    df = pd.read_csv(DATA_PATH)
    log.info("Loaded %s rows x %s columns", *df.shape)
    assert not df.duplicated().any() and df[core.RAW_FEATURES + [core.TARGET]].notna().all().all()

    X, y = df[core.RAW_FEATURES], df[core.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    log.info("Train %s (failures %d) | Test %s (failures %d)", X_tr.shape, y_tr.sum(), X_te.shape, y_te.sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = [], {}
    for name, (est, grid) in model_spaces().items():
        log.info("Tuning %s ...", name)
        gs = GridSearchCV(build_pipeline(est), grid, scoring="average_precision", cv=cv, n_jobs=1, refit=True)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        oof = cross_val_predict(build_pipeline(est).set_params(**gs.best_params_), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = best_threshold(y_tr, oof)
        prob_te = best.predict_proba(X_te)[:, 1]
        row = {
            "model": name,
            "best_params": {k.replace("clf__", ""): (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()},
            "cv_pr_auc_mean": float(gs.best_score_),
            "cv_pr_auc_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
            "cv_f2_at_tuned_threshold": float(metrics_at(y_tr, oof, thr)["f2"]),
            "roc_auc": float(roc_auc_score(y_te, prob_te)),
            "pr_auc": float(average_precision_score(y_te, prob_te)),
            "test_default_threshold": metrics_at(y_te, prob_te, 0.5),
            "test_tuned_threshold": metrics_at(y_te, prob_te, thr),
        }
        results.append(row)
        fitted[name] = best
        log.info("  %s | CV PR-AUC %.3f | tuned thr %.3f | test recall %.3f precision %.3f", name,
                 row["cv_pr_auc_mean"], thr, row["test_tuned_threshold"]["recall"], row["test_tuned_threshold"]["precision"])

    # ---- model selection uses TRAINING cross-validation only (test set stays untouched) ----
    chosen = max(results, key=lambda r: (r["cv_f2_at_tuned_threshold"], r["cv_pr_auc_mean"]))
    name, final = chosen["model"], fitted[chosen["model"]]
    thr = chosen["test_tuned_threshold"]["threshold"]
    log.info("Selected model (by CV F2 at tuned threshold): %s", name)

    # ---- explainability: permutation importance on the held-out test set ----
    imp_rows = []
    r = permutation_importance(final, X_te, y_te, scoring="average_precision", n_repeats=10, random_state=SEED, n_jobs=1)
    for i, c in enumerate(core.RAW_FEATURES):
        imp_rows.append({"level": "raw input", "feature": c, "importance_mean": r.importances_mean[i], "importance_std": r.importances_std[i]})
    pre, clf = final[:-1], final[-1]
    Xt = pre.transform(X_te)
    names = list(pre[-1].get_feature_names_out()) if hasattr(pre[-1], "get_feature_names_out") else list(range(Xt.shape[1]))
    r2 = permutation_importance(clf, Xt, y_te, scoring="average_precision", n_repeats=10, random_state=SEED, n_jobs=1)
    for i, c in enumerate(names):
        imp_rows.append({"level": "engineered / encoded", "feature": c, "importance_mean": r2.importances_mean[i], "importance_std": r2.importances_std[i]})
    imp = pd.DataFrame(imp_rows).sort_values(["level", "importance_mean"], ascending=[True, False])

    # ---- persist everything ----
    core.OUTPUT_DIR.mkdir(exist_ok=True)
    core.MODEL_PATH.parent.mkdir(exist_ok=True)
    medians = {c: float(X_tr[c].median()) for c in core.RAW_NUMERIC}
    medians["Type"] = X_tr["Type"].mode()[0]
    artifact = {
        "model": final, "threshold": thr, "model_name": name, "raw_features": core.RAW_FEATURES,
        "training_ranges": {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in core.RAW_NUMERIC},
        "medians": medians, "random_state": SEED, "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(artifact, core.MODEL_PATH, compress=3)

    table = pd.DataFrame([{
        "model": r["model"], "cv_pr_auc": r["cv_pr_auc_mean"], "threshold": r["test_tuned_threshold"]["threshold"],
        "accuracy": r["test_tuned_threshold"]["accuracy"], "precision": r["test_tuned_threshold"]["precision"],
        "recall": r["test_tuned_threshold"]["recall"], "f1": r["test_tuned_threshold"]["f1"], "f2": r["test_tuned_threshold"]["f2"],
        "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
        "false_negatives": r["test_tuned_threshold"]["confusion_matrix"]["fn"],
        "false_positives": r["test_tuned_threshold"]["confusion_matrix"]["fp"]} for r in results]).round(4)
    table.to_csv(core.OUTPUT_DIR / "model_comparison.csv", index=False)
    imp.round(6).to_csv(core.OUTPUT_DIR / "feature_importance.csv", index=False)
    evaluation = {
        "selected_model": name, "selection_rule": "highest 5-fold CV F2 at the tuned threshold on the training set (test set not used for selection)",
        "f_beta": F_BETA, "random_state": SEED, "test_size": TEST_SIZE,
        "train_records": int(len(X_tr)), "test_records": int(len(X_te)),
        "test_failures": int(y_te.sum()), "test_failure_rate": float(y_te.mean()),
        "majority_class_accuracy": float(1 - y_te.mean()),
        "models": results, "risk_rule": {"high": ">= threshold", "medium": ">= threshold / 2", "low": "otherwise"},
        "trained_at": artifact["trained_at"],
    }
    with open(core.OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, indent=2)
    with open(core.OUTPUT_DIR / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(dataset_stats(df), fh, indent=2)
    log.info("Saved artifact (%.1f MB) and outputs.", core.MODEL_PATH.stat().st_size / 1e6)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
