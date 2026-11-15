"""Train, compare and save churn models.

Usage:  python -m src.churn.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, average_precision_score, classification_report, f1_score,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .data import load
from .features import CATEGORICAL, NUMERIC, add_features, preprocessor

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def candidates():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, class_weight="balanced", random_state=SEED),
    }


def best_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1 on the training folds."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def main() -> None:
    df = add_features(load())
    X, y = df[NUMERIC + CATEGORICAL], df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in candidates().items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {"cv_roc_auc_mean": round(scores.mean(), 4), "cv_roc_auc_std": round(scores.std(), 4)}
        print(f"{name:22s} ROC-AUC {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    best = Pipeline([("prep", preprocessor()), ("model", candidates()[best_name])]).fit(X_train, y_train)
    # Tune the decision threshold on out-of-fold predictions so the test set stays untouched.
    oof = cross_val_predict(Pipeline([("prep", preprocessor()), ("model", candidates()[best_name])]), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    test = {
        "model": best_name,
        "threshold": round(threshold, 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "report": classification_report(y_test, pred, output_dict=True),
        "baseline_churn_rate": round(float(y.mean()), 4),
    }
    print(f"\nBest: {best_name} | test ROC-AUC {test['roc_auc']} | F1 {test['f1']} @ threshold {test['threshold']}")

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": best, "threshold": threshold, "features": NUMERIC + CATEGORICAL}, ROOT / "models" / "churn_model.joblib")
    reports = ROOT / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps({"cv": results, "test": test}, indent=2))

    # Figures: ROC curve + top drivers (logistic regression coefficients for interpretability).
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("ROC curve (held-out test set)")
    fig.tight_layout()
    fig.savefig(reports / "figures" / "roc_curve.png", dpi=150)

    lr = Pipeline([("prep", preprocessor()), ("model", candidates()["logistic_regression"])]).fit(X_train, y_train)
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["model"].coef_[0], index=[n.split("__", 1)[1] for n in names]).sort_values()
    top = pd.concat([coefs.head(8), coefs.tail(8)])
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax, color=["#16a34a" if v < 0 else "#dc2626" for v in top])
    ax.set_title("Churn drivers (red = more churn)")
    ax.axvline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "churn_drivers.png", dpi=150)
    print("Saved model, metrics and figures.")


if __name__ == "__main__":
    main()
