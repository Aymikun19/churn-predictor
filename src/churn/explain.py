"""Explain logistic-regression churn scores.

For a linear model the log-odds is a sum of per-column terms (coefficient x transformed value), so a
customer's score can be decomposed exactly. Each term is measured against the average customer, then
one-hot columns are added back up under the original feature they came from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_features

PRETTY = {
    "tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "num_services": "Number of add-ons", "avg_monthly_spend": "Average monthly spend", "gender": "Gender",
    "SeniorCitizen": "Senior citizen", "Partner": "Partner", "Dependents": "Dependents",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "tenure_band": "Tenure band",
}


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


def _source(name: str, features: list[str]) -> str:
    """Map a transformed column name like 'cat__Contract_Two year' back to 'Contract'."""
    bare = name.split("__", 1)[-1]
    for f in sorted(features, key=len, reverse=True):
        if bare == f or bare.startswith(f + "_"):
            return f
    return bare


def contributions(bundle: dict, rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Per-feature log-odds contribution for each row, relative to the reference population's mean."""
    pipe = bundle["pipeline"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    feats = bundle["features"]
    Xr = _dense(prep.transform(add_features(rows)[feats]))
    base = _dense(prep.transform(add_features(reference)[feats])).mean(axis=0)
    terms = (Xr - base) * model.coef_[0]
    names = [_source(n, feats) for n in prep.get_feature_names_out()]
    df = pd.DataFrame(terms, columns=names)
    return df.T.groupby(level=0).sum().T.rename(columns=PRETTY)


def top_drivers(contrib: pd.Series, n: int = 6) -> pd.DataFrame:
    """The n features pushing the score most, in either direction."""
    s = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(n)
    return pd.DataFrame({"feature": s.index, "effect": s.values})
