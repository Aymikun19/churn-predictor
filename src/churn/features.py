"""Feature engineering shared by training, the notebook and the Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", *SERVICE_COLS, "Contract", "PaperlessBilling", "PaymentMethod", "tenure_band",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["num_services"] = (out[SERVICE_COLS] == "Yes").sum(axis=1)
    # Average real spend per month; for tenure 0 fall back to the current monthly charge.
    out["avg_monthly_spend"] = np.where(out["tenure"] > 0, out["TotalCharges"] / out["tenure"].clip(lower=1), out["MonthlyCharges"])
    out["tenure_band"] = pd.cut(
        out["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"]
    ).astype(str)
    return out


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ]
    )
