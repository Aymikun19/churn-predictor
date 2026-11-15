import pandas as pd

from src.churn.features import add_features


def row(**kw):
    base = dict(tenure=10, MonthlyCharges=50.0, TotalCharges=480.0, OnlineSecurity="Yes", OnlineBackup="No",
                DeviceProtection="No internet service", TechSupport="Yes", StreamingTV="No", StreamingMovies="Yes")
    base.update(kw)
    return pd.DataFrame([base])


def test_counts_services():
    assert add_features(row())["num_services"].iloc[0] == 3


def test_avg_spend_handles_new_customers():
    out = add_features(row(tenure=0, TotalCharges=0.0, MonthlyCharges=70.0))
    assert out["avg_monthly_spend"].iloc[0] == 70.0
    assert out["tenure_band"].iloc[0] == "0-6m"


def test_avg_spend_regular():
    assert add_features(row())["avg_monthly_spend"].iloc[0] == 48.0
