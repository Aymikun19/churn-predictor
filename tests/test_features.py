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


def test_contributions_explain_the_log_odds():
    import joblib
    import numpy as np
    from pathlib import Path
    from src.churn.data import load
    from src.churn.explain import contributions
    from src.churn.features import add_features

    bundle = joblib.load(Path(__file__).parents[1] / "models" / "churn_model.joblib")
    df = load().drop(columns=["Churn"])
    rows = df.head(5)
    c = contributions(bundle, rows, df)
    pipe = bundle["pipeline"]
    logit = pipe.decision_function(add_features(rows)[bundle["features"]])
    base = pipe.decision_function(add_features(df)[bundle["features"]]).mean()
    # Contributions are exact for a linear model: base + sum(contributions) = row log-odds.
    assert np.allclose(base + c.sum(axis=1).values, logit, atol=1e-6)
