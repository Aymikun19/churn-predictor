"""Download and clean the IBM Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "telco_churn.csv"


def download(force: bool = False) -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if force or not RAW.exists():
        urlretrieve(URL, RAW)
    return RAW


def load() -> pd.DataFrame:
    """Load the raw CSV and fix its known quirks."""
    df = pd.read_csv(download())
    # TotalCharges is text, with blanks for brand-new customers (tenure 0). They have paid nothing yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", "0"))
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])
