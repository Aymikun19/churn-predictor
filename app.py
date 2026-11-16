"""Streamlit app: score a customer's churn risk.  Run with:  streamlit run app.py"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

LOGO = Path(__file__).parent / "assets" / "logo.png"

from src.churn.features import add_features

MODEL = Path(__file__).parent / "models" / "churn_model.joblib"

st.set_page_config(page_title="Churn Risk Scorer", page_icon=str(LOGO), layout="centered")
st.logo(str(LOGO), size="large")
st.title("Customer Churn Risk")
st.caption("Logistic-regression pipeline trained on the IBM Telco dataset. See the README for metrics.")

if not MODEL.exists():
    st.error("Model not found. Run `python -m src.churn.train` first.")
    st.stop()
bundle = joblib.load(MODEL)

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 5)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 80.0)
with c2:
    senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    addons = st.multiselect("Add-on services", ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"], ["StreamingTV"])

no_net = internet == "No"
row = {
    "gender": "Female", "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": internet,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
}
for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
    row[s] = "No internet service" if no_net else ("Yes" if s in addons else "No")

X = add_features(pd.DataFrame([row]))[bundle["features"]]
p = float(bundle["pipeline"].predict_proba(X)[0, 1])
high = p >= bundle["threshold"]

st.divider()
st.metric("Churn probability", f"{p:.0%}", "HIGH RISK" if high else "low risk", delta_color="inverse" if high else "normal")
st.progress(p)
if high:
    tips = []
    if contract == "Month-to-month":
        tips.append("Offer a discount to move to a 1-year contract.")
    if payment == "Electronic check":
        tips.append("Encourage switching to automatic payments.")
    if "TechSupport" not in addons and not no_net:
        tips.append("Bundle free tech support for 3 months.")
    if tenure < 6:
        tips.append("Schedule an onboarding check-in call.")
    st.subheader("Suggested retention actions")
    for t in tips or ["Reach out with a personalised loyalty offer."]:
        st.write("•", t)
