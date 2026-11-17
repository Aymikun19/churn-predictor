"""Streamlit app: score, explain and batch-score customer churn risk.  Run with:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.churn.data import clean, load
from src.churn.explain import contributions, top_drivers
from src.churn.features import add_features

ROOT = Path(__file__).parent
LOGO = ROOT / "assets" / "logo.png"
MODEL = ROOT / "models" / "churn_model.joblib"
METRICS = ROOT / "reports" / "metrics.json"
SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

st.set_page_config(page_title="Churn Risk Scorer", page_icon=str(LOGO), layout="wide")
st.logo(str(LOGO), size="large")

if not MODEL.exists():
    st.error("Model not found. Run `python -m src.churn.train` first.")
    st.stop()


@st.cache_resource
def bundle():
    return joblib.load(MODEL)


@st.cache_data
def population() -> pd.DataFrame:
    return load()


B = bundle()
POP = population()
REF = POP.drop(columns=["Churn"])


def score(df: pd.DataFrame) -> pd.Series:
    return pd.Series(B["pipeline"].predict_proba(add_features(df)[B["features"]])[:, 1], index=df.index)


def customer(**kw) -> pd.DataFrame:
    no_net = kw["internet"] == "No"
    row = {
        "gender": "Female", "SeniorCitizen": kw["senior"], "Partner": kw["partner"], "Dependents": kw["dependents"],
        "tenure": kw["tenure"], "PhoneService": "Yes", "MultipleLines": "No", "InternetService": kw["internet"],
        "Contract": kw["contract"], "PaperlessBilling": kw["paperless"], "PaymentMethod": kw["payment"],
        "MonthlyCharges": kw["monthly"], "TotalCharges": kw["monthly"] * kw["tenure"],
    }
    for s in SERVICES:
        row[s] = "No internet service" if no_net else ("Yes" if s in kw["addons"] else "No")
    return pd.DataFrame([row])


st.title("Customer Churn Risk")
st.caption(
    f"Logistic-regression pipeline trained on {len(POP):,} IBM Telco customers. "
    f"Customers scoring above the tuned threshold of {B['threshold']:.0%} are flagged as high risk."
)

tab_score, tab_whatif, tab_batch, tab_model = st.tabs(["Score a customer", "What-if analysis", "Batch scoring", "Model performance"])

with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 5)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 80.0)
    senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    addons = st.multiselect("Add-on services", SERVICES, ["StreamingTV"])

profile = dict(tenure=tenure, contract=contract, internet=internet, payment=payment, monthly=monthly,
               senior=senior, partner=partner, dependents=dependents, paperless=paperless, addons=addons)
row = customer(**profile)
p = float(score(row).iloc[0])
high = p >= B["threshold"]
avg = float(score(REF).mean())

with tab_score:
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{p:.0%}", "High risk" if high else "Low risk", delta_color="inverse" if high else "normal")
    c2.metric("Average customer", f"{avg:.0%}")
    c3.metric("Riskier than", f"{(score(REF) < p).mean():.0%}", help="Share of customers in the training data with a lower score")
    st.progress(p)

    st.subheader("Why this score")
    st.caption("Each bar is how much a feature moves this customer's log-odds of churning compared with the average customer. Red pushes risk up, blue pulls it down.")
    drivers = top_drivers(contributions(B, row, REF).iloc[0], n=8).iloc[::-1]
    drivers["direction"] = drivers["effect"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    st.vega_lite_chart(
        drivers,
        {
            "mark": {"type": "bar", "cornerRadius": 3},
            "encoding": {
                "y": {"field": "feature", "type": "nominal", "sort": None, "title": None, "axis": {"labelOverlap": False, "labelLimit": 220, "labelFontSize": 13}},
                "x": {"field": "effect", "type": "quantitative", "title": "Effect on log-odds"},
                "color": {"field": "direction", "type": "nominal", "scale": {"domain": ["Raises risk", "Lowers risk"], "range": ["#dc2626", "#2563eb"]}, "legend": {"orient": "bottom", "title": None}},
                "tooltip": [{"field": "feature"}, {"field": "effect", "format": "+.2f"}],
            },
            "height": {"step": 36},
        },
        width="stretch",
    )

    if high:
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount to move to a 1-year contract.")
        if payment == "Electronic check":
            tips.append("Encourage switching to automatic payments.")
        if "TechSupport" not in addons and internet != "No":
            tips.append("Bundle free tech support for 3 months.")
        if tenure < 6:
            tips.append("Schedule an onboarding check-in call.")
        st.subheader("Suggested retention actions")
        for t in tips or ["Reach out with a personalised loyalty offer."]:
            st.markdown(f"- {t}")

with tab_whatif:
    st.subheader("How would the score change?")
    st.caption("Everything else about the customer stays as set in the sidebar.")
    months = list(range(0, 73, 3))
    curve = pd.DataFrame({"Tenure (months)": months, "Churn probability": [float(score(customer(**{**profile, "tenure": m})).iloc[0]) for m in months]})
    st.markdown("**Tenure**")
    st.line_chart(curve, x="Tenure (months)", y="Churn probability", height=260)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Contract type**")
        con = pd.DataFrame({"Contract": ["Month-to-month", "One year", "Two year"]})
        con["Churn probability"] = [float(score(customer(**{**profile, "contract": c})).iloc[0]) for c in con["Contract"]]
        st.bar_chart(con, x="Contract", y="Churn probability", height=240)
    with cols[1]:
        st.markdown("**Payment method**")
        pay = pd.DataFrame({"Payment": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]})
        pay["Churn probability"] = [float(score(customer(**{**profile, "payment": m})).iloc[0]) for m in pay["Payment"]]
        st.bar_chart(pay, x="Payment", y="Churn probability", height=240)

    best = min(
        [("contract", c) for c in ["One year", "Two year"]] + [("payment", m) for m in ["Bank transfer (automatic)", "Credit card (automatic)"]],
        key=lambda kv: float(score(customer(**{**profile, kv[0]: kv[1]})).iloc[0]),
    )
    p_best = float(score(customer(**{**profile, best[0]: best[1]})).iloc[0])
    if p_best < p:
        st.success(f"The single biggest lever: switching {best[0]} to **{best[1]}** lowers the probability from {p:.0%} to {p_best:.0%}.")

with tab_batch:
    st.subheader("Score a whole customer list")
    st.caption("Upload a CSV in the IBM Telco format (the same columns as the training data), or try it on a sample.")
    up = st.file_uploader("Customer CSV", type="csv")
    use_sample = st.toggle("Use a sample of 500 customers from the training data", value=up is None)
    raw = pd.read_csv(up) if up is not None else (pd.read_csv(ROOT / "data" / "telco_churn.csv").sample(500, random_state=7) if use_sample else None)
    if raw is not None:
        needed = set(B["features"]) - {"num_services", "avg_monthly_spend", "tenure_band"}
        missing = needed - set(raw.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
        else:
            ids = raw["customerID"] if "customerID" in raw else pd.Series(range(1, len(raw) + 1), index=raw.index)
            df = clean(raw)
            probs = score(df)
            out = pd.DataFrame({"customerID": ids.values, "churn_probability": probs.round(4).values, "high_risk": (probs >= B["threshold"]).values})
            out = pd.concat([out, raw.drop(columns=["customerID"], errors="ignore").reset_index(drop=True)], axis=1).sort_values("churn_probability", ascending=False)
            m1, m2, m3 = st.columns(3)
            m1.metric("Customers scored", f"{len(out):,}")
            m2.metric("Flagged high risk", f"{int(out['high_risk'].sum()):,}", f"{out['high_risk'].mean():.0%} of the list", delta_color="off")
            m3.metric("Monthly revenue at risk", f"${out.loc[out['high_risk'], 'MonthlyCharges'].sum():,.0f}")
            hist = pd.cut(out["churn_probability"], bins=[i / 10 for i in range(11)], include_lowest=True).value_counts().sort_index()
            st.bar_chart(pd.DataFrame({"Probability band": [f"{int(b.left * 100)}-{int(b.right * 100)}%" for b in hist.index], "Customers": hist.values}), x="Probability band", y="Customers", height=220)
            st.markdown("**Highest-risk customers**")
            st.dataframe(out.head(25)[["customerID", "churn_probability", "Contract", "tenure", "MonthlyCharges", "PaymentMethod"]], hide_index=True, width="stretch",
                         column_config={"churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="percent")})
            st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv", type="primary")

with tab_model:
    st.subheader("How good is the model?")
    if METRICS.exists():
        m = json.loads(METRICS.read_text())
        t = m["test"]
        a, b, c, d = st.columns(4)
        a.metric("ROC-AUC (test)", f"{t['roc_auc']:.3f}")
        b.metric("PR-AUC (test)", f"{t['pr_auc']:.3f}")
        c.metric("Churner recall", f"{t['report']['1']['recall']:.0%}")
        d.metric("Churner precision", f"{t['report']['1']['precision']:.0%}")
        cv = pd.DataFrame([{"Model": k.replace("_", " ").title(), "CV ROC-AUC": v["cv_roc_auc_mean"], "± std": v["cv_roc_auc_std"]} for k, v in m["cv"].items()])
        st.markdown("**5-fold cross-validation**")
        st.dataframe(cv, hide_index=True, width="stretch")
    figs = ROOT / "reports" / "figures"
    cols = st.columns(2)
    for col, (name, cap) in zip(cols, [("roc_curve.png", "ROC curve on the held-out test set"), ("churn_drivers.png", "Strongest churn drivers")]):
        if (figs / name).exists():
            col.image(str(figs / name), caption=cap, width="stretch")
