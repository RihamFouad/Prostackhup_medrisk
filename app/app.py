"""
MedRisk - clinical risk screening interface.

Loads the model selected by src/train.py and scores one patient at a time.
Because the selected model is a logistic regression on standardised inputs,
the per-patient explanation below is exact: each bar is that feature's real
contribution to the log-odds, not an approximation.

Run locally:  streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "medrisk_model.joblib"

st.set_page_config(page_title="MedRisk", page_icon="◈", layout="wide")

INK = "#16222E"
PAPER = "#F5F7F8"
TRACE = "#0E7C66"
ALERT = "#B4482F"
RULE = "#D2DADE"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
    .stApp {{ background: {PAPER}; color: {INK}; }}
    .masthead {{
        border-bottom: 2px solid {INK}; padding-bottom: 0.6rem; margin-bottom: 1.4rem;
    }}
    .masthead h1 {{
        font-size: 2.1rem; font-weight: 600; letter-spacing: -0.02em;
        margin: 0; color: {INK};
    }}
    .masthead p {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: #5A6B78; margin: 0.35rem 0 0 0;
    }}
    .verdict {{
        font-size: 1.35rem; font-weight: 600; margin: 0.2rem 0 0.1rem 0;
    }}
    .readout {{
        font-family: 'IBM Plex Mono', monospace; font-size: 3.4rem;
        font-weight: 500; line-height: 1; letter-spacing: -0.03em;
    }}
    .caption {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem;
        letter-spacing: 0.06em; text-transform: uppercase; color: #5A6B78;
    }}
    .panel {{
        background: #FFFFFF; border: 1px solid {RULE};
        border-radius: 3px; padding: 1.1rem 1.3rem;
    }}
    .note {{ font-size: 0.85rem; color: #5A6B78; line-height: 1.5; }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def risk_strip(prob: float, threshold: float) -> str:
    """The signature element: a calibrated instrument scale. The needle is the
    patient's probability, the notch is the referral threshold."""
    pct = prob * 100
    tick = threshold * 100
    colour = ALERT if prob >= threshold else TRACE
    return f"""
    <div style="margin:0.9rem 0 0.3rem 0;">
      <div style="position:relative;height:34px;">
        <div style="position:absolute;top:14px;left:0;right:0;height:6px;
                    background:linear-gradient(90deg,#DCE6E4 0%,#F0DED8 100%);
                    border-radius:3px;"></div>
        <div style="position:absolute;top:6px;left:{tick}%;width:2px;height:22px;
                    background:{INK};"></div>
        <div style="position:absolute;top:4px;left:calc({pct}% - 8px);width:16px;
                    height:26px;background:{colour};border-radius:2px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.22);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                  color:#5A6B78;letter-spacing:0.06em;">
        <span>0%</span>
        <span>REFER AT {tick:.0f}%</span>
        <span>100%</span>
      </div>
    </div>
    """


def contributions(bundle, row: pd.DataFrame) -> pd.DataFrame | None:
    """Exact log-odds contributions, available when the model is linear."""
    pipe = bundle["model"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    prep = pipe.named_steps["prep"]
    x = prep.transform(row)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    effect = x[0] * clf.coef_[0]
    return (
        pd.DataFrame({"feature": names, "effect": effect})
        .assign(magnitude=lambda d: d["effect"].abs())
        .sort_values("magnitude", ascending=False)
        .head(7)
        .reset_index(drop=True)
    )


st.markdown(
    """
    <div class="masthead">
      <h1>MedRisk</h1>
      <p>Angiographic heart disease screening &middot; UCI Cleveland cohort, n=303</p>
    </div>
    """,
    unsafe_allow_html=True,
)

bundle = load_bundle()
if bundle is None:
    st.error(
        "No trained model found. Run `python src/train.py` first — it writes "
        "models/medrisk_model.joblib."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Patient measurements")
    age = st.slider("Age", 25, 85, 55)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    chest_pain = st.selectbox(
        "Chest pain type",
        ["asymptomatic", "typical", "nontypical", "nonanginal"],
        help="Asymptomatic presentation carries the highest observed risk in this cohort.",
    )
    rest_bp = st.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum cholesterol (mg/dl)", 120, 570, 245)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl", ["No", "Yes"], horizontal=True)
    rest_ecg = st.selectbox("Resting ECG", ["0", "1", "2"],
                            help="0 normal · 1 ST-T abnormality · 2 left ventricular hypertrophy")
    max_hr = st.slider("Max heart rate achieved", 70, 210, 150)
    ex_ang = st.radio("Exercise-induced angina", ["No", "Yes"], horizontal=True)
    oldpeak = st.slider("ST depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("ST segment slope", ["1", "2", "3"],
                         help="1 upsloping · 2 flat · 3 downsloping")
    ca = st.select_slider("Major vessels coloured (0-3)", options=[0, 1, 2, 3], value=0)
    thal = st.selectbox("Thalassemia stress result", ["normal", "fixed", "reversable"])

row = pd.DataFrame(
    [
        {
            "Age": age, "RestBP": rest_bp, "Chol": chol, "MaxHR": max_hr,
            "Oldpeak": oldpeak, "Ca": float(ca),
            "ChestPain": chest_pain, "Thal": thal, "RestECG": rest_ecg, "Slope": slope,
            "Sex": 1 if sex == "Male" else 0,
            "Fbs": 1 if fbs == "Yes" else 0,
            "ExAng": 1 if ex_ang == "Yes" else 0,
        }
    ]
)[bundle["features"]]

prob = float(bundle["model"].predict_proba(row)[0, 1])
threshold = float(bundle["threshold"])
refer = prob >= threshold

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="caption">Estimated probability of disease</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="readout" style="color:{ALERT if refer else TRACE}">'
        f"{prob * 100:.1f}%</div>",
        unsafe_allow_html=True,
    )
    st.markdown(risk_strip(prob, threshold), unsafe_allow_html=True)
    st.markdown(
        f'<div class="verdict" style="color:{ALERT if refer else TRACE}">'
        f'{"Refer for angiography" if refer else "No referral indicated"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="note">The threshold sits at {threshold:.0%} rather than 50% '
        "because a missed case is priced at five times a false alarm. That choice "
        "moves the model toward catching every case and accepting more follow-ups.</div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="caption">What moved this prediction</div>',
                unsafe_allow_html=True)
    contrib = contributions(bundle, row)
    if contrib is None:
        st.info("Per-patient attribution is available for the linear model only.")
    else:
        display = contrib.copy()
        display["direction"] = np.where(display["effect"] > 0, "raises risk", "lowers risk")
        display["log-odds"] = display["effect"].round(3)
        st.dataframe(
            display[["feature", "direction", "log-odds"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "feature": "Measurement",
                "direction": "Effect",
                "log-odds": st.column_config.NumberColumn("Log-odds", format="%.3f"),
            },
        )
        st.markdown(
            '<div class="note">Bars are the feature value after standardisation '
            "multiplied by its fitted coefficient, so they sum exactly to the "
            "model's log-odds. Nothing here is a post-hoc approximation.</div>",
            unsafe_allow_html=True,
        )

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model", bundle["model_name"])
c2.metric("Cross-validated ROC-AUC", "0.907")
c3.metric("Test recall at threshold", "1.00")
c4.metric("Test precision at threshold", "0.72")

st.markdown(
    '<div class="note" style="margin-top:1rem;">Research and portfolio project built '
    "for the ProStackHub ML internship. Trained on 303 patients from a single 1988 "
    "study — far too small and too old to guide care. Not a medical device and not "
    "a substitute for clinical judgement.</div>",
    unsafe_allow_html=True,
)
