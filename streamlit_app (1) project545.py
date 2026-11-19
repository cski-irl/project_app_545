"""
Streamlit app for the next-session injury-risk model from 545_Model__Project_.ipynb.

The saved file is the full sklearn Pipeline (preprocessing + tuned RandomForest),
so it takes a DataFrame with the raw feature columns and does its own
imputation, scaling and one-hot encoding.

Run:  streamlit run streamlit_app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("project_regression_model.joblib")  # name used in the notebook's last cell

# Same engineered features as the notebook (cell 3).
ROLL_COLS = ["recovery_score", "sleep_quality", "fatigue_index", "training_load"]


# --------------------------------------------------------------------------------------
# Load model and read everything we need back out of the fitted pipeline
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        f"Model file `{MODEL_PATH}` not found. Put the file saved by the notebook "
        "next to this script."
    )
    st.stop()

model = load_model(MODEL_PATH)
prep = model.named_steps["prep"]

# Column lists exactly as the ColumnTransformer was fitted.
numeric_features = list(prep.transformers_[0][2])
categorical_features = list(prep.transformers_[1][2])
FEATURES = numeric_features + categorical_features

# Training medians (from the imputer) make sensible defaults; scaler std gives step sizes.
num_pipe = prep.named_transformers_["num"]
medians = dict(zip(numeric_features, num_pipe.named_steps["impute"].statistics_))
stds = dict(zip(numeric_features, num_pipe.named_steps["scale"].scale_))

# Category options the encoder saw during training.
onehot = prep.named_transformers_["cat"].named_steps["onehot"]
cat_options = {c: list(cats) for c, cats in zip(categorical_features, onehot.categories_)}

base_numeric = [c for c in numeric_features if not c.endswith(("_roll5", "_trend3"))]
INT_COLS = {"step_count", "age"}


def pretty(col):
    return col.replace("_", " ").title()


def add_history_features(hist):
    """Recreate the notebook's rolling means and trends for one athlete's history."""
    hist = hist.copy()
    if "session_id" in hist.columns:
        hist = hist.sort_values("session_id")
    hist = hist.reset_index(drop=True)
    for col in ROLL_COLS:
        hist[f"{col}_roll5"] = hist[col].rolling(5, min_periods=1).mean()
        hist[f"{col}_trend3"] = (hist[col] - hist[col].shift(3)).fillna(0)
    return hist


def show_prediction(row):
    proba = float(model.predict_proba(row[FEATURES])[:, 1][0])
    pred = int(model.predict(row[FEATURES])[0])
    st.subheader("Prediction for the next session")
    c1, c2 = st.columns(2)
    c1.metric("Probability of high-risk next session", f"{proba:.1%}")
    c2.metric("Predicted class", "HIGH RISK" if pred == 1 else "Not high risk")
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(
        "High risk = the athlete's next session has injury_occurred == 2. "
        "Class uses the model's default 0.5 cut-off (the model was trained with "
        "balanced class weights, so probabilities run higher than the ~15% base rate)."
    )


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("Athlete Next-Session Injury Risk Predictor")
st.write(
    """
    Predicts whether an athlete's **next** training session will be **high risk**,
    using a tuned Random Forest trained on the Multimodal Sports Injury dataset.
    """
)

mode = st.radio(
    "Input method",
    ["Enter one session manually", "Upload an athlete's session history (CSV)"],
    horizontal=True,
)

# ---- Mode 1: manual entry ------------------------------------------------------------
if mode.startswith("Enter"):
    st.write("Defaults are the training-set medians.")
    values = {}

    st.markdown("#### Athlete & session context")
    cols = st.columns(len(categorical_features))
    for col, box in zip(categorical_features, cols):
        values[col] = box.selectbox(pretty(col), cat_options[col])

    st.markdown("#### Current session measurements")
    grid = st.columns(3)
    for i, col in enumerate(base_numeric):
        default = float(medians[col])
        step = float(stds[col]) / 10 if stds[col] > 0 else 0.1
        with grid[i % 3]:
            if col in INT_COLS:
                values[col] = st.number_input(pretty(col), value=int(round(default)), step=1)
            else:
                values[col] = st.number_input(
                    pretty(col), value=round(default, 3), step=round(step, 3) or 0.001,
                    format="%.3f",
                )

    with st.expander("Recent history (rolling averages and 3-session trends)"):
        st.write(
            "By default these assume no history: each 5-session average equals the "
            "current value and each trend is 0 (same as an athlete's first session in "
            "the notebook). Override if you know the athlete's recent sessions."
        )
        override = st.checkbox("Enter history values manually")
        for col in ROLL_COLS:
            r, t = f"{col}_roll5", f"{col}_trend3"
            if override:
                a, b = st.columns(2)
                values[r] = a.number_input(f"{pretty(col)} — 5-session avg",
                                           value=float(values[col]), format="%.3f")
                values[t] = b.number_input(f"{pretty(col)} — change vs 3 sessions ago",
                                           value=0.0, format="%.3f")
            else:
                values[r] = float(values[col])
                values[t] = 0.0

    row = pd.DataFrame([values])
    for c in numeric_features:
        row[c] = row[c].astype(float)
    show_prediction(row)

# ---- Mode 2: CSV history -------------------------------------------------------------
else:
    st.write(
        "Upload a CSV of **one athlete's** sessions with the original dataset columns "
        f"({', '.join(base_numeric + categorical_features)}), plus `session_id` for ordering. "
        "Rolling averages and trends are computed as in the notebook, and the prediction "
        "is for the session after the last row."
    )
    file = st.file_uploader("Session history CSV", type="csv")
    if file is not None:
        hist = pd.read_csv(file)
        needed = base_numeric + categorical_features
        missing = [c for c in needed if c not in hist.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        if "athlete_id" in hist.columns and hist["athlete_id"].nunique() > 1:
            athlete = st.selectbox("Athlete", sorted(hist["athlete_id"].unique()))
            hist = hist[hist["athlete_id"] == athlete]

        hist = add_history_features(hist)
        st.dataframe(hist.tail(5), use_container_width=True)
        show_prediction(hist.tail(1))
