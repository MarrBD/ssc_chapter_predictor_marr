import streamlit as st
import pandas as pd
import numpy as np
import joblib

LEVELS = {
    "SSC": {
        "model_path": "ssc_chapter_importance_model.pkl",
        "data_path": "ssc_chapters_data.csv",
    },
}

st.set_page_config(page_title="Chapter Importance Predictor", page_icon="📊", layout="wide")

st.sidebar.title("⚙️ Settings")

level = st.sidebar.selectbox("শিক্ষাস্তর (Level)", list(LEVELS.keys()))

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

bundle = load_model(LEVELS[level]["model_path"])
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

df = load_data(LEVELS[level]["data_path"])

subject = st.sidebar.selectbox("বিষয় (Subject)", sorted(df["subject"].unique()))
target_year = st.sidebar.number_input("কোন বছরের জন্য predict করবেন?", min_value=2025, max_value=2035, value=2026, step=1)
top_n = st.sidebar.slider("কতগুলো Chapter দেখাবেন?", 3, 15, 5)

def predict_importance(df, model, scaler, feature_cols):
    df_encoded = pd.get_dummies(df, columns=["subject"], prefix="subject")
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X = df_encoded[feature_cols]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return np.clip(preds, 0, 100)

df["predicted_importance"] = predict_importance(df, model, scaler, feature_cols)

is_odd_year = target_year % 2 != 0
adjustment_direction = 1 if is_odd_year else -1
df["adjusted_importance"] = df["predicted_importance"] + (adjustment_direction * df["parity_bias"] * 0.5)
df["adjusted_importance"] = np.clip(df["adjusted_importance"], 0, 100)

st.title("📊 SSC Chapter Importance Predictor")
st.caption(f"MARR (230321059) | EUB, CSE, Batch 26 | ML Course Project")

subject_df = df[df["subject"] == subject].sort_values("adjusted_importance", ascending=False).head(top_n)

st.subheader(f"{subject} — {target_year} সালের জন্য সবচেয়ে গুরুত্বপূর্ণ {top_n}টি Chapter")

for i, row in enumerate(subject_df.itertuples(), 1):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{i}. {row.chapter_name_en}** ({row.chapter_name_bn})")
        st.progress(min(int(row.adjusted_importance), 100))
    with col2:
        st.metric("Score", f"{row.adjusted_importance:.1f}")

st.divider()
st.subheader("বিস্তারিত টেবিল")
st.dataframe(
    subject_df[["chapter_name_en", "chapter_name_bn", "predicted_importance", "adjusted_importance"]]
    .rename(columns={
        "chapter_name_en": "Chapter (EN)",
        "chapter_name_bn": "Chapter (BN)",
        "predicted_importance": "Base Score",
        "adjusted_importance": f"Adjusted Score ({target_year})"
    }),
    use_container_width=True,
    hide_index=True
)
