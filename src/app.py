import os
import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(model_dir, "model.joblib")
    vec_path = os.path.join(model_dir, "vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        st.error("⚠️ Model files missing! Please ensure model.joblib and vectorizer.joblib are in /models/")
        st.stop()

    model = joblib.load(model_path)
    vec = joblib.load(vec_path)
    return vec, model


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

vec, model = load_models()

# Small header info
st.caption("✅ Model and vectorizer loaded successfully.")
st.caption("🤖 Trained Logistic Regression model for Fake News Classification.")
st.markdown("**📊 Model Accuracy:** 98.4% on test dataset")

st.divider()

mode = st.radio("Choose mode", ["Single text", "Upload CSV"])

if mode == "Single text":
    text = st.text_area("Paste article or sentence:")
    if st.button("Predict"):
        if not text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            X = vec.transform([text])
            pred = model.predict(X)[0]
            result = "⚠️ FAKE" if pred == 1 else "✅ REAL"
            st.success(f"Prediction: {result}")
else:
    uploaded = st.file_uploader("Upload CSV (must contain a 'text' column)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            X = vec.transform(df["text"].astype(str))
            preds = model.predict(X)
            df["prediction"] = ["FAKE" if p == 1 else "REAL" for p in preds]
            st.dataframe(df.head(20))
            st.download_button("⬇️ Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Jeewant Malviya**  |  [View Source on GitHub](https://github.com/Jeewant/fake-news-detector)")
