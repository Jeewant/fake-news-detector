import os
import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# MODEL LOADING FUNCTION
# ----------------------------
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    model_path = os.path.join(model_dir, "model.joblib")
    vec_path = os.path.join(model_dir, "vectorizer.joblib")

    # Debug info for logs (helps when deploying)
    st.write("== DEBUG INFO ==")
    st.write(f"models_dir: {model_dir}")
    st.write(f"Expecting model: {model_path}")
    st.write(f"Expecting vectorizer: {vec_path}")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        st.error("Model files missing! Please ensure model.joblib and vectorizer.joblib are in /models/")
        st.stop()

    model = joblib.load(model_path)
    vec = joblib.load(vec_path)

    st.success("✅ Model and vectorizer loaded successfully.")
    return vec, model


# ----------------------------
# STREAMLIT UI SETUP
# ----------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

vec, model = load_models()

mode = st.radio("Choose mode", ["Single text", "Upload CSV"])

# ----------------------------
# SINGLE TEXT MODE
# ----------------------------
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

# ----------------------------
# UPLOAD CSV MODE
# ----------------------------
else:
    uploaded = st.file_uploader("Upload CSV (must contain a 'text' column)", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()

        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            X = vec.transform(df["text"].astype(str))
            preds = model.predict(X)
            df["prediction"] = ["FAKE" if p == 1 else "REAL" for p in preds]
            st.dataframe(df.head(20))
            st.download_button("⬇️ Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")
