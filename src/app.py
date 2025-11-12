# ---- DEBUG-ENABLED app.py top (paste over the top of your current file) ----
import os
import joblib
import traceback
import streamlit as st
import pandas as pd

# show immediate debug info on the Streamlit page
st.set_page_config(page_title="Fake News Detector (debug)")
st.write("== DEBUG INFO ==")
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
st.write("models_dir (resolved):", models_dir)

# safe attempt to list files in the models dir
try:
    files = os.listdir(models_dir)
    st.write("models/ listing:", files)
except Exception as e:
    st.write("Could not list models/ directory:", repr(e))

# defensive loader that surfaces any exception in the UI (and re-raises)
@st.cache_resource
def load_models_debug():
    model_path = os.path.join(models_dir, "model.joblib")
    vec_path = os.path.join(models_dir, "vectorizer.joblib")
    st.write("Expecting model at:", model_path)
    st.write("Expecting vectorizer at:", vec_path)

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing file: {model_path}")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Missing file: {vec_path}")

        # attempt to load and measure sizes
        st.write("model size (bytes):", os.path.getsize(model_path))
        st.write("vectorizer size (bytes):", os.path.getsize(vec_path))

        vec = joblib.load(vec_path)
        model = joblib.load(model_path)
        st.write("Model and vectorizer loaded successfully.")
        return vec, model
    except Exception as e:
        st.error("Exception while loading models. See traceback below.")
        st.text(traceback.format_exc())
        # re-raise so Streamlit's logs also capture it
        raise

# Try to load models and catch top-level errors to show them in the UI
try:
    vec, model = load_models_debug()
except Exception:
    st.stop()  # stop further execution so we can see debug info

# ---- continue with the rest of your app code as-is below ----
# (from here on paste the rest of your original app.py code starting
#  from the radio UI, prediction code etc.)

import streamlit as st
import joblib
import pandas as pd
import os

@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    vec = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return vec, model

st.set_page_config(page_title="Fake News Detector")
st.title("📰 Fake News Detector")

vec, model = load_models()

mode = st.radio("Choose mode", ["Single text", "Upload CSV"])

if mode == "Single text":
    text = st.text_area("Paste article or sentence:")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter text to analyze.")
        else:
            X = vec.transform([text])
            pred = model.predict(X)[0]
            st.success("Prediction: ⚠️ FAKE" if pred == 1 else "Prediction: ✅ REAL")

else:
    uploaded = st.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            X = vec.transform(df["text"].astype(str))
            preds = model.predict(X)
            df["prediction"] = ["FAKE" if p == 1 else "REAL" for p in preds]
            st.dataframe(df.head(20))
            st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")
