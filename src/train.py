import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "fake_or_real_news.csv")
df = pd.read_csv(data_path)

# Combine title + text if available
if "title" in df.columns and "text" in df.columns:
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

# Map label column to numeric if needed
if df["label"].dtype == object:
    df["label"] = df["label"].map(lambda x: 1 if str(x).lower() in ["fake", "1", "true"] else 0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))
print("Report:\n", classification_report(y_test, preds))

# Save model and vectorizer
model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.joblib"))
joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
print("✅ Model and vectorizer saved in 'models/'")

