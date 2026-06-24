import ollama
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. LOAD DATA ──────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# ── 2. TRAIN MODEL ────────────────────────────────────────
X = df[iris.feature_names]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 3. EVALUATE ───────────────────────────────────────────
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=iris.target_names)

# ── 4. LLM INTERPRETATION ─────────────────────────────────
prompt = f"""
You are a senior data scientist. Analyze this model performance report and provide insights.

Model: Random Forest Classifier
Dataset: Iris (150 samples, 3 classes)
Accuracy: {accuracy:.4f}

Classification Report:
{report}

Provide:
1. Overall performance assessment
2. Which class is hardest to predict and why
3. Production readiness verdict
"""

print("Running model analysis with LLM...\n")

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": prompt}]
)

print("=" * 60)
print("MODEL PERFORMANCE REPORT")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nLLM Analysis:")
print("=" * 60)
print(response.message.content)