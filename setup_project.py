import os
import nbformat as nbf

BASE_DIR = r"C:\Users\lenovo\OneDrive\Documents\Social-Media-Sentiment-Analysis-Dashboard"
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "main.ipynb")

os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)

nb = nbf.v4.new_notebook()

cells = []

# ==========================================================
# 1. TITLE MARKDOWN
# ==========================================================
cells.append(nbf.v4.new_markdown_cell("""
# 📡 Sentiment Analysis Project  
### Reputation Radar - ML Pipeline
"""))

# ==========================================================
# 2. IMPORTS
# ==========================================================
cells.append(nbf.v4.new_code_cell("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
"""))

# ==========================================================
# 3. LOAD / CREATE DATA
# ==========================================================
cells.append(nbf.v4.new_code_cell("""
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1200,
    n_features=10,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X)
df["target"] = y

df.head()
"""))

# ==========================================================
# 4. DATA VISUALIZATION
# ==========================================================
cells.append(nbf.v4.new_code_cell("""
df["target"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.show()
"""))

# ==========================================================
# 5. TRAIN TEST SPLIT
# ==========================================================
cells.append(nbf.v4.new_code_cell("""
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""))

# ==========================================================
# 6. MODEL TRAINING
# ==========================================================
cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
"""))

# ==========================================================
# 7. PREDICTION
# ==========================================================
cells.append(nbf.v4.new_code_cell("""
y_pred = model.predict(X_test)
"""))

# ==========================================================
# 8. EVALUATION
# ==========================================================
cells.append(nbf.v4.new_code_cell("""
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
"""))

# ==========================================================
# SAVE
# ==========================================================
nb["cells"] = cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("✔ Notebook created at:", NOTEBOOK_PATH)