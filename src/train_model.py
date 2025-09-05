# src/train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# -- config
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_xgb.joblib")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_names.joblib")

# create folders
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)

# 1) Load data if available, else generate synthetic data
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
if os.path.exists(csv_path):
    print("Loading real dataset:", csv_path)
    df = pd.read_csv(csv_path)
    # assume the target column is named 'Class' like Kaggle creditcard dataset
    if 'Class' not in df.columns:
        raise SystemExit("Expected 'Class' column in CSV")
    X = df.drop(columns=['Class'])
    y = df['Class']
else:
    print("No CSV found — generating synthetic imbalanced dataset (for demo).")
    # generate imbalanced data: 100k rows, 20 features, 0.5% positive (fraud)
    X_np, y = make_classification(
        n_samples=100000,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.995, 0.005],
        flip_y=0.01,
        random_state=42
    )
    feature_names = [f"f{i}" for i in range(X_np.shape[1])]
    X = pd.DataFrame(X_np, columns=feature_names)

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Compute scale_pos_weight for XGBoost
num_neg = sum(y_train == 0)
num_pos = sum(y_train == 1)
scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0
print(f"scale_pos_weight = {scale_pos_weight:.2f} (neg {num_neg} / pos {num_pos})")

# 4) Fit XGBoost classifier
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# 5) Evaluate
y_pred = model.predict(X_test)
print("Classification report on test set:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 6) Save model + feature names
joblib.dump(model, MODEL_PATH)
print("Saved model to", MODEL_PATH)
joblib.dump(list(X.columns), FEATURES_PATH)
print("Saved feature names to", FEATURES_PATH)
