#!/usr/bin/env python3
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Example external model registry for ms_ml_benchmark v2.py
# Provide either an estimator instance or (estimator, param_grid) pairs.
# If you want pretrained models, save them as .joblib/.pkl and pass the directory.

EXTERNAL_MODELS = {
    "External Logistic Regression": (
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "kbest__k": [50, 100, 200],
        },
    ),
    "External Random Forest": (
        RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
        },
    ),
}
