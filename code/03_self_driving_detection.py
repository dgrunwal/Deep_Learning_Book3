"""
03_self_driving_detection.py
============================
ML STRATEGY POINT: Multi-task learning -- one model predicting several labels
at once, where each task can help the others.

The overview's self-driving example: a single image may contain pedestrians,
cars, stop signs, and traffic lights, so each example has FOUR labels at once
(a 4-vector), not one. We train a single multi-output classifier that predicts
all four from the same input, exactly as a self-driving vision system would.
Split conventions (test_size=0.20, random_state=42) match Book 1; the model
builds on the LogisticRegression you already know, wrapped to emit four labels.
"""

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("SELF-DRIVING DETECTION -- multi-task (multi-label) learning")
print("=" * 60)

LABELS = ["pedestrian", "car", "stop_sign", "traffic_light"]

# Each "image" (feature row) carries 4 independent yes/no labels at once.
X, Y = make_multilabel_classification(n_samples=1500, n_features=20,
                                      n_classes=4, n_labels=2, random_state=42)
X_tr, X_dev, Y_tr, Y_dev = train_test_split(X, Y, test_size=0.20, random_state=42)

# ONE model, four outputs -- the essence of multi-task learning.
model = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X_tr, Y_tr)
pred = model.predict(X_dev)

print("\n  Per-task detection accuracy on the dev set:")
for i, name in enumerate(LABELS):
    acc = accuracy_score(Y_dev[:, i], pred[:, i])
    print(f"    {name:14s}: {acc*100:5.1f}%")

# Show one example image's predicted label vector vs the truth.
print("\n  Example -- one image, four labels at once:")
print(f"    predicted: {dict(zip(LABELS, pred[0].tolist()))}")
print(f"    actual:    {dict(zip(LABELS, Y_dev[0].tolist()))}")

print("\nSTRATEGY TAKEAWAY: instead of four separate detectors, one shared network")
print("learns all tasks together -- features useful for 'car' also help 'stop_sign'.")
