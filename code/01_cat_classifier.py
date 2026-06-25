"""
01_cat_classifier.py
====================
ML STRATEGY POINT: Single-number evaluation metric, and precision vs recall.

The overview says progress is faster when you have ONE number to compare ideas.
Here we train two classifiers ("A" and "B") on a binary "cat / not-cat" style
problem, then show why precision and recall alone can be ambiguous -- and how a
single combined metric (F1) lets you pick a winner at a glance.

CONTINUES FROM BOOK 1: this reuses the two models you already met in
"ML For Beginners" -- LogisticRegression and RandomForest -- and the same
split conventions (test_size=0.20, random_state=42, stratify=y). The dataset
is synthesized as a stand-in for "cat (1) vs not-cat (0)".
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

print("CAT CLASSIFIER -- single-number metric, precision & recall")
print("=" * 60)

# 1 = cat, 0 = not-cat. A small, slightly imbalanced dataset.
X, y = make_classification(n_samples=1200, n_features=12, n_informative=6,
                           weights=[0.6, 0.4], random_state=42)
X_tr, X_dev, y_tr, y_dev = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Two competing "ideas" -- classifier A and classifier B
A = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
B = RandomForestClassifier(n_estimators=60, random_state=42).fit(X_tr, y_tr)


def report(name, model):
    pred = model.predict(X_dev)
    # precision: of those we CALLED cats, how many were cats?
    # recall:    of all real cats, how many did we find?
    p = precision_score(y_dev, pred)
    r = recall_score(y_dev, pred)
    f = f1_score(y_dev, pred)        # single number that balances p and r
    print(f"  {name}:  precision={p:.3f}  recall={r:.3f}  -> F1={f:.3f}")
    return f


print("\nComparing two classifiers on the dev set:")
fa = report("Classifier A", A)
fb = report("Classifier B", B)

winner = "A" if fa >= fb else "B"
print(f"\nPrecision/recall can disagree, but the single F1 metric is decisive.")
print(f"  -> Pick Classifier {winner} (higher F1).")
print("\nSTRATEGY TAKEAWAY: one real-number metric turns a fuzzy two-number")
print("comparison into an instant 'better or worse?' decision.")
