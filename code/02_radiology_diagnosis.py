"""
02_radiology_diagnosis.py
=========================
ML STRATEGY POINT: Human-level performance as a proxy for Bayes error, and
splitting error into AVOIDABLE BIAS vs VARIANCE.

CONTINUES FROM "ML FOR BEGINNERS" (BOOK 1)
------------------------------------------
This program deliberately reuses the SAME dataset and SAME setup conventions as
the ml_demo.py program from "ML For Beginners":
    - load_breast_cancer()  (569 samples, 30 features, ~37% malignant / 63% benign)
    - train_test_split(..., test_size=0.20, random_state=42, stratify=y)
    - StandardScaler fitted on train only (no data leakage)
    - LogisticRegression and RandomForest, the same two models
So if you completed Book 1, every variable name and number here will look
familiar. Book 1 taught you to READ the results (accuracy, precision, recall).
This book teaches you to ACT on them with strategy: turning training and dev
error into avoidable bias and variance, and deciding what to fix next.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("RADIOLOGY DIAGNOSIS -- human-level error, avoidable bias vs variance")
print("Reusing the Breast Cancer Wisconsin dataset from 'ML For Beginners'")
print("=" * 64)

# --- Same data and same split conventions as Book 1's ml_demo.py ---
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # same 80/20 split as Book 1
    random_state=42,       # same reproducibility seed as Book 1
    stratify=y)            # same class-ratio preservation as Book 1

# --- Same scaling discipline as Book 1: fit on train, transform both ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# In Book 1 the held-out 20% was called the TEST set. In ML strategy we read it
# as the DEV set -- the data we tune against and diagnose on. Same numbers,
# strategic lens. (model.score returns ACCURACY; error = 1 - accuracy.)
model = LogisticRegression(max_iter=5000).fit(X_train_scaled, y_train)

train_err = 1.0 - model.score(X_train_scaled, y_train)
dev_err = 1.0 - model.score(X_test_scaled, y_test)

# Human-level error from the radiology example: a team of expert doctors
# achieves the lowest error, so we use it as our Bayes-error proxy.
human_level = 0.005        # 0.5% -- team of experienced doctors

avoidable_bias = train_err - human_level    # gap to the best possible
variance = dev_err - train_err              # gap from train to dev

print(f"\n  Human-level error (Bayes proxy): {human_level*100:5.2f}%")
print(f"  Training error:                  {train_err*100:5.2f}%")
print(f"  Dev error (Book 1's test set):   {dev_err*100:5.2f}%")
print(f"\n  Avoidable bias (train - human):  {avoidable_bias*100:5.2f}%")
print(f"  Variance      (dev - train):     {variance*100:5.2f}%")

print("\n  Diagnosis of what to work on next:")
if avoidable_bias > variance:
    print("    -> Avoidable bias dominates: focus on FITTING better")
    print("       (bigger model, train longer, better features).")
else:
    print("    -> Variance dominates: focus on GENERALIZING better")
    print("       (more data, regularization).")

print("\nSTRATEGY TAKEAWAY: Book 1 got logistic regression to ~98% accuracy on")
print("this data. Here we reframe that same result through human-level error to")
print("see WHICH knob -- bias or variance -- will improve it next.")
