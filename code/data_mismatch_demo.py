"""
data_mismatch_demo.py
=====================
ML STRATEGY POINT: Data mismatch -- what happens when your training data comes
from a DIFFERENT distribution than your dev/test data, and how the TRAINING-DEV
set lets you tell data mismatch apart from a plain variance problem.

CONTINUES FROM "ML FOR BEGINNERS" (BOOK 1) AND THIS BOOK'S EARLIER CHAPTERS
--------------------------------------------------------------------------
This reuses the SAME conventions you already know:
    - load_breast_cancer()  (the Book 1 dataset)
    - StandardScaler fitted on TRAIN ONLY (no data leakage)
    - LogisticRegression, the same model family
    - random_state=42 so the run is reproducible

In Chapter 7 you split error into AVOIDABLE BIAS and VARIANCE by comparing
human-level, training, and dev error. That worked because train and dev came
from the SAME distribution. Here they do NOT: we deliberately make the dev/test
data "harder" by adding noise, mimicking the cat-photo example where training
images are clean web photos but dev images are blurry phone uploads. Two things
now change between train and dev at once -- the data the model has seen, AND the
distribution -- so the old gap no longer tells us which is to blame.

The fix from the chapter: carve out a TRAINING-DEV set -- same distribution as
training, but NOT trained on. The three gaps then separate cleanly:
    human  -> train      = avoidable bias
    train  -> train-dev  = variance   (same distribution, just unseen)
    train-dev -> dev     = DATA MISMATCH (distribution changed)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("DATA MISMATCH -- diagnosing it with a training-dev set")
print("=" * 60)

rng = np.random.RandomState(42)          # reproducible noise
X, y = load_breast_cancer(return_X_y=True)

# Split off the dev/test portion first; these will become the "target"
# distribution (the data we truly care about getting right).
X_train_full, X_target, y_train_full, y_target = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y)

# Carve a TRAINING-DEV set out of the training data. SAME distribution as
# training, but the model will NOT be trained on it.
X_train, X_traindev, y_train, y_traindev = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)

# Split the target data into dev and test (both the harder distribution).
X_dev, X_test, y_dev, y_test = train_test_split(
    X_target, y_target, test_size=0.50, random_state=42, stratify=y_target)

# Scale using statistics learned on the TRAINING set only (Book 1 discipline).
scaler = StandardScaler().fit(X_train)
X_train_s    = scaler.transform(X_train)
X_traindev_s = scaler.transform(X_traindev)
X_dev_s      = scaler.transform(X_dev)
X_test_s     = scaler.transform(X_test)

# --- Make the TARGET distribution different (and harder) -------------------
# Adding feature noise to dev/test ONLY is the tabular stand-in for the
# cat-photo example: train = clean web images, dev/test = blurry phone photos.
# The training and training-dev data stay clean; only the target shifts.
noise = rng.normal(0.0, 2.0, size=X_dev_s.shape)
X_dev_s  = X_dev_s  + noise
X_test_s = X_test_s + rng.normal(0.0, 2.0, size=X_test_s.shape)

# Train ONLY on the training set -- never on training-dev, dev, or test.
model = LogisticRegression(max_iter=5000).fit(X_train_s, y_train)

def err(name, Xs, yy):
    e = 1.0 - model.score(Xs, yy)
    print(f"    {name:<22} {e*100:5.2f}%")
    return e

human_level = 0.005                       # expert proxy for Bayes error
print("\n  Error at each stage:")
print(f"    {'human-level (proxy)':<22} {human_level*100:5.2f}%")
train_err    = err("training error",      X_train_s,    y_train)
traindev_err = err("training-dev error",  X_traindev_s, y_traindev)
dev_err      = err("dev error",           X_dev_s,      y_dev)
test_err     = err("test error",          X_test_s,     y_test)

# --- Read the gaps: each gap isolates ONE problem --------------------------
avoidable_bias = train_err    - human_level   # human  -> train
variance       = traindev_err - train_err     # train  -> train-dev (same dist)
data_mismatch  = dev_err      - traindev_err  # train-dev -> dev (dist changed)
overfit_dev    = test_err     - dev_err       # dev    -> test (tuning leak)

print("\n  Diagnostic gaps:")
print(f"    Avoidable bias (human -> train):      {avoidable_bias*100:5.2f}%")
print(f"    Variance       (train -> train-dev):  {variance*100:5.2f}%")
print(f"    DATA MISMATCH  (train-dev -> dev):     {data_mismatch*100:5.2f}%")
print(f"    Overfit to dev (dev -> test):         {overfit_dev*100:5.2f}%")

biggest = max(("avoidable bias", avoidable_bias),
              ("variance", variance),
              ("data mismatch", data_mismatch),
              key=lambda t: t[1])[0]
print(f"\n  Largest gap -> focus next on: {biggest.upper()}")
print("\nSTRATEGY TAKEAWAY: the training-dev set splits the old train->dev gap in")
print("two. A jump at train->train-dev is VARIANCE; a jump only at train-dev->dev")
print("is DATA MISMATCH -- the model learned the wrong distribution, so make the")
print("training data look more like the target (e.g. artificial data synthesis).")
