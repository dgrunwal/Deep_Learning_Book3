"""
04_speech_recognition.py
========================
ML STRATEGY POINT: End-to-end learning vs a hand-engineered pipeline, plus a
quick look at transfer learning.

The overview contrasts traditional speech recognition (audio -> hand-designed
features -> phonemes -> words) with END-TO-END learning (audio -> transcript in
one network). It also notes that with limited data, a pipeline using good
hand-built components can still win. We illustrate the trade-off on a digit
"audio-style" dataset by comparing:
    (a) a PIPELINE: hand-crafted feature reduction (PCA) -> classifier
    (b) an END-TO-END model: classifier straight from raw inputs
and then show TRANSFER LEARNING: reuse features learned on many classes to
help a task with very little data. Split conventions (test_size=0.20,
random_state=42) and the LogisticRegression model match Book 1.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("SPEECH RECOGNITION -- end-to-end vs pipeline, and transfer learning")
print("=" * 60)

# Digits (8x8 = 64 "signal" features) stand in for short audio clips -> labels.
X, y = load_digits(return_X_y=True)
X_tr, X_dev, y_tr, y_dev = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# (a) PIPELINE: a hand-designed feature stage (PCA) feeding a classifier --
#     analogous to extracting features/phonemes before recognizing words.
pipeline = make_pipeline(PCA(n_components=20),
                         LogisticRegression(max_iter=5000)).fit(X_tr, y_tr)
pipe_acc = pipeline.score(X_dev, y_dev)

# (b) END-TO-END: classifier straight from the raw inputs, no feature stage --
#     "let the data speak".
end2end = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
e2e_acc = end2end.score(X_dev, y_dev)

print("\n  Approach comparison on the dev set:")
print(f"    Pipeline  (PCA features -> classifier): {pipe_acc*100:5.1f}%")
print(f"    End-to-end (raw input  -> classifier):  {e2e_acc*100:5.1f}%")
print("    With plenty of data, end-to-end often matches or beats the pipeline.")

# ---- Transfer learning demo ----------------------------------------------
# Pretend digits 0-7 are an abundant "source task" and digits 8-9 are a new
# task with only a little data. We reuse the PCA features learned on the source.
print("\n  Transfer learning -- reuse features from a data-rich task:")
src = (y_tr <= 7)
new = (y_tr >= 8)

feature_extractor = PCA(n_components=20).fit(X_tr[src])   # learned on source task
X_new = feature_extractor.transform(X_tr[new])
X_new_dev = feature_extractor.transform(X_dev[y_dev >= 8])

clf = LogisticRegression(max_iter=5000).fit(X_new, y_tr[new])
transfer_acc = clf.score(X_new_dev, y_dev[y_dev >= 8])
print(f"    New task (digits 8-9) using transferred features: {transfer_acc*100:5.1f}%")

print("\nSTRATEGY TAKEAWAY: end-to-end shines with abundant data; pipelines and")
print("transfer learning help when data for the target task is scarce.")
