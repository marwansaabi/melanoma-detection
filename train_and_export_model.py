"""
Fit the final SVM on the full 51-image dataset and export it, so the
Streamlit app can load a ready model instead of retraining on every run.

Same hyperparameters as the cross-validated model in melanoma_detector.py
and the report (C=1.5, RBF kernel, class_weight='balanced', seed 42) plus
probability=True, which only adds Platt-scaled confidence estimates for the
UI - it does not change the decision boundary or the reported accuracy.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from melanoma_detector import (
    RANDOM_STATE,
    extract_all_features,
    read_images,
    read_labels,
    segment_image,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
parser.add_argument("--out", type=Path, default=Path("app_assets/model.joblib"))
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)

labels = read_labels(args.dataset_dir / "list.csv")
images, manual_masks, classes, ids = read_images(args.dataset_dir, labels)
print(f"Training on {len(images)} images ({sum(classes)} melanoma, {len(classes)-sum(classes)} benign)")

predicted_masks = [segment_image(img, gt_mask=gt)[0] for img, gt in zip(images, manual_masks)]

feature_names = None
X_list = []
for img, mask in zip(images, predicted_masks):
    feats = extract_all_features(img, mask)
    if feature_names is None:
        feature_names = list(feats.keys())
    X_list.append([feats[k] for k in feature_names])
X = np.array(X_list, dtype=float)
y = np.array(classes)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.5, gamma="scale", random_state=RANDOM_STATE,
        class_weight="balanced", probability=True),
)
model.fit(X, y)
print(f"Fitted on {X.shape[0]} x {X.shape[1]} features. Train accuracy: {model.score(X, y):.3f}")
print("(Train accuracy is expected to look better than the cross-validated 90.2% - "
      "it is not evaluated on held-out data. The CV number is the honest one.)")

joblib.dump({"model": model, "feature_names": feature_names}, args.out)
print(f"Saved to {args.out}")
