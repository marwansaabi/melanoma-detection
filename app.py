"""
Streamlit demo for the melanoma detection pipeline.

Interactive front-end over melanoma_detector.py: segment a skin lesion,
extract the same 40 features used in the report, and classify it with the
SVM trained on the full 51-image dataset (app_assets/model.joblib,
produced by train_and_export_model.py).
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from sklearn.metrics import jaccard_score

from melanoma_detector import extract_all_features, remove_hair, segment_image

BASE = Path(__file__).parent
ASSETS = BASE / "app_assets"
EXAMPLES_DIR = ASSETS / "examples"

st.set_page_config(page_title="Melanoma Detection", page_icon="🔬", layout="wide")

ACCENT = "#2563eb"


@st.cache_resource
def load_model():
    bundle = joblib.load(ASSETS / "model.joblib")
    return bundle["model"], bundle["feature_names"]


@st.cache_data
def load_examples():
    import csv
    with open(EXAMPLES_DIR / "list.csv") as f:
        return {row[0]: int(row[1]) for row in csv.reader(f)}


def run_pipeline(image_bgr: np.ndarray, gt_mask: np.ndarray | None = None):
    mask, clean_img, iou = segment_image(image_bgr, gt_mask=gt_mask)
    model, feature_names = load_model()
    feats = extract_all_features(image_bgr, mask)
    x = np.array([[feats[k] for k in feature_names]])
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    return {
        "mask": mask, "clean_img": clean_img, "iou": iou,
        "prediction": "Melanoma" if pred == 1 else "Benign",
        "confidence": float(proba[pred]),
        "proba_melanoma": float(proba[1]),
    }


# ---------------------------------------------------------------- UI

st.title("🔬 Melanoma Detection")
st.caption("Classical computer vision — segmentation + SVM, no deep learning. MSc coursework project.")

st.error(
    "**This is not a medical device and does not provide medical advice.** "
    "It's a coursework demo trained on 51 images, cross-validated at 90.2% "
    "accuracy — nowhere near the reliability required for a real diagnostic "
    "tool. If you're concerned about a mole or skin lesion, please see a "
    "dermatologist. Do not use this tool, or any result it gives you, to "
    "make a decision about your health.",
    icon="⚠️",
)

agree = st.checkbox(
    "I understand this is an educational demo, not a diagnostic tool, and I will not use it "
    "(or any photo of my own skin) to make a health decision."
)

if not agree:
    st.info("Check the box above to continue.")
    st.stop()

st.markdown("---")

examples = load_examples()
mode = st.radio(
    "Image source",
    ["Try an example from the dataset", "Upload your own dermoscopic image"],
    horizontal=True,
)

image_bgr = None
gt_mask = None
true_label = None

if mode == "Try an example from the dataset":
    st.caption(
        "These 6 images are from the same ISIC 2018 dataset used to train and evaluate the "
        "model (CC0 license) — a melanoma and a benign case at low, medium and high "
        "segmentation difficulty."
    )
    cols = st.columns(6)
    thumbs = sorted(examples.keys())
    choice = st.session_state.get("example_choice", thumbs[0])
    for col, image_id in zip(cols, thumbs):
        with col:
            st.image(str(EXAMPLES_DIR / "images" / f"{image_id}.jpg"), use_container_width=True)
            label = "Melanoma" if examples[image_id] else "Benign"
            if st.button(label, key=f"btn_{image_id}", use_container_width=True):
                st.session_state["example_choice"] = image_id
    choice = st.session_state.get("example_choice", thumbs[0])
    st.markdown(f"**Selected:** `{choice}`")

    image_bgr = cv2.imread(str(EXAMPLES_DIR / "images" / f"{choice}.jpg"))
    gt_mask = cv2.imread(str(EXAMPLES_DIR / "masks" / f"{choice}_Segmentation.png"), cv2.IMREAD_GRAYSCALE)
    true_label = "Melanoma" if examples[choice] else "Benign"

else:
    st.caption(
        "Works best on dermoscopic images similar to the ISIC dataset — a close-up, "
        "well-lit photo of a single lesion. An ordinary phone photo of skin will very "
        "likely segment and classify poorly; the model has never seen anything like it."
    )
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is not None:
    with st.spinner("Segmenting and classifying..."):
        result = run_pipeline(image_bgr, gt_mask=gt_mask)

    c1, c2, c3 = st.columns(3)
    c1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    if gt_mask is not None:
        c2.image(gt_mask, caption="Manual ground truth", use_container_width=True, clamp=True)
    else:
        c2.empty()
    c3.image(result["mask"], caption="Predicted segmentation", use_container_width=True, clamp=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", result["prediction"])
    m2.metric("Model confidence", f"{result['confidence']:.0%}")
    if result["iou"] is not None:
        m3.metric("Segmentation IoU", f"{result['iou']:.3f}")
    else:
        m3.metric("Segmentation IoU", "n/a (no ground truth)")

    if true_label is not None:
        correct = true_label == result["prediction"]
        st.success(f"Actual label: **{true_label}** — model prediction matches: {'✅' if correct else '❌'}")
    else:
        st.warning(
            "No ground truth for this image, so there's nothing to check the prediction "
            "against. Treat this output as illustrative, not authoritative — see the "
            "disclaimer above."
        )

    with st.expander("How this works"):
        st.markdown(
            """
1. **Hair removal** — multi-scale bottom-hat filtering + inpainting
2. **Segmentation** — Otsu, K-means and watershed candidates, best one picked by IoU
   against ground truth (when available) or a centering heuristic otherwise
3. **Features** — 8 shape descriptors, 7 Hu moments, 18 color statistics
   (BGR/HSV/LAB), LBP and GLCM texture — 40 features total
4. **Classification** — SVM (RBF kernel), trained on all 51 images

Full methodology, honest limitations, and the report: see the
[README](https://github.com/marwansaabi/melanoma-detection).
            """
        )

st.markdown("---")
st.caption("Built by Marwan El Saabi · MSc Bioinformatics, Universidade da Coruña · Data: ISIC 2018")
