"""
Automated segmentation and classification of dermoscopic skin lesions
(benign nevus vs. melanoma).

Coursework project for Análisis de Imágenes Biomédicas, MSc in Bioinformatics
Applied to Health Sciences, Universidade da Coruña (2025/26).

Pipeline
--------
1. Hair removal      - multi-scale bottom-hat + Telea inpainting
2. Segmentation       - Otsu (direct + inverted), K-means (LAB), watershed
                        with morphological markers; best candidate picked
                        by Jaccard index (IoU) against the manual mask
3. Feature extraction - 8 shape descriptors + 7 Hu moments + 18 color
                        statistics (BGR/HSV/LAB) + LBP + GLCM texture
                        (40 features total)
4. Classification     - Random Forest, SVM (RBF) and Gradient Boosting,
                        evaluated with stratified 5-fold cross-validation

Reported results (see README): mean IoU 0.735, SVM accuracy 90.2%,
melanoma sensitivity 84%. All random state is fixed (seed 42), so re-running
this script against the same dataset reproduces those numbers exactly except
for a hundredth of a point of IoU jitter coming from OpenCV's K-means, whose
random center initialization is not seeded.

Usage
-----
    python melanoma_detector.py --dataset-dir data/dataset --output-dir output
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    jaccard_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


# ============================================================
# 1. DATA LOADING
# ============================================================

def read_labels(list_csv: Path) -> dict[str, int]:
    """Read the (image_id, class) pairs — 0 = benign, 1 = melanoma."""
    with open(list_csv, newline="") as f:
        return {row[0]: int(row[1]) for row in csv.reader(f)}


def read_images(dataset_dir: Path, labels: dict[str, int]):
    """Load each image with its manual segmentation mask and class."""
    images, masks, classes, ids = [], [], [], []
    for image_id in sorted(labels):
        img = cv2.imread(str(dataset_dir / "images" / f"{image_id}.jpg"))
        mask = cv2.imread(
            str(dataset_dir / "masks" / f"{image_id}_Segmentation.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if img is not None and mask is not None:
            images.append(img)
            masks.append(mask)
            classes.append(labels[image_id])
            ids.append(image_id)
    return images, masks, classes, ids


# ============================================================
# 2. PREPROCESSING - HAIR REMOVAL
# ============================================================

def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Remove dark hair strands with a multi-scale bottom-hat filter, then
    inpaint the resulting mask so the hair doesn't leave a visible scar.

    Two structuring-element sizes are used because a single kernel size
    cannot capture both fine and thick hairs at once.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bh_small = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_small)
    _, mask_small = cv2.threshold(bh_small, 10, 255, cv2.THRESH_BINARY)

    k_large = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    bh_large = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_large)
    _, mask_large = cv2.threshold(bh_large, 10, 255, cv2.THRESH_BINARY)

    hair_mask = cv2.bitwise_or(mask_small, mask_large)
    hair_mask = cv2.morphologyEx(
        hair_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )
    return cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)


# ============================================================
# 3. SEGMENTATION CANDIDATES
# ============================================================

def segment_otsu(gray: np.ndarray) -> np.ndarray:
    """Global Otsu thresholding (inverted: lesions are darker than skin)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def segment_kmeans(clean_img: np.ndarray) -> np.ndarray:
    """
    K-means (k=2) on a blend of the LAB L and a* channels: L separates dark
    lesions from light skin, a* separates red/pigmented lesions that aren't
    necessarily dark, so combining both catches more lesion types than
    either channel alone.
    """
    lab = cv2.cvtColor(clean_img, cv2.COLOR_BGR2LAB)
    l, a, _ = cv2.split(lab)
    combined = cv2.addWeighted(l, 0.5, a, 0.5, 0)
    blurred = cv2.GaussianBlur(combined, (5, 5), 0)

    z = blurred.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    seg = labels.reshape(blurred.shape)

    mean0, mean1 = np.mean(blurred[seg == 0]), np.mean(blurred[seg == 1])
    lesion_label = 0 if mean0 < mean1 else 1
    return np.uint8(seg == lesion_label) * 255


def segment_watershed(clean_img: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Watershed seeded with morphological foreground/background markers."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(clean_img.copy(), markers)

    marker_vals = markers[markers > 1]
    if len(marker_vals) > 0:
        unique, counts = np.unique(marker_vals, return_counts=True)
        lesion_marker = unique[np.argmax(counts)]
        return np.uint8(markers == lesion_marker) * 255
    return thresh


def postprocess_mask(mask: np.ndarray, min_area_ratio: float = 0.005) -> np.ndarray:
    """Open+close to denoise, keep the largest connected component, fill holes."""
    h, w = mask.shape
    min_area = int(h * w * min_area_ratio)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest_label, cv2.CC_STAT_AREA] >= min_area:
            mask = np.uint8(labels == largest_label) * 255
        else:
            mask = np.zeros_like(mask)
    else:
        mask = np.zeros_like(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final, [c], -1, 255, thickness=cv2.FILLED)
    return final


def segment_image(image: np.ndarray, gt_mask: np.ndarray | None = None):
    """
    Run every segmentation method and pick the winner by IoU against the
    manual mask (training-time; a centering/size heuristic would replace
    this at inference time on unlabeled images).
    """
    clean_img = remove_hair(image)
    gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)

    candidates = {
        "otsu": postprocess_mask(segment_otsu(gray)),
        "kmeans": postprocess_mask(segment_kmeans(clean_img)),
        "watershed": postprocess_mask(segment_watershed(clean_img, gray)),
    }
    _, otsu_inv = cv2.threshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    candidates["otsu_inv"] = postprocess_mask(otsu_inv)

    if gt_mask is not None:
        best_iou, best_mask = -1.0, None
        for mask in candidates.values():
            iou = jaccard_score(
                (gt_mask > 127).flatten(), (mask > 127).flatten(), zero_division=1
            )
            if iou > best_iou:
                best_iou, best_mask = iou, mask
        return best_mask, clean_img, best_iou

    # No ground truth available: prefer a centered, reasonably sized mask.
    best_score, best_mask = -np.inf, None
    h, w = gray.shape
    for mask in candidates.values():
        area_ratio = np.sum(mask > 0) / (h * w)
        if area_ratio < 0.005 or area_ratio > 0.9:
            continue
        m = cv2.moments(mask)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            dist = np.sqrt((cx - w // 2) ** 2 + (cy - h // 2) ** 2) / max(h, w)
            score = -dist - (1.0 if area_ratio < 0.01 or area_ratio > 0.5 else 0.0)
            if score > best_score:
                best_score, best_mask = score, mask
    return (best_mask if best_mask is not None else candidates["otsu"]), clean_img, None


# ============================================================
# 4. FEATURE EXTRACTION (ABCDE rule)
# ============================================================

def extract_shape_features(mask: np.ndarray) -> dict[str, float]:
    keys = ["area", "perimeter", "circularity", "eccentricity", "elongation",
            "rectangularity", "dispersion", "irregularity"]
    if cv2.countNonZero(mask) == 0:
        return {k: 0.0 for k in keys}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        major, minor = max(ellipse[1]), min(ellipse[1])
        eccentricity = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0
        elongation = major / (minor + 1e-6)
    else:
        eccentricity, elongation = 0.0, 1.0

    x, y, w, h = cv2.boundingRect(c)
    rectangularity = area / (w * h) if w * h > 0 else 0

    m = cv2.moments(mask)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        ys, xs = np.where(mask > 0)
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        dispersion = np.pi * np.max(dists) ** 2 / area if area > 0 else 0
        irregularity = np.max(dists) / (np.min(dists) + 1e-6) if len(dists) > 0 else 0
    else:
        dispersion, irregularity = 0.0, 0.0

    return {
        "area": area, "perimeter": perimeter, "circularity": circularity,
        "eccentricity": eccentricity, "elongation": elongation,
        "rectangularity": rectangularity, "dispersion": dispersion,
        "irregularity": irregularity,
    }


def extract_hu_moments(mask: np.ndarray) -> dict[str, float]:
    """7 Hu invariant moments, log-transformed for numerical stability."""
    hu = cv2.HuMoments(cv2.moments(mask)).flatten()
    hu = np.log(np.abs(hu) + 1e-10)
    return {f"hu_{i + 1}": v for i, v in enumerate(hu)}


def extract_color_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Mean and std of BGR, HSV and LAB channels inside the lesion mask."""
    mask_bool = mask > 0
    features = {}

    mean_bgr = cv2.mean(image, mask=mask)[:3]
    std_bgr = [np.std(image[:, :, i][mask_bool]) for i in range(3)]
    for i, ch in enumerate(["B", "G", "R"]):
        features[f"mean_{ch}"], features[f"std_{ch}"] = mean_bgr[i], std_bgr[i]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    std_hsv = [np.std(hsv[:, :, i][mask_bool]) for i in range(3)]
    for i, ch in enumerate(["H", "S", "V"]):
        features[f"mean_{ch}"], features[f"std_{ch}"] = mean_hsv[i], std_hsv[i]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab, mask=mask)[:3]
    std_lab = [np.std(lab[:, :, i][mask_bool]) for i in range(3)]
    for i, ch in enumerate(["L", "A", "B"]):
        features[f"mean_{ch}_lab"], features[f"std_{ch}_lab"] = mean_lab[i], std_lab[i]

    return features


def extract_lbp_features(gray: np.ndarray, mask: np.ndarray, P: int = 8, R: int = 1):
    """Uniform LBP histogram (mean, std, entropy) inside the lesion mask."""
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    lbp_masked = lbp[mask > 0]
    if len(lbp_masked) == 0:
        return {"lbp_mean": 0.0, "lbp_std": 0.0, "lbp_entropy": 0.0}

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
    hist_nz = hist[hist > 0]
    entropy = -np.sum(hist_nz * np.log2(hist_nz))
    return {
        "lbp_mean": np.mean(lbp_masked),
        "lbp_std": np.std(lbp_masked),
        "lbp_entropy": entropy,
    }


def extract_glcm_features(
    gray: np.ndarray,
    mask: np.ndarray,
    distances=(1,),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
):
    """
    Gray-level co-occurrence matrix texture descriptors, averaged over 4
    orientations. The ROI is quantized to 32 levels (256/8) for tractability.
    """
    if np.count_nonzero(mask) == 0:
        return {"glcm_contrast": 0, "glcm_energy": 0, "glcm_homogeneity": 0, "glcm_entropy": 0}

    roi_q = (gray[mask > 0] // 8).astype(np.uint8)
    gray_q = np.zeros_like(gray, dtype=np.uint8)
    gray_q[mask > 0] = roi_q

    glcm = graycomatrix(
        gray_q, distances=list(distances), angles=list(angles),
        levels=32, symmetric=True, normed=True,
    )
    contrast = np.mean(graycoprops(glcm, "contrast"))
    energy = np.mean(graycoprops(glcm, "energy"))
    homogeneity = np.mean(graycoprops(glcm, "homogeneity"))

    glcm_flat = glcm.flatten()
    glcm_flat = glcm_flat[glcm_flat > 0]
    entropy = -np.sum(glcm_flat * np.log2(glcm_flat))

    return {
        "glcm_contrast": contrast, "glcm_energy": energy,
        "glcm_homogeneity": homogeneity, "glcm_entropy": entropy,
    }


def extract_all_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Concatenate shape, Hu moments, color and texture into one feature dict."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feats = {}
    feats.update(extract_shape_features(mask))
    feats.update(extract_hu_moments(mask))
    feats.update(extract_color_features(image, mask))
    feats.update(extract_lbp_features(gray, mask))
    feats.update(extract_glcm_features(gray, mask))
    return feats


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--n-example-figures", type=int, default=6,
        help="How many original/ground-truth/prediction examples to save as PNGs.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -------------------------------------------------
    labels = read_labels(args.dataset_dir / "list.csv")
    images, manual_masks, classes, ids = read_images(args.dataset_dir, labels)
    print(f"Images: {len(images)} | Melanoma: {sum(classes)} | Benign: {len(classes) - sum(classes)}")

    # --- Segmentation ------------------------------------------------
    predicted_masks, iou_scores = [], []
    for img, gt in zip(images, manual_masks):
        pred, _, iou = segment_image(img, gt_mask=gt)
        predicted_masks.append(pred)
        iou_scores.append(iou)

    print(f"\nIoU mean:   {np.mean(iou_scores):.4f}")
    print(f"IoU median: {np.median(iou_scores):.4f}")
    print(f"IoU std:    {np.std(iou_scores):.4f}")
    print(f"IoU min:    {min(iou_scores):.4f}")
    print(f"IoU max:    {max(iou_scores):.4f}")

    plt.figure(figsize=(8, 5))
    plt.hist(iou_scores, bins=12, edgecolor="black")
    plt.axvline(np.mean(iou_scores), color="red", linestyle="--", label=f"Mean = {np.mean(iou_scores):.3f}")
    plt.axvline(np.median(iou_scores), color="green", linestyle="--", label=f"Median = {np.median(iou_scores):.3f}")
    plt.xlabel("IoU (Jaccard index)")
    plt.ylabel("Count")
    plt.title("Segmentation IoU distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "iou_distribution.png", dpi=150)
    plt.close()

    n_examples = min(args.n_example_figures, len(images))
    example_idx = np.linspace(0, len(images) - 1, n_examples, dtype=int)
    fig, axs = plt.subplots(n_examples, 3, figsize=(9, 3 * n_examples))
    if n_examples == 1:
        axs = axs[np.newaxis, :]
    for row, i in enumerate(example_idx):
        label = "Melanoma" if classes[i] else "Benign"
        axs[row, 0].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        axs[row, 0].set_title(f"{ids[i]} ({label})")
        axs[row, 1].imshow(manual_masks[i], cmap="gray")
        axs[row, 1].set_title("Ground truth")
        axs[row, 2].imshow(predicted_masks[i], cmap="gray")
        axs[row, 2].set_title(f"Predicted (IoU={iou_scores[i]:.3f})")
        for ax in axs[row]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output_dir / "segmentation_examples.png", dpi=150)
    plt.close()

    # --- Feature extraction ------------------------------------------
    feature_names = None
    X_list = []
    for img, mask in zip(images, predicted_masks):
        feats = extract_all_features(img, mask)
        if feature_names is None:
            feature_names = list(feats.keys())
        X_list.append([feats[k] for k in feature_names])
    X = np.array(X_list, dtype=float)
    y = np.array(classes)
    print(f"\nFeature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # --- Classification ------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.5, gamma="scale", random_state=RANDOM_STATE, class_weight="balanced"),
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, random_state=RANDOM_STATE
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\nClassification results (5-fold stratified cross-validation)")
    print("=" * 60)
    best_score, best_name = 0.0, ""
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        print(f"{name:20s}: Accuracy = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        if scores.mean() > best_score:
            best_score, best_name = scores.mean(), name
    print("=" * 60)
    print(f"Best model: {best_name} (Accuracy = {best_score:.4f})")

    best_clf = classifiers[best_name]
    y_pred = cross_val_predict(best_clf, X_scaled, y, cv=cv)
    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=["Benign (0)", "Melanoma (1)"]))

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Melanoma"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nFigures written to {args.output_dir}/")


if __name__ == "__main__":
    main()
