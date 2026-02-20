"""
utils/evaluate_model.py
-----------------------
Evaluate the trained model on the validation split and produce:
  • Classification report (precision / recall / F1)
  • Confusion matrix image
  • ROC curve

Usage
-----
    python utils/evaluate_model.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    sys.exit("❌  TensorFlow not found. pip install tensorflow")

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)

IMG_SIZE   = 128
BATCH_SIZE = 32
DATASET    = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "mask_detector.h5")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "model")


def main():
    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(DATASET):
        sys.exit(f"Dataset not found: {DATASET}")

    print("Loading model …")
    model = load_model(MODEL_PATH)

    gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val = gen.flow_from_directory(
        DATASET,
        target_size  = (IMG_SIZE, IMG_SIZE),
        batch_size   = BATCH_SIZE,
        class_mode   = "binary",
        subset       = "validation",
        shuffle      = False,
    )

    print("Predicting …")
    preds_prob = model.predict(val, verbose=1).ravel()
    preds      = (preds_prob > 0.5).astype(int)
    labels     = val.classes
    class_names= list(val.class_indices.keys())

    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(labels, preds, target_names=class_names))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm   = confusion_matrix(labels, preds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(class_names, rotation=15)
    axes[0].set_yticklabels(class_names)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i,j], ha="center", va="center",
                         color="white" if cm[i,j] > cm.max()/2 else "black",
                         fontsize=18, fontweight="bold")
    plt.colorbar(im, ax=axes[0])

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, preds_prob)
    roc_auc     = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="#1f6feb", lw=2,
                 label=f"ROC (AUC = {roc_auc:.3f})")
    axes[1].plot([0,1],[0,1], linestyle="--", color="grey")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=.3)

    out_path = os.path.join(OUT_DIR, "evaluation.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\n📊  Evaluation plots saved → {out_path}")


if __name__ == "__main__":
    main()
