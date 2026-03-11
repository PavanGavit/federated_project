"""
evaluate.py — Standalone evaluation script.

Loads the best saved global model and evaluates it on the test set,
printing a full classification report and saving all visualisations
to outputs/logs/eval_*.png.

Usage:
    python evaluate.py [iid|non_iid]
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

import config
from model import get_model, set_parameters, evaluate_model, DEVICE
from dataset import build_federated_datasets


# ─────────────────────────────────────────────────────
def load_model():
    model = get_model()
    ckpt  = os.path.join(config.MODEL_DIR, "best_global_model.pt")
    if not os.path.exists(ckpt):
        print(f"[Evaluate] ⚠ Checkpoint not found at {ckpt}."
               " Please run train.py first.")
        sys.exit(1)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[Evaluate] Loaded: {ckpt}")
    return model


def get_probabilities(model, loader):
    all_probs, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.numpy())
    return np.array(all_probs), np.array(all_targets)


def plot_confusion_matrix(targets, preds, save_path):
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES,
                linewidths=0.5, ax=ax, annot_kws={"size": 14})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix – Global Test Set",
                 fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] Confusion matrix → {save_path}")


def plot_roc_curves(targets, probs, save_path):
    n_classes = config.NUM_CLASSES
    y_bin = label_binarize(targets, classes=list(range(n_classes)))
    colors = ["#4CAF50", "#2196F3", "#F44336"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cls, col) in enumerate(zip(config.CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc      = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls.capitalize()} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves – One-vs-Rest", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Evaluate] ROC curves → {save_path}")


def main():
    dist = sys.argv[1].lower() if len(sys.argv) > 1 else config.DISTRIBUTION
    print(f"\n[Evaluate] Distribution mode: {dist.upper()}")

    _, test_loader, _, _ = build_federated_datasets(dist)
    model = load_model()

    loss, acc, preds, targets = evaluate_model(model, test_loader)
    probs, _ = get_probabilities(model, test_loader)

    print(f"\n{'='*55}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"{'='*55}")
    print("\nClassification Report:")
    print(classification_report(targets, preds,
                                 target_names=config.CLASSES, digits=4))

    # Save plots
    os.makedirs(config.LOG_DIR, exist_ok=True)
    plot_confusion_matrix(targets, preds,
                          os.path.join(config.LOG_DIR, f"eval_cm_{dist}.png"))
    plot_roc_curves(targets, probs,
                    os.path.join(config.LOG_DIR, f"eval_roc_{dist}.png"))

    print("\n[Evaluate] Done.")


if __name__ == "__main__":
    main()
