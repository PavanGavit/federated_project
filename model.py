"""
model.py — DiagnosisNet: a lightweight CNN for breast-ultrasound classification.

Architecture:
  • Feature extractor: 4 conv blocks with BatchNorm + MaxPool + Dropout
  • Classifier head:   Global Average Pooling → FC(256) → FC(num_classes)
  • Input:  RGB 224×224
  • Output: logits for [normal, benign, malignant]

Utility functions:
  • get_model()         — instantiate and move to device
  • get_parameters()    — extract flattened numpy weights (for Flower)
  • set_parameters()    — restore weights from numpy list (for Flower)
  • evaluate_model()    — run inference on a DataLoader, return metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import List
import config

# ─────────────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
)
print(f"[Model] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# DiagnosisNet
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DiagnosisNet(nn.Module):
    """
    Custom CNN designed for medical image classification.
    Tested on 224×224 RGB breast-ultrasound images.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32, dropout=0.10),   # 224 → 112
            ConvBlock(32,  64, dropout=0.20),   # 112 →  56
            ConvBlock(64,  128, dropout=0.25),  #  56 →  28
            ConvBlock(128, 256, dropout=0.30),  #  28 →  14
        )
        self.gap = nn.AdaptiveAvgPool2d(1)      # → (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────
def get_model() -> DiagnosisNet:
    return DiagnosisNet(num_classes=config.NUM_CLASSES).to(DEVICE)


# ─────────────────────────────────────────────
# Flower weight helpers
# ─────────────────────────────────────────────
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract all trainable parameters as a list of numpy arrays."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load numpy parameter list into the model in-place."""
    params_dict = zip(model.parameters(), parameters)
    for param, new_val in params_dict:
        param.data = torch.tensor(new_val, dtype=param.dtype).to(DEVICE)


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────
def evaluate_model(model: nn.Module, loader, criterion=None):
    """
    Run inference on `loader`, return:
        loss (float), accuracy (float),
        preds (list), targets (list)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else float("inf")
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy, all_preds, all_targets


def print_classification_report(all_targets, all_preds):
    print("\n" + "="*55)
    print("Classification Report")
    print("="*55)
    print(classification_report(all_targets, all_preds,
                                 target_names=config.CLASSES, digits=4))
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)
    print("="*55 + "\n")
