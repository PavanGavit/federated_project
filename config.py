"""
config.py — Central configuration for the FL-Project federated learning pipeline.
Breast Ultrasound Dataset: normal / benign / malignant
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
CLASSES        = ["normal", "benign", "malignant"]
NUM_CLASSES    = len(CLASSES)
IMAGE_SIZE     = (224, 224)          # resize all images to this
# Mask images are excluded from training (only originals are used)

# ─────────────────────────────────────────────
# Federated Learning
# ─────────────────────────────────────────────
NUM_CLIENTS       = 5                # simulated medical institutions
NUM_ROUNDS        = 10               # global FL rounds
LOCAL_EPOCHS      = 3                # local training epochs per round
MIN_FIT_CLIENTS   = 5
MIN_AVAIL_CLIENTS = 5
FRACTION_FIT      = 1.0

# ─────────────────────────────────────────────
# Data Distribution
# ─────────────────────────────────────────────
DISTRIBUTION = "non_iid"            # "iid" | "non_iid"
ALPHA        = 0.5                  # Dirichlet α for Non-IID (lower = more skewed)

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
SEED          = 42

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
TEST_SPLIT   = 0.20                 # fraction of total data held out as global test set
VAL_SPLIT    = 0.10                 # fraction held out per-client for local validation

# ─────────────────────────────────────────────
# Hardware  (auto-detected — do not hardcode)
# ─────────────────────────────────────────────
import torch  # noqa: E402

if torch.cuda.is_available():
    DEVICE      = "cuda"
    NUM_WORKERS = 4      # background DataLoader threads — fast on GPU machines
elif torch.backends.mps.is_available():
    DEVICE      = "mps"  # Apple Silicon GPU
    NUM_WORKERS = 2
else:
    DEVICE      = "cpu"
    NUM_WORKERS = 0      # extra threads add overhead on CPU-only

del torch  # keep the config namespace clean
