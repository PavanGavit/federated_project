# FL-Project — User Manual

Federated Learning for Breast Ultrasound Classification  
Model: **DiagnosisNet** · Aggregation: **FedAvg** · Framework: **Flower**

---

## Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (pick one)
python train.py non_iid

# 3. Open the web interface (separate terminal)
python web_app.py
# → Dashboard  http://localhost:5000/
# → Inference  http://localhost:5000/inference
```

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Run Commands](#run-commands)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Web Interface](#web-interface)
   - [Gradio Demo](#gradio-demo)
5. [Web Interface User Manual](#web-interface-user-manual)
6. [Dataset](#dataset)
7. [Outputs Reference](#outputs-reference)
8. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
FL-Project/
│
├── data/                          ← Dataset (DO NOT modify)
│   ├── benign/
│   ├── malignant/
│   └── normal/
│
├── outputs/                       ← Auto-created on first run
│   ├── models/
│   │   └── best_global_model.pt   ← best checkpoint saved here
│   └── logs/
│       ├── training_metrics.json  ← live metrics (read by dashboard)
│       ├── fl_results_iid.csv
│       ├── fl_results_non_iid.csv
│       ├── partition_iid.json
│       ├── partition_non_iid.json
│       ├── eval_cm_*.png          ← confusion matrix images
│       └── eval_roc_*.png         ← ROC curve images
│
├── templates/
│   ├── dashboard.html             ← Live training dashboard
│   └── inference.html             ← Image upload & prediction page
│
├── config.py           ← All settings (edit this to change behaviour)
├── dataset.py          ← Data loading & partitioning
├── model.py            ← DiagnosisNet architecture
├── client.py           ← Flower federated client
├── server.py           ← Flower FedAvg server
├── metrics_logger.py   ← Real-time JSON metrics writer
├── train.py            ← Training entry point (CLI)
├── evaluate.py         ← Standalone evaluation script
├── web_app.py          ← Flask web server (dashboard + inference)
├── app.py              ← Gradio demo (alternative UI)
└── requirements.txt
```

---

## Installation

### Requirements
- Python 3.10 or newer
- pip

### Steps

```bash
# (Recommended) Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

This installs: `torch`, `torchvision`, `flwr`, `flask`, `gradio`, `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`, `numpy`, `pandas`.

---

## Configuration

All settings live in **`config.py`**. Edit this file to change the experiment without touching any other file.

| Setting | Default | Description |
|---------|---------|-------------|
| `NUM_CLIENTS` | `5` | Number of simulated hospitals |
| `NUM_ROUNDS` | `10` | Number of global FL rounds |
| `LOCAL_EPOCHS` | `3` | Training epochs per client per round |
| `DISTRIBUTION` | `"non_iid"` | Default distribution mode |
| `ALPHA` | `0.5` | Dirichlet skew for Non-IID (lower = more skewed) |
| `BATCH_SIZE` | `16` | Mini-batch size |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `TEST_SPLIT` | `0.20` | Fraction of data held out as global test set |

---

## Run Commands

### Training

Run in a terminal from the project root directory.

```bash
# Non-IID distribution (default, recommended)
python train.py non_iid

# IID distribution
python train.py iid

# Run both IID and Non-IID back-to-back
python train.py both

# Use whatever DISTRIBUTION is set in config.py
python train.py
```

**What happens during training:**
- All 5 clients train locally each round
- FedAvg aggregates their weights
- Best model is saved automatically to `outputs/models/best_global_model.pt`
- Per-round metrics are written live to `outputs/logs/training_metrics.json`
- CSV log saved to `outputs/logs/fl_results_<dist>.csv`

> **Tip:** Open the web dashboard in a browser *before* starting training so you can watch metrics update live.

---

### Evaluation

Evaluate the saved best model on the test set.

```bash
# Evaluate using Non-IID test split
python evaluate.py non_iid

# Evaluate using IID test split
python evaluate.py iid

# Use the default from config.py
python evaluate.py
```

**Output:**
- Classification report printed to terminal (precision, recall, F1 per class)
- `outputs/logs/eval_cm_<dist>.png` — Confusion matrix heatmap
- `outputs/logs/eval_roc_<dist>.png` — Per-class ROC curves (AUC scores)

> ⚠️ Requires `best_global_model.pt` to exist. Run training first.

---

### Web Interface

```bash
python web_app.py
```

Opens a Flask server at **http://localhost:5000**

| URL | Page |
|-----|------|
| `http://localhost:5000/` | 📊 Live Training Dashboard |
| `http://localhost:5000/inference` | 🔬 Image Upload & Inference |

**To watch training live:** open the dashboard URL in your browser, *then* start training in a separate terminal. The dashboard polls for updates every 2 seconds automatically.

---

### Gradio Demo

An alternative all-in-one UI (includes training curves, partition visualiser, and inference):

```bash
python app.py
```

Opens at **http://localhost:7860** in your browser automatically.

---

## Web Interface User Manual

### 📊 Training Dashboard (`/`)

The dashboard updates automatically every 2 seconds while training is running.

#### Status Banner
- Shows current training status: **Idle** / **Training** (pulsing blue) / **Complete** (green)
- Shows the distribution mode (IID or NON-IID)
- Round progress bar

#### Stat Cards
| Card | Description |
|------|-------------|
| Current Round | Which FL round is in progress |
| Global Accuracy | Latest accuracy of the aggregated global model |
| Global Loss | Latest loss of the global model on the test set |
| Best Accuracy | Highest accuracy achieved across all rounds so far |

#### Charts
- **Global Model Accuracy** — Accuracy per round. Green = server-side eval, Blue dashed = FedAvg weighted average
- **Global Model Loss** — Loss per round. Orange = server-side eval, Red dashed = FedAvg weighted average

#### Live Client Cards
One card per hospital (Client 0–4). Each shows:
- Current round's final **loss** and **accuracy** for that client
- **Epoch mini-bars** — bar height represents loss per epoch within the current round (taller = higher loss)

#### Round-by-Round Table
Full summary table, newest round first. Columns:
- `Round` — round number
- `Global Loss / Acc` — server-side evaluation of the aggregated model
- `FedAvg Loss / Acc` — weighted average of client evaluation reports
- `C0–C4 Acc` — each client's local accuracy for that round

---

### 🔬 Inference Page (`/inference`)

#### How to use:
1. **Upload an image** — click the upload area or drag-and-drop any ultrasound PNG/JPG
2. The image preview appears immediately on the left
3. Click **Run Diagnosis**
4. Results appear on the right:
   - **Prediction badge** — predicted class coloured green (Normal), blue (Benign), or red (Malignant)
   - **Confidence** — percentage certainty of the top prediction
   - **Confidence bars** — animated probability bar for each of the 3 classes
5. Click **Clear** to reset and upload a different image

> ⚠️ Requires `best_global_model.pt` to exist. If no checkpoint is found, the server will use an untrained (random) model and predictions will be meaningless.

---

## Dataset

**Breast Ultrasound Images Dataset (BUSI)**

| Property | Value |
|----------|-------|
| Classes | `normal`, `benign`, `malignant` |
| Total Images | 780 PNG files |
| Mask Images | Included in folders but **automatically excluded** from training |
| Resolution | Average 500×500 px (all resized to 224×224 before training) |
| Patients | 600 female patients, ages 25–75 |
| Collected | 2018 |

Place images in:
```
data/normal/       ← normal ultrasound scans
data/benign/       ← benign lesion scans
data/malignant/    ← malignant lesion scans
```

---

## Outputs Reference

| File | Created by | Description |
|------|-----------|-------------|
| `outputs/models/best_global_model.pt` | `train.py` | PyTorch state dict of best accuracy checkpoint |
| `outputs/logs/training_metrics.json` | `train.py` (live) | Full per-round, per-client, per-epoch metrics |
| `outputs/logs/fl_results_iid.csv` | `train.py` | Round, loss, accuracy CSV for IID run |
| `outputs/logs/fl_results_non_iid.csv` | `train.py` | Round, loss, accuracy CSV for Non-IID run |
| `outputs/logs/partition_iid.json` | `train.py` | Class counts per client for IID partition |
| `outputs/logs/partition_non_iid.json` | `train.py` | Class counts per client for Non-IID partition |
| `outputs/logs/eval_cm_<dist>.png` | `evaluate.py` | Confusion matrix heatmap image |
| `outputs/logs/eval_roc_<dist>.png` | `evaluate.py` | Per-class ROC curves with AUC scores |

---

## Troubleshooting

**`No checkpoint found — using untrained model`**  
→ You haven't run training yet, or training didn't complete. Run `python train.py non_iid` first.

**Dashboard shows "Waiting for training to start…"**  
→ Training hasn't been started. Run `python train.py non_iid` in another terminal.  
→ Or the `outputs/logs/training_metrics.json` file doesn't exist yet (created when training begins).

**`Class folder not found` error**  
→ The `data/` directory is missing one of `normal/`, `benign/`, or `malignant/`. Check that the dataset is correctly placed.

**Flask port 5000 already in use**  
→ Change the port at the bottom of `web_app.py`: `app.run(port=5001)`

**Out of memory during training**  
→ Reduce `BATCH_SIZE` in `config.py` (e.g. to `8`).  
→ Reduce `NUM_CLIENTS` to lower parallel memory usage during simulation.

**Training is very slow**  
→ Reduce `NUM_ROUNDS` or `LOCAL_EPOCHS` in `config.py` for a quick test run.  
→ Enable GPU by ensuring CUDA is installed; `config.py` auto-detects it.
