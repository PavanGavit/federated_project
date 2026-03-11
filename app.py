"""
app.py — Gradio interface for evaluating and demonstrating the federated DiagnosisNet model.

Features:
  1. Single-image diagnosis (upload → prediction + confidence bar chart)
  2. Global model evaluation on the full test set (metrics + confusion matrix)
  3. Training result plots (FL round curves from saved CSV logs)
  4. Federated partition visualisation (IID vs Non-IID class distribution)
"""

import os, json, io, csv
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gradio as gr

import config
from model import get_model, set_parameters, evaluate_model, DEVICE, DiagnosisNet
from dataset import build_federated_datasets, EVAL_TRANSFORMS
from sklearn.metrics import confusion_matrix, classification_report

# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────
def _load_best_model() -> DiagnosisNet:
    model = get_model()
    ckpt  = os.path.join(config.MODEL_DIR, "best_global_model.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[App] Loaded checkpoint: {ckpt}")
    else:
        print("[App] ⚠ No checkpoint found; using untrained model.")
    model.eval()
    return model


MODEL: DiagnosisNet | None = None


def _get_model() -> DiagnosisNet:
    global MODEL
    if MODEL is None:
        MODEL = _load_best_model()
    return MODEL


# ──────────────────────────────────────────────────────
# 1. Single-Image Prediction
# ──────────────────────────────────────────────────────
def predict_image(uploaded_img):
    """Predict class probabilities for a single uploaded image."""
    if uploaded_img is None:
        return "Please upload an image.", None

    # uploaded_img arrives as a numpy array from Gradio
    pil_img = Image.fromarray(uploaded_img).convert("RGB")
    tensor  = EVAL_TRANSFORMS(pil_img).unsqueeze(0).to(DEVICE)

    model = _get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = config.CLASSES[pred_idx]
    confidence = probs[pred_idx]

    # Build confidence bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#4CAF50", "#2196F3", "#F44336"]
    bars = ax.barh(config.CLASSES, probs * 100, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("DiagnosisNet Prediction", fontsize=13, fontweight="bold")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    result_text = (
        f"**Prediction:** `{pred_label.upper()}`\n\n"
        f"**Confidence:** `{confidence*100:.2f}%`\n\n"
        f"Normal: {probs[0]*100:.1f}%  | "
        f"Benign: {probs[1]*100:.1f}%  | "
        f"Malignant: {probs[2]*100:.1f}%"
    )
    return result_text, fig


# ──────────────────────────────────────────────────────
# 2. Global Model Evaluation
# ──────────────────────────────────────────────────────
def evaluate_global_model(distribution):
    """Evaluate the best saved model on the global test split."""
    _, test_loader, _, _ = build_federated_datasets(distribution)
    model = _load_best_model()

    loss, acc, preds, targets = evaluate_model(model, test_loader)

    # Confusion matrix heatmap
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASSES, yticklabels=config.CLASSES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Confusion Matrix – Global Test Set", fontsize=13, fontweight="bold")
    fig_cm.tight_layout()

    # Classification report text
    report = classification_report(targets, preds,
                                   target_names=config.CLASSES, digits=4)
    summary = (
        f"**Test Loss:** `{loss:.4f}`\n\n"
        f"**Test Accuracy:** `{acc*100:.2f}%`\n\n"
        f"```\n{report}\n```"
    )
    return summary, fig_cm


# ──────────────────────────────────────────────────────
# 3. Training Curves
# ──────────────────────────────────────────────────────
def plot_training_curves():
    """Plot loss and accuracy curves from saved CSV logs (IID & Non-IID)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = {"iid": "#2196F3", "non_iid": "#F44336"}

    found_any = False
    for dist_key, color in palette.items():
        csv_path = os.path.join(config.LOG_DIR, f"fl_results_{dist_key}.csv")
        if not os.path.exists(csv_path):
            continue
        found_any = True
        rounds, losses, accs = [], [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rounds.append(int(row["round"]))
                losses.append(float(row["loss"]))
                accs.append(float(row["accuracy"]) * 100)

        label = "IID" if dist_key == "iid" else "Non-IID"
        axes[0].plot(rounds, losses, marker="o", color=color, label=label, linewidth=2)
        axes[1].plot(rounds, accs,   marker="o", color=color, label=label, linewidth=2)

    if not found_any:
        for ax in axes:
            ax.text(0.5, 0.5, "No training logs found.\nRun train.py first.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        axes[0].set_title("Federated Loss per Round",      fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Round"); axes[0].set_ylabel("Loss")
        axes[1].set_title("Federated Accuracy per Round",  fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Round"); axes[1].set_ylabel("Accuracy (%)")
        for ax in axes:
            ax.legend(); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3)

    fig.suptitle("FL Training Curves: IID vs Non-IID", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────
# 4. Partition Visualisation
# ──────────────────────────────────────────────────────
def plot_partition(distribution):
    """
    Bar chart showing the label distribution per client for a given distribution mode.
    Requires partition JSON created during training.
    """
    json_path = os.path.join(config.LOG_DIR, f"partition_{distribution}.json")
    if not os.path.exists(json_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No partition data found.\nRun train.py first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    with open(json_path) as f:
        info = json.load(f)

    # info: {client_N: {normal: int, benign: int, malignant: int}}
    clients = sorted(info.keys())
    x = np.arange(len(clients))
    width = 0.25
    palette = {"normal": "#4CAF50", "benign": "#2196F3", "malignant": "#F44336"}

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, cls in enumerate(config.CLASSES):
        counts = [info[c].get(cls, 0) for c in clients]
        ax.bar(x + i * width, counts, width, label=cls.capitalize(),
               color=list(palette.values())[i], edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("client_", "Client ") for c in clients])
    ax.set_xlabel("Client (Medical Institution)", fontsize=11)
    ax.set_ylabel("Number of Images",             fontsize=11)
    ax.set_title(f"Data Partition – {distribution.upper().replace('_', '-')}",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════
def build_ui():
    dist_choices = ["iid", "non_iid"]

    with gr.Blocks(
        title="FL-Project · DiagnosisNet",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="teal",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .header-box { text-align:center; padding:20px 0 10px; }
        .header-box h1 { font-size:2rem; font-weight:800; }
        .header-box p  { color:#666; font-size:1rem; }
        .tab-content   { padding:20px; }
        """,
    ) as demo:

        # ── Header ─────────────────────────────────
        with gr.Row(elem_classes="header-box"):
            gr.HTML("""
            <div class="header-box">
              <h1>🩺 FL-Project · DiagnosisNet</h1>
              <p>Federated Learning for Breast Ultrasound Classification
                 &nbsp;|&nbsp; Normal · Benign · Malignant</p>
            </div>
            """)

        # ── Tabs ───────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Predict ─────────────────────
            with gr.TabItem("🔬 Predict"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(label="Upload Breast Ultrasound Image",
                                             type="numpy", height=300)
                        predict_btn = gr.Button("Run Diagnosis", variant="primary")
                    with gr.Column(scale=1):
                        result_md  = gr.Markdown(label="Prediction Result")
                        chart_out  = gr.Plot(label="Confidence Chart")

                predict_btn.click(fn=predict_image,
                                  inputs=img_input,
                                  outputs=[result_md, chart_out])

            # ── Tab 2: Evaluate Model ───────────────
            with gr.TabItem("📊 Evaluate Model"):
                with gr.Row():
                    dist_radio_eval = gr.Radio(dist_choices, value="non_iid",
                                               label="Training Distribution")
                    eval_btn = gr.Button("Evaluate on Test Set", variant="primary")
                with gr.Row():
                    eval_summary = gr.Markdown(label="Metrics")
                    cm_plot      = gr.Plot(label="Confusion Matrix")

                eval_btn.click(fn=evaluate_global_model,
                               inputs=dist_radio_eval,
                               outputs=[eval_summary, cm_plot])

            # ── Tab 3: Training Curves ──────────────
            with gr.TabItem("📈 Training Curves"):
                curves_btn  = gr.Button("Load Training Curves", variant="secondary")
                curves_plot = gr.Plot(label="IID vs Non-IID FL Rounds")
                curves_btn.click(fn=plot_training_curves, outputs=curves_plot)

            # ── Tab 4: Partition Visualiser ─────────
            with gr.TabItem("🗂 Data Partition"):
                dist_radio_part = gr.Radio(dist_choices, value="non_iid",
                                           label="Distribution Mode")
                part_btn  = gr.Button("Show Partition", variant="secondary")
                part_plot = gr.Plot(label="Class Distribution per Client")
                part_btn.click(fn=plot_partition,
                               inputs=dist_radio_part,
                               outputs=part_plot)

            # ── Tab 5: About ────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(f"""
## About This Project

### Dataset
**Breast Ultrasound Dataset (BUSI)**
- 600 female patients, ages 25–75, collected 2018
- **780 images** (PNG, avg 500×500 px) across 3 classes:
  `normal` · `benign` · `malignant`

### Federated Setup
| Parameter | Value |
|-----------|-------|
| Framework | Flower (flwr) |
| Aggregation | FedAvg |
| Clients | {config.NUM_CLIENTS} (simulated medical institutions) |
| Rounds | {config.NUM_ROUNDS} |
| Local Epochs | {config.LOCAL_EPOCHS} |
| Partitioning | IID & Non-IID (Dirichlet α={config.ALPHA}) |

### Model — DiagnosisNet
- 4× ConvBlocks (Conv→BN→ReLU×2 → MaxPool → Dropout)
- Global Average Pooling → FC(256) → FC(3)
- Input: 224×224 RGB

### Communication Pattern
Synchronous cross-silo federated learning with horizontal data partitioning.
""")

    return demo


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
