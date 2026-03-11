"""
web_app.py — Flask server for the FL-Project web interface.

Routes:
  GET  /                  → Training Dashboard (live metrics)
  GET  /inference         → Inference Page (upload + predict)
  GET  /api/metrics       → JSON: current training_metrics.json
  GET  /api/status        → JSON: {status, current_round, total_rounds}
  POST /api/predict       → JSON: {prediction, confidence, probabilities}
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request, abort

import config
import metrics_logger
from model import get_model, DEVICE, DiagnosisNet
from dataset import EVAL_TRANSFORMS

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

# ──────────────────────────────────────────────────────────
# Model cache (loaded once on first /api/predict request)
# ──────────────────────────────────────────────────────────
_MODEL: DiagnosisNet | None = None

def _get_model() -> DiagnosisNet:
    global _MODEL
    if _MODEL is None:
        m = get_model()
        ckpt = os.path.join(config.MODEL_DIR, "best_global_model.pt")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=DEVICE)
            m.load_state_dict(state)
            print(f"[WebApp] Loaded checkpoint: {ckpt}")
        else:
            print("[WebApp] ⚠ No checkpoint found — using untrained model.")
        m.eval()
        _MODEL = m
    return _MODEL


# ──────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html",
                           num_rounds=config.NUM_ROUNDS,
                           num_clients=config.NUM_CLIENTS)


@app.route("/inference")
def inference_page():
    return render_template("inference.html",
                           classes=config.CLASSES)


# ──────────────────────────────────────────────────────────
# API — Metrics
# ──────────────────────────────────────────────────────────
@app.route("/api/metrics")
def api_metrics():
    """Return the full training metrics JSON."""
    data = metrics_logger.read_metrics()
    return jsonify(data)


@app.route("/api/status")
def api_status():
    """Return a lightweight status summary for the dashboard header."""
    data = metrics_logger.read_metrics()
    completed_rounds = len(data.get("rounds", []))
    return jsonify({
        "status":           data.get("status", "idle"),
        "distribution":     data.get("distribution", "—"),
        "current_round":    completed_rounds,
        "total_rounds":     config.NUM_ROUNDS,
        "num_clients":      config.NUM_CLIENTS,
        "started_at":       data.get("started_at"),
        "completed_at":     data.get("completed_at"),
    })


# ──────────────────────────────────────────────────────────
# API — Inference
# ──────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept a multipart image upload, run DiagnosisNet, return predictions.
    Form field name: 'image'
    """
    if "image" not in request.files:
        abort(400, "No image field in form data.")
    file = request.files["image"]
    if not file.filename:
        abort(400, "Empty filename.")

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        abort(400, f"Cannot open image: {e}")

    tensor = EVAL_TRANSFORMS(pil_img).unsqueeze(0).to(DEVICE)

    model = _get_model()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_label = config.CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    return jsonify({
        "prediction":    pred_label,
        "confidence":    round(confidence * 100, 2),
        "probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(config.CLASSES, probs)
        },
    })


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[WebApp] Starting FL-Project web interface …")
    print(f"[WebApp] Dashboard  →  http://127.0.0.1:5000/")
    print(f"[WebApp] Inference  →  http://127.0.0.1:5000/inference\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
