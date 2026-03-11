"""
metrics_logger.py — Thread-safe, atomic JSON metrics writer.

Written to during training; read by the Flask API to serve the dashboard.

Schema:
{
  "status":       "idle" | "training" | "complete",
  "distribution": "iid" | "non_iid",
  "started_at":   ISO string | null,
  "completed_at": ISO string | null,
  "config":       { num_clients, num_rounds, local_epochs, ... },
  "rounds": [
    {
      "round": 1,
      "clients": {
        "0": {
          "epochs": [ {"epoch":1,"loss":0.9,"accuracy":0.6}, ... ],
          "final_loss": 0.8,
          "final_accuracy": 0.65,
          "num_examples": 120
        },
        ...
      },
      "global_eval": { "loss": 0.75, "accuracy": 0.68 },   # server-side eval
      "fedavg_eval": { "loss": 0.76, "accuracy": 0.67 }    # client-reported weighted avg
    }
  ]
}
"""

import os
import json
import threading
from datetime import datetime

import config

METRICS_PATH = os.path.join(config.LOG_DIR, "training_metrics.json")

_lock = threading.Lock()


# ──────────────────────────────────────────────────────────
# Internal read / write helpers (always hold _lock when calling)
# ──────────────────────────────────────────────────────────
def _read() -> dict:
    if not os.path.exists(METRICS_PATH):
        return _empty_state()
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(state: dict) -> None:
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    tmp = METRICS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, METRICS_PATH)   # atomic on most OSes


def _empty_state() -> dict:
    return {
        "status":       "idle",
        "distribution": config.DISTRIBUTION,
        "started_at":   None,
        "completed_at": None,
        "config": {
            "num_clients":   config.NUM_CLIENTS,
            "num_rounds":    config.NUM_ROUNDS,
            "local_epochs":  config.LOCAL_EPOCHS,
            "batch_size":    config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "alpha":         config.ALPHA,
            "image_size":    list(config.IMAGE_SIZE),
        },
        "rounds": [],
    }


def _get_or_create_round(state: dict, rnd: int) -> dict:
    """Return the round dict for `rnd`, creating it if absent."""
    for r in state["rounds"]:
        if r["round"] == rnd:
            return r
    entry = {
        "round":       rnd,
        "clients":     {},
        "global_eval": None,
        "fedavg_eval": None,
    }
    state["rounds"].append(entry)
    state["rounds"].sort(key=lambda x: x["round"])
    return entry


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────
def init_session(distribution: str) -> None:
    """Call once at the start of a training run."""
    with _lock:
        state = _empty_state()
        state["distribution"] = distribution
        state["config"]["distribution"] = distribution
        state["status"]     = "training"
        state["started_at"] = datetime.now().isoformat()
        _write(state)


def set_status(status: str) -> None:
    """Update top-level status ('training' | 'complete')."""
    with _lock:
        state = _read()
        state["status"] = status
        if status == "complete":
            state["completed_at"] = datetime.now().isoformat()
        _write(state)


def log_client_epoch(rnd: int, client_id: int,
                     epoch: int, loss: float, accuracy: float) -> None:
    """Record one epoch result for a specific client in a round."""
    with _lock:
        state = _read()
        round_entry = _get_or_create_round(state, rnd)
        ckey = str(client_id)
        if ckey not in round_entry["clients"]:
            round_entry["clients"][ckey] = {
                "epochs": [],
                "final_loss":     None,
                "final_accuracy": None,
                "num_examples":   None,
            }
        round_entry["clients"][ckey]["epochs"].append({
            "epoch":    epoch,
            "loss":     round(loss, 6),
            "accuracy": round(accuracy, 6),
        })
        _write(state)


def log_client_round(rnd: int, client_id: int,
                     final_loss: float, final_accuracy: float,
                     num_examples: int) -> None:
    """Record the final (post-training) metrics for a client in a round."""
    with _lock:
        state = _read()
        round_entry = _get_or_create_round(state, rnd)
        ckey = str(client_id)
        if ckey not in round_entry["clients"]:
            round_entry["clients"][ckey] = {"epochs": []}
        round_entry["clients"][ckey]["final_loss"]     = round(final_loss, 6)
        round_entry["clients"][ckey]["final_accuracy"] = round(final_accuracy, 6)
        round_entry["clients"][ckey]["num_examples"]   = num_examples
        _write(state)


def log_global_eval(rnd: int, loss: float, accuracy: float) -> None:
    """Record server-side global model evaluation after FedAvg."""
    with _lock:
        state = _read()
        round_entry = _get_or_create_round(state, rnd)
        round_entry["global_eval"] = {
            "loss":     round(loss, 6),
            "accuracy": round(accuracy, 6),
        }
        _write(state)


def log_fedavg_eval(rnd: int, loss: float, accuracy: float) -> None:
    """Record the weighted-average of client evaluation metrics."""
    with _lock:
        state = _read()
        round_entry = _get_or_create_round(state, rnd)
        round_entry["fedavg_eval"] = {
            "loss":     round(loss, 6),
            "accuracy": round(accuracy, 6),
        }
        _write(state)


def read_metrics() -> dict:
    """Return the current full metrics state (thread-safe)."""
    with _lock:
        return _read()
