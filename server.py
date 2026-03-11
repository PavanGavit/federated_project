"""
server.py — Pure-Python FedAvg simulation runner. No Ray required.

The FL loop is implemented manually:
  For each round:
    1. Send global weights to every client via FlowerClient.fit()
    2. Collect updated weights + num_examples from each client
    3. FedAvg: weighted average of all client weights
    4. Server-side evaluate on the hold-out test set
    5. Each client calls evaluate() → weighted-average for logging

Metrics are written live to:
  • outputs/logs/training_metrics.json  (read by the web dashboard)
  • outputs/logs/fl_results_<dist>.csv  (round-level CSV)
"""

import os
import csv
import json
import copy

import numpy as np
import torch

import config
import metrics_logger
from model import get_model, get_parameters, set_parameters, evaluate_model, DEVICE
from dataset import build_federated_datasets
from client import FlowerClient            # use class directly — no Flower runtime


# ─────────────────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────────────────
def _init_csv(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "loss", "accuracy"])


def _append_csv(log_path: str, rnd: int, loss: float, acc: float) -> None:
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([rnd, f"{loss:.6f}", f"{acc:.6f}"])


# ─────────────────────────────────────────────────────────
# FedAvg weight aggregation
# ─────────────────────────────────────────────────────────
def fedavg(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """
    Weighted average of model parameters.

    Args:
        results: list of (parameters, num_examples) tuples from each client.

    Returns:
        Aggregated parameter list.
    """
    total_examples = sum(n for _, n in results)
    aggregated = []
    for layer_idx in range(len(results[0][0])):
        weighted_sum = sum(
            params[layer_idx] * (n / total_examples)
            for params, n in results
        )
        aggregated.append(weighted_sum)
    return aggregated


# ─────────────────────────────────────────────────────────
# Server-side evaluation + checkpoint saving
# ─────────────────────────────────────────────────────────
_best_acc = 0.0


def server_evaluate(global_params: list[np.ndarray],
                    test_loader,
                    server_round: int) -> tuple[float, float]:
    """Evaluate the aggregated global model on the held-out test set."""
    global _best_acc

    model = get_model()
    set_parameters(model, global_params)
    loss, acc, _, _ = evaluate_model(model, test_loader)

    print(f"  [Server Eval Round {server_round:>2}] loss={loss:.4f}  acc={acc:.4f}")
    metrics_logger.log_global_eval(server_round, loss, acc)

    if acc > _best_acc:
        _best_acc = acc
        ckpt_path = os.path.join(config.MODEL_DIR, "best_global_model.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  ✓ New best model saved ({acc:.4f}) → {ckpt_path}")

    return float(loss), float(acc)


# ─────────────────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────────────────
def run_simulation(distribution: str = config.DISTRIBUTION):
    global _best_acc
    _best_acc = 0.0   # reset between IID and Non-IID runs

    print("\n" + "="*60)
    print("  FL-Project  |  Federated Learning Simulation")
    print(f"  Distribution : {distribution.upper()}")
    print(f"  Clients      : {config.NUM_CLIENTS}")
    print(f"  Rounds       : {config.NUM_ROUNDS}")
    print(f"  Local Epochs : {config.LOCAL_EPOCHS}")
    print("="*60 + "\n")

    # ── Init metrics log ───────────────────────────────────
    metrics_logger.init_session(distribution)

    # ── Build datasets ─────────────────────────────────────
    client_loaders, test_loader, _, partition_info = build_federated_datasets(distribution)

    # Save partition info
    info_path = os.path.join(config.LOG_DIR, f"partition_{distribution}.json")
    with open(info_path, "w") as f:
        json.dump(partition_info, f, indent=2)
    print(f"[Server] Partition info saved → {info_path}\n")

    # ── CSV log ────────────────────────────────────────────
    log_path = os.path.join(config.LOG_DIR, f"fl_results_{distribution}.csv")
    _init_csv(log_path)

    # ── Instantiate one client per hospital ────────────────
    clients = [FlowerClient(cid) for cid in range(config.NUM_CLIENTS)]

    # ── Initial global parameters (random fresh model) ─────
    init_model = get_model()
    global_params = get_parameters(init_model)

    round_history = []

    # ══════════════════════════════════════════════════════
    # FL ROUNDS
    # ══════════════════════════════════════════════════════
    for rnd in range(1, config.NUM_ROUNDS + 1):
        print(f"\n{'━'*60}")
        print(f"  Round {rnd}/{config.NUM_ROUNDS}")
        print(f"{'━'*60}")

        fit_cfg = {
            "lr":           config.LEARNING_RATE,
            "local_epochs": config.LOCAL_EPOCHS,
            "server_round": rnd,
        }

        # ── Step 1: Local training on every client ─────────
        fit_results = []
        for client in clients:
            updated_params, num_examples, _ = client.fit(
                parameters=copy.deepcopy(global_params),
                config_dict=fit_cfg,
            )
            fit_results.append((updated_params, num_examples))

        # ── Step 2: FedAvg aggregation ─────────────────────
        global_params = fedavg(fit_results)
        print(f"\n  [Server] FedAvg aggregation complete (round {rnd})")

        # ── Step 3: Server-side global eval ────────────────
        g_loss, g_acc = server_evaluate(global_params, test_loader, rnd)

        # ── Step 4: Client-side evaluation ─────────────────
        eval_cfg = {}
        eval_results = []
        for client in clients:
            c_loss, c_n, c_metrics = client.evaluate(
                parameters=copy.deepcopy(global_params),
                config_dict=eval_cfg,
            )
            eval_results.append((c_loss, c_n, c_metrics.get("accuracy", 0.0)))

        # Weighted average of client evaluations
        total_n  = sum(n for _, n, _ in eval_results)
        avg_loss = sum(l * n for l, n, _ in eval_results) / total_n
        avg_acc  = sum(a * n for _, n, a in eval_results) / total_n

        print(f"\n{'─'*55}")
        print(f"  [Round {rnd:>2}] FedAvg eval → "
              f"loss={avg_loss:.4f}  accuracy={avg_acc:.4f}")
        print(f"{'─'*55}")

        # Log FedAvg weighted eval
        metrics_logger.log_fedavg_eval(rnd, avg_loss, avg_acc)
        _append_csv(log_path, rnd, avg_loss, avg_acc)
        round_history.append({
            "round":    rnd,
            "loss":     avg_loss,
            "accuracy": avg_acc,
        })

    # ══════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════
    metrics_logger.set_status("complete")

    print(f"\n{'='*60}")
    print("  [Server] Simulation complete!")
    print(f"  Results CSV  → {log_path}")
    print(f"  Best Acc     → {_best_acc*100:.2f}%")
    print(f"{'='*60}\n")

    return round_history


if __name__ == "__main__":
    import sys
    dist = sys.argv[1] if len(sys.argv) > 1 else config.DISTRIBUTION
    run_simulation(distribution=dist)
