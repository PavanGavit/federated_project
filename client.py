"""
client.py — FL Client definition (no Flower runtime required).

Each FlowerClient simulates one medical institution.
server.py calls .fit() and .evaluate() directly in a Python for-loop —
no Ray, no Flower simulation engine needed.

Metrics are written to metrics_logger at every epoch and round end.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import config
import metrics_logger
from model import get_model, get_parameters, set_parameters, evaluate_model, DEVICE
from dataset import build_federated_datasets


# ─────────────────────────────────────────────────────────
# Pre-build all client loaders once (avoids redundant disk IO)
# ─────────────────────────────────────────────────────────
_CLIENT_LOADERS = None
_GLOBAL_TEST_LOADER = None

def _ensure_datasets():
    global _CLIENT_LOADERS, _GLOBAL_TEST_LOADER
    if _CLIENT_LOADERS is None:
        _CLIENT_LOADERS, _GLOBAL_TEST_LOADER, _, _ = build_federated_datasets(
            distribution=config.DISTRIBUTION
        )


# ─────────────────────────────────────────────────────────
# FlowerClient
# ─────────────────────────────────────────────────────────
class FlowerClient:
    """
    Represents a single federated client (medical institution).

    Parameters
    ----------
    client_id : int
        Index of this client in [0, NUM_CLIENTS).
    """

    def __init__(self, client_id: int):
        _ensure_datasets()
        self.client_id    = client_id
        self.train_loader = _CLIENT_LOADERS[client_id]
        self.model        = get_model()
        self.criterion    = nn.CrossEntropyLoss()

    # ── Core interface (called directly by server.py) ─────
    def get_parameters(self, config_dict=None):
        return get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray],
            config_dict: dict) -> tuple[list[np.ndarray], int, dict]:
        """Receive global weights, train locally, return updated weights."""
        set_parameters(self.model, parameters)

        lr           = config_dict.get("lr",           config.LEARNING_RATE)
        epochs       = config_dict.get("local_epochs",  config.LOCAL_EPOCHS)
        server_round = config_dict.get("server_round",  1)

        optimizer = Adam(self.model.parameters(),
                         lr=lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        self.model.train()
        last_loss, last_acc = 0.0, 0.0

        for epoch in range(epochs):
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(images)
                loss   = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total   += images.size(0)
            scheduler.step()

            last_loss = running_loss / (total or 1)
            last_acc  = correct / (total or 1)
            print(f"  [Client {self.client_id}] Epoch {epoch+1}/{epochs} "
                  f"- loss: {last_loss:.4f}  acc: {last_acc:.4f}")

            # ── Log epoch metrics ──────────────────────────
            metrics_logger.log_client_epoch(
                rnd=server_round,
                client_id=self.client_id,
                epoch=epoch + 1,
                loss=last_loss,
                accuracy=last_acc,
            )

        num_examples = len(self.train_loader.dataset)

        # ── Log final round metrics for this client ────────
        metrics_logger.log_client_round(
            rnd=server_round,
            client_id=self.client_id,
            final_loss=last_loss,
            final_accuracy=last_acc,
            num_examples=num_examples,
        )

        return get_parameters(self.model), num_examples, {}

    def evaluate(self, parameters: list[np.ndarray],
                 config_dict: dict) -> tuple[float, int, dict]:
        """Evaluate global model weights on the shared test set."""
        set_parameters(self.model, parameters)
        loss, acc, _, _ = evaluate_model(self.model, _GLOBAL_TEST_LOADER)
        num_examples     = len(_GLOBAL_TEST_LOADER.dataset)
        return float(loss), num_examples, {"accuracy": float(acc)}
