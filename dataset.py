"""
dataset.py — Dataset loading, preprocessing, and federated partitioning.

Supports:
  • IID  : data shuffled and split equally among clients
  • Non-IID: Dirichlet allocation (α) per class label
Mask images (filenames ending in '_mask') are excluded.
"""

import os
import re
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import config

# ───────────────────────────────────────────────────────────────
# Transforms
# ───────────────────────────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ───────────────────────────────────────────────────────────────
# Core Dataset Class
# ───────────────────────────────────────────────────────────────
class BreastUltrasoundDataset(Dataset):
    """
    Loads breast-ultrasound images (PNG) from data/{normal,benign,malignant}/.
    Mask images (contain '_mask' in filename) are excluded.
    """

    LABEL_MAP = {cls: idx for idx, cls in enumerate(config.CLASSES)}

    def __init__(self, root: str = config.DATA_DIR, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for class_name in config.CLASSES:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class folder not found: {class_dir}")
            label = self.LABEL_MAP[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(".png") and "_mask" not in fname.lower():
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"[Dataset] Loaded {len(self.samples)} images across {config.NUM_CLASSES} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict:
        counts = {c: 0 for c in config.CLASSES}
        for _, lbl in self.samples:
            counts[config.CLASSES[lbl]] += 1
        return counts


# ───────────────────────────────────────────────────────────────
# Train/Test Split helpers
# ───────────────────────────────────────────────────────────────
def stratified_split(dataset: BreastUltrasoundDataset,
                     test_ratio: float = config.TEST_SPLIT,
                     seed: int = config.SEED):
    """
    Returns (train_indices, test_indices) using stratified sampling.
    """
    rng = random.Random(seed)
    labels = [lbl for _, lbl in dataset.samples]

    class_indices: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(i)

    train_idx, test_idx = [], []
    for lbl, idxs in class_indices.items():
        rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_ratio))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])

    return train_idx, test_idx


# ───────────────────────────────────────────────────────────────
# Federated Partitioning
# ───────────────────────────────────────────────────────────────
def iid_partition(train_indices: list[int],
                  num_clients: int = config.NUM_CLIENTS,
                  seed: int = config.SEED) -> list[list[int]]:
    """
    Evenly shuffle and split indices among clients (IID).
    """
    rng = random.Random(seed)
    shuffled = list(train_indices)
    rng.shuffle(shuffled)
    splits = np.array_split(shuffled, num_clients)
    return [list(s) for s in splits]


def non_iid_partition(train_indices: list[int],
                      labels: list[int],
                      num_clients: int = config.NUM_CLIENTS,
                      alpha: float = config.ALPHA,
                      seed: int = config.SEED) -> list[list[int]]:
    """
    Dirichlet-based Non-IID partition.
    Lower α → more heterogeneous label distribution per client.
    """
    np.random.seed(seed)
    num_classes = config.NUM_CLASSES

    # Group indices by class
    class_indices: dict[int, list[int]] = {}
    for idx in train_indices:
        lbl = labels[idx]
        class_indices.setdefault(lbl, []).append(idx)

    # Dirichlet draw
    client_buckets: list[list[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idxs = np.array(class_indices.get(c, []))
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idxs)).astype(int)
        # fix rounding
        proportions[-1] = len(idxs) - proportions[:-1].sum()
        ptr = 0
        for cid, cnt in enumerate(proportions):
            client_buckets[cid].extend(idxs[ptr: ptr + cnt].tolist())
            ptr += cnt

    return client_buckets


# ───────────────────────────────────────────────────────────────
# High-level builder
# ───────────────────────────────────────────────────────────────
def build_federated_datasets(distribution: str = config.DISTRIBUTION):
    """
    Returns:
        client_train_loaders : list of DataLoader (one per client)
        global_test_loader   : DataLoader for global evaluation
        full_dataset         : BreastUltrasoundDataset (with eval transform)
        partition_info       : dict with class counts per client (for logging)
    """
    # Full datasets (train augments, eval clean)
    train_ds = BreastUltrasoundDataset(transform=TRAIN_TRANSFORMS)
    eval_ds  = BreastUltrasoundDataset(transform=EVAL_TRANSFORMS)

    all_labels = [lbl for _, lbl in train_ds.samples]
    train_idx, test_idx = stratified_split(train_ds)

    # Partition
    if distribution == "iid":
        client_partitions = iid_partition(train_idx)
    else:
        client_partitions = non_iid_partition(train_idx, all_labels)

    # Build per-client DataLoaders (train datasets use augmentation)
    client_train_loaders = []
    partition_info = {}
    for cid, idxs in enumerate(client_partitions):
        subset = Subset(train_ds, idxs)
        loader = DataLoader(subset, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=0, pin_memory=False)
        client_train_loaders.append(loader)
        # Count labels for diagnostics
        label_counts = [0] * config.NUM_CLASSES
        for i in idxs:
            label_counts[all_labels[i]] += 1
        partition_info[f"client_{cid}"] = {
            cls: label_counts[j] for j, cls in enumerate(config.CLASSES)
        }

    # Global test DataLoader
    test_subset  = Subset(eval_ds, test_idx)
    global_test_loader = DataLoader(test_subset, batch_size=config.BATCH_SIZE,
                                    shuffle=False, num_workers=0)

    return client_train_loaders, global_test_loader, eval_ds, partition_info


def get_single_client_loader(client_id: int,
                              distribution: str = config.DISTRIBUTION):
    """Lightweight helper used by Flower clients during simulation."""
    loaders, _, _, _ = build_federated_datasets(distribution)
    return loaders[client_id]
