"""
train.py — Entry point for running IID and/or Non-IID federated training.

Usage:
    python train.py              # uses config.DISTRIBUTION
    python train.py iid
    python train.py non_iid
    python train.py both         # runs IID then Non-IID sequentially
"""

import sys
import config
from server import run_simulation
from typing import List


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else config.DISTRIBUTION

    if mode == "both":
        print("\n>>> Running IID Simulation <<<")
        run_simulation("iid")
        print("\n>>> Running Non-IID Simulation <<<")
        run_simulation("non_iid")
    elif mode in ("iid", "non_iid"):
        run_simulation(mode)
    else:
        print(f"[Error] Unknown distribution: '{mode}'. Use iid | non_iid | both")
        sys.exit(1)


if __name__ == "__main__":
    main()
