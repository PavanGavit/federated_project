"""
check_hardware.py — Hardware detection and readiness report.

Run this before training to confirm what device will be used:
    python check_hardware.py
"""

import sys
import platform

# ── Core imports ─────────────────────────────────────────────
try:
    import torch
except ImportError:
    print("❌  PyTorch is not installed. Run:  pip install -r requirements.txt")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}i{RESET}  {msg}")
def sep():      print(f"  {'─'*54}")

# ─────────────────────────────────────────────────────────────
def check_system():
    print(f"\n{BOLD}  System{RESET}")
    sep()
    info(f"OS          : {platform.system()} {platform.release()} ({platform.machine()})")
    info(f"Python      : {sys.version.split()[0]}")
    info(f"PyTorch     : {torch.__version__}")

# ─────────────────────────────────────────────────────────────
def check_cuda():
    print(f"\n{BOLD}  NVIDIA CUDA (GPU){RESET}")
    sep()

    if not torch.cuda.is_available():
        fail("CUDA is NOT available")
        warn("Either no NVIDIA GPU detected, or CPU-only PyTorch is installed.")
        warn("To install CUDA PyTorch (RTX 30/40 series):")
        print(f"  {DIM}pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121{RESET}")
        return False

    n = torch.cuda.device_count()
    ok(f"CUDA available  ({n} device{'s' if n > 1 else ''} found)")

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        info(f"GPU [{i}]     : {props.name}")
        info(f"VRAM [{i}]    : {vram_gb:.1f} GB")
        info(f"Compute [{i}] : SM {props.major}.{props.minor}")

        # Recommend batch size based on VRAM
        if vram_gb >= 20:
            batch_rec = 128
        elif vram_gb >= 10:
            batch_rec = 64
        elif vram_gb >= 6:
            batch_rec = 32
        else:
            batch_rec = 16

        ok(f"Recommended BATCH_SIZE for this GPU: {batch_rec}")

    # Quick smoke test
    try:
        t = torch.zeros(1, device="cuda")
        del t
        ok("CUDA allocation test passed")
    except Exception as e:
        fail(f"CUDA allocation failed: {e}")
        return False

    return True

# ─────────────────────────────────────────────────────────────
def check_mps():
    print(f"\n{BOLD}  Apple Silicon MPS{RESET}")
    sep()
    if not hasattr(torch.backends, "mps"):
        fail("MPS backend not present (PyTorch < 1.12 or non-macOS)")
        return False
    if not torch.backends.mps.is_available():
        fail("MPS is NOT available (not Apple Silicon, or not macOS 12.3+)")
        return False
    ok("MPS (Apple Silicon GPU) is available")
    return True

# ─────────────────────────────────────────────────────────────
def check_cpu():
    print(f"\n{BOLD}  CPU Fallback{RESET}")
    sep()
    import os
    cores = os.cpu_count()
    ok(f"CPU available  ({cores} logical cores)")
    info("Training on CPU is slow. Consider a machine with an NVIDIA GPU.")

# ─────────────────────────────────────────────────────────────
def print_recommendation(cuda_ok, mps_ok):
    print(f"\n{BOLD}  Selected Device (what config.py will use){RESET}")
    sep()

    if cuda_ok:
        ok(f"{GREEN}{BOLD}CUDA (NVIDIA GPU){RESET} — best performance")
        info("config.py → DEVICE='cuda', NUM_WORKERS=4")
    elif mps_ok:
        ok(f"{YELLOW}{BOLD}MPS (Apple Silicon){RESET} — good performance")
        info("config.py → DEVICE='mps', NUM_WORKERS=2")
    else:
        warn(f"{RED}CPU only{RESET} — training will be slow")
        info("config.py → DEVICE='cpu', NUM_WORKERS=0")
        info("To speed up: install CUDA PyTorch on a machine with an NVIDIA GPU.")

    print()

# ─────────────────────────────────────────────────────────────
def main():
    print(f"\n{'═'*58}")
    print(f"  FL-Project  |  Hardware Check")
    print(f"{'═'*58}")

    check_system()
    cuda_ok = check_cuda()
    mps_ok  = check_mps() if not cuda_ok else False
    check_cpu()
    print_recommendation(cuda_ok, mps_ok)

if __name__ == "__main__":
    main()
