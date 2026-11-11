"""Portable command entry point; no downloads or raw-file writes."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cos1.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="COS1: audit, FFT resampling and real-signal plots"
    )
    p.add_argument(
        "command", choices=["audit", "resample", "validate", "figures", "all"]
    )
    run(p.parse_args().command)
