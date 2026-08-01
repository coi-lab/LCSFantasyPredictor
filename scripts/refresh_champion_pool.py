"""Refresh draft evidence and current champion recommendations in order."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFRESH_COMMANDS = (
    (sys.executable, "-m", "champion_prediction.draft_actions"),
    (sys.executable, "-m", "champion_prediction.simple_predictor"),
)


def refresh_champion_pool() -> None:
    """Rebuild the draft database before generating champion recommendations."""
    for command in REFRESH_COMMANDS:
        print(f"Running: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    refresh_champion_pool()
