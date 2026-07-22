import subprocess
import sys
from pathlib import Path


def main() -> None:
    r"""Starts the Marimo analysis notebook."""
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "analysis.py"
    result = subprocess.run([sys.executable, "-m", "marimo", "run", str(notebook)], check=False)
    raise SystemExit(result.returncode)
