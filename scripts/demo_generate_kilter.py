#!/usr/bin/env python3
"""Convenience wrapper: generate and visualize Kilter routes."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    # Delegate to the generic demo so board-specific wrappers stay tiny.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "demo_generate_and_visualize.py"),
        "--board",
        "kilter",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))
