#!/usr/bin/env python3
"""Convenience wrapper: predict grade for a TB2 frames string."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "demo_predict_grade.py"),
        "--board",
        "tb2",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))
