"""Path discovery helpers for scripts that can be launched from any directory."""
from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upward until the repository root markers are found.

    The project root is identified by both ``pyproject.toml`` and ``configs``.
    If neither marker pair is found, the resolved starting directory is returned
    so callers still have a deterministic base path.
    """
    current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    return current
