from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    return current
