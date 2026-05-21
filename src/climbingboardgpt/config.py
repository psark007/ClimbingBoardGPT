"""
Board configuration management for ClimbingBoardGPT.

This module handles loading and parsing board-specific configuration from
JSON files. Each board (TB2, Kilter) has different:
- Layout IDs
- Role ID mappings (start/middle/finish/foot)
- Angle cutoffs
- Database paths
- Token prefixes

The config-driven approach means adding a new board only requires
creating a new JSON file, not modifying code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import find_project_root


@dataclass(frozen=True)
class BoardConfig:
    """Configuration for a single climbing board.
    
    This dataclass stores all board-specific settings needed for
    data loading, tokenization, and model training.
    
    Attributes:
        board_key: Short identifier (e.g., "tb2", "kilter")
        display_name: Human-readable name (e.g., "Tension Board 2 Mirror")
        token_prefix: Namespace for hold tokens (e.g., "TB2", "KILTER")
        db_path: Path to the SQLite database
        layout_id: Which layout in the database to use
        max_angle: Filter out routes steeper than this (None = no filter)
        min_fa_date: Filter out routes first ascended before this date
        placement_y_max: Filter out placements above this Y coordinate
        include_mirror_placement_id: Whether to include mirror info (TB2 only)
        role_definitions: Maps semantic role names to numeric IDs
        boardlib_database_command: Command to download the database
        boardlib_images_command: Command to download board images
        notes: Additional notes about the configuration
    """
    board_key: str
    display_name: str
    token_prefix: str
    db_path: Path
    layout_id: int
    max_angle: float | None
    min_fa_date: str | None
    placement_y_max: float | None
    include_mirror_placement_id: bool
    role_definitions: dict[str, int]
    boardlib_database_command: str | None = None
    boardlib_images_command: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def role_id_to_name(self) -> dict[int, str]:
        """Reverse mapping from numeric role IDs to semantic role names.
        
        Example: {5: 'start', 6: 'middle', 7: 'finish', 8: 'foot'} for TB2
        """
        return {int(role_id): name for name, role_id in self.role_definitions.items()}

    @property
    def board_token(self) -> str:
        """The special token representing this board.
        
        Example: "<BOARD_TB2>" or "<BOARD_KILTER>"
        """
        return f"<BOARD_{self.token_prefix}>"

    def resolve_db_path(self, project_root: Path | None = None) -> Path:
        """Resolve the database path relative to the project root.
        
        If db_path is absolute, return it as-is.
        Otherwise, resolve it relative to the project root.
        """
        project_root = project_root or find_project_root()
        return self.db_path if self.db_path.is_absolute() else project_root / self.db_path


def load_board_config(board_key: str, config_dir: str | Path | None = None) -> BoardConfig:
    """Load a single board configuration from a JSON file.
    
    Args:
        board_key: Board identifier (e.g., "tb2", "kilter")
        config_dir: Directory containing config JSON files
        
    Returns:
        BoardConfig dataclass with all board settings
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    project_root = find_project_root()
    config_dir = Path(config_dir) if config_dir is not None else project_root / "configs"
    path = config_dir / f"{board_key}.json"
    if not path.exists():
        available = sorted(p.stem for p in config_dir.glob("*.json"))
        raise FileNotFoundError(
            f"Unknown board config '{board_key}'. Available: {available}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    return BoardConfig(
        board_key=str(payload["board_key"]),
        display_name=str(payload["display_name"]),
        token_prefix=str(payload["token_prefix"]),
        db_path=Path(payload["db_path"]),
        layout_id=int(payload["layout_id"]),
        max_angle=None if payload.get("max_angle") is None else float(payload["max_angle"]),
        min_fa_date=payload.get("min_fa_date"),
        placement_y_max=None if payload.get("placement_y_max") is None else float(payload["placement_y_max"]),
        include_mirror_placement_id=bool(payload.get("include_mirror_placement_id", False)),
        role_definitions={str(k): int(v) for k, v in payload["role_definitions"].items()},
        boardlib_database_command=payload.get("boardlib_database_command"),
        boardlib_images_command=payload.get("boardlib_images_command"),
        notes=tuple(payload.get("notes", [])),
    )


def load_board_configs(board_keys: list[str] | tuple[str, ...]) -> list[BoardConfig]:
    """Load multiple board configurations.
    
    Args:
        board_keys: List of board identifiers
        
    Returns:
        List of BoardConfig dataclasses
    """
    return [load_board_config(board_key) for board_key in board_keys]


def parse_board_keys(value: str | None, default: tuple[str, ...] = ("tb2", "kilter")) -> list[str]:
    """Parse a comma-separated string of board keys.
    
    Args:
        value: Comma-separated string (e.g., "tb2,kilter") or None
        default: Default board keys if value is None or empty
        
    Returns:
        List of board key strings
    """
    if value is None or not value.strip():
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]