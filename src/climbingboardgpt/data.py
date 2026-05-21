"""
Database loading for ClimbingBoardGPT.

This module queries SQLite databases for climb and placement data,
applying board-specific filters defined in the configuration.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import BoardConfig
from .paths import find_project_root


def build_climbs_query(config: BoardConfig) -> tuple[str, list]:
    """Build a SQL query for climbs data with board-specific filters.
    
    The query joins climbs, layouts, products, climb_stats, and difficulty_grades
    tables, applying filters for:
    - layout_id: Which board layout to use
    - max_angle: Exclude routes steeper than this
    - min_fa_date: Exclude routes first ascended before this date
    - display_difficulty IS NOT NULL: Only routes with difficulty ratings
    - is_listed = 1: Only publicly listed routes
    
    Args:
        config: Board configuration
        
    Returns:
        Tuple of (SQL query string, list of query parameters)
    """
    conditions = [
        "cs.display_difficulty IS NOT NULL",
        "c.is_listed = 1",
        "c.layout_id = ?",
    ]
    params: list = [config.layout_id]

    if config.max_angle is not None:
        conditions.append("cs.angle <= ?")
        params.append(config.max_angle)

    if config.min_fa_date is not None:
        conditions.append("cs.fa_at > ?")
        params.append(config.min_fa_date)

    query = f"""
    SELECT
        c.uuid,
        c.name AS climb_name,
        c.setter_username,
        c.layout_id AS layout_id,
        c.description,
        c.is_nomatch,
        c.is_listed,
        l.name AS layout_name,
        p.name AS board_name,
        c.frames,
        cs.angle,
        cs.display_difficulty,
        dg.boulder_name AS boulder_grade,
        cs.ascensionist_count,
        cs.quality_average,
        cs.fa_at
    FROM climbs c
    JOIN layouts l ON c.layout_id = l.id
    JOIN products p ON l.product_id = p.id
    JOIN climb_stats cs ON c.uuid = cs.climb_uuid
    JOIN difficulty_grades dg ON ROUND(cs.display_difficulty) = dg.difficulty
    WHERE {' AND '.join(conditions)}
    """
    return query, params


def build_placements_query(config: BoardConfig) -> tuple[str, list]:
    """Build a SQL query for placement data with board-specific filters.
    
    The query retrieves hold positions, default roles, material types,
    and (optionally) mirror placement IDs for symmetric holds.
    
    Args:
        config: Board configuration
        
    Returns:
        Tuple of (SQL query string, list of query parameters)
    """
    params: list = [config.layout_id]
    y_condition = ""
    if config.placement_y_max is not None:
        y_condition = " AND h.y <= ?"
        params.append(config.placement_y_max)

    if config.include_mirror_placement_id:
        # TB2 has mirrored holds — include the mirror placement ID
        query = f"""
        SELECT
            p.id AS placement_id,
            h.x,
            h.y,
            p.default_placement_role_id AS default_role_id,
            p.set_id AS set_id,
            s.name AS set_name,
            p_mirror.id AS mirror_placement_id
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        JOIN sets s ON p.set_id = s.id
        LEFT JOIN holes h_mirror ON h.mirrored_hole_id = h_mirror.id
        LEFT JOIN placements p_mirror
            ON p_mirror.hole_id = h_mirror.id
           AND p_mirror.layout_id = p.layout_id
        WHERE p.layout_id = ?{y_condition}
        """
    else:
        # Kilter doesn't have mirrored holds
        query = f"""
        SELECT
            p.id AS placement_id,
            h.x,
            h.y,
            p.default_placement_role_id AS default_role_id,
            p.set_id AS set_id,
            s.name AS set_name,
            NULL AS mirror_placement_id
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        JOIN sets s ON p.set_id = s.id
        WHERE p.layout_id = ?{y_condition}
        """
    return query, params


def load_board_data(
    config: BoardConfig,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load climbs and placements data for a single board.
    
    Args:
        config: Board configuration
        project_root: Path to project root (for resolving db_path)
        
    Returns:
        Tuple of (climbs DataFrame, placements DataFrame)
    """
    project_root = Path(project_root) if project_root is not None else find_project_root()
    db_path = config.resolve_db_path(project_root)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Could not find database for board '{config.board_key}': {db_path}"
        )

    climbs_query, climbs_params = build_climbs_query(config)
    placements_query, placements_params = build_placements_query(config)

    with sqlite3.connect(db_path) as conn:
        df_climbs = pd.read_sql_query(climbs_query, conn, params=climbs_params)
        df_placements = pd.read_sql_query(placements_query, conn, params=placements_params)

    # Add board identifiers for multi-board processing
    df_climbs["board_key"] = config.board_key
    df_climbs["board_token_prefix"] = config.token_prefix
    df_climbs["board_display_name"] = config.display_name

    df_placements["board_key"] = config.board_key
    df_placements["board_token_prefix"] = config.token_prefix
    df_placements["board_display_name"] = config.display_name

    return df_climbs, df_placements


def load_multi_board_data(
    configs: list[BoardConfig],
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate data from multiple boards.
    
    This function loads data from each board's database and concatenates
    them into unified DataFrames. Board identifiers are preserved in
    the board_key column.
    
    Args:
        configs: List of board configurations
        project_root: Path to project root
        
    Returns:
        Tuple of (combined climbs DataFrame, combined placements DataFrame)
    """
    climb_frames = []
    placement_frames = []

    for config in configs:
        climbs, placements = load_board_data(config, project_root=project_root)
        climb_frames.append(climbs)
        placement_frames.append(placements)

    return (
        pd.concat(climb_frames, ignore_index=True),
        pd.concat(placement_frames, ignore_index=True),
    )