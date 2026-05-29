"""Grade-scale helpers for BoardLib display difficulty and grouped V grades."""
from __future__ import annotations

# BoardLib display difficulties are integer-like values. This project groups
# them into V grades so TB2 and Kilter can share a compact grade-token space.
GRADE_TO_V = {
    10: 0, 11: 0, 12: 0,
    13: 1, 14: 1,
    15: 2,
    16: 3, 17: 3,
    18: 4, 19: 4,
    20: 5, 21: 5,
    22: 6,
    23: 7,
    24: 8, 25: 8,
    26: 9,
    27: 10,
    28: 11,
    29: 12,
    30: 13,
    31: 14,
    32: 15,
    33: 16,
}


def to_grouped_v(display_difficulty: float) -> int:
    """Map a continuous display difficulty to the nearest grouped V grade."""
    rounded = int(round(float(display_difficulty)))
    rounded = max(min(rounded, max(GRADE_TO_V)), min(GRADE_TO_V))
    return GRADE_TO_V[rounded]


def grade_token(display_difficulty: float) -> str:
    """Return the grade-conditioning token for a display difficulty value."""
    return f"<GRADE_V{to_grouped_v(display_difficulty)}>"
