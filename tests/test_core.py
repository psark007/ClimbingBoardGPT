from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from climbingboardgpt.config import BoardConfig
from climbingboardgpt.evaluation import nearest_real_route_same_board
from climbingboardgpt.generation import validity_summary
from climbingboardgpt.grades import to_grouped_v
from climbingboardgpt.tokenization import (
    parse_frames,
    parse_tokens,
    tokenize_route,
    tokens_to_hold_records,
)


class CoreBehaviorTest(unittest.TestCase):
    def test_to_grouped_v_clamps_and_maps_display_difficulty(self):
        self.assertEqual(to_grouped_v(10), 0)
        self.assertEqual(to_grouped_v(22), 6)
        self.assertEqual(to_grouped_v(99), 16)

    def test_parse_frames_extracts_placement_role_pairs(self):
        self.assertEqual(parse_frames("p344r5p369r6p603r7"), [(344, 5), (369, 6), (603, 7)])
        self.assertEqual(parse_frames(None), [])

    def test_shared_token_parser_handles_repr_and_sequence_strings(self):
        self.assertEqual(parse_tokens("['<BOS>', '<EOS>']"), ["<BOS>", "<EOS>"])
        self.assertEqual(parse_tokens("<BOS> <EOS>"), ["<BOS>", "<EOS>"])

    def test_tokens_to_hold_records_extracts_shared_shape(self):
        records = tokens_to_hold_records(["<BOS>", "<TB2_p344_start>", "<TB2_p603_finish>"])
        self.assertEqual(
            records,
            [
                {
                    "token": "<TB2_p344_start>",
                    "board_token_prefix": "TB2",
                    "board_prefix": "TB2",
                    "placement_id": 344,
                    "role": "start",
                },
                {
                    "token": "<TB2_p603_finish>",
                    "board_token_prefix": "TB2",
                    "board_prefix": "TB2",
                    "placement_id": 603,
                    "role": "finish",
                },
            ],
        )

    def test_validity_summary_flags_basic_valid_route(self):
        summary = validity_summary(
            ["<TB2_p344_start>", "<TB2_p369_middle>", "<TB2_p603_finish>"],
            requested_board_prefix="TB2",
        )
        self.assertTrue(summary["basic_valid"])
        self.assertEqual(summary["n_hold_tokens"], 3)

    def test_tokenize_route_uses_canonical_hold_order(self):
        config = BoardConfig(
            board_key="tb2",
            display_name="TB2",
            token_prefix="TB2",
            db_path=Path("unused.db"),
            layout_id=1,
            max_angle=None,
            min_fa_date=None,
            placement_y_max=None,
            include_mirror_placement_id=False,
            role_definitions={"start": 5, "middle": 6, "finish": 7, "foot": 8},
        )
        placement_lookup = {
            ("tb2", 603): {"x": 0, "y": 120},
            ("tb2", 344): {"x": 0, "y": 10},
            ("tb2", 369): {"x": 0, "y": 80},
        }
        row = {
            "frames": "p603r7p369r6p344r5",
            "angle": 40,
            "display_difficulty": 22,
        }

        self.assertEqual(
            tokenize_route(row, config, placement_lookup),
            [
                "<BOS>",
                "<BOARD_TB2>",
                "<ANGLE_40>",
                "<GRADE_V6>",
                "<TB2_p344_start>",
                "<TB2_p369_middle>",
                "<TB2_p603_finish>",
                "<EOS>",
            ],
        )

    def test_nearest_real_route_same_board_finds_best_jaccard_match(self):
        real_df = pd.DataFrame(
            [
                {
                    "board_key": "tb2",
                    "hold_set": frozenset({1, 2, 3}),
                    "uuid": "a",
                    "climb_name": "close",
                    "grouped_v": 5,
                    "angle": 40,
                },
                {
                    "board_key": "tb2",
                    "hold_set": frozenset({7, 8}),
                    "uuid": "b",
                    "climb_name": "far",
                    "grouped_v": 8,
                    "angle": 45,
                },
            ]
        )

        result = nearest_real_route_same_board(frozenset({1, 2, 4}), "tb2", real_df)

        self.assertEqual(result["nearest_real_uuid"], "a")
        self.assertAlmostEqual(result["nearest_real_jaccard"], 0.5)


if __name__ == "__main__":
    unittest.main()
