import unittest

import numpy as np

from examples.LIBERO.eval_files.eval_lap_checkpoint import build_eval_row


class EvalLAPCheckpointTest(unittest.TestCase):
    def test_build_eval_row_records_language_and_interp_error(self):
        gt_action = np.zeros((4, 7), dtype=np.float32)
        gt_action[:, 0] = 2.0
        gt_action[:, 6] = 1.0

        row = build_eval_row(
            sample_id=3,
            instruction="pick up the cup",
            pred_language_action="move forward 8 cm, close gripper",
            gt_action=gt_action,
            horizon=4,
        )

        self.assertEqual(row["sample_id"], 3)
        self.assertEqual(row["instruction"], "pick up the cup")
        self.assertEqual(row["pred_language_action"], "move forward 8 cm, close gripper")
        self.assertIn("move forward 8 cm", row["gt_language_action"])
        self.assertTrue(row["parse_ok"])
        self.assertTrue(row["gripper_match"])
        self.assertAlmostEqual(row["lap_interp_mse"], 0.0)

    def test_build_eval_row_marks_bad_parse_without_crashing(self):
        gt_action = np.zeros((2, 7), dtype=np.float32)

        row = build_eval_row(
            sample_id=1,
            instruction="open the drawer",
            pred_language_action=None,
            gt_action=gt_action,
            horizon=2,
        )

        self.assertFalse(row["parse_ok"])
        self.assertIsNone(row["lap_interp_mse"])
        self.assertIsNone(row["gripper_match"])


if __name__ == "__main__":
    unittest.main()
