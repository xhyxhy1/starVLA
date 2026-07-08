import unittest

import numpy as np

from starVLA.model.modules.action_model.lap_textualize import language_action_to_interpolated_action


class LAPTextualizeTest(unittest.TestCase):
    def test_language_action_to_interpolated_action_parses_lap_text(self):
        action = language_action_to_interpolated_action(
            "move forward 8 cm, move left 4 cm, move up 2 cm, "
            "tilt right 16 degrees, tilt forward 8 degrees, "
            "rotate clockwise 4 degrees, close gripper",
            horizon=4,
        )

        self.assertEqual(action.shape, (4, 7))
        np.testing.assert_allclose(action[:, 0], np.full(4, 2.0))
        np.testing.assert_allclose(action[:, 1], np.full(4, 1.0))
        np.testing.assert_allclose(action[:, 2], np.full(4, 0.5))
        np.testing.assert_allclose(action[:, 3], np.full(4, -np.deg2rad(4.0)))
        np.testing.assert_allclose(action[:, 4], np.full(4, np.deg2rad(2.0)))
        np.testing.assert_allclose(action[:, 5], np.full(4, -np.deg2rad(1.0)))
        np.testing.assert_allclose(action[:, 6], np.ones(4))

    def test_language_action_to_interpolated_action_tolerates_missing_components(self):
        action = language_action_to_interpolated_action("open gripper", horizon=3)

        self.assertEqual(action.shape, (3, 7))
        np.testing.assert_allclose(action[:, :6], np.zeros((3, 6)))
        np.testing.assert_allclose(action[:, 6], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
