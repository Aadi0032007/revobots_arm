import unittest
from unittest.mock import patch

import cv2
import numpy as np

from revobots_arm.hand_eye.calibration import HandEyeCalibration, solve_hand_eye
from revobots_arm.inverse_kinematics.core import rotvec_to_matrix


def pose(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotvec_to_matrix(rotation)
    transform[:3, 3] = translation
    return transform


class HandEyeTests(unittest.TestCase):
    def test_recovers_camera_to_link_from_a_stationary_board(self):
        link_from_camera = pose([0.08, -0.12, 0.32], [0.05, -0.02, 0.1])
        base_from_board = pose([0, 0.4, 0], [0.25, -0.1, 0.45])
        base_from_links = [
            pose([0, 0, 0], [0.1, 0, 0.2]),
            pose([0.3, 0.05, 0], [0.1, 0.1, 0.2]),
            pose([-0.1, 0.35, 0.15], [0.16, 0.04, 0.3]),
            pose([0.05, -0.2, 0.2], [0.25, -0.03, 0.17]),
        ]
        camera_from_boards = [
            np.linalg.solve(link @ link_from_camera, base_from_board)
            for link in base_from_links
        ]
        solution = solve_hand_eye(base_from_links, camera_from_boards)
        # The Python binding is absent from some OpenCV builds.
        with patch("revobots_arm.hand_eye.calibration.hasattr", return_value=False):
            fallback = solve_hand_eye(base_from_links, camera_from_boards)
        for transform, translation_rms, rotation_rms in (solution, fallback):
            np.testing.assert_allclose(transform, link_from_camera, atol=1e-7)
            self.assertLess(translation_rms, 1e-5)
            self.assertLess(rotation_rms, 1e-5)

    def test_single_axis_motion_is_rejected(self):
        links = [pose([0, 0, angle], [0, 0, 0]) for angle in (0, 0.25, 0.5)]
        with self.assertRaisesRegex(ValueError, "second axis"):
            solve_hand_eye(links, [np.linalg.inv(link) for link in links])

    def test_pixel_conversion_uses_current_link_pose(self):
        camera_matrix = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
        link_from_camera = pose([0.1, 0.2, 0.3], [0.01, 0.02, 0.07])
        calibration = HandEyeCalibration(
            camera="wrist", robot_id="nero", base_link="base_link", camera_link="link7",
            image_size_wh=(640, 480), camera_matrix=camera_matrix.tolist(),
            dist_coeffs=[0] * 5, link_from_camera=link_from_camera.tolist(),
            board_square_mm=35, board_marker_mm=26, sample_frames=[1, 5, 12],
            reprojection_rms_px=0.2, board_translation_rms_mm=1.0, board_rotation_rms_deg=0.2,
        )
        base_from_link = pose([-0.1, 0.2, 0.03], [0.4, 0.2, 0.3])
        camera_point = np.array([0.01, -0.03, 0.6])
        pixel, _ = cv2.projectPoints(
            camera_point.reshape(1, 3), np.zeros(3), np.zeros(3), camera_matrix, None
        )
        expected = (base_from_link @ link_from_camera @ np.r_[camera_point, 1])[:3]
        np.testing.assert_allclose(
            calibration.pixel_to_base(tuple(pixel.ravel()), 0.6, base_from_link), expected
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            calibration.pixel_to_base(tuple(pixel.ravel()), 0.0, base_from_link)


if __name__ == "__main__":
    unittest.main()
