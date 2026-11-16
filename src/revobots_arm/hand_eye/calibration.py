"""Eye-in-hand transforms and calibration from observed board corners."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np

from revobots_arm.inverse_kinematics.core import matrix_to_rotvec


@dataclass
class HandEyeCalibration:
    camera: str
    robot_id: str | None
    base_link: str
    camera_link: str
    image_size_wh: tuple[int, int]
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    link_from_camera: list[list[float]]
    board_square_mm: float
    board_marker_mm: float
    sample_frames: list[int]
    reprojection_rms_px: float
    board_translation_rms_mm: float
    board_rotation_rms_deg: float
    version: int = 1

    def __post_init__(self) -> None:
        K = np.asarray(self.camera_matrix, dtype=float)
        D = np.asarray(self.dist_coeffs, dtype=float)
        X = np.asarray(self.link_from_camera, dtype=float)
        R = X[:3, :3] if X.shape == (4, 4) else np.empty((0, 0))
        if (
            self.version != 1
            or len(self.image_size_wh) != 2
            or min(self.image_size_wh) <= 0
            or K.shape != (3, 3)
            or not np.isfinite(K).all()
            or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or D.shape != (5,)
            or not np.isfinite(D).all()
            or X.shape != (4, 4)
            or not np.isfinite(X).all()
            or not np.allclose(X[3], [0, 0, 0, 1])
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R), 1.0)
        ):
            raise ValueError("Invalid or unsupported hand-eye calibration")

    @classmethod
    def load(cls, path: str | Path) -> HandEyeCalibration:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def base_from_camera(self, base_from_link: np.ndarray) -> np.ndarray:
        """Camera-to-base transform using the mounting link pose at image capture."""
        return np.asarray(base_from_link) @ np.asarray(self.link_from_camera)

    def camera_to_base(self, point_m: np.ndarray, base_from_link: np.ndarray) -> np.ndarray:
        """Transform one point or an array of points from camera to base, in metres."""
        transform = self.base_from_camera(base_from_link)
        return np.asarray(point_m) @ transform[:3, :3].T + transform[:3, 3]

    def pixel_to_base(
        self, pixel_xy: tuple[float, float], depth_m: float, base_from_link: np.ndarray
    ) -> np.ndarray:
        """Deproject a colour pixel using axial depth already aligned to that pixel."""
        if not np.isfinite(depth_m) or depth_m <= 0:
            raise ValueError("Aligned depth must be positive and finite")
        pixel = np.asarray(pixel_xy, dtype=float).reshape(1, 1, 2)
        if (
            not np.isfinite(pixel).all()
            or np.any(pixel < 0)
            or np.any(pixel >= self.image_size_wh)
        ):
            raise ValueError("Pixel is outside the calibrated colour image")
        normalized = cv2.undistortPoints(
            pixel, np.asarray(self.camera_matrix, dtype=float), np.asarray(self.dist_coeffs, dtype=float)
        ).reshape(2)
        return self.camera_to_base(depth_m * np.r_[normalized, 1.0], base_from_link)


def _solve_relative_motions(links: np.ndarray, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve A X = X B for OpenCV builds missing the Python hand-eye binding."""
    left, right = [], []
    for i, j in combinations(range(len(links)), 2):
        left.append(np.linalg.solve(links[j], links[i]))
        right.append(boards[j] @ np.linalg.inv(boards[i]))
    left, right = np.asarray(left), np.asarray(right)

    # Rotation vectors satisfy log(R_A) = R_X log(R_B).
    left_vectors = np.array([matrix_to_rotvec(A[:3, :3]) for A in left])
    right_vectors = np.array([matrix_to_rotvec(B[:3, :3]) for B in right])
    U, _, Vt = np.linalg.svd(left_vectors.T @ right_vectors)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    # (R_A - I) t_X = R_X t_B - t_A.
    coefficients = (left[:, :3, :3] - np.eye(3)).reshape(-1, 3)
    values = (right[:, :3, 3] @ R.T - left[:, :3, 3]).ravel()
    t, _, rank, _ = np.linalg.lstsq(coefficients, values, rcond=None)
    if rank != 3:
        raise ValueError("Hand-eye translation is underconstrained. Record more varied views.")
    return R, t


def solve_hand_eye(
    base_from_links: list[np.ndarray], camera_from_boards: list[np.ndarray]
) -> tuple[np.ndarray, float, float]:
    """Return link-from-camera and the RMS spread of the board in the base frame."""
    if len(base_from_links) != len(camera_from_boards) or len(base_from_links) < 3:
        raise ValueError("Hand-eye calibration needs at least three paired poses")
    links = np.asarray(base_from_links, dtype=float)
    boards = np.asarray(camera_from_boards, dtype=float)
    rotations = [matrix_to_rotvec(link[:3, :3] @ links[0, :3, :3].T) for link in links[1:]]
    if np.linalg.svd(rotations, compute_uv=False)[1] < np.radians(8):
        raise ValueError("Too little wrist rotation about a second axis. Record more varied views.")

    if hasattr(cv2, "calibrateHandEye"):
        R, t = cv2.calibrateHandEye(
            list(links[:, :3, :3]), list(links[:, :3, 3]),
            list(boards[:, :3, :3]), list(boards[:, :3, 3]),
            method=cv2.CALIB_HAND_EYE_PARK,
        )
    else:
        R, t = _solve_relative_motions(links, boards)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.ravel()
    if not np.isfinite(X).all() or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Hand-eye solve failed. Check the board and record more varied views.")

    poses = links @ X @ boards
    positions = poses[:, :3, 3]
    position_errors = positions - positions.mean(axis=0)
    translation_rms_mm = 1000 * np.sqrt(np.mean(np.sum(position_errors ** 2, axis=1)))
    U, _, Vt = np.linalg.svd(poses[:, :3, :3].mean(axis=0))
    mean_rotation = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    angles = [np.linalg.norm(matrix_to_rotvec(mean_rotation.T @ pose[:3, :3])) for pose in poses]
    rotation_rms_deg = np.degrees(np.sqrt(np.mean(np.square(angles))))
    return X, float(translation_rms_mm), float(rotation_rms_deg)
