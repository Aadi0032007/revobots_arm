"""Generate a ChArUco target and calibrate from a finished Nero recording."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from revobots_arm.inverse_kinematics.nero import nero_kinematics
from revobots_arm.recording.session import SessionFrames

from .calibration import HandEyeCalibration, solve_hand_eye


def make_board(square_mm: float, marker_mm: float):
    if not 0 < marker_mm < square_mm < float("inf"):
        raise ValueError("Board dimensions must satisfy 0 < marker-mm < square-mm")
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "CharucoDetector"):
        raise ValueError("This OpenCV build needs ArUco and CharucoDetector support")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    return aruco.CharucoBoard((7, 5), square_mm / 1000, marker_mm / 1000, dictionary)


def board_png(board) -> bytes:
    image = board.generateImage((2180, 1580), marginSize=40)
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the calibration board")
    return png.tobytes()


def detect_view(gray, board, detector):
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 8 or board.checkCharucoCornersCollinear(ids):
        return None
    return board.getChessboardCorners()[ids.ravel()], corners


def collect_views(session: SessionFrames, camera: str, board, kin):
    try:
        times = np.array([float(row["timestamp"]) for row in session.rows])
        joints = np.array([kin.motor_to_q(row["observation"]) for row in session.rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The session must contain timestamps and all seven measured Nero joints") from exc
    if not np.isfinite(times).all() or not np.isfinite(joints).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Expected increasing timestamps and finite measured joint angles")

    detector = cv2.aruco.CharucoDetector(board)
    accepted_joints = []
    views = []
    # Joint feedback and video frames are paired by the recorder. Use settled
    # views so capture and feedback latency do not turn into calibration error.
    before = np.searchsorted(times, times - 0.3, side="right") - 1
    after = np.searchsorted(times, times + 0.3)
    image_size = None
    with session:
        for index, (row, images) in enumerate(session):
            left, right = before[index], after[index]
            if left < 0 or right >= len(times):
                continue
            q = joints[index]
            if np.max(np.abs(joints[left : right + 1] - q)) > np.radians(0.4):
                continue
            if any(np.max(np.abs(q - previous)) < np.radians(2) for previous in accepted_joints):
                continue
            gray = cv2.cvtColor(images[camera], cv2.COLOR_RGB2GRAY)
            detected = detect_view(gray, board, detector)
            if detected is None:
                continue
            image_size = gray.shape[::-1]
            accepted_joints.append(q)
            views.append((row["frame_index"], kin.fk(q), *detected))

    return views, image_size


def calibrate(session_dir: Path, camera: str, board) -> HandEyeCalibration:
    session = SessionFrames(session_dir)
    camera_info = session.meta.get("cameras", {}).get(camera)
    if camera_info is None or "depth" in camera_info:
        raise ValueError(f"Expected a recorded colour camera. Available channels: {session.camera_keys}")
    camera_calibration = session.meta.get("camera_calibration", {}).get(camera, {})
    kin = nero_kinematics()
    views, image_size = collect_views(session, camera, board, kin)
    return solve_views(views, image_size, board, kin, camera, session.meta.get("robot_id"), camera_calibration)


def solve_views(views, image_size, board, kin, camera, robot_id, camera_calibration):
    if len(views) < 6:
        raise ValueError(f"Found {len(views)} settled board views. Need at least six distinct views.")
    if len(views) > 40:
        views = [views[i] for i in np.linspace(0, len(views) - 1, 40, dtype=int)]
    indices, links, objects, pixels = zip(*views)
    reprojection, K, D, rotations, translations = cv2.calibrateCamera(
        list(objects), list(pixels), image_size, None, None
    )
    boards = []
    for rotation, translation in zip(rotations, translations):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = translation.ravel()
        boards.append(pose)
    X, board_mm, board_deg = solve_hand_eye(list(links), boards)
    print(f"{len(views)} views · reprojection RMS {reprojection:.2f} px · board RMS {board_mm:.1f} mm / {board_deg:.2f}°")
    if not np.isfinite([reprojection, board_mm, board_deg]).all() or reprojection > 1.0 or board_mm > 10 or board_deg > 2:
        raise ValueError("Calibration residuals exceed 1 px, 10 mm or 2°. Check board scale, camera and settled views.")
    return HandEyeCalibration(
        camera=camera,
        robot_id=robot_id,
        base_link=kin.chain.base_link,
        camera_link=kin.chain.tip_link,
        image_size_wh=image_size,
        camera_matrix=K.tolist(),
        dist_coeffs=D.ravel().tolist(),
        link_from_camera=X.tolist(),
        board_square_mm=1000 * board.getSquareLength(),
        board_marker_mm=1000 * board.getMarkerLength(),
        sample_frames=list(indices),
        reprojection_rms_px=float(reprojection),
        board_translation_rms_mm=board_mm,
        board_rotation_rms_deg=board_deg,
        camera_serial=camera_calibration.get("serial"),
        image_rotation=camera_calibration.get("rotation"),
    )


def save_calibration(calibration, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(asdict(calibration), stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("board", help="Write a printable 7 × 5 ChArUco PNG")
    solve = sub.add_parser("calibrate", help="Calibrate an eye-in-hand Nero wrist camera")
    solve.add_argument("--session", type=Path, required=True, help="Finalized session_<n> directory")
    solve.add_argument("--camera", required=True, help="Recorded colour camera key")
    for command in (render, solve):
        command.add_argument("--square-mm", type=float, default=35.0)
        command.add_argument("--marker-mm", type=float, default=26.0)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        board = make_board(args.square_mm, args.marker_mm)
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "board":
            if output.suffix.lower() != ".png":
                raise ValueError("Board output must use a .png extension")
            output.write_bytes(board_png(board))
            print(f"Saved {output}. Measure a printed square and marker before calibrating.")
        else:
            calibration = calibrate(args.session.expanduser(), args.camera, board)
            save_calibration(calibration, output)
            print(f"Saved {output}")
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        parser.exit(1, f"Hand-eye calibration: {exc}\n")


if __name__ == "__main__":
    main()
