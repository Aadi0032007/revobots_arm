"""Validate the camera and tool configuration shared by both local consoles."""

from dataclasses import dataclass
from pathlib import Path
import math

from revobots_arm.cameras.realsense import RealSenseCapture
from revobots_arm.config import ArmConfig
from revobots_arm.hand_eye.calibration import HandEyeCalibration
from revobots_arm.inverse_kinematics.nero import vector
from .motion import NeroPickMotion


@dataclass(frozen=True)
class GraspSetup:
    calibration: str | Path
    tcp_xyz_m: str
    table_z_m: float
    tcp_rpy_deg: str | None = None
    clearance_m: float = 0.010
    max_travel_m: float = 0.4
    gripper_open: float | None = None
    gripper_close: float | None = None

    def camera_calibration(self, cfg: ArmConfig) -> HandEyeCalibration:
        if cfg.robot_type != "agx_nero_native":
            raise ValueError("Hover and grasp requires a native Nero preset")
        if not 10 <= cfg.fps <= 60:
            raise ValueError("Use arm fps 10–60")
        if (not math.isfinite(self.table_z_m) or not 0.005 <= self.clearance_m <= 0.1
                or not 0 < self.max_travel_m <= 0.5):
            raise ValueError("Use a finite table height, clearance 5–100 mm, and travel at most 0.5 m")
        calibration = HandEyeCalibration.load(Path(self.calibration).expanduser())
        if calibration.robot_id and calibration.robot_id != cfg.robot.get("id"):
            raise ValueError("Hand-eye calibration was recorded for a different robot ID")
        camera = cfg.camera(calibration.camera)
        if not camera.enabled or not camera.is_realsense or not camera.use_depth:
            raise ValueError("Configure the calibrated wrist camera as an enabled RealSense with depth")
        if calibration.image_rotation is not None and camera.rotation != calibration.image_rotation:
            raise ValueError("Camera rotation does not match hand-eye calibration")
        if camera.frame_shape[:2][::-1] != tuple(calibration.image_size_wh):
            raise ValueError("Camera image size does not match hand-eye calibration")
        if (self.gripper_open is None) != (self.gripper_close is None):
            raise ValueError("Specify both gripper positions to enable Grasp")
        if self.gripper_open is not None:
            limit = 0.1 if cfg.robot.get("gripper_mode", "width") == "width" else 100
            if not cfg.robot.get("with_gripper") or not 0 <= self.gripper_close < self.gripper_open <= limit:
                raise ValueError(f"Enable the gripper and enter 0 ≤ close < open ≤ {limit}")
        return calibration

    def motion(self, cfg: ArmConfig, robot, calibration: HandEyeCalibration) -> NeroPickMotion:
        return NeroPickMotion(
            robot, calibration, tcp_xyz_m=vector(self.tcp_xyz_m, 3),
            tcp_rpy_deg=vector(self.tcp_rpy_deg, 3) if self.tcp_rpy_deg else None,
            table_z_m=self.table_z_m, clearance_m=self.clearance_m,
            max_travel_m=self.max_travel_m, fps=cfg.fps,
            gripper_open=self.gripper_open, gripper_close=self.gripper_close,
        )


def check_camera_serial(camera: RealSenseCapture, calibration: HandEyeCalibration) -> None:
    if calibration.camera_serial and camera.serial != calibration.camera_serial:
        raise ValueError("RealSense serial does not match hand-eye calibration")
