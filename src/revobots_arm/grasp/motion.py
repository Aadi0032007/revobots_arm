"""Preflight and execute top-down tabletop moves at the Nero tool centre point."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Event
import time
from typing import Callable

import numpy as np

from revobots_arm.hand_eye.calibration import HandEyeCalibration
from revobots_arm.inverse_kinematics.core import matrix_to_rotvec, rpy_to_matrix
from revobots_arm.inverse_kinematics.nero import nero_kinematics, plan_cartesian
from revobots_arm.robots.nero.driver import NeroJointRobot


POSE_TOL_RAD = math.radians(0.75)
POSE_TOL_M = 0.005


class MotionStopped(RuntimeError):
    pass


def check_stop(stop: Event) -> None:
    if stop.is_set():
        raise MotionStopped("Stopped")


def surface_depth_m(depth: np.ndarray, pixel: tuple[int, int]) -> float:
    """Use samples near the marked surface while excluding holes and other surfaces."""
    x, y = pixel
    if depth.ndim != 3 or depth.shape[2] != 1 or depth.dtype != np.uint16:
        raise ValueError("Expected an aligned depth image in uint16 millimetres")
    h, w, _ = depth.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Target is outside the image")
    samples = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3, 0].astype(int)
    centre = int(depth[y, x, 0])
    near = samples[(samples > 0) & (samples < 65535) & (np.abs(samples - centre) <= 15)]
    if centre in (0, 65535) or near.size < 5:
        raise ValueError("Insufficient depth on the marked surface. Select a point away from its edge.")
    return float(np.median(near)) / 1000


@dataclass(frozen=True)
class Step:
    name: str
    path: np.ndarray | None = None
    dt: float = 0.0
    gripper: float | None = None


@dataclass(frozen=True)
class Plan:
    kind: str
    start: np.ndarray
    hover: np.ndarray
    grasp: np.ndarray | None
    steps: tuple[Step, ...]


class NeroPickMotion:
    def __init__(
        self, robot: NeroJointRobot, calibration: HandEyeCalibration, *,
        tcp_xyz_m: np.ndarray, tcp_rpy_deg: np.ndarray | None, table_z_m: float,
        clearance_m: float, max_travel_m: float, fps: int,
        gripper_open: float | None, gripper_close: float | None,
    ) -> None:
        self.robot = robot
        self.calibration = calibration
        self.wrist = nero_kinematics()
        self.tool = nero_kinematics(tip_link="gripper_tcp", tcp_xyz_m=tcp_xyz_m, tcp_rpy_deg=tcp_rpy_deg)
        if (calibration.base_link, calibration.camera_link) != ("base_link", "link7"):
            raise ValueError("Grasp requires a native Nero base_link/link7 hand-eye calibration")
        self.floor = table_z_m + clearance_m
        self.max_travel = max_travel_m
        self.fps = fps
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close

    @property
    def can_grasp(self) -> bool:
        return self.gripper_open is not None and self.gripper_close is not None

    def joints(self) -> np.ndarray:
        return self.tool.motor_to_q(self.robot.get_observation())

    def check_clearance(self, q: np.ndarray) -> None:
        base_from_wrist = self.wrist.fk(q)
        heights = {
            "gripper tip": self.tool.fk(q)[2, 3],
            "wrist": base_from_wrist[2, 3],
            "camera": self.calibration.base_from_camera(base_from_wrist)[2, 3],
        }
        for part, z in heights.items():
            if not np.isfinite(z) or z < self.floor:
                raise ValueError(f"{part.capitalize()} would be below the configured table clearance")

    def plan(
        self, kind: str, target: np.ndarray, start: np.ndarray, hover_m: float,
        depth_m: float, yaw_deg: float, stop: Event,
    ) -> Plan:
        if kind not in ("hover", "grasp"):
            raise ValueError("Choose Hover or Grasp")
        if kind == "grasp" and not self.can_grasp:
            raise ValueError("Configure gripper open and close positions to grasp")
        if not 0.02 <= hover_m <= 0.30 or not 0 <= depth_m <= 0.10 or not -180 <= yaw_deg <= 180:
            raise ValueError("Use hover 20–300 mm, grasp depth 0–100 mm, and yaw −180–180 degrees")
        check_stop(stop)
        initial = self.tool.fk(start)
        self.check_clearance(start)
        if not np.all(np.isfinite(target)) or target[2] < self.floor:
            raise ValueError("Marked point is below the configured table clearance")
        hover_xyz = target + [0, 0, hover_m]
        grasp_xyz = target - [0, 0, depth_m] if kind == "grasp" else None
        if grasp_xyz is not None and grasp_xyz[2] < self.floor:
            raise ValueError("Grasp is below the configured table clearance. Reduce grasp depth.")
        rotation = rpy_to_matrix(math.pi, 0, math.radians(yaw_deg))
        travel_height = max(initial[2, 3], hover_xyz[2])
        above_start = np.r_[initial[:2, 3], travel_height]
        above_target = np.r_[hover_xyz[:2], travel_height]
        steps: list[Step] = []
        q = start.copy()

        def move(name: str, xyz: np.ndarray, orientation: np.ndarray, speed: float) -> None:
            nonlocal q
            check_stop(stop)
            current = self.tool.fk(q)
            distance = np.linalg.norm(xyz - current[:3, 3])
            angle = np.linalg.norm(matrix_to_rotvec(current[:3, :3].T @ orientation))
            if np.linalg.norm(xyz - initial[:3, 3]) > self.max_travel:
                raise ValueError(f"{name} exceeds the configured travel limit")
            if distance < 0.001 and angle < math.radians(0.5):
                return
            destination = np.eye(4)
            destination[:3, :3], destination[:3, 3] = orientation, xyz
            duration = max(1.5, 1.9 * distance / speed, 1.9 * angle / 0.6)
            try:
                path, dt = plan_cartesian(self.tool, q, destination, duration_s=duration, fps=self.fps)
                for waypoint in path:
                    check_stop(stop)
                    self.check_clearance(waypoint)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
            steps.append(Step(name, path, dt))
            q = path[-1]

        move("Raise to travel height", above_start, initial[:3, :3], 0.04)
        move("Move above target", above_target, rotation, 0.04)
        move("Move to hover", hover_xyz, rotation, 0.04)
        if grasp_xyz is not None:
            steps.append(Step("Open gripper", gripper=self.gripper_open))
            move("Descend", grasp_xyz, rotation, 0.02)
            steps.append(Step("Close gripper", gripper=self.gripper_close))
            move("Lift", hover_xyz, rotation, 0.03)
        return Plan(kind, start.copy(), hover_xyz, grasp_xyz, tuple(steps))

    def execute(self, plan: Plan, stop: Event, progress: Callable[[str], None]) -> None:
        check_stop(stop)
        q = self.joints()
        start_displacement = np.linalg.norm(self.tool.fk(q)[:3, 3] - self.tool.fk(plan.start)[:3, 3])
        if np.max(np.abs(q - plan.start)) > POSE_TOL_RAD or start_displacement > POSE_TOL_M:
            raise ValueError("Arm moved after planning. Preview the path again.")
        self.check_clearance(q)
        enabled = False
        try:
            if plan.steps:
                check_stop(stop)
                self.robot.enable()
                enabled = True
            for step in plan.steps:
                check_stop(stop)
                progress(step.name)
                if step.path is None:
                    self._grip(step, q, stop)
                    continue
                for command in step.path:
                    check_stop(stop)
                    started = time.monotonic()
                    self.check_clearance(self.joints())
                    check_stop(stop)
                    self.robot.send_action(self.tool.q_to_motor(command))
                    if stop.wait(max(0, step.dt - (time.monotonic() - started))):
                        raise MotionStopped("Stopped")
                q = step.path[-1]
                self._wait_for_pose(q, stop)
        except Exception as exc:
            if enabled:
                try:
                    self.robot.send_action(self.tool.q_to_motor(self.joints()))
                except Exception as hold_error:
                    raise RuntimeError(f"{exc}. Could not command a hold: {hold_error}") from exc
            raise

    def _wait_for_pose(self, q: np.ndarray, stop: Event) -> None:
        deadline = time.monotonic() + 2
        target = self.tool.fk(q)[:3, 3]
        while time.monotonic() < deadline:
            check_stop(stop)
            actual = self.joints()
            self.check_clearance(actual)
            if (np.max(np.abs(actual - q)) <= POSE_TOL_RAD
                    and np.linalg.norm(self.tool.fk(actual)[:3, 3] - target) <= POSE_TOL_M):
                return
            stop.wait(0.03)
        raise TimeoutError("Arm feedback did not reach the planned pose")

    def _grip(self, step: Step, q: np.ndarray, stop: Event) -> None:
        self._wait_for_pose(q, stop)
        check_stop(stop)
        self.robot.send_action({**self.tool.q_to_motor(q), "gripper.pos": step.gripper})
        if step.name == "Close gripper":
            if stop.wait(0.7):
                raise MotionStopped("Stopped")
            return
        tolerance = 0.004 if self.robot.config.gripper_mode == "width" else 4.0
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            check_stop(stop)
            if abs(self.robot.get_observation()["gripper.pos"] - step.gripper) <= tolerance:
                return
            stop.wait(0.03)
        raise TimeoutError("Gripper feedback did not reach the open position")
