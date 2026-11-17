"""Manual joint and Cartesian trajectories based on measured feedback."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
from threading import Event
import time

import numpy as np

from revobots_arm.grasp.motion import MotionStopped, check_stop
from revobots_arm.inverse_kinematics.core import matrix_to_pose, pose_to_matrix
from revobots_arm.inverse_kinematics.nero import nero_kinematics, plan_cartesian
from revobots_arm.inverse_kinematics.so101 import so101_kinematics


@dataclass(frozen=True)
class ManualPlan:
    start: dict[str, float]
    target: dict[str, float]
    samples: list[dict[str, float]]
    dt: float


class ManualControl:
    def __init__(self, cfg, robot):
        self.robot = robot
        self.fps = cfg.fps
        self.mapped_nero = cfg.robot_type == "agx_nero_follower"
        self.kin = None
        self.full_pose = cfg.robot_type == "agx_nero_native"
        self.degrees = self.full_pose or self.mapped_nero or cfg.robot.get("use_degrees", False)
        self.gripper_mode = cfg.robot.get("gripper_mode", "width") if self.full_pose else None
        self.limits = {}
        if self.full_pose:
            self.kin = nero_kinematics()
        elif cfg.robot_type == "so101_follower" and self.degrees:
            self.kin = so101_kinematics()
        if self.kin:
            self.limits.update(zip(self.kin.keys, zip(
                np.degrees(self.kin.chain.lower + self.kin.limit_margin),
                np.degrees(self.kin.chain.upper - self.kin.limit_margin),
            )))
        self.limits["gripper.pos"] = (0, 0.1 if self.gripper_mode == "width" else 100)

    def position_fields(self, observation: dict) -> list[dict]:
        fields = []
        for name in self.robot.action_features:
            if not name.endswith(".pos") or name not in observation or not math.isfinite(float(observation[name])):
                continue
            unit = "°" if self.degrees else "normalized"
            if name == "gripper.pos":
                unit = {"width": "m", "angle": "°", "auto": "%"}.get(self.gripper_mode, "driver units")
            low, high = self.limits.get(name, (-180, 180) if self.degrees else (-100, 100))
            fields.append({"key": name, "value": float(observation[name]),
                           "min": float(low), "max": float(high), "unit": unit})
        return fields

    def pose(self, observation: dict) -> dict | None:
        if self.kin is None:
            return None
        xyz, rpy = matrix_to_pose(self.kin.fk_motors(observation))
        if not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError("Tool position feedback is not finite")
        return {"xyz": (1000 * xyz).tolist(), "rpy": np.degrees(rpy).tolist(),
                "base": self.kin.chain.base_link, "tip": self.kin.chain.tip_link,
                "orientation": self.full_pose}

    def joints_plan(self, requested: dict, observation: dict) -> ManualPlan:
        if self.mapped_nero:
            raise ValueError("Select a native Nero preset to command individual joints")
        fields = {field["key"]: field for field in self.position_fields(observation)}
        if not requested or requested.keys() - fields.keys():
            raise ValueError("Choose one or more available joint targets")
        target, start = {}, {}
        for key, raw in requested.items():
            field = fields[key]
            value = float(raw)
            if not math.isfinite(value) or not field["min"] <= value <= field["max"]:
                raise ValueError(f"{key} must be between {field['min']:.3f} and {field['max']:.3f}")
            target[key], start[key] = value, field["value"]
        duration = max(1.0, 1.9 * max(
            abs(target[key] - start[key]) / (0.02 if fields[key]["unit"] == "m" else 12)
            for key in target
        ))
        count = math.ceil(duration * self.fps)
        samples = []
        for i in range(1, count + 1):
            t = i / count
            fraction = t**3 * (10 - 15 * t + 6 * t**2)
            samples.append({key: start[key] + fraction * (value - start[key]) for key, value in target.items()})
        # All joints are part of the starting pose even if only one is being edited.
        return ManualPlan({key: field["value"] for key, field in fields.items()}, target, samples, duration / count)

    def pose_plan(self, xyz: list, rpy: list, observation: dict) -> ManualPlan:
        if self.kin is None:
            raise ValueError("Tool pose control needs native Nero or SO-101 configured in degrees")
        xyz = np.asarray(xyz, dtype=float) / 1000
        rpy = np.radians(np.asarray(rpy, dtype=float))
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError("Enter three finite position and orientation values")
        start = self.kin.motor_to_q(observation)
        current_xyz, current_rpy = matrix_to_pose(self.kin.fk(start))
        distance = float(np.linalg.norm(xyz - current_xyz))
        if distance > 0.5:
            raise ValueError("Move the tool by at most 500 mm per command")
        target = pose_to_matrix(xyz, rpy if self.full_pose else current_rpy)
        duration = max(3.0, distance / 0.02)
        path, dt = plan_cartesian(
            self.kin, start, target, duration_s=duration, fps=self.fps,
            orientation_weight=1.0 if self.full_pose else 0.0,
        )
        samples = [self.kin.q_to_motor(q) for q in path]
        return ManualPlan(dict(self.kin.q_to_motor(start)), samples[-1], samples, dt)

    def hold(self) -> None:
        if self.mapped_nero:
            # The mapped command API can transform or zero wrist joints.
            # Hold the seven measured hardware angles directly through the connected SDK.
            streamer = getattr(self.robot, "_streamer", None)
            if streamer is not None:
                streamer.stop()
                self.robot._streamer = None
            with getattr(self.robot, "_sdk_lock", nullcontext()):
                arm = self.robot.arm
                feedback = arm.get_joint_angles()
                if feedback is None or len(feedback.msg) != 7 or not np.isfinite(feedback.msg).all():
                    raise RuntimeError("No Nero joint feedback for position hold")
                arm.move_js(list(feedback.msg))
            return
        observation = self.robot.get_observation()
        self.robot.send_action({key: observation[key] for key in self.robot.action_features if key in observation})

    def execute(self, plan: ManualPlan, stop: Event) -> None:
        check_stop(stop)
        feedback = self.robot.get_observation()
        for key, start in plan.start.items():
            tolerance = 0.003 if self.limits.get(key, (None, None))[1] == 0.1 else 1.0
            measured = float(feedback[key])
            if not math.isfinite(measured) or abs(measured - start) > tolerance:
                raise ValueError("Arm moved after preview. Preview a new path.")
        enabled = False
        try:
            if self.full_pose:
                self.robot.enable()
            enabled = True
            for action in plan.samples:
                check_stop(stop)
                tick = time.monotonic()
                self.robot.send_action(action)
                if stop.wait(max(0.0, plan.dt - (time.monotonic() - tick))):
                    raise MotionStopped("Stopped")
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                check_stop(stop)
                feedback = self.robot.get_observation()
                if all(abs(float(feedback[key]) - value) <= (0.004 if self.limits.get(key, (None, None))[1] == 0.1 else 1.0)
                       for key, value in plan.target.items()):
                    return
                stop.wait(0.03)
            raise TimeoutError("Joint feedback did not reach the previewed target")
        except Exception:
            if enabled:
                self.hold()
            raise
