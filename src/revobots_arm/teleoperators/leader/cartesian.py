"""Retarget a leader's tool motion from calibrated joints to native follower joints."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from revobots_arm.config.schema import ArmConfig, CartesianIKConfig
from revobots_arm.inverse_kinematics.core import (
    ArmKinematics,
    ArmKinematicsConfig,
    SerialChain,
    matrix_to_rotvec,
    rotvec_to_matrix,
    rpy_to_matrix,
)
from revobots_arm.inverse_kinematics.nero import NERO_TIP_LINK, NERO_URDF, NERO_WRIST_JOINTS
from revobots_arm.inverse_kinematics.so101 import SO101_TIP_LINK, SO101_URDF
from revobots_arm.teleoperators.leader.joint_stream import leader_joint_calibration
from revobots_arm.teleoperators.leader.koch import KOCH_ARM_MOTORS, KOCH_URDF
from revobots_arm.utils.parsing import parse_csv, parse_offsets


MODELS = {
    "koch_leader": (KOCH_URDF, "gripper_tcp"),
    "so101_leader": (SO101_URDF, SO101_TIP_LINK),
    "so101_follower": (SO101_URDF, SO101_TIP_LINK),
    "agx_nero_native": (NERO_URDF, NERO_TIP_LINK),
}
logger = logging.getLogger(__name__)


def _kinematics(cfg: CartesianIKConfig, role: str, device_type: str) -> ArmKinematics:
    remote_leader = role == "leader" and device_type.endswith("_remote")
    device_type = device_type.removesuffix("_remote")
    default_path, default_tip = MODELS.get(device_type, (None, None))
    path = getattr(cfg, f"{role}_urdf") or default_path
    tip = getattr(cfg, f"{role}_tip_link") or default_tip
    base = getattr(cfg, f"{role}_base_link")
    if not path or not tip:
        raise ValueError(
            f"Leader IK needs --ik.{role}_urdf and --ik.{role}_tip_link for {device_type}"
        )
    path = Path(path).expanduser()
    chain = SerialChain.from_urdf(path, tip, base)
    if not chain.active or any(joint.type == "prismatic" for joint in chain.active):
        raise ValueError(f"Leader IK requires a rotary-joint serial chain for the {role}")
    bundled_koch = device_type == "koch_leader" and path.resolve() == KOCH_URDF.resolve()
    motors = parse_csv(getattr(cfg, f"{role}_motors")) or (
        list(KOCH_ARM_MOTORS) if bundled_koch else chain.joint_names
    )
    if remote_leader:
        if any(value is not None for value in (
            cfg.leader_calibration, cfg.leader_invert_joints, cfg.leader_joint_offsets_deg
        )):
            raise ValueError("Set remote leader calibration on revobots-arm-leader-client")
        signs, offsets = {}, {}  # UDP angles already use the leader URDF's convention.
    elif role == "leader":
        signs, offsets = leader_joint_calibration(cfg, bundled_koch=bundled_koch)
    else:
        signs = dict.fromkeys(parse_csv(cfg.follower_invert_joints), -1.0)
        offsets = parse_offsets(cfg.follower_joint_offsets_deg)
    if len(set(motors)) != len(motors) or (signs.keys() | offsets.keys()) - set(motors):
        raise ValueError(f"Leader IK has duplicate or unknown {role} motor names")
    if not all(math.isfinite(offset) for offset in offsets.values()):
        raise ValueError(f"Leader IK {role} joint offsets must be finite")
    return ArmKinematics(
        ArmKinematicsConfig(
            urdf_path=str(path),
            tip_link=tip,
            base_link=base,
            motors=motors,
            urdf_joints=chain.joint_names,
            joint_signs=signs,
            joint_offsets_deg=offsets,
            preferred_orientation_motors=(
                NERO_WRIST_JOINTS if role == "follower" and cfg.wrist_priority else ()
            ),
        )
    )


def _limited(vector: np.ndarray, limit: float) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector * (limit / length) if length > limit else vector


class CartesianRetargeter:
    """Follow motion relative to the first paired observation, seeded from live feedback."""

    def __init__(self, cfg: ArmConfig) -> None:
        if cfg.teleop_type == "gamepad_ee":
            raise ValueError("--ik=true expects a robot leader. Gamepad presets already use IK.")
        self.leader = _kinematics(cfg.ik, "leader", cfg.teleop_type)
        koch_to_nero = (
            Path(self.leader.cfg.urdf_path).resolve() == KOCH_URDF.resolve()
            and cfg.robot_type == "agx_nero_native"
        )
        ik = cfg.ik
        if koch_to_nero and ik.follower_urdf is None and ik.follower_tip_link is None:
            ik = replace(ik, follower_tip_link="gripper_tcp")
        self.follower = _kinematics(ik, "follower", cfg.robot_type)
        self.config = ik
        self.position_scale = ik.position_scale
        if self.position_scale is None:
            self.position_scale = 2.75 if koch_to_nero else 1.0
        if not math.isfinite(cfg.fps) or cfg.fps <= 0:
            raise ValueError("fps must be finite and positive for leader IK")
        if not math.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("ik.position_scale must be finite and positive")
        if not math.isfinite(cfg.ik.orientation_weight) or not 0 <= cfg.ik.orientation_weight <= 1:
            raise ValueError("ik.orientation_weight must be in [0, 1]")
        if not math.isfinite(cfg.ik.max_joint_step_deg) or cfg.ik.max_joint_step_deg <= 0:
            raise ValueError("ik.max_joint_step_deg must be finite and positive")
        self.auto_alignment = koch_to_nero and ik.frame_rpy_deg is None
        rpy_spec = "0,0,0" if ik.frame_rpy_deg is None else ik.frame_rpy_deg
        rpy = [float(value) for value in parse_csv(rpy_spec)]
        if len(rpy) != 3 or not all(math.isfinite(value) for value in rpy):
            raise ValueError("ik.frame_rpy_deg requires three finite angles (roll,pitch,yaw)")
        self.alignment = rpy_to_matrix(*np.radians(rpy))
        # At low control rates, cap the interval so a slow tick cannot request a large move.
        period = min(1 / cfg.fps, 0.1)
        self.linear_step = 0.06 * period
        self.angular_step = math.radians(30) * period
        self.joint_step = math.radians(cfg.ik.max_joint_step_deg)
        self.hold_keys: tuple[str, ...] = ()
        self.reset()

    def validate_devices(self, leader, follower) -> None:
        leader_features = getattr(leader, "action_features", None)
        if leader_features is not None:
            missing = set(self.leader.keys) - leader_features.keys()
            if missing:
                raise ValueError(f"Leader IK leader driver lacks {sorted(missing)}")
        missing = set(self.follower.keys) - follower.action_features.keys()
        if missing:
            raise ValueError(f"Leader IK follower driver cannot command {sorted(missing)}")
        missing = follower.action_features.keys() - follower.observation_features.keys()
        if missing:
            raise ValueError(f"Leader IK follower driver cannot read back {sorted(missing)}")
        self.hold_keys = tuple(
            key for key in follower.action_features if key not in self.follower.keys
        )

    def reset(self) -> None:
        self.reference: tuple[np.ndarray, np.ndarray] | None = None

    def action(self, leader_action, observation) -> dict[str, float] | None:
        if leader_action is None:
            self.reset()
            return None
        try:
            leader_q = self.leader.motor_to_q(leader_action)
            follower_q = self.follower.motor_to_q(observation)
            hold = {key: float(observation[key]) for key in self.hold_keys}
        except (KeyError, TypeError, ValueError, OverflowError):
            self.reset()
            return None
        if not (
            np.all(np.isfinite(leader_q))
            and np.all(np.isfinite(follower_q))
            and all(math.isfinite(value) for value in hold.values())
        ):
            self.reset()
            return None

        leader_pose = self.leader.fk(leader_q)
        follower_pose = self.follower.fk(follower_q)
        if self.reference is None:
            self.reference = (leader_pose, follower_pose)
            if self.auto_alignment:
                # Match forward in the horizontal plane using both tools at the anchor.
                lx, ly = leader_pose[:2, 3]
                fx, fy = follower_pose[:2, 3]
                lead_heading = math.atan2(ly, lx) if math.hypot(lx, ly) > 1e-3 else math.pi
                follower_heading = math.atan2(fy, fx) if math.hypot(fx, fy) > 1e-3 else 0.0
                self.alignment = rpy_to_matrix(0.0, 0.0, follower_heading - lead_heading)
            logger.info("Leader IK anchored to current leader and follower poses")
            return None

        leader_ref, follower_ref = self.reference
        align = self.alignment
        target_xyz = follower_ref[:3, 3] + self.position_scale * (
            align @ (leader_pose[:3, 3] - leader_ref[:3, 3])
        )
        leader_rotation = leader_pose[:3, :3] @ leader_ref[:3, :3].T
        target_rotation = align @ leader_rotation @ align.T @ follower_ref[:3, :3]
        dp = _limited(target_xyz - follower_pose[:3, 3], self.linear_step)
        dr = _limited(
            matrix_to_rotvec(target_rotation @ follower_pose[:3, :3].T), self.angular_step
        )
        if np.linalg.norm(dp) < 1e-4 and (
            self.config.orientation_weight == 0 or np.linalg.norm(dr) < 1e-3
        ):
            return None

        target = follower_pose.copy()
        target[:3, 3] += dp
        target[:3, :3] = rotvec_to_matrix(dr) @ follower_pose[:3, :3]
        result = self.follower.solve(
            target, follower_q, orientation_weight=self.config.orientation_weight
        )
        if not np.all(np.isfinite(result.q)) or not result.pos_err <= 0.001:
            return None
        delta = result.q - follower_q
        largest = float(np.max(np.abs(delta)))
        if largest < 1e-6:
            return None
        q = follower_q + delta * min(1.0, self.joint_step / largest)
        return {**self.follower.q_to_motor(q), **hold}
