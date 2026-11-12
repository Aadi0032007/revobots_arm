from __future__ import annotations
import os
from typing import Any
from revobots_arm.config.schema import (
    ArmConfig,
    DatasetConfig,
    PolicyConfig,
    ReplayConfig,
    LeaderClientConfig,
    serial_port,
)
# ── presets ──────────────────────────────────────────────────────────────────

KOCH_LEADER_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_roll",
    "wrist_flex",
    "gripper",
]


def koch_leader_remote(port: int = 5001) -> dict[str, Any]:
    return dict(
        type="koch_leader_remote",
        host="0.0.0.0",
        port=port,
        max_action_age_s=0.5,
        waiting_log_period_s=2.0,
    )


def so101_leader_remote(port: int = 5002) -> dict[str, Any]:
    return dict(
        type="so101_leader_remote",
        host="0.0.0.0",
        port=port,
        max_action_age_s=0.5,
        waiting_log_period_s=2.0,
    )


def so101_follower_robot() -> dict[str, Any]:
    return dict(
        type="so101_follower",
        port=serial_port("COM4", "/dev/ttyACM0"),
        id="aadi_so101_follower",  # calibration file key (lerobot-calibrate --robot.id)
        use_degrees=True,
        max_relative_target=None,
        disable_torque_on_disconnect=True,
    )


def gamepad_ee_teleop(**fields: Any) -> dict[str, Any]:
    """Gamepad Cartesian jog through URDF IK (see gamepad_teleop.py). Defaults = SO-101.
    Any field of GamepadEETeleopConfig can be passed, e.g. max_linear_speed=0.05, invert="pitch"."""
    return dict(type="gamepad_ee", **fields)


def nero() -> ArmConfig:
    """AgileX Nero follower over CAN, driven by a Koch leader on another PC (default)."""
    return ArmConfig(
        name="nero",
        robot=dict(
            type="agx_nero_follower",
            id="revobots_nero",
            interface="socketcan",
            channel="can-nero",  # USB-CAN (gs_usb); Jetson onboard MTTCAN is can0
            firmware_version="v121",  # this Nero reports 1.22; do not flash, profile only
            bitrate=1_000_000,
            speed_percent=20,
            max_relative_target=None,  # degrees per step, None = unlimited
            disable_on_disconnect=True,
            teleop_action_joints=list(KOCH_LEADER_JOINTS),
        ),
        teleop=koch_leader_remote(),
        dataset=DatasetConfig(
            repo_id="Aadi/revobots_nero_v1", tags=["revobots", "nero", "agilex", "arm"]
        ),
        policy=PolicyConfig(
            path="outputs/train/act_nero/checkpoints/last/pretrained_model"
        ),
        replay=ReplayConfig(ramp_s=2.0),
    )


def nero_cartesian() -> ArmConfig:
    """Native seven-joint interface for host IK; no Koch leader retargeting."""
    from revobots_arm.robots.nero.driver import NERO_JOINTS, NERO_URDF

    cfg = nero()
    cfg.name = "nero_cartesian"
    cfg.robot = dict(
        type="agx_nero_native",
        id="revobots_nero",
        interface="socketcan",
        channel="can-nero",
        firmware_version="v121",
        speed_percent=20,
        max_relative_target=10.0,
        disable_on_disconnect=False,
        with_gripper=False,
    )
    cfg.teleop = dict(
        type="gamepad_ee",
        urdf_path=str(NERO_URDF),
        tip_link="link7",
        base_link="base_link",
        motors=list(NERO_JOINTS),
        orientation_mode="frame",
        orientation_weight=1.0,
        max_pos_error_m=0.001,
        max_rot_error_deg=1.5,
        max_joint_step_deg=1.0,
        max_linear_speed=0.04,
        max_angular_speed=0.5,
        require_deadman=True,
        gripper_key=None,
    )
    # Native seven-joint datasets/policies have a different action schema.
    cfg.dataset.repo_id = "Aadi/revobots_nero_cartesian_v1"
    cfg.policy.path = ""
    return cfg


def nero_gamepad() -> ArmConfig:
    cfg = nero_cartesian()
    cfg.name = "nero_gamepad"
    # Nero controls: sticks are live, LB opens, RB closes; X/Y/B select poses.
    # Auto gripper mode exposes 0..100% in whichever SDK mode is reported.
    cfg.robot.update(
        with_gripper=True,
        gripper_mode="auto",
        rest_on_start=True,
        rest_deg=[0.742, -93.381, -0.228, 100.216, -2.091, 3.129, 78.982],
        rest_pose_file="/home/revolabs/revobots/porter/arm_rest_pose.json",
    )
    cfg.teleop.update(
        initial_speed_level=5,
        swap_stick_x=True,
        require_deadman=False,
        gripper_needs_deadman=False,
        gripper_key="gripper.pos",
        gripper_min=0.0,
        gripper_max=100.0,
        gripper_speed=60.0,
        max_tracking_error_deg=10.0,
        invert="x,y,roll",
        invert_gripper=True,
        pick_on_x=True,
        pick_pose_file="/home/revolabs/revobots/porter/arm_pick_pose.json",
        pick_deg=[0.163, -49.290, 2.444, 78.796, 0.509, 5.388, 56.427],
        pick_speed_deg_s=90.0,
        pick_min_time_s=2.0,
        place_on_y=True,
        place_pose_file="/home/revolabs/revobots/porter/arm_place_pose.json",
        place_deg=[-2.396, -90.607, 1.549, 112.206, -4.412, -3.649, 64.670],
        place_speed_deg_s=90.0,
        place_min_time_s=2.0,
        look_on_b=True,
        look_pose_file="/home/revolabs/revobots/porter/arm_look_pose.json",
        look_deg=[1.388, -48.965, 2.914, 78.624, 1.302, 2.826, -25.573],
        look_speed_deg_s=90.0,
        look_min_time_s=2.0,
    )
    return cfg


def configure_ik_follower(cfg: ArmConfig) -> None:
    """Use native joint targets whenever leader Cartesian retargeting is enabled."""
    if cfg.robot_type == "agx_nero_follower":
        native = nero_cartesian()
        legacy = nero()
        overrides = dict(cfg.robot)
        overrides.pop("teleop_action_joints", None)
        if overrides.get("max_relative_target") is None:
            overrides.pop("max_relative_target", None)
        cfg.robot = {**native.robot, **overrides, "type": "agx_nero_native"}
        if cfg.dataset.repo_id == legacy.dataset.repo_id:
            cfg.dataset.repo_id = native.dataset.repo_id
        if cfg.policy.path == legacy.policy.path:
            cfg.policy.path = native.policy.path
    if cfg.robot_type in ("koch_follower", "so100_follower", "so101_follower"):
        cfg.robot["use_degrees"] = True
    if cfg.robot_type == "agx_nero_native":
        cfg.robot["rest_on_start"] = False


def koch() -> ArmConfig:
    """Koch follower (Dynamixel, serial) driven by a Koch leader on another PC."""
    return ArmConfig(
        name="koch",
        robot=dict(
            type="koch_follower",
            port=serial_port("COM4", "/dev/ttyACM0"),
            id="revobots_koch_follower",
            use_degrees=False,  # False → body joints in -100..100, gripper 0..100
            max_relative_target=None,
            disable_torque_on_disconnect=True,
        ),
        teleop=koch_leader_remote(),
        dataset=DatasetConfig(
            repo_id="Aadi/revobots_koch_v1", tags=["revobots", "koch", "arm"]
        ),
        policy=PolicyConfig(
            path="outputs/train/act_koch/checkpoints/last/pretrained_model"
        ),
    )


def so101() -> ArmConfig:
    """SO-101 follower + SO-101 leader, both plugged into the robot PC."""
    return ArmConfig(
        name="so101",
        robot=so101_follower_robot(),
        teleop=dict(
            type="so101_leader",
            port=serial_port("COM5", "/dev/ttyACM1"),
            id="aadi_so101_leader",  # calibration file key (lerobot-calibrate --teleop.id)
            use_degrees=True,
        ),
        dataset=DatasetConfig(
            repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]
        ),
        policy=PolicyConfig(
            path="outputs/train/act_so101/checkpoints/last/pretrained_model"
        ),
    )


def so101_remote() -> ArmConfig:
    """SO-101 follower on the robot PC, SO-101 leader on another PC (so_leader_remote, UDP 5002)."""
    return ArmConfig(
        name="so101_remote",
        robot=so101_follower_robot(),
        teleop=so101_leader_remote(),
        leader_client=LeaderClientConfig(
            port=5002,
            com=serial_port("COM5", "/dev/ttyACM1"),
            id="aadi_so101_leader",
            use_degrees=True,
        ),
        dataset=DatasetConfig(
            repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]
        ),
        policy=PolicyConfig(
            path="outputs/train/act_so101/checkpoints/last/pretrained_model"
        ),
    )


def so101_gamepad() -> ArmConfig:
    """SO-101 follower driven in Cartesian space from a USB gamepad (URDF IK, no leader arm)."""
    return ArmConfig(
        name="so101_gamepad",
        robot=so101_follower_robot(),
        teleop=gamepad_ee_teleop(initial_speed_level=5),
        dataset=DatasetConfig(
            repo_id="Aadi/revobots_so101_v1", tags=["revobots", "so101", "arm"]
        ),
        policy=PolicyConfig(
            path="outputs/train/act_so101/checkpoints/last/pretrained_model"
        ),
    )


PRESETS = {
    "nero": nero,
    "nero_cartesian": nero_cartesian,
    "nero_gamepad": nero_gamepad,
    "koch": koch,
    "so101": so101,
    "so101_remote": so101_remote,
    "so101_gamepad": so101_gamepad,
}
DEFAULT_PRESET = "nero"
PRESET_ENV_VAR = "REVOBOTS_ARM_PRESET"


def load(preset: str | None = None) -> ArmConfig:
    name = preset or os.environ.get(PRESET_ENV_VAR) or DEFAULT_PRESET
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; available: {sorted(PRESETS)}"
        ) from None
