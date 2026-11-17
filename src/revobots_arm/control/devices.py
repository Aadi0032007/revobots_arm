"""Device settings and input loops for the local arm console."""

from copy import deepcopy
from dataclasses import asdict, replace
import inspect
import math
import time

from revobots_arm.config import CameraConfig
from revobots_arm.config.presets import configure_ik_follower
from revobots_arm.config.schema import CartesianIKConfig
from revobots_arm.grasp.motion import check_stop
from revobots_arm.robots.nero.driver import NeroTrackingError
from revobots_arm.robots.registry import make_robot
from revobots_arm.teleoperators.registry import make_teleop
from revobots_arm.transport.ai_link import UdpJsonListener
from revobots_arm.utils.bootstrap import ensure_lerobot_importable


SERIAL = {"koch_follower", "so100_follower", "so101_follower"}
FOLLOWERS = SERIAL | {"agx_nero_native", "agx_nero_follower"}
LEADERS = {f"{family}_leader{suffix}" for family in ("koch", "so100", "so101") for suffix in ("", "_remote")}
INPUTS = LEADERS | {"manual", "gamepad_ee"}


def setup_values(cfg) -> dict:
    return {"preset": cfg.name, "robot": cfg.robot, "teleop": cfg.teleop, "ik": asdict(cfg.ik),
            "fps": cfg.fps, "cameras": [asdict(camera) for camera in cfg.cameras], "wrist": ""}


def make_config(baseline, setup: dict):
    cfg = deepcopy(baseline)
    cfg.robot = dict(setup["robot"])
    cfg.teleop = dict(setup["teleop"])
    cfg.ik = CartesianIKConfig(**setup["ik"])
    cfg.fps = int(setup["fps"])
    if cfg.robot_type not in FOLLOWERS or cfg.teleop_type not in INPUTS:
        raise ValueError("Select a supported follower and input")
    if not 10 <= cfg.fps <= 60:
        raise ValueError("Use an arm rate between 10 and 60 FPS")
    if cfg.ik.enabled:
        if cfg.teleop_type not in LEADERS:
            raise ValueError("Leader Cartesian IK requires a leader arm input")
        configure_ik_follower(cfg)
    if cfg.robot_type == "agx_nero_native":
        cfg.robot.update(enable_on_connect=False, rest_on_start=False, disable_on_disconnect=False)
    if len(setup["cameras"]) > 8:
        raise ValueError("Configure at most eight cameras")
    cfg.cameras = [CameraConfig(**spec) for spec in setup["cameras"]]
    names = set()
    for camera in cfg.cameras:
        camera.validate()
        if (not camera.name or not all(c.isalnum() or c in "_-" for c in camera.name)
                or camera.name in names or camera.name.endswith("_depth")):
            raise ValueError("Use unique camera names with letters, numbers, hyphens, or underscores")
        names.add(camera.name)
        if not str(camera.source).strip() or not (0 < camera.width <= 4096 and 0 < camera.height <= 4096 and 0 < camera.fps <= 120):
            raise ValueError(f"Enter a valid source, resolution, and FPS for {camera.name}")
    wrist = setup.get("wrist") or None
    if wrist:
        camera = cfg.camera(wrist)
        if not camera.enabled or not camera.is_realsense or not camera.use_depth:
            raise ValueError("Choose an enabled depth RealSense as the grasp camera")
    return cfg, wrist


class SerialPreview:
    """Read the calibrated motor positions without changing torque or configuration."""

    def __init__(self, robot):
        self.robot = robot

    def __getattr__(self, name):
        return getattr(self.robot, name)

    @property
    def is_connected(self):
        return self.robot.bus.is_connected

    def connect(self, calibrate=False):
        self.robot.bus.connect(handshake=False)

    def get_observation(self):
        return {f"{key}.pos": value for key, value in self.robot.bus.sync_read("Present_Position").items()}

    def disconnect(self):
        self.robot.bus.disconnect(disable_torque=False)


def open_robot(cfg, *, live: bool):
    if cfg.robot_type != "agx_nero_native":
        if cfg.robot_type == "agx_nero_follower" and not live:
            raise ValueError("Use native Nero for preview mode or start with --execute for the mapped driver")
        ensure_lerobot_importable(cfg.lerobot_src)
    robot = make_robot(cfg)
    return SerialPreview(robot) if not live and cfg.robot_type in SERIAL else robot


def checked_action(action: dict, keys, *, complete: bool = False) -> dict[str, float]:
    if action.keys() - keys or (complete and keys - action.keys()):
        raise ValueError("Input action keys do not match the connected follower")
    result = {key: float(value) for key, value in action.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Input returned a non-finite motor target")
    return result


class RemoteJointInput:
    """Consume the leader client's joint packets without blocking the control loop."""

    def __init__(self, spec):
        port, host = spec["port"], spec.get("host", "0.0.0.0")
        self.max_age = float(spec.get("max_action_age_s", 0.5))
        if type(port) is not int or not 1 <= port <= 65535 or not 0 < self.max_age < float("inf"):
            raise ValueError("Specify a valid UDP port and positive input timeout")
        self.listener = UdpJsonListener(host, port, "console-leader", session_packets=True)
        self.is_connected = False

    def connect(self):
        self.listener.start()
        self.is_connected = True

    def disconnect(self):
        self.listener.stop()
        self.is_connected = False

    def get_action(self):
        packet = self.listener.latest(self.max_age)
        if packet is None or packet.get("type") is not None or not isinstance(packet.get("action"), dict):
            return {}
        return packet["action"]


def run_teleop(cfg, robot, manual, stop, report, publish) -> None:
    """Read an existing leader or gamepad adapter without opening a second follower."""
    if cfg.teleop_type == "manual":
        raise ValueError("Choose a leader or gamepad in Setup")
    remote = cfg.teleop_type.endswith("_remote")
    if (cfg.teleop_type in LEADERS and not remote) or cfg.robot_type != "agx_nero_native":
        ensure_lerobot_importable(cfg.lerobot_src)
    if cfg.teleop_type in LEADERS and not cfg.ik.enabled:
        if manual.full_pose:
            raise ValueError("Select Cartesian IK for a leader and native Nero")
        if manual.mapped_nero and not remote:
            raise ValueError("Run the mapped Nero leader client over UDP, using 127.0.0.1 on the same machine")
        if not remote and bool(cfg.teleop.get("use_degrees")) != bool(cfg.robot.get("use_degrees")):
            raise ValueError("Use the same position units for the leader and follower")
    if cfg.teleop_type == "gamepad_ee":
        if not manual.full_pose and (cfg.robot_type != "so101_follower" or not manual.degrees):
            raise ValueError("Gamepad jog needs native Nero or SO-101 configured in degrees")
        options = dict(cfg.teleop)
        options.update(require_deadman=True, gripper_needs_deadman=True,
                       pick_on_x=False, place_on_y=False, look_on_b=False)
        if cfg.robot_type == "agx_nero_native" and "urdf_path" not in options:
            from revobots_arm.config.presets import nero_cartesian
            options = {**nero_cartesian().teleop, **options}
        if "gripper.pos" not in robot.action_features:
            options["gripper_key"] = None
        elif manual.full_pose and "gripper_key" not in cfg.teleop:
            low, high = manual.limits["gripper.pos"]
            options.update(gripper_key="gripper.pos", gripper_min=low, gripper_max=high,
                           gripper_speed=0.02 if manual.gripper_mode == "width" else 60)
        device = make_teleop(replace(cfg, teleop=options))
        retarget = None
    elif cfg.ik.enabled:
        from revobots_arm.teleoperators.leader.cartesian import CartesianRetargeter
        from revobots_arm.teleoperators.leader.joint_stream import make_ik_input

        device = make_ik_input(cfg)
        retarget = CartesianRetargeter(cfg)
        retarget.validate_devices(device, robot)
    elif remote:
        device, retarget = RemoteJointInput(cfg.teleop), None
    else:
        device, retarget = make_teleop(cfg), None
    keys = set(robot.action_features)
    sent = False
    active = False
    try:
        if "calibrate" in inspect.signature(device.connect).parameters:
            device.connect(calibrate=False)
        else:
            device.connect()
        report("Waiting for fresh teleop input")
        while not stop.is_set():
            tick = time.monotonic()
            observation = robot.get_observation()
            publish(observation)
            if getattr(device, "needs_seed", False):
                device.seed_from_observation(observation)
            if hasattr(device, "update_observation"):
                device.update_observation(observation)
            action = device.get_action()
            if retarget is not None:
                action = retarget.action(action or None, observation)
            if not action or not getattr(device, "command_ready", True):
                if active:
                    manual.hold()
                    report("Input paused. Holding position.")
                active = False
            else:
                action = checked_action(action, keys, complete=manual.mapped_nero)
                check_stop(stop)
                if not sent and manual.full_pose:
                    robot.enable()
                check_stop(stop)
                sent = True
                try:
                    robot.send_action(action)
                except NeroTrackingError as exc:
                    if retarget is not None:
                        retarget.reset()
                    elif hasattr(device, "on_target_rejected"):
                        device.on_target_rejected(exc.measured_action)
                    else:
                        raise
                    report("Input paused while the arm catches up")
                else:
                    if not active:
                        report("Teleop active")
                    active = True
            stop.wait(max(0.0, 1 / cfg.fps - (time.monotonic() - tick)))
    finally:
        try:
            if sent:
                manual.hold()
        finally:
            if getattr(device, "is_connected", False):
                device.disconnect()
