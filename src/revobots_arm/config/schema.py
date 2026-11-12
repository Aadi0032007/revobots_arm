"""All settings for the Revobots arm, as plain Python. No JSON.

Same idea as the AGV repo's ``LAB/config.py``: dataclasses with sensible defaults,
edited in place. Presets are defined at the bottom; pick one with
``--preset nero|koch|so101|so101_remote|...`` on any script, or ``REVOBOTS_ARM_PRESET``.
The default is ``DEFAULT_PRESET``.

``robot`` and ``teleop`` are dicts because they map 1:1 onto LeRobot's own config
dataclasses (``KochFollowerConfig``, ``AgxNeroFollowerConfig``, ``KochLeaderRemoteConfig``
...): ``type`` is the registered name, every other key is a field of that dataclass.
Cameras are NOT part of the robot dict: they are owned by ``cameras.py`` (frame bus,
hardware decode, RealSense depth) and merged into the observation by ``RobotWithCameras``.
"""

from __future__ import annotations

import ipaddress
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IS_LINUX = platform.system() == "Linux"
IS_JETSON = IS_LINUX and Path("/etc/nv_tegra_release").exists()


def video(n: int) -> int | str:
    """Camera source for index ``n``: ``/dev/videoN`` on Linux, the bare index elsewhere."""
    return f"/dev/video{n}" if IS_LINUX else n


def serial_port(windows: str, linux: str) -> str:
    return linux if IS_LINUX else windows


# ── cameras ──────────────────────────────────────────────────────────────────


CAMERA_TYPES = ("opencv", "intelrealsense")  # same names as LeRobot's camera registry


def depth_key(camera_name: str) -> str:
    """Observation key / frame-bus name of a camera's depth stream (LeRobot convention)."""
    return f"{camera_name}_depth"


@dataclass
class CameraConfig:
    """One camera. ``name`` becomes ``observation.images.<name>`` and the frame bus
    region ``lab_<name>``; with ``use_depth=True`` there is also
    ``observation.images.<name>_depth`` (uint16 mm, (H, W, 1)) and ``lab_<name>_depth``.

    ``type="opencv"``          ``source`` = index (Windows) or ``/dev/videoN`` (Linux).
                               ``hw_decode`` / ``pixel_format`` apply.
    ``type="intelrealsense"``  ``source`` = serial number (or unique device name) as printed by
                               ``python -m revobots_arm.cameras.diagnostics find``. Captured with pyrealsense2 directly
                               (AGV LAB style), arrays identical to LeRobot's ``RealSenseCamera``;
                               RGB is always on. ``use_depth`` adds the depth stream.
    """

    name: str
    source: int | str
    type: str = "opencv"  # opencv | intelrealsense
    width: int = 640
    height: int = 480
    fps: int = 30
    # opencv on Jetson: v4l2src → nvv4l2decoder/nvvidconv GStreamer pipeline through OpenCV.
    # If that pipeline fails to open, capture falls back to plain OpenCV V4L2.
    hw_decode: bool = IS_JETSON
    pixel_format: str = "MJPG"  # MJPG | YUYV (opencv only)
    use_rgb: bool = True  # legacy setting, ignored by RealSense
    use_depth: bool = False
    warmup_s: int = (
        1  # intelrealsense: warm-up before the first frame (>= 1 s, as LeRobot)
    )
    # Publish every frame to shared memory (/dev/shm/lab_<name>[, lab_<name>_depth]) so
    # inference and any other process can read frames while teleop owns the device.
    publish_frames: bool = True
    rotation: int = 0  # 0 | 90 | 180 | 270, applied after capture
    enabled: bool = True
    open_timeout_sec: float = 12.0

    @property
    def is_realsense(self) -> bool:
        return self.type == "intelrealsense"

    @property
    def rgb_enabled(self) -> bool:
        return self.is_realsense or self.use_rgb

    def validate(self) -> None:
        if self.type not in CAMERA_TYPES:
            raise ValueError(
                f"camera {self.name!r}: type must be one of {CAMERA_TYPES}, got {self.type!r}"
            )
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError(
                f"camera {self.name!r}: rotation must be 0, 90, 180 or 270"
            )
        if not self.is_realsense and (self.use_depth or not self.use_rgb):
            raise ValueError(f"camera {self.name!r}: depth needs type=intelrealsense")

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """(height, width, 3) of the colour frames this camera delivers, after rotation."""
        if self.rotation in (90, 270):
            return (self.width, self.height, 3)
        return (self.height, self.width, 3)

    @property
    def depth_shape(self) -> tuple[int, int, int]:
        """(height, width, 1) of the depth maps (same resolution as colour)."""
        h, w, _ = self.frame_shape
        return (h, w, 1)

    def observation_shapes(self) -> dict[str, tuple[int, int, int]]:
        """What this camera adds to ``observation_features``: ``{name: (H,W,3)}`` and,
        with depth, ``{name_depth: (H,W,1)}`` (LeRobot's ``<cam>_depth`` convention)."""
        out: dict[str, tuple[int, int, int]] = {}
        if self.rgb_enabled:
            out[self.name] = self.frame_shape
        if self.use_depth:
            out[depth_key(self.name)] = self.depth_shape
        return out


def realsense(
    name: str, serial_number_or_name: str, use_depth: bool = False, **fields: Any
) -> CameraConfig:
    """An Intel RealSense camera. ``serial_number_or_name`` as ``python -m revobots_arm.cameras.diagnostics find`` prints it.
    ``use_depth=True`` records ``observation.images.<name>_depth`` too."""
    return CameraConfig(
        name=name,
        source=serial_number_or_name,
        type="intelrealsense",
        use_depth=use_depth,
        **fields,
    )


def default_cameras() -> list[CameraConfig]:
    """The two-camera rig. Not used by any preset by default: cameras are OFF unless given
    on the command line (``--cameras.wrist.source=0 --cameras.eye.source=2``) or a preset
    sets ``cameras=default_cameras()``. A RealSense with depth looks like
    ``realsense("eye", "123456789012", use_depth=True)``."""
    return [
        CameraConfig(name="wrist", source=video(0)),
        CameraConfig(name="eye", source=video(2)),
    ]


# ── sub-configs ──────────────────────────────────────────────────────────────


@dataclass
class LeaderClientConfig:
    """Defaults for ``leader_client.py`` on the PC the leader arm is plugged into.

    Which client runs (Koch or SO-101) follows the preset's ``teleop["type"]``:
    ``koch_leader_remote`` → Koch client, ``so101_leader_remote`` → SO client.
    """

    robot_ip: str = "192.168.1.42"
    port: int = 5001  # must match teleop["port"] on the robot PC
    com: str = serial_port("COM3", "/dev/ttyACM1")
    id: str = "aadi"  # calibration file key of the leader arm
    fps: int = 60
    gripper_open_pos: float = 45.0  # Koch leader only: gripper spring-back position
    use_degrees: bool = True  # SO leader only: must match the follower's use_degrees


@dataclass
class RecordConfig:
    continuous: bool = (
        False  # teleop: start on readiness, stop on exit; mark episodes offline
    )
    # CLI: --display_data=true --display_mode=rerun --display_ip=... --display_port=... --play_sounds=false
    display_data: bool = False
    display_mode: str = "rerun"  # rerun | foxglove
    display_ip: str | None = None  # rerun: remote viewer ip; foxglove: bind interface
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = False  # spoken prompts off; --play_sounds=true to enable
    codec: str = "mp4v"  # OpenCV FOURCC for the raw session mp4s
    redo_starts_recording: bool = (
        True  # teleop: after Left/r start the next take at once
    )
    frame_stale_s: float = 0.5  # warn when a camera frame is older than this
    # record.py (timed, lerobot-record style). CLI: --dataset.num_episodes / episode_time_s / reset_time_s
    num_episodes: int = 50
    episode_time_s: float = 30.0
    reset_time_s: float = 10.0


@dataclass
class DatasetConfig:
    """Same field names as lerobot-record's --dataset.* where they overlap."""

    repo_id: str = "Aadi/revobots_arm_v1"
    root: str | None = None  # None → $HF_LEROBOT_HOME/<repo_id>
    push_to_hub: bool = False
    private: bool | None = None
    tags: list[str] = field(default_factory=lambda: ["revobots", "arm"])
    video: bool = True  # convert.py: encode videos (False → PNG images)
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    # convert.py: how depth maps are stored (LeRobot's 12-bit depth encoder, metres; LeRobot's defaults).
    # CLI: --dataset.depth_encoder.depth_max=1.0 --dataset.depth_encoder.use_log=false
    # D405 (7-50 cm): depth_max=1.0 with use_log=False keeps every millimetre exact.
    depth_min: float = 0.01
    depth_max: float = 10.0
    depth_shift: float = 3.5  # log mode only
    depth_use_log: bool = True
    episode: int = 0  # replay.py: --dataset.episode


@dataclass
class ReplayConfig:
    ramp_s: float = 1.5  # glide from the current pose to the first action


@dataclass
class PolicyConfig:
    path: str = "outputs/train/act_arm/checkpoints/last/pretrained_model"
    dataset_repo_id: str | None = None  # None → dataset.repo_id
    dataset_root: str | None = None
    device: str = "cuda"
    duration_s: float = 60.0
    fps: int | None = None  # None → ArmConfig.fps


@dataclass
class AiConfig:
    """Teleop ↔ inference link. Teleop owns the arm and the cameras; inference is a
    separate process that reads frames from the frame bus and joint state from
    ``state_port``, and sends actions to ``action_port``. Teleop only applies them
    while AI control is on (``a`` key)."""

    # OFF by default: no UDP sockets are opened and the `a` key does nothing.
    # Turn on with --ai.enabled=true (required on the teleop side before running inference.py).
    enabled: bool = False
    host: str = "127.0.0.1"
    action_port: int = 5010  # inference → teleop   {"seq","t","action":{...}}
    state_port: int = 5011  # teleop → inference   {"seq","t","observation":{...},...}
    action_ttl_s: float = 0.5  # older AI actions are ignored (arm holds)
    bus_timeout_s: float = 5.0  # inference waits this long for the first frame/state
    # Frame bus gaps: inference reuses the last frame at most this many times in a row,
    # then stops with an error naming the camera and the frame bus. A fresh frame resets it.
    max_repeated_frames: int = 30
    # While AI drives, keep reading the leader so a human can take over by moving it.
    # With koch_leader_remote this blocks when the leader client is not streaming.
    read_leader_while_ai: bool = True
    # Leader movement (max joint delta from its pose when AI was enabled) that hands
    # control back to the human. Units follow the leader: Koch ±100 range, SO-101 degrees.
    human_takeover_threshold: float = 8.0


@dataclass
class StreamConfig:
    """RTSP stream for the dashboard: the AGV ``LAB/cameras.py`` GstRtspServer.

    The process that owns the cameras (teleop, record.py, ``cameras test|preview``) serves
    ``rtsp://<robot pc>:<port>/<mount>``; MediaMTX pulls it and republishes it to the
    dashboard, exactly as with the AGV's USB AI camera. ``camera`` (default: the first
    enabled camera) is served at ``/<mount>``, every other enabled colour camera at
    ``/<camera name>``. Frames are read from the frame bus ``lab_<name>`` (so the camera
    needs ``publish_frames=True``); an encoder only runs while a client pulls that mount.
    Needs PyGObject and gst-rtsp-server (Linux / Jetson); without them the server logs an
    error and the rest keeps running.
    """

    # OFF by default. CLI: --stream.enabled=true [--stream.camera=eye] [--stream.hw_encode=true]
    enabled: bool = False
    bind: str = "0.0.0.0"
    port: int = 8556  # same as the AGV; change it if both run on one PC
    mount: str = "ai"  # the path MediaMTX pulls, same as the AGV's USB AI camera
    camera: str | None = (
        None  # camera served at /<mount>; None = first enabled colour camera
    )
    hw_encode: bool = (
        False  # True = Jetson NVENC (nvv4l2h264enc); False = x264enc (AGV default)
    )
    bitrate_kbps: int = 1500
    # Frame bus gaps: the last frame is re-sent at most this many times in a row; then the
    # mount stops and logs which camera / frame bus stalled. A fresh frame resets the count.
    max_repeated_frames: int = 30
    # More CLOSE_WAIT sockets than this on the port (checked every interval) logs an error.
    # The AGV exits for a systemd restart there; the arm only logs, because exiting
    # teleop would skip robot.disconnect() and lose the session being recorded.
    close_wait_interval_s: float = 30.0
    close_wait_max: int = 50
    # With RTSP and RealSense depth enabled, send distance samples at 3 Hz.
    depth_robot_id: str | None = None  # fleet ID, defaults to the configured arm ID
    depth_host: str = "127.0.0.1"
    depth_port: int = 57110


# ── top level ────────────────────────────────────────────────────────────────


@dataclass
class CartesianIKConfig:
    enabled: bool = False
    leader_urdf: str | None = None
    leader_base_link: str | None = None
    leader_tip_link: str | None = None
    leader_motors: str | None = None  # comma-separated, in URDF chain order
    leader_invert_joints: str | None = None
    leader_joint_offsets_deg: str | None = None
    follower_urdf: str | None = None
    follower_base_link: str | None = None
    follower_tip_link: str | None = None
    follower_motors: str | None = None
    follower_invert_joints: str = ""
    follower_joint_offsets_deg: str = ""
    position_scale: float | None = None  # default 2.75 for Koch to Nero, otherwise 1
    frame_rpy_deg: str | None = None  # otherwise auto-align Koch to Nero, identity for other pairs
    orientation_weight: float = 1.0
    max_joint_step_deg: float = 1.0
    leader_calibration: str | None = None


@dataclass
class ArmConfig:
    name: str
    robot: dict[str, Any]
    teleop: dict[str, Any]
    # No cameras unless given: --cameras.wrist.source=0 --cameras.eye.source=2
    cameras: list[CameraConfig] = field(default_factory=list)
    fps: int = 30
    # Empty unless given: --dataset.single_task="Pick the cube"
    task: str = ""
    sessions_root: Path = Path("~/.cache/revobots/arm/sessions").expanduser()
    leader_client: LeaderClientConfig = field(default_factory=LeaderClientConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    # Path to <lerobot repo>/src when LeRobot is neither installed nor next to this folder.
    lerobot_src: str | None = None
    ik: CartesianIKConfig = field(default_factory=CartesianIKConfig)

    @property
    def robot_type(self) -> str:
        return str(self.robot["type"])

    @property
    def teleop_type(self) -> str:
        return str(self.teleop["type"])

    @property
    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self.enabled_cameras]

    @property
    def camera_observation_keys(self) -> list[str]:
        """Image keys the cameras put into the observation: ``<name>`` and ``<name>_depth``."""
        keys: list[str] = []
        for c in self.enabled_cameras:
            keys.extend(c.observation_shapes())
        return keys

    def camera(self, name: str) -> CameraConfig:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(
            f"no camera named {name!r}; have {[c.name for c in self.cameras]}"
        )

    def stream_mounts(self) -> dict[str, CameraConfig]:
        """RTSP mount → camera: ``stream.camera`` (default the first enabled colour camera) at
        ``/<stream.mount>``, every other enabled colour camera at ``/<name>``.
        Raises ``ValueError`` for a setup the stream cannot serve."""
        cams = [c for c in self.enabled_cameras if c.rgb_enabled]
        if not cams:
            raise ValueError(
                "the stream needs a colour camera: add --cameras.<name>.source=..., "
                "or drop --stream.enabled=true"
            )
        name = cams[0].name if self.stream.camera is None else str(self.stream.camera)
        main = next((c for c in cams if c.name == name), None)
        if main is None:
            raise ValueError(
                f"stream.camera={name!r} is not an enabled colour camera; have {[c.name for c in cams]}"
            )
        mounts = {str(self.stream.mount).strip("/"): main}
        for c in cams:
            if c is main:
                continue
            if c.name in mounts:
                raise ValueError(
                    f"camera {c.name!r} and stream.mount both want /{c.name}; change --stream.mount"
                )
            mounts[c.name] = c
        off_bus = [c.name for c in mounts.values() if not c.publish_frames]
        if off_bus:
            raise ValueError(
                f"the stream reads the frame bus, but {off_bus} has publish_frames=False"
            )
        if any(c.is_realsense and c.use_depth for c in mounts.values()):
            try:
                ipaddress.IPv4Address(str(self.stream.depth_host))
            except ValueError as exc:
                raise ValueError(
                    "stream.depth_host must be a numeric IPv4 address"
                ) from exc
            port = self.stream.depth_port
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("stream.depth_port must be an integer in 1..65535")
        return mounts
