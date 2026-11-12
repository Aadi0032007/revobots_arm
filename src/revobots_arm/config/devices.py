"""Glue between ``config.py`` and LeRobot: build the follower and the leader objects.

The LeRobot robot is always built WITHOUT cameras. Cameras are owned by
``cameras.py`` (frame bus, hardware decode) and merged into the observation by
``cameras.RobotWithCameras``, so the dataset features end up identical to what
``lerobot-record`` would produce with the cameras inside the robot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from revobots_arm import config as _config
from revobots_arm.utils.bootstrap import ensure_lerobot_importable
from revobots_arm.config import ArmConfig
from revobots_arm.config.presets import configure_ik_follower

from revobots_arm.robots.registry import (
    build_robot_config as build_robot_config,
    make_robot as make_robot,
)
from revobots_arm.teleoperators.registry import (
    build_teleop_config as build_teleop_config,
    make_teleop as make_teleop,
)

logger = logging.getLogger(__name__)

# ── config loading ────────────────────────────────────────────────────────────


OVERRIDES_HELP = """\
Configuration arguments use the same names as the lerobot CLI (lerobot-teleoperate / -record / -replay):
  --robot.type=so101_follower --robot.port=COM4 --robot.id=aadi_so101_follower
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=aadi_so101_leader
  --teleop.type=gamepad_ee --port=5001   UDP gamepad on 0.0.0.0; omit --port for local USB
  --debug=true   Print received UDP gamepad packets and active mapped inputs (default: false)
  --fps=30 --display_data=true --display_mode=rerun --play_sounds=false
  --dataset.repo_id=Aadi/so101_v1 --dataset.single_task="Pick the cube" --dataset.root=...
  --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10
  --dataset.episode=0 --dataset.push_to_hub=true --dataset.video=false
  --dataset.depth_encoder.depth_max=1.0 --dataset.depth_encoder.use_log=false   (convert: depth storage)
  --policy.path=<ckpt> --policy.device=cuda --dataset_repo_id=... --single_task=... --control_time_s=60
Extra sections of this folder, same style:
  --cameras.wrist.source=0 --cameras.eye.enabled=false   or   --cameras=none
  --cameras.eye.type=intelrealsense --cameras.eye.source=<serial> --cameras.eye.use_depth=true
  --stream.enabled=true --stream.camera=eye --stream.port=8556 --stream.hw_encode=true   (RTSP for the dashboard)
  --ik=true   Retarget a robot leader's tool motion through follower IK
  --teleop.wrist_priority=true   Favor the Nero wrist during gamepad IK rotations
  --ik.wrist_priority=true       Favor the Nero wrist during leader IK rotations
  --sessions_root=<dir> --ai.action_port=5010 --record.codec=mp4v --replay.ramp_s=0
Anything not given comes from the preset (--preset, default 'nero'). Setting --robot.type or
--teleop.type discards the preset's other keys for that section. Values are parsed as
int/float/bool/none/JSON when they look like one, else kept as text."""

# Repo-style names → where they live in ArmConfig.
ALIASES = {
    "ik": "ik.enabled",
    "port": "teleop.udp_port",
    "debug": "teleop.debug",
    "dataset.single_task": "task",
    "single_task": "task",
    "dataset.fps": "fps",
    "dataset.num_episodes": "record.num_episodes",
    "dataset.episode_time_s": "record.episode_time_s",
    "dataset.reset_time_s": "record.reset_time_s",
    "dataset.depth_encoder.depth_min": "dataset.depth_min",
    "dataset.depth_encoder.depth_max": "dataset.depth_max",
    "dataset.depth_encoder.shift": "dataset.depth_shift",
    "dataset.depth_encoder.use_log": "dataset.depth_use_log",
    "display_data": "record.display_data",
    "display_mode": "record.display_mode",
    "display_ip": "record.display_ip",
    "display_port": "record.display_port",
    "display_compressed_images": "record.display_compressed_images",
    "play_sounds": "record.play_sounds",
    "dataset_repo_id": "policy.dataset_repo_id",
    "root": "policy.dataset_root",
    "control_time_s": "policy.duration_s",
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "on", ""):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def make_parser(description: str | None) -> argparse.ArgumentParser:
    """Parser with --preset, lerobot-style epilog, and no prefix matching (so unknown
    --robot.* style args are never mistaken for abbreviations of known flags)."""
    # Windows redirected terminals may use a legacy code page.
    encoding = sys.stdout.encoding or "utf-8"
    if description:
        description = description.encode(encoding, errors="backslashreplace").decode(
            encoding
        )
    parser = argparse.ArgumentParser(
        description=description,
        epilog=OVERRIDES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_preset_arg(parser)
    parser.add_argument(
        "--deployment",
        help="JSON file of dotted configuration overrides; or REVOBOTS_ARM_DEPLOYMENT",
    )
    return parser


def add_bool_flag(
    parser: argparse.ArgumentParser, name: str, help: str, default: bool = False
) -> None:
    """``--resume``, ``--resume=true`` and ``--resume false`` all work (lerobot style)."""
    parser.add_argument(
        name, nargs="?", const=True, default=default, type=str2bool, help=help
    )


def add_preset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        default=None,
        choices=sorted(_config.PRESETS),
        help=f"Config preset from src/config.py (default: ${_config.PRESET_ENV_VAR} or '{_config.DEFAULT_PRESET}')",
    )
    if not parser.epilog:
        parser.epilog = OVERRIDES_HELP
        parser.formatter_class = argparse.RawDescriptionHelpFormatter


def load_config(
    preset: str | None = None,
    overrides: list[str] | dict[str, str] | None = None,
    deployment: str | None = None,
) -> ArmConfig:
    """Pick a preset from ``config.py``, apply CLI overrides, make sure ``lerobot`` is importable.

    ``overrides`` is the unknown-args list from ``parser.parse_known_args()`` (or an
    already parsed ``{"robot.port": "COM4"}`` dict).
    """
    cfg = _config.load(preset)
    deployment = deployment or os.environ.get("REVOBOTS_ARM_DEPLOYMENT")
    if deployment:
        values = json.loads(Path(deployment).expanduser().read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(
                "Deployment configuration must be an object of dotted overrides"
            )
        apply_overrides(
            cfg,
            {
                k: json.dumps(v) if not isinstance(v, str) else v
                for k, v in values.items()
            },
        )
    if overrides:
        parsed = (
            parse_overrides(overrides)
            if isinstance(overrides, list)
            else dict(overrides)
        )
        apply_overrides(cfg, parsed)
    if cfg.fps <= 0:
        raise ValueError("fps must be positive")
    if cfg.ik.enabled:
        configure_ik_follower(cfg)
    ensure_lerobot_importable(cfg.lerobot_src)
    return cfg


# ── CLI overrides ─────────────────────────────────────────────────────────────

_SECTION_DATACLASSES = (
    "leader_client",
    "record",
    "dataset",
    "replay",
    "policy",
    "ai",
    "stream",
    "ik",
)
_SCALARS = ("fps", "task", "sessions_root", "lerobot_src")


def parse_overrides(tokens: list[str]) -> dict[str, str]:
    """``["--robot.port=COM4", "--fps", "30"]`` → ``{"robot.port": "COM4", "fps": "30"}``."""
    out: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise SystemExit(
                f"unexpected argument {tok!r}. Overrides look like --robot.port=COM4.\n\n{OVERRIDES_HELP}"
            )
        body = tok[2:]
        if "=" in body:
            key, val = body.split("=", 1)
            i += 1
        else:
            key = body
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1]
                i += 2
            else:
                val = "true"
                i += 1
        out[key.strip()] = val
    return out


def coerce(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def overridden(overrides: list[str] | dict[str, str] | None, *keys: str) -> bool:
    """Whether any of ``keys`` (after alias resolution) was given on the command line."""
    if not overrides:
        return False
    parsed = (
        parse_overrides(overrides) if isinstance(overrides, list) else dict(overrides)
    )
    given = {ALIASES.get(k, k) for k in parsed}
    return any(ALIASES.get(k, k) in given for k in keys)


def apply_overrides(cfg: ArmConfig, overrides: dict[str, str]) -> ArmConfig:
    # Type changes first: they reset their section, then the remaining keys fill it.
    ordered = sorted(
        overrides.items(),
        key=lambda kv: 0 if kv[0] in ("robot.type", "teleop.type") else 1,
    )
    for raw_key, raw in ordered:
        key = ALIASES.get(raw_key, raw_key)
        parts = key.split(".")
        head = parts[0]
        val = coerce(raw)
        if key in (
            "teleop.debug", "ik.enabled", "teleop.wrist_priority", "ik.wrist_priority"
        ):
            val = str2bool(raw)

        if head in ("robot", "teleop"):
            if len(parts) != 2:
                raise SystemExit(f"--{key}: expected --{head}.<field>=<value>")
            section = getattr(cfg, head)
            if parts[1] == "type":
                section.clear()
                section["type"] = str(val)
            else:
                section[parts[1]] = val

        elif head == "cameras":
            if len(parts) == 1:
                if (
                    val is None
                    or val is False
                    or str(val).lower() in ("none", "off", "")
                ):
                    cfg.cameras = []
                else:
                    raise SystemExit(
                        "use --cameras=none, or --cameras.<name>.<field>=<value>"
                    )
            elif len(parts) == 3:
                name, field_name = parts[1], parts[2]
                cam = next((c for c in cfg.cameras if c.name == name), None)
                if cam is None:
                    cam = _config.CameraConfig(name=name, source=0)
                    cfg.cameras.append(cam)
                if (
                    field_name == "serial_number_or_name"
                ):  # LeRobot's RealSense field name
                    cam.type = "intelrealsense"
                    field_name = "source"
                if field_name == "source" and cam.is_realsense:
                    val = raw.strip()  # serial numbers are digits: keep them as text
                if not hasattr(cam, field_name):
                    raise SystemExit(
                        f"--{key}: CameraConfig has no field '{field_name}'"
                    )
                setattr(cam, field_name, val)
            else:
                raise SystemExit(
                    f"--{key}: expected --cameras.<name>.<field>=<value> or --cameras=none"
                )

        elif head in _SCALARS and len(parts) == 1:
            if head == "sessions_root":
                val = Path(str(val)).expanduser()
            setattr(cfg, head, val)

        elif head in _SECTION_DATACLASSES and len(parts) == 2:
            sub = getattr(cfg, head)
            if not hasattr(sub, parts[1]):
                raise SystemExit(
                    f"--{key}: {type(sub).__name__} has no field '{parts[1]}'"
                )
            setattr(sub, parts[1], val)

        else:
            raise SystemExit(f"unknown override --{key}\n\n{OVERRIDES_HELP}")
    if "port" in overrides and cfg.teleop_type != "gamepad_ee":
        raise SystemExit("--port selects UDP input for --teleop.type=gamepad_ee only")
    for cam in cfg.cameras:
        if cam.is_realsense:
            cam.source = str(
                cam.source
            )  # --cameras.x.type given after --cameras.x.source
        try:
            cam.validate()
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if cfg.stream.enabled:
        try:
            cfg.stream_mounts()
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    return cfg


# ── config dict → LeRobot config dataclass ───────────────────────────────────


def _describe_camera(c: Any) -> str:
    extras = []
    if c.is_realsense:
        extras.append("realsense")
        if c.use_depth:
            extras.append("depth")
    elif c.hw_decode:
        extras.append("hw")
    if c.publish_frames:
        extras.append("bus")
    return f"{c.name}({c.width}x{c.height}@{c.fps}{''.join(', ' + e for e in extras)})"


def describe(cfg: ArmConfig) -> str:
    cams = ", ".join(_describe_camera(c) for c in cfg.enabled_cameras) or (
        "none  (add --cameras.wrist.source=0 --cameras.eye.source=2, or "
        "--cameras.eye.type=intelrealsense --cameras.eye.source=<serial> --cameras.eye.use_depth=true)"
    )
    ai = (
        f"on  action udp:{cfg.ai.action_port}  state udp:{cfg.ai.state_port}"
        if cfg.ai.enabled
        else "off (--ai.enabled=true to allow inference.py / the `a` key)"
    )
    return (
        f"preset={cfg.name}\n"
        f"  robot : {cfg.robot_type}  {_short(cfg.robot)}\n"
        f"  teleop: {cfg.teleop_type}  {_short(cfg.teleop)}\n"
        f"  cams  : {cams}\n"
        f"  stream: {_describe_stream(cfg)}\n"
        f"  fps   : {cfg.fps}\n"
        f"  ai    : {ai}\n"
        f"  task  : {cfg.task!r}"
        + ("  (--dataset.single_task=... to set)" if not cfg.task else "")
    )


def _describe_stream(cfg: ArmConfig) -> str:
    s = cfg.stream
    if not s.enabled:
        return f"off (--stream.enabled=true serves rtsp://<this pc>:{s.port}/{s.mount} for MediaMTX)"
    try:
        mounts = cfg.stream_mounts()
    except ValueError as exc:
        return f"error: {exc}"
    paths = "  ".join(f"/{m} <- {c.name}" for m, c in mounts.items())
    return f"rtsp {s.bind}:{s.port}  {paths}  ({'nvv4l2h264enc' if s.hw_encode else 'x264enc'} {s.bitrate_kbps} kbps)"


def _short(section: dict[str, Any]) -> str:
    return " ".join(
        f"{k}={v}" for k, v in section.items() if k != "type" and v is not None
    )
