"""Read leader joints locally or send calibrated URDF joint degrees over UDP."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import replace
from pathlib import Path

from revobots_arm.config import ArmConfig
from revobots_arm.config.schema import CartesianIKConfig
from revobots_arm.teleoperators.leader.koch import (
    KOCH_REVERSED,
    KOCH_URDF,
    RAHUL_OFFSETS_DEG,
    KochDegreeLeader,
)
from revobots_arm.teleoperators.registry import make_teleop
from revobots_arm.transport.ai_link import UdpJsonListener, UdpJsonSender
from revobots_arm.utils.parsing import parse_csv, parse_offsets


PACKET_TYPE = "leader_urdf_degrees"


def leader_joint_calibration(
    cfg: CartesianIKConfig, *, bundled_koch: bool
) -> tuple[dict[str, float], dict[str, float]]:
    inverted = (
        KOCH_REVERSED
        if bundled_koch and cfg.leader_invert_joints is None
        else parse_csv(cfg.leader_invert_joints)
    )
    offsets = parse_offsets(cfg.leader_joint_offsets_deg)
    if cfg.leader_calibration is not None:
        if not bundled_koch or cfg.leader_calibration != "rahul":
            raise ValueError("--ik.leader_calibration=rahul requires the bundled Koch leader model")
        if cfg.leader_joint_offsets_deg is None:
            offsets = RAHUL_OFFSETS_DEG
    if not all(math.isfinite(offset) for offset in offsets.values()):
        raise ValueError("Leader IK leader joint offsets must be finite")
    return dict.fromkeys(inverted, -1.0), offsets


def _degree_leader(cfg: ArmConfig, spec: dict):
    spec = dict(spec)
    kind = spec["type"]
    if kind in ("so101_leader", "so100_leader"):
        spec["use_degrees"] = True
    leader = make_teleop(replace(cfg, teleop=spec))
    if kind == "koch_leader":
        return KochDegreeLeader(leader)
    return leader


class RemoteDegreeLeader:
    def __init__(self, spec: dict) -> None:
        self.kind = spec["type"].removesuffix("_remote")
        self.host = spec.get("host", "0.0.0.0")
        self.port = spec.get("port", 5002 if spec["type"].startswith("so") else 5001)
        self.max_age = float(spec.get("max_action_age_s", 0.5))
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("Remote leader port must be an integer in 1..65535")
        if not math.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError("Remote leader max_action_age_s must be finite and positive")
        self.listener: UdpJsonListener | None = None
        self.session: str | None = None

    @property
    def is_connected(self) -> bool:
        return self.listener is not None

    def connect(self) -> None:
        if self.listener is None:
            listener = UdpJsonListener(
                self.host, self.port, "ik-leader", packet_type=PACKET_TYPE
            )
            listener.start()
            self.listener = listener

    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self.session = None

    def get_action(self) -> dict:
        packet = self.listener.latest(self.max_age) if self.listener else None
        if packet is None or packet.get("leader_type") != self.kind:
            return {}
        if packet["session"] != self.session:
            self.session = packet["session"]
            return {}  # force the follower to re-anchor after a sender restart
        action = packet.get("action")
        return action if isinstance(action, dict) else {}


def make_ik_input(cfg: ArmConfig):
    if cfg.teleop_type.endswith("_remote"):
        return RemoteDegreeLeader(cfg.teleop)
    return _degree_leader(cfg, cfg.teleop)


def stream_joint_degrees(
    cfg: ArmConfig,
    *,
    ip: str,
    port: int,
    com: str,
    arm_id: str,
    fps: int,
    gripper_open_pos: float,
) -> None:
    if fps <= 0 or not 1 <= port <= 65535:
        raise ValueError("Leader stream needs positive fps and a UDP port in 1..65535")
    kind = cfg.teleop_type.removesuffix("_remote")
    spec = {
        key: value for key, value in cfg.teleop.items()
        if key not in ("host", "port", "max_action_age_s", "waiting_log_period_s")
    }
    spec.update(type=kind, port=com, id=arm_id)
    if kind == "koch_leader":
        spec["gripper_open_pos"] = gripper_open_pos
    bundled_koch = kind == "koch_leader" and (
        cfg.ik.leader_urdf is None
        or Path(cfg.ik.leader_urdf).expanduser().resolve() == KOCH_URDF.resolve()
    )
    signs, offsets = leader_joint_calibration(cfg.ik, bundled_koch=bundled_koch)
    leader = _degree_leader(cfg, spec)
    corrected_keys = {f"{motor}.pos": motor for motor in signs.keys() | offsets.keys()}
    features = getattr(leader, "action_features", None)
    missing = corrected_keys.keys() - features.keys() if features is not None else set()
    if missing:
        raise ValueError(f"Leader calibration refers to unknown joint actions: {sorted(missing)}")
    sender = UdpJsonSender(ip, port)
    session = uuid.uuid4().hex
    header = {"type": PACKET_TYPE, "session": session, "leader_type": kind}
    try:
        leader.connect()
        while True:
            tick = time.monotonic()
            angles = {key: float(value) for key, value in leader.get_action().items()}
            for key, motor in corrected_keys.items():
                if key in angles:
                    angles[key] = signs.get(motor, 1.0) * angles[key] + offsets.get(motor, 0.0)
            sender.send({**header, "action": angles})
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        if leader.is_connected:
            leader.disconnect()
