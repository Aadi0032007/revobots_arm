"""Teleoperator factory: local adapters first, then LeRobot."""

from __future__ import annotations
from typing import Any
from revobots_arm.config.schema import ArmConfig
from revobots_arm.utils.registry import DeviceRegistry, instantiate
from revobots_arm.utils.lerobot import _choice_class, _normalise_common

teleoperators = DeviceRegistry("teleop")
teleoperators.register(
    "gamepad_ee",
    "revobots_arm.teleoperators.gamepad.controller:GamepadEETeleopConfig",
    "revobots_arm.teleoperators.gamepad.controller:GamepadEETeleop",
)

teleoperators.register_adapter(
    "koch_leader", "revobots_arm.teleoperators.leader.koch:local"
)

teleoperators.register_adapter(
    "koch_leader_remote", "revobots_arm.teleoperators.leader.koch:remote"
)

teleoperators.register_adapter(
    "so101_leader", "revobots_arm.teleoperators.leader.so101:local"
)

teleoperators.register_adapter(
    "so101_leader_remote", "revobots_arm.teleoperators.leader.so101:remote"
)

teleoperators.register_adapter(
    "so100_leader", "revobots_arm.teleoperators.leader.so100:local"
)

teleoperators.register_adapter(
    "so100_leader_remote", "revobots_arm.teleoperators.leader.so100:remote"
)


def build_teleop_config(cfg: ArmConfig) -> Any:
    if cfg.teleop_type in teleoperators:
        return teleoperators.config(cfg.teleop_type, cfg.teleop)
    from lerobot.teleoperators import TeleoperatorConfig

    spec = dict(cfg.teleop)
    cls = _choice_class(TeleoperatorConfig, str(spec.pop("type")), "teleoperator")
    _normalise_common(spec)
    return instantiate(cls, spec, "teleop")


def make_teleop(cfg: ArmConfig):
    config = build_teleop_config(cfg)
    if cfg.teleop_type in teleoperators:
        return teleoperators.create(cfg.teleop_type, config)
    from lerobot.teleoperators import make_teleoperator_from_config

    return make_teleoperator_from_config(config)
