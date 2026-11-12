"""Robot factory: local adapters first, then the existing LeRobot registry."""

from __future__ import annotations
import dataclasses
from typing import Any
from revobots_arm.config.schema import ArmConfig
from revobots_arm.utils.registry import DeviceRegistry, instantiate
from revobots_arm.utils.lerobot import _choice_class, _normalise_common

robots = DeviceRegistry("robot")
robots.register(
    "agx_nero_native",
    "revobots_arm.robots.nero.driver:NeroRobotConfig",
    "revobots_arm.robots.nero.driver:NeroJointRobot",
)

robots.register_adapter("koch_follower", "revobots_arm.robots.koch.follower:adapter")

robots.register_adapter("so101_follower", "revobots_arm.robots.so101.follower:adapter")

robots.register_adapter("so100_follower", "revobots_arm.robots.so100.follower:adapter")

robots.register_adapter(
    "agx_nero_follower", "revobots_arm.robots.nero.follower:adapter"
)


def build_robot_config(cfg: ArmConfig) -> Any:
    if cfg.robot_type in robots:
        return robots.config(cfg.robot_type, cfg.robot)
    from lerobot.robots import RobotConfig

    spec = dict(cfg.robot)
    cls = _choice_class(RobotConfig, str(spec.pop("type")), "robot")
    _normalise_common(spec)
    if "cameras" in {f.name for f in dataclasses.fields(cls)}:
        spec["cameras"] = {}
    return instantiate(cls, spec, "robot")


def make_robot(cfg: ArmConfig):
    config = build_robot_config(cfg)
    if cfg.robot_type in robots:
        return robots.create(cfg.robot_type, config)
    from lerobot.robots import make_robot_from_config

    return make_robot_from_config(config)
