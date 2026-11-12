"""KOCH leader integration for local serial and remote UDP operation."""

from revobots_arm.utils.bootstrap import ASSET_ROOT
from revobots_arm.utils.lerobot import LeRobotAdapter

KOCH_URDF = ASSET_ROOT / "koch" / "koch_v1_1.urdf"
KOCH_ARM_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
KOCH_REVERSED = ("shoulder_pan", "shoulder_lift", "wrist_flex", "wrist_roll")
# Converted from sign * (q_encoder - zero) to sign * q_encoder + offset.
RAHUL_OFFSETS_DEG = dict(zip(KOCH_ARM_MOTORS, (1.0, 0.9, 4.4, -5.6, -7.9)))

local = LeRobotAdapter("teleop", "koch_leader", ("lerobot.teleoperators.koch_leader",))
remote = LeRobotAdapter(
    "teleop", "koch_leader_remote", ("lerobot.teleoperators.koch_leader_remote",)
)


def run_remote_client(**kwargs):
    """Run LeRobot's sender on the computer connected to the physical leader."""
    from revobots_arm.utils.lerobot import import_device_module

    return import_device_module(
        ("lerobot.teleoperators.koch_leader_remote.koch_leader_remote_client",)
    ).run_remote_client(**kwargs)


class KochDegreeLeader:
    """Read arm degrees around the encoder midpoint used by the Koch URDF calibration."""

    def __init__(self, leader) -> None:
        self.leader = leader

    def __getattr__(self, name: str):
        return getattr(self.leader, name)

    def get_action(self) -> dict[str, float]:
        action = self.leader.get_action()
        bus = self.leader.bus
        names = {
            bus.motors[name].id: name
            for name in KOCH_ARM_MOTORS
            if f"{name}.pos" in action
        }
        # Match rahul_revobots_ai: undo LeRobot's range normalization and drive mode,
        # then measure from encoder count 2048. DEGREES mode centers at the calibrated range instead.
        positions = bus._unnormalize(
            {motor: action[f"{name}.pos"] for motor, name in names.items()}
        )
        degrees = {
            f"{names[motor]}.pos": (steps - 2048) * 360.0 / 4096
            for motor, steps in positions.items()
        }
        return {**action, **degrees}
