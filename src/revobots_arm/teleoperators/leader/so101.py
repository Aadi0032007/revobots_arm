"""SO101 leader integration for local serial and remote UDP operation."""

from revobots_arm.utils.lerobot import LeRobotAdapter

local = LeRobotAdapter(
    "teleop",
    "so101_leader",
    ("lerobot.teleoperators.so_leader", "lerobot.teleoperators.so101_leader"),
)
remote = LeRobotAdapter(
    "teleop",
    "so101_leader_remote",
    (
        "lerobot.teleoperators.so_leader_remote",
        "lerobot.teleoperators.so101_leader_remote",
    ),
)


def run_remote_client(**kwargs):
    """Run LeRobot's sender on the computer connected to the physical leader."""
    from revobots_arm.utils.lerobot import import_device_module

    return import_device_module(
        (
            "lerobot.teleoperators.so_leader_remote.so_leader_remote_client",
            "lerobot.teleoperators.so101_leader_remote.so101_leader_remote_client",
        )
    ).run_remote_client(**kwargs)
