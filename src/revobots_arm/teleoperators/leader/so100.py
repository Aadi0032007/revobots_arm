"""SO100 leader integration for local serial and remote UDP operation."""

from revobots_arm.utils.lerobot import LeRobotAdapter

local = LeRobotAdapter(
    "teleop",
    "so100_leader",
    ("lerobot.teleoperators.so_leader", "lerobot.teleoperators.so100_leader"),
)
remote = LeRobotAdapter(
    "teleop",
    "so100_leader_remote",
    (
        "lerobot.teleoperators.so_leader_remote",
        "lerobot.teleoperators.so100_leader_remote",
    ),
)
