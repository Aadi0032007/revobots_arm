"""SO101 follower: LeRobot owns motor control, calibration and torque handling."""

from revobots_arm.utils.lerobot import LeRobotAdapter

adapter = LeRobotAdapter(
    "robot",
    "so101_follower",
    ("lerobot.robots.so_follower", "lerobot.robots.so101_follower"),
)
