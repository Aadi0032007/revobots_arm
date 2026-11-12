"""SO100 follower: LeRobot owns motor control, calibration and torque handling."""

from revobots_arm.utils.lerobot import LeRobotAdapter

adapter = LeRobotAdapter(
    "robot",
    "so100_follower",
    ("lerobot.robots.so_follower", "lerobot.robots.so100_follower"),
)
