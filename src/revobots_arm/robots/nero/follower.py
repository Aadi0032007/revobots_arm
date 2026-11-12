"""NERO follower: LeRobot owns motor control, calibration and torque handling."""

from revobots_arm.utils.lerobot import LeRobotAdapter

adapter = LeRobotAdapter(
    "robot", "agx_nero_follower", ("lerobot.robots.agx_nero_follower",)
)
