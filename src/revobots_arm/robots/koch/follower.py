"""KOCH follower: LeRobot owns motor control, calibration and torque handling."""

from revobots_arm.utils.lerobot import LeRobotAdapter

adapter = LeRobotAdapter("robot", "koch_follower", ("lerobot.robots.koch_follower",))
