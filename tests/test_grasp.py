from __future__ import annotations

from threading import Event
import time
import unittest
from unittest.mock import Mock

import numpy as np

from revobots_arm.cameras.realsense import RealSenseCapture
from revobots_arm.config import CameraConfig
from revobots_arm.grasp.motion import MotionStopped, NeroPickMotion, Plan, surface_depth_m
from revobots_arm.grasp.session import PickSession
from revobots_arm.hand_eye.calibration import HandEyeCalibration


def calibration() -> HandEyeCalibration:
    return HandEyeCalibration(
        camera="wrist", robot_id=None, base_link="base_link", camera_link="link7",
        image_size_wh=(9, 9), camera_matrix=[[8, 0, 4], [0, 8, 4], [0, 0, 1]],
        dist_coeffs=[0] * 5, link_from_camera=np.eye(4).tolist(),
        board_square_mm=35, board_marker_mm=26, sample_frames=[],
        reprojection_rms_px=0, board_translation_rms_mm=0, board_rotation_rms_deg=0,
    )


class ImmediateEvent(Event):
    """Execute a simulated robot's planned commands without wall-clock delays."""

    def wait(self, timeout=None):
        return self.is_set()


class GraspTests(unittest.TestCase):
    def test_depth_selection_keeps_only_samples_from_the_marked_surface(self):
        depth = np.full((9, 9, 1), 400, dtype=np.uint16)
        depth[2:4, :, 0] = 700
        self.assertAlmostEqual(surface_depth_m(depth, (4, 4)), 0.4)
        depth[4, 4, 0] = 0
        with self.assertRaisesRegex(ValueError, "Insufficient depth"):
            surface_depth_m(depth, (4, 4))

    def test_sdk_aligned_depth_is_converted_to_colour_axis_after_rotation(self):
        cfg = CameraConfig("wrist", "123", type="intelrealsense", use_depth=True, publish_frames=False, rotation=90)
        capture = RealSenseCapture(cfg, align_depth=True)
        capture._depth_axis = np.array([0.1, 0, np.sqrt(0.99)])
        capture._depth_origin = np.array([0.02, 0, 0.005])
        native_ray = np.array([0.25, -0.2, 1])
        colour_point = native_ray * 0.42
        native_depth = capture._depth_axis @ (colour_point - capture._depth_origin)
        self.assertAlmostEqual(capture.colour_depth_m(native_depth, np.array([0.2, 0.25, 1])), 0.42)

    def test_preflight_and_execution_order_using_native_nero_ik(self):
        robot = Mock()
        robot.config.gripper_mode = "width"
        motion = NeroPickMotion(
            robot, calibration(), tcp_xyz_m=np.array([0, 0, 0.138]), tcp_rpy_deg=None,
            table_z_m=0, clearance_m=0.01, max_travel_m=0.4, fps=30,
            gripper_open=0.07, gripper_close=0,
        )
        q = np.radians([-83.5897, -23.2644, -74.6794, 107.5022, -17.5783, 26.8008, 78.5652])
        start = {**motion.tool.q_to_motor(q), "gripper.pos": 0.04}
        feedback = start.copy()
        robot.get_observation.side_effect = lambda: feedback.copy()
        robot.send_action.side_effect = feedback.update
        target = np.array([0.265, 0, 0.14])
        with self.assertRaisesRegex(ValueError, "table clearance"):
            motion.plan("grasp", np.array([0.265, 0, 0.014]), q, 0.07, 0.01, 0, Event())
        plan = motion.plan("grasp", target, q, 0.07, 0.01, 0, Event())
        robot.send_action.assert_not_called()
        robot.enable.assert_not_called()

        feedback["joint1.pos"] += 2
        with self.assertRaisesRegex(ValueError, "moved after planning"):
            motion.execute(plan, ImmediateEvent(), lambda _: None)
        robot.enable.assert_not_called()
        feedback.update(start)

        phases = []
        motion.execute(plan, ImmediateEvent(), phases.append)
        self.assertEqual(phases[-4:], ["Open gripper", "Descend", "Close gripper", "Lift"])
        grips = [call.args[0]["gripper.pos"] for call in robot.send_action.call_args_list if "gripper.pos" in call.args[0]]
        self.assertEqual(grips, [0.07, 0])
        np.testing.assert_allclose(motion.tool.fk_motors(feedback)[:3, 3], plan.hover, atol=0.001)

        robot.reset_mock()
        feedback.update(start)
        stop = ImmediateEvent()
        with self.assertRaises(MotionStopped):
            motion.execute(plan, stop, lambda _: stop.set())
        self.assertEqual(robot.send_action.call_args_list[-1].args[0], motion.tool.q_to_motor(q))
        self.assertTrue(all("gripper.pos" not in call.args[0] for call in robot.send_action.call_args_list))

    def test_target_requires_a_fresh_settled_frame_and_confirmation_is_single_use(self):
        camera, motion = Mock(), Mock()
        colour = np.zeros((9, 9, 3), dtype=np.uint8)
        depth = np.full((9, 9, 1), 400, dtype=np.uint16)
        camera.read_aligned_pair.side_effect = lambda: (time.time(), colour, depth)
        camera.colour_depth_m.side_effect = lambda value, _: value
        motion.calibration = calibration()
        motion.wrist.fk.return_value = np.eye(4)
        motion.joints.return_value = np.zeros(7)
        motion.can_grasp = True
        point = np.array([0, 0, 0.4])
        motion.plan.return_value = Plan("hover", np.zeros(7), point + [0, 0, 0.1], None, ())

        def wait_for(state):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = session.status()
                if status["state"] == state:
                    return status
                time.sleep(0.005)
            self.fail(f"Expected {state}, got {session.status()}")

        session = PickSession(camera, motion, execute=True)
        try:
            self.assertEqual(session.frame()[1], "")
            _, key = session.frame()
            self.assertTrue(key)
            motion.joints.return_value = np.ones(7)
            with self.assertRaisesRegex(ValueError, "Arm moved"):
                session.select(key, 4, 4)
            motion.joints.return_value = np.zeros(7)
            session.select(key, 4, 4)
            self.assertEqual(session.status()["target_mm"], [0, 0, 400])
            session.prepare("hover", 100, 20, 0)
            plan_id = wait_for("planned")["plan_id"]
            self.assertTrue(plan_id)
            motion.execute.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Preview the path"):
                session.confirm("wrong")
            with unittest.mock.patch.object(camera, "read_aligned_pair", return_value=(time.time() - 2, colour, depth)):
                with self.assertRaisesRegex(ValueError, "fresh RealSense"):
                    session.confirm(plan_id)
            motion.execute.assert_not_called()
            session.confirm(plan_id)
            wait_for("done")
            motion.execute.assert_called_once()
            with self.assertRaises(ValueError):
                session.confirm(plan_id)
        finally:
            session.close()
