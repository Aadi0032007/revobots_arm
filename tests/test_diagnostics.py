from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

from revobots_arm import diagnostics
from revobots_arm.config import load
from revobots_arm.transport.frame_bus import FrameBusPublisher


class DiagnosticsTests(unittest.TestCase):
    def test_tailscale_reports_both_address_families(self):
        with patch.object(diagnostics.shutil, "which", return_value="tailscale"), patch.object(
            diagnostics, "run_local_command", return_value="100.64.1.2\nfd7a:115c:a1e0::1"
        ) as command:
            result = diagnostics.tailscale_addresses()
        command.assert_called_once_with(["tailscale", "ip"])
        self.assertEqual(result["addresses"], ["100.64.1.2", "fd7a:115c:a1e0::1"])

    def test_readability_rejects_missing_or_nonfinite_joint_feedback(self):
        for positions in ({}, {"joint1.pos": float("nan")}, {"joint1.pos": True}):
            with self.subTest(positions=positions), self.assertRaisesRegex(ValueError, "joint1.pos"):
                diagnostics.sample_positions(lambda: positions, {"joint1.pos"}, 0.001)

    def test_serial_probe_uses_bus_without_configuring_or_disabling_torque(self):
        bus = Mock(is_connected=True, motors={"elbow_flex": object()})
        bus.sync_read.return_value = {"elbow_flex": 2048}
        with patch.object(diagnostics, "ensure_lerobot_importable"), patch(
            "revobots_arm.config.devices.make_robot", return_value=Mock(bus=bus)
        ):
            result = diagnostics.arm_feedback(load("so101"), "robot", 0.001)
        bus.connect.assert_called_once_with(handshake=False)
        bus.sync_read.assert_called_with("Present_Position", normalize=False)
        bus.disconnect.assert_called_once_with(disable_torque=False)
        self.assertEqual(result["positions"], {"elbow_flex": 2048})
        self.assertEqual(result["unit"], "encoder counts")

    def test_nero_probe_disables_enable_rest_and_gripper(self):
        robot = Mock(observation_features={"joint1.pos": float})
        robot.get_observation.return_value = {"joint1.pos": 12.0}
        cfg = load("nero_gamepad")
        with patch("revobots_arm.robots.nero.driver.NeroJointRobot", return_value=robot) as factory:
            result = diagnostics.arm_feedback(cfg, "robot", 0.001)
        probe_config = factory.call_args.args[0]
        self.assertEqual(probe_config.channel, cfg.robot["channel"])
        self.assertFalse(probe_config.enable_on_connect)
        self.assertFalse(probe_config.rest_on_start)
        self.assertFalse(probe_config.with_gripper)
        robot.connect.assert_called_once_with(calibrate=False)
        robot.disconnect.assert_called_once_with()
        robot.send_action.assert_not_called()
        self.assertEqual(result["unit"], "degrees")

    def test_camera_probe_reports_age_and_rejects_a_stalled_publisher(self):
        key = f"diag_{uuid4().hex[:12]}_depth"
        publisher = FrameBusPublisher(key, height=2, width=3, channels=1, dtype=np.uint16)
        frame = np.ones((2, 3, 1), dtype=np.uint16)
        stop = threading.Event()

        def publish():
            while not stop.wait(0.01):
                publisher.publish(frame, time.time())

        thread = threading.Thread(target=publish)
        try:
            publisher.publish(frame, time.time())
            self.assertFalse(diagnostics.camera_feedback(key, 0.02)["ok"])
            thread.start()
            result = diagnostics.camera_feedback(key, 0.25)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["shape"], [2, 3, 1])
            self.assertEqual(result["dtype"], "uint16")
            self.assertGreaterEqual(result["frame_age"]["p95_ms"], 0)
        finally:
            stop.set()
            if thread.ident is not None:
                thread.join()
            publisher.close()

    def test_udp_connection_and_inter_arrival_measurement(self):
        sender = ("127.0.0.1", 6000)
        with patch.object(diagnostics.socket, "socket") as socket_factory, patch.object(
            diagnostics.time, "monotonic", side_effect=[0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3]
        ):
            receiver = socket_factory.return_value.__enter__.return_value
            receiver.recvfrom.side_effect = [
                (b"broken", sender), (b'{"seq":1}', sender), (b'{"seq":2}', sender), TimeoutError(),
            ]
            result = diagnostics.udp_feedback("127.0.0.1", 5001, 1.0)
        receiver.bind.assert_called_once_with(("127.0.0.1", 5001))
        socket_factory.return_value.__exit__.assert_called_once()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["json_packets"], 2)
        self.assertEqual(result["invalid_packets"], 1)
        self.assertEqual(result["source_ips"], ["127.0.0.1"])
        self.assertEqual(result["arrival_gap"]["p95_ms"], 100)

    def test_so_remote_joint_client_opts_into_nero_mapping_only_for_nero(self):
        from revobots_arm.teleoperators.leader import client

        for preset in ("nero", "so101_remote"):
            cfg = load(preset)
            cfg.teleop["type"] = "so101_leader_remote"
            with self.subTest(preset=preset), patch.object(client, "load_config", return_value=cfg), patch(
                "sys.argv", ["revobots-arm-leader-client"]
            ), patch.dict("sys.modules", {"lerobot.utils.utils": Mock(init_logging=Mock())}), patch(
                "revobots_arm.teleoperators.leader.so101.run_remote_client"
            ) as run:
                client.main()
                self.assertTrue(run.call_args.kwargs["use_degrees"])
                self.assertEqual(run.call_args.kwargs.get("is_follower_nero"), True if preset == "nero" else None)
