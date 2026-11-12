import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from revobots_arm.config.devices import load_config
from revobots_arm.teleoperators.leader.cartesian import CartesianRetargeter
from revobots_arm.teleoperators.leader.koch import KOCH_ARM_MOTORS
from revobots_arm.teleoperators.leader.joint_stream import (
    PACKET_TYPE,
    RemoteDegreeLeader,
    stream_joint_degrees,
)
from revobots_arm.transport.ai_link import UdpJsonListener


class LeaderIKTests(unittest.TestCase):
    def test_so101_drives_all_seven_native_nero_targets(self):
        with patch("revobots_arm.config.devices.ensure_lerobot_importable"):
            legacy = load_config("nero", ["--ik=false"])
            cfg = load_config("nero", ["--ik=true", "--teleop.type=so101_leader"])
        self.assertEqual(legacy.robot_type, "agx_nero_follower")
        self.assertEqual(cfg.robot_type, "agx_nero_native")
        self.assertFalse(cfg.robot["rest_on_start"])
        self.assertNotIn("teleop_action_joints", cfg.robot)

        retarget = CartesianRetargeter(cfg)
        leader_start = retarget.leader.q_to_motor(np.zeros(5))
        leader_move = retarget.leader.q_to_motor(np.radians([3, 2, -4, 1, 5]))
        follower_start = retarget.follower.q_to_motor(np.radians([0, 20, 0, 60, 0, 10, 0]))
        feedback = follower_start
        self.assertIsNone(retarget.action(leader_start, feedback))
        for _ in range(25):
            command = retarget.action(leader_move, feedback)
            if command is None:
                break
            self.assertEqual(list(command), [f"joint{i}.pos" for i in range(1, 8)])
            self.assertLessEqual(max(abs(command[k] - feedback[k]) for k in command), 1.00001)
            feedback = command
        expected_delta = (
            retarget.leader.fk_motors(leader_move)[:3, 3]
            - retarget.leader.fk_motors(leader_start)[:3, 3]
        )
        actual_delta = (
            retarget.follower.fk_motors(feedback)[:3, 3]
            - retarget.follower.fk_motors(follower_start)[:3, 3]
        )
        np.testing.assert_allclose(actual_delta, expected_delta, atol=0.001)
        self.assertGreater(abs(feedback["joint3.pos"] - follower_start["joint3.pos"]), 0.1)
        self.assertGreater(abs(feedback["joint6.pos"] - follower_start["joint6.pos"]), 0.1)
        self.assertIsNone(retarget.action(None, feedback))
        self.assertIsNone(retarget.action(leader_start, feedback))  # no jump on reconnect

    def test_koch_uses_the_bundled_jaw_pivot_and_rahul_calibration(self):
        with patch("revobots_arm.config.devices.ensure_lerobot_importable"):
            cfg = load_config("nero", [
                "--ik=true", "--teleop.type=koch_leader", "--ik.leader_calibration=rahul",
            ])
        retarget = CartesianRetargeter(cfg)
        leader = retarget.leader
        self.assertEqual(leader.chain.joint_names, [f"joint_{i}" for i in range(1, 6)])
        self.assertEqual(leader.keys, [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
        ])
        zero = dict(zip(leader.keys, (1.0, 0.9, -4.4, -5.6, -7.9)))
        np.testing.assert_allclose(leader.motor_to_q(zero), np.zeros(5), atol=1e-15)
        np.testing.assert_allclose(leader.fk_motors(zero)[:3, 3], [-0.16143, -0.008753, 0.1817999])
        self.assertEqual(retarget.follower.chain.tip_link, "gripper_tcp")
        self.assertEqual(retarget.position_scale, 2.75)

        feedback = retarget.follower.q_to_motor(np.radians([0, 20, 0, 60, 0, 10, 0]))
        self.assertIsNone(retarget.action(zero, feedback))
        lead_xy = leader.fk_motors(zero)[:2, 3]
        follower_xy = retarget.follower.fk_motors(feedback)[:2, 3]
        np.testing.assert_allclose(
            retarget.alignment[:2, :2] @ (lead_xy / np.linalg.norm(lead_xy)),
            follower_xy / np.linalg.norm(follower_xy),
        )
        action = retarget.action({**zero, "shoulder_pan.pos": 3.0}, feedback)
        self.assertIsNotNone(action)
        self.assertEqual(set(action), set(feedback))

    def test_koch_accepts_a_custom_urdf_joint_mapping(self):
        from revobots_arm.inverse_kinematics.so101 import SO101_URDF, SO101_TIP_LINK

        with patch("revobots_arm.config.devices.ensure_lerobot_importable"):
            cfg = load_config("nero", ["--ik=true", "--teleop.type=koch_leader"])
        # A fixture chain with different motor labels exercises the generic binding.
        cfg.ik.leader_urdf = str(SO101_URDF)
        cfg.ik.leader_tip_link = SO101_TIP_LINK
        cfg.ik.leader_motors = "m1,m2,m3,m4,m5"
        retarget = CartesianRetargeter(cfg)
        self.assertEqual(retarget.leader.keys, [f"m{i}.pos" for i in range(1, 6)])
        self.assertEqual(retarget.leader.chain.joint_names[0], "shoulder_pan")
        self.assertEqual(retarget.position_scale, 1.0)

    def test_remote_leader_reanchors_on_sender_restart(self):
        leader = RemoteDegreeLeader({"type": "so101_leader_remote"})
        leader.listener = Mock()
        old = {"session": "old", "leader_type": "so101_leader", "action": {"shoulder_pan.pos": 10.0}}
        new = {**old, "session": "new", "action": {"shoulder_pan.pos": 40.0}}
        wrong = {**old, "leader_type": "koch_leader"}
        leader.listener.latest.side_effect = [old, old, wrong, None, new, new]
        self.assertEqual(leader.get_action(), {})
        self.assertEqual(leader.get_action(), old["action"])
        self.assertEqual(leader.get_action(), {})
        self.assertEqual(leader.get_action(), {})
        self.assertEqual(leader.get_action(), {})
        self.assertEqual(leader.get_action(), new["action"])

    def test_koch_sender_calibrates_urdf_degrees_for_the_nero_receiver(self):
        with patch("revobots_arm.config.devices.ensure_lerobot_importable"):
            sender_cfg = load_config("nero", ["--ik=true", "--ik.leader_calibration=rahul"])
            follower_cfg = load_config("nero", ["--ik=true"])
        leader = Mock(is_connected=True)
        leader.bus.motors = {
            name: SimpleNamespace(id=index) for index, name in enumerate(KOCH_ARM_MOTORS, 1)
        }
        leader.bus._unnormalize.return_value = {1: 2304, 2: 2048, 3: 2048, 4: 2048, 5: 2048}
        reading = dict.fromkeys((f"{name}.pos" for name in KOCH_ARM_MOTORS), 15)
        reading["gripper.pos"] = 55
        leader.action_features = dict.fromkeys(reading, float)
        leader.get_action.side_effect = [reading, KeyboardInterrupt()]
        with (
            patch("revobots_arm.teleoperators.leader.joint_stream.make_teleop", return_value=leader) as factory,
            patch("revobots_arm.teleoperators.leader.joint_stream.UdpJsonSender") as sender,
            patch("revobots_arm.teleoperators.leader.joint_stream.time.sleep"),
        ):
            stream_joint_degrees(
                sender_cfg, ip="127.0.0.1", port=5001, com="COM3", arm_id="rahul", fps=60,
                gripper_open_pos=45.0,
            )
        self.assertEqual(factory.call_args.args[0].teleop_type, "koch_leader")
        leader.bus._unnormalize.assert_called_once_with(dict.fromkeys(range(1, 6), 15))
        payload = sender.return_value.send.call_args.args[0]
        self.assertEqual(payload["type"], PACKET_TYPE)
        self.assertEqual(payload["leader_type"], "koch_leader")
        self.assertEqual(payload["action"]["gripper.pos"], 55.0)
        leader.disconnect.assert_called_once()

        receiver = RemoteDegreeLeader(follower_cfg.teleop)
        receiver.listener = Mock()
        receiver.listener.latest.return_value = payload
        self.assertEqual(receiver.get_action(), {})  # first packet resets the pose anchor
        model = CartesianRetargeter(follower_cfg).leader
        np.testing.assert_allclose(
            model.motor_to_q(receiver.get_action()), np.radians([-21.5, 0.9, 4.4, -5.6, -7.9]),
        )

        follower_cfg.ik.leader_calibration = "rahul"
        with self.assertRaisesRegex(ValueError, "revobots-arm-leader-client"):
            CartesianRetargeter(follower_cfg)

    def test_tagged_listener_filters_legacy_and_reordered_packets_and_accepts_restart(self):
        listener = UdpJsonListener("127.0.0.1", 5002, packet_type=PACKET_TYPE)
        packets = [
            {"seq": 1000, "action": {}},
            {"type": "leader_joint_degrees", "session": "encoder", "seq": 10},
            {"type": PACKET_TYPE, "session": "old", "seq": 8},
            {"type": PACKET_TYPE, "session": "old", "seq": 4},
            {"type": PACKET_TYPE, "session": "new", "seq": 1},
        ]
        listener._sock = Mock()
        listener._sock.recvfrom.side_effect = [
            *((json.dumps(packet).encode(), None) for packet in packets), OSError()
        ]
        listener._run()
        self.assertEqual(listener.received, 2)
        self.assertEqual(listener.latest(), packets[-1])
