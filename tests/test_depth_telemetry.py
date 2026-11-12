"""Offline depth checks: synthetic arrays/shared memory, no camera or cloud."""
import contextlib
import json
import sys
import unittest
import uuid
from unittest.mock import Mock, patch

import numpy as np

from revobots_arm.telemetry.depth import DepthSampler, center_depth, run
from revobots_arm.transport.frame_bus import FrameBusPublisher


class DepthTests(unittest.TestCase):
    def test_centre_patch_ignores_background_and_invalid_depth(self):
        depth = np.full((25, 25, 1), 9000, np.uint16)
        depth[7:18, 7:18] = 250
        depth[12, 12, 0] = 0
        depth[11, 12, 0] = 65535
        metres, fraction = center_depth(depth)
        self.assertEqual(metres, 0.25)
        self.assertAlmostEqual(fraction, 119 / 121)
        self.assertEqual(center_depth(np.zeros((11, 11), np.uint16)), (None, 0))

    def test_bus_must_be_millimetre_uint16_with_valid_shape(self):
        for frame in (np.zeros((11, 11, 3), np.uint16),
                      np.zeros((11, 11), np.uint8), np.zeros((3, 3), np.uint16)):
            with self.assertRaises(ValueError):
                center_depth(frame)
        with self.assertRaises(ValueError):
            center_depth(np.zeros((11, 11), np.uint16), 4)

    def test_duplicate_frames_are_not_recalculated_and_stale_value_is_cleared(self):
        reader = Mock(last_frame_idx=1)
        reader.read_new.side_effect = [(100, np.full((11, 11, 1), 300, np.uint16)),
                                      (None, None), (None, None)]
        factory = Mock(return_value=reader)
        sampler = DepthSampler("wrist", "base", reader_factory=factory)
        with patch("revobots_arm.telemetry.depth.center_depth", wraps=center_depth) as calc:
            first = sampler.sample(now=100, monotonic=10)
            duplicate = sampler.sample(now=100.3, monotonic=10.3)
            stale = sampler.sample(now=102, monotonic=12)
        self.assertEqual(first["depth_m"], 0.3)
        self.assertTrue(duplicate["valid"])
        self.assertEqual(stale["status"], "stale")
        self.assertIsNone(stale["depth_m"])
        calc.assert_called_once()
        factory.assert_called_once_with("wrist_depth")
        reader.close.assert_called_once()

    def test_invalid_new_depth_does_not_retain_previous_distance(self):
        reader = Mock(last_frame_idx=1)
        reader.read_new.side_effect = [(100, np.full((11, 11), 250, np.uint16)),
                                      (100.3, np.zeros((11, 11), np.uint16))]
        sampler = DepthSampler("wrist", "base", reader_factory=Mock(return_value=reader))
        sampler.sample(now=100, monotonic=10)
        invalid = sampler.sample(now=100.3, monotonic=10.3)
        self.assertEqual(invalid["status"], "invalid")
        self.assertFalse(invalid["valid"])
        self.assertIsNone(invalid["depth_m"])
        sampler.close()

    def test_future_timestamp_and_wrong_bus_dtype_are_errors(self):
        for ts, dtype in ((103, np.uint16), (100, np.uint8)):
            reader = Mock(last_frame_idx=1)
            reader.read_new.return_value = (ts, np.full((11, 11), 20, dtype))
            sampler = DepthSampler("wrist", "base", reader_factory=Mock(return_value=reader))
            result = sampler.sample(now=100, monotonic=10)
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["depth_m"])
            reader.close.assert_called_once()

    def test_live_depth_bus_and_replacement_publisher(self):
        camera = "test_" + uuid.uuid4().hex
        tracker = patch("multiprocessing.resource_tracker.unregister") if sys.platform == "win32" else contextlib.nullcontext()
        with tracker:
            sampler = DepthSampler(camera, "base")
            pub = FrameBusPublisher(camera + "_depth", 11, 11, 1, dtype=np.uint16)
            try:
                pub.publish(np.full((11, 11, 1), 400, np.uint16), 100)
                self.assertEqual(sampler.sample(now=100, monotonic=10)["depth_m"], 0.4)
                pub.close()
                self.assertFalse(sampler.sample(now=102, monotonic=12)["valid"])
                pub = FrameBusPublisher(camera + "_depth", 11, 11, 1, dtype=np.uint16)
                pub.publish(np.full((11, 11, 1), 200, np.uint16), 102.3)
                self.assertEqual(sampler.sample(now=102.3, monotonic=12.3)["depth_m"], 0.2)
            finally:
                sampler.close()
                pub.close()

    def test_three_hz_worker_does_not_catch_up_or_open_camera(self):
        stop = Mock()
        stop.is_set.side_effect = [False, False, True]
        sampler = Mock()
        sampler.sample.return_value = {"depth_m": 0.3}
        with patch("revobots_arm.telemetry.depth.socket.socket") as sock, \
             patch("revobots_arm.telemetry.depth.time.monotonic", side_effect=[0, 0.02, 0.34, 0.36]):
            run(sampler, ("127.0.0.1", 57110), stop=stop)
        self.assertEqual(sampler.sample.call_count, 2)
        for call in stop.wait.call_args_list:
            self.assertAlmostEqual(call.args[0], 1 / 3 - 0.02)
        udp = sock.return_value.__enter__.return_value
        udp.setblocking.assert_called_once_with(False)
        self.assertEqual(json.loads(udp.sendto.call_args.args[0]), {"depth_m": 0.3})
        sampler.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
