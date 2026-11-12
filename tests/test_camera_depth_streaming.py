import unittest
from unittest.mock import Mock, patch

from revobots_arm.cameras.manager import CamerasManager, make_capture
from revobots_arm.config import CameraConfig, load
from revobots_arm.config.devices import apply_overrides


class CameraDepthStreamingTests(unittest.TestCase):
    def test_realsense_keeps_rgb_with_and_without_depth(self):
        for use_depth in (False, True):
            with self.subTest(use_depth=use_depth):
                cfg = load("nero_gamepad")
                apply_overrides(
                    cfg,
                    {
                        "cameras.wrist.type": "intelrealsense",
                        "cameras.wrist.source": "123456789012",
                        "cameras.wrist.use_rgb": "false",
                        "cameras.wrist.use_depth": str(use_depth),
                        "stream.enabled": "true",
                    },
                )
                camera = cfg.camera("wrist")
                self.assertIs(cfg.stream_mounts()["ai"], camera)
                self.assertTrue(make_capture(camera).use_rgb)
                expected = ["wrist", "wrist_depth"] if use_depth else ["wrist"]
                self.assertEqual(cfg.camera_observation_keys, expected)

    def test_distance_worker_is_scoped_to_running_streamed_depth_cameras(self):
        for stream, depth, opened, rtsp_ready in (
            (True, True, True, True),
            (False, True, True, True),
            (True, False, True, True),
            (True, True, False, True),
            (True, True, True, False),
        ):
            with self.subTest(stream=stream, depth=depth, opened=opened, rtsp=rtsp_ready):
                cfg = load("nero_gamepad")
                cfg.cameras = [
                    CameraConfig("wrist", "123", type="intelrealsense", use_depth=depth)
                ]
                cfg.stream.enabled = stream
                cfg.stream.depth_robot_id = "base-001"
                cfg.stream.depth_host = "192.168.1.20"
                cfg.stream.depth_port = 57111
                manager = CamerasManager(cfg)
                capture = Mock()
                capture.start.return_value = opened
                rtsp = Mock()
                rtsp.start.return_value = rtsp_ready
                stopped = []

                def run_worker(_sampler, _destination, *, stop):
                    stopped.append(stop.wait(2))

                with (
                    patch("revobots_arm.cameras.manager.make_capture", return_value=capture),
                    patch("revobots_arm.cameras.manager.RtspServer", return_value=rtsp),
                    patch(
                        "revobots_arm.cameras.manager.run_depth_telemetry",
                        side_effect=run_worker,
                    ) as run,
                ):
                    manager.start()
                    if opened:
                        manager.start()
                    manager.stop()

                if stream and depth and opened and rtsp_ready:
                    run.assert_called_once()
                    sampler, destination = run.call_args.args
                    self.assertEqual((sampler.camera, sampler.robot_id), ("wrist", "base-001"))
                    self.assertEqual(destination, ("192.168.1.20", 57111))
                    self.assertEqual(stopped, [True])
                else:
                    run.assert_not_called()
