from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch
from uuid import uuid4

import numpy as np

from revobots_arm.transport.frame_bus import FrameBusPublisher
from revobots_arm.vision import Detection, draw_detections
from revobots_arm.vision import cli
from revobots_arm.vision.gemini import GeminiDetector
from revobots_arm.vision.launch import VisionProcess
from revobots_arm.vision.yolo import YoloDetector


class VisionTests(unittest.TestCase):
    def test_yolo_tracks_and_preserves_pixel_outlines(self):
        class Boxes:
            xyxy = np.array([[-3, 12, 48, 50]])
            cls = np.array([2])
            conf = np.array([0.9])
            id = np.array([11])

            def __len__(self):
                return len(self.xyxy)

        polygon = np.array([[0, 12], [40, 12], [20, 35]])
        result = SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(xy=[polygon]), names={2: "bin"})
        model = Mock(task="segment")
        model.track.return_value = [result]
        model.predict.return_value = [result]
        sdk = SimpleNamespace(YOLO=Mock(return_value=model))
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        with patch.dict(sys.modules, {"ultralytics": sdk}):
            tracking = YoloDetector("seg.pt", tracker="bytetrack.yaml")
            detector = YoloDetector("seg.pt")
        tracked = tracking.detect(frame)[0]
        self.assertEqual(tracked, Detection("bin", (0, 12, 48, 40), 0.9, ((0, 12), (40, 12), (20, 35)), 11))
        model.track.assert_called_once_with(
            frame, persist=True, tracker="bytetrack.yaml", conf=0.25, device=None, verbose=False
        )
        detector.detect(frame)
        model.predict.assert_called_once_with(frame, conf=0.25, device=None, verbose=False)
        self.assertTrue(draw_detections(frame, [tracked]).any())
        self.assertFalse(frame.any())

    def test_gemini_scales_yxyx_to_original_pixels(self):
        sdk = Mock()
        google = SimpleNamespace(genai=sdk)
        frame = np.zeros((120, 400, 3), dtype=np.uint8)
        client = sdk.Client.return_value
        client.models.generate_content.return_value.text = (
            '[{"label": "bin", "box_2d": [100, 250, 800, 750]}]'
        )
        with patch.dict(sys.modules, {"google": google}):
            detector = GeminiDetector("model", "Find the bin", api_key="test-key")
        result = detector.detect(frame)[0]
        self.assertEqual(result, Detection("bin", (100, 12, 300, 96)))
        self.assertEqual(result.center_xy, (200, 54))
        self.assertEqual(client.models.generate_content.call_args.kwargs["model"], "model")
        client.models.generate_content.return_value.text = '[{"box_2d": [1, 2]}]'
        with self.assertRaisesRegex(ValueError, "valid list"):
            detector.detect(frame)
        detector.close()
        client.close.assert_called_once()

    def test_cli_reads_named_camera_bus_and_emits_json(self):
        camera = f"vision_test_{uuid4().hex}"
        frame = np.full((12, 16, 3), 31, dtype=np.uint8)
        publisher = FrameBusPublisher(camera, height=12, width=16, channels=3)
        self.addCleanup(publisher.close)
        timestamp = time.time()
        publisher.publish(frame, timestamp)
        detector = Mock()
        detector.detect.return_value = [Detection("bin", (1, 2, 3, 4))]
        output = io.StringIO()
        argv = ["vision", "yolo", "--camera", camera, "--model", "det.pt"]
        with patch.object(sys, "argv", argv), patch.object(cli, "YoloDetector", return_value=detector), redirect_stdout(output):
            cli.main()
        np.testing.assert_array_equal(detector.detect.call_args.args[0], frame)
        result = json.loads(output.getvalue())
        self.assertEqual(result["camera"], camera)
        self.assertEqual(result["capture_time_s"], timestamp)
        self.assertEqual(result["detections"][0]["xyxy"], [1, 2, 3, 4])

    def test_recorded_colour_frame_keeps_its_frame_index(self):
        rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
        session = Mock(meta={"cameras": {"wrist": {"video": "wrist.mp4"}}})
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock()
        session.__iter__ = Mock(return_value=iter([
            ({"frame_index": 17, "timestamp": 0.6}, {"wrist": rgb})
        ]))
        args = SimpleNamespace(image=None, session=Path("session_0"), camera="wrist")
        with patch.object(cli, "SessionFrames", return_value=session):
            [(frame, metadata)] = list(cli.frames(args))
        np.testing.assert_array_equal(frame, [[[30, 20, 10]]])
        self.assertEqual(metadata, {"camera": "wrist", "frame_index": 17, "session_time_s": 0.6})

    def test_launcher_forwards_arguments_and_stops_the_child(self):
        args = ["yolo", "--session", "saved run/session_0", "--camera", "wrist",
                "--model", "det.pt", "--follow"]
        path = Path("output/vision.jsonl")
        process = VisionProcess(args, path)
        output = mock_open()
        with patch.object(Path, "mkdir") as mkdir, patch.object(Path, "open", output), patch(
            "revobots_arm.vision.launch.subprocess.Popen"
        ) as popen:
            child = popen.return_value
            child.poll.return_value = None
            process.start()
            process.stop()
            process.stop()
        output.assert_called_once_with("wb")
        mkdir.assert_not_called()
        popen.assert_called_once_with(
            [sys.executable, "-m", "revobots_arm.vision.cli", *args],
            stdin=subprocess.DEVNULL,
            stdout=output.return_value.__enter__.return_value,
        )
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=2.0)
