"""Run vision on an image, a recorded camera, or the live arm frame bus."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, closing, redirect_stdout
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from revobots_arm.recording.session import SessionFrames
from revobots_arm.transport.frame_bus import FrameBusReader

from .detection import Detector, draw_detections
from .gemini import GeminiDetector
from .yolo import YoloDetector


def frames(args: argparse.Namespace):
    if args.image is not None:
        data = np.frombuffer(args.image.expanduser().read_bytes(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode image {args.image}")
        yield frame, {"image": str(args.image)}
        return

    if args.session is not None:
        session = SessionFrames(args.session.expanduser())
        info = session.meta.get("cameras", {}).get(args.camera)
        if info is None or "depth" in info:
            raise ValueError(f"Expected a recorded colour camera. Channels: {session.camera_keys}")
        with session:
            for row, images in session:
                frame = cv2.cvtColor(images[args.camera], cv2.COLOR_RGB2BGR)
                yield frame, {
                    "camera": args.camera,
                    "frame_index": row["frame_index"],
                    "session_time_s": row["timestamp"],
                }
        return

    with closing(FrameBusReader(args.camera)) as reader:
        deadline = time.monotonic() + 5.0
        while True:
            timestamp, frame = reader.read_new()
            if frame is not None and timestamp is not None and 0 <= time.time() - timestamp <= 1.0:
                yield frame, {
                    "camera": args.camera,
                    "frame_number": reader.last_frame_idx,
                    "capture_time_s": timestamp,
                }
                deadline = time.monotonic() + 5.0
            elif time.monotonic() >= deadline:
                raise RuntimeError(f"No fresh frames from {args.camera!r}. Check the camera and publish_frames.")
            else:
                time.sleep(0.02)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="backend", required=True)
    yolo = sub.add_parser("yolo", help="Local YOLO detection, segmentation or RGB tracking")
    yolo.add_argument("--confidence", type=float, default=0.25)
    yolo.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
    yolo.add_argument("--tracker", help="Tracker config for consecutive frames, e.g. bytetrack.yaml")
    gemini = sub.add_parser("gemini", help="Prompt-based object boxes via the Gemini API")
    gemini.add_argument("--prompt", required=True, help="Describe the object(s) to find")

    for backend, default_fps in ((yolo, 10.0), (gemini, 1.0)):
        source = backend.add_mutually_exclusive_group(required=True)
        source.add_argument("--image", type=Path, help="Image file")
        source.add_argument("--camera", help="Live or recorded colour camera key")
        backend.add_argument("--session", type=Path, help="Read --camera from a finalized session")
        backend.add_argument("--model", required=True, help="YOLO checkpoint or Gemini model ID")
        backend.add_argument("--follow", action="store_true", help="Process successive camera frames")
        backend.add_argument(
            "--max-fps", type=float, default=default_fps,
            help="Maximum detection rate with --follow",
        )
        backend.add_argument("--annotated", type=Path, help="Write a single annotated image")

    args = parser.parse_args(argv)
    if args.image is not None and (args.follow or args.session is not None):
        parser.error("--follow and --session require --camera")
    if args.follow and args.annotated is not None:
        parser.error("--annotated writes a single frame. Use it without --follow")
    if args.backend == "yolo" and args.tracker and not args.follow:
        parser.error("--tracker requires --follow")
    if not math.isfinite(args.max_fps) or args.max_fps <= 0:
        parser.error("--max-fps must be positive and finite")
    return parser, args


def main() -> None:
    parser, args = parse_args()
    try:
        with ExitStack() as stack:
            detector: Detector
            with redirect_stdout(sys.stderr):
                if args.backend == "yolo":
                    detector = YoloDetector(
                        args.model, confidence=args.confidence, device=args.device,
                        tracker=args.tracker,
                    )
                else:
                    detector = stack.enter_context(closing(GeminiDetector(args.model, args.prompt)))
            source = stack.enter_context(closing(frames(args)))
            processed = False
            for frame, metadata in source:
                started = time.monotonic()
                with redirect_stdout(sys.stderr):
                    detections = detector.detect(frame)
                if args.annotated is not None:
                    output = args.annotated.expanduser()
                    if args.image is not None and output.resolve() == args.image.expanduser().resolve():
                        raise ValueError("Choose a different path for the annotated image")
                    ok, encoded = cv2.imencode(output.suffix, draw_detections(frame, detections))
                    if not ok:
                        raise RuntimeError(f"Could not encode {output}")
                    output.write_bytes(encoded.tobytes())
                result = {
                    **metadata,
                    "image_size_wh": frame.shape[:2][::-1],
                    "detections": [asdict(item) for item in detections],
                }
                print(json.dumps(result, allow_nan=False), flush=True)
                processed = True
                if not args.follow:
                    break
                time.sleep(max(0.0, 1.0 / args.max_fps - (time.monotonic() - started)))
            if not processed:
                raise RuntimeError("No frames available in the session")
    except KeyboardInterrupt:
        pass
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        parser.exit(1, f"Vision: {exc}\n")


if __name__ == "__main__":
    main()
