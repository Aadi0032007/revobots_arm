from __future__ import annotations
import logging
import time
from typing import Any
import cv2
import numpy as np
from revobots_arm.config import ArmConfig, depth_key

logger = logging.getLogger(__name__)
import glob
import platform
from .manager import CamerasManager
from .manager import RobotWithCameras as RobotWithCameras, make_capture as make_capture
from .base import Frame as Frame, FramePair as FramePair
from .opencv import UsbCameraCapture as UsbCameraCapture
from .realsense import (
    RealSenseCapture as RealSenseCapture,
    resolve_realsense_serial as resolve_realsense_serial,
)
from revobots_arm.streaming.rtsp import RtspServer as RtspServer
from .realsense import _import_rs


def colorize_depth(depth: np.ndarray, max_mm: float = 4000.0) -> np.ndarray:
    """uint16 mm depth → BGR colour map for display (0 .. max_mm)."""
    d = depth[..., 0] if depth.ndim == 3 else depth
    scaled = cv2.convertScaleAbs(d, alpha=255.0 / max_mm)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def find_opencv_cameras() -> None:
    """Probe candidate OpenCV devices and print their default modes."""
    if platform.system() == "Linux":
        targets: list[int | str] = sorted(glob.glob("/dev/video*"))
        api = cv2.CAP_V4L2
    else:
        targets = list(range(10))
        api = cv2.CAP_ANY
    print(
        f"--- OpenCV cameras (type=opencv) ---\n{'source':<14} {'default mode':<16} note"
    )
    found = 0
    for target in targets:
        cap = cv2.VideoCapture(target, api)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, _ = cap.read()
        cap.release()
        found += 1
        print(
            f"{str(target):<14} {f'{w}x{h}@{fps:.0f}':<16} {'ok' if ok else 'opened but no frame (metadata node?)'}"
        )
    if not found:
        print("none")
    print()


def _rs_info(dev: Any, key: Any) -> str:
    try:
        return str(dev.get_info(key)) if dev.supports(key) else "?"
    except Exception:  # noqa: BLE001
        return "?"


def find_realsense_cameras() -> None:
    """List RealSense devices: serial, name, firmware, USB type and the default colour /
    depth modes (what ``lerobot-find-cameras realsense`` prints), straight from pyrealsense2."""
    print("--- RealSense cameras (type=intelrealsense, source=<serial number>) ---")
    try:
        rs = _import_rs()
    except RuntimeError as exc:
        print(f"{exc}; skipping.\n")
        return
    try:
        devices = list(rs.context().query_devices())
    except Exception as exc:  # noqa: BLE001
        print(f"RealSense search failed: {exc}\n")
        return
    if not devices:
        print("none\n")
        return
    for dev in devices:
        modes = []
        try:
            for sensor in dev.query_sensors():
                for prof in sensor.get_stream_profiles():
                    if prof.is_video_stream_profile() and prof.is_default():
                        v = prof.as_video_stream_profile()
                        modes.append(
                            f"{v.stream_name()} {v.width()}x{v.height()}@{v.fps()} {v.format().name}"
                        )
        except Exception as exc:  # noqa: BLE001
            modes.append(f"profiles unavailable ({exc})")
        print(
            f"serial {_rs_info(dev, rs.camera_info.serial_number)}  "
            f"name {_rs_info(dev, rs.camera_info.name)!r}  "
            f"fw {_rs_info(dev, rs.camera_info.firmware_version)}  "
            f"usb {_rs_info(dev, rs.camera_info.usb_type_descriptor)}"
        )
        print(f"    default: {', '.join(modes) or '?'}")
    print(
        "\nUse: --cameras.<name>.type=intelrealsense --cameras.<name>.source=<serial> [--cameras.<name>.use_depth=true]\n"
    )


def find_cameras(camera_type: str | None = None) -> None:
    if camera_type in (None, "opencv"):
        find_opencv_cameras()
    if camera_type in (None, "realsense"):
        find_realsense_cameras()


def test_cameras(cfg: ArmConfig, seconds: float) -> None:
    cams = CamerasManager(cfg)
    cams.start()
    if cams.failed:
        print(f"failed to open: {cams.failed}")
    if not cams.captures:
        return
    missing = cams.wait_ready(5.0)
    if missing:
        print(f"no frames within 5s from: {missing}")
    start_counts = {n: c.frames_read for n, c in cams.captures.items()}
    t0 = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - t0
    print(
        f"\n{'camera':<8} {'backend':<10} {'fps':>6}  {'colour (BGR)':<16} {'depth (mm)':<16} bus"
    )
    for name, cap in cams.captures.items():
        fps = (cap.frames_read - start_counts[name]) / elapsed
        _, frame = cap.read_latest()
        _, dmap = cap.read_latest_depth()
        shape = str(tuple(frame.shape)) if frame is not None else "-"
        dshape = str(tuple(dmap.shape)) if dmap is not None else "-"
        flag = "" if fps >= 0.9 * cap.cfg.fps else "   <- below configured fps"
        bus = ", ".join(cap.bus_names) or "no"
        print(
            f"{name:<8} {cap.backend:<10} {fps:>6.1f}  {shape:<16} {dshape:<16} {bus}{flag}"
        )
    cams.stop()


def preview(cfg: ArmConfig) -> None:
    cams = CamerasManager(cfg)
    cams.start()
    if not cams.captures:
        print("No cameras opened.")
        return
    try:
        while True:
            for name, cap in cams.captures.items():
                _, bgr = cap.read_latest()
                if bgr is not None:
                    cv2.imshow(name, bgr)
                _, dmap = cap.read_latest_depth()
                if dmap is not None:
                    cv2.imshow(depth_key(name), colorize_depth(dmap))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cams.stop()
        cv2.destroyAllWindows()


def main() -> None:
    from revobots_arm.config.devices import load_config, make_parser

    parser = make_parser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_find = sub.add_parser(
        "find", help="List OpenCV devices and RealSense serial numbers"
    )
    p_find.add_argument(
        "camera_type", nargs="?", default=None, choices=["opencv", "realsense"]
    )
    p_test = sub.add_parser(
        "test", help="Start the configured cameras and measure delivered fps"
    )
    p_test.add_argument(
        "--record_time_s",
        type=float,
        default=5.0,
        help="How long to measure (like lerobot-find-cameras)",
    )
    sub.add_parser(
        "preview",
        help="Live windows of the configured cameras (colour + depth), q to quit",
    )
    args, overrides = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.command == "find":
        find_cameras(args.camera_type)
        return
    cfg = load_config(args.preset, overrides, deployment=args.deployment)
    if args.command == "test":
        test_cameras(cfg, args.record_time_s)
    elif args.command == "preview":
        preview(cfg)


if __name__ == "__main__":
    main()
