from __future__ import annotations
import logging
import time
from typing import Any, Optional
import cv2
import numpy as np
from revobots_arm.config import CameraConfig

logger = logging.getLogger(__name__)
from .base import _CaptureBase, _ROTATIONS

_DEPTH_UNIT_M = 0.001


def _import_rs() -> Any:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed (pip install pyrealsense2; pyrealsense2-macosx on macOS)"
        ) from exc
    return rs


def _rs_valid(frame: Any) -> bool:
    """pyrealsense2 frames are falsy when that stream is missing from the frameset."""
    try:
        return bool(frame)
    except Exception:  # noqa: BLE001
        return frame is not None


def resolve_realsense_serial(rs: Any, source: int | str) -> str:
    """Serial number for ``source``: digits are a serial number, anything else a unique
    device name (same rule as LeRobot's ``RealSenseCamera``)."""
    src = str(source).strip()
    if src.isdigit():
        return src
    found = [
        (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
        for d in rs.context().query_devices()
    ]
    matches = [sn for name, sn in found if name == src]
    if not matches:
        raise RuntimeError(f"no RealSense named {src!r}; connected: {found}")
    if len(matches) > 1:
        raise RuntimeError(
            f"several RealSense named {src!r} ({matches}); use the serial number"
        )
    return matches[0]


class RealSenseCapture(_CaptureBase):
    """Background reader for one Intel RealSense (D4xx / D5xx), straight on pyrealsense2.

    Same design as the AGV ``LAB/cameras.py`` RealSenseCapture and the OpenCV path above:
    this thread is the only owner of the device; every frameset goes to the 1-slot holder
    (teleop / record, read with ``read_latest_pair``) and, with ``publish_frames=True``, to
    the shared-memory frame bus so any other process reads it without opening the camera:

        lab_<name>          BGR colour     uint8   (H, W, 3)
        lab_<name>_depth    depth in mm    uint16  (H, W, 1)

    The arrays match LeRobot's ``RealSenseCamera`` so the dataset is the one
    ``lerobot-record`` would write: colour and depth are enabled at the same
    width/height/fps and taken from ONE frameset (colour N and depth N belong together);
    depth is normally the unaligned z16 map, uint16 mm,
    shape (H, W, 1); a warm-up of at least 1 s runs before the first frame counts;
    rotation is applied after capture to both streams. Colour stays BGR on the bus;
    ``RobotWithCameras`` converts it to RGB for the observation, as LeRobot does.

    Depth units: LeRobot passes the raw z16 values through and calls them millimetres,
    true at the D4xx default of 1 mm. A D405 ships at 0.1 mm, so the sensor is asked for
    1 mm units before streaming, and if it refuses the values are rescaled here. Either
    way ``<name>_depth`` really is millimetres.
    """

    backend = "realsense"
    lost_after_s = 10.0  # no complete frameset for this long → reconnect

    def __init__(self, cfg: CameraConfig, *, align_depth: bool = False) -> None:
        super().__init__(cfg)
        if align_depth and not cfg.use_depth:
            raise ValueError("Aligned depth requires depth capture")
        self._align_depth = align_depth
        self._aligner = None
        self._aligned_pair = (None, None, None)
        self._depth_axis = self._depth_origin = None
        self.serial: Optional[str] = None
        self._mm_per_unit = 1.0  # raw z16 → mm; 1.0 when the sensor runs at 1 mm units

    # ── device setup ─────────────────────────────────────────────────────────

    def _request_mm_units(self, rs: Any, serial: str) -> None:
        """Ask the depth sensor for 1 mm units before streaming. Best effort."""
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            try:
                sensor = dev.first_depth_sensor()
                opt = rs.option.depth_units
                if sensor.supports(opt) and not sensor.is_option_read_only(opt):
                    if abs(float(sensor.get_option(opt)) - _DEPTH_UNIT_M) > 1e-9:
                        sensor.set_option(opt, _DEPTH_UNIT_M)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s: could not set depth units to 1 mm: %s", self.name, exc
                )
            return

    def _complete(self, frames: Any) -> bool:
        """True when every enabled stream is present in the frameset."""
        if frames is None:
            return False
        if self.use_rgb and not _rs_valid(frames.get_color_frame()):
            return False
        if self.use_depth and not _rs_valid(frames.get_depth_frame()):
            return False
        return True

    def _open(self) -> Any:
        """Start the pipeline, check the negotiated modes and depth unit, run the warm-up.
        Returns the running ``rs.pipeline``; raises on any failure."""
        rs = _import_rs()
        cfg = self.cfg
        serial = resolve_realsense_serial(rs, cfg.source)
        if cfg.use_depth:
            self._request_mm_units(rs, serial)

        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        # cfg.width x cfg.height is the capture size; rotation is applied afterwards.
        rs_cfg.enable_stream(
            rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps
        )
        if cfg.use_depth:
            rs_cfg.enable_stream(
                rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
            )

        pipe = rs.pipeline()
        profile = pipe.start(rs_cfg)
        try:
            for stream, on in (
                (rs.stream.color, True),
                (rs.stream.depth, cfg.use_depth),
            ):
                if not on:
                    continue
                vsp = profile.get_stream(stream).as_video_stream_profile()
                if (vsp.width(), vsp.height(), vsp.fps()) != (
                    cfg.width,
                    cfg.height,
                    cfg.fps,
                ):
                    raise RuntimeError(
                        f"{stream} negotiated {vsp.width()}x{vsp.height()}@{vsp.fps()}, "
                        f"config asks {cfg.width}x{cfg.height}@{cfg.fps}"
                    )

            self._mm_per_unit = 1.0
            if cfg.use_depth:
                scale_m = float(
                    profile.get_device().first_depth_sensor().get_depth_scale()
                )
                factor = scale_m / _DEPTH_UNIT_M
                if abs(factor - 1.0) > 1e-6:
                    self._mm_per_unit = factor
                    logger.warning(
                        "%s: depth unit is %.4g mm, rescaling to mm in software",
                        self.name,
                        factor,
                    )

            # Warm-up like LeRobot (>= 1 s): the first framesets after start are often incomplete.
            warm = max(1.0, float(cfg.warmup_s))
            deadline = time.monotonic() + warm
            got = False
            while time.monotonic() < deadline:
                ok, frames = pipe.try_wait_for_frames(
                    timeout_ms=int(warm * 1000) + 1000
                )
                got = got or (ok and self._complete(frames))
            if not got:
                raise RuntimeError(
                    f"no complete frameset during the {warm:.0f}s warm-up"
                )
            if self._align_depth:
                extrinsics = profile.get_stream(rs.stream.depth).get_extrinsics_to(
                    profile.get_stream(rs.stream.color)
                )
                # The third column of the depth-to-colour rotation is the native depth Z axis.
                self._depth_axis = np.asarray(extrinsics.rotation[6:9], dtype=float)
                self._depth_origin = np.asarray(extrinsics.translation, dtype=float)
                self._aligner = rs.align(rs.stream.color)
        except Exception:
            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass
            raise

        self.serial = serial
        return pipe

    def colour_depth_m(self, depth_m: float, colour_ray: np.ndarray) -> float:
        """Convert a reprojected depth sample's native Z to Z along the colour ray."""
        if self._depth_axis is None or self._depth_origin is None:
            raise ValueError("Depth-to-colour alignment is unavailable")
        x, y, z = colour_ray
        rotation = int(self.cfg.rotation or 0)
        ray = {0: (x, y, z), 90: (y, -x, z), 180: (-x, -y, z), 270: (-y, x, z)}[rotation]
        projection = float(self._depth_axis @ ray)
        if not np.isfinite(projection) or projection <= 0:
            raise ValueError("Invalid depth projection at the selected pixel")
        return (depth_m + float(self._depth_axis @ self._depth_origin)) / projection

    def read_aligned_pair(self):
        with self._lock:
            return self._aligned_pair

    def stop(self) -> None:
        super().stop()
        with self._lock:
            self._aligned_pair = (None, None, None)

    def _depth_mm(self, raw: np.ndarray) -> np.ndarray:
        if self._mm_per_unit == 1.0:
            return raw
        return (
            np.rint(raw.astype(np.float32) * self._mm_per_unit)
            .clip(0, 65535)
            .astype(np.uint16)
        )

    # ── reader loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        backoff = 1.0
        rotate = _ROTATIONS.get(int(self.cfg.rotation or 0))
        while not self._stop.is_set():
            pipe = self._initial
            self._initial = None
            if pipe is None:
                try:
                    pipe = self._open()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: RealSense reopen failed: %s", self.name, exc)
                    pipe = None
            if pipe is None:
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 1.0

            last_ok = time.monotonic()
            while not self._stop.is_set():
                try:
                    ok, frames = pipe.try_wait_for_frames(timeout_ms=500)
                    captured_at = time.time()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: wait_for_frames raised: %s", self.name, exc)
                    break
                if not ok or not self._complete(frames):
                    if time.monotonic() - last_ok > self.lost_after_s:
                        break
                    continue
                last_ok = time.monotonic()

                # Copy out: librealsense recycles these buffers once the frameset is released.
                frame = depth = None
                if self.use_rgb:
                    frame = np.array(
                        frames.get_color_frame().get_data(), dtype=np.uint8, copy=True
                    )
                    if rotate is not None:
                        frame = cv2.rotate(frame, rotate)
                if self.use_depth:
                    depth = np.array(
                        frames.get_depth_frame().get_data(), dtype=np.uint16, copy=True
                    )
                    depth = self._depth_mm(depth)
                    if rotate is not None:
                        depth = cv2.rotate(depth, rotate)
                    depth = depth[..., np.newaxis]  # (H, W) → (H, W, 1), LeRobot layout
                # One _store per frameset: holder and frame bus get colour and depth together.
                self._store(frame, depth, captured_at if self._align_depth else time.time())
                if self._aligner is not None:
                    try:
                        aligned = self._aligner.process(frames).get_depth_frame()
                    except RuntimeError as exc:
                        logger.warning("%s: depth alignment failed: %s", self.name, exc)
                        continue
                    if _rs_valid(aligned):
                        aligned_mm = self._depth_mm(np.array(aligned.get_data(), dtype=np.uint16, copy=True))
                        if rotate is not None:
                            aligned_mm = cv2.rotate(aligned_mm, rotate)
                        with self._lock:
                            self._aligned_pair = (captured_at, frame, aligned_mm[..., np.newaxis])

            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass
            if not self._stop.is_set():
                logger.warning(
                    "%s: RealSense stream lost, reconnecting in %.1fs",
                    self.name,
                    backoff,
                )
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)
