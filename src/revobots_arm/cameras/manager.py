from __future__ import annotations
import logging
import threading
import time
from typing import Any, Optional
import cv2
import numpy as np
from revobots_arm.config import ArmConfig, CameraConfig, depth_key

logger = logging.getLogger(__name__)
from .base import _CaptureBase, Frame, FramePair
from .opencv import UsbCameraCapture
from .realsense import RealSenseCapture
from revobots_arm.streaming.rtsp import RtspServer
from revobots_arm.telemetry.depth import DepthSampler, run as run_depth_telemetry


def make_capture(cfg: CameraConfig) -> _CaptureBase:
    if cfg.is_realsense:
        return RealSenseCapture(cfg)
    return UsbCameraCapture(cfg)


class CamerasManager:
    """Own enabled cameras, RTSP streaming, and distance sampling for streamed depth cameras."""

    def __init__(self, cfg: ArmConfig, *, aligned_camera: str | None = None) -> None:
        self._cfg = cfg
        self._aligned_camera = aligned_camera
        self.captures: dict[str, _CaptureBase] = {}
        self.failed: list[str] = []
        self.rtsp: Optional[RtspServer] = None
        self._depth_stop = threading.Event()
        self._depth_workers: list[threading.Thread] = []

    @property
    def expected_names(self) -> list[str]:
        return self._cfg.camera_names

    def calibration_metadata(self) -> dict:
        return {
            name: {"rotation": cap.cfg.rotation, "serial": getattr(cap, "serial", None)}
            for name, cap in self.captures.items()
        }

    def start(self) -> list[str]:
        if self.captures or self.rtsp is not None:
            return list(self.captures)
        for cam in self._cfg.enabled_cameras:
            try:
                cap = RealSenseCapture(cam, align_depth=True) if cam.name == self._aligned_camera else make_capture(cam)
            except Exception as exc:  # noqa: BLE001 (bad config)
                logger.error("%s: %s", cam.name, exc)
                self.failed.append(cam.name)
                continue
            if cap.start():
                self.captures[cam.name] = cap
            else:
                self.failed.append(cam.name)
        if self._cfg.stream.enabled:
            server = RtspServer(self._cfg)
            if server.start():
                self.rtsp = server
                self._start_depth_workers()
        return list(self.captures)

    def _start_depth_workers(self) -> None:
        self._depth_stop.clear()
        stream = self._cfg.stream
        robot_id = str(stream.depth_robot_id or self._cfg.robot.get("id") or "").strip()
        robot_id = robot_id or self._cfg.name
        destination = (str(stream.depth_host), stream.depth_port)
        for cam in self._cfg.enabled_cameras:
            if not (cam.is_realsense and cam.use_depth and cam.name in self.captures):
                continue
            sampler = DepthSampler(cam.name, robot_id)
            worker = threading.Thread(
                target=run_depth_telemetry,
                args=(sampler, destination),
                kwargs={"stop": self._depth_stop},
                daemon=True,
                name=f"depth-{cam.name}",
            )
            worker.start()
            self._depth_workers.append(worker)
            logger.info(
                "%s: 3 Hz depth distance -> %s:%d (%s)", cam.name, *destination, robot_id
            )

    def stop(self) -> None:
        self._depth_stop.set()
        for worker in self._depth_workers:
            worker.join()
        self._depth_workers.clear()
        if self.rtsp is not None:  # feeders first: they read the buses the captures own
            self.rtsp.stop()
            self.rtsp = None
        for cap in self.captures.values():
            cap.stop()
        self.captures.clear()

    def read(self, name: str) -> Frame:
        cap = self.captures.get(name)
        return (None, None) if cap is None else cap.read_latest()

    def read_depth(self, name: str) -> Frame:
        cap = self.captures.get(name)
        return (None, None) if cap is None else cap.read_latest_depth()

    def read_pair(self, name: str) -> FramePair:
        """(ts, colour, depth) from one snapshot of the camera's holder."""
        cap = self.captures.get(name)
        return (None, None, None) if cap is None else cap.read_latest_pair()

    def names(self) -> list[str]:
        return list(self.captures)

    def shapes(self) -> dict[str, tuple[int, int, int]]:
        """Configured observation shapes per running camera: ``<name>`` → (H, W, 3) and,
        with depth, ``<name>_depth`` → (H, W, 1). This is what goes into the features."""
        out: dict[str, tuple[int, int, int]] = {}
        for cap in self.captures.values():
            out.update(cap.observation_shapes())
        return out

    def wait_ready(self, timeout_s: float) -> list[str]:
        """Block until every camera delivered its frames; return the names that did not."""
        deadline = time.monotonic() + timeout_s
        missing = []
        for name, cap in self.captures.items():
            if not cap.wait_for_frame(max(0.0, deadline - time.monotonic())):
                missing.append(name)
        return missing


# ══════════════════════════════════════════════════════════════════════════════
#  RobotWithCameras — merge frames into a LeRobot robot's observation
# ══════════════════════════════════════════════════════════════════════════════


class RobotWithCameras:
    """A LeRobot robot (built without cameras) plus this module's cameras.

    Exposes the members the LeRobot scripts touch (``observation_features``,
    ``action_features``, ``cameras``, ``name``, ``robot_type``, ``connect`` ...) and
    forwards everything else to the wrapped robot. Observation keys match what a
    LeRobot robot with the same cameras would produce: ``<name>`` RGB uint8 (H, W, 3)
    and ``<name>_depth`` uint16 mm (H, W, 1).
    """

    def __init__(
        self,
        robot,
        cameras: CamerasManager,
        stale_s: float = 0.5,
        ready_timeout_s: float = 10.0,
    ) -> None:
        self._robot = robot
        self._cams = cameras
        self._stale_s = float(stale_s)
        self._ready_timeout_s = float(ready_timeout_s)
        self._warned_stale: set[str] = set()
        self._warned_resize: set[str] = set()

    # -- LeRobot-facing surface ---------------------------------------------

    @property
    def cameras(self) -> dict[str, _CaptureBase]:
        return dict(self._cams.captures)

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._robot.observation_features, **self._cams.shapes()}

    @property
    def action_features(self) -> dict[str, Any]:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def id(self):
        return self._robot.id

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._robot.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        missing = [n for n in self._cams.expected_names if n not in self._cams.captures]
        if missing:
            raise RuntimeError(
                f"cameras failed to open: {missing}. Fix the config or disable them."
            )
        self._robot.connect(calibrate)
        not_ready = self._cams.wait_ready(self._ready_timeout_s)
        if not_ready:
            self._robot.disconnect()
            raise RuntimeError(
                f"cameras delivered no frame within {self._ready_timeout_s:.0f}s: {not_ready}"
            )

    def disconnect(self) -> None:
        self._robot.disconnect()

    def calibrate(self) -> None:
        self._robot.calibrate()

    def configure(self) -> None:
        self._robot.configure()

    def _check_age(self, key: str, ts: Optional[float], now: float) -> None:
        age = now - float(ts or 0.0)
        if age > self._stale_s:
            if key not in self._warned_stale:
                logger.warning("camera %s frame is %.0f ms old", key, age * 1e3)
                self._warned_stale.add(key)
        else:
            self._warned_stale.discard(key)

    def _fit(
        self, key: str, image: np.ndarray, shape: tuple[int, int, int]
    ) -> np.ndarray:
        h, w, _ = shape
        if image.shape[0] == h and image.shape[1] == w:
            return image
        if key not in self._warned_resize:
            logger.warning(
                "camera %s delivers %sx%s but config says %sx%s; resizing",
                key,
                image.shape[1],
                image.shape[0],
                w,
                h,
            )
            self._warned_resize.add(key)
        depth = image.ndim == 3 and image.shape[2] == 1
        interp = cv2.INTER_NEAREST if depth else cv2.INTER_AREA
        out = cv2.resize(image, (w, h), interpolation=interp)
        return out[..., np.newaxis] if depth and out.ndim == 2 else out

    def get_observation(self) -> dict[str, Any]:
        obs = self._robot.get_observation()
        now = time.time()
        for name, cap in self._cams.captures.items():
            shapes = cap.observation_shapes()
            # One snapshot of the holder: colour N and depth N from the same frameset,
            # so a recorded frame never mixes two capture instants.
            ts, bgr, dmap = cap.read_latest_pair()
            if cap.use_rgb:
                if bgr is None:
                    raise RuntimeError(f"camera {name!r} has no frame")
                self._check_age(name, ts, now)
                bgr = self._fit(name, bgr, shapes[name])
                obs[name] = cv2.cvtColor(
                    bgr, cv2.COLOR_BGR2RGB
                )  # LeRobot convention is RGB
            if cap.use_depth:
                key = depth_key(name)
                if dmap is None:
                    raise RuntimeError(f"camera {name!r} has no depth frame")
                self._check_age(key, ts, now)
                obs[key] = self._fit(key, dmap, shapes[key])  # uint16 mm, (H, W, 1)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self._robot.send_action(action)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._robot, item)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
