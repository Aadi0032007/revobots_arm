"""Associate a clicked frame with settled joint feedback and manage confirmed moves."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from numbers import Real
import secrets
from threading import Event, Lock
import time

import numpy as np

from revobots_arm.cameras.realsense import RealSenseCapture
from .motion import MotionStopped, NeroPickMotion, Plan, surface_depth_m


logger = logging.getLogger(__name__)
FRAME_WINDOW_S = 0.6
TARGET_WINDOW_S = 120
PLAN_WINDOW_S = 20
BUSY = {"planning", "moving", "stopping"}


@dataclass(frozen=True)
class CapturedTargetFrame:
    key: str
    timestamp: float
    depth: np.ndarray
    joints: np.ndarray


class PickSession:
    def __init__(self, camera: RealSenseCapture, motion: NeroPickMotion, *, execute: bool) -> None:
        self.camera = camera
        self.motion = motion
        self.live = execute
        self.token = secrets.token_urlsafe(24)
        self._lock = Lock()
        self._stop = Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nero-pick")
        self._frames: deque[CapturedTargetFrame] = deque(maxlen=8)
        self._feedback: tuple[float, np.ndarray] | None = None
        self._target: np.ndarray | None = None
        self._target_at = self._plan_at = 0.0
        self._plan: Plan | None = None
        self._plan_id = ""
        self._state = "idle"
        self._message = "Mark a point on the wrist camera image."
        self._closed = False

    def _set_status(self, state: str, message: str) -> None:
        self._state, self._message = state, message

    def status(self) -> dict:
        with self._lock:
            now = time.monotonic()
            if self._target is not None and self._state not in BUSY and now - self._target_at >= TARGET_WINDOW_S:
                self._target = self._plan = None
                self._set_status("idle", "Target expired. Select a point on the live image.")
            if self._plan and now - self._plan_at >= PLAN_WINDOW_S:
                self._plan = None
                self._set_status("target", "Plan expired. Preview again to continue.")
            plan = self._plan
            return {
                "state": self._state, "message": self._message,
                "live": self.live, "can_grasp": self.motion.can_grasp,
                "target_mm": self._mm(self._target),
                "plan_id": self._plan_id if plan else "",
                "hover_mm": self._mm(plan.hover) if plan else None,
                "grasp_mm": self._mm(plan.grasp) if plan else None,
                "steps": [step.name for step in plan.steps] if plan else [],
            }

    @staticmethod
    def _mm(point: np.ndarray | None) -> list[float] | None:
        return np.round(point * 1000, 1).tolist() if point is not None else None

    def _require_idle(self) -> None:
        if self._closed or self._state in BUSY:
            raise ValueError("Wait for the current task or press Stop")

    def _fresh_pair(self) -> tuple[float, np.ndarray, np.ndarray]:
        stamp, colour, depth = self.camera.read_aligned_pair()
        if stamp is None or colour is None or depth is None or not 0 <= time.time() - stamp <= FRAME_WINDOW_S:
            raise ValueError("Waiting for a fresh RealSense colour/depth pair")
        size = tuple(self.motion.calibration.image_size_wh)
        if colour.shape[:2][::-1] != size or depth.shape[:2][::-1] != size:
            raise ValueError("Camera image size does not match hand-eye calibration")
        return stamp, colour, depth

    def frame(self) -> tuple[np.ndarray, str]:
        with self._lock:
            if self._closed:
                raise ValueError("Console is shutting down")
            stamp, colour, depth = self._fresh_pair()
            if self._state in BUSY:
                return colour, ""
            q = self.motion.joints()
            now = time.time()
            previous = self._feedback
            self._feedback = now, q.copy()
            if previous is not None and np.max(np.abs(q - previous[1])) > np.radians(0.15):
                self._frames.clear()
            if self._frames and self._frames[-1].timestamp == stamp:
                return colour, self._frames[-1].key
            if (previous is not None and 0 <= stamp - previous[0] <= FRAME_WINDOW_S
                    and np.max(np.abs(q - previous[1])) <= np.radians(0.15)):
                snapshot = CapturedTargetFrame(secrets.token_hex(10), stamp, depth, q.copy())
                self._frames.append(snapshot)
                return colour, snapshot.key
            return colour, ""

    def preview_frame(self) -> np.ndarray:
        return self._fresh_pair()[1]

    def select(self, key: str, x: int, y: int) -> None:
        if type(x) is not int or type(y) is not int:
            raise ValueError("Expected an integer image pixel")
        with self._lock:
            self._require_idle()
            self._target = self._plan = None
            self._set_status("idle", "Mark a point on the wrist camera image.")
            snapshot = next((frame for frame in self._frames if frame.key == key), None)
            if snapshot is None or not 0 <= time.time() - snapshot.timestamp <= FRAME_WINDOW_S:
                raise ValueError("Frame is stale or the arm is settling. Click the live image again.")
            if np.max(np.abs(self.motion.joints() - snapshot.joints)) > np.radians(0.15):
                raise ValueError("Arm moved since the image was captured. Select the target again.")
            ray = self.motion.calibration.pixel_ray((x, y))
            depth_m = self.camera.colour_depth_m(surface_depth_m(snapshot.depth, (x, y)), ray)
            if not np.isfinite(depth_m) or depth_m <= 0:
                raise ValueError("The marked pixel has no valid colour-frame depth")
            point = self.motion.calibration.camera_to_base(depth_m * ray, self.motion.wrist.fk(snapshot.joints))
            self._target, self._target_at, self._plan = point, time.monotonic(), None
            self._set_status("target", "Target marked. Preview Hover or Grasp.")

    def clear(self) -> None:
        with self._lock:
            self._require_idle()
            self._target = self._plan = None
            self._set_status("idle", "Mark a point on the wrist camera image.")

    def prepare(self, kind: str, hover_mm: float, depth_mm: float, yaw_deg: float) -> None:
        if kind not in ("hover", "grasp"):
            raise ValueError("Choose Hover or Grasp")
        if not all(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)
                   for value in (hover_mm, depth_mm, yaw_deg)):
            raise ValueError("Enter finite numeric motion settings")
        with self._lock:
            self._require_idle()
            if self._target is None or time.monotonic() - self._target_at >= TARGET_WINDOW_S:
                raise ValueError("Mark a fresh target in the camera image")
            self._plan = None
            self._stop.clear()
            self._set_status("planning", f"Checking {kind} path…")
            self._pool.submit(self._prepare, kind, self._target.copy(), hover_mm / 1000, depth_mm / 1000, yaw_deg)

    def _prepare(self, kind: str, target: np.ndarray, hover_m: float, depth_m: float, yaw_deg: float) -> None:
        try:
            plan = self.motion.plan(kind, target, self.motion.joints(), hover_m, depth_m, yaw_deg, self._stop)
            with self._lock:
                if self._stop.is_set():
                    self._set_status("stopped", "Stopped. Select a fresh target.")
                    return
                self._plan, self._plan_at = plan, time.monotonic()
                self._plan_id = secrets.token_urlsafe(12)
                msg = f"{kind.capitalize()} path checked. Confirm within {PLAN_WINDOW_S} seconds." if self.live else "Path checked in preview mode."
                self._set_status("planned", msg)
        except Exception as exc:
            self._fail(exc)

    def confirm(self, plan_id: str) -> None:
        with self._lock:
            self._require_idle()
            if not self.live:
                raise ValueError("Start the console with --execute to enable motion")
            if (self._plan is None or plan_id != self._plan_id
                    or time.monotonic() - self._plan_at >= PLAN_WINDOW_S
                    or time.monotonic() - self._target_at >= TARGET_WINDOW_S):
                raise ValueError("Preview the path again before confirming")
            self._fresh_pair()
            plan, self._plan = self._plan, None
            self._frames.clear()
            self._feedback = None
            self._set_status("moving", "Checking starting pose…")
            self._pool.submit(self._execute, plan)

    def _execute(self, plan: Plan) -> None:
        def progress(message: str) -> None:
            with self._lock:
                self._message = message if self._state == "moving" else self._message

        try:
            self.motion.execute(plan, self._stop, progress)
            with self._lock:
                if self._stop.is_set():
                    self._set_status("stopped", "Stopped. Select a fresh target.")
                else:
                    # After a grasp the object may have moved. A fresh selection is required.
                    if plan.kind == "grasp":
                        self._target = None
                    self._set_status("done", f"{plan.kind.capitalize()} sequence complete.")
        except Exception as exc:
            self._fail(exc, clear_target=True)

    def _fail(self, exc: Exception, *, clear_target: bool = False) -> None:
        if not isinstance(exc, (ValueError, MotionStopped)):
            logger.exception("Hover or grasp failed")
        with self._lock:
            if clear_target:
                self._target = None
            self._set_status("stopped" if isinstance(exc, MotionStopped) else "error", str(exc))

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            self._target = self._plan = None
            self._frames.clear()
            self._feedback = None
            state = "stopping" if self._state in BUSY else "stopped"
            self._set_status(state, "Stopping…" if state == "stopping" else "Stopped. Select a fresh target.")

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.stop()
        self._pool.shutdown(wait=True)
