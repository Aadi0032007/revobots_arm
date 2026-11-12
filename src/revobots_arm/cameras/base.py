from __future__ import annotations
import logging
import threading
import time
from typing import Any, Optional
import cv2
import numpy as np
from revobots_arm.config import CameraConfig, depth_key

logger = logging.getLogger(__name__)
_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

Frame = tuple[Optional[float], Optional[np.ndarray]]
# (timestamp, BGR colour, depth) taken under one lock: always the same frameset.
FramePair = tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]


# ══════════════════════════════════════════════════════════════════════════════
#  Shared capture machinery: 1-slot holder + frame bus publishing
# ══════════════════════════════════════════════════════════════════════════════


class _CaptureBase:
    """Common part of both backends: open on a helper thread (so a wedged device
    cannot hang startup), run the reader on a daemon thread, keep the newest colour
    and depth frame in a lock-protected 1-slot holder and mirror them to the frame
    bus. Consumers never block capture."""

    backend = "none"

    def __init__(self, cfg: CameraConfig) -> None:
        cfg.validate()
        self.name = cfg.name
        self.cfg = cfg
        self.frames_read = 0

        self._frame: Optional[np.ndarray] = None  # BGR uint8
        self._depth: Optional[np.ndarray] = None  # uint16 mm (H, W, 1)
        self._ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initial: Any = None  # device handle from start()

        self._publish_enabled = bool(cfg.publish_frames)
        self._publishers: dict[str, Any] = {}  # bus name -> FrameBusPublisher
        self._bus_disabled: set[str] = set()

    # ── what this camera contributes to the observation ──────────────────────

    @property
    def use_rgb(self) -> bool:
        return self.cfg.rgb_enabled

    @property
    def use_depth(self) -> bool:
        return self.cfg.use_depth

    def observation_shapes(self) -> dict[str, tuple[int, int, int]]:
        return self.cfg.observation_shapes()

    @property
    def bus_names(self) -> list[str]:
        return list(self._publishers)

    # ── reading ──────────────────────────────────────────────────────────────

    def read_latest(self) -> Frame:
        """(timestamp, BGR frame) from the 1-slot holder. Non-blocking, not copied."""
        with self._lock:
            if self._frame is None:
                return None, None
            return self._ts, self._frame

    def read_latest_depth(self) -> Frame:
        """(timestamp, depth uint16 (H, W, 1) in mm). Non-blocking, not copied."""
        with self._lock:
            if self._depth is None:
                return None, None
            return self._ts, self._depth

    def read_latest_pair(self) -> FramePair:
        """(timestamp, BGR frame, depth) under ONE lock acquisition, so colour and depth
        come from the same frameset. Either array is None when that stream is off.
        Non-blocking, not copied. Use this, not two separate reads, when both are needed."""
        with self._lock:
            if self._frame is None and self._depth is None:
                return None, None, None
            return self._ts, self._frame, self._depth

    def has_frame(self) -> bool:
        with self._lock:
            return (not self.use_rgb or self._frame is not None) and (
                not self.use_depth or self._depth is not None
            )

    def wait_for_frame(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.has_frame():
                return True
            time.sleep(0.02)
        return self.has_frame()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _open(self) -> Any:
        """Open the device and return a handle. Raise or return None on failure."""
        raise NotImplementedError

    def _run(self) -> None:
        """Reader loop; must poll ``self._stop`` and call ``_store`` per frame."""
        raise NotImplementedError

    def start(self, open_timeout_sec: float | None = None) -> bool:
        """Open the device (on a helper thread) and start the reader. False if unusable."""
        timeout = float(open_timeout_sec or self.cfg.open_timeout_sec)
        result: dict[str, Any] = {}

        def _open() -> None:
            try:
                result["handle"] = self._open()
            except Exception as exc:  # noqa: BLE001
                result["err"] = exc

        opener = threading.Thread(
            target=_open, daemon=True, name=f"cam-open-{self.name}"
        )
        opener.start()
        opener.join(timeout=max(1.0, timeout))
        if opener.is_alive():
            logger.error(
                "%s: open timed out after %.0fs (%r)",
                self.name,
                timeout,
                self.cfg.source,
            )
            return False
        if "err" in result:
            logger.error("%s: open error: %s", self.name, result["err"])
            return False
        handle = result.get("handle")
        if handle is None:
            logger.error("%s: cannot open %r", self.name, self.cfg.source)
            return False

        self._initial = handle
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"cam-{self.name}"
        )
        self._thread.start()
        streams = "+".join(
            s for s, on in (("rgb", self.use_rgb), ("depth", self.use_depth)) if on
        )
        logger.info(
            "%s: started %sx%s@%s %s via %s%s",
            self.name,
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
            streams,
            self.backend,
            " [bus]" if self._publish_enabled else "",
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._frame = None
            self._depth = None
        for bus, pub in self._publishers.items():
            try:
                pub.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: publisher %s close error: %s", self.name, bus, exc)
        self._publishers.clear()

    # ── storing + publishing ─────────────────────────────────────────────────

    def _store(
        self, frame: Optional[np.ndarray], depth: Optional[np.ndarray], now: float
    ) -> None:
        with self._lock:  # in-process consumers (teleop / record)
            if frame is not None:
                self._frame = frame
            if depth is not None:
                self._depth = depth
            self._ts = now
            self.frames_read += 1

        if not self._publish_enabled:  # out-of-process consumers (inference)
            return
        if frame is not None:
            self._publish(self.name, frame, now)
        if depth is not None:
            self._publish(depth_key(self.name), depth, now)

    def _publish(self, bus: str, array: np.ndarray, now: float) -> None:
        if bus in self._bus_disabled:
            return
        pub = self._publishers.get(bus)
        if pub is None:
            if array.ndim != 3:
                logger.warning(
                    "%s: bus %s disabled, unexpected ndim=%s",
                    self.name,
                    bus,
                    array.ndim,
                )
                self._bus_disabled.add(bus)
                return
            try:
                from revobots_arm.transport.frame_bus import FrameBusPublisher

                h, w, c = array.shape
                pub = FrameBusPublisher(
                    bus, height=h, width=w, channels=c, dtype=array.dtype
                )
                self._publishers[bus] = pub
                logger.info(
                    "%s: frame bus → %s (%sx%sx%s %s)",
                    self.name,
                    pub.name,
                    w,
                    h,
                    c,
                    array.dtype,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: frame bus %s init failed: %s", self.name, bus, exc)
                self._bus_disabled.add(bus)
                return
        try:
            pub.publish(np.ascontiguousarray(array), now)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "%s: bus %s publish error: %s (bus disabled)", self.name, bus, exc
            )
            try:
                pub.close()
            except Exception:  # noqa: BLE001
                pass
            self._publishers.pop(bus, None)
            self._bus_disabled.add(bus)


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV / GStreamer capture — sole owner of the device
# ══════════════════════════════════════════════════════════════════════════════
