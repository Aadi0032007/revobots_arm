from __future__ import annotations
import logging
import time
from typing import Optional
import cv2

logger = logging.getLogger(__name__)
import platform
from .base import _CaptureBase, _ROTATIONS


class UsbCameraCapture(_CaptureBase):
    """Background reader for one OpenCV-accessible camera (USB/UVC)."""

    def _gst_pipeline(self) -> str:
        cfg = self.cfg
        fmt = cfg.pixel_format.upper()
        # io-mode=2 = GST_V4L2_IO_MMAP, the explicit, reliable choice for UVC devices.
        if fmt == "YUYV":
            return (
                f"v4l2src device={cfg.source} io-mode=2 ! "
                f"video/x-raw,format=YUY2,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
        return (
            f"v4l2src device={cfg.source} io-mode=2 ! "
            f"image/jpeg,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1 ! "
            "nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def _open(self) -> Optional[cv2.VideoCapture]:
        cfg = self.cfg
        if cfg.hw_decode:
            pipeline = self._gst_pipeline()
            logger.info("%s: gst → %s", self.name, pipeline)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self.backend = "gstreamer"
                return cap
            cap.release()
            logger.warning(
                "%s: hardware-decode pipeline failed to open, falling back to OpenCV",
                self.name,
            )

        api = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        cap = cv2.VideoCapture(cfg.source, api)
        if not cap.isOpened():
            cap.release()
            return None
        if cfg.pixel_format:
            try:
                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*cfg.pixel_format.upper()[:4]),
                )
            except Exception:  # noqa: BLE001
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        self.backend = "opencv"
        return cap

    def _run(self) -> None:
        backoff = 1.0
        rotate = _ROTATIONS.get(int(self.cfg.rotation or 0))
        while not self._stop.is_set():
            cap = self._initial
            self._initial = None
            if cap is None:
                try:
                    cap = self._open()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s: reopen failed: %s", self.name, exc)
                    cap = None
            if cap is None:
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            backoff = 1.0

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if rotate is not None:
                    frame = cv2.rotate(frame, rotate)
                self._store(frame, None, time.time())

            cap.release()
            if not self._stop.is_set():
                logger.warning(
                    "%s: stream lost, reconnecting in %.1fs", self.name, backoff
                )
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 10.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Intel RealSense capture — pyrealsense2 directly (AGV LAB style), colour + depth
# ══════════════════════════════════════════════════════════════════════════════

# LeRobot's depth contract: uint16 where one unit is 1 mm (z16 at depth_units = 0.001 m).
