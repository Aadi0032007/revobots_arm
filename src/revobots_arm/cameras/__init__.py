"""Camera backends and observation manager (SDKs load when selected)."""

from .manager import CamerasManager, RobotWithCameras, make_capture
from .opencv import UsbCameraCapture
from .realsense import RealSenseCapture

__all__ = [
    "CamerasManager",
    "RobotWithCameras",
    "make_capture",
    "UsbCameraCapture",
    "RealSenseCapture",
]
