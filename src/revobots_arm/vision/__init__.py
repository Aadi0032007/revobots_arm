"""Camera-independent image processing for Revobots Arm."""

from .detection import Detection, Detector, draw_detections

__all__ = ["Detection", "Detector", "draw_detections"]
