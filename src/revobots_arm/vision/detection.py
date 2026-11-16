"""Shared pixel coordinates and rendering for vision backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A box in original-image pixels as (left, top, right, bottom) edges."""

    label: str
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None
    polygon_xy: tuple[tuple[float, float], ...] | None = None
    track_id: int | None = None

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Process a uint8 BGR colour frame."""
        ...


def validate_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
        raise ValueError("Expected a nonempty uint8 BGR image with shape (height, width, 3)")


def clip_box(
    xyxy: Iterable[float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    box = np.asarray(tuple(xyxy), dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Expected four finite box coordinates")
    x1, y1, x2, y2 = np.clip(box, 0, (width, height, width, height)).tolist()
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def draw_detections(frame_bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Draw boxes and available outlines on a copy of the frame."""
    validate_frame(frame_bgr)
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    color = (0, 220, 0)
    for item in detections:
        box = clip_box(item.xyxy, width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        if item.polygon_xy:
            polygon = np.rint(item.polygon_xy).astype(np.int32)
            cv2.polylines(output, [polygon], True, color, 2)
        cv2.rectangle(output, (x1, y1), (min(x2, width - 1), min(y2, height - 1)), color, 2)
        label = item.label
        if item.track_id is not None:
            label = f"{label} #{item.track_id}"
        if item.confidence is not None:
            label = f"{label} {item.confidence:.2f}"
        cv2.putText(output, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output
