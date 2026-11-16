"""Local YOLO boxes, instance outlines, and optional RGB tracking."""

from __future__ import annotations

import numpy as np

from .detection import Detection, clip_box, validate_frame


class YoloDetector:
    """Use one instance per camera when a tracker is configured."""

    def __init__(
        self,
        model: str,
        *,
        confidence: float = 0.25,
        device: str | None = None,
        tracker: str | None = None,
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("YOLO confidence must be in [0, 1]")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO needs the revobots-arm[vision-yolo] extra") from exc
        self._model = YOLO(model)
        if self._model.task not in {"detect", "segment"}:
            raise ValueError("Use a YOLO detection or instance segmentation checkpoint")
        self._options = {"conf": confidence, "device": device, "verbose": False}
        self._tracker = tracker

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        validate_frame(frame_bgr)
        if self._tracker is None:
            result = self._model.predict(frame_bgr, **self._options)[0]
        else:
            result = self._model.track(
                frame_bgr, persist=True, tracker=self._tracker, **self._options
            )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        height, width = frame_bgr.shape[:2]
        ids = [None] * len(boxes) if boxes.id is None else boxes.id.tolist()
        polygons = [None] * len(boxes) if result.masks is None else result.masks.xy
        detections = []
        for xyxy, class_id, score, track_id, polygon in zip(
            boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), ids, polygons, strict=True
        ):
            box = clip_box(xyxy, width, height)
            if box is None:
                continue
            outline = None
            if polygon is not None and len(polygon) >= 3:
                outline = tuple(map(tuple, np.clip(polygon, 0, (width, height)).tolist()))
            detections.append(
                Detection(
                    label=result.names[int(class_id)],
                    xyxy=box,
                    confidence=float(score),
                    polygon_xy=outline,
                    track_id=None if track_id is None else int(track_id),
                )
            )
        return detections
