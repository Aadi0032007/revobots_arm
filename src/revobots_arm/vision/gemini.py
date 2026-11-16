"""Prompted object localization using the Gemini API."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .detection import Detection, clip_box, validate_frame


_BOX_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "box_2d": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["label", "box_2d"],
    },
}


class GeminiDetector:
    def __init__(self, model: str, prompt: str, *, api_key: str | None = None) -> None:
        if not prompt.strip():
            raise ValueError("Gemini needs a prompt describing the objects to find")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Gemini needs the revobots-arm[vision-gemini] extra") from exc
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini")
        self._client = genai.Client(api_key=key)
        self._sdk = genai
        self._model = model
        self._prompt = (
            "Find objects matching the request in this image. Return one label and one "
            "box_2d per object. Use [ymin, xmin, ymax, xmax] on a 0 to 1000 grid. "
            "Return an empty array if no object matches.\n"
            f"Request: {prompt}"
        )

    def close(self) -> None:
        self._client.close()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        validate_frame(frame_bgr)
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("Could not encode the frame for Gemini")
        image = self._sdk.types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg")
        config = self._sdk.types.GenerateContentConfig(
            response_mime_type="application/json", response_json_schema=_BOX_SCHEMA
        )
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=[image, self._prompt], config=config
            )
        except self._sdk.errors.APIError as exc:
            raise RuntimeError(f"Gemini request failed (status {exc.code})") from exc

        height, width = frame_bgr.shape[:2]
        detections = []
        try:
            items = json.loads(response.text or "")
            if not isinstance(items, list):
                raise ValueError("Expected a JSON array")
            for item in items:
                label = item["label"]
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("Expected a nonempty object label")
                y1, x1, y2, x2 = map(float, item["box_2d"])
                pixel_edges = (
                    x1 * width / 1000, y1 * height / 1000,
                    x2 * width / 1000, y2 * height / 1000,
                )
                box = clip_box(pixel_edges, width, height)
                if box is not None:
                    detections.append(Detection(label=label, xyxy=box))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Gemini did not return a valid list of boxes") from exc
        return detections
