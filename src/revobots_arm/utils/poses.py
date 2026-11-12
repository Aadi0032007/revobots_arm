"""Load and validate saved joint poses in degrees, independently of any robot."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from collections.abc import Sequence
import numpy as np


def load_joint_pose(
    path: str | None,
    fallback: Sequence[float] | None,
    joint_count: int,
    name: str,
    logger: logging.Logger,
    keys: Sequence[str] = (),
) -> np.ndarray | None:
    values = None
    if path and path.strip():
        source = Path(path).expanduser()
        if source.is_file():
            data = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{name} pose file must contain a JSON object")
            for key in keys or ("joints_deg", f"{name}_deg"):
                if data.get(key) is not None:
                    values = data[key]
                    break
            logger.info("%s pose from %s", name, source)
        else:
            logger.warning("%s pose file missing: %s", name, source)
    if values is None:
        values = fallback
    if values is None:
        return None
    try:
        degrees = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} pose must contain {joint_count} finite joint degrees"
        ) from exc
    if degrees.shape != (joint_count,) or not np.all(np.isfinite(degrees)):
        raise ValueError(f"{name} pose must contain {joint_count} finite joint degrees")
    return degrees
