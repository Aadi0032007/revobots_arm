"""Read the arm's millimetre depth bus at 3 Hz and send scalar UDP telemetry.

The camera manager can start this worker, or it can run as a separate process.
The measurement is axial depth at the centre of the unaligned depth image.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import signal
import socket
import threading
import time
import uuid

import numpy as np

from revobots_arm.transport.frame_bus import FrameBusReader

logger = logging.getLogger(__name__)


def center_depth(depth: np.ndarray, patch_size: int = 11) -> tuple[float | None, float]:
    """Median metres and valid fraction from a uint16 millimetre centre patch."""
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    if depth.dtype != np.uint16:
        raise ValueError("depth bus must contain uint16 millimetres")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or min(depth.shape) < patch_size:
        raise ValueError("depth frame is too small or has an invalid shape")
    cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
    radius = patch_size // 2
    patch = depth[cy-radius:cy+radius+1, cx-radius:cx+radius+1]
    valid = patch[(patch > 0) & (patch < 65535)]
    fraction = float(valid.size / patch.size)
    return (float(np.median(valid)) / 1000.0 if valid.size else None), fraction


class DepthSampler:
    """Latest-only consumer. Reattaches after a stopped/restarted publisher."""

    def __init__(self, camera: str, robot_id: str, *, patch_size: int = 11,
                 stale_sec: float = 1.5, reader_factory=FrameBusReader) -> None:
        if not camera.strip() or not robot_id.strip():
            raise ValueError("camera and robot_id must be nonempty")
        if patch_size < 1 or patch_size % 2 != 1:
            raise ValueError("patch_size must be a positive odd integer")
        if not math.isfinite(stale_sec) or stale_sec <= 0:
            raise ValueError("stale_sec must be finite and positive")
        self.camera, self.robot_id = camera, robot_id
        self.patch_size, self.stale_sec = patch_size, stale_sec
        self._reader_factory = reader_factory
        self._reader = None
        self._frame_ts = None
        self._frame_idx = None
        self._depth_m = None
        self._fraction = 0.0
        self._status = "missing"
        self._last_fresh = None
        self._session = uuid.uuid4().hex
        self._seq = 0

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def sample(self, *, now: float | None = None,
               monotonic: float | None = None) -> dict:
        now = time.time() if now is None else now
        monotonic = time.monotonic() if monotonic is None else monotonic
        try:
            if self._reader is None:
                self._reader = self._reader_factory(f"{self.camera}_depth")
            ts, frame = self._reader.read_new()
            if ts is not None and frame is not None:
                ts = float(ts)
                if not math.isfinite(ts) or ts <= 0 or ts > now + 0.5:
                    raise ValueError("invalid depth frame timestamp")
                value, fraction = center_depth(frame, self.patch_size)
                self._frame_ts = ts
                self._frame_idx = self._reader.last_frame_idx
                self._last_fresh = monotonic
                self._depth_m, self._fraction = value, fraction
                self._status = "ok" if value is not None else "invalid"
            elif self._last_fresh is None or monotonic - self._last_fresh > self.stale_sec:
                # A reader of an unlinked bus can retain its old mapping forever.
                # Drop it so the next tick can attach to the replacement bus.
                self.close()
        except (RuntimeError, ValueError, OSError) as exc:
            if self._status != "error":
                logger.warning("depth bus %s: %s", self.camera, exc)
            self._status, self._depth_m, self._fraction = "error", None, 0.0
            self.close()

        age = None if self._frame_ts is None else max(0.0, now - self._frame_ts)
        stale = (age is not None and age > self.stale_sec) or (
            self._last_fresh is not None and monotonic - self._last_fresh > self.stale_sec)
        status = "stale" if stale and self._status != "error" else self._status
        valid = status == "ok"
        self._seq += 1
        return {
            "type": "arm_depth", "version": 1, "robot_id": self.robot_id,
            "camera": self.camera, "session": self._session, "seq": self._seq,
            "sample_ts": now, "frame_ts": self._frame_ts,
            "frame_idx": self._frame_idx, "reference": "depth_center",
            "depth_m": self._depth_m if valid else None,
            "valid": valid, "valid_ratio": self._fraction,
            "status": status,
        }


def run(sampler: DepthSampler, destination: tuple[str, int], *, hz: float = 3.0,
        stop: threading.Event | None = None) -> None:
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be finite and positive")
    stop = stop if stop is not None else threading.Event()
    period = 1.0 / hz
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.setblocking(False)
            failed = False
            while not stop.is_set():
                started = time.monotonic()
                packet = sampler.sample()
                try:
                    udp.sendto(json.dumps(packet, allow_nan=False).encode(), destination)
                    failed = False
                except OSError as exc:
                    if not failed:
                        logger.warning("depth telemetry send failed: %s", exc)
                    failed = True
                # No catch-up burst and no queue of past depth frames.
                stop.wait(max(0.0, period - (time.monotonic() - started)))
    finally:
        sampler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="arm camera name, without lab_ or _depth")
    parser.add_argument("--robot-id", required=True, help="base fleet identity receiving this depth")
    parser.add_argument("--host", default="127.0.0.1", help="receiver's numeric IPv4 address")
    parser.add_argument("--port", type=int, default=57110)
    parser.add_argument("--hz", type=float, default=3.0)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--stale-sec", type=float, default=1.5)
    args = parser.parse_args()
    try:
        ipaddress.IPv4Address(args.host)
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be in 1..65535")
        if not math.isfinite(args.hz) or args.hz <= 0:
            raise ValueError("hz must be finite and positive")
        sampler = DepthSampler(args.camera, args.robot_id, patch_size=args.patch_size,
                               stale_sec=args.stale_sec)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    logger.info("lab_%s_depth -> %s:%d at %g Hz (depth-image centre)",
                args.camera, args.host, args.port, args.hz)
    try:
        run(sampler, (args.host, args.port), hz=args.hz, stop=stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
