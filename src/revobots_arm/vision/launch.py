"""Launch the same standalone vision CLI alongside teleoperation."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys


logger = logging.getLogger(__name__)


class VisionProcess:
    def __init__(self, args: list[str], output: Path) -> None:
        self._command = [sys.executable, "-m", "revobots_arm.vision.cli", *args]
        self.output = output.expanduser()
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        try:
            with self.output.open("wb") as stdout:
                self._process = subprocess.Popen(
                    self._command, stdin=subprocess.DEVNULL, stdout=stdout
                )
        except OSError as exc:
            logger.error("Could not launch vision: %s", exc)
            return
        logger.info("Vision started. Detection output: %s", self.output)

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        status = process.poll()
        if status is not None:
            if status:
                logger.warning("Vision exited with status %d", status)
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
