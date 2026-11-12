"""Indexed recordings, validated annotations, and atomic session exports."""

from __future__ import annotations

from array import array
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import cv2
import numpy as np

from revobots_arm.recording.session import (
    DATA_FILENAME,
    META_FILENAME,
    find_sessions,
    load_meta,
)


class Recording:
    """Index JSONL offsets rather than loading an entire continuous run into memory."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.meta = load_meta(self.path)
        self.fps = float(self.meta["fps"])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Invalid FPS in {self.path}")
        self.offsets = array("Q")
        data = self.path / DATA_FILENAME
        with data.open("rb") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(position)
        self.count = len(self.offsets)
        if not self.count or self.meta.get("num_frames") != self.count:
            raise ValueError(
                f"{self.path}: recording is empty, incomplete, or has inconsistent frame counts"
            )
        self.cameras = self.meta.get("cameras", {})
        signature = [self.meta]
        for info in self.cameras.values():
            media = self.media_path(info.get("video") or info.get("depth", ""))
            stat = media.stat()
            signature.append((media.name, stat.st_size, stat.st_mtime_ns))
            h, w = int(info["height"]), int(info["width"])
            if h <= 0 or w <= 0:
                raise ValueError("Camera dimensions must be positive")
            if "depth" in info:
                if stat.st_size != self.count * h * w * 2:
                    raise ValueError(f"Depth frame count does not match JSONL: {media}")
            else:
                cap = cv2.VideoCapture(str(media))
                try:
                    if (
                        not cap.isOpened()
                        or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != self.count
                    ):
                        raise ValueError(
                            f"Video frame count does not match JSONL: {media}"
                        )
                finally:
                    cap.release()
        stat = data.stat()
        signature.append((stat.st_size, stat.st_mtime_ns))
        self.signature = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()

    def media_path(self, filename: str) -> Path:
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError("Recording media filenames must be local basenames")
        path = (self.path / filename).resolve()
        if not filename or not path.is_relative_to(self.path) or not path.is_file():
            raise ValueError(f"Invalid or missing recording media: {filename}")
        return path

    def row(self, index: int) -> dict:
        if not 0 <= index < self.count:
            raise ValueError("Frame index is outside the recording")
        with (self.path / DATA_FILENAME).open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())

    def rows(self, start: int, end: int):
        with (self.path / DATA_FILENAME).open("rb") as stream:
            for index in range(start, end):
                stream.seek(self.offsets[index])
                yield json.loads(stream.readline())

    def frame(self, camera: str, index: int) -> np.ndarray:
        if camera not in self.cameras or not 0 <= index < self.count:
            raise ValueError("Unknown camera or invalid frame index")
        info = self.cameras[camera]
        if "depth" in info:
            size = int(info["height"]) * int(info["width"])
            with self.media_path(info["depth"]).open("rb") as stream:
                stream.seek(index * size * 2)
                frame = np.frombuffer(stream.read(size * 2), dtype=np.uint16).reshape(
                    info["height"], info["width"]
                )
            # Visualization only; the original uint16 maps are copied unchanged during export.
            return cv2.applyColorMap(
                np.uint8(np.clip(frame.astype(float) / 4000 * 255, 0, 255)),
                cv2.COLORMAP_TURBO,
            )
        cap = cv2.VideoCapture(str(self.media_path(info["video"])))
        try:
            seek_frame(cap, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode {camera} frame {index}")
            return frame
        finally:
            cap.release()


def seek_frame(cap, index: int) -> None:
    """Use the decoder's frame seek; fall back to sequential decoding if unavailable."""
    if (
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == index
    ):
        return
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise ValueError("Video decoder cannot seek to a known frame")
    for _ in range(index):
        if not cap.grab():
            raise ValueError("Video ended while seeking")


class Project:
    def __init__(self, path: Path):
        path = path.expanduser().resolve()
        if path.is_file() and path.name in (DATA_FILENAME, META_FILENAME):
            path = path.parent
        self.path = path
        paths = find_sessions(path)
        if not paths:
            raise ValueError(
                "Expected a finalized continuous session or a run containing sessions (meta.json + data.jsonl)"
            )
        self.recordings = {str(i): Recording(p) for i, p in enumerate(paths)}
        self.annotation_path = path / "annotations.json"
        self.revision = 0
        self.episodes: list[dict] = []
        if self.annotation_path.exists():
            saved = json.loads(self.annotation_path.read_text(encoding="utf-8"))
            signatures = {key: rec.signature for key, rec in self.recordings.items()}
            if saved.get("version") != 1 or saved.get("sources") != signatures:
                raise ValueError(
                    "Saved annotations belong to a different or modified source recording"
                )
            self.episodes = self.validate(saved["episodes"])
            self.revision = int(saved.get("revision", 0))

    def validate(self, episodes: list) -> list[dict]:
        if not isinstance(episodes, list):
            raise ValueError("Episodes must be a list")
        result = []
        ids = set()
        for episode in episodes:
            if not isinstance(episode, dict):
                raise ValueError("Each episode must be an object")
            source = episode.get("source")
            start, end = episode.get("start"), episode.get("end")
            if (
                not isinstance(source, str)
                or source not in self.recordings
                or type(start) is not int
                or type(end) is not int
            ):
                raise ValueError(
                    "Each episode needs a valid source and integer frame boundaries"
                )
            if not 0 <= start < end <= self.recordings[source].count:
                raise ValueError(
                    "Episode boundaries must satisfy 0 <= start < end <= frame count"
                )
            identifier = episode.get("id")
            task = episode.get("task", "")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("Episode IDs must be unique nonempty strings")
            if not isinstance(task, str) or len(task) > 4000:
                raise ValueError("Task must be text of at most 4000 characters")
            ids.add(identifier)
            result.append(
                dict(id=identifier, source=source, start=start, end=end, task=task)
            )
        for source in self.recordings:
            ordered = sorted(
                (e for e in result if e["source"] == source), key=lambda e: e["start"]
            )
            if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("Episodes cannot overlap within the same recording")
        return result

    def save(self, episodes: list, revision: int) -> dict:
        if revision != self.revision:
            raise ValueError(
                "Annotations changed in another tab; reload before editing"
            )
        clean = self.validate(episodes)
        payload = dict(
            version=1,
            revision=self.revision + 1,
            sources={k: r.signature for k, r in self.recordings.items()},
            episodes=clean,
        )
        atomic_json(self.annotation_path, payload)
        self.episodes, self.revision = clean, payload["revision"]
        return payload

    def describe(self) -> dict:
        return dict(
            path=str(self.path),
            revision=self.revision,
            episodes=self.episodes,
            sources=[
                dict(
                    id=k,
                    name=r.path.name,
                    frames=r.count,
                    fps=r.fps,
                    cameras=list(r.cameras),
                    duration=r.count / r.fps,
                    task=r.meta.get("task", ""),
                )
                for k, r in self.recordings.items()
            ],
        )


def atomic_json(path: Path, payload: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def export_episodes(
    project: Project,
    episodes: list,
    output: Path,
    progress: Callable[[str], None] = lambda _: None,
) -> Path:
    """Write the established session format, preserving all channels and source timestamps."""
    episodes = project.validate(episodes)
    if not episodes:
        raise ValueError("Mark at least one episode before exporting")
    output = output.resolve()
    if output.exists() or output == project.path or output.is_relative_to(project.path):
        raise ValueError("Export must be a new directory outside the source recording")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".episode-export-", dir=output.parent))
    try:
        for source in {e["source"] for e in episodes}:
            rec = project.recordings[source]
            if Recording(rec.path).signature != rec.signature:
                raise ValueError("Source recording changed; reload before exporting")
        for number, episode in enumerate(episodes):
            rec = project.recordings[episode["source"]]
            start, end = episode["start"], episode["end"]
            session = staging / f"session_{number}"
            session.mkdir()
            progress(
                f"Exporting episode {number + 1}/{len(episodes)}: frames {start}–{end - 1}"
            )
            first_time = float(rec.row(start)["timestamp"])
            with (session / DATA_FILENAME).open("w", encoding="utf-8") as dest:
                for index, row in enumerate(rec.rows(start, end), start=start):
                    row.update(
                        source_frame_index=index,
                        source_timestamp=row["timestamp"],
                        frame_index=index - start,
                        timestamp=round(float(row["timestamp"]) - first_time, 6),
                        task=episode["task"] or row.get("task"),
                    )
                    dest.write(json.dumps(row, allow_nan=False) + "\n")
            for camera, info in rec.cameras.items():
                if "depth" in info:
                    frame_bytes = int(info["height"]) * int(info["width"]) * 2
                    with (
                        rec.media_path(info["depth"]).open("rb") as src,
                        (session / info["depth"]).open("wb") as dest,
                    ):
                        src.seek(start * frame_bytes)
                        remaining = (end - start) * frame_bytes
                        while remaining:
                            chunk = src.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                raise ValueError("Truncated depth recording")
                            dest.write(chunk)
                            remaining -= len(chunk)
                else:
                    cap = cv2.VideoCapture(str(rec.media_path(info["video"])))
                    writer = cv2.VideoWriter(
                        str(session / info["video"]),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        rec.fps,
                        (int(info["width"]), int(info["height"])),
                    )
                    try:
                        if not writer.isOpened():
                            raise ValueError(
                                f"Cannot create episode video for {camera}"
                            )
                        seek_frame(cap, start)
                        for _ in range(start, end):
                            ok, frame = cap.read()
                            if not ok:
                                raise ValueError(f"Truncated video: {camera}")
                            writer.write(frame)
                    finally:
                        cap.release()
                        writer.release()
            meta = copy.deepcopy(rec.meta)
            meta.update(
                session_name=session.name,
                num_frames=end - start,
                fps=rec.fps,
                codec="mp4v",
                duration_s=(end - start) / rec.fps,
                continuous=False,
                task=episode["task"] or meta.get("task"),
                annotation=dict(
                    source=str(rec.path),
                    signature=rec.signature,
                    start=start,
                    end=end,
                    id=episode["id"],
                ),
            )
            atomic_json(session / META_FILENAME, meta)
            Recording(session)  # Fail atomically if any channel is short or misaligned.
        atomic_json(
            staging / "export.json",
            dict(version=1, source=str(project.path), episodes=episodes),
        )
        staging.rename(output)
        return output
    except BaseException:
        shutil.rmtree(staging)
        raise
