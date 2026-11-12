"""Local-only annotation dashboard. No additional web framework is required."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

import cv2

from .data import Project, export_episodes

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "web"


class Dashboard:
    def __init__(self, project: Project, output: Path):
        self.project = project
        self.output = output.resolve()
        if self.output == project.path or self.output.is_relative_to(project.path):
            raise ValueError("Output must be outside the source recording directory")
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="episode-export"
        )
        self.job = dict(state="idle", message="Ready", output=None)

    def start_export(self, request: dict) -> dict:
        with self.lock:
            if self.job["state"] == "running":
                raise ValueError("An export or conversion is already running")
            if request.get("revision") != self.project.revision:
                raise ValueError("Save annotations before exporting")
            episodes = self.project.validate(self.project.episodes)
            if not episodes:
                raise ValueError("Mark at least one episode before exporting")
            convert = request.get("convert", False)
            if type(convert) is not bool:
                raise ValueError("convert must be a boolean")
            repo_id = request.get("repo_id", "")
            if convert and (
                not isinstance(repo_id, str)
                or len(repo_id.split("/")) != 2
                or not all(repo_id.split("/"))
            ):
                raise ValueError("Enter a dataset repo ID such as username/pick-cube")
            stamp = (
                datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
            )
            destination = self.output / f"session_{stamp}"
            self.job = dict(
                state="running", message="Preparing episode export", output=None
            )
            self.pool.submit(self._export, episodes, destination, convert, repo_id)
            return dict(self.job)

    def _export(self, episodes, destination, convert, repo_id):
        def progress(message):
            with self.lock:
                self.job["message"] = message

        try:
            output = export_episodes(self.project, episodes, destination, progress)
            with self.lock:
                self.job["output"] = str(output)
            if convert:
                progress(
                    "Converting with the existing LeRobot pipeline (no Hub upload)"
                )
                from revobots_arm.recording.conversion import run_convert

                dataset = run_convert(
                    output,
                    repo_id=repo_id,
                    root=self.output / (destination.name + "_hf"),
                    push_to_hub=False,
                )
                message = f"Dataset created: {dataset}"
            else:
                message = f"Sessions exported: {output}"
            with self.lock:
                self.job.update(state="done", message=message)
        except (Exception, SystemExit) as exc:
            logger.exception("Episode export/conversion failed")
            with self.lock:
                self.job.update(state="error", message=str(exc) or type(exc).__name__)


def create_server(dashboard: Dashboard, port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log the per-session authorization token in frame URLs.

        def send(self, status, data, content_type="application/json"):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, allow_nan=False).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            query = parse_qs(urlsplit(self.path).query)
            token = (
                self.headers.get("X-Annotation-Token") or query.get("token", [""])[0]
            )
            return secrets.compare_digest(token, dashboard.token)

        def do_GET(self):
            url = urlsplit(self.path)
            static = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if url.path in static:
                name, kind = static[url.path]
                return self.send(200, (STATIC / name).read_bytes(), kind)
            if not self.authorized():
                return self.send(
                    403,
                    {"error": "Open the dashboard using the URL printed by the CLI"},
                )
            try:
                if url.path == "/api/project":
                    with dashboard.lock:
                        return self.send(200, dashboard.project.describe())
                if url.path == "/api/job":
                    with dashboard.lock:
                        return self.send(200, dict(dashboard.job))
                if url.path == "/api/frame":
                    q = parse_qs(url.query)
                    rec = dashboard.project.recordings[q["source"][0]]
                    frame = rec.frame(q["camera"][0], int(q["frame"][0]))
                    if frame.shape[1] > 1280:
                        frame = cv2.resize(
                            frame,
                            (
                                1280,
                                max(1, round(frame.shape[0] * 1280 / frame.shape[1])),
                            ),
                        )
                    ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88]
                    )
                    if not ok:
                        raise ValueError("Cannot encode preview frame")
                    return self.send(200, encoded.tobytes(), "image/jpeg")
                return self.send(404, {"error": "Not found"})
            except (ValueError, KeyError, OSError) as exc:
                return self.send(400, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return self.send(403, {"error": "Invalid dashboard token"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")
                route = urlsplit(self.path).path
                if route == "/api/annotations":
                    with dashboard.lock:
                        result = dashboard.project.save(
                            request["episodes"], request["revision"]
                        )
                    return self.send(200, result)
                if route == "/api/export":
                    return self.send(202, dashboard.start_export(request))
                return self.send(404, {"error": "Not found"})
            except (ValueError, KeyError, OSError) as exc:
                return self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(
        description="Mark episodes in a finalized continuous recording and export aligned sessions."
    )
    parser.add_argument(
        "continuous_data_path",
        type=Path,
        help="Session folder, run folder, or data.jsonl path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Export parent directory (default: sibling <source>_annotated)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local dashboard port; 0 selects a free port (default: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the local URL without opening a browser",
    )
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dashboard = None
    try:
        project = Project(args.continuous_data_path)
        output = args.output or project.path.with_name(project.path.name + "_annotated")
        dashboard = Dashboard(project, output)
        server = create_server(dashboard, args.port)
    except (ValueError, KeyError, OSError) as exc:
        if dashboard:
            dashboard.pool.shutdown(wait=False)
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_port}/#token={dashboard.token}"
    print(
        f"\nEpisode annotation dashboard: {url}\nSource: {project.path}\nExports: {output}\nCtrl-C to stop. Running exports finish before exit.\n",
        flush=True,
    )
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        dashboard.pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
