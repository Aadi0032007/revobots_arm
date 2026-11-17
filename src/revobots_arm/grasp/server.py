"""Run the local Nero hover and grasp console."""

from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
from urllib.parse import urlsplit
import webbrowser

import cv2

from revobots_arm.cameras.realsense import RealSenseCapture
from revobots_arm.config.devices import load_config, make_parser
from revobots_arm.robots.registry import make_robot
from .session import PickSession
from .setup import GraspSetup, check_camera_serial


STATIC = Path(__file__).with_name("web")


def pick_action(session: PickSession, action: str, request: dict) -> dict:
    if action == "target":
        session.select(request["frame"], request["x"], request["y"])
    elif action == "clear":
        session.clear()
    elif action == "plan":
        session.prepare(request["kind"], request["hover_mm"], request["depth_mm"], request["yaw_deg"])
    elif action == "confirm":
        session.confirm(request["plan_id"])
    elif action == "stop":
        session.stop()
    else:
        raise LookupError(action)
    return session.status()


def create_server(session, port: int = 0, *, console_assets: Path | None = None) -> ThreadingHTTPServer:
    console = console_assets is not None
    assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript"),
        "/style.css": ("style.css", "text/css"),
    }
    class Handler(BaseHTTPRequestHandler):
        timeout = 4

        def log_message(self, *_):
            pass

        def respond(self, status, data, content_type="application/json", frame_key=None):
            if isinstance(data, dict):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            ancestors = "'self'" if console else "'none'"
            self.send_header("Content-Security-Policy", f"default-src 'self'; img-src 'self' blob:; frame-ancestors {ancestors}")
            if frame_key is not None:
                self.send_header("X-Frame-Key", frame_key)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorize(self):
            expected_host = f"127.0.0.1:{self.server.server_port}"
            token = self.headers.get("X-Grasp-Token", "")
            if (self.headers.get("Host") == expected_host and token.isascii()
                    and secrets.compare_digest(token, session.token)):
                return True
            self.respond(403, {"error": "Open the local console URL printed by the CLI"})
            return False

        def do_GET(self):
            url = urlsplit(self.path)
            grab = not console or url.path.startswith("/grasp/")
            path = url.path.removeprefix("/grasp") if console and grab else url.path
            if path in assets:
                filename, mime = assets[path]
                directory = STATIC if grab else console_assets
                return self.respond(200, (directory / filename).read_bytes(), mime)
            if not self.authorize():
                return
            try:
                if path == "/api/frame" and grab:
                    image, key = session.pick_frame() if console else session.frame()
                    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 83])
                    if not ok:
                        raise RuntimeError("Could not encode camera frame")
                    return self.respond(200, jpg.tobytes(), "image/jpeg", key)
                if path == "/api/status" and grab:
                    return self.respond(200, session.pick_status() if console else session.status())
                if not grab and console:
                    result = session.get(path, url.query)
                    if isinstance(result, tuple):
                        return self.respond(200, *result)
                    return self.respond(200, result)
                raise LookupError(path)
            except LookupError:
                return self.respond(404, {"error": "Not found"})
            except (ValueError, RuntimeError, OSError, ImportError, cv2.error) as exc:
                return self.respond(409, {"error": str(exc)})

        def do_POST(self):
            if not self.authorize():
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= (65536 if console else 4096):
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")
                route = urlsplit(self.path).path
                if console:
                    result = (session.pick_post(route.removeprefix("/grasp/api/"), request)
                              if route.startswith("/grasp/api/") else session.post(route, request))
                else:
                    result = pick_action(session, route.removeprefix("/api/"), request)
                return self.respond(200, result)
            except (ValueError, KeyError, TypeError, OverflowError, RuntimeError, OSError, ImportError, cv2.error) as exc:
                return self.respond(400, {"error": str(exc)})
            except LookupError:
                return self.respond(404, {"error": "Not found"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--calibration", required=True, type=Path, help="Nero wrist hand-eye calibration JSON")
    parser.add_argument("--tcp-xyz-m", required=True, help="Measured grasp point x,y,z in gripper_base, in metres")
    parser.add_argument("--tcp-rpy-deg", help="Tool axes roll,pitch,yaw in gripper_base (tool +Z is approach direction)")
    parser.add_argument("--table-z-m", required=True, type=float, help="Measured tabletop height in the robot base frame")
    parser.add_argument("--clearance-m", type=float, default=0.010, help="Minimum checked point height above the table")
    parser.add_argument("--max-travel-m", type=float, default=0.4, help="Maximum tool displacement per task")
    parser.add_argument("--gripper-open", type=float, help="Open position in the configured driver units")
    parser.add_argument("--gripper-close", type=float, help="Close position in the configured driver units")
    parser.add_argument("--execute", action="store_true", help="Allow confirmed motion from the local browser")
    parser.add_argument("--port", type=int, default=0, help="Loopback HTTP port (0 selects a free port)")
    parser.add_argument("--no-browser", action="store_true", help="Print the local URL without opening a browser")
    args, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(args.preset or "nero_gamepad", overrides, args.deployment, require_lerobot=False)
        if not 0 <= args.port <= 65535:
            raise ValueError("Use port 0–65535")
        setup = GraspSetup(args.calibration, args.tcp_xyz_m, args.table_z_m, args.tcp_rpy_deg,
                           args.clearance_m, args.max_travel_m, args.gripper_open, args.gripper_close)
        calibration = setup.camera_calibration(cfg)
        cfg.robot.update(enable_on_connect=False, rest_on_start=False, disable_on_disconnect=False)
        camera_cfg = cfg.camera(calibration.camera)
        camera = RealSenseCapture(replace(camera_cfg, publish_frames=False), align_depth=True)
        robot = make_robot(cfg)
        motion = setup.motion(cfg, robot, calibration)
        session = PickSession(camera, motion, execute=args.execute)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"Hover and grasp: {exc}\n")

    try:
        with create_server(session, args.port) as server:
            robot.connect(calibrate=False)
            if not camera.start() or not camera.wait_for_frame(5):
                raise RuntimeError("Could not start the wrist RealSense. Check the source and stream mode.")
            check_camera_serial(camera, calibration)
            url = f"http://127.0.0.1:{server.server_port}/#token={session.token}"
            print(f"Hover and grasp console ({'motion enabled' if args.execute else 'preview mode'}): {url}", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ImportError, ValueError) as exc:
        parser.exit(1, f"Hover and grasp: {exc}\n")
    finally:
        session.close()
        camera.stop()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
