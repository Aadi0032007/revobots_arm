"""Own one follower, its cameras, and the current operation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from pathlib import Path
import secrets
from threading import Event, RLock
import time
from urllib.parse import parse_qs

import cv2
import numpy as np

from revobots_arm.cameras.manager import CamerasManager
from revobots_arm.config.presets import PRESETS
from revobots_arm.grasp.motion import MotionStopped, check_stop
from revobots_arm.grasp.server import pick_action
from revobots_arm.grasp.session import BUSY, PLAN_WINDOW_S, PickSession
from revobots_arm.grasp.setup import GraspSetup, check_camera_serial
from revobots_arm.hand_eye.cli import board_png, detect_view, make_board, save_calibration, solve_views

from .devices import make_config, open_robot, run_teleop, setup_values
from .motion import ManualControl
from .policy import PolicyRun


logger = logging.getLogger(__name__)


@dataclass
class Pending:
    kind: str
    value: object
    detail: dict
    token: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    expiry: float = field(default_factory=lambda: time.monotonic() + PLAN_WINDOW_S)


class ArmConsole:
    def __init__(self, baseline, *, live=False):
        self.baseline = baseline
        self.live = live
        self.token = secrets.token_urlsafe(24)
        self.robot = self.cameras = self.manual = self.pick = self.cfg = None
        self.wrist = None
        self._lock = RLock()
        self._stop = Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arm-console")
        self._job = ""
        self._message = "Choose a follower and cameras in Setup."
        self._feedback = {}
        self._pending = None
        self._eye_views = []
        self._eye_settings = self._eye_size = None
        self._eye_file = ""
        self._closing = False

    def _pick_busy(self):
        return self.pick is not None and self.pick.status()["state"] in BUSY

    def _idle(self):
        if self._closing or self._job or self._pick_busy():
            raise ValueError("Wait for the active task or press Stop")

    def _connected(self):
        if self.robot is None:
            raise ValueError("Connect the follower in Setup")

    def _start(self, name, message, work, *args):
        self._idle()
        self._pending = None
        if self.pick:
            self.pick.stop()
        self._stop.clear()
        self._job, self._message = name, message
        self._pool.submit(self._work, work, args)

    def _work(self, work, args):
        try:
            message, pending = work(*args)
            with self._lock:
                stopped = self._stop.is_set()
                self._message = "Stopped." if stopped else message
                self._pending = None if stopped else pending
        except MotionStopped:
            self._report("Stopped.")
        except Exception as exc:
            logger.exception("Arm console task failed")
            self._report(f"Error: {exc}")
        finally:
            with self._lock:
                self._job = ""

    def _report(self, message):
        with self._lock:
            self._message = message

    def _publish(self, feedback):
        with self._lock:
            self._feedback = feedback

    def _connect(self, config, wrist):
        robot = open_robot(config, live=self.live)
        cameras = CamerasManager(config, aligned_camera=wrist)
        try:
            robot.connect(calibrate=False)
            check_stop(self._stop)
            manual = ManualControl(config, robot)
            feedback = robot.get_observation()
            cameras.start()
            check_stop(self._stop)
        except Exception:
            try:
                cameras.stop()
            finally:
                if robot.is_connected:
                    robot.disconnect()
            raise
        with self._lock:
            self.robot, self.cameras, self.manual = robot, cameras, manual
            self.cfg, self.wrist, self._feedback = config, wrist, feedback
        failed = f" Camera(s) failed: {', '.join(cameras.failed)}." if cameras.failed else ""
        return f"Follower connected.{failed}", None

    def _disconnect(self):
        if self.pick:
            self.pick.close()
        try:
            if self.cameras:
                self.cameras.stop()
        finally:
            if self.robot is not None and self.robot.is_connected:
                self.robot.disconnect()
            with self._lock:
                self.robot = self.cameras = self.manual = self.pick = self.cfg = None
                self.wrist, self._feedback = None, {}
                self._reset_eye()
                self._eye_file = ""
        return "Follower disconnected.", None

    def _preview(self, kind, request):
        feedback = self.robot.get_observation()
        self._publish(feedback)
        if kind == "joints":
            plan = self.manual.joints_plan(request["targets"], feedback)
        else:
            plan = self.manual.pose_plan(request["xyz"], request["rpy"], feedback)
        return "Path preview ready.", Pending(kind, plan, {
            "targets": plan.target, "seconds": round(len(plan.samples) * plan.dt, 1),
        })

    def _run_plan(self, plan):
        self.manual.execute(plan, self._stop)
        return "Move complete.", None

    def _teleop(self):
        run_teleop(self.cfg, self.robot, self.manual, self._stop, self._report, self._publish)
        return "Teleop stopped.", None

    def _preview_policy(self, request):
        policy = PolicyRun(self.cfg, request)
        first = policy.run(self.robot, self.cameras, self.manual, self._stop, preview=True)
        return "First policy action ready.", Pending("policy", policy, {"targets": first, "seconds": policy.duration})

    def _run_policy(self, policy):
        policy.run(self.robot, self.cameras, self.manual, self._stop, preview=False)
        return "Policy run complete.", None

    def _configure_pick(self, request):
        setup = GraspSetup(**request)
        calibration = setup.camera_calibration(self.cfg)
        if self.wrist != calibration.camera or self.wrist not in self.cameras.captures:
            raise ValueError("Select and connect the calibrated grasp camera in Setup")
        camera = self.cameras.captures[self.wrist]
        check_camera_serial(camera, calibration)
        motion = setup.motion(self.cfg, self.robot, calibration)
        if self.pick:
            self.pick.close()
        self._pending = None
        self.pick = PickSession(camera, motion, execute=self.live)

    def _reset_eye(self):
        self._eye_views.clear()
        self._eye_settings = self._eye_size = None

    def _sample_eye(self, camera, square, marker):
        settings = camera, square, marker
        if self._eye_settings is not None and self._eye_settings != settings:
            raise ValueError("Reset the samples before changing the camera or board dimensions")
        if len(self._eye_views) >= 40:
            raise ValueError("Solve from the 40 collected views")
        board = make_board(square, marker)
        kin = self.manual.kin
        start = kin.motor_to_q(self.robot.get_observation())
        if self._stop.wait(0.3):
            check_stop(self._stop)
        boundary = time.time()
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            stamp, frame = self.cameras.read(camera)
            if stamp is not None and stamp >= boundary and frame is not None:
                break
            if self._stop.wait(0.01):
                check_stop(self._stop)
        else:
            raise ValueError("Waiting for a fresh wrist camera frame")
        joints = kin.motor_to_q(self.robot.get_observation())
        if np.max(np.abs(joints - start)) > np.radians(0.4):
            raise ValueError("Wait for the arm to settle before capturing")
        if any(np.max(np.abs(joints - existing)) < np.radians(2) for existing, _ in self._eye_views):
            raise ValueError("Move or rotate the wrist to collect a different view")
        view = detect_view(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board, cv2.aruco.CharucoDetector(board))
        if view is None:
            raise ValueError("Show more of the ChArUco board in the wrist camera")
        size = frame.shape[:2][::-1]
        with self._lock:
            check_stop(self._stop)
            if self._eye_size is not None and self._eye_size != size:
                raise ValueError("The camera resolution changed. Reset the samples.")
            self._eye_settings, self._eye_size = settings, size
            self._eye_views.append((joints, view))
            return f"Captured view {len(self._eye_views)}.", None

    def _solve_eye(self, output):
        if self._eye_settings is None:
            raise ValueError("Capture at least six distinct ChArUco views")
        camera, square, marker = self._eye_settings
        kin = self.manual.kin
        views = [(i, kin.fk(q), *view) for i, (q, view) in enumerate(self._eye_views, 1)]
        result = solve_views(views, self._eye_size, make_board(square, marker), kin, camera,
                             self.cfg.robot.get("id"), self.cameras.calibration_metadata()[camera])
        check_stop(self._stop)
        save_calibration(result, output)
        with self._lock:
            self._eye_file = str(output)
        return f"Saved calibration. Reprojection RMS {result.reprojection_rms_px:.2f} px.", None

    def _status(self):
        with self._lock:
            busy = self._pick_busy()
            error = ""
            if self.robot is not None and not self._job and not busy and not self._closing:
                try:
                    self._feedback = self.robot.get_observation()
                except (OSError, RuntimeError, ValueError) as exc:
                    error = str(exc)
            fields, pose = [], None
            if self.manual is not None and self._feedback:
                try:
                    fields = self.manual.position_fields(self._feedback)
                    pose = self.manual.pose(self._feedback)
                except (KeyError, TypeError, ValueError) as exc:
                    error = str(exc)
            if self._pending and self._pending.expiry <= time.monotonic():
                self._pending = None
                self._message = "Preview expired. Generate a new preview."
            pending = self._pending
            return {
                "live": self.live, "connected": self.robot is not None, "robot": self.cfg.robot_type if self.cfg else "",
                "input": self.cfg.teleop_type if self.cfg else "", "job": self._job or ("grasp" if busy else ""),
                "message": self._message, "feedback_error": error,
                "joints": fields, "pose": pose, "joint_control": bool(self.manual and not self.manual.mapped_nero),
                "cameras": self.cameras.names() if self.cameras else [],
                "failed_cameras": self.cameras.failed if self.cameras else [],
                "wrist": self.wrist, "grasp_ready": self.pick is not None,
                "eye_count": len(self._eye_views), "eye_file": self._eye_file,
                "preview": {**pending.detail, "id": pending.token, "kind": pending.kind} if pending else None,
            }

    def get(self, path, query):
        params = parse_qs(query)
        if path == "/api/status":
            return self._status()
        if path == "/api/config":
            preset = params.get("preset", [self.baseline.name])[0]
            cfg = self._preset(preset)
            return {"setup": setup_values(cfg), "presets": list(PRESETS),
                    "policy": {"path": cfg.policy.path, "dataset": cfg.policy.dataset_repo_id or cfg.dataset.repo_id,
                               "device": cfg.policy.device, "task": cfg.task, "duration": cfg.policy.duration_s}}
        if path == "/api/frame":
            with self._lock:
                camera = params.get("camera", [""])[0]
                stamp, frame = self.cameras.read(camera) if self.cameras else (None, None)
            if stamp is None or frame is None or not 0 <= time.time() - stamp <= 1:
                raise ValueError("Waiting for a live camera frame")
            ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise ValueError("Could not encode camera image")
            return data.tobytes(), "image/jpeg"
        if path == "/api/board":
            board = make_board(float(params.get("square", [35])[0]), float(params.get("marker", [26])[0]))
            return board_png(board), "image/png"
        raise LookupError(path)

    def _preset(self, name):
        if name == self.baseline.name:
            return self.baseline
        if name not in PRESETS:
            raise ValueError("Choose a known profile")
        cfg = PRESETS[name]()
        cfg.lerobot_src = self.baseline.lerobot_src
        return cfg

    def post(self, path, request):
        with self._lock:
            if path == "/api/stop":
                self.stop()
                return {"ok": True}
            self._idle()
            if path == "/api/connect":
                if self.robot is not None:
                    raise ValueError("Disconnect before applying a new setup")
                cfg, wrist = make_config(self._preset(request.get("preset", self.baseline.name)), request)
                self._start("connect", "Opening follower and cameras…", self._connect, cfg, wrist)
                return {"ok": True}
            self._connected()
            if path == "/api/disconnect":
                self._start("disconnect", "Closing follower and cameras…", self._disconnect)
            elif path in ("/api/joints", "/api/pose"):
                self._start("planning", "Checking path…", self._preview, path.rsplit("/", 1)[1], request)
            elif path == "/api/execute":
                if not self.live:
                    raise ValueError("Start with --execute to allow movement")
                pending = self._pending
                if pending is None or request.get("id") != pending.token or time.monotonic() >= pending.expiry:
                    raise ValueError("Generate a fresh preview before confirming")
                work = self._run_policy if pending.kind == "policy" else self._run_plan
                self._start(pending.kind, "Running confirmed operation…", work, pending.value)
            elif path == "/api/teleop":
                if not self.live:
                    raise ValueError("Start with --execute to allow teleop")
                self._start("teleop", "Opening input device…", self._teleop)
            elif path == "/api/policy":
                self._start("planning", "Loading policy and sampling an action…", self._preview_policy, request)
            elif path == "/api/grasp-setup":
                self._configure_pick(request)
            elif path == "/api/eye-reset":
                self._reset_eye()
                self._message = "Calibration samples cleared."
            elif path == "/api/eye-sample":
                camera = request["camera"]
                if not self.manual.full_pose or camera not in self.cameras.captures:
                    raise ValueError("Connect native Nero and a wrist camera for live hand–eye calibration")
                self._start("calibration", "Waiting for a settled board view…", self._sample_eye,
                            camera, float(request["square"]), float(request["marker"]))
            elif path == "/api/eye-solve":
                output = Path(request["output"]).expanduser().resolve()
                if output.suffix.lower() != ".json" or output.exists():
                    raise ValueError("Enter a new calibration .json file path")
                self._start("calibration", "Solving hand–eye calibration…", self._solve_eye, output)
            else:
                raise LookupError(path)
            return {"ok": True}

    def pick_frame(self):
        with self._lock:
            if self.pick is None:
                raise ValueError("Apply grasp settings first")
            if self._job:
                return self.pick.preview_frame(), ""
            return self.pick.frame()

    def pick_status(self):
        with self._lock:
            if self.pick is None:
                raise ValueError("Apply grasp settings first")
            state = dict(self.pick.status())
            if self._job:
                state.update(state="stopping", message=self._message, target_mm=None, plan_id="")
            return state

    def pick_post(self, action, request):
        with self._lock:
            if action == "stop":
                self.stop()
                return self.pick_status()
            if self._job or self._closing or self.pick is None:
                raise ValueError("Hover and grasp is unavailable while another operation is active")
            self._pending = None
            return pick_action(self.pick, action, request)

    def stop(self):
        with self._lock:
            self._stop.set()
            self._pending = None
            busy = self._job or self._pick_busy()
            if self.pick:
                self.pick.stop()
            self._message = "Stopping…" if busy else "Stopped."

    def close(self):
        with self._lock:
            self._closing = True
            self.stop()
        self._pool.shutdown(wait=True)
        self._disconnect()
