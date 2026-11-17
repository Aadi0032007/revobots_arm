"""Run the LeRobot inference API on the console's existing devices."""

import time

from revobots_arm.cameras.manager import RobotWithCameras
from revobots_arm.config import depth_key
from revobots_arm.grasp.motion import check_stop
from revobots_arm.utils.bootstrap import ensure_lerobot_importable

from .devices import checked_action


class _PreviewComplete(Exception):
    pass


class PolicyRun:
    def __init__(self, cfg, request):
        ensure_lerobot_importable(cfg.lerobot_src)
        try:
            from lerobot.scripts.lerobot_inference_api import build_policy_pipeline, run_inference
        except ImportError as exc:
            raise ImportError("Install the LeRobot fork that provides lerobot_inference_api") from exc
        self.duration = float(request["duration"])
        if not 0.1 <= self.duration <= 120:
            raise ValueError("Set policy run time between 0.1 and 120 seconds")
        self.fps = int(cfg.policy.fps or cfg.fps)
        if not 1 <= self.fps <= 60:
            raise ValueError("Set policy FPS between 1 and 60")
        self.task = request.get("task") or None
        self.infer = run_inference
        path = request["path"].strip()
        dataset = request["dataset"].strip()
        if not path or not dataset:
            raise ValueError("Enter a policy checkpoint and dataset repo ID")
        self.pipeline = build_policy_pipeline(policy_path=path, dataset_repo_id=dataset,
                                              device=request.get("device") or None,
                                              dataset_root=cfg.policy.dataset_root or cfg.dataset.root)
        features = self.pipeline[5]
        self.actions = set(features.get("action", {}).get("names") or [])
        self.state = set(features.get("observation.state", {}).get("names") or [])
        self.images = {key.removeprefix("observation.images.") for key in features
                       if key.startswith("observation.images.")}

    def run(self, robot, cameras, manual, stop, *, preview):
        if self.actions != set(robot.action_features):
            raise ValueError("Policy action names differ from the connected follower")
        if self.state - robot.observation_features.keys():
            raise ValueError("Policy state names differ from the connected follower")
        if self.images - cameras.shapes().keys():
            raise ValueError(f"Missing policy camera channels: {sorted(self.images - cameras.shapes().keys())}")
        adapter = _PolicyRobot(robot, cameras, manual, stop, self.images, preview)
        try:
            self.infer(robot=adapter, pipeline=self.pipeline, duration_s=self.duration, fps=self.fps,
                       single_task=self.task, display_data=False)
        except _PreviewComplete:
            return adapter.first
        finally:
            if adapter.sent:
                manual.hold()
        if preview:
            raise RuntimeError("The policy did not produce an action")


class _PolicyRobot(RobotWithCameras):
    def __init__(self, robot, cameras, manual, stop, images, preview):
        super().__init__(robot, cameras)
        self.manual, self.stop, self.images, self.preview = manual, stop, images, preview
        self.sent = False
        self.first = None

    def get_observation(self):
        check_stop(self.stop)
        return super().get_observation()

    def _check_age(self, key, ts, now):
        if key in self.images and (ts is None or not 0 <= now - ts <= self._stale_s):
            raise RuntimeError(f"Policy camera {key} stopped delivering fresh frames")

    def send_action(self, action):
        check_stop(self.stop)
        now = time.time()
        for name in self._cams.captures:
            if {name, depth_key(name)} & self.images:
                self._check_age(name if name in self.images else depth_key(name), self._cams.read_pair(name)[0], now)
        action = checked_action(action, set(self.action_features), complete=True)
        for key, value in action.items():
            low, high = self.manual.limits.get(key, (-float("inf"), float("inf")))
            if not low <= value <= high:
                raise ValueError(f"Policy target {key} is outside the configured range")
        if self.preview:
            self.first = action
            raise _PreviewComplete
        if not self.sent and self.manual.full_pose:
            self._robot.enable()
        check_stop(self.stop)
        self.sent = True
        return super().send_action(action)
