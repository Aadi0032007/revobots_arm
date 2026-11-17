import contextlib
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

from revobots_arm.config.presets import nero_cartesian
from revobots_arm.control.devices import setup_values
from revobots_arm.control.session import ArmConsole
from revobots_arm.robots.nero.driver import NERO_JOINTS


class FakeNero:
    def __init__(self):
        self.is_connected = False
        self.action_features = {f"{name}.pos": float for name in NERO_JOINTS}
        self.position = dict.fromkeys(self.action_features, 0.0)
        self.enable = Mock()
        self.send_action = Mock(side_effect=self.position.update)

    def connect(self, calibrate=False):
        self.is_connected = True

    def get_observation(self):
        return self.position.copy()

    def disconnect(self):
        self.is_connected = False


class ControlTests(unittest.TestCase):
    def wait_idle(self, session):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = session.get("/api/status", "")
            if not state["job"]:
                return state
            time.sleep(0.01)
        self.fail("Console did not become idle")

    @contextlib.contextmanager
    def connected(self, live):
        preset = nero_cartesian()
        robot = FakeNero()
        console = ArmConsole(preset, live=live)
        try:
            with patch("revobots_arm.control.session.open_robot", return_value=robot):
                console.post("/api/connect", setup_values(preset))
                self.assertTrue(self.wait_idle(console)["connected"])
            yield console, robot
        finally:
            console.close()

    def test_preview_is_read_only_and_live_confirm_is_single_use(self):
        for live in (False, True):
            with self.subTest(live=live), self.connected(live) as (console, robot):
                first = next(iter(robot.action_features))
                console.post("/api/joints", {"targets": {first: 1}})
                state = self.wait_idle(console)
                self.assertEqual(state["preview"]["targets"], {first: 1})
                robot.send_action.assert_not_called()
                robot.enable.assert_not_called()
                request = {"id": state["preview"]["id"]}
                if live:
                    console.post("/api/execute", request)
                    self.assertEqual(self.wait_idle(console)["message"], "Move complete.")
                    self.assertAlmostEqual(robot.position[first], 1)
                with self.assertRaises(ValueError):
                    console.post("/api/execute", request)

    def test_browser_and_grasp_operations_share_exclusive_arm_access(self):
        with self.connected(live=True) as (console, robot):
            pick = console.pick = Mock()
            pick.status.return_value = {"state": "moving"}
            with self.assertRaisesRegex(ValueError, "active task"):
                console.post("/api/teleop", {})
            pick.status.return_value = {"state": "idle"}
            entered = Event()

            def teleop(cfg, robot, manual, stop, report, publish):
                entered.set()
                if not stop.wait(5):
                    raise RuntimeError("Stop was not relayed")

            with patch("revobots_arm.control.session.run_teleop", side_effect=teleop):
                console.post("/api/teleop", {})
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(ValueError, "another operation"):
                    console.pick_post("confirm", {})
                console.pick_post("stop", {})
                self.assertEqual(self.wait_idle(console)["message"], "Stopped.")
                pick.stop.assert_called()
                pick.confirm.assert_not_called()
                robot.send_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
