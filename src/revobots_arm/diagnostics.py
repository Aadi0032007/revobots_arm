"""Inventory the host and check robot feedback, camera frame buses, and network input."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import importlib
from importlib import metadata, util
import ipaddress
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

from revobots_arm.config import ArmConfig
from revobots_arm.config.devices import load_config, make_parser
from revobots_arm.utils.bootstrap import PACKAGE_ROOT, ensure_lerobot_importable


def timings(samples: list[float]) -> dict[str, float]:
    """Summarize durations in seconds as milliseconds using a nearest-rank p95."""
    if not samples:
        return {}
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(ordered) * 1000, 3),
        "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1] * 1000, 3),
        "max_ms": round(ordered[-1] * 1000, 3),
    }


def run_local_command(args: list[str]) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, errors="replace", timeout=5,
    ).stdout.strip()


def tailscale_addresses() -> dict[str, Any]:
    executable = shutil.which("tailscale")
    if executable is None:
        return {"status": "CLI unavailable", "addresses": []}
    try:
        addresses = [str(ipaddress.ip_address(line)) for line in run_local_command([executable, "ip"]).splitlines()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"status": "address lookup failed or Tailscale is disconnected", "addresses": []}
    return {"status": "available" if addresses else "no address", "addresses": addresses}


def hardware_inventory() -> dict[str, Any]:
    system = platform.system()
    serial: list[str] = []
    cameras: list[str] = []
    can: dict[str, str] = {}
    errors: list[str] = []
    if system == "Linux":
        dev = Path("/dev")
        serial = [str(path) for pattern in ("ttyACM*", "ttyUSB*", "serial/by-id/*") for path in dev.glob(pattern)]
        cameras = [str(path) for path in dev.glob("video*")]
        for interface in Path("/sys/class/net").glob("*"):
            try:
                if (interface / "type").read_text().strip() == "280":  # Linux ARPHRD_CAN
                    can[interface.name] = (interface / "operstate").read_text().strip()
            except OSError:
                continue  # A USB adapter may disappear during enumeration.
    elif system == "Windows":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM") as key:
                serial = [winreg.EnumValue(key, i)[1] for i in range(winreg.QueryInfoKey(key)[1])]
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"Serial port enumeration failed: {exc}")
        powershell = shutil.which("powershell.exe")
        if powershell:
            try:
                cameras = run_local_command([
                    powershell, "-NoProfile", "-NonInteractive", "-Command",
                    "Get-PnpDevice -PresentOnly -Class Camera,Image -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FriendlyName",
                ]).splitlines()
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"Windows camera enumeration failed: {exc}")
        else:
            errors.append("Windows camera enumeration needs PowerShell")
    elif system == "Darwin":
        serial = [str(path) for path in Path("/dev").glob("cu.*")]
        try:
            report = run_local_command(["system_profiler", "SPCameraDataType", "-json"])
            cameras = [item["_name"] for item in json.loads(report).get("SPCameraDataType", []) if "_name" in item]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"macOS camera enumeration failed: {exc}")

    realsense = []
    if util.find_spec("pyrealsense2") is not None:
        try:
            rs = importlib.import_module("pyrealsense2")
            for device in rs.context().query_devices():
                realsense.append({
                    "name": device.get_info(rs.camera_info.name),
                    "serial": device.get_info(rs.camera_info.serial_number),
                })
        except (ImportError, RuntimeError, OSError) as exc:
            errors.append(f"RealSense enumeration failed: {exc}")
    return {
        "serial_ports": sorted(set(serial)), "can_interfaces": can,
        "camera_devices": sorted(set(cameras)), "realsense": realsense, "errors": errors,
    }


def environment_info(cfg: ArmConfig) -> dict[str, Any]:
    try:
        ensure_lerobot_importable(cfg.lerobot_src)
    except ImportError:
        pass
    packages = {}
    for name in ("revobots-arm", "lerobot", "numpy", "opencv-python", "opencv-python-headless",
                 "pygame", "pyrealsense2", "pyAgxArm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0], "executable": sys.executable,
        "package_source": str(PACKAGE_ROOT), "packages": packages,
        "lerobot_available": util.find_spec("lerobot") is not None,
    }


def camera_feedback(name: str, seconds: float) -> dict[str, Any]:
    from revobots_arm.transport.frame_bus import FrameBusReader

    reader = FrameBusReader(name)
    read_times, ages = [], []
    shape = dtype = None
    start = time.monotonic()
    try:
        while time.monotonic() - start < seconds:
            before = time.perf_counter()
            stamp, frame = reader.read_new()
            elapsed = time.perf_counter() - before
            if stamp is not None and frame is not None and frame.size:
                read_times.append(elapsed)
                shape, dtype = list(frame.shape), str(frame.dtype)
                age = time.time() - stamp  # FrameBusPublisher uses the local Unix clock.
                if math.isfinite(age) and age >= 0:
                    ages.append(age)
            time.sleep(min(0.005, max(0.0, seconds - (time.monotonic() - start))))
    finally:
        attached = reader.attached
        reader.close()
    result = {
        "ok": len(read_times) >= 2, "attached": attached, "frames": len(read_times),
        "observed_fps": round(len(read_times) / (time.monotonic() - start), 1),
        "shape": shape, "dtype": dtype,
        "read_time": timings(read_times), "frame_age": timings(ages),
    }
    if not result["ok"]:
        result["error"] = "No advancing frames. Start the camera owner with publish_frames=true."
    return result


def udp_feedback(bind: str, port: int, seconds: float) -> dict[str, Any]:
    received, invalid = [], 0
    sources: set[str] = set()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        if sys.platform == "win32":
            receiver.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        receiver.bind((bind, port))
        stop = time.monotonic() + seconds
        while (remaining := stop - time.monotonic()) > 0:
            receiver.settimeout(remaining)
            try:
                payload, (address, _) = receiver.recvfrom(65535)
            except TimeoutError:
                break
            arrived = time.monotonic()
            try:
                packet = json.loads(payload)
            except (UnicodeError, ValueError):
                packet = None
            if isinstance(packet, dict) and packet:
                received.append(arrived)
                sources.add(address)
            else:
                invalid += 1
    return {
        "ok": bool(received), "json_packets": len(received), "invalid_packets": invalid,
        "source_ips": sorted(sources),
        "arrival_gap": timings([b - a for a, b in zip(received, received[1:])]),
    }


def sample_positions(read: Callable[[], Mapping[str, Any]], expected: set[str], seconds: float) -> dict[str, Any]:
    durations = []
    positions: dict[str, float] = {}
    start = time.monotonic()
    while not durations or time.monotonic() - start < seconds:
        before = time.perf_counter()
        values = read()
        durations.append(time.perf_counter() - before)
        positions = {
            key: float(values[key]) for key in sorted(expected)
            if isinstance(values.get(key), Real) and not isinstance(values[key], bool)
            and math.isfinite(values[key])
        }
        if missing := sorted(expected - positions.keys()):
            raise ValueError(f"Missing or invalid feedback for {', '.join(missing)}")
        time.sleep(min(0.01, max(0.0, seconds - (time.monotonic() - start))))
    return {"ok": True, "reads": len(durations), "positions": positions, "read_time": timings(durations)}


def arm_feedback(cfg: ArmConfig, role: str, seconds: float) -> dict[str, Any]:
    connection_start = time.perf_counter()
    if role == "robot" and cfg.robot_type in {"agx_nero_native", "agx_nero_follower"}:
        from revobots_arm.robots.nero.driver import NeroJointRobot, NeroRobotConfig

        transport = {key: cfg.robot[key] for key in ("interface", "channel", "bitrate", "firmware_version") if key in cfg.robot}
        robot = NeroJointRobot(NeroRobotConfig(**transport, enable_on_connect=False))
        robot.connect(calibrate=False)
        try:
            connection_ms = (time.perf_counter() - connection_start) * 1000
            result = sample_positions(robot.get_observation, set(robot.observation_features), seconds)
        finally:
            robot.disconnect()
        return {**result, "unit": "degrees", "connect_ms": round(connection_ms, 3)}

    serial_robots = {"koch_follower", "so100_follower", "so101_follower"}
    serial_leaders = {"koch_leader", "so100_leader", "so101_leader"}
    if (role == "robot" and cfg.robot_type not in serial_robots) or (
        role == "leader" and cfg.teleop_type not in serial_leaders
    ):
        raise ValueError(f"No read-only {role} probe for this type. Use --udp-port for a remote input.")
    from revobots_arm.config.devices import make_robot, make_teleop

    ensure_lerobot_importable(cfg.lerobot_src)
    device = make_robot(cfg) if role == "robot" else make_teleop(cfg)
    bus = device.bus
    try:
        bus.connect(handshake=False)
        connection_ms = (time.perf_counter() - connection_start) * 1000
        result = sample_positions(lambda: bus.sync_read("Present_Position", normalize=False), set(bus.motors), seconds)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    return {**result, "unit": "encoder counts", "connect_ms": round(connection_ms, 3)}


def checked(operation: Callable, *args: Any) -> dict[str, Any]:
    try:
        return operation(*args)
    except Exception as exc:  # A device or optional dependency failure belongs in the report.
        return {"ok": False, "error": str(exc) or type(exc).__name__}


def print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    for name, fields in report.items():
        status = "  PASS" if fields.get("ok") is True else "  FAIL" if fields.get("ok") is False else ""
        print(f"[{name}]{status}")
        for key, value in fields.items():
            if key != "ok":
                print(f"  {key}: {json.dumps(value, ensure_ascii=True)}")


def add_probe_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    parser.add_argument("--robot", action="store_true", help="Read follower feedback with teleop stopped")
    parser.add_argument("--leader", action="store_true", help="Read a local serial leader with teleop stopped")


def info_main() -> None:
    parser = make_parser("Show host, environment, Tailscale addresses, and detected device endpoints.")
    add_probe_options(parser)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.preset, overrides, args.deployment, require_lerobot=False)
    report = {
        "machine": {
            "host": socket.gethostname(), "os": platform.platform(),
            "architecture": platform.machine(), "logical_cpus": os.cpu_count(),
        },
        "environment": environment_info(cfg),
        "tailscale": tailscale_addresses(),
        "devices": hardware_inventory(),
        "configuration": {
            "preset": cfg.name, "robot_type": cfg.robot_type, "teleop_type": cfg.teleop_type,
            "robot_endpoint": cfg.robot.get("port", cfg.robot.get("channel")),
            "teleop_endpoint": cfg.teleop.get("port", cfg.teleop.get("udp_port")),
            "cameras": {camera.name: camera.source for camera in cfg.enabled_cameras},
        },
    }
    for role in ("robot", "leader"):
        if getattr(args, role):
            report[f"{role} feedback"] = checked(arm_feedback, cfg, role, 0.25)
    print_report(report, args.json)
    if any(part.get("ok") is False for part in report.values()):
        raise SystemExit(1)


def test_main() -> None:
    parser = make_parser("Check dependencies, read feedback, and measure camera and UDP receive timing.")
    add_probe_options(parser)
    parser.add_argument("--camera", action="append", default=[], help="Read a live shared-memory key, such as wrist2D_depth")
    parser.add_argument("--udp-port", type=int, help="Listen for JSON input on a free UDP port")
    parser.add_argument("--bind", default="0.0.0.0", help="Local IPv4 address for the UDP receiver")
    parser.add_argument("--seconds", type=float, default=2.0, help="Sampling duration per check")
    args, overrides = parser.parse_known_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.udp_port is not None and not 1 <= args.udp_port <= 65535:
        parser.error("--udp-port must be between 1 and 65535")
    report = {}
    if args.robot or args.leader:
        cfg = load_config(args.preset, overrides, args.deployment, require_lerobot=False)
        for role in ("robot", "leader"):
            if getattr(args, role):
                report[f"{role} feedback"] = checked(arm_feedback, cfg, role, args.seconds)
    elif overrides:
        parser.error("configuration overrides require --robot or --leader")
    for name in dict.fromkeys(args.camera):
        report[f"camera {name}"] = checked(camera_feedback, name, args.seconds)
    if args.udp_port is not None:
        report[f"UDP {args.bind}:{args.udp_port}"] = checked(udp_feedback, args.bind, args.udp_port, args.seconds)
    if not report:
        for module in ("numpy", "cv2"):
            try:
                imported = importlib.import_module(module)
                report[module] = {"ok": True, "version": imported.__version__}
            except (ImportError, OSError) as exc:
                report[module] = {"ok": False, "error": str(exc)}
    print_report(report, args.json)
    if any(not part["ok"] for part in report.values()):
        raise SystemExit(1)
