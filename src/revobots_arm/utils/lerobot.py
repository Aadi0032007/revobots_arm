"""LeRobot registry integration; optional device families load only when selected."""

from __future__ import annotations
import importlib
import dataclasses
from dataclasses import dataclass
from .registry import instantiate
from pathlib import Path
from typing import Any


def import_device_module(candidates: tuple[str, ...]):
    """Try known package layouts; propagate missing SDKs inside an installed driver."""
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name and not name.startswith(str(exc.name) + "."):
                raise
    raise ModuleNotFoundError(
        "Compatible LeRobot device package required; tried: " + ", ".join(candidates)
    )


@dataclass(frozen=True)
class LeRobotAdapter:
    """Explicit local integration with an existing LeRobot driver, without cloning it."""

    kind: str
    type_name: str
    modules: tuple[str, ...]

    def config(self, spec: dict[str, Any]) -> Any:
        import_device_module(self.modules)
        if self.kind == "robot":
            from lerobot.robots import RobotConfig as Config
        else:
            from lerobot.teleoperators import TeleoperatorConfig as Config
        cls = _choice_class(Config, self.type_name, self.kind)
        values = {k: v for k, v in spec.items() if k != "type"}
        _normalise_common(values)
        if self.kind == "robot" and "cameras" in {
            f.name for f in dataclasses.fields(cls)
        }:
            values["cameras"] = {}
        return instantiate(cls, values, self.kind)

    def create(self, config: Any) -> Any:
        if self.kind == "robot":
            from lerobot.robots import make_robot_from_config

            return make_robot_from_config(config)
        from lerobot.teleoperators import make_teleoperator_from_config

        return make_teleoperator_from_config(config)


def _choice_class(registry: Any, type_name: str, what: str) -> type:
    try:
        return registry.get_choice_class(type_name)
    except KeyError:
        known = sorted(registry.get_known_choices())
        raise ValueError(
            f"Unknown {what} type '{type_name}'. Registered types: {known}"
        ) from None


def _normalise_common(spec: dict[str, Any]) -> None:
    if spec.get("calibration_dir"):
        spec["calibration_dir"] = Path(spec["calibration_dir"]).expanduser()
    else:
        spec.pop("calibration_dir", None)
    # `ensure_safe_goal_position` checks `isinstance(x, float)`, so an int would fail.
    mrt = spec.get("max_relative_target")
    if isinstance(mrt, int) and not isinstance(mrt, bool):
        spec["max_relative_target"] = float(mrt)
    elif isinstance(mrt, dict):
        spec["max_relative_target"] = {k: float(v) for k, v in mrt.items()}
