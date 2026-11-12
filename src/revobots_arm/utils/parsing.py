"""Shared parsing of joint names and calibration offsets."""


def parse_csv(text: str | None) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def parse_offsets(text: str | None) -> dict[str, float]:
    """``"elbow_flex:5,wrist_flex:-3"`` -> ``{"elbow_flex": 5.0, "wrist_flex": -3.0}``."""
    out: dict[str, float] = {}
    for item in parse_csv(text):
        name, _, val = item.partition(":")
        out[name.strip()] = float(val)
    return out
