"""Make ``lerobot`` importable without installing or modifying anything.

Resolution order:

1. ``lerobot`` is already importable (pip / editable install)  -> nothing to do
2. ``$LEROBOT_SRC``                                             -> ``<lerobot repo>/src``
3. ``ArmConfig.lerobot_src`` from ``config.py``                  -> same
4. ``../../src`` relative to this file                           -> works while this folder
                                                                   lives inside the LeRobot repo
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT.parent
ARM_ROOT = SRC_DIR.parent
ASSET_ROOT = PACKAGE_ROOT / "assets"
DEFAULT_LEROBOT_SRC = ARM_ROOT.parent / "src"  # .../<lerobot repo>/src

ENV_VAR = "LEROBOT_SRC"


def ensure_lerobot_importable(
    config_hint: str | os.PathLike | None = None,
) -> Path | None:
    """Put a LeRobot source tree on ``sys.path`` if ``lerobot`` cannot be imported yet.

    Returns the path that was added, or ``None`` when ``lerobot`` was already importable.
    """
    if importlib.util.find_spec("lerobot") is not None:
        return None

    for candidate in (os.environ.get(ENV_VAR), config_hint, DEFAULT_LEROBOT_SRC):
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "lerobot" / "__init__.py").is_file():
            sys.path.insert(0, str(path))
            return path

    raise ImportError(
        "Could not import `lerobot`. Either install the LeRobot repo into this environment "
        "(`pip install -e <lerobot repo>`), set the LEROBOT_SRC environment variable to "
        "`<lerobot repo>/src`, or set `lerobot_src` in src/config.py."
    )
