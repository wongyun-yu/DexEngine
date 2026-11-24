"""Locations inside the code checkout (no third-party imports).

The repository root holds the package (`dexengine/`), the YAML configuration (`configs/`) and the binary robot,
hand and scene files (`assets/`). Data never lives here; data paths come from `dexengine.core.layout`.
"""

from __future__ import annotations

from pathlib import Path

#: The checkout root (the parent of the `dexengine/` package).
REPO_ROOT = Path(__file__).resolve().parents[2]
#: YAML/JSON configuration read by the pipeline.
CONFIGS_DIR = REPO_ROOT / "configs"
#: Binary assets (robots, hands, scene parts).
ASSETS_DIR = REPO_ROOT / "assets"
#: Robot and hand description files.
ROBOT_ASSETS_DIR = ASSETS_DIR / "robots"
#: Hand assets, one folder per retargeting hand spec directory (`<spec>/mjcf`, vendor URDFs, scene blocks).
HAND_ASSETS_DIR = ROBOT_ASSETS_DIR / "hands"
#: Merged arm + hand URDFs, one folder per combo (`<combo><variant>/<combo><variant>.urdf`).
FULL_ROBOT_URDF_DIR = ROBOT_ASSETS_DIR / "full_robot_gen"


def hand_rig_dir(spec_dir: str) -> Path:
    """The retargeting rig folder of one hand spec directory (`right.xml`, `left.xml`, `bimanual.xml`,
    `retarget_prior.yaml`, mesh `assets/`)."""
    return HAND_ASSETS_DIR / spec_dir / "mjcf"
