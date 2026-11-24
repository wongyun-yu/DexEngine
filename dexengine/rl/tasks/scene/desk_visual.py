"""The visual desk entity of the scene ({ENV}/Desk): the one function that makes it.

One desk for every scene and every process (the box desk and the desk choice are gone): the
Willow table (NVIDIA Commercial Assets Pack; USD in cm, unit scale DESK_VISUAL_UNIT), rotated 90° about z (asset x =
world y), read from the local look mirror (dexengine.record.looks.look_assets.willow_usd — downloaded from NVIDIA's S3,
sha256-pinned, never in a tree; missing = ConfigError naming the download step). It is visual only: the desk collides
through its cuboids, and `dexengine.core.desk.apply_desk_top` fits the visual and the cuboids as one piece.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from dexengine.core import desk as DK
from dexengine.core.config import ConfigError
from dexengine.record.looks import look_assets as LA

DESK_PRIM_PATH = "{ENV_REGEX_NS}/Desk"
WILLOW_ROT = (0.707, 0.0, 0.0, 0.707)  # 90° about z, the same rotation as the collision top cuboid
MARK = "[DESK]"


def desk_cfg(x: float, environ=None) -> AssetBaseCfg:
    """The {ENV}/Desk entity, centered at x. Prints the [DESK] marker."""
    environ = os.environ if environ is None else environ
    if gone := [k for k in DK.REMOVED_DESK_ENV if k in environ]:
        raise ConfigError(f"{MARK} {gone} removed — one desk (Willow from the look mirror) for every scene")
    usd = LA.willow_usd()
    ent = AssetBaseCfg(
        prim_path=DESK_PRIM_PATH,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, 0.0, 0.0), rot=WILLOW_ROT),
        spawn=sim_utils.UsdFileCfg(usd_path=usd, scale=(DK.DESK_VISUAL_UNIT,) * 3),
    )
    print(f"{MARK} desk=willow usd={usd} · visual only (collision = the desk cuboids)", flush=True)
    return ent
