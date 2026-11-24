"""The render look — the one statement of the background and lights every render process draws (it
replaces the sourced shell preset file and its two shell round trips).

Who reads it
    dexengine.rl.launch.modes.render_look_env — the env of a render or replay process (RL videos · replays · the
    record stage-2 replay · the one-hand FROZEN_RENDER_OUT render). Training and judging never draw (dexengine.rl.app).
    dexengine.record.looks.materials.base_roots — the MDLs look D's mirror must carry (look D draws over this look).
    dexengine.rl.render.scene_look applies the values on the built stage (knob names = the env names below).

Values (name NAME): plaster walls · fieldstone floor · brushed-aluminium desk top · key light 3200 · dome
1600 (0.15 grey) · the ego camera at a 12 mm focal length for videos (at the hardware 8 mm the head housing enters the
frame bottom; training and evaluation keep 8 mm). The MDLs are read from the local look mirror (look_assets, no S3
fallback). A value the caller already set (non-empty) wins — an A/B render stays possible; the ego focal length
follows RENDER_EGO_FOCAL_MM. There is one look: a PRESET other than NAME dies (the old `none` preset had no caller).
Standard library only.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

NAME = "V5_08_bright"
MARK = "[RENDER_LOOK]"
MATERIALS_REL = "Assets/Isaac/4.5/NVIDIA/Materials"  # same layout on S3 and in the local mirror
#: The background MDLs (env knob -> path under MATERIALS_REL), in binding order.
MDL = {
    "DEXENGINE_AUG_ROOM": "vMaterials_2/Plaster/Plaster_Wall.mdl",
    "DEXENGINE_AUG_FLOOR_MDL": "Base/Stone/Fieldstone.mdl",
    "DEXENGINE_AUG_DESK_MDL": "vMaterials_2/Metal/Aluminum_Brushed.mdl",
}
VALUES = {
    "DEXENGINE_KEY_LIGHT": "3200",
    "DEXENGINE_AUG_LIGHT": "1600,0.15,0.15,0.15",
    "DEXENGINE_LOG_EGO_PITCH": "1",  # the ego camera's extra downward pitch is logged
}
FOCAL_ENV, FOCAL_KNOB, FOCAL_MM = "RENDER_EGO_FOCAL_MM", "DEXENGINE_CAMERA_FOCAL_MM", "12"
PRESET_ENV = "PRESET"
_ROOT_ENV = ("DEXENGINE_MAIN_ROOT", "DEXENGINE_LAYOUT_CONFIG")  # what decides the data root (core.layout.roots)


def mdl_rels() -> list[str]:
    """Mirror-relative paths of the look's MDLs (MDL order)."""
    return [f"{MATERIALS_REL}/{rel}" for rel in MDL.values()]


@contextmanager
def _root_env(environ):
    """os.environ with the data-root variables of `environ` (the render process's env) while the mirror is resolved."""
    keep = {k: os.environ.get(k) for k in _ROOT_ENV}
    try:
        for k in _ROOT_ENV:
            if environ.get(k):
                os.environ[k] = environ[k]
            else:
                os.environ.pop(k, None)
        yield
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def mirror_dir(tree, environ) -> Path:
    """The look mirror a render process started from `tree` with `environ` reads (look_assets.MIRROR_REL under that
    process's data root)."""
    from dexengine.core.layout import roots
    from dexengine.record.looks import look_assets as LA

    with _root_env(environ):
        return Path(roots.data_root(str(tree))).resolve() / LA.MIRROR_REL


def env(tree, environ) -> dict[str, str]:
    """The look's env for a render process: the variables whose value differs from `environ` (a caller's non-empty
    value wins)."""
    preset = environ.get(PRESET_ENV, "")
    if preset not in ("", NAME):
        raise ValueError(f"{MARK} {PRESET_ENV}={preset!r}: one render look ({NAME}) — presets are gone (render_unify)")
    mats = mirror_dir(tree, environ) / MATERIALS_REL
    want = {k: str(mats / rel) for k, rel in MDL.items()} | VALUES
    out = {k: v for k, v in want.items() if not environ.get(k)}
    focal = environ.get(FOCAL_ENV) or FOCAL_MM
    if environ.get(FOCAL_KNOB) != focal:
        out[FOCAL_KNOB] = focal
    return out
