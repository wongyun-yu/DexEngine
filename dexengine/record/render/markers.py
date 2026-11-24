"""P&P place markers: flat display-only prims (no collision, rigid body or physics API; cast no shadows).
Placed only in the render session (render.episode.load_traj) and removed by render.clip_session._clear.

Switch: the caller's env DENSE_PLACE_MARKER=outline|contour|disc|ring|x-red|x-black|pad-red -> render.clip stamps it
  into the clip plan field place_marker (stamp_plan). The session reads only the plan field (style), so env inherited
  by a reused Kit worker does not leak into other clips. Every drawn episode checks render_meta.place_marker == the
  requested switch (check_episodes).
  Default off = no prims, no meta, no plan field and no render fingerprint tail (output bytes unchanged).
Shapes (chosen: ring):
  outline = a plate filling the end-pose object's floor footprint (xy convex hull of the bottom band points)
  contour = a line around the same footprint (band LINE_GAP … LINE_GAP + LINE_W outside it)
  disc    = a disc of the object's size (diameter = longer side of the end-pose object xy extent; no orientation)
  ring    = a ring around the same circle (radius r + LINE_GAP … r + LINE_GAP + LINE_W)
  height = end-pose object bottom (desk, support top, on the receiving object) + LIFT … + LIFT + PLATE_T.
  No object-shaped ghost: an opaque one overlaps the real object from the first frame of an in-place clip and
  flickers.
Colour: COLOR shared by the four shapes (cyan, OmniPBR opaque + self emission EMISSIVE). Look D has the translucency
  path (rtx.translucency) off and the render settings are not changed: a 0.5 marker was invisible in a translucency
  trial.
Place: p_end and R_end of the single source e5_place_ref.place_target (variant tree folder; tree frame, last frame of
  that variant's trajectory_kinematic.npz; frozen copy in the folder REF_DIR_ENV names, md5 REF_MD5).
  Tree frame -> env frame = the session env's reference trajectory transform as it is
  (trajectory_manager._rotate_pos_batch, _rotate_quat_batch, loader clip_dz_of) + the render world shift
  (render.body world_shift_m) + the env origin.
Check (measured vs stated): p_start moved by the same transform must be within POS_TOL_M of an object root of the
  session's first state (this also matches names to scene objects: the nearest one) · the end rotation is checked
  against Rz·R; otherwise it dies. frame='body' (receiving object is the main object) does not exist yet: dies.
Marker: ``[PLACE_MARKER] style=… obj=… pos=… quat=…`` (one line per object when on; X is
  ``style=x color=red|black``).
X marker (x-red, x-black: the shape taped with electrical tape on the real robot): two strips crossing at 90° (±45°
  to env +x) · width X_TAPE_W 19 mm · arm end = centre + r + X_PAD (same rule as the ring's outer radius) · matte
  (OmniPBR, no emission, X_COLORS).
Support pad (pad-red: a marker only, like a pad on the floor, red, one size and shape; none where the place is
  already obvious, like a scale): the same disc for every clip (radius PAD_R, thickness PLATE_T, matte X_COLORS, no
  emission) · centre = centre of the end-pose object xy extent · height = end-pose object bottom (supporting surface)
  + LIFT. Place = the support top under the end pose (support_top): on the desk (None) or our support discs (name
  SUPPORT_DISC_PREFIX) the pad is placed; on a data object (scale, plate, bowl …) it is not (skip=receiver line).
  When our support top is smaller than the pad, the pad stays as it is (one size) and the marker records overhang_m.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dexengine.core.errors import InputMissingError

ENV = "DENSE_PLACE_MARKER"
PLAN_KEY = "place_marker"  # clip plan (plan.json) field: sets the render session's switch (style)
# x-<colour> = tape X · pad-<colour> = support pad (colour field X_COLORS)
STYLES = ("outline", "contour", "disc", "ring", "x-red", "x-black", "pad-red")
MARK = "[PLACE_MARKER]"
#: env: the folder of the frozen copy of the place single source (e5_place_ref.py, md5 markers_geom.REF_MD5). No
#: default: a run with markers on dies without it (ref_file).
REF_DIR_ENV = "DENSE_PLACE_REF_DIR"
REF_DIR = os.environ.get(REF_DIR_ENV, "")
LINE_GAP, LINE_W = 0.004, 0.006  # contour, ring: 6 mm wide, 4 mm outside the footprint (circle)


def _valid(v: str, src: str) -> str | None:
    v = (v or "").strip()
    if not v:
        return None
    if v not in STYLES:
        raise SystemExit(f"{MARK} {src}={v!r}: one of {STYLES} only (unknown values die)")
    return v


def env_style() -> str | None:
    """The caller's switch (the process that plans the render): env DENSE_PLACE_MARKER. Stamped into the plan
    (stamp_plan) and part of the render fingerprint tail (tag)."""
    return _valid(os.environ.get(ENV, ""), ENV)


def style() -> str | None:
    """The render session's switch: set by PLAN_KEY of that job's plan (COLLECT_PLAN file; explicit per job). A reused
    Kit worker (session.persist) inherits env from its launcher and keeps it for other clips' jobs, so the session
    does not read env (no markers leaking into non-P&P clips). Only processes without a plan (closed-loop eval etc.)
    read env."""
    pp = os.environ.get("COLLECT_PLAN", "")
    if pp:
        import json

        return _valid(json.loads(Path(pp).read_text()).get(PLAN_KEY, ""), f"plan {PLAN_KEY}")
    return env_style()


def stamp_plan(plan: dict) -> dict:
    """Stamp the caller's env switch into that clip's plan (off = plan unchanged, byte-identical)."""
    s = env_style()
    if s:
        plan[PLAN_KEY] = s
    return plan


def check_episodes(dirs) -> None:
    """Check that each drawn episode's render_meta place_marker shape == the caller's env switch (off = no field);
    otherwise die."""
    import json

    want = env_style()
    for d in dirs:
        pm = json.loads((Path(d) / "render_meta.json").read_text()).get("place_marker")
        got = pm["style"] if pm else None
        if got != want:
            raise SystemExit(
                f"{MARK} {d}: drawn marker {got} != requested {want} (reused worker env mix-up or an old cache)"
            )


def code_files() -> list[Path]:
    """The marker code = every render/markers*.py (a rule, so a file split off later is covered too)."""
    return sorted(Path(__file__).parent.glob("markers*.py"))


def ref_file() -> Path:
    """The frozen place single source REF_DIR/e5_place_ref.py; dies when REF_DIR_ENV is unset."""
    if not REF_DIR:
        raise InputMissingError(f"{MARK} {REF_DIR_ENV} unset: the folder of the frozen e5_place_ref.py (no default)")
    return Path(REF_DIR) / "e5_place_ref.py"


def tag() -> str:
    """Render fingerprint tail: off = empty string (the old fingerprint) · on = shape + the bytes of the marker code
    (code_files) and the single source (a cache without markers is not reused)."""
    s = env_style()
    if s is None:
        return ""
    code = b"".join(p.read_bytes() for p in code_files())
    h = hashlib.sha256(code + ref_file().read_bytes()).hexdigest()[:8]
    return f"+place_marker_{s}_{h}"


def show(raw_env, binds, P0, tree_dir: Path, k: int, shift_z: float) -> dict | None:
    """Place the marker for episode k (Kit); built by render.markers_scene."""
    from dexengine.record.render import markers_scene as MS

    return MS.show(raw_env, binds, P0, tree_dir, k, shift_z)


def clear(stage) -> list[str]:
    """Remove the marker prims (Kit)."""
    from dexengine.record.render import markers_scene as MS

    return MS.clear(stage)
