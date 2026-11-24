"""The hand-root rotation of the rows an f0 writer changes.

The qpos is the one owner of the hand pose: the chain re-derives every downstream wrist cache from its FK (body →
wrist stages), so a repaired wrist rotation written only into the f0 wrist cache never reaches a final tree. The
writers call ``turn_rows`` for a side whose solve turned the wrist: the rotation columns of each changed row are set so
the palm FK rotation equals that row's wrist-cache quaternion, through the hand root's convention owner
(placement.retime_variants.BaseMap, fitted from the tree's own scene). The cache rows are checked against the FK both
before (the cache must be the FK of the qpos it came with) and after the write. A side whose solve did not turn the
wrist keeps its rotation columns (turn_rows reads no model; those bytes stay). The translation columns of every written
row go through the root's axes read from the scene (``root_axes``; compiled once per tree, ``scene_model``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

#: Per-side record the writers add to the F0_REPAIR_INPLACE side (and the window sides); a side that turned the
#: wrist without it was written before the rotation reached the qpos (cache_verdict: STALE).
ROOT_ROT = "root_rot"
#: The wrist-cache rows must be the palm FK of the qpos rows they came with (a float32 cache: ~1e-7 rad).
CACHE_FK_TOL_RAD = 1e-4
#: The written rows' palm FK against the cache rows they claim: rotation, and position (the writer's affine
#: translation map is good to repair_geom.WRIST_FIT_MAX_RES; a palm off the root's rotation centre would miss it).
WRITE_TOL_RAD = 1e-6
WRITE_TOL_M = 2e-4
WHY_UNWRITTEN = "wrist rotation of the repair not in the qpos (written before the root-rotation fix)"
#: The compiled scene of the trees this process wrote last (root_axes and turn_rows read the same scene for each
#: written side): keyed on the scene path and its bytes. Neither reader changes the model (each makes its own MjData).
_SCENE: dict = {}
_SCENE_MAX = 4


def turns(dq) -> bool:
    """True when the solve turned the wrist (any rotation component of dq is not exactly zero)."""
    return bool(np.any(np.asarray(dq, np.float64)[3:6] != 0.0))


def _rot(quat_wxyz) -> Rot:
    q = np.asarray(quat_wxyz, np.float64)
    return Rot.from_quat(q[..., [1, 2, 3, 0]])


def _wrap(v):
    return np.angle(np.exp(1j * np.asarray(v, np.float64)))


def _near(owner, cols, ref):
    """The spelling of ``cols`` (owner's canonical form or its other spelling, each 2*pi-aligned) closest to ``ref``
    column by column — the written rows stay continuous with the rows around them."""
    c1 = ref + _wrap(cols - ref)
    c2 = ref + _wrap(np.asarray(owner.alt_euler(cols), np.float64) - ref)
    pick = np.abs(c2 - ref).sum(axis=1) < np.abs(c1 - ref).sum(axis=1)
    return np.where(pick[:, None], c2, c1)


def _palm(model, rows, side: str, hand: str) -> tuple[np.ndarray, Rot]:
    from dexengine.data.base_layer import geom as G  # noqa: PLC0415 — palm FK owner (HandSpec.palm_fk_body_right)

    fk = G.palm_fk(model, rows, [side], hand)
    return np.stack([f[0][0] for f in fk]), Rot.from_matrix(np.stack([f[0][1] for f in fk]))


def scene_model(tree: Path):
    """The tree's scene.xml compiled (mujoco.MjModel), once per process for the same path and bytes."""
    import mujoco  # noqa: PLC0415

    p = Path(tree) / "scene.xml"
    key = (str(p), hashlib.sha256(p.read_bytes()).hexdigest())
    if key not in _SCENE:
        if len(_SCENE) >= _SCENE_MAX:
            _SCENE.clear()
        _SCENE[key] = mujoco.MjModel.from_xml_path(str(p))
    return _SCENE[key]


def root_axes(tree: Path, off: int) -> np.ndarray:
    """The hand root's translation axes (3x3: world displacement per unit of qpos[off:off+3]) from the tree's scene —
    the convention owner's A (placement.retime_variants.BaseMap). The writers' wrist map takes it
    (repair_geom.fit_wrist_qpos_map); it does not depend on how the clip moves."""
    from dexengine.placement import retime_variants as RV  # noqa: PLC0415 — hand-root convention owner

    return RV.BaseMap(scene_model(tree), off).A


def turn_rows(tree: Path, off: int, side: str, hand: str, rows: np.ndarray, quat_old, quat_new, pos_new) -> dict:
    """Set IN PLACE the rotation columns (off+3:off+6) of ``rows`` (N, nq) so each row's palm FK rotation equals
    ``quat_new`` (N, 4 wxyz, the wrist-cache rows written for them; ``pos_new`` (N, 3) their positions, which the
    rows' translation columns already hold). ``quat_old`` (N, 4) = the cache rows the rows came with: checked against
    the FK of ``rows`` first. Rows whose quat_new equals quat_old bit for bit keep their bytes. Returns the side record
    (ROOT_ROT); dies when either check misses."""
    from dexengine.placement import retime_variants as RV  # noqa: PLC0415 — hand-root convention owner

    model = scene_model(tree)
    owner = RV.BaseMap(model, off)
    quat_old, quat_new = np.asarray(quat_old, np.float64), np.asarray(quat_new, np.float64)
    sel = np.flatnonzero(np.any(quat_new != quat_old, axis=1))
    rec = {"rows": int(len(sel)), "owner": f"{owner.seq}/{''.join('+' if s > 0 else '-' for s in owner.sign)}"}
    if not len(sel):
        return rec
    sub = np.array(rows[sel], np.float64)
    _, palm_old = _palm(model, sub, side, hand)
    before = float((palm_old.inv() * _rot(quat_old[sel])).magnitude().max())
    if not before <= CACHE_FK_TOL_RAD:
        raise ValueError(
            f"[F0_ROOT_ROT] ✗ {Path(tree).name} [{side}]: wrist cache ↔ palm FK of its qpos {before:.2e} rad > "
            f"{CACHE_FK_TOL_RAD:g} — the cache did not come from this qpos (stale cache)"
        )
    ref = sub[:, off + 3 : off + 6]
    base_old = owner.rot(ref)
    base_new = _rot(quat_new[sel]) * (palm_old.inv() * base_old)
    cols = owner.from_world(np.zeros((len(sel), 3)), base_new)[:, 3:6]
    sub[:, off + 3 : off + 6] = _near(owner, cols, ref)
    p_fk, r_fk = _palm(model, sub, side, hand)
    after = float((r_fk.inv() * _rot(quat_new[sel])).magnitude().max())
    miss = float(np.abs(p_fk - np.asarray(pos_new, np.float64)[sel]).max())
    if not (after <= WRITE_TOL_RAD and miss <= WRITE_TOL_M):
        raise ValueError(
            f"[F0_ROOT_ROT] ✗ {Path(tree).name} [{side}]: written rows' palm FK ↔ cache {after:.2e} rad · {miss:.2e} m"
        )
    rows[sel, off + 3 : off + 6] = sub[:, off + 3 : off + 6]
    turn = float((base_old.inv() * base_new).magnitude().max())
    rec.update(
        turn_deg=round(float(np.degrees(turn)), 3),
        cache_fk_rad=float(f"{before:.2e}"),
        fk_err_rad=float(f"{after:.2e}"),
        fk_pos_mm=round(miss * 1000, 4),
    )
    print(
        f"[F0_ROOT_ROT] {Path(tree).name} side={side} rows={len(sel)} turn={rec['turn_deg']}deg "
        f"cache↔FK before={before:.1e}rad after={after:.1e}rad {miss * 1000:.3f}mm owner={rec['owner']}",
        flush=True,
    )
    return rec


def unwritten(tree_dir: Path) -> bool:
    """True when the shadow's repair turned a wrist but its marker has no ROOT_ROT record for that side (written by a
    writer that put the rotation in the wrist cache only). Window sides always carry the record."""
    f = Path(tree_dir) / "F0_REPAIR_INPLACE.json"
    if not f.is_file():
        return False
    j = json.loads(f.read_text())
    for s in (j.get("sides") or {}).values():
        if ROOT_ROT in s:
            continue
        if j.get("mode") == "window" or (s.get("dq") is not None and turns(s["dq"])):
            return True
    return False
