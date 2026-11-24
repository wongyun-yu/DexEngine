"""f0 object repair outputs: the window rows of repair_obj and the frame-0 separation's object rows (repair_sep) written
into the tree in place (write_obj_window), and the rows' content hash the downstream keys read (obj_rows_sha)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from dexengine.core.paths import WRIST_SIDECAR_SFX
from dexengine.placement.judge.f0_key import OBJ_WINDOW_MARKER
from dexengine.retarget.f0 import repair_root as RT
from dexengine.retarget.f0.repair_cfg import HC
from dexengine.retarget.f0.repair_geom import _aa_to_R, _quat_wxyz_to_R, _R_to_quat_wxyz, fit_wrist_qpos_map
from dexengine.retarget.f0.repair_sep import SEP_FIELD


def obj_rows_sha(sides_rows: dict) -> str:
    """The content hash of the rows the object repair wrote: side name and each change row (float64, rounded to 1e-9
    so a re-run of the same solve hashes the same). The downstream keys read it from the marker (judge input_clip)."""
    h = hashlib.sha256()
    for side in sorted(sides_rows):
        h.update(side.encode())
        h.update(np.round(np.asarray(sides_rows[side], np.float64), 9).tobytes())
    return h.hexdigest()


def _window_side(qpos, wp_f, side, r, fit, tree):
    """Apply one side's window rows: wrist cache pose, hand-root translation (the root's axes and offset), hand-root
    rotation (repair_root.turn_rows, for the rows that turn the wrist) and fingers. Returns the new (positions,
    quaternions) of the cache; `qpos` is changed in place and r gets its ROOT_ROT record."""
    with np.load(wp_f) as wr:
        pos, quat = wr["positions"].astype(np.float64), wr["quaternions"].astype(np.float64)
    quat_in = quat.copy()
    off = r["_internal"][6]
    A, b, _res = fit
    Ainv = np.linalg.inv(A)
    turned = []
    for f, dq in enumerate(np.asarray(r["obj_window"], np.float64)):
        if f >= len(qpos) or f >= len(pos):
            break
        pos[f] = pos[f] + dq[:3]
        quat[f] = _R_to_quat_wxyz(_aa_to_R(dq[3:6]) @ _quat_wxyz_to_R(quat[f]))
        qpos[f, off : off + 3] = Ainv @ (pos[f] - b)
        indep = qpos[f, off + 6 : off + HC.H][HC.indep_idx]
        qpos[f, off + 6 : off + HC.H] = HC.expand(indep + dq[6:])
        if RT.turns(dq):
            turned.append(f)
    r[RT.ROOT_ROT] = {"rows": 0}
    if turned:
        sub = qpos[turned]
        r[RT.ROOT_ROT] = RT.turn_rows(tree, off, side, HC.name, sub, quat_in[turned], quat[turned], pos[turned])
        qpos[turned] = sub
    return pos, quat


def write_obj_window(seq: str, sides: dict, data_root: Path, wrist_root: Path, sep: dict | None = None) -> None:
    """In-place write of the object repair (repair_obj): each side's rows `obj_window` (window + fade, one change per
    frame from frame 0) applied to the tree's qpos and the side's wrist cache. The length and the timeline are
    unchanged (no pad). The originals are backed up once as *.pre_f0r.npz; the marker F0_OBJ_WINDOW.json carries the
    rows' hash (obj_rows_sha), and F0_REPAIR_INPLACE.json (mode window, P = the existing pad) is written for the
    idempotence check and the ledger — a desk repair of the other side written after this replaces it. `sep`
    (repair_sep.moved: {mover: {qs, rows, partner}}) adds each moved main object's rows to its position columns, and
    the markers carry them (field SEP_FIELD; the hash covers them). Without it the outputs are the ones before."""
    d = data_root / seq
    kin = d / "0/trajectory_kinematic.npz"
    npz = np.load(kin, allow_pickle=True)
    qpos = npz["qpos"].copy()
    fits = {
        s: fit_wrist_qpos_map(
            npz["qpos"][:, r["_internal"][6] : r["_internal"][6] + 6],
            np.load(wrist_root / f"wrist_poses_{s}" / f"{seq}.npz")["positions"].astype(np.float64),
            RT.root_axes(d, r["_internal"][6]),
            where=f"{seq} [{s}] window",
        )
        for s, r in sides.items()
    }
    bkin = d / f"0/trajectory_kinematic{WRIST_SIDECAR_SFX}"
    if not bkin.exists():
        shutil.copy2(kin, bkin)
    for s, r in sides.items():
        wp_f = wrist_root / f"wrist_poses_{s}" / f"{seq}.npz"
        bwp = wp_f.with_suffix(WRIST_SIDECAR_SFX)
        if not bwp.exists():
            shutil.copy2(wp_f, bwp)
        pos, quat = _window_side(qpos, wp_f, s, r, fits[s], d)
        np.savez(wp_f, positions=pos.astype(np.float32), quaternions=quat.astype(np.float32))
    for mv in (sep or {}).values():
        qpos[: len(mv["rows"]), mv["qs"] : mv["qs"] + 3] += mv["rows"]
    np.savez(kin, **{**{k: npz[k] for k in npz.files}, "qpos": qpos})
    _write_window_markers(d, seq, sides, sep)


def _sep_meta(sep: dict) -> dict:
    """The markers' SEP_FIELD record: per moved main object its partner, rows and the frame-0 move (mm, unit)."""
    out = {}
    for n, mv in sorted(sep.items()):
        v = np.asarray(mv["rows"][0], np.float64)
        mm = float(np.linalg.norm(v)) * 1000.0
        out[n] = {"partner": mv["partner"], "frames": len(mv["rows"]), "move_mm": round(mm, 3)}
        out[n]["dir"] = [round(float(x), 4) for x in v / max(np.linalg.norm(v), 1e-12)]
    return out


def _write_window_markers(d: Path, seq: str, sides: dict, sep: dict | None = None) -> None:
    """write_obj_window step: F0_OBJ_WINDOW.json and the F0_REPAIR_INPLACE.json of a window-only repair."""
    fj = d / "F0_REPAIR.json"
    P_base = int(json.loads(fj.read_text()).get("n_blend_effective", 0) or 0) if fj.exists() else 0
    rows = {s: r["obj_window"] for s, r in sides.items()}
    rows.update({f"{SEP_FIELD}/{n}": mv["rows"] for n, mv in (sep or {}).items()})
    meta = {
        "hand": HC.name,
        "rows_sha": obj_rows_sha(rows),
        "sides": {
            s: {
                "frames": int(len(r["obj_window"])),
                "window": int(r["obj_window_n"]),
                "sat_mm": r["obj_window_sat_mm"],
                "carried": r["obj_window_carried"],
            }
            for s, r in sides.items()
        },
    }
    if sep:
        meta[SEP_FIELD] = _sep_meta(sep)
    (d / OBJ_WINDOW_MARKER).write_text(json.dumps(meta, indent=1))
    skip = ("_internal", "obj_window")
    inplace = {
        "hand": HC.name,
        "mode": "window",
        "P": P_base,
        "P_base": P_base,
        "sides": {s: {k: v for k, v in r.items() if k not in skip} for s, r in sides.items()},
        "note": f"window: rows 0… of the repaired sides rewritten in place (length unchanged; {OBJ_WINDOW_MARKER})",
    }
    if sep:
        inplace[SEP_FIELD] = meta[SEP_FIELD]
    (d / "F0_REPAIR_INPLACE.json").write_text(json.dumps(inplace, indent=1, default=str))
