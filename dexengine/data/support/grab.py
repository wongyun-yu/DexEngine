#!/usr/bin/env python3
"""GRAB desk oracle (D10-b): measures the desk top z from the raw sequence's `table.params` + `table.ply` (the desk the
official viewer draws),
compares it with the per-clip `desk_z` in INTAKE `support_audit.json` (rest estimate = delivered z0 object bottom), and
writes the result into the same ledger **as oracle fields only** (values are not changed; independent oracle).

  desk_z_oracle       raw desk top z (delivered tree frame) · plane height at the first-frame object xy, median over 40
      sample frames
  oracle_diff_m       desk_z_oracle − desk_z
  support_status      |diff| > ORACLE_WARN_M → "warn" (+why). The training-candidate exclusion threshold (ORACLE_DROP_M)
      is applied by dexengine.data.candidates reading this field.
  desk_footprint_raw  xy convex hull of the real table top (base frame), per take (independent of windows); base copies
      it into the tree's support.
D14′ wrist (below the desk inside the table outline) is measured **per tree (piece) on its window**: the case stage
(cases.taxonomy, MANO wrist) calls the owning function `wrist_footprint` per tree. The take ledger holds no wrist values
(so sibling pieces' wrists never mix).
Reads: data/grab/raw/grab__s*.zip (+ table.ply from tools__object_meshes__contact_meshes.zip) · base trees (z shift =
median raw−tree over the first 10 % of frames)
Usage: $RETARGET_PY dexengine/data/support/grab.py [--write-intake] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import InputMissingError
from dexengine.data import rules as RR  # noqa: E402  — owner of the hand axis

# The hand (spec directory) is read **at run time** (inside main). Reading it at module top level would make even
# dexengine.data.candidates, which imports only this file's constants (ORACLE_DROP_M · ORACLE_FIELDS), fail to import
# without DEXENGINE_HAND, and with it the environment's hand would silently leak in.
ORACLE_WARN_M = 0.01
# D14' footprint margin (m): up to 10 cm inside the table edge counts as "on the desk" (conservatively includes arms
# brushing the edge)
FP_MARGIN = 0.10


def _in_poly(pt, poly, margin=0.0):
    """Whether a point lies inside the convex polygon (+margin): inside, or within margin of an edge."""
    n = len(poly)
    inside = True
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cr = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cr < 0:
            inside = False
    if inside:
        return True
    if margin <= 0:
        return False
    d = min(_seg_dist(pt, poly[i], poly[(i + 1) % n]) for i in range(n))
    return d <= margin


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


#: Digits of the ledger footprint (m, 4th decimal = 0.1 mm). The hull is computed from points **rounded** to these
#: digits (footprint_hull).
FP_DECIMALS = 4


def footprint_hull(xy) -> np.ndarray:
    """(K,2) table outline: counter-clockwise · strictly convex · no duplicates. Hull vertices are rounded to the stored
    digits and **the hull of those points is computed again**. Stopping after rounding turns two nearly coincident
    vertices into a 0.1 mm (or zero-length) edge whose direction is arbitrary, and `_in_poly` (left half-plane per edge)
    then reads points in the middle of the table as outside (measured: 473 of 1,317 outlines written that way had a
    flipped short edge). The recomputed hull keeps no such vertices."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    pts = np.asarray(xy, float)[:, :2]
    pts = np.unique(np.round(pts[ConvexHull(pts).vertices], FP_DECIMALS), axis=0)
    return pts[ConvexHull(pts).vertices]


def convex_ccw_or_die(poly) -> np.ndarray:
    """Check the premise of `_in_poly` (counter-clockwise · strictly convex · no zero-length edge); a violation raises
    ValueError (a wrist inside the outline is never silently read as outside). Outlines written in the old order (hull →
    round) fail here; rerun the oracle."""
    p = np.asarray(poly, float)
    e = np.roll(p, -1, axis=0) - p
    f = np.roll(e, -1, axis=0)
    turn = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
    if len(p) < 3 or (np.einsum("ij,ij->i", e, e) == 0).any() or (turn <= 0).any():
        raise ValueError(
            f"[grab_oracle] footprint is not a counter-clockwise strictly convex polygon"
            f"(turns {np.round(turn, 8).tolist()}) — "
            "rewrite the outline with dexengine/data/support/grab.py --write-intake (footprint_hull)"
        )
    return p


# exclusion threshold read by dexengine.data.candidates, owned here (D10-b: everything > 2 cm is doorknob 19, sequences
# not starting on the desk)
ORACLE_DROP_M = 0.02

#: Frame in which the D14' wrist is measured: the **human hand (MANO)**, since it must be hand-independent (D39: "wuji
#: is
#: no hand's base"). This frame, not a value, gives the hand independence.
#: There is **no** robot-frame path in this file, neither through arguments nor through env (no silent substitute).
WRIST_FRAME = "mano"

#: Ledger fields **owned** by this file, per table (take) and independent of windows. base_adapter copies them into the
#: tree's support (SUPPORT_COPIED) and base_layer.stale.support_drift compares them with the ledger.
ORACLE_FIELDS = ("desk_z_oracle", "oracle_diff_m", "oracle_source", "oracle_dxy_m", "desk_footprint_raw")
#: D14′ wrist fields, measured on the tree window (`wrist_footprint`). The same names in old ledgers (union over all
#: segments) are removed on rerun.
WRIST_FIELDS = ("wrist_inside_fp_frac", "wrist_below_fp_m", "wrist_below_any_m")


def wrist_footprint(wrist, hull, desk_z: float) -> dict:
    """D14′: MANO wrists (N,3) of one tree window (base frame, sides concatenated) · table outline hull (K,2) · desk
    height → {wrist_inside_fp_frac, wrist_below_fp_m (lowest wrist inside outline+FP_MARGIN − desk; None if never
    inside), wrist_below_any_m}.
    An infinite desk model would also count hands lowered **beside** the table as 'wrist below'; only the outline
    counts."""
    wr, hull = np.asarray(wrist, float).reshape(-1, 3), convex_ccw_or_die(hull)
    inside = np.array([_in_poly(pt[:2], hull, FP_MARGIN) for pt in wr], bool)
    dz = wr[:, 2] - float(desk_z)
    return {
        "wrist_inside_fp_frac": round(float(inside.mean()), 3) if len(wr) else None,
        "wrist_below_fp_m": round(float(dz[inside].min()), 4) if inside.any() else None,
        "wrist_below_any_m": round(float(dz.min()), 4) if len(wr) else None,
    }


def table_mesh(raw_dir: Path):
    # heavy dependency inside the function, so dexengine.data.candidates importing only constants (ORACLE_DROP_M) stays
    # light
    import trimesh

    with zipfile.ZipFile(raw_dir / "tools__object_meshes__contact_meshes.zip") as z:
        import io

        m = trimesh.load(
            io.BytesIO(z.read("contact_meshes/table.ply")), file_type="ply", process=False
        )  # read as bytes, without a path resolver
    return np.asarray(
        m.vertices, float
    )  # mesh y = thickness axis (±2.7 mm); which face is up depends on the pose (table_world)


def raw_clip_id(member: str) -> str:
    """Source zip member (`s10/alarmclock_lift_Retake.npz`) → ledger clip id (`s10__alarmclock_lift_retake`).
    Ledger ids are lower case; among source names only the `_Retake` takes carry capitals. Without matching case those
    takes would drop out without an oracle (no outline, no desk height). A set mismatch makes main fail (source ↔ ledger
    difference 0). Names go through the normalisation owner (paths.canonical_clip); ledger keys use the same function
    for the join."""
    return P.canonical_clip(f"{member.split('/')[0]}__{Path(member).stem}".lower(), "grab")


def ledger_clips(sa: dict) -> dict:
    """Clip fields of INTAKE support_audit.json → {canonical clip id: its record (the same object; edits stay in sa)}.
    Two keys colliding after normalisation fail (the join cannot count one clip twice)."""
    out: dict = {}
    for k, rec in sa["clips"].items():
        c = P.canonical_clip(k, "grab")
        if c in out:
            raise SystemExit(
                f"[grab_oracle] ledger clip {k!r} collides with another key after normalisation ({c}) — not writing"
            )
        out[c] = rec
    return out


#: Frames whose table thickness axis (mesh y) tilts more than this from vertical are left out of the desk height
#: samples,
#: so frames where the flat board's pose estimate wobbled (measured: some frames of one of 1,335 takes at 49°) do not
#: distort the 'top' height.
TABLE_TILT_MAX_DEG = 30.0


def table_world(V, rotvec, transl):
    """(world coordinates of the table vertices, top-face vertices or None). Rotation convention from the official GRAB
    `tools/objectmodel.py:95-97` (github.com/otaheri/GRAB): `vertices = v_template @ batch_rodrigues(global_orient) +
    transl` (row vector × R = Rᵀ·v; the official viewer `examples/visualize_grab.py:85-91` draws the table with this
    ObjectModel). Top = the face whose normal points up in the world, chosen by the z sign of the world direction of
    mesh +y (the second row of R). The board is flat and symmetric, so the source pose is recorded flipped in some takes
    and frames (measured: 142 takes flipped on every frame · 175 mixed per frame); fixing 'mesh y>0 = top' would read
    the underside (−5.4 mm) on those frames. None when the thickness axis tilts more than TABLE_TILT_MAX_DEG."""
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    rm = R.from_rotvec(rotvec).as_matrix()
    W = np.asarray(V, float) @ rm + np.asarray(transl, float)
    up = float(rm[1, 2])  # world z component of mesh +y (world = Rᵀ·v → second row of R)
    if abs(up) < np.cos(np.radians(TABLE_TILT_MAX_DEG)):
        return W, None
    return W, W[(V[:, 1] > 0) if up > 0 else (V[:, 1] < 0)]


def _parse_args() -> argparse.Namespace:
    """Command line: --write-intake, --limit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-intake", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def _tree_for(base: Path, clip: str, S: dict) -> Path | None:
    """The base tree to align with: the clip's own tree, else its first segment (None = no tree, counted in S)."""
    tree = base / clip
    # when the parent clip was split into segments the trees are <clip>_segNN; the first segment is the reference
    # tree for table alignment
    segs = sorted(base.glob(f"{clip}_seg[0-9][0-9]"))
    if not (tree / "0/trajectory_kinematic.npz").exists():
        if not segs or not (segs[0] / "0/trajectory_kinematic.npz").exists():
            S["no_tree"] += 1
            return None
        tree = segs[0]
    return tree


def _table_tops(V, tr, ao, ot, S: dict) -> tuple[list, np.ndarray | None]:
    """Desk top height under the object's first position on ~40 sampled frames (tilted frames are skipped and counted)
    and the first flat top (footprint source)."""
    tops, top0 = [], None
    for fr in range(0, len(tr), max(1, len(tr) // 40)):
        _, top = table_world(V, ao[fr], tr[fr])
        if top is None:
            S["tilted_frames"] += 1
            continue
        top0 = top if top0 is None else top0  # footprint = first flat sample frame
        c = np.linalg.lstsq(np.c_[top[:, 0], top[:, 1], np.ones(len(top))], top[:, 2], rcond=None)[0]
        tops.append(c[0] * ot[0, 0] + c[1] * ot[0, 1] + c[2])
    return tops, top0


def _record_footprint(rec: dict, ti: dict, ot, idx, Q, s: int, top0, R) -> None:
    """D14': real table footprint (xy convex hull of the top, tree frame) and the raw→tree xy offset in rec.

    wrist_below_fp_m counts only "wrist below the desk inside the footprint (+FP_MARGIN)": an infinite desk model would
    create phantom 'wrist below' readings for hands lowered **beside** the table (0.4–0.7 m from its centre).
    """
    # base layer = Rz(yaw)·capture (D5); raw xy is turned by the same yaw
    psi = float(((ti.get("frame") or {}).get("yaw_deg")) or 0.0)
    Ryw = R.from_euler("z", psi, degrees=True).as_matrix()[:2, :2]
    # raw→tree xy offset after yaw (should be near 0; any remainder is corrected and recorded)
    dxy = np.median(ot[idx, :2] @ Ryw.T - Q[:, s : s + 2], axis=0)
    hull = footprint_hull(top0[:, :2] @ Ryw.T - dxy[None, :])  # hull → round → hull again
    rec["desk_footprint_raw"] = hull.tolist()
    rec["oracle_dxy_m"] = np.round(dxy, 4).tolist()
    for k in WRIST_FIELDS:
        rec.pop(k, None)


def _flag_diff(rec: dict, diff: float, S: dict) -> None:
    """Warn when the raw desk top differs from the rest estimate by more than ORACLE_WARN_M; count drops."""
    if abs(diff) > ORACLE_WARN_M:
        rec["support_status"] = "warn"
        S["warn"] += 1
        rec["why"] = f"raw desk top differs from the rest estimate desk_z by {diff * 100:+.1f} cm" + (
            " — looks like a sequence that does not start on the desk (over the exclusion threshold)"
            if abs(diff) > ORACLE_DROP_M
            else ""
        )
        if abs(diff) > ORACLE_DROP_M:
            S["drop"] += 1


def _oracle_clip(zf, nm: str, clip: str, tree: Path, clips: dict, V, S: dict, R) -> float:
    """Desk height of one clip from the raw table pose + mesh, written into its ledger record; returns raw − rest."""
    import io

    # npz loads lazily, so the zip member is read as bytes first
    d = np.load(io.BytesIO(zf.read(nm)), allow_pickle=True)
    ot = np.asarray(d["object"].item()["params"]["transl"])
    tp = d["table"].item()["params"]
    tr, ao = np.asarray(tp["transl"]), np.asarray(tp["global_orient"])
    ti = json.loads((tree / "task_info.json").read_text())
    Q = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    T = len(Q)
    s = ti["objects"][0]["qpos_start"]
    oz = Q[:, s + 2]
    idx = np.linspace(0, len(ot) - 1, T).round().astype(int)
    dz = ot[idx, 2] - oz
    shift = float(np.median(dz[: max(5, T // 10)]))
    tops, top0 = _table_tops(V, tr, ao, ot, S)
    if not tops:
        raise InputMissingError(
            f"[grab_oracle] {clip}: no flat table frame (thickness axis > "
            f"{TABLE_TILT_MAX_DEG}° everywhere) — the desk height cannot be measured"
        )
    oracle = float(np.median(tops)) - shift
    diff = oracle - float(clips[clip]["desk_z"])
    rec = clips[clip]
    rec["desk_z_oracle"] = round(oracle, 4)
    rec["oracle_diff_m"] = round(diff, 4)
    rec["oracle_source"] = "grab raw table.params+table.ply"
    _record_footprint(rec, ti, ot, idx, Q, s, top0, R)
    _flag_diff(rec, diff, S)
    return diff


def _summary(diffs: list, S: dict) -> dict:
    """Oracle summary stored in support_audit.json: diff percentiles and the warn/drop counts."""
    dd = np.array(diffs)
    summ = {
        "tool": "dexengine/data/support/grab.py",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n": int(len(dd)),
        "diff_p5": round(float(np.percentile(dd, 5)), 4),
        "diff_p50": round(float(np.median(dd)), 4),
        "diff_p95": round(float(np.percentile(dd, 95)), 4),
        "warn_gt_m": ORACLE_WARN_M,
        "n_warn": S["warn"],
        "drop_gt_m": ORACLE_DROP_M,
        "n_drop": S["drop"],
        "counts": S,
    }
    return summ


def _scan_raw(a, raw_dir: Path, clips: dict, base: Path, V, S: dict, R) -> tuple[list, set, list]:
    """Oracle every raw GRAB member that has a ledger clip and a base tree (first --limit clips when set); returns the
    diffs, the clips seen and the raw members without a ledger clip."""
    done = 0
    diffs = []
    seen, raw_only = set(), []
    for z in sorted(raw_dir.glob("grab__s*.zip")):
        zf = zipfile.ZipFile(z)
        for nm in zf.namelist():
            if not nm.endswith(".npz"):
                continue
            clip = raw_clip_id(nm)
            if clip not in clips:
                S["no_clip"] += 1
                raw_only.append(nm)
                continue
            seen.add(clip)
            tree = _tree_for(base, clip, S)
            if tree is None:
                continue
            diff = _oracle_clip(zf, nm, clip, tree, clips, V, S, R)
            diffs.append(diff)
            S["ok"] += 1
            done += 1
            if a.limit and done >= a.limit:
                break
        if a.limit and done >= a.limit:
            break
    return diffs, seen, raw_only


def main() -> int:
    a = _parse_args()
    t0 = time.time()
    HAND = RR.hand_dir()  # running hand (DEXENGINE_HAND); data.rules fails loudly without it
    print(
        f"[grab_oracle] per-table fields {ORACLE_FIELDS} — the D14′ wrist ({WRIST_FRAME}) is measured per "
        "tree by the case stage with wrist_footprint. "
        f"The hand ({HAND}) enters only the desk and object z",
        flush=True,
    )
    from scipy.spatial.transform import Rotation as R

    raw_dir = P.layer_path("raw", "grab")
    idir = P.layer_path("intake", "grab")
    sa_p = idir / "support_audit.json"
    sa = json.loads(sa_p.read_text())
    clips = ledger_clips(sa)
    V = table_mesh(raw_dir)
    base = P.layer_path("base", "grab", hand=HAND, emb="bimanual")
    S = {"ok": 0, "no_clip": 0, "no_tree": 0, "warn": 0, "drop": 0, "tilted_frames": 0}
    diffs, seen, raw_only = _scan_raw(a, raw_dir, clips, base, V, S, R)
    if not a.limit and (raw_only or seen != set(clips)):
        raise SystemExit(
            f"[grab_oracle] source ↔ ledger clip sets differ — source only {len(raw_only)} {raw_only[:4]} · "
            f"ledger only {len(set(clips) - seen)} {sorted(set(clips) - seen)[:4]} "
            "(raw_clip_id rule) — not writing"
        )
    summ = _summary(diffs, S)
    print(
        f"[grab_oracle] {S} · diff p5/p50/p95 = {summ['diff_p5']:+.4f}/{summ['diff_p50']:+.4f}/{summ['diff_p95']:+.4f} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    if a.write_intake:
        sa["oracle"] = summ
        tmp = sa_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sa, ensure_ascii=False, indent=1))
        tmp.replace(sa_p)
        print(f"[grab_oracle] INTAKE written → {sa_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
