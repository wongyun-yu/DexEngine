"""Articulated **asset** spec: one canonical joint and the USD joint limits per asset, derived from every clip of the
base layer (D1-j (a) · D1-g).

Why: the USD and surface points exist **once per asset** (asset id = artifact_paths.object_asset_id: the object id; for
paired articulations object, moving part and kind), while a tree's `objects[].articulation` states the joint **per
clip**.
OakInk2 (two rigid bodies → articulation, D1-g) takes each tree's own first frame as joint 0, so child_pose, axis,
anchor and range differ per tree. Hence the spec:
  canonical joint = the statement of the tree that **reproduces the most asset trees within the D1-g thresholds** (ties:
              smaller max error, then clip name): kind · axis · anchor · child_pose · parts.
  reexpress  = tree joint values α in canonical coordinates: θ = s·α + θ_off (s = sign of the base-frame axis · θ_off =
              the angle/offset matching the tree's first-frame pose). It fits when the moving body origin's position and
              rotation lie inside the kind's band (oakink2.artic_fit.band; hinge = D1-g 1 cm · 5° · slide = rail play
              band), the same ruler as the converter FK oracle.
  2π rule    = a hinge θ_off is the same pose every 2π. reexpress alone picks the representative: the tree interval's
              midpoint closest to the window midpoint. Window = the **minimum-width window** holding the fitting trees,
              each shifted by 2πk (hinge_window · midpoint ∈ (−π, π]); independent of which tree is canonical, the
              quaternion sign and which side of ±π (depending on those inflates the limits by 2π). The converter (G5)
              reads the same θ_off via reexpress(spec[asset], tree, (spec lo, spec hi)).
  limits     = union of the canonical-coordinate intervals of the fitting trees placed by that rule (D1-j (a): the
              asset's max opening over its clips).
  n_restate  = number of trees to rewrite (needs_restate: different statement or (s, θ_off) ≠ (1, 0)) · misfit = trees
              outside the thresholds. Rewriting trees to the canonical statement is the converter's job
              (oakink2.artic_convert); trees that do not fit are not articulated (D1-g: they stay two rigid bodies).
  FK oracle  = before output, per tree on the output dict: artic_moving_vertices(spec, s·α + θ_off) ≈
              artic_moving_vertices(tree, α) (fk_oracle; canonical tree 1e-6 m · fit = D1-g), which the bake readback
              and spawn smoke tests, comparing specs only, cannot see.
The USD and surface points are functions of the canonical statement and limits alone: rewriting trees does not change
them (fixed point: rewritten trees all carry the canonical statement with θ_off 0 and the same intervals → same window,
same limits), so the bake does not wait for tree restatement. Whether a tree's joint values match the USD coordinates is
checked per tree by the consumers (trainability, entry) with needs_restate.
The FK convention belongs to coords.artic_moving_vertices (body pose = image of the origin and three unit axes).

Update (asset_spec prev): published assets keep their statement **fixed** (re-deriving it would make already converted
      trees non-canonical) and the window = published window ∪ fitting tree intervals, never shrinking (arctic). OakInk2
      is measured from the spec file's source record section (hand-independent, measured once and fixed;
      dexengine.data.artic.range_src.derive, keeping published statements by keeps_published; the window never shrinks).
      Input collection, merging and writing are done by the stage dexengine.data.artic.spec_stage (update).
Output: `artic_ranges.json` in the audit folder of the axis hand (AXIS_HAND, owner of shared outputs;
      data.rules.axis_output); consumers (converter · objects.usd_bake · artic.surface · trainability) read that one
          file.
Usage: (write the spec) python3 dexengine/data/artic/spec_stage.py --dataset oakink2 [--dry-run]
       (training-entry child) python3 dexengine/data/artic/ranges.py --not-canonical-stdin <ds>
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import artifact_paths
from dexengine.data.artic import (
    limits as AL,  # noqa: E402 — owner of the physical joint ranges (effective limits, clamping)
)
from dexengine.data.artic.fk import artic_moving_vertices
from dexengine.data.artic.limits import (  # noqa: E402,F401 — re-export (the converter reads it through this module)
    clamp,
    mark,
)
from dexengine.data.oakink2.artic_fit import (  # noqa: E402, F401 — owner of the D1-g thresholds and per-kind bands
    POS_TOL_M,
    ROT_TOL_DEG,
    band,
    in_band,
)

HELP = (
    "Per articulated asset: canonical joint · USD joint limits (max opening over all clips, D1-j (a)) · trees to "
    "rewrite."
)
#: Canonical statement: equal fields mean the same joint (same zero, axis, parts). Reexpression then leaves s = 1 and,
#: for hinges, only a 2πk shift (needs_restate). Part roles/sides (`parts.*_role`) are facts of the tree, not statement.
STATEMENT = ("type", "axis", "pos", "child_pose", "parts")
#: fixed = fixed attachments (id · mesh · link · pose; part of the statement)
_PART_KEYS = ("base", "moving", "base_id", "moving_id", "fixed")
_FRAME = np.vstack([np.zeros(3), np.eye(3)])  # body origin + unit axes: the four points whose FK image gives the pose
TWO_PI = 2.0 * math.pi
WRAP_TOL = 1e-9  # numerical slack (rad) for 2π window boundaries and ties


def body_pose(art: dict, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame pose of the moving body (R 3×3, origin 3), read from the image under the FK owner."""
    v = artic_moving_vertices(art, _FRAME, float(theta))
    return (v[1:] - v[0]).T, v[0]


def base_frame_joint(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """(axis, anchor) in the base frame: the spec's axis and anchor (child frame) moved through the FK owner's image:
    anchor = image of the anchor point (fixed while the joint moves) · axis = image of anchor+axis − image of the
    anchor."""
    anc = np.asarray(spec.get("pos") or [0, 0, 0], np.float64)
    ax = np.asarray(spec["axis"], np.float64)
    ax = ax / np.linalg.norm(ax)
    a0, a1 = artic_moving_vertices(spec, np.stack([anc, anc + ax]), 0.0)
    d = a1 - a0
    return d / np.linalg.norm(d), a0


def _base_axis(art: dict) -> np.ndarray:
    r0, o0 = body_pose(art, 0.0)
    r1, o1 = body_pose(art, 1e-3)
    if art["type"] == "hinge":
        w = Rot.from_matrix(r1 @ r0.T).as_rotvec()
        return w / np.linalg.norm(w)
    return (o1 - o0) / np.linalg.norm(o1 - o0)


def _samples(art: dict) -> list[float]:
    """D1-g comparison samples α: 5 points over the tree range ∪ {0} (shared by reexpress and the FK oracle)."""
    lo, hi = (float(x) for x in art["range"])
    return sorted(set(np.linspace(lo, hi, 5).tolist()) | {0.0})


def _relate(canon: dict, art: dict) -> dict:
    """Tree joint `art` ↔ canonical `canon`: axis sign s · first-frame offset off (hinges before the 2π representative:
    [−π, π]; the relative rotation quaternion sign is fixed to w ≥ 0 for bitwise determinism) · body origin position
    error
    (m) and rotation error (°) over the range · D1-g fit. Fit and errors do not depend on 2π (the pose is 2π periodic).
    A different kind fails (the asset id holds the kind). The ValueError text reaches the exclusion ledger (through
    canon_joint and spec_lookup.rest_joints) and stays as written."""
    if canon["type"] != art["type"]:
        raise ValueError(f"joint type {art['type']} ≠ canonical {canon['type']} — the asset id must carry the type")
    a_c = _base_axis(canon)
    s = 1.0 if float(a_c @ _base_axis(art)) >= 0 else -1.0
    rc, oc = body_pose(canon, 0.0)
    rt, ot = body_pose(art, 0.0)
    if art["type"] == "hinge":
        q = Rot.from_matrix(rc.T @ rt).as_quat()  # xyzw · rotation from canonical zero to tree zero
        k = rc.T @ a_c  # axis of that rotation (canonical body frame)
        if q[3] < 0 or (q[3] == 0 and float(q[:3] @ k) < 0):
            q = -q  # q ≡ −q; one representative (off ∈ [−π, π])
        off = 2.0 * math.atan2(float(q[:3] @ k), float(q[3]))
    else:
        off = float(a_c @ (ot - oc))
    dp = dr = 0.0
    for al in _samples(art):
        r1, o1 = body_pose(art, al)
        r2, o2 = body_pose(canon, s * al + off)
        dp = max(dp, float(np.linalg.norm(o1 - o2)))
        dr = max(dr, float(np.degrees(Rot.from_matrix(r1.T @ r2).magnitude())))
    return {"s": s, "off": off, "pos_err_m": dp, "rot_err_deg": dr, "fits": in_band(canon["type"], dp, dr)}


def _interval(art: dict, s: float, off: float) -> tuple[float, float]:
    """Canonical-coordinate interval of the tree range; same formula as the converter's joint values θ = s·α + off (same
    floats)."""
    lo, hi = sorted(s * float(x) + off for x in art["range"])
    return lo, hi


def hinge_window(ivs: list[tuple[float, float]]) -> tuple[float, float]:
    """**Minimum-width window** (lo, hi) holding the hinge tree intervals (canonical coordinates, any 2π
    representative),
    each shifted by 2πk: the 2π rule of the limits. The window's left end is some tree's lo and the rest take their
    first
    position to its right (a minimum-width window always has this form) · ties = midpoint closer to 0 · the window is
    shifted by 2π so its midpoint ∈ (−π, π] (independent of which tree is canonical; solving again with restated trees
    gives the same window)."""
    best = None
    for left, _ in ivs:
        right = max(h + TWO_PI * math.ceil((left - lo) / TWO_PI - WRAP_TOL) for lo, h in ivs)
        shift = TWO_PI * math.ceil(((left + right) / 2 - math.pi) / TWO_PI)
        key = (round(right - left, 9), abs((left + right) / 2 - shift), (left + right) / 2 - shift)
        if best is None or key < best[0]:
            best = (key, (left - shift, right - shift))
    return best[1]


def reexpress(canon: dict, art: dict, window: tuple[float, float]) -> dict:
    """Tree joint `art` in the coordinates of canonical `canon`: θ = s·α + θ_off. **Owner of the hinge 2π
    representative**:
    the tree interval's midpoint closest to the window midpoint (k = round(Δ/2π)); window = spec (lo, hi) (bake limits
    and
    converter qpos read the same window). A fitting tree landing outside the window fails (the spec does not cover that
    tree: a stale spec; this ValueError text reaches the exclusion ledger and stays as written). + errors · fit
    (_relate).
    When the canonical joint has a physical range (phys), the interval is clipped into the effective limits
    (dexengine.data.artic.limits.clip_iv; the converter clamps to them) and phys is returned too."""
    r = _relate(canon, art)
    lo, hi = _interval(art, r["s"], r["off"])
    k = round(((window[0] + window[1]) - (lo + hi)) / (2 * TWO_PI)) if art["type"] == "hinge" else 0
    off = r["off"] + TWO_PI * k
    iv = AL.clip_iv(_interval(art, r["s"], off), canon.get("phys"))
    if r["fits"] and (iv[0] < window[0] - WRAP_TOL or iv[1] > window[1] + WRAP_TOL):
        raise ValueError(
            f"matching tree interval {iv} is outside the window {window} — the spec does not cover this tree "
            "(stale spec)"
        )
    return {**r, "theta_off": off, "interval": iv, "phys": canon.get("phys")}


_SPEC: dict = {}


def load_spec() -> dict:
    """Asset table {"<dataset>/<asset id>": entry} of the committed axis-hand spec (location =
    data.rules.axis_output(ARTIC_SPEC)),
    the one table consumers (trainability, training entry) read. Read once per file (path, mtime)."""
    from dexengine.data import rules as RR  # noqa: PLC0415

    f = RR.axis_output(RR.ARTIC_SPEC)
    key = (str(f), f.stat().st_mtime_ns)
    if key not in _SPEC:
        _SPEC.clear()
        _SPEC[key] = json.loads(f.read_text(encoding="utf-8"))["objects"]
    return _SPEC[key]


def not_canonical(ds: str, obj_id: str, art: dict, spec: dict | None = None) -> str:
    """Why an articulated tree cannot be trained ('' = trainable): its asset (artifact_paths.object_asset_id) must be in
    the
    spec and needs_restate(spec, art, reexpress(spec, art, (spec lo, spec hi))) must be false. Trees with only the
    statement equal and θ_off = 2πk, trees outside the window (reexpress ValueError) and trees of another kind are
    caught
    too. Trainability (dexengine.placement.train_ready T3 `artic_not_canonical`) and the training entry
    (dexengine/rl/cli.py, fatal) call this one function."""
    asset = f"{ds}/{artifact_paths.object_asset_id(obj_id, art)}"
    sp = (load_spec() if spec is None else spec).get(asset)
    if sp is None:
        return f"{asset}: asset not in the spec (artic_ranges.json)"
    try:
        rx = reexpress(sp, art, (sp["lo"], sp["hi"]))
    except ValueError as e:
        return f"{asset}: cannot move to canonical coordinates — {e}"
    if needs_restate(sp, art, rx):
        return f"{asset}: not the canonical joint (statement, or s={rx['s']:+.0f} · θ_off={rx['theta_off']:.4f})"
    return ""


def statement(art: dict) -> dict:
    """Joint statement (STATEMENT); parts as paths and ids only."""
    out = {k: art.get(k) for k in STATEMENT}
    out["parts"] = {k: v for k, v in (art.get("parts") or {}).items() if k in _PART_KEYS}
    return out


def needs_restate(canon: dict, art: dict, rx: dict) -> bool:
    """Whether the tree must be rewritten: the statement differs from the canonical one or the joint-value coordinates
    differ ((s, θ_off) ≠ (1, 0); a hinge with the same statement may still carry a 2πk shift). Consumers (trainability)
    read this predicate too; checking statement equality alone would pass trees whose α lies outside the limits."""
    return statement(art) != statement(canon) or (rx["s"], rx["theta_off"]) != (1.0, 0.0)


def canonical(recs: list[tuple[str, dict]]) -> tuple[str, dict, dict]:
    """(canonical clip, canonical joint, {clip: _relate}): the tree reproducing the most trees within the thresholds
    (ties = max error, then clip order). Independent of 2π. One candidate per distinct statement (equal statements give
    equal results)."""
    best, seen = None, set()
    for clip, cand in sorted(recs, key=lambda r: r[0]):
        sig = json.dumps(statement(cand), sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        rel = {c: _relate(cand, a) for c, a in recs}
        fit = [r for r in rel.values() if r["fits"]]
        key = (-len(fit), max(r["pos_err_m"] for r in fit), clip)
        if best is None or key < best[0]:
            best = (key, clip, cand, rel)
    return best[1], best[2], best[3]


def _fk_diff(spec: dict, art: dict, win: tuple[float, float]) -> tuple[dict, float, float, float]:
    """Move tree `art` into spec coordinates (reexpress) and take, over the sample α, the max of body origin difference
    (m) · rotation difference (°) · max difference of the four points (m)."""
    rx = reexpress(spec, art, win)
    dp = dr = d4 = 0.0
    for al in _samples(art):
        vt = artic_moving_vertices(art, _FRAME, al)
        vs = artic_moving_vertices(spec, _FRAME, rx["s"] * al + rx["theta_off"])
        dp = max(dp, float(np.linalg.norm(vt[0] - vs[0])))
        dr = max(dr, float(np.degrees(Rot.from_matrix((vt[1:] - vt[0]) @ (vs[1:] - vs[0]).T).magnitude())))
        d4 = max(d4, float(np.abs(vt - vs).max()))
    return rx, dp, dr, d4


def fk_oracle(
    spec: dict, recs: list[tuple[str, dict]], floor: tuple[float, float] | None = None, canon_art: dict | None = None
) -> list[str]:
    """Output spec ↔ tree FK oracle on **the dict** the bake and converter read: per tree at θ = s·α + θ_off
    (reexpress(spec, tree, (lo, hi)), the converter path), the body origin and rotation difference between
    artic_moving_vertices(spec, θ) and artic_moving_vertices(tree, α) · fit (D1-g) = outside the spec misfit list ·
    the canonical tree within 1e-6 m on the four points (statement as is) · min/max of the fitting intervals placed in
    that
    window = lo/hi (the converter's qpos fill the baked limits exactly) · hinge limits locally minimal (shifting one
    fitting tree by ±2π must not shrink the width, i.e. no 2π inflation, which the 2π-periodic FK pose check cannot
    see).
    Returns the problems (empty = pass).
    `canon_art` = the statement when the canonical statement is a fixed value instead of a tree record
    (dexengine.data.artic.range_src's
    kept assets); the 1e-6 canonical check compares against it, and a tree record of the same name is measured by band
    only."""
    bad, win, placed = [], (spec["lo"], spec["hi"]), []
    if canon_art is not None:
        d4 = _fk_diff(spec, canon_art, win)[3]
        if d4 > 1e-6:
            bad.append(
                f"{spec['ref_clip']}(fixed value): spec dict FK difference {d4:.2e} m > 1e-6 — the spec does not carry "
                "the fixed statement as is"
            )
    for c, art in recs:
        rx, dp, dr, d4 = _fk_diff(spec, art, win)
        fits = in_band(spec["type"], dp, dr)  # same band as _relate (compared with the spec misfit list)
        if fits != (c not in spec["misfit"]):
            bad.append(
                f"{c}: spec dict FK difference {dp * 1e3:.2f} mm · {dr:.2f}° — fit {fits} ≠ spec "
                f"{c not in spec['misfit']}"
            )
        if canon_art is None and c == spec["ref_clip"] and d4 > 1e-6:
            bad.append(
                f"{c}(canonical): spec dict FK difference {d4:.2e} m > 1e-6 — the spec does not carry the "
                "canonical statement as is"
            )
        if fits:
            placed.append(rx["interval"])
    if floor is not None:  # published window (update; those trees may be missing from this collection)
        placed.append(tuple(floor))
    if placed and max(abs(min(p[0] for p in placed) - win[0]), abs(max(p[1] for p in placed) - win[1])) > WRAP_TOL:
        bad.append(f"union of placed intervals ≠ spec limits {win}")
    for i, (lo, hi) in enumerate(placed if spec["type"] == "hinge" else []):
        rest = placed[:i] + placed[i + 1 :]
        for d in (-TWO_PI, TWO_PI):
            w = max([hi + d] + [p[1] for p in rest]) - min([lo + d] + [p[0] for p in rest])
            if w < win[1] - win[0] - WRAP_TOL:
                bad.append(
                    f"limits inflated by 2π: shifting interval {(lo, hi)} by {d:+.4f} changes the width "
                    f"{win[1] - win[0]:.4f} → {w:.4f}"
                )
    return bad


def _pose_dev(p: dict, q: dict) -> tuple[float, float]:
    """Difference of two link-frame poses {pos, quat_wxyz}: (position m, rotation °)."""
    dp = float(np.linalg.norm(np.asarray(p["pos"], float) - np.asarray(q["pos"], float)))
    ra, rb = (Rot.from_quat(np.asarray(x["quat_wxyz"], float)[[1, 2, 3, 0]]) for x in (p, q))
    return dp, float(np.degrees((ra.inv() * rb).magnitude()))


def fixed_medoid(fixed: list[dict], arts: list[dict]) -> list[dict]:
    """Canonical fixed attachment pose: per part, among the fitting trees' poses the one whose max difference to the
    other
    trees' poses (divided by the D1-g thresholds) is smallest (medoid; ties = tree order). Fixed attachments are one
    body
    **within** a clip (1 cm · 5°), but their placement varies a little between clips (microwave lever: 0.2–1.3 cm
    against
    the body), so one canonical tree's pose would push the trees at the other end past the threshold.
    No tree has a part id set different from the canonical one (the asset id holds the fixed ids)."""
    out = []
    for f in fixed:
        cands = [g["pose"] for a in arts for g in (a.get("parts") or {}).get("fixed") or [] if g["id"] == f["id"]]
        score = [
            max(max(dp / POS_TOL_M, dr / ROT_TOL_DEG) for dp, dr in (_pose_dev(c, o) for o in cands)) for c in cands
        ]
        out.append({**f, "pose": cands[int(np.argmin(score))]})
    return out


def asset_spec(
    ds: str, oid: str, asset: str, recs: list[tuple[str, dict]], prev: dict | None = None, fixed_canon: bool = False
) -> dict:
    """Spec of one asset: canonical statement · limits (union of the canonical-coordinate intervals of the fitting trees
    placed by the 2π rule; minimum-width window for hinges) · trees to rewrite (needs_restate) · trees outside the
    thresholds. Before output the FK oracle compares the output dict with the trees (a mismatch fails).
    `prev` = the published entry (update): the statement stays **fixed** (not re-derived; changing it would make every
    converted tree non-canonical) · window = published window ∪ fitting tree intervals (never shrinks; a widened window
    that does not cover the published one fails).
    `fixed_canon` = the canonical statement is the published entry itself (dexengine.data.artic.range_src's kept assets,
    whose source tree is no longer recorded): the 1e-6 canonical check compares against the published statement
    (fk_oracle canon_art)."""
    if prev is None:
        ref, canon, rel = canonical(recs)
    else:
        ref, canon = prev["ref_clip"], prev
        rel = {c: _relate(canon, a) for c, a in recs}
    fit = [(c, a) for c, a in recs if rel[c]["fits"]]
    ivs = [AL.clip_iv(_interval(a, rel[c]["s"], rel[c]["off"]), canon.get("phys")) for c, a in fit]
    floor = None if prev is None else (float(prev["lo"]), float(prev["hi"]))
    if floor is not None:
        ivs.append(floor)
    win = hinge_window(ivs) if canon["type"] == "hinge" else (min(i[0] for i in ivs), max(i[1] for i in ivs))
    rx = {c: reexpress(canon, a, win) for c, a in fit}
    placed = [r["interval"] for r in rx.values()] + ([floor] if floor is not None else [])
    lo, hi = min(p[0] for p in placed), max(p[1] for p in placed)
    if floor is not None and (lo > floor[0] + WRAP_TOL or hi < floor[1] - WRAP_TOL):
        raise ValueError(
            f"{ds}/{asset}: widened window ({lo:.6f}, {hi:.6f}) does not cover the published window {floor} (2π shift)"
        )
    misfit = sorted(c for c in rel if c not in rx)
    restate = sorted(c for c, a in recs if c not in rx or needs_restate(canon, a, rx[c]))
    st = statement(canon)
    if prev is None and st["parts"].get("fixed"):  # fixed attachment pose = medoid of the fitting trees (fixed_medoid)
        st["parts"] = {**st["parts"], "fixed": fixed_medoid(st["parts"]["fixed"], [a for c, a in fit])}
    fitted = list(rx.values())
    out = {
        "dataset": ds,
        "obj_id": oid,
        "asset": asset,
        **st,
        "lo": lo,
        "hi": hi,
        "span_max": hi - lo,
        "n_clips": len(recs),
        "ref_clip": ref,
        "n_fit": len(rx),
        "misfit": misfit,
        "n_restate": len(restate),
        "pos_err_max_m": round(max((r["pos_err_m"] for r in fitted), default=0.0), 6),
        "rot_err_max_deg": round(max((r["rot_err_deg"] for r in fitted), default=0.0), 4),
    }
    if canon.get("phys"):  # a published asset's physical range follows like the statement (not re-measured)
        out["phys"] = canon["phys"]
    canon_art = {**statement(prev), "range": [float(prev["lo"]), float(prev["hi"])]} if fixed_canon else None
    bad = fk_oracle(out, recs, floor, canon_art)
    if bad:
        raise ValueError(f"[ARTIC_FK_ORACLE] {ds}/{asset} output dict ≠ tree FK: {bad[:3]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=HELP)
    ap.add_argument(
        "--not-canonical-stdin",
        metavar="DS",
        default="",
        help="for the training entry: for each JSON [[object id, joint statement], …] on stdin, print the "
        "not_canonical(DS, …) reasons as one JSON list line in the same order ('' = trainable); the "
        "entry does not load numpy before Kit, so this child process decides",
    )
    a = ap.parse_args()
    if a.not_canonical_stdin:
        print(
            json.dumps(
                [not_canonical(a.not_canonical_stdin, o, art) for o, art in json.load(sys.stdin)], ensure_ascii=False
            )
        )
        return 0
    raise SystemExit(
        "[artic_ranges] the spec is written by the stage dexengine/data/artic/spec_stage.py --dataset <ds>"
        " (called by stage 1 step 2b); this file holds the spec rule (asset_spec) and the consumer predicates"
    )


if __name__ == "__main__":
    sys.exit(main())
