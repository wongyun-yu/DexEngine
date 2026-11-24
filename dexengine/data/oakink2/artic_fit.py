#!/usr/bin/env python3
"""OakInk2 part pairs → joint-axis fit and joint-value recovery (the D1-g measurement).

What it measures: for every tree with parts of the same instance under the official part tree (`oakink2.parts`, D1-d;
any role, supplementary parts included), it **clusters the parts by relative motion** (part_clusters, the one rule the
converter reads too). For two bodies it fits whether the motion of moving relative to base is one joint (fixed axis)
by least squares and measures the **residual**. Measurement and conversion use the same functions on the same frames
(no extraction).

Model: p_rel(t) = c + R(n̂, α(t))·(p_rel(0) − c) + h·α(t)·n̂ ,  R_rel(t) = R(n̂, α(t))·R_rel(0)
  n̂ = principal direction of the rotation vectors v(t)=log(R_rel(t)R_rel(0)^T) (|θ|-weighted SVD) · α(t) = v(t)·n̂ ·
  c·h = linear least squares.
Classes (D1-g thresholds: rotation 5° · translation 1 cm):
  rigid      max |α| < 5° and relative position change < 1 cm: the parts move together (one body, no joint)
  prismatic  max |α| < 5°, line-fit residual < 1 cm, travel ≥ 1 cm
  revolute   rotation residual < 5°, position residual < 1 cm and |h·Δα| < 1 cm
  screw      the same residual conditions + axial travel |h·Δα| ≥ 1 cm
  free       anything else: not one joint (assembly, picking) → two separate rigid bodies as in D1-b
Pairs whose spec is a slide are re-measured with fit_slide: with more than 5° rotation from the first frame the classes
above skip the slide test and take the hinge model, reading a drawer tilting on its rail as free. Bands come from
band(kind): slides use the rail play band (oakink2.part_kinds.SLIDE_PLAY_* · rotation 8° · translation 1.5 cm), hinges
and rigid joints the D1-g thresholds above.
Output: `oakink2_artic_fit_<date>/{fit.tsv,summary.md}` in the active hand's audit folder (data.rules.audit_out;
        per hand, since it reads the hand's base trees).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.data import (
    raw_sample as RS,  # noqa: E402 — owner of the sampling rule and version of supplementary part poses
)
from dexengine.data.oakink2 import (
    attach as AT,  # noqa: E402 — owner of the attached-part measurements (attach pose, lever rule ruler)
)
from dexengine.data.oakink2 import part_kinds as PK  # noqa: E402 — owner of the per-object joint kinds
from dexengine.data.oakink2 import parts as OP  # noqa: E402

ROT_TOL_DEG = 5.0  # D1-g
POS_TOL_M = 0.01  # D1-g
#: Reasons for scenes the part merge rule cannot decide; the converter fails with them.
PART_MERGE_FAIL = (
    "duplicate",
    "multi_moving",
    "multi_fixed_drifts",
    "no_body",
    "raw_time",
    "kind_mismatch",
    *PK.ATTACH_MOVED.values(),
)


class PartMergeError(ValueError):
    """The part merge rule (part_clusters) cannot decide; `code` ∈ PART_MERGE_FAIL."""

    def __init__(self, code: str, msg: str):
        if code not in PART_MERGE_FAIL:
            raise ValueError(f"unknown merge failure reason {code!r}")
        super().__init__(f"{code}: {msg}")
        self.code = code


class PartExcluded(PartMergeError):
    """The spec **judged this tree unconvertible**: the motion disagrees with the spec (kind, attachment, lever) or
    falls
    outside the converter contract (one articulated object and one joint per tree). Unlike broken input (PartMergeError:
    the same part id twice, no body), the verdict is an **explicit exclusion** with a reason word (`code`; the converter
    writes it to the exclusion ledger, stage rc 0), while broken input is an error (rc≠0). The raising site decides
    which;
    no other table sorts the reason words. The message is stored in the exclusion ledger as is."""


def band(kind: str) -> tuple[float, float]:
    """Verification band (position m, rotation °) of a joint kind. Read by the kind check (part_clusters · fit_slide),
    by
    the canonical comparison through in_band (dexengine.data.artic.ranges._relate; converter, spec stage, trainability
    T3 and training entry all use that one function) and by the FK oracle (converter, spec). Slides (prismatic · slide)
    = rail play band (owner oakink2.part_kinds.SLIDE_PLAY_*) · hinges and rigid joints = D1-g thresholds. Unknown kinds
    fail."""
    if kind in ("prismatic", "slide"):
        return PK.SLIDE_PLAY_POS_M, PK.SLIDE_PLAY_ROT_DEG
    if kind in ("revolute", "hinge", "rigid"):
        return POS_TOL_M, ROT_TOL_DEG
    raise ValueError(f"[ARTIC_FIT] unknown joint kind {kind!r} — no verification band")


def in_band(kind: str, pos_m: float, rot_deg: float) -> bool:
    """Whether an error (position m, rotation °) is inside the kind's band; the one check the canonical comparison and
    FK oracle read."""
    pos_tol, rot_tol = band(kind)
    return pos_m <= pos_tol and rot_deg <= rot_tol


def _quat_xyzw(q7: np.ndarray) -> np.ndarray:
    """Tree object block, 7 columns = [xyz, qw, qx, qy, qz] (wxyz) → scipy xyzw."""
    return q7[:, [4, 5, 6, 3]]


def block(q: np.ndarray, o: dict) -> np.ndarray:
    """The 7 free-joint columns of object `o` in the tree qpos."""
    s = int(o["qpos_start"])
    return q[:, s : s + 7]


def path_m(q7: np.ndarray) -> float:
    """Path length (m) of the object origin; the ruler for which body moved less (base)."""
    return float(np.linalg.norm(np.diff(q7[:, :3], axis=0), axis=1).sum())


def rel_pose(qA: np.ndarray, qB: np.ndarray):
    """Pose sequence of body B seen in body A's frame (positions (T,3), rotations Rotation)."""
    RA = Rot.from_quat(_quat_xyzw(qA))
    return RA.inv().apply(qB[:, :3] - qA[:, :3]), RA.inv() * Rot.from_quat(_quat_xyzw(qB))


def joint_values(qA: np.ndarray, qB: np.ndarray, fit: dict) -> np.ndarray:
    """Joint value α(t) of body B relative to body A, measured from the first frame (revolute: angle about the axis in
    rad · otherwise: axial travel in m)."""
    prel, Rrel = rel_pose(qA, qB)
    n = np.asarray(fit["axis"], float)
    if fit["kind"] == "revolute":
        # unwrap the ±π branch into a continuous joint angle (joints that turn far jump within [-π, π])
        return np.unwrap((Rrel * Rrel[0].inv()).as_rotvec() @ n)
    return (prel - prel[0]) @ n


def rigid_together(qA: np.ndarray, qB: np.ndarray) -> bool:
    """Whether two bodies move as one; exactly the rigid test of fit_pair (thresholds not restated). The frame in which
    the relative position is measured can flip the result at the boundary, so rigid **in either direction** counts as
    one
    body; then fit_pair never returns rigid between the representatives of two clusters."""
    return fit_pair(qA, qB)["kind"] == "rigid" or fit_pair(qB, qA)["kind"] == "rigid"


def part_instances(objs: list[dict]) -> dict[str, list[dict]]:
    """Part objects of the same instance under the official part tree (oakink2.parts, D1-d): {instance root: [objects]}.
    Non-part objects are left out."""
    by: dict[str, list[dict]] = {}
    for o in objs:
        rid = OP.rid_of(o["obj_id"])
        if OP.is_part(rid):
            by.setdefault(OP.root_of(rid), []).append(o)
    return by


def part_clusters(objs: list[dict], q: np.ndarray) -> dict:
    """Merge decision for the parts of an instance: the one D1-g pairing rule (read by the measurer main and the
    converter convert_tree). **The per-object joint spec (oakink2.part_kinds) sets the kind; motion only verifies it**
    (so the same object never changes kind between clips).

    Free parts are not merged (ordinary objects). The other parts = the body cluster (body + parts attached fixed or as
    a
    lever) + the jointed-part cluster (jointed part + parts attached fixed to it; at most one per instance, ensured by
    the
    spec check).
    Verification (a mismatch fails, so nothing drops silently into a static slot, two main objects or main-object
    parts):
      fixed · deformable  rigid with the attachment representative (fit_pair; thresholds = D1-g rotation 5° ·
          translation 1 cm)
      lever  tip displacement (distance the lever surface travels from rest) ≤ LEVER_TIP_M and the share of the jointed
             part's motion with the lever turned < LEVER_DRIVE_SHARE (oakink2.attach.lever_fixed); the lever is fixed to
             the attachment representative in its rest pose
      joint  fit_pair kind of body ↔ jointed part ∈ {spec kind, rigid (did not move in this clip)}
    Decisions:
      no_parts      no instance with two or more parts to merge; nothing to do
      fixed_only    no jointed part: one rigidly joined object (body + attachments)
      one_cluster   the jointed part did not move in this clip (fit rigid); the joint comes from the spec asset
          (converter's job)
      two_clusters  the jointed part moved; the fit is this tree's joint
      multi         two or more instances (rule = oakink2.multi_instance)
    The decision carries `levers` = {obj_id · tip_m · drive_share} per fixed lever (record; only when there is a lever).
    Failures (PartMergeError): duplicate · no_body (the body or an attachment parent is missing from the tree) ·
    raw_time
    (a supplementary part posed at another time than the body: the raw_sample version differs; such relative motion is
    not
    measured, and it is a defect to fix, so an error, not an explicit exclusion) · kind_mismatch (the joint motion
    disagrees with the spec kind) · an attached part moving outside its band (reason word = that kind's
    oakink2.part_kinds.ATTACH_MOVED: fixed · lever (two joints: pulled far or pushing the jointed part) ·
    deformable)."""
    inst = {r: [o for o in v if PK.merges(OP.rid_of(o["obj_id"]))] for r, v in part_instances(objs).items()}
    inst = {r: v for r, v in inst.items() if len(v) >= 2}
    if not inst:
        return {"status": "no_parts"}
    for r, v in inst.items():
        rids = [OP.rid_of(o["obj_id"]) for o in v]
        dup = sorted({x for x in rids if rids.count(x) > 1})
        if dup:
            raise PartMergeError(
                "duplicate",
                f"{OP.name_of(r)}: the same part id more than once {dup} — supplement copy (regenerate base)",
            )
    if len(inst) > 1:
        from dexengine.data.oakink2 import multi_instance as MI  # noqa: PLC0415 — circular import

        return MI.decide(inst, q)
    ((root, members),) = inst.items()
    return _one_instance(root, members, q)


def _one_instance(root: str, members: list[dict], q: np.ndarray) -> dict:
    """Decision for one instance (rule in the part_clusters docstring)."""
    by = {OP.rid_of(o["obj_id"]): o for o in members}
    supp = [o for o in members if o.get("supplement")]
    _check_supplement_time(root, supp)
    body = PK.body_of(root)
    if body not in by:
        raise PartMergeError(
            "no_body",
            f"{OP.name_of(root)}: body {body}({OP.name_of(body)}) not in the tree "
            f"— nowhere to attach parts {sorted(by)}",
        )
    joint = [r for r in by if PK.kind_of(r)[0] in PK.JOINT_KINDS]
    jfit = {j: _joint_fit(block(q, by[body]), block(q, by[j]), PK.kind_of(j)[0]) for j in joint}  # at most one
    # What a lever can push = joint values of jointed parts that moved with their spec kind (another kind fails below as
    # kind_mismatch).
    drive = [
        joint_values(block(q, by[body]), block(q, by[j]), f) for j, f in jfit.items() if f["kind"] == PK.kind_of(j)[0]
    ]
    groups, levers = _attach_groups(root, by, body, joint, drive, q)
    clusters = [{"anchor": by[g], "fixed": groups[g], "path_m": path_m(block(q, by[g]))} for g in [body, *joint]]
    if supp:  # supplementary parts whose version was checked; a marker shared by measurer and converter
        print(RS.mark(len(supp), f"D1-g {OP.name_of(root)}"), flush=True)
    out = {"instance": root, "same_grid": len(supp), **({"levers": levers} if levers else {})}
    if not joint:
        return {"status": "fixed_only", **out, "clusters": clusters}
    return _joint_decision(by, body, joint, jfit, groups, clusters, out)


def _check_supplement_time(root: str, supp: list[dict]) -> None:
    """Refuse supplementary parts sampled at another time: relative motion is not measured across time bases."""
    for o in supp:
        why = RS.stamp_reason(o["supplement"].get("sample"), f"{OP.name_of(root)}: supplementary part {o['obj_id']}")
        if why:
            raise PartMergeError("raw_time", why)


def _attach_groups(
    root: str, by: dict, body: str, joint: list, drive: list, q: np.ndarray
) -> tuple[dict[str, list[dict]], list]:
    """Group every non-anchor part under its owner (the body or a jointed part, found by climbing the attachment
    chain). A fixed part must move rigidly with its owner; a lever must stay at rest against the joint drive."""

    def owner(rid):  # climb the attachment chain to the body or a jointed part
        while True:
            k, p = PK.kind_of(rid)
            if k == "body" or k in PK.JOINT_KINDS:
                return rid
            if p not in by:
                raise PartMergeError(
                    "no_body", f"{OP.name_of(root)}: {rid}({OP.name_of(rid)}) has parent {p}, which is not in the tree"
                )
            rid = p

    groups: dict[str, list[dict]] = {body: [], **{j: [] for j in joint}}
    levers = []
    for rid, o in sorted(by.items()):
        if rid in groups:
            continue
        own = owner(rid)
        groups[own].append(o)
        k = PK.kind_of(rid)[0]
        if k != "lever" and not rigid_together(block(q, by[own]), block(q, o)):
            f = fit_pair(block(q, by[own]), block(q, o))
            raise PartExcluded(
                PK.ATTACH_MOVED[k],
                f"{OP.name_of(rid)} moves separately from {OP.name_of(own)} although the spec says {k} "
                f"(rotation {f['theta_max_deg']:.1f}° · translation {f['pos_span_m'] * 1e3:.0f} mm)",
            )
        if k == "lever":
            lv, why = AT.lever_fixed(
                *rel_pose(block(q, by[own]), block(q, o)), o, drive[0] if drive else None, ROT_TOL_DEG
            )
            if why:
                raise PartExcluded(
                    PK.ATTACH_MOVED[k], f"{OP.name_of(rid)} is a lever (spec) of {OP.name_of(own)} — {why}"
                )
            levers.append(lv)
    return groups, levers


def _joint_decision(by: dict, body: str, joint: list, jfit: dict, groups: dict, clusters: list, out: dict) -> dict:
    """Decision of an instance with a jointed part: its measured kind must be the spec kind or rigid; a rigid one makes
    the whole instance one cluster (one_cluster), a moved one keeps body and part apart (two_clusters)."""
    jk = PK.kind_of(joint[0])[0]
    fit = jfit[joint[0]]
    if fit["kind"] not in (jk, "rigid"):
        raise PartExcluded(
            "kind_mismatch",
            f"{OP.name_of(joint[0])} with {OP.name_of(body)} is {jk} in the spec but "
            f"measured {fit['kind']} (rotation residual {fit.get('rot_resid_deg', 0):.1f}° · position residual "
            f"{fit.get('pos_resid_m', 0) * 1e3:.0f} mm · rotation {fit['theta_max_deg']:.1f}°)",
        )
    if fit["kind"] == "rigid":
        one = [
            {
                "anchor": by[body],
                "fixed": groups[body] + [by[joint[0]]] + groups[joint[0]],
                "path_m": clusters[0]["path_m"],
            }
        ]
        return {"status": "one_cluster", **out, "clusters": one, "joint": by[joint[0]], "joint_kind": jk}
    return {"status": "two_clusters", **out, "clusters": clusters, "fit": fit, "joint_kind": jk}


def _joint_fit(qA: np.ndarray, qB: np.ndarray, jk: str) -> dict:
    """Measurement of body ↔ jointed part (spec kind `jk`): fit_pair, and when the spec is a slide but that
    classification
    gives no slide, the slide model directly (fit_slide · band = band). The kind check and the lever drive share
    (part_clusters) read this one measurement."""
    fit = fit_pair(qA, qB)
    if jk == "prismatic" and fit["kind"] not in (jk, "rigid"):
        alt = fit_slide(qA, qB)
        if alt["kind"] in (jk, "rigid"):
            fit = alt
    return fit


def fit_slide(qA: np.ndarray, qB: np.ndarray) -> dict:
    """Measure a pair whose spec is a slide **with the slide model directly** (_joint_fit calls it after fit_pair). Rail
    =
    principal direction of the displacement relative to the first frame (same formula as fit_pair's prismatic) · pose
    reference = first frame (the reference of the converter's child_pose). Residuals = max off-rail (m) · max rotation
    from the first frame (°) · band = band('prismatic').
      rigid      displacement and rotation both inside the band (a still drawer; the joint comes from the spec asset)
      prismatic  rotation and off-rail residual inside the band and displacement at least the band
      free       anything else"""
    RA = Rot.from_quat(_quat_xyzw(qA))
    RB = Rot.from_quat(_quat_xyzw(qB))
    Rrel = RA.inv() * RB
    prel = RA.inv().apply(qB[:, :3] - qA[:, :3])
    th = float(np.degrees((Rrel * Rrel[0].inv()).magnitude()).max())
    d = prel - prel[0]
    span = float(np.linalg.norm(d, axis=1).max())
    pos_tol, rot_tol = band("prismatic")
    out = {"n_frames": len(qA), "theta_max_deg": th, "pos_span_m": span, "rot_resid_deg": th}
    if th < rot_tol and span < pos_tol:
        out.update(kind="rigid", pos_resid_m=span)
        return out
    ax = np.linalg.svd(d, full_matrices=False)[2][0]
    ax /= np.linalg.norm(ax)
    perp = float(np.linalg.norm(d - np.outer(d @ ax, ax), axis=1).max())
    out.update(
        kind="prismatic" if (th < rot_tol and perp < pos_tol) else "free",
        axis=ax.tolist(),
        pos_resid_m=perp,
        travel_m=span,
    )
    return out


def fit_pair(qA: np.ndarray, qB: np.ndarray) -> dict:
    RA = Rot.from_quat(_quat_xyzw(qA))
    RB = Rot.from_quat(_quat_xyzw(qB))
    pA = qA[:, :3]
    pB = qB[:, :3]
    Rrel = RA.inv() * RB  # pose of moving relative to base
    prel = RA.inv().apply(pB - pA)  # position of moving relative to base
    D = Rrel * Rrel[0].inv()  # relative rotation from frame 0
    v = D.as_rotvec()
    th = np.linalg.norm(v, axis=1)
    out = {
        "n_frames": len(qA),
        "theta_max_deg": float(np.degrees(th.max())),
        "pos_span_m": float(np.linalg.norm(prel - prel[0], axis=1).max()),
    }
    if out["theta_max_deg"] < ROT_TOL_DEG:  # no rotation → rigid/prismatic
        d = prel - prel[0]
        span = np.linalg.norm(d, axis=1).max()
        if span < POS_TOL_M:
            out.update(kind="rigid", rot_resid_deg=out["theta_max_deg"], pos_resid_m=float(span))
            return out
        u, s, _ = np.linalg.svd(d - d.mean(0), full_matrices=False)
        ax = (np.linalg.svd(d, full_matrices=False)[2])[0]
        ax /= np.linalg.norm(ax)
        perp = np.linalg.norm(d - np.outer(d @ ax, ax), axis=1).max()
        out.update(
            kind="prismatic" if perp < POS_TOL_M else "free",
            axis=ax.tolist(),
            rot_resid_deg=out["theta_max_deg"],
            pos_resid_m=float(perp),
            travel_m=float(span),
        )
        return out
    w = th[:, None]  # principal axis = first singular vector of the |θ|-weighted SVD
    n = np.linalg.svd(v * w, full_matrices=False)[2][0]
    n /= np.linalg.norm(n)
    if np.dot(v[th.argmax()], n) < 0:
        n = -n
    al = v @ n  # axial component (signed joint angle)
    Rp = Rot.from_rotvec(np.outer(al, n))  # rotation about the axis only
    rot_res = np.degrees((Rp.inv() * D).magnitude())  # residual rotation off the axis
    A = []
    b = []  # p_rel(t) − R_t·p0 = (I − R_t)c + h·α·n̂
    Rm = Rp.as_matrix()
    for t in range(len(al)):
        A.append(np.hstack([np.eye(3) - Rm[t], (al[t] * n).reshape(3, 1)]))
        b.append(prel[t] - Rm[t] @ prel[0])
    x, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    c, h = x[:3], float(x[3])
    pred = np.einsum("tij,j->ti", Rm, prel[0] - c) + c + np.outer(al * h, n)
    pos_res = float(np.linalg.norm(prel - pred, axis=1).max())
    rr = float(rot_res.max())
    travel = float(abs(h) * (al.max() - al.min()))
    kind = "free"
    if rr < ROT_TOL_DEG and pos_res < POS_TOL_M:
        kind = "screw" if travel >= POS_TOL_M else "revolute"
    out.update(
        kind=kind,
        axis=n.tolist(),
        pivot=c.tolist(),
        pitch_m_per_rad=h,
        rot_resid_deg=rr,
        pos_resid_m=pos_res,
        travel_m=travel,
        angle_range_deg=float(np.degrees(al.max() - al.min())),
    )
    return out


if __name__ == "__main__":
    from dexengine.data.oakink2.artic_fit_report import main  # the measurement report CLI of this module

    raise SystemExit(main())
