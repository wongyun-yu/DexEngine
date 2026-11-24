"""Helpers of the cross repair (cross_repair.py): link sides, the object relative-shift audit value, the per-set
statistics, the post-repair tracking-error recompute and the per-clip ledger line."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from dexengine.core.paths import mode_record  # record-by-mode owner (a dry run never writes the write run's ledger)
from dexengine.placement.collision import cross_sets as CC
from dexengine.placement.collision import det_math as D
from dexengine.placement.repair import wrist_shift as WS

#: The repaired collision sets (pairs with a hand link in them): crossed hand/arm · hand <-> opposite object · arm/hand
#: <-> torso. The finger indices, abduction joints and limits come from the tree + URDF (cross_sets.HandLayout).
SET_REPAIR = ("1_cross", "2_hand_obj", "4_torso")
#: Header of the per-clip ledger TSV (data — keep the column names).
LEDGER_HEADER = (
    "clip\tT\tstatus\trepair_frames\ttouched_frames\tbefore_pen_frames\tbefore_max_mm\tafter_pen_frames\t"
    "after_max_mm\tspawn_before_mm\tspawn_after_mm\twrist_p50_mm\twrist_max_mm\tfinger_p50_deg\tfinger_max_deg\t"
    "arm_p50_deg\tarm_max_deg\tobjobj_max_mm\tarm_only_max_mm\tunresolved\tsec\tts\tobj_rel_shift_max_mm"
)
OBJ_REL_KEY = "obj_rel_shift_max_mm"  # report key of the audit value — obj_rel_shift_mm's max over the frames


def side_of_link(link: str, arm) -> str | None:
    """The side of a robot link of a body with arm ``arm`` (ArmSpec): an arm link's (ArmSpec.body_side), else a hand
    link's (every merged hand link starts with its side — merge.hand_subtree); None for any other link."""
    side = arm.body_side(link)
    if side is not None:
        return side
    return next((s for s in ("right", "left") if link.startswith(f"{s}_")), None)


def is_hand_link(link: str, arm) -> bool:
    """Whether ``link`` is in a hand group (cross_sets.group_of under the ArmSpec ``arm``)."""
    return CC.group_of(link, arm) in ("RH", "LH")


def obj_rel_shift_mm(dobj: dict, obj: dict, shared: bool) -> float:
    """Audit column: two held objects moved by different amounts change their relative pose (a shared object takes
    both moves) — the largest per-frame difference of the two sides' object shifts (m -> mm). 0 when the clip does not
    hold two separate objects."""
    if obj.get("right") is None or obj.get("left") is None or shared:
        return 0.0
    return float(D.norm(np.asarray(dobj["right"], float) - np.asarray(dobj["left"], float)).max(initial=0.0) * 1000)


def _set_stat(v: np.ndarray) -> dict:
    pen = bool((v < 0).any())
    return {
        "pen_frames": int((v < 0).sum()),
        "max_depth_mm": round(float(-v.min() * 1000), 2) if pen else 0.0,
        "min_dist_mm": round(float(v.min() * 1000), 2),
        "pen_at_frame0": bool(v[0] < 0),
    }


def _palm_tracking_error(model, arm_names, arm_q_all, qpos_new, side: str, tree: Path, base, arm: str, hand: str):
    """Per frame, the (position m, rotation deg) between the palm-frame FK of the repaired arm joints and the palm
    target (tree wrist position + R_palm·palm_off). The same target expression as the arm IK (including the palm
    offset). The orientation is the palm body quaternion (wxyz) from 0/body_positions.npz. None on failure."""
    from dexengine.core import hands as HS
    from dexengine.placement import arm_ik as AIP

    bp = tree / "0/body_positions.npz"
    if not bp.is_file():
        return None
    # The palm FK body (the body wrist_poses extracted the pose from) is HandSpec.palm_fk_body_right. FK body and URDF
    # palm frame names can differ per hand (DG-5F-S: rig `link_mount` <-> URDF `link_base`); their relation (a pure
    # translation) is palm_frame_offset(hand) in the target formula. This identity check stays outside the try below
    # (a numeric diagnostic failure does not flip the result): a tree / hand mismatch is an input error.
    b = np.load(bp, allow_pickle=True)
    names = [str(x) for x in b[f"{side}_hand_body_names"]]
    palm_body = HS.HAND_REGISTRY[hand].palm_fk_body_right.replace("right_", f"{side}_", 1)
    if palm_body not in names:
        raise RuntimeError(
            f"[cross_repair] {tree.name}: {side} palm FK body {palm_body!r} is not in body_positions "
            f"(hand {hand}, bodies {names[:6]}…) — the tree and the hand do not match"
        )
    try:
        _, ee_frame, _, _, _ = AIP.resolve_combo(arm, hand, side)
        quat = np.asarray(b[f"{side}_hand_body_quat"][:, names.index(palm_body), :], float)  # wxyz
        T = len(arm_q_all)
        if quat.shape[0] != T or qpos_new.shape[0] != T:
            return None
        off_q = model.layout.hands[side]
        palm_off = AIP.palm_frame_offset(hand)
        fid = model.model.getFrameId(ee_frame)
        # det_math + DetFK throughout (the bits are the same on every host — CrossSetModel's FK)
        qn = quat / np.maximum(D.norm(quat), 1e-12)[:, None]
        w, x, y, z_ = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
        R = np.stack(
            [
                np.stack([1 - 2 * (y * y + z_ * z_), 2 * (x * y - z_ * w), 2 * (x * z_ + y * w)], -1),
                np.stack([2 * (x * y + z_ * w), 1 - 2 * (x * x + z_ * z_), 2 * (y * z_ - x * w)], -1),
                np.stack([2 * (x * z_ - y * w), 2 * (y * z_ + x * w), 1 - 2 * (x * x + y * y)], -1),
            ],
            -2,
        )
        tgt, Rt = WS.ee_target(qpos_new[:, off_q : off_q + 3], R, palm_off, model.desk_h, det=True)
        tgt = tgt - np.asarray(base, float)
        q = np.broadcast_to(np.asarray(model.pin.neutral(model.model), float), (T, model.model.nq)).copy()
        for k, jn in enumerate(arm_names):
            q[:, model.jq[jn]] = arm_q_all[:, k]
        Rf, pf = model.fk.frames(q, [fid])
        pe = D.norm(pf[:, 0] - tgt)
        c = (D.trace_AtB(Rf[:, 0], Rt) - 1.0) * 0.5
        re_ = D.acos_deg(c)
        return pe, re_
    except Exception as e:  # noqa: BLE001 — a diagnostic recompute failure must not flip the result
        print(f"[cross_repair] tracking recompute failed ({side}): {e}", flush=True)
        return None


def _ledger(report: dict, a):
    """Append one TSV line per clip (header on a new file). Same meaning as the f0 ledger — a failure is recorded and
    placement continues. A dry run writes to the mode-tagged ledger (paths.mode_record ``.dryrun``)."""
    if not a.ledger:
        return
    p = mode_record(a.ledger, "dryrun" if a.dry_run else None)  # a dry run's line never joins the write ledger
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        with open(p, "w") as f:
            f.write(LEDGER_HEADER + "\n")
    st = report.get("stats") or {}
    wr = st.get("wrist_mm", {})
    fd = st.get("finger_deg", {})
    ad = st.get("arm_deg", {})
    bh, ah = report["before_hand"], report.get("after_hand", report["before_hand"])
    row = [
        report["clip"],
        report["T"],
        report["status"],
        report["n_repair_frames"],
        st.get("frames_touched", 0),
        sum(bh[s]["pen_frames"] for s in SET_REPAIR),
        max(bh[s]["max_depth_mm"] for s in SET_REPAIR),
        sum(ah[s]["pen_frames"] for s in SET_REPAIR),
        max(ah[s]["max_depth_mm"] for s in SET_REPAIR),
        report["spawn_before_mm"],
        report.get("spawn_after_mm", report["spawn_before_mm"]),
        max((wr.get(k, {}).get("p50", 0.0) for k in ("right", "left")), default=0.0),
        max((wr.get(k, {}).get("max", 0.0) for k in ("right", "left")), default=0.0),
        max((fd.get(k, {}).get("p50", 0.0) for k in ("right", "left")), default=0.0),
        max((fd.get(k, {}).get("max", 0.0) for k in ("right", "left")), default=0.0),
        max((ad.get(k, {}).get("p50", 0.0) for k in ("right", "left")), default=0.0),
        max((ad.get(k, {}).get("max", 0.0) for k in ("right", "left")), default=0.0),
        report["obj_obj_max_depth_mm"],
        report["arm_only_max_depth_mm"],
        len(report.get("unresolved_frames", [])),
        report["sec"],
        time.strftime("%FT%T"),
        report[OBJ_REL_KEY],
    ]
    with open(p, "a") as f:
        f.write("\t".join(str(x) for x in row) + "\n")
