"""Case classification of one base tree (classify_tree): contact sidecars, hand × object masks, the
assignment, wrist measures, the case name and its flags. Thresholds TH_* are owned here (thresholds() lists them for
the census); the CLI and the census are dexengine/data/cases/taxonomy.py."""

from __future__ import annotations

import json

import numpy as np

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.data.cases import (
    assign_r as AR2,  # ASSIGN-2: rule R and its validity gate; this file only passes the inputs
)
from dexengine.data.cases import assign_r_inputs as ARI
from dexengine.data.cases import (
    assign_version as AR,  # assignment versions (one shared ACTIVE version, hand registry HANDS)
)
from dexengine.data.cases import frame_layers as FL  # frame layers per hand × object (A3)
from dexengine.data.cases import gate as CGATE
from dexengine.data.cases import (
    hand_object_assign as H,  # hand–object assignment (A3–A7, timeline) and its thresholds
)
from dexengine.data.cases.assign_version import (
    ASSIGN_KEY,  # assignment record key (written here, read by the roles stage)
)
from dexengine.data.contact import (
    fingerprint as SF,  # sidecar input fingerprint, checked against this base before reading
)
from dexengine.data.contact import geom as CG
from dexengine.data.contact.source import CONTACT_SOURCE
from dexengine.data.mano_frame import mano_keypoints_base
from dexengine.data.support import plane as SA

#: Case record version (this one adds case.move_runs, the motion layer of hand_object_assign.move_runs).
VERSION = "2026-10-14a"


# ── Thresholds of the case flags (A8–A11) — the assignment thresholds belong to hand_object_assign ──
TH_CONTACT_M = CG.KEYPOINT_THRESHOLD_M  # keypoint↔surface contact (2 cm) — dexengine.data.contact.common owns


#   this threshold, here it is only reused (object–object)
TH_TOUCH_ONLY_FRAC = 0.05  # touch_only flag: inactive hand with contact fraction ≥ 5 % (information —


#   activity is the grasp total alone, hand_object_assign.TH_ACTIVE_MIN_S)
TH_WRIST_DESK_M = 0.02  # lowest wrist − desk < 2 cm → hand_below_desk


TH_IDLE_MOVE_M = 0.10  # wrist path of an inactive hand > 10 cm → idle_hand_moves


TH_OBJOBJ_STRIDE = 10  # object–object contact is approximated with a frame stride (informational)


def thresholds() -> dict:
    """The TH_* thresholds of this module, in definition order (recorded in the case census)."""
    return {
        "TH_CONTACT_M": TH_CONTACT_M,
        "TH_TOUCH_ONLY_FRAC": TH_TOUCH_ONLY_FRAC,
        "TH_WRIST_DESK_M": TH_WRIST_DESK_M,
        "TH_IDLE_MOVE_M": TH_IDLE_MOVE_M,
        "TH_OBJOBJ_STRIDE": TH_OBJOBJ_STRIDE,
    }


HAND = RR.hand_dir()


#: MANO fingertip contact threshold, read from the measured calibration file (no number here). The sidecar threshold
#: (2 cm on keypoints) is not reused: MANO distances run a few mm larger than robot surface distances, so 2 cm would
#: miss
#: some robot contacts. A missing file is an error; falling back to 2 cm would move case boundaries.
def _mano_thr() -> float:
    from dexengine.data.contact.source import TIP_DISTANCE_FILE as f

    if not f.is_file():
        raise FileNotFoundError(
            f"{f} is missing: it holds the measured MANO contact threshold; the sidecar threshold is not used instead"
        )
    return float(json.loads(f.read_text())["best_thr_m"])


def _assign_mode(assign_version: str) -> str:
    """Assignment version to use: the explicit one (read-only reports) or the shared version (unregistered hands
    fail)."""
    mode = assign_version or AR.hand_version(HAND)
    if mode not in AR.ASSIGN_VERSIONS:
        raise KeyError(f"unknown assignment version {mode!r} — {AR.ASSIGN_VERSIONS}")
    return mode


def _record_active(rec: dict, hands: list) -> list:
    """Active hands of the chosen assignment record, in hand order: the case's `active` (object_roles.from_assign relies
    on it)."""
    return [s for s in hands if (rec["per_side"].get(s) or {}).get("active")]


def _assign_record(
    mode: str,
    ds: str,
    ti: dict,
    A: dict,
    objs: list,
    hands: list,
    T: int,
    fps: float,
    kpm: dict,
    mano: dict,
    thr: float,
    objsrc: dict,
) -> dict:
    """base `rules.ASSIGN` record. ASSIGN-1 is hand_object_assign.record as is; ASSIGN-2 passes the rule-R inputs
    (assign_r_inputs.build, which also reads the source statements) to assign_r.assign2 (validity gate)."""
    if mode == AR.ASSIGN_VERSION:
        return {**H.record(A), "contact_source": CONTACT_SOURCE}
    F, task_objects = ARI.build(ds, ti, objs, hands, T, fps, kpm, mano, thr, objsrc)
    return {**AR2.assign2(A, F, task_objects), "contact_source": CONTACT_SOURCE}


def _read_sidecars(ds: str, clip: str, ti: dict, q, hands: list) -> tuple[dict, dict]:
    """Each side's MANO contact sidecar, read once and checked against the input fingerprint recomputed from this base
    → (side → (npz, object index, contact threshold), side → input id)."""
    _mano: dict = {}
    _cin: dict = {}  # inputs this case read (input_id per side); roles check them against their own base
    _tin = SF.TreeInputs(P.main_root(), ds, clip, ti, q, HAND)  # data root, not the code tree
    for s in hands:
        f = P.sidecar_file("contact_all", ds, clip, s, HAND)  # path = dexengine.core.layout.sidecar_rel
        # Checking the length T alone would let a sidecar from another object-pose generation through. Compare with the
        # input fingerprint recomputed from this base; a mismatch or unknown fails (this also covers missing and
        # length).
        _exp = _tin.expected(SF.LAYER_ALL, s)
        SF.check(f, _exp, ds=ds)
        _cin[s] = SF.input_id(_exp)
        z = np.load(f, allow_pickle=True)
        _mano[s] = (z, {str(n): i for i, n in enumerate(z["obj_names"])}, float(z["contact_threshold_m"]))
    return _mano, _cin


def _object_masks(ds: str, clip: str, objs: list, q, hands: list, fps: float, _mano: dict, _MANO_THR, tips_v: dict):
    """A3 frame layers per hand × object (frame_layers.hand_object_masks) and the per-object inputs of the
    assignment → (masks, obj_pos, obj_rot, _objsrc)."""
    masks, obj_pos, obj_rot = {}, {}, {}
    # object → (o7, joint, joint column, small object, surface samples); ASSIGN-2 inputs (_assign_record)
    _objsrc: dict = {}
    for o in objs:
        st = int(o["qpos_start"])
        o7 = q[:, st : st + 7]
        art = o.get("articulation")
        hinge = q[:, art["qpos_index"]] if art else None
        surf = CG.ensure_surface(ds, o)
        _sz = np.load(surf)
        _pts = _sz["points"] if "points" in _sz.files else np.concatenate([_sz["base_points"], _sz["moving_points"]])
        small = float((_pts.max(0) - _pts.min(0)).max()) < H.TH_SMALL_OBJ_M
        _objsrc[o["name"]] = (o7, art, hinge, small, _sz)
        rot = FL.angular_speed_dps(o7[:, 3:7], fps)  # in-place rotation (counts toward carry), every object
        obj_pos[o["name"]] = o7[:, :3]
        obj_rot[o["name"]] = rot
        for s in hands:
            z, oidx, _ = _mano[s]
            if o["name"] not in oidx:  # skipping silently would make the object look unused
                raise KeyError(f"{clip}/{s}: object {o['name']} missing from the sidecar (stale sidecar)")
            oi = oidx[o["name"]]
            d = np.asarray(z["dist"][:, oi, :], float)  # (T, 5 fingers)
            wit = np.asarray(z["witness_obj"][:, oi, :, :], float)  # (T, 5, 3)
            dp = np.asarray(z["palm_dist"][:, oi], float)  # (T,) — for MANO, measured from the wrist keypoint
            # threshold = measured threshold (_mano_thr fails when it is missing)
            masks[(s, o["name"])] = FL.hand_object_masks(
                d, wit, dp, _MANO_THR, small, tips_v[s], fps, rot_dps=rot, hinge=hinge
            )
    return masks, obj_pos, obj_rot, _objsrc


def _wrist_measures(q, lay, hands: list, _kpm: dict, sup: dict, desk_z):
    """Wrist heights (MANO keypoints; robot values for the oracle), the D14′ footprint depth and wrist paths."""
    wrist_p50_robot = {s: float(np.median(q[:, lay.hands[s] + 2])) for s in hands}
    # Wrist height (D18/D14) is also measured on the MANO wrist keypoint; the retargeted robot wrist would assign the
    # axis
    # differently per hand. Robot values stay in `*_robot_m` for the oracle. Frame: MANO keypoints share the base frame
    # (adapter offset 0; the hot3d yaw leaves z unchanged); only h2o also applies the desk-plane rotation (same owner as
    # the sidecar, h2o_frame). Keypoints use the same loader as tips_v above (_kpm, same window and rotation rules).
    wrist_min = {s: float(_kpm["wrist"][s][:, 2].min()) for s in hands}
    wrist_p50 = {s: float(np.median(_kpm["wrist"][s][:, 2])) for s in hands}
    # D14′: when the source has a real table outline (support.desk_footprint_raw), measure the wrist depth inside it
    # with
    # the MANO wrists of this tree window (dexengine.data.support.grab.wrist_footprint). Pieces re-measure on their own
    # window.
    wrist_fp = None
    if sup.get("desk_footprint_raw") and desk_z is not None:
        from dexengine.data.support import (
            grab as GDO,  # the heavy dependency stays inside that function
        )

        wrist_fp = GDO.wrist_footprint(
            np.concatenate([_kpm["wrist"][s] for s in hands]), sup["desk_footprint_raw"], desk_z
        )
    wrist_desk = {s: (round(wrist_min[s] - float(desk_z), 4) if desk_z is not None else None) for s in hands}
    wrist_desk_p50 = {
        s: (round(wrist_p50[s] - float(desk_z), 4) if desk_z is not None else None) for s in hands
    }  # D18: tells standing manipulation apart (median)
    wrist_path = {
        s: float(np.linalg.norm(np.diff(q[:, lay.hands[s] : lay.hands[s] + 3], axis=0), axis=1).sum()) for s in hands
    }
    return wrist_p50_robot, wrist_fp, wrist_desk, wrist_desk_p50, wrist_path


def _objobj_contact(ds: str, objs: list, used: list, q, T: int):
    """Object–object contact fraction of the first two used objects (informational; err:<type> when unmeasurable)."""
    objobj = None
    if len(used) >= 2:
        try:
            oa, ob = [o for o in objs if o["name"] in used[:2]]
            za = np.load(CG.ensure_surface(ds, oa))
            pa = za["points"] if "points" in za.files else za["base_points"]
            sa = pa[np.random.default_rng(0).choice(len(pa), min(256, len(pa)), replace=False)]
            idx = np.arange(0, T, TH_OBJOBJ_STRIDE)
            from scipy.spatial.transform import Rotation as Rot

            qa = q[idx, oa["qpos_start"] : oa["qpos_start"] + 7]
            Ra = Rot.from_quat(qa[:, [4, 5, 6, 3]]).as_matrix()
            pw = np.einsum("tij,kj->tki", Ra, sa) + qa[:, None, :3]
            d, _, _ = CG.tip_distances(
                pw,
                q[idx, ob["qpos_start"] : ob["qpos_start"] + 7],
                CG.ensure_surface(ds, ob),
                ob.get("articulation"),
                q[idx, ob["articulation"]["qpos_index"]] if ob.get("articulation") else None,
            )
            objobj = round(float((d.min(1) < TH_CONTACT_M).mean()), 3)
        except Exception as e:  # noqa: BLE001  informational axis: any failure is kept as err:<type> in the record
            objobj = f"err:{type(e).__name__}"
    return objobj


def _case_name(clip: str, hands: list, active, used: list, share: dict, A: dict, T: int, fps: float, objs: list):
    """The case name (mutually exclusive, first match top → bottom) and the handover gate verdicts → (case, hgate)."""
    # ── Case decision (mutually exclusive, first match top→bottom) ────────────
    # handover unit → reasons it is not a handover (renames the case only; the assignment is unchanged)
    hgate: dict = {}
    if len(hands) == 1:
        case = "UNI_R" if hands[0] == "right" else "UNI_L"
    elif len(active) == 0:
        case = "BI_NONE"
    elif len(active) == 1:
        case = "BI_ONE_ACTIVE"
    else:
        # assist is not sharing (it is the original hand's object) — unless the assignment kept that unit for both
        # hands (the helping hand also carried it ≥ hand_object_assign.TH_CARRY_MIN_S): then it is BI_SHARED_ASSIST.
        # The assignment's shared unit always counts, with its kind (the case name and the roles agree).
        shared = {k: v["kind"] for k, v in share.items() if v["kind"] != "assist"}
        if A["shared"]:
            shared[A["shared"]] = A["share_kind"]
        hgate = CGATE.gate(
            [u for u, k in shared.items() if k == "handover"],
            A["grasp_s"],
            T / fps,
            CGATE.art_units(H.units_of(objs)[1], objs),
        )
        for u, w in hgate.items():
            verdict = "not a handover → mixed" if CGATE.is_out(w) else "handover kept"
            print(f"{CGATE.MARK} {clip} {u} {verdict} — {' · '.join(w)}", flush=True)
            if CGATE.is_out(w):
                shared[u] = "mixed"
        n_used = len(used)
        if n_used == 1:
            k = shared.get(used[0])
            if (
                k is None and used[0] in share
            ):  # 1 object · 2 active hands, but the second hand is assist → in effect a one-hand object
                case = "BI_ONE_ACTIVE"
            else:
                case = {
                    "sim": "BI_SHARED_SIM",
                    "handover": "BI_HANDOVER",
                    "mixed": "BI_SHARED_MIXED",
                    "parts": "BI_PARTS",
                    "assist": "BI_SHARED_ASSIST",
                }.get(k, "UNCLASSIFIED")
        elif n_used == 2:
            if not shared:
                case = "BI_TWO_OBJECTS"
            else:
                kinds = set(shared.values())
                case = (
                    "BI_PARTS_PLUS"
                    if "parts" in kinds
                    else ("BI_HANDOVER_PLUS" if kinds <= {"handover"} else "BI_SHARED_PLUS")
                )
        else:
            case = "BI_MULTI"
    return case, hgate


def _flags(ti, objs, used, wrist_desk, desk_mode, arts, objobj, share, hands, active, wrist_path, touch_only) -> list:
    """Case flags (information next to the case name; quality flags of the tree are appended)."""
    flags = []
    if len(used) < len(objs):
        flags.append("extras_present")
    if any(v is not None and v < TH_WRIST_DESK_M for v in wrist_desk.values()):
        flags.append("hand_below_desk")
    if desk_mode == SA.DESK_NONE:
        flags.append("desk_none")
    if desk_mode == SA.DESK_HEIGHT_UNKNOWN:
        flags.append("desk_unknown")
    if arts and (len(arts) >= 2 or any(len(o["articulation"].get("parts") or {}) >= 3 for o in arts)):
        flags.append("art_complex")
    if isinstance(objobj, float) and objobj > 0.05:
        flags.append("obj_obj_contact")
    if any(v["kind"] == "assist" for v in share.values()):
        flags.append("assist_hand")
    flags.extend(f"idle_hand_moves_{s[0]}" for s in hands if s not in active and wrist_path[s] > TH_IDLE_MOVE_M)
    flags.extend(f"touch_only_{s[0]}" for s in touch_only)
    for f in (ti.get("quality") or {}).get("flags", []) or []:
        if f not in flags:
            flags.append(f)
    return flags


def _obj_kind(o):
    if o.get("articulation"):
        return (o["articulation"] or {}).get("type") or "articulated"  # arctic: hinge (has a joint column)
    g = o.get("articulation_group")
    if g:  # oakink2: a part pair (2 rigid bodies); the fit kind is fit.type of the part entry
        fit = (g.get("fit") or {}).get("type") if g.get("role") == "part" else None
        return f"part-{fit}" if fit else "part-base"
    return "rigid"


def _load_tree(ds: str, emb: str, clip: str):
    """The base tree's task_info, qpos, frame count, frame rate and qpos layout."""
    tree = P.layer_path("base", ds, hand=HAND, emb=emb, clip=clip)
    ti = json.loads((tree / "task_info.json").read_text())
    z = np.load(tree / "0/trajectory_kinematic.npz")
    q = z["qpos"]
    T = len(q)
    fps = float(z["frequency"]) if "frequency" in z.files else (1.0 / float(ti.get("ref_dt", 1 / 30)))
    lay = P.qpos_layout.read(str(tree))
    return ti, q, T, fps, lay


def _assign_tree(mode, ds: str, clip: str, ti: dict, q, T: int, fps: float, hands: list, objs: list):
    """Frame layers per hand × object → assignment, motion layer and assignment record
    → (A, move, assign_rec, contact input fingerprint, MANO keypoints)."""
    # A3 frame layers per hand × object: contact, hold, carry, timeline grasp (defined in
    # frame_layers.hand_object_masks).
    # Each side's MANO sidecar (built before roles) is read once.
    _MANO_THR = _mano_thr()
    _mano, _cin = _read_sidecars(ds, clip, ti, q, hands)
    # Hand speed also comes from the MANO fingertips: robot fingertip speed would shift the comove decision by a frame
    # or two per hand, so timeline boundaries, and the _tN units cut at them, would differ per hand.
    # Window and rotation rules match the sidecar (compute_side).
    _kpm = mano_keypoints_base(ti, ds, clip, T, hands)
    tips_v = {s: H.smooth_velocity(_kpm["tips"][s].mean(1), fps)[1] for s in hands}
    masks, obj_pos, obj_rot, _objsrc = _object_masks(ds, clip, objs, q, hands, fps, _mano, _MANO_THR, tips_v)
    # A4–A7 and the timeline come from the assignment (hand_object_assign.assign); the case only adds a name and flags.
    A = H.assign(T, fps, hands, objs, masks, obj_pos)
    # Motion layer (carry ∧ body motion ∨ joint drive), read only by the split rule (segment.split, D36). The assignment
    # is unchanged.
    move = H.move_runs(T, fps, hands, objs, masks, obj_pos, obj_rot)
    assign_rec = _assign_record(mode, ds, ti, A, objs, hands, T, fps, _kpm, _mano, _MANO_THR, _objsrc)
    return A, move, assign_rec, _cin, _kpm


def classify_tree(ds: str, emb: str, clip: str, assign_version: str = "") -> dict:
    """assign_version = assignment version to use; empty = the shared version
    (dexengine.data.cases.assign_version.hand_version). An explicit version is only for read-only reports (measuring
    ASSIGN-2 ahead for hands still on ASSIGN-1)."""
    mode = _assign_mode(assign_version)
    ti, q, T, fps, lay = _load_tree(ds, emb, clip)
    hands = sorted(lay.hands)  # ['left','right'] etc.
    objs = ti.get("objects") or []
    if not objs:
        return {"clip": clip, "case": "UNCLASSIFIED", "why": "no objects[]"}
    A, move, assign_rec, _cin, _kpm = _assign_tree(mode, ds, clip, ti, q, T, fps, hands, objs)
    active, used, share = A["active"], A["used"], A["share"]
    # Active hands of the case = active hands of the chosen assignment record (roles-stage invariant,
    # object_roles.from_assign). Case name, flags and timeline stay the ASSIGN-1 measurement: the D36 piece registry
    # reads the timeline, and changing it would rename pieces.
    case_active = _record_active(assign_rec, hands)
    touch_only = [s for s in hands if s not in active and A["hand_frac"][s] >= TH_TOUCH_ONLY_FRAC]
    # A8 articulation · A9 support surface · A10/11
    arts = [o for o in objs if o.get("articulation")]
    art_kinds = sorted({o["articulation"]["type"] for o in arts})
    sup = ti.get("support") or {}
    desk_mode = sup.get("desk_mode")
    desk_z = sup.get("desk_z")
    wrist_p50_robot, wrist_fp, wrist_desk, wrist_desk_p50, wrist_path = _wrist_measures(
        q, lay, hands, _kpm, sup, desk_z
    )
    objobj = _objobj_contact(ds, objs, used, q, T)
    case, hgate = _case_name(clip, hands, active, used, share, A, T, fps, objs)
    flags = _flags(ti, objs, used, wrist_desk, desk_mode, arts, objobj, share, hands, active, wrist_path, touch_only)
    if arts:
        case = f"{case}+ART{len(arts)}"

    obj_kind = {o["name"]: _obj_kind(o) for o in objs}
    return {
        "clip": clip,
        "dataset": ds,
        "emb": emb,
        "T": T,
        "fps": fps,
        "case": case,
        "flags": flags,
        "version": VERSION,
        "contact_source": CONTACT_SOURCE,
        "contact_input": {"schema": SF.FP_SCHEMA, "sides": _cin},
        "n_objects": len(objs),
        "n_used": len(used),
        "n_extra": len(objs) - len(used),
        "obj_ids": sorted({str(o["obj_id"]) for o in objs}),
        "obj_kind": obj_kind,
        "n_articulated": len(arts),
        "hands": hands,
        "active": case_active,
        "used": used,
        "touched": A["touched"],
        "units": A["units"],
        "part_groups": A["part_groups"],
        "hand_total_s": A["hand_total_s"],
        "hand_frac": {s: round(float(v), 3) for s, v in A["hand_frac"].items()},
        "contact_frac": A["contact_frac"],
        "grasp_s": A["grasp_s"],
        "share": share,
        "handover_gate": hgate,
        "timeline": A["timeline"],
        "move_runs": move,
        "articulation": art_kinds,
        "desk_mode": desk_mode,
        "desk_z": desk_z,
        "wrist_desk_m": wrist_desk,
        "wrist_desk_p50_m": wrist_desk_p50,
        "wrist_fp": wrist_fp,
        "wrist_source": CONTACT_SOURCE,
        "wrist_desk_p50_robot_m": {
            s: (round(wrist_p50_robot[s] - float(desk_z), 4) if desk_z is not None else None) for s in hands
        },
        "wrist_path_m": {s: round(v, 3) for s, v in wrist_path.items()},
        "artic_drive": A["artic_drive"],  # D27 evidence (empty = no articulation drive)
        "obj_obj_contact_frac": objobj,
        "main_by_adapter": {o["name"]: o.get("side") for o in objs if o.get("role") == "main"},
        ASSIGN_KEY: assign_rec,
    }
