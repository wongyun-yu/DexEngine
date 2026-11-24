"""Contact derivation of one combo tree: the run's roots (set once by _configure_roots from --combo/--dataset/--emb/
--src-layer), the time map from the reference base (build_map) and the per-tree derivation (process_tree).
The roots live here with their readers; derive.main reads them as derive_tree.<NAME> after configuring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from dexengine.core.paths import (
    ARM_IDS,
    HAND_IDS,
    IK_SIDECAR_SFX,
    TREE_SFX,
    contact_derive_dir,
    data_rel,
    hand_spec_dir,
    layer_path,
    main_root,
    qpos_layout,
    ref_base_dir,
    robots_dir,
    sidecar_file,
    split_combo,
    trees_dir,
)
from dexengine.data.contact import fingerprint as SF
from dexengine.data.contact.derive_steps import SIDES, _assign_from_layout, _candidates, collision_mod, derive_side
from dexengine.data.contact.derive_util import (
    DISCRIM_MIN_MM,
    DT,
    ERROR_HEAD,
    FAIL_HEAD,
    RESID_MAX_MM,
    MapFail,
    _atomic_savez,
    _obj_resid_mm,
    _obj_resid_tree_mm,
    _p_base_from,
    _placement,
    _stored_src_idx,
    ok_status,
)

#: Row statuses the idle-side synthesis reads back from the rows written above it (the same names).
NO_CANON, IDLE_NO_MAIN = f"{ERROR_HEAD}_NO_CANON", "IDLE_NO_MAIN"


# main() fills the five roots below from --combo/--dataset/--emb/--src-layer (_configure_roots). All are **full paths
# including the combo** (the trees layer puts an <emb> node after the combo, so the layer owner assembles it — the
# caller does not append `/ combo`).
CANON_DIR: Path = None  # contact sidecar canon (contact_base layer)


DATASET: str = ""  # the dataset axis (fingerprint check of the canon against the reference base)


HAND: str = ""  # the hand of the reference base (provenance label only)


TRAJ: Path = None  # reference universe (paths.ref_base_dir) — children = base clip directories


GEN_V2: Path = None  # this combo's tree root (paths.trees_dir)


ROBOTS: Path = None  # this combo's output root (paths.robots_dir)


OUT_ROOT: Path = None  # this combo's derived output (paths.contact_derive_dir)


def _configure_roots(combo: str, dataset: str, emb: str, src_layer: str) -> str:
    global CANON_DIR, TRAJ, GEN_V2, ROBOTS, OUT_ROOT, DATASET, HAND
    try:
        _arm, hand, _var = split_combo(combo, ARM_IDS, HAND_IDS)
    except ValueError as e:
        raise SystemExit(f"[contact_derive] {e}") from e
    DATASET, HAND = dataset, hand
    # every root comes from the layer owner as is — all layers live under the data root (paths.main_root)
    CANON_DIR = layer_path("contact_base", dataset)
    TRAJ = ref_base_dir(dataset, hand, emb, src_layer)
    GEN_V2 = trees_dir(combo, dataset, emb)
    ROBOTS = robots_dir(combo, dataset)
    OUT_ROOT = contact_derive_dir(combo, dataset)
    return hand


class _MapInputs(NamedTuple):
    """Inputs of one tree's time map (read and checked by _map_inputs)."""

    ti: dict
    qb: np.ndarray
    qt: np.ndarray
    T_base: int
    T_fin: int
    hqd: int
    ob_off: int
    starts_t: list
    tail: int
    P: int
    place: object
    base_place: object


def _map_inputs(tree: Path, base: str) -> _MapInputs:
    """Task info, reference/tree qpos, hand width, object columns (checked against the reference), pad P, tail trim and
    both placements of one tree."""
    ti = json.loads((tree / "task_info.json").read_text())
    base_tree = TRAJ / base
    kb = base_tree / "0" / "trajectory_kinematic.npz"
    kt = tree / "0" / "trajectory_kinematic.npz"
    if not kb.exists():  # an input deficit, not a structural property of the clip (ERROR_MAP)
        raise FileNotFoundError(f"base tree missing: {kb}")
    qb = np.load(kb)["qpos"].astype(np.float64)
    qt = np.load(kt)["qpos"].astype(np.float64)
    T_base, T_fin = len(qb), len(qt)
    hqd = int(ti.get("hand_qpos_dim", 18))
    # Object start column of the reference universe = hand count × H, owned by the qpos_layout reader (bimanual 2·H,
    # single-hand emb source 1·H: the tree gets an idle block so it is 2·H, but the reference stays as the source).
    _lay_b = qpos_layout.read(str(base_tree), nq=qb.shape[1])
    ob_off = len(_lay_b.hands) * _lay_b.hand_qpos_dim
    # Oracle: the slot-count arithmetic (D − 2H)//7 in _obj_resid_* below floors away a trailing joint column —
    # proceed only if it matches the statement (object_starts) (arctic: (60−52)//7 = 1 = 1 object, by check, not luck).
    _lay_t = qpos_layout.read(str(tree), nq=qt.shape[1])
    # Object columns are read only from the statement (object_starts) — in an articulated tree
    # `[object 7][joint 1][object 7]` 7-stride arithmetic starts the second object at the joint column (a 726 mm
    # residual), while the slot-count guard (D−2H)//7 passes by luck because its floor swallows the joint column.
    # The base and the tree differ in object columns only by the hand-block width (ob_off − 2H) — that is compared.
    starts_t = _lay_t.object_starts()
    if [c - 2 * hqd for c in starts_t] != [c - ob_off for c in _lay_b.object_starts()]:
        raise MapFail(
            f"object columns mismatch: tree {starts_t}(2H={2 * hqd}) vs base {_lay_b.object_starts()}(ob_off={ob_off})"
        )
    tail = int(ti.get("tail_trim_frames", 0) or 0)
    f0 = tree / "F0_REPAIR.json"
    P = 0
    if f0.exists():  # a marker that exists but cannot be read dies (never P = 0)
        P = int(json.loads(f0.read_text()).get("n_blend_effective", 0) or 0)
    # trimmed_tail_frames answers "how many frames longer than the reference the staging was, so it was cut", so
    # the post-bake time axis is already the reference (= base) length. A contract violation is caught here.
    trimmed = int(ti.get("trimmed_tail_frames", 0) or 0)
    if not 0 <= trimmed <= 2:
        raise MapFail(f"trimmed_tail_frames={trimmed} (contract is 0..2)")
    place = _placement(ti)
    base_place = _placement(json.loads((base_tree / "task_info.json").read_text()))
    return _MapInputs(ti, qb, qt, T_base, T_fin, hqd, ob_off, starts_t, tail, P, place, base_place)


def _xr_gates(name, order, T_pre, tau, m: _MapInputs, xr_p: Path, pre_rt_p: Path, tried, undeclared):
    """G2 on a cross-repair (xr) tree, in two stages: (resid, discrim, name + "+xr"), or None with the reason in tried.

    xr moves the wrist **with the grasped object** (carry_object) by up to 30mm, so the final object ≠ rigid placement
    (base) — 815 of the 2,130 canon trees hit G2 at 5~20mm. Raising the tolerance blinds the oracle (it overlaps the
    wrong map's residual, discrim ≈ 20mm). Instead it splits into two stages, and **each stage still expects 0**:
      G2a rigidity : the snapshot just before xr (pre_xr, pre-retiming time axis j) == fwd(base[j−P])
      G2b time map : the final tree == the snapshot just before retiming (pre_rt, after xr) resampled by tau
    """
    qb, qt, hqd, place, ob_off, starts_t = m.qb, m.qt, m.hqd, m.place, m.ob_off, m.starts_t
    T_base, T_fin, P = m.T_base, m.T_fin, m.P
    if order != "f0_then_rt":
        tried.append(f"{name}:XR trees support only f0_then_rt (order {order})")
        return None
    try:
        q_xr0 = np.load(xr_p)["qpos"].astype(np.float64)
        q_rt0 = np.load(pre_rt_p)["qpos"].astype(np.float64)
    except (OSError, KeyError) as e:
        undeclared.append(name)
        tried.append(f"{name}:XR snapshot read failed {e}")
        return None
    if len(q_xr0) != T_pre or len(q_rt0) != T_pre:
        tried.append(f"{name}:XR snapshot length pre_xr {len(q_xr0)} / pre_rt {len(q_rt0)} != T_pre {T_pre}")
        return None
    j = np.arange(T_pre, dtype=float)
    resid_a = _obj_resid_mm(np.clip(j - P, 0.0, T_base - 1.0), qb, q_xr0, hqd, place, ob_off, starts_t)
    if float(resid_a.max()) > RESID_MAX_MM:
        tried.append(f"{name}:XR-RIGID {resid_a.max():.4f}mm")
        return None
    p_pre = np.interp(np.arange(T_fin) * DT, tau, j)
    resid = _obj_resid_tree_mm(p_pre, q_rt0, qt, hqd, starts_t)  # G2b residual
    if float(resid.max()) > RESID_MAX_MM:
        tried.append(f"{name}:RESID(xr) {resid.max():.4f}mm")
        return None
    shifted = np.clip(p_pre + 2.0, 0.0, T_pre - 1.0)
    discrim = float(_obj_resid_tree_mm(shifted, q_rt0, qt, hqd, starts_t).max())
    name = name + "+xr"  # mark it in the ledger's strategy column
    return resid, discrim, name


def _map_info(tree: Path, name, order, T_pre, p_base, m: _MapInputs, resid, discrim) -> dict:
    """The accepted map's record: strategy, axes, residuals, weak-gate flag and the src_idx watch value."""
    ti, qb, qt, T_base, T_fin, hqd, ob_off, _, tail, P, place, base_place = m
    # Watch only: difference (frames) between the recorded src_idx and the map we used. Not an adoption criterion.
    si = _stored_src_idx(tree)
    if si is not None and len(si) == T_fin:
        sdev = float(np.abs(np.clip(si - P, 0.0, T_base - 1.0) - p_base).max())
    else:
        sdev = float("nan")
    info = {
        "strategy": name,
        "order": order,
        "T_base": T_base,
        "T_pre": T_pre,
        "P": P,
        "tail": tail,
        "T_fin": T_fin,
        "hqd": hqd,
        "ob_off": ob_off,
        "place": place,
        "base_place": base_place,
        "resid": resid,
        "discrim": discrim,
        "ti": ti,
        "qb": qb,
        "qt": qt,
        "weak": (discrim < DISCRIM_MIN_MM),
        "srcidx_dev": sdev,
    }
    return info


def _try_candidate(tree: Path, name, order, T_pre, mk, m: _MapInputs, xr_p: Path, pre_rt_p: Path, tried, undeclared):
    """One synthesized time map through G1 (length), G2 (object residual) and G2b (discriminative power):
    (p_base, info), or None with the reason appended to tried (undeclared also gets the name on an unexpected error)."""
    qb, qt, hqd, place, ob_off, starts_t = m.qb, m.qt, m.hqd, m.place, m.ob_off, m.starts_t
    T_base, T_fin, P = m.T_base, m.T_fin, m.P
    try:
        tau = mk(T_pre)
    except MapFail as e:
        tried.append(f"{name}:{e}")
        return None
    except Exception as e:  # noqa: BLE001  any other failure of one candidate is recorded as undeclared; the next runs
        undeclared.append(name)
        tried.append(f"{name}:{type(e).__name__}:{e}")
        return None
    T_new = int(np.floor(tau[-1] / DT + 1e-9)) + 1
    expect = T_new if order == "f0_then_rt" else T_new + P
    if expect != T_fin:  # G1 length
        tried.append(f"{name}:LEN {expect}!={T_fin}")
        return None
    p_base = _p_base_from(tau, T_pre, T_fin, P, order, T_base)
    if xr_p.exists():
        gates = _xr_gates(name, order, T_pre, tau, m, xr_p, pre_rt_p, tried, undeclared)
        if gates is None:
            return None
        resid, discrim, name = gates
    else:
        resid = _obj_resid_mm(p_base, qb, qt, hqd, place, ob_off, starts_t)  # G2 residual
        if float(resid.max()) > RESID_MAX_MM:
            tried.append(f"{name}:RESID {resid.max():.4f}mm")
            return None
        # G2b discriminative power: the residual of a map deliberately shifted by 2 frames
        shifted = np.clip(p_base + 2.0, 0.0, T_base - 1.0)
        discrim = float(_obj_resid_mm(shifted, qb, qt, hqd, place, ob_off, starts_t).max())
    info = _map_info(tree, name, order, T_pre, p_base, m, resid, discrim)
    return p_base, info


def build_map(tree: Path, base: str, combo: str):
    """(p_base, info) — synthesis + verification. MapFail on failure."""
    m = _map_inputs(tree, base)
    ti, T_base, T_fin, P = m.ti, m.T_base, m.T_fin, m.P
    tried, undeclared = [], []  # undeclared: a candidate died of something other than its own gates
    xr_p = tree / "0" / "trajectory_kinematic.pre_xr.npz"  # snapshot just before cross-repair
    pre_rt_p = tree / "0" / f"trajectory_kinematic{IK_SIDECAR_SFX}"  # snapshot just before retiming
    for name, order, T_pre, mk in _candidates(tree, base, combo, ti, T_base, T_fin, P):
        found = _try_candidate(tree, name, order, T_pre, mk, m, xr_p, pre_rt_p, tried, undeclared)
        if found is not None:
            return found
    if undeclared:  # an input deficit or an unexpected exception decided it — ERROR_MAP
        raise RuntimeError(f"undeclared failure during synthesis ({','.join(undeclared)}) — " + " ; ".join(tried))
    raise MapFail("synthesis failed — " + " ; ".join(tried) if tried else "no synthesis candidate")


def _canon_expected(base: str, info: dict, side: str) -> dict:
    """The fingerprint the canon of (base clip, side) must carry — computed from the reference base (TRAJ/base)."""
    bt = TRAJ / base
    if "base_ti" not in info:
        info["base_ti"] = json.loads((bt / "task_info.json").read_text())
    if "base_tin" not in info:
        info["base_tin"] = SF.TreeInputs(main_root(), DATASET, base, info["base_ti"], info["qb"], HAND)
    return info["base_tin"].expected(SF.LAYER_BASE, side)


def _row_head(base: str, side: str, info: dict, p_base) -> dict:
    """Ledger row of one side with the accepted map's columns (status is added by the caller)."""
    row = {
        "clip": base,
        "side": side,
        "strategy": info["strategy"],
        "order": info["order"],
        "T_base": info["T_base"],
        "T_pre": info["T_pre"],
        "P": info["P"],
        "tail": info["tail"],
        "T_fin": info["T_fin"],
        "resid_p50_mm": f"{np.percentile(info['resid'], 50):.3e}",
        "resid_max_mm": f"{info['resid'].max():.3e}",
        "discrim_mm": f"{info['discrim']:.2f}",
        "srcidx_dev_f": ("-" if info["srcidx_dev"] != info["srcidx_dev"] else f"{info['srcidx_dev']:.3e}"),
        "map_first": f"{p_base[0]:.3f}",
        "map_last": f"{p_base[-1]:.3f}",
    }
    return row


def _tree_assignment(ti: dict, side: str, mesh_res: str, slot_res: int, layout_owned: bool) -> tuple:
    """(mesh, slot column) the tree states for one side, to compare with the canon and the resolver."""
    if layout_owned:
        # base-layer tree: the sole owner of the assignment is objects[] (qpos_layout). The legacy
        # `<side>_object_mesh_dir/_slot` is a forwarded copy (duplicated statement), so after supplementary parts
        # or reassignment (D1-e·D6) it is None or stale — not a comparison target
        # (comparing it gives FAIL_ASSIGN false alarms). Compare canon ↔ resolver only.
        mesh_tree, slot_ti = mesh_res, slot_res
    else:
        mesh_tree = str(ti.get(f"{side}_object_mesh_dir", "")).split("/")[-1]
        # the legacy statement is a slot number — convert to a column to compare (the resolver returns columns)
        _ls = ti.get(f"{side}_object_slot")
        slot_ti = (2 * int(ti.get("hand_qpos_dim", 18)) + 7 * int(_ls)) if _ls is not None else slot_res
    return mesh_tree, slot_ti


def _zero_idle(payload: dict, base: str, side: str) -> None:
    """D16 single-hand rule: the idle hand is parked (beside the torso) and never touches the object — all
    contact/penetration labels 0, recorded with a marker."""
    for k in ("contact", "penetration", "contact_bin", "palm_penetration"):
        if k in payload:
            payload[k] = np.zeros_like(payload[k])
    payload["derive_idle"] = np.int32(1)
    print(f"[IDLE_SIDE] contact_derive {base} [{side}]: idle side → contact and penetration 0", flush=True)


def _side_gates(
    tree: Path, side: str, base: str, info: dict, assign: dict, layout_owned: bool, canon_p: Path, row: dict
) -> bool:
    """Fill row's canon columns and run the gates before derivation: idle side without a main object, canon input
    fingerprint, G3 left/right assignment. True = the row is final (its status is set)."""
    ti = info["ti"]
    mesh_res, slot_res = assign[side]
    mesh_tree, slot_ti = _tree_assignment(ti, side, mesh_res, slot_res, layout_owned)
    with np.load(canon_p) as z:
        mesh_canon = str(z["tool_mesh_id"])
        base_contact = int(np.any(z["contact"], axis=(1, 2)).sum())
    row.update(mesh_canon=mesh_canon, mesh_tree=mesh_tree, slot_tree=slot_res, contact_frames_base=base_contact)
    # a hand with no main object is treated as an idle hand
    _side_idle = side in collision_mod().idle_sides_of_tree(tree) or assign[side] == ("", -1)
    if assign[side] == ("", -1):
        # D23: the idle side has no main object (no column in the tree) — with no object pose to derive from,
        # derive is not called (it would die on a 0-norm quaternion); _synthesize_idle borrows the active side's
        # structure to make 0 contact (the same path as a single-hand emb).
        row["status"] = IDLE_NO_MAIN
        return True
    # The canon is combo-independent but is built from **one hand's base** object poses. It must carry the
    # same input fingerprint as this combo's reference base (object-pose generation · MANO · code version) — a
    # canon made from an older base derives contact on the old poses. Unknown/stale/missing → fail, not pass.
    # The idle side (D16) is exempt for the same reason as G3 below: its contact is zeroed, the canon is not used.
    try:
        if not _side_idle:
            SF.check(canon_p, _canon_expected(base, info, side), ds=DATASET)
    except (SF.SidecarFingerprintError, ValueError, KeyError, OSError) as e:
        row["status"] = f"{ERROR_HEAD}_SIDECAR_FP:{type(e).__name__}:{e}"[:400]  # stale input: re-derive
        return True
    # G3 left/right assignment: canon · task_info · resolver must all agree. **The idle side (D16/D23) is not
    # compared** — with no main object the resolver returns ""/-1, and the canon sidecar still carries the old
    # assignment (e.g. the shared grab bowl), so it always disagrees. Contact is synthesized as 0 below.
    if (not _side_idle) and (not (mesh_canon == mesh_tree == mesh_res) or slot_ti != slot_res):
        row["status"] = (
            f"{FAIL_HEAD}_ASSIGN:canon={mesh_canon},ti={mesh_tree},res={mesh_res}@{slot_res},ti_slot={slot_ti}"
        )
        return True
    return False


def _side_row(
    tree: Path, side: str, base: str, info: dict, p_base, assign: dict, layout_owned: bool, in_place, overwrite
):
    """(row, (dst, payload) or None) of one side: canon present, the gates of _side_gates, then the derived labels."""
    # location = dexengine.core.layout.sidecar_rel
    canon_p = sidecar_file("contact_base", DATASET, base, side, hand_spec_dir(HAND))
    row = _row_head(base, side, info, p_base)
    if not canon_p.exists():
        row["status"] = NO_CANON
        return row, None  # input deficit
    if _side_gates(tree, side, base, info, assign, layout_owned, canon_p, row):
        return row, None
    dst = tree / "0" / (f"contact_mano_{side}.npz" if in_place else f"contact_mano_{side}.derived.npz")
    if dst.exists() and not overwrite and not in_place:
        row["status"] = "SKIP_HAVE"
        return row, None
    payload, and_dropped, canon = derive_side(canon_p, p_base, info, side)
    if side in collision_mod().idle_sides_of_tree(tree):
        _zero_idle(payload, base, side)
    n_contact = int(np.any(payload["contact"], axis=(1, 2)).sum())
    row.update(
        contact_frames_out=n_contact,
        contact_frac_out=round(n_contact / max(info["T_fin"], 1), 4),
        pen_frames_out=int(np.any(payload["penetration"], axis=(1, 2)).sum()),
        and_dropped=and_dropped,
        status=ok_status(n_contact, weak=info["weak"]),
    )
    return row, (dst, payload)


def _idle_payload(ref_payload: dict, side: str, ref_side: str) -> dict:
    """0-contact payload for an idle side with the active side's structure (dist 1 m = no contact)."""
    synth = {}
    for k, v in ref_payload.items():
        v = np.asarray(v)
        if k in ("contact", "penetration", "contact_bin", "palm_penetration"):
            synth[k] = np.zeros_like(v)
        elif k in ("dist", "effective_dist", "palm_dist"):
            synth[k] = np.full_like(v, 1.0)  # 1 m = no contact
        elif k.startswith(("witness", "palm_witness")):
            synth[k] = np.zeros_like(v)
        elif k == "side":
            synth[k] = np.array(side)
        else:
            synth[k] = v
    synth["derive_idle"] = np.int32(1)
    synth["derive_synth_from"] = np.array(ref_side)
    synth["derive_rule"] = np.array(
        "idle_synth(zero contact; 2a hand absent from the one-hand emb source / D23 idle side has no main object)"
    )
    return synth


def _synthesize_idle(tree: Path, rows: list[dict], payloads: dict, assign: dict, base: str, in_place: bool) -> None:
    """Single-hand emb: an idle hand absent from the source (NO_CANON) or without a main object gets a **0 contact**
    file built from the active side's payload structure (the gate requires contact_mano_<side>.npz on both sides);
    its row and payloads entry are filled in. A missing file on a non-idle side still fails."""
    _idle = set(collision_mod().idle_sides_of_tree(tree)) | {
        s for s in SIDES if assign[s] == ("", -1)
    }  # idle sides ∪ hands with no main object
    for row in rows:
        if row.get("status") not in (NO_CANON, IDLE_NO_MAIN) or row["side"] not in _idle or not payloads:
            continue
        side = row["side"]
        ref_side, (_, ref_payload) = next(iter(payloads.items()))
        synth = _idle_payload(ref_payload, side, ref_side)
        dst = tree / "0" / (f"contact_mano_{side}.npz" if in_place else f"contact_mano_{side}.derived.npz")
        payloads[side] = (dst, synth)
        row.update(
            status=ok_status(0, idle_synth=True),
            contact_frames_out=0,
            contact_frac_out=0.0,
            pen_frames_out=0,
            mesh_canon="-",
            mesh_tree="-",
            slot_tree="-",
            contact_frames_base=0,
        )
        print(
            f"[IDLE_SIDE] contact_derive {base} [{side}]: hand absent from the source → 0 contact synthesised "
            f"with the {ref_side} structure",
            flush=True,
        )


def process_tree(tree: Path, combo: str, apply: bool, in_place: bool, overwrite: bool) -> list[dict]:
    """Ledger rows (one per side) of one combo tree: time map, per-side derivation, idle-side synthesis and, with
    apply, the atomic write of each payload."""
    base = tree.name[: -len(TREE_SFX)] if tree.name.endswith(TREE_SFX) else tree.name
    rows: list[dict] = []
    try:
        p_base, info = build_map(tree, base, combo)
    except MapFail as e:
        return [{"clip": base, "side": s, "status": f"{FAIL_HEAD}_MAP:{e}"} for s in SIDES]
    except Exception as e:  # noqa: BLE001 — undeclared: ERROR (re-bake debt), never the terminal quarantine
        return [{"clip": base, "side": s, "status": f"{ERROR_HEAD}_MAP:{type(e).__name__}:{e}"} for s in SIDES]
    info["base_clip"] = base
    try:
        assign = _assign_from_layout(tree, info["qt"].shape[1])
    except Exception as e:  # noqa: BLE001 — a layout the reader refuses is an artifact defect: ERROR
        return [{"clip": base, "side": s, "status": f"{ERROR_HEAD}_RESOLVE:{type(e).__name__}:{e}"} for s in SIDES]
    info["slot"] = {s: assign[s][1] for s in SIDES}
    layout_owned = (
        qpos_layout.read(str(tree), nq=info["qt"].shape[1]).source == "task_info.layout"
    )  # base layer = owned by objects[] (legacy fields ignored)

    payloads = {}
    for side in SIDES:
        row, out = _side_row(tree, side, base, info, p_base, assign, layout_owned, in_place, overwrite)
        if out is not None:
            payloads[side] = out
        rows.append(row)
    _synthesize_idle(tree, rows, payloads, assign, base, in_place)
    if apply:
        for side, (dst, payload) in payloads.items():
            # No replacement backup (.pre_derive.npz) is kept — a backup of a derived file is a second source.
            # It re-derives from the canon (contact_mano_base) to the same result; the ledger holds the history.
            _atomic_savez(dst, payload)
            print(
                f"[contact_derive] APPLY{'(IN-PLACE)' if in_place else ''} {base} [{side}] → {data_rel(dst)}",
                flush=True,
            )
    return rows
