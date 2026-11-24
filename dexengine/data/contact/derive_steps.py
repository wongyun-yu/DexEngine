"""Steps of the contact derivation (contact.derive): arm and collision owners, time-map candidates, deriving one side
from the canon, the side assignment from the layout, and the run report."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from dexengine.core.paths import ARM_IDS, HAND_IDS, data_rel, mode_record, qpos_layout, split_combo
from dexengine.data import side_objects as SO  # noqa: E402  hand↔object assignment canon
from dexengine.data.contact.derive_util import (
    BOOL_KEYS,
    DT,
    ERROR_HEAD,
    FAIL_HEAD,
    NON_TIME_KEYS,
    MapFail,
    _fwd,
    _inv,
    _obj_pose,
    _tau_stored,
)

SIDES = qpos_layout.SIDES  # owner of the side vocabulary
_ARM_REG = None


def arm_registry():
    """ArmSpec canon (velocity limits) — loaded only once."""
    global _ARM_REG
    if _ARM_REG is None:
        from dexengine.core import paths as _paths  # noqa: PLC0415

        _ARM_REG = _paths.load_registry("arm_specs").ARM_REGISTRY
    return _ARM_REG


_COLL = None


def collision_mod():
    """The owner module of the idle side (D16), collision.idle_arm.idle_sides_of_tree — values are not restated here."""
    global _COLL
    if _COLL is None:
        from dexengine.placement.collision import idle_arm as m  # noqa: PLC0415  (placement sits above data)

        _COLL = m
    return _COLL


def _arm_of(combo: str) -> str:
    arm = split_combo(combo, ARM_IDS, HAND_IDS)[0]  # the one combo parser (paths); ValueError when it cannot split
    if arm not in arm_registry():
        raise KeyError(f"ArmSpec not registered: {arm} (combo {combo})")
    return arm


def _refuse(why: str):
    raise MapFail(why)


def _tau_inherit(pl_name: str, ik_suffix: str, safety: float, T_pre: int):
    """legacy inherited warp — REMOVED.

    It replayed retime_to_limits.compute_warp over the removed first arm's global IK caches and
    velocity caps. A tree that can only be reproduced this way now fails this candidate loudly
    (MapFail with the reason — the caller records it and tries the stored/v5/combo maps)."""
    raise MapFail(
        f"the inherited warp (legacy arm global IK cache·velocity limits, {pl_name} suffix={ik_suffix!r}) "
        "was removed with that arm — only stored time maps·v5·combo warps can be reproduced"
    )


def _candidates(tree: Path, base: str, combo: str, ti: dict, T_base: int, T_fin: int, P: int):
    """Candidate list of (name, order, T_pre, tau builder) — only what the recorded sources allow."""
    _arm_of(combo)  # the combo's arm must be registered (KeyError otherwise)
    safety = float(ti.get("retime_safety", 1.15))
    tail = int(ti.get("tail_trim_frames", 0) or 0)
    out = []
    # ① The time map **stored** in the tree wins — it is the original record, not a reproduction,
    #    and it does not depend on the sidecar (`*.pre_rt.npz`), so one tree is self-contained.
    #    It is adopted only after passing the same G1 (length) · G2 (object residual) gates.
    try:
        tau_st, T_pre_st = _tau_stored(tree)
    except MapFail:
        tau_st = None
    if tau_st is not None:
        # v5 chain invariant: length just before retiming == base − tail trim + pad
        if T_pre_st == T_base - tail + P:
            out.append(("stored", "f0_then_rt", T_pre_st, lambda tp, t=tau_st: t))
    T_pre_legacy = T_base - tail
    rv2 = str(ti.get("retime_v2", ""))
    if rv2 == "combo":
        # The arm-only combo warp (stage IK) was deleted with its bake path (one retiming implementation) —
        # a tree baked that way has no reproducible map here: refuse with the reason rather than "no candidate".
        out.append(
            (
                "combo",
                "rt_then_f0",
                T_pre_legacy,
                lambda tp: _refuse(
                    "the combo warp (stage IK · arm only) was removed with the bake path — only stored time "
                    "maps can be reproduced"
                ),
            )
        )
    if rv2 in ("inherit", ""):
        pl_name = ti.get("retimed_from", f"{base}_pl")
        suffix = ti.get("retime_ik_suffix", "")
        out.append(("inherit", "rt_then_f0", T_pre_legacy, lambda tp: _tau_inherit(pl_name, suffix, safety, tp)))
    if rv2 == "none":
        # A tree with no retiming at all (bake only) — identity tau.
        out.append(("none", "rt_then_f0", T_pre_legacy, lambda tp: np.arange(tp, dtype=float) * DT))
    return out


def derive_side(canon_path: Path, p_base: np.ndarray, info: dict, side: str):
    """Canon npz → tree-time-axis payload (boolean AND · continuous linear)."""
    with np.load(canon_path, allow_pickle=True) as z:
        canon = {k: z[k] for k in z.files}
    T_base = info["T_base"]
    T_fin = len(p_base)
    lo = np.clip(np.floor(p_base).astype(int), 0, T_base - 1)
    hi = np.clip(np.ceil(p_base).astype(int), 0, T_base - 1)
    w = (p_base - lo).astype(np.float64)

    # witness_obj is a **point on the object surface**. Interpolating it directly in scene coordinates leaves the
    # surface wherever the object rotates (measured p50 0.4mm · max 25mm). Moving it into object-local coordinates,
    # interpolating there and restoring with **the tree's object pose** keeps it on the surface and fixes placement too.
    Rb = pb = Rt = pt = None
    if info.get("qb") is not None:
        col = int(info["slot"][side])  # main-object start column (tree coords)
        Rb, pb = _obj_pose(info["qb"], info["ob_off"] + (col - 2 * info["hqd"]))
        Rt, pt = _obj_pose(info["qt"], col)

    out: dict[str, np.ndarray] = {}
    and_dropped = 0
    for k, v in canon.items():
        v = np.asarray(v)
        if k in NON_TIME_KEYS or v.ndim == 0 or v.shape[0] != T_base:
            out[k] = v  # scalars, names, thresholds: as is
            continue
        if k == "frames":
            out[k] = np.arange(T_fin, dtype=v.dtype)
            continue
        if v.dtype == np.bool_ or k in BOOL_KEYS:
            a, b = v[lo], v[hi]
            out[k] = a & b  # AND rule (conservative)
            if k == "contact":
                and_dropped += int((np.any(a | b, axis=(1, 2)) & ~np.any(a & b, axis=(1, 2))).sum())
            continue
        if v.dtype.kind not in "fiu":
            out[k] = v
            continue
        out[k] = _interp_continuous(k, v, lo, hi, w, (Rb, pb, Rt, pt), info)

    _stamp_derivation(out, p_base, info, T_base, Rb)
    return out, and_dropped, canon


def _interp_continuous(k: str, v: np.ndarray, lo, hi, w, poses: tuple, info: dict) -> np.ndarray:
    """One continuous channel on the tree time axis: linear interpolation between the bracketing source frames;
    object witnesses go through object-local coordinates, other witness positions get the placement re-applied."""
    Rb, pb, Rt, pt = poses
    T_base, T_fin = len(v), len(lo)
    f = v.astype(np.float64)
    shp = (T_fin,) + f.shape[1:]
    wb = w.reshape((T_fin,) + (1,) * (f.ndim - 1))
    if k.endswith("witness_obj") and Rb is not None:
        # interpolate in object-local coordinates → restore with the tree's object pose (stays on the surface)
        loc = np.einsum(
            "tij,t...j->t...i", np.transpose(Rb, (0, 2, 1)), f - pb.reshape((T_base,) + (1,) * (f.ndim - 2) + (3,))
        )
        li = loc[lo] * (1.0 - wb) + loc[hi] * wb
        r = np.einsum("tij,t...j->t...i", Rt, li) + pt.reshape((T_fin,) + (1,) * (f.ndim - 2) + (3,))
    else:
        r = f[lo] * (1.0 - wb) + f[hi] * wb  # linear interpolation
        if "witness" in k and r.shape[-1] == 3:  # position → re-apply the placement
            r = _fwd(_inv(r, *info["base_place"]), *info["place"])
    return r.astype(v.dtype).reshape(shp)


def _stamp_derivation(out: dict, p_base: np.ndarray, info: dict, T_base: int, Rb) -> None:
    """Record how the payload was derived: rule, map strategy/order, base clip, map, residual, pad, trim, placement."""
    out["derive_rule"] = "bool=AND(bracket) cont=linear witness_hand=scene " + (
        "witness_obj=obj_local" if Rb is not None else "witness_obj=scene"
    )
    out["derive_strategy"] = info["strategy"]
    out["derive_order"] = info["order"]
    out["derive_base_clip"] = info["base_clip"]
    out["derive_map"] = p_base.astype(np.float32)
    out["derive_obj_resid_mm"] = info["resid"].astype(np.float32)
    out["derive_T_base"] = np.int32(T_base)
    out["derive_pad"] = np.int32(info["P"])
    out["derive_tail_trim"] = np.int32(info["tail"])
    out["derive_placement"] = np.asarray(
        [info["place"][0], info["place"][1], info["place"][2], info["place"][3][0], info["place"][3][1]],
        dtype=np.float64,
    )


def _assign_from_layout(tree: Path, nq: int) -> dict[str, tuple[str, int]]:
    """{'right': (mesh id, slot), 'left': (…)} — owner = qpos_layout (tree task_info).

    base-layer tree: the main object from objects[] (with a shared object both sides may be on the **same column** —
    legal). Legacy tree (no qpos_layout): the reader's legacy branch
    reproduces the legacy slot convention, and the mesh id is read from the scene.xml object order
    (side_objects.scene_object_ids — TACO legacy only).
    slot = (main-object column − 2·H)/7 — raises if the objects are not contiguous 7-blocks after the hands
    (no silent default-layout fallback).
    """
    lay = qpos_layout.read(str(tree), nq=nq)
    h2 = 2 * lay.hand_qpos_dim
    layout = lay.source == "task_info.layout"
    ids_by_col = {int(o["qpos_start"]): str(o.get("obj_id") or "") for o in lay.objects} if layout else {}
    scene_ids = [] if layout else SO.scene_object_ids(tree)
    out: dict[str, tuple[str, int]] = {}
    for side in SIDES:
        st = lay.main.get(side)
        if st is None:
            # No main object = an idle hand (2a single-hand emb) **or** a hand active under D23 but with 0 contact
            # (it is in case.active, so idle_sides does not catch it).
            # Neither case has an object pose to derive from, so 0 contact is synthesized — the same criterion as the
            # manipulating-hand owner (Layout.manip_sides = hand ∧ main object).
            out[side] = ("", -1)
            continue
        # The second value is the **main-object start column** (tree coordinates) — a slot (st−2H)/7 is
        # undefined in an articulated tree where a joint column is interleaved.
        if layout:
            mid = ids_by_col.get(st, "")
        else:
            if (st - h2) % 7:
                raise ValueError(
                    f"{tree.name}: {side} main-object column {st} does not follow the legacy 2H+7k convention "
                    f"(H={lay.hand_qpos_dim})"
                )
            k = (st - h2) // 7
            mid = scene_ids[k] if k < len(scene_ids) else ""
        out[side] = (mid, st)
    return out


def _report(a, rows: list[dict], ledger: Path, out_dir: Path, mode: str | None, cum: tuple, t0: float) -> int:
    """This run's counts → the human line, the machine summary (--summary-json) and the rc (1 = any FAIL_* / ERROR_*).
    ERROR clips (any side) over this run's trees that are systematic (ledger_states.error_systematic — one owner with
    the gate) print [CD_ERROR_FAIL] and set error_systematic: stage2 then stops without writing the status.
    ``ledger`` = the record this run wrote (mode-tagged for a DRY run) · ``mode`` names the lists the same way."""
    from dexengine.placement import ledger_states as LS  # noqa: PLC0415

    ok, fails, errors, zeros, weak = cum
    local_fails = [r for r in rows if str(r.get("status", "")).startswith(FAIL_HEAD)]
    local_errs = [r for r in rows if str(r.get("status", "")).startswith(ERROR_HEAD)]
    err_clips, n_clips = len({r["clip"] for r in local_errs}), len({r["clip"] for r in rows})
    systematic = LS.error_systematic(err_clips, n_clips)
    if systematic:
        print(
            f"[CD_ERROR_FAIL] errors (missing input · undeclared exception) {err_clips}/{n_clips} clips — "
            "systematic (≥ "
            f"{LS.ERROR_SYSTEMATIC_MIN} clips · all or ≥ {LS.ERROR_SYSTEMATIC_FRAC:.0%}) · not a clip verdict",
            flush=True,
        )
    print(
        f"[contact_derive] {a.combo}: this run {len(rows) - len(local_fails) - len(local_errs)}/{len(rows)} · "
        f"cumulative OK {ok} · failed (cumulative) {len(fails)} · errors (cumulative) {len(errors)} · zero "
        f"contact (cumulative) {len(zeros)} · "
        f"weak gate (cumulative) {len(weak)} · {time.time() - t0:.1f}s",
        flush=True,
    )
    if a.summary_json:
        _write_summary_json(a, rows, local_fails, local_errs, systematic, cum)
    print(
        f"[contact_derive] ledger {data_rel(ledger)}"
        + (f" · failed (cumulative) {data_rel(mode_record(out_dir / 'fails.txt', mode))}" if fails else "")
        + (f" · errors (cumulative) {data_rel(mode_record(out_dir / 'errors.txt', mode))}" if errors else ""),
        flush=True,
    )
    return 1 if local_fails or local_errs else 0


def _write_summary_json(a, rows: list[dict], local_fails: list, local_errs: list, systematic: bool, cum: tuple) -> None:
    """--summary-json: this run's and the cumulative counts as JSON, written atomically (temp + rename)."""
    ok, fails, errors, zeros, weak = cum
    # this JSON owns the counts; the line above is for humans, so changing its format never breaks a consumer.
    sp = Path(a.summary_json)
    tmp = sp.with_name(sp.name + f".tmp{os.getpid()}")
    tmp.write_text(
        json.dumps(
            {
                "combo": a.combo,
                "dataset": a.dataset,
                "emb": a.emb,
                "src_layer": a.src_layer,
                "run_total": len(rows),
                "run_ok": len(rows) - len(local_fails) - len(local_errs),
                "run_fail": len(local_fails),
                "run_error": len(local_errs),
                "error_systematic": systematic,
                "cum_ok": ok,
                "cum_fail": len(fails),
                "cum_error": len(errors),
                "cum_zero": len(zeros),
                "cum_weak": len(weak),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    os.replace(tmp, sp)
    print(f"[CONTACT_DERIVE_SUMMARY] {sp}", flush=True)
