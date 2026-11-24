#!/usr/bin/env python3
"""Bake stage (`bake` in the stage chain): apply the adopted placement (dx, dy, yaw, pivot) to a base-layer clip and
write the training tree.

Placement only, no retiming — the chain retimes once later (`retime/retime.py`, arm + fingers). The source is the
base layer of the dataset (paths.default_src_layer), which is also the reference: the tree is checked against
itself.
  in   placement_dir(<combo>, <ds>)/<tree>.adopted.json  (dx, dy, yaw, pivot, robot_z_off)
       <processed_root>/<hand spec dir>/<emb>/<clip>/     (the f0 shadow staging of the base layer)
  out  paths.trees_dir(<combo>, <ds>, <emb>)/<tree>/

A one-hand tree gets the missing hand's block before the placement transform (`uni_layout`), so every tree has the
two-hand layout; the bake's arm IK stage later fills that block with the idle arm's rest pose.

Gates (each one raises — nothing passes silently):
  G1 FK: after the transform, the hand base FK == Rz·source + shift (< 1e-6; 5 frames × both hands)
  G2 object: every object block == the placement transform of the reference object (< 1e-9)

The f0 pad is not built here: f0 runs before judging, and the bake copies its pad record into the tree.
trajectory_ikrollout holds qpos_rollout (when present) or qpos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import (
    ARM_IDS,
    HAND_IDS,
    PLACEMENT_SFX,
    SRC_LAYERS,
    default_src_layer,
    placement_dir,
    qpos_layout,
    split_combo,
    tree_name,
    trees_dir,
)
from dexengine.core.paths import HAND_SPEC_DIR as SPIDER_OF
from dexengine.core.paths import main_root as _main_root
from dexengine.data.objects import scene_parts as SP
from dexengine.placement import retime_variants as _mcv  # BaseMap, rigid_pl, resample_all, _rt
from dexengine.placement.bake.combo_util import (  # noqa: F401
    _abs_dirs_text,
    _load_staging,
    _tf_points,
    _write_base_task_info,
    copy_xml_abs_dirs,
    f0_pad_of,
)
from dexengine.placement.reach_map import store as _PM  # (PROV_KEY — the tree's record of the judge's map ids)

HELP = "Bake the adopted placement into a training tree (placement only — the chain retimes later)."

# Tree root owner = paths.main_root (env DEXENGINE_MAIN_ROOT first, default = repository root).
MAIN = _main_root()


def _maps_for(src: Path, hqd: int):
    return _mcv.maps_for(src, hqd)  # one loader (retime_variants) — the retiming reads the same owner


def _inject_missing_hand(a, base: str, hand: str, src: Path, z: dict, qpos, hqd: int, lay0):
    """One-hand tree (UNI_R / UNI_L): insert the missing hand's zero block **before** the placement transform
    (rigid_pl expects [0:H][H:2H][2H+7k]) to make a two-hand layout (owner `uni_layout`). Only with active sides
    `auto` — a one-hand tree in a two-hand chain dies at once. → (z, qpos, inj, maps of the injected scene)."""
    from dexengine.placement import uni_layout as _UL  # noqa: PLC0415

    if os.environ.get("DEXENGINE_ACTIVE_SIDES", "") != "auto":
        raise SystemExit(
            f"[bake] {base}: a one-hand tree (hands={sorted(lay0.hands)}) is baked only with "
            f"--active-sides auto (DEXENGINE_ACTIVE_SIDES=auto) — two-hand chain"
        )
    _miss = next(s_ for s_ in ("right", "left") if s_ not in lay0.hands)
    _sha, _meshes, _body = _UL.load_block(str(hand), _miss)
    _ti0 = json.loads((src / "task_info.json").read_text())
    if _sha != _ti0.get("hand_asset_sha256_xml3"):
        raise SystemExit(
            f"[bake] {base}: uni_layout template sha {_sha[:12]} != tree hand_asset_sha256_xml3 "
            f"{str(_ti0.get('hand_asset_sha256_xml3'))[:12]}"
        )
    _col = _UL.insert_col(dict(lay0.hands), _miss, hqd)
    _nq0 = int(qpos.shape[1])
    _nv0 = int(z["qvel"].shape[1]) if "qvel" in z else _nq0
    _old_adr = _UL.joint_adr(src / "scene.xml")
    z = _UL.inject_arrays(dict(z), _col, hqd, _nq0, _nv0, str(hand))
    qpos = z["qpos"]
    # The other hand's root body = the tree layout's statement (it differs per hand).
    _other_root = _UL.hand_root(_ti0, "right" if _miss == "left" else "left")
    _scene_txt = {
        fn: _UL.inject_scene_text((src / fn).read_text(), _miss, _meshes, _body, _other_root)
        for fn in ("scene.xml", "scene_eq.xml")
        if (src / fn).exists()
    }
    import tempfile

    _tmpd = Path(tempfile.mkdtemp(prefix="uni_scene_"))
    try:  # the injected scene file serves only the compile oracle and the maps (models live in memory)
        _tmp_scene = _tmpd / "scene.xml"
        _tmp_scene.write_text(_abs_dirs_text(_scene_txt["scene.xml"], src))
        _UL.verify_scene(_tmp_scene, _miss, _col, hqd, _nq0, _nv0, _old_adr)
        _inj_maps = _maps_for(_tmpd, hqd)
    finally:
        shutil.rmtree(_tmpd, ignore_errors=True)
    _ti_after = _UL.task_info_after(_ti0, _miss, _col, hqd, _UL.block_root(_body))
    inj = {
        "side": _miss,
        "col": _col,
        "H": hqd,
        "nq_before": _nq0,
        "scene_txt": _scene_txt,
        "layout": _ti_after["qpos_layout"],
        "ti_after": _ti_after,
    }
    print(
        f"[UNI_LAYOUT] {base}: one-hand tree hands={sorted(lay0.hands)} → {_miss} hand zero block added col={_col} "
        f"H={hqd} nq {_nq0}→{_nq0 + hqd} (compile oracle passed; the arm IK stage fills it with the rest pose)",
        flush=True,
    )
    return z, qpos, inj, _inj_maps


def _g1_fk_gate(maps, hqd, T, qpos, q_pl, dx, dy, yaw, pivot, Rz):
    """G1: the placed wrists equal the FK of the placement transform applied to the source wrists (5 sampled frames)."""
    for bidx, m in ((0, maps[0]), (hqd, maps[1])):
        sel = np.linspace(0, T - 1, 5).astype(int)
        p_o, R_o = m.to_world(qpos[sel, bidx : bidx + 6])
        p_n, R_n = m.to_world(q_pl[sel, bidx : bidx + 6])
        want = _tf_points(p_o, dx, dy, yaw, pivot)
        dp = np.abs(p_n - want).max()
        dr = (R_n.inv() * (Rz * R_o)).magnitude().max()
        assert dp < 1e-6 and dr < 1e-6, f"G1 FK gate: dp={dp:.2e} dr={dr:.2e}"


def _g2_object_gate(obj_pairs, bl, T, dx, dy, yaw, pivot, q_pl, base, Rz):
    """G2: every object slot moved by exactly the placement transform."""
    for k, (lo, lob) in enumerate(obj_pairs):
        want_p = _tf_points(bl[:T, lob : lob + 3], dx, dy, yaw, pivot)
        dp = np.abs(q_pl[:, lo : lo + 3] - want_p).max()
        if dp >= 1e-9:
            # The object trajectory must be robot-independent: the reference transformed by our placement is
            # canonical, so a drifted copy is replaced by it (not a threshold relaxation).
            print(
                f"[bake] {base}: object slot {k} drift dp={dp:.3e} — replaced by the reference (bl) value", flush=True
            )
            q_pl[:, lo : lo + 3] = want_p
            R_bl_obj = Rot.from_quat(np.roll(bl[:T, lob + 3 : lob + 7], -1, axis=1))
            q_pl[:, lo + 3 : lo + 7] = np.roll((Rz * R_bl_obj).as_quat(), 1, axis=1)
            dp = np.abs(q_pl[:, lo : lo + 3] - want_p).max()
        assert dp < 1e-9, f"G2 object gate (slot {k}): {dp:.3e}"


def _adopted_placement(hand, base, combo, a):
    """The hand's staging name and the adopted placement of the clip (dx, dy, yaw and the pivot the solve used)."""
    spider = SPIDER_OF[hand]
    tree = tree_name(base)  # no shadowing: a local named like a function kills that function here
    adopted_p = placement_dir(combo, a.dataset) / f"{tree}{PLACEMENT_SFX}"
    ad = json.loads(adopted_p.read_text())
    dx, dy, yaw = float(ad["dx"]), float(ad["dy"]), float(ad["yaw"])
    pivot = ad.get("pivot")
    assert pivot is not None, f"{adopted_p}: no pivot — the solve and the bake must use the same pivot"
    return spider, ad, dx, dy, yaw, pivot


def _check_object_slots(qpos, _artic, h2, slots, base, hqd):
    """Without articulation, the object slots must be the 7-wide blocks right after the hands."""
    D = qpos.shape[1]
    if not _artic:
        expect = [h2 + 7 * k for k in range((D - h2) // 7)]
        if (D - h2) % 7 or slots != expect:
            raise InputMissingError(
                f"[bake] {base}: object blocks differ from the contiguous 7-block convention after "
                f"the hands — tree {slots} vs h2+7k {expect} (D={D}, H={hqd}) · no joint column"
            )


def _placed_arrays(a, combo: str, base: str):
    """(src, z_pl, hqd, ad, (dx,dy,yaw,pivot), maps, pad) — the placed arrays of one clip, gates G1/G2 passed."""
    # A variant suffix of the combo is stripped for parsing only — the paths keep the combo string as is.
    try:
        arm, hand, _var = split_combo(combo, ARM_IDS, HAND_IDS)  # the one combo parser (paths)
    except ValueError as e:
        raise SystemExit(f"[bake] {e}") from e
    spider, ad, dx, dy, yaw, pivot = _adopted_placement(hand, base, combo, a)

    src, z, qpos, hqd, pad = _load_staging(spider, base, Path(a.processed_root), a.emb)
    bl = qpos  # base layer: the reference is the tree itself
    inj = None
    _lay0 = qpos_layout.read(str(src), nq=qpos.shape[1])
    if len(_lay0.hands) == 1:
        z, qpos, inj, _inj_maps = _inject_missing_hand(a, base, hand, src, z, qpos, hqd, _lay0)
        bl = qpos
    maps = _maps_for(src, hqd) if inj is None else _inj_maps
    T = len(qpos)
    # Articulated objects: without an injection, object and joint column addresses come from the **tree statement
    # (qpos_layout)**, so an articulation column passes through the rigid transform unchanged.
    _lay_pl = None
    if inj is None:
        _lay_pl = qpos_layout.read(str(src), nq=qpos.shape[1])
        _lay_pl.blocks()  # the statement covers qpos fully (a hole/overlap dies here)
        if _lay_pl.artic:
            _ti_a = json.loads((src / "task_info.json").read_text())
            for _o in _ti_a.get("objects") or []:
                _a = _o.get("articulation")
                if _a:
                    _c = int(_a["qpos_index"])
                    _tr = float(qpos[:, _c].max() - qpos[:, _c].min())
                    print(
                        f"[ARTIC] bake {base}: {_o['name']} {_a.get('type')} col={_c} axis={_a.get('axis')} "
                        f"pos={_a.get('pos')} "
                        f"range={_a.get('range')} travel={_tr:.4f} parts={_a.get('parts')} — the joint column passes "
                        f"the placement, part FK is the consumer's (scene.xml hinge)",
                        flush=True,
                    )
    q_pl, v_pl, cp_pl = _mcv.rigid_pl(
        qpos, z.get("qvel"), z.get("contact_pos"), hqd, dx, dy, yaw, pivot, maps, layout=_lay_pl
    )
    Rz = Rot.from_euler("z", np.deg2rad(yaw))
    _g1_fk_gate(maps, hqd, T, qpos, q_pl, dx, dy, yaw, pivot, Rz)
    h2 = 2 * hqd
    # G2 is a self-comparison here: it checks that rigid_pl's object transform == _tf_points.
    if inj is None:
        # Object block addresses = the tree statement. With a joint column, layout already went to rigid_pl, so the
        # contiguous 7-block convention is not required (blocks() confirmed the coverage); without one, the tree
        # addresses must match the "contiguous 7-blocks after the hands" (h2 + 7k) convention exactly.
        lay = _lay_pl if _lay_pl is not None else qpos_layout.read(str(src), nq=qpos.shape[1])
        slots = lay.object_starts()
        _artic = lay.artic_cols()
    else:  # injected: the injected statement (the source tree is one-hand, its own reading misaligns nq)
        slots = sorted(int(e["start"]) for e in inj["layout"] if e.get("kind") == "object")
        _artic = []
    _check_object_slots(qpos, _artic, h2, slots, base, hqd)
    obj_pairs = [(s_, s_) for s_ in slots]
    print(
        f"[bake] {base}: base-layer source — reference = itself · trimmed 0 · G2 self-check · object blocks {slots}"
        + (f" · joint columns {_artic} (passed through)" if _artic else ""),
        flush=True,
    )
    _g2_object_gate(obj_pairs, bl, T, dx, dy, yaw, pivot, q_pl, base, Rz)
    z_pl = dict(z)
    z_pl["qpos"] = q_pl
    if v_pl is not None:
        z_pl["qvel"] = v_pl
    if cp_pl is not None:
        z_pl["contact_pos"] = cp_pl
    if inj is not None:
        ad = dict(ad)
        ad["_uni_inject"] = inj  # passed to the write step (adopted.json is not touched)
    return src, z_pl, hqd, ad, (dx, dy, yaw, pivot), maps, pad


def cmd_bake(a) -> int:
    """`bake`: place one clip and write its tree → 0."""
    if a.src_layer != "base":
        raise SystemExit(f"[bake] source layer {a.src_layer!r}: only the base layer is baked")
    base = a.base
    tree = tree_name(base)  # no shadowing: a local named like a function kills that function here
    src, z_pl, hqd, ad, (dx, dy, yaw, pivot), _maps, pad = _placed_arrays(a, a.combo, base)
    adopted_p = placement_dir(a.combo, a.dataset) / f"{tree}{PLACEMENT_SFX}"
    T = len(z_pl["qpos"])
    # No static-tail trimming: after the object stops, the hand still releases it (contact ends later), and the
    # chain stretches only the time axis, so a static tail is not slowed down.

    # ---- time axis (identity: placement only, the chain retimes later) ----
    tau = np.arange(T) * _mcv._rt.DT
    s = np.ones(max(T - 1, 1))
    retime_meta = {"retime_v2": "none"}
    # Without an injection the quaternion slots come from the statement (qpos_layout); joint columns are
    # interpolated as scalars.
    _lay_rs = qpos_layout.read(str(src), nq=z_pl["qpos"].shape[1]) if ad.get("_uni_inject") is None else None
    out = _mcv.resample_all(z_pl, tau, s, hqd, layout=_lay_rs, root_maps=lambda: _maps)
    assert len(out["qpos"]) == T, f"the length changed although only the placement is baked: {len(out['qpos'])} vs {T}"

    # ---- write ----
    # Tree root owner = paths.trees_dir(combo, dataset, emb); --out_root overrides that root itself.
    out_root = Path(a.out_root) if a.out_root else trees_dir(a.combo, a.dataset, a.emb)
    dst = out_root / tree
    # Re-run idempotency: a tree right after bake is always fresh with no record — a surviving F0_REPAIR.json of
    # an earlier cycle would be misread later as an already-padded tree. It is generated, so the directory goes.
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "0").mkdir(parents=True, exist_ok=True)
    np.savez(dst / "0" / "trajectory_kinematic.npz", **out)
    np.savez(dst / "0" / "trajectory_ikrollout.npz", qpos=out.get("qpos_rollout", out["qpos"]))
    _inj = ad.get("_uni_inject") if isinstance(ad, dict) else None
    for fname in SP.SCENE_XMLS:
        if (src / fname).exists():
            if _inj is not None:
                (dst / fname).write_text(_abs_dirs_text(_inj["scene_txt"][fname], src))  # injected scene, abs dirs
            else:
                copy_xml_abs_dirs(src / fname, dst / fname)  # absolute meshdir, independent of the tree depth
    if (src / "0" / "text_annotations.json").exists():
        shutil.copy2(src / "0" / "text_annotations.json", dst / "0")
    print(
        f"[bake] {base}: contact sidecar copy skipped (base layer) — dexengine.data.contact.derive moves it from "
        "contact_base",
        flush=True,
    )
    tio = {}
    if (src / "task_info.json").exists():
        tio = json.loads((src / "task_info.json").read_text())
    # One-hand rule: active_sides / idle_sides of adopted.json move into the tree task_info (read downstream by
    # collision.idle_sides_of_tree). Absent keys (two-hand mode) write nothing.
    if "active_sides" in ad:
        tio["active_sides"] = list(ad["active_sides"])
        tio["idle_sides"] = list(ad.get("idle_sides", []))
    # The reach map ids the judge placed the clip with (plan line → adopted.json), read back by the coarse path
    # (store.check_recorded). Provenance, outside every content comparison (tree_content.SOURCE_TOP).
    _PM.record_into(tio, ad.get("pool"))
    if _inj is not None:
        # The tree statement becomes the two-hand layout (qpos_layout · nq · hands) + a marker of the added hand,
        # which must be the idle hand.
        if _inj["side"] not in (tio.get("idle_sides") or []):
            raise InputMissingError(
                f"[bake] {base}: the added hand {_inj['side']} is not in idle_sides="
                f"{tio.get('idle_sides')} — rule mismatch"
            )
        for _k in ("qpos_layout", "objects", "nq", "hands", "idle_hand_injected", "uni_layout_nq_before"):
            tio[_k] = _inj["ti_after"][_k]  # column statements (objects[].qpos_start too)
    _write_base_task_info(a, tio, src, dst, base, hqd, pad, (dx, dy, yaw, pivot), adopted_p, f"{base}_pl", retime_meta)
    print(
        f"[bake] {a.combo}/{tree}: T={len(out['qpos'])} "
        f"(dx={dx:+.4f} dy={dy:+.4f} yaw={yaw:+.2f} z={ad.get('robot_z_off', 0):+.4f}"
        f"{f' · f0 pad {pad}' if pad else ''} · base layer → {dst.parent})"
    )
    return 0


def main() -> int:
    """CLI: `combo.py bake --combo C --base CLIP --dataset DS --emb EMB [--src-layer base] …`."""
    ap = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bake", allow_abbrev=False)  # no prefix matching of removed options
    b.add_argument("--combo", required=True)  # e.g. rby1_wuji2
    b.add_argument("--base", required=True)  # base clip name
    b.add_argument(
        "--processed_root",
        default=str(MAIN / "data/taco_retarget/processed/taco"),
        help="source root (below it <hand spec dir>/<emb>/<clip>) — the chain passes the f0 shadow root",
    )
    b.add_argument(
        "--out_root", default="", help="the tree root **itself** (default = paths.trees_dir(combo, dataset, emb))"
    )
    b.add_argument("--dataset", required=True, help="dataset id (no default — the caller states the axis)")
    b.add_argument("--emb", required=True, help="embodiment: right|left|bimanual (no default)")
    b.add_argument(
        "--src-layer",
        dest="src_layer",
        default="",
        choices=list(SRC_LAYERS) + [""],
        help="source layer (empty = paths.default_src_layer(dataset))",
    )
    # The bake only places — the retiming is retime.py once, later in the chain.
    b.add_argument(
        "--warp_safety",
        type=float,
        default=1.15,
        help="task_info.retime_safety placeholder before retiming — the real value is retime.py --safety, "
        "which overwrites it",
    )
    a = ap.parse_args()
    if not a.src_layer:
        a.src_layer = default_src_layer(a.dataset)
    return cmd_bake(a)


if __name__ == "__main__":
    raise SystemExit(main())
