"""Base layer, one segment of a clip: slice the arrays and attach raw supplements, roles and articulated part
groups, then write the tree (driven by base_layer.clip.adapt_clip)."""

from __future__ import annotations

import io
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np

from dexengine.core import stdio
from dexengine.data import raw_sample as RS
from dexengine.data import tree_io as TT
from dexengine.data.base_layer.common import ADAPTER_VERSION, JUMP_RAD, ROOT, _dest
from dexengine.data.base_layer.geom import (
    load_model,
    recompute_qvel,
    slice_arrays,
)
from dexengine.data.base_layer.hand_dir import PER_CLIP_YAW, supplements_for
from dexengine.data.base_layer.stale import retarget_input, support_for
from dexengine.data.base_layer.supplements import (
    ensure_supplement_asset,
    insert_supplement_body,
    supplement_input,
    supplement_mark,
)


def _segment_arrays(ds, hand, out_clip, sg, kin, T, objs, roots, layout, nq, psi, dt):
    """Slice the segment and attach ⑥ raw supplement objects (per-segment XML copies) →
    (a, b, Ts, kin_s, dropped_k, sups, sup_dup, roots_s, layout_s, objs_base, nq_s)."""
    a, b = sg["start"], sg["end"]
    Ts = b - a
    kin_s, dropped_k = slice_arrays({k: v for k, v in kin.items() if k != "qpos_rollout_ik"}, a, b, T)
    # ⑥ Raw supplement objects: if a pose sequence was exported under this segment's
    #   tree name, attach the body (the XML becomes a per-segment copy)
    from dexengine.data.oakink2 import parts as OP  # owner of the part id (rid) rule

    sups, sup_dup = supplements_for(ds, out_clip, {OP.rid_of(o["obj_id"]) for o in objs}, yaw=psi)
    if sup_dup:
        print(
            f"[SUPP_DUP] {out_clip}: {len(sup_dup)} supplements with the same part id as the handoff "
            f"not loaded {sup_dup}",
            flush=True,
        )
    roots_s, layout_s, objs_base, nq_s = roots, layout, objs, nq
    if sups:
        print(RS.mark(len(sups), out_clip), flush=True)
        roots_s = {n: ET.fromstring(ET.tostring(r)) for n, r in roots.items()}
        layout_s = list(layout)
        objs_base = [dict(o) for o in objs]
        for k2, sp in enumerate(sups):
            if len(sp["pos"]) != Ts:
                raise ValueError(f"supplement object {sp['rid']} pose series length {len(sp['pos'])} != segment {Ts}")
            oid, rel = ensure_supplement_asset(ds, sp["rid"], sp["mesh_src"])
            name = f"supp_object_{k2}"
            insert_supplement_body(roots_s, name, oid, ds, hand)
            block = np.concatenate([sp["pos"], sp["quat"]], axis=1)
            for key in ("qpos", "qpos_rollout"):
                if key in kin_s:
                    kin_s[key] = np.concatenate([kin_s[key], block], axis=1)
            objs_base.append(
                {
                    "name": name,
                    "role": "extra",
                    "side": None,
                    "obj_id": oid,
                    "mesh_dir": rel,
                    "convex_dir": rel,
                    "spider_mesh_dir": None,
                    "spider_convex_dir": None,
                    "qpos_start": nq_s,
                    "qpos_width": 7,
                    "slot": len(objs_base),
                    "npz_key": None,
                    "articulation": None,
                    "supplement": {
                        "source": "oakink2_raw_objects",
                        "raw_obj": sp["rid"],
                        "for_hand": sp["hand"],
                        "contact_frac_raw": sp["contact_frac"],
                        "file": sp["file"],
                        "sample": sp["sample"],
                        "convex": "none(visual copy)",
                    },
                }
            )
            layout_s.append({"name": name, "start": nq_s, "width": 7, "kind": "object", "side": None, "role": "extra"})
            nq_s += 7
        model_s = load_model(ET.tostring(roots_s["scene.xml"], encoding="unicode"), ds, hand)
        if model_s.nq != nq_s:
            raise ValueError(f"XML nq after the supplement {model_s.nq} != {nq_s}")
        kin_s["qvel"] = recompute_qvel(model_s, kin_s["qpos"], dt)
    return a, b, Ts, kin_s, dropped_k, sups, sup_dup, roots_s, layout_s, objs_base, nq_s


def _segment_roles(ds, emb, want, hqd, objs_base, kin_s, res):
    """Delivered roles, ④ articulated part groups (OakInk2 part tree) and ⑤ finger-jump markers →
    (objs_s, slots, shared, jumps)."""
    from dexengine.data.contact import geom as CG

    # Delivered roles pass through (who handles what = hand_object_assign at stage 3, applied by object_roles at
    # 4)
    objs_s = [dict(o) for o in objs_base]
    chosen = {
        s: next((o["name"] for o in objs_s if o["role"] == "main" and o["side"] in (s, "both")), None) for s in want
    }
    slots = {s: next(o["slot"] for o in objs_s if o["name"] == n) for s, n in chosen.items() if n}
    shared = emb == "bimanual" and len(set(chosen.values())) == 1 and all(chosen.values())
    # ④ Articulated part groups — identity is decided **only by the official OakInk2 part tree** (oakink2.parts,
    # D1-d). "Same object number = part pair" would misread
    #    the category number as an instance and group distinct objects (beaker + test tube, …). base = the part
    #    that moved least in the segment
    #    (the body at rest); the rest have their relative motion fitted.
    groups = {}
    if ds == "oakink2":
        from dexengine.data.oakink2 import parts as OP

        for o in objs_s:
            rt = OP.root_of(OP.rid_of(o["obj_id"]))
            if rt:
                groups.setdefault(rt, []).append(o)
    for g, members in groups.items():
        if len(members) < 2:
            continue

        def span(o):
            return float(
                (
                    kin_s["qpos"][:, o["qpos_start"] : o["qpos_start"] + 3].max(0)
                    - kin_s["qpos"][:, o["qpos_start"] : o["qpos_start"] + 3].min(0)
                ).max()
            )

        members.sort(key=lambda o: (round(span(o), 3), o["obj_id"]))
        base = members[0]
        res["art_groups"] += 1
        for o in members:
            fit = (
                None
                if o is base
                else CG.fit_articulation(
                    kin_s["qpos"][:, base["qpos_start"] : base["qpos_start"] + 7],
                    kin_s["qpos"][:, o["qpos_start"] : o["qpos_start"] + 7],
                )
            )
            o["articulation_group"] = {
                "group": g,
                "instance": OP.name_of(g),
                "part_name": OP.name_of(OP.rid_of(o["obj_id"])),
                "parts": [m["name"] for m in members],
                "role": "base" if o is base else "part",
                "base": base["name"],
                "fit": fit,
                "identity_source": "object_part_tree.json",
            }
    # ⑤ Markers
    jumps = {
        s: int((np.abs(np.diff(kin_s["qpos"][:, i * hqd + 6 : (i + 1) * hqd], axis=0)).max(1) > JUMP_RAD).sum())
        for i, s in enumerate(want)
    }
    return objs_s, slots, shared, jumps


def _write_scene(tmp, src, roots_s: dict, ds: str, hand: str, merged, sups, scene_parts) -> None:
    """The scene XML files (rewritten when duplicates were merged, objects supplemented or parts re-derived;
    otherwise copied) with their asset references fixed."""
    from dexengine.data import intake_ingest as IH

    for n, root in roots_s.items():
        if merged or sups or scene_parts:
            ET.indent(root, space="  ")
            (tmp / n).write_text('<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n")
        else:
            shutil.copy2(src / n, tmp / n)
    IH.fix_scene_refs(tmp, ds, hand, {hand: hand})


def _write_arrays(tmp, src, kin_s: dict, kin: dict, rollout, a: int, b: int, T: int, dropped_k) -> dict:
    """The kinematic, IK-rollout and diagnostics arrays sliced to the segment → keys dropped per file (length ≠ T)."""
    np.savez(tmp / "0/trajectory_kinematic.npz", **kin_s)
    if rollout is not None:
        ro = dict(rollout.items())
        ro["qpos"] = kin["qpos_rollout_ik"]
        ro_s, _ = slice_arrays(ro, a, b, T)
        np.savez(tmp / "0/trajectory_ikrollout.npz", **ro_s)
    dropped = {"trajectory_kinematic.npz": dropped_k} if dropped_k else {}
    if (src / "0/retarget_diagnostics.npz").exists():
        dg, dr = slice_arrays(dict(np.load(src / "0/retarget_diagnostics.npz")), a, b, T)
        np.savez(tmp / "0/retarget_diagnostics.npz", **dg)
        if dr:
            dropped["retarget_diagnostics.npz"] = dr
    return dropped


def _segment_block(sg: dict, Ts: int, ti: dict, T_mano: int, T: int, off: int) -> dict:
    """task_info.segment: the segment window, its length and rate, and the MANO ↔ kinematic frame map."""
    return {
        **sg,
        "T": Ts,
        "fps": 1.0 / float(ti["ref_dt"]),
        "frame_map": {
            "T_mano": T_mano,
            "T_kin": T,
            "offset": off,
            "rule": "kin[i] == mano[i] (tail Δ frames dropped; measured by matching object poses)",
        },
    }


def _frame_block(ds: str, psi: float, dstat: dict, tstat: dict, h2o_tilt, tilt_version) -> dict:
    """task_info.frame: the yaw that maps the capture frame to the base frame (and the H2O desk tilt)."""
    return {
        "convention": "taco_hand_dir_per_clip" if ds in PER_CLIP_YAW else "taco_hand_dir",
        "yaw_deg": round(psi, 4),
        "hand_dir_deg_dataset": dstat["mean_deg"],
        "hand_dir_spread_deg_dataset": dstat["spread_deg"],
        "hand_dir_deg_taco": tstat["mean_deg"],
        "rule": "base = Rz(yaw)·capture, yaw = dir(taco) − dir(ds), dir = circular mean of the MANO "
        "wrist→fingertip xy angle (INTAKE/hand_dir.json); z unchanged",
        "h2o_tilt_d10e2": None
        if h2o_tilt is None
        else {
            "tilt_deg": round(h2o_tilt["tilt_deg"], 2),
            "desk_z": round(h2o_tilt["desk_z"], 4),
            "pivot_capture": [round(x, 4) for x in h2o_tilt["pivot"]],
            "pivot_source": h2o_tilt["pivot_source"],
            "version": tilt_version,
            "rule": "capture' = R(n_plane→+z)·(capture − pivot) + pivot, then yaw; desk_z = pivot.z "
            "(INTAKE/h2o_table_plane.json, source depth)",
        },
    }


def _provenance(c: dict, out_clip: str, dropped: dict) -> dict:
    """task_info.provenance: adapter version, time, inputs and their fingerprints, dropped diagnostics."""
    ds, emb, clip, src, scene_parts, T, nq0 = (c[k] for k in ("ds", "emb", "clip", "src", "scene_parts", "T", "nq0"))
    return {
        "adapter": "dexengine/data/base_adapter.py",
        "adapter_version": ADAPTER_VERSION,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "retarget_dir": os.path.relpath(src, ROOT),
        "retarget_input": retarget_input(src),
        "supplement_input": supplement_input(ds, out_clip),
        "scene_parts": scene_parts,
        "mano_dir": os.path.relpath(_dest("mano", ds, emb=emb, clip=clip), ROOT),
        "T_full": T,
        "nq_delivered": nq0,
        "qvel": "recomputed with mujoco.mj_differentiatePos",
        "diagnostics_dropped(length≠T)": dropped,
    }


def _task_info(
    c: dict,
    sg: dict,
    out_clip: str,
    a: int,
    b: int,
    Ts: int,
    layout_s,
    nq_s,
    objs_s: list,
    slots: dict,
    shared,
    jumps: dict,
    sup_dup,
    dropped: dict,
) -> dict:
    """task_info.json of one segment: the delivered task_info with the base fields (hand, segment, frame, layout,
    objects, support, quality, provenance) and the object slots."""
    ds, hand, clip, job, ti, T, merged, want, hqd = (
        c[k] for k in ("ds", "hand", "clip", "job", "ti", "T", "merged", "want", "hqd")
    )
    T_mano, off, h2o_tilt, tilt_version, psi, dstat, tstat, quality, sha = (
        c[k] for k in ("T_mano", "off", "h2o_tilt", "tilt_version", "psi", "dstat", "tstat", "quality", "sha")
    )
    info = dict(ti)
    for s in (
        "right",
        "left",
    ):  # the retarget keys follow the reassignment and the merge too (the two views agree)
        m = next((o for o in objs_s if o["role"] == "main" and o["side"] in (s, "both")), None)
        info[f"{s}_object_mesh_dir"] = m["spider_mesh_dir"] if m else None
        info[f"{s}_object_convex_dir"] = m["spider_convex_dir"] if m else None
    info.update(
        {
            "robot_type": hand,
            "robot_type_delivered": ti.get("robot_type"),
            "hand": hand,
            "hand_asset_sha256_xml3": sha,
            "hand_qpos_dim": hqd,
            "hands": want,
            "clip_id": out_clip,
            "parent_clip": clip,
            "segment": _segment_block(sg, Ts, ti, T_mano, T, off),
            "frame": _frame_block(ds, psi, dstat, tstat, h2o_tilt, tilt_version),
            "qpos_layout": layout_s,
            "nq": nq_s,
            "objects": objs_s,
            "shared_object": shared,
            "merged_duplicate": merged,
            "supplemented_objects": [o["supplement"] for o in objs_s if o.get("supplement")] or None,
            "support": support_for(job["sup"], a, b, off, psi),
            "quality": {**quality, "finger_jump_frames": jumps, "finger_jump_threshold_rad": JUMP_RAD},
            "provenance": _provenance(c, out_clip, dropped),
        }
    )
    for k in ("right", "left"):
        info.pop(f"{k}_object_slot", None)
    for s, sl in slots.items():
        info[f"{s}_object_slot"] = sl
    info.update(supplement_mark(info, sup_dup))
    return info


def _body_positions(tmp) -> None:
    """Body positions by FK into tmp/0/body_positions.npz; a failed FK raises with the tail of its output."""
    from dexengine.placement.fk import body_positions as BP

    buf = io.StringIO()
    with stdio.capture(buf):
        ok = BP.process_sequence(str(tmp), verbose=False)
    if not ok or not (tmp / "0/body_positions.npz").exists():
        raise RuntimeError("body_positions FK failed: " + buf.getvalue()[-300:])


def _write_segment(
    c: dict,
    sg,
    out_clip,
    dst,
    a,
    b,
    Ts,
    kin_s,
    dropped_k,
    sups,
    sup_dup,
    roots_s,
    layout_s,
    nq_s,
    objs_s,
    slots,
    shared,
    jumps,
    res,
):
    """Write one base tree (tmp folder → body positions FK → atomic replace of dst)."""
    src, kin, rollout, T, merged, scene_parts = (c[k] for k in ("src", "kin", "rollout", "T", "merged", "scene_parts"))
    # Write
    tmp = TT.tree_tmp(dst)
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "0").mkdir(parents=True)
    _write_scene(tmp, src, roots_s, c["ds"], c["hand"], merged, sups, scene_parts)
    dropped = _write_arrays(tmp, src, kin_s, kin, rollout, a, b, T, dropped_k)
    info = _task_info(c, sg, out_clip, a, b, Ts, layout_s, nq_s, objs_s, slots, shared, jumps, sup_dup, dropped)
    (tmp / "task_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    _body_positions(tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    res["out"].append(out_clip)
    if merged:
        res["merged"] += 1
    if sups:
        res["supplemented"] += 1
