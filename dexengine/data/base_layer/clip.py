"""Base layer: one delivered retargeting clip -> its stage-1 trees (adapt_clip; driven by the base_adapter entry)."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.data import tree_io as TT
from dexengine.data.base_layer.clip_segment import _segment_arrays, _segment_roles, _write_segment
from dexengine.data.base_layer.common import SEG_FMT, _dest, data_root, dl
from dexengine.data.base_layer.geom import (
    _hand_qpos_dim,
    build_objects,
    check_fk_oracle,
    clip_hand_dir_deg,
    find_duplicate,
    load_model,
    palm_fk,
    qpos_layout_from_root,
    recompute_qvel,
    remove_body_from_xml,
    rotate_world,
    segments_for,
    transform_world,
)
from dexengine.data.base_layer.hand_dir import PER_CLIP_YAW, mano_to_kin_offset
from dexengine.data.base_layer.stale import tree_stale_reason
from dexengine.data.objects import scene_parts as SP


# ── One clip ───────────────────────────────────────────────────────────────────
def _load_delivered(src):
    """The delivered tree: task_info, kinematic arrays (+ the IK rollout), lengths and the scene XML roots."""
    ti = json.loads((src / "task_info.json").read_text())
    kin = dict(np.load(src / "0/trajectory_kinematic.npz", allow_pickle=False))
    kin = {k: (v.copy() if v.ndim else v) for k, v in kin.items()}
    rollout = (
        dict(np.load(src / "0/trajectory_ikrollout.npz")) if (src / "0/trajectory_ikrollout.npz").exists() else None
    )
    if rollout is not None:
        kin["qpos_rollout_ik"] = rollout["qpos"].copy()
    T = int(kin["qpos"].shape[0])
    nq0 = int(kin["qpos"].shape[1])
    roots = {n: ET.parse(src / n).getroot() for n in SP.SCENE_XMLS if (src / n).exists()}
    return ti, kin, rollout, T, nq0, roots


def _rederive_parts(roots: dict, ds: str, hand: str, clip: str) -> list:
    """⑦ Re-stamp the scene's convex part groups with the current decomposition (before load_model)."""
    from dexengine.data import intake_ingest as IH

    # ⑦ convex parts = decomposed now (not copied from the numbering used when the delivered scene was retargeted);
    # before load_model (it cannot open without the parts)
    scene_parts = []
    for n, root in roots.items():
        for ch in SP.rederive(root, IH.scene_assets_root(ds, hand)):
            scene_parts.append({"file": n, **ch})
            print(
                f"[SCENE_PARTS] {ds}/{clip} {n} {ch['dir']} scene={len(ch['scene'])} now={len(ch['now'])}",
                flush=True,
            )
    return scene_parts


def _merge_duplicates(ds: str, ti: dict, kin: dict, roots: dict, layout: list, objs: list):
    """① Merge duplicate object bodies (in place on ti · kin · roots) → (merged record | None, layout, objs)."""
    merged = None
    # ① Merge duplicates (repeat until none are left — in hocap the main object is in the
    #   extra list too, so they come 2 at a time)
    while True:
        dup = find_duplicate(objs, kin)
        if not dup:
            break
        keep, drop = dup
        nqc = int(kin["qpos"].shape[1])
        s0 = drop["qpos_start"]
        cols = np.r_[0:s0, s0 + 7 : nqc]
        merged = merged or {"kept": [], "removed": [], "xml": {}}
        merged["kept"].append(keep["name"])
        merged["removed"].append(drop["name"])
        for n, root in roots.items():
            merged["xml"].setdefault(n, []).append(remove_body_from_xml(root, drop["name"]))
        for key in ("qpos", "qpos_rollout", "qpos_rollout_ik"):
            if key in kin:
                kin[key] = kin[key][:, cols]
        if drop["npz_key"]:
            kin.pop(drop["npz_key"], None)
        if drop["role"] == "main":
            dside = drop["name"].split("_")[0]
            ti[f"{dside}_object_mesh_dir"] = None
            ti[f"{dside}_object_convex_dir"] = None
        layout = qpos_layout_from_root(roots["scene.xml"])
        objs = build_objects(ds, ti, layout, kin)
        if sum(e["width"] for e in layout) != nqc - 7:
            raise ValueError("layout width after the merge is not nq−7")
    return merged, layout, objs


def _hand_layout(layout: list, emb: str, hand: str):
    """The hand entries of the layout, checked against the embodiment and the HandSpec width → (want, hqd)."""
    hands = [e for e in layout if e["kind"] == "hand"]
    want = {"right": ["right"], "left": ["left"], "bimanual": ["right", "left"]}[emb]
    if [h["side"] for h in hands] != want:
        raise ValueError(f"hand set {[h['side'] for h in hands]} != {want}")
    hqd = hands[0]["width"]
    # check that what the structure picked really is that hand **against the declaration**; a wrong pick silently
    # gives another answer.
    _want_hqd = _hand_qpos_dim(hand)
    if _want_hqd and hqd != _want_hqd:
        raise ValueError(
            f"hand width {hqd} != HandSpec.hand_qpos_dim {_want_hqd} ({hand}) — "
            f"root bodies {[h['name'] for h in hands]} were taken as hands. Check the rig and the scene"
        )
    return want, hqd


def _h2o_tilt(ds: str, clip: str, model, kin: dict, layout: list, mano: dict, probe, want: list, hand: str, res: dict):
    """①' h2o D10-e'': rotate the desk plane normal onto +z (kin and mano) → (h2o_tilt | None, mano, tilt version)."""
    tilt_version = None
    h2o_tilt = None
    if ds == "h2o":
        from dexengine.data.readers import h2o_frame as HF

        h2o_tilt = HF.tilt_transform(clip)
        tilt_version = HF.TILT_VERSION
        before = palm_fk(model, kin["qpos"][probe], want, hand)
        transform_world(kin, layout, h2o_tilt["M"], keys=("qpos", "qpos_rollout", "qpos_rollout_ik"))
        check_fk_oracle(model, kin, want, probe, before, h2o_tilt["M"], "h2o desk rotation(D10-e'')", hand)
        mano = HF.apply_rigid_keypoints(mano, h2o_tilt["M"])
        print(HF.tilt_marker(clip, h2o_tilt, "base_adapter"), flush=True)
        res["h2o_tilt_deg"] = h2o_tilt["tilt_deg"]
    return h2o_tilt, mano, tilt_version


def _yaw_align(ds: str, clip: str, job: dict, mano: dict, want: list, model, kin: dict, layout: list, probe, hand: str):
    """② Yaw alignment of kin (+ oracle: palm FK == Rz·old palm FK) → (psi, dstat, tstat)."""
    # ② yaw alignment (+ oracle: palm FK == Rz·old palm FK)
    psi, dstat, tstat = job["yaw"]
    if ds in PER_CLIP_YAW:
        d_clip = clip_hand_dir_deg(mano, want)
        psi = float((tstat["mean_deg"] - d_clip + 180.0) % 360.0 - 180.0)
        dstat = {**dstat, "clip_dir_deg": round(d_clip, 2)}
        print(f"[YAW_CLIP] {clip}: clip_dir={d_clip:+.1f}° → yaw={psi:+.1f}° (taco {tstat['mean_deg']}°)", flush=True)
    before = palm_fk(model, kin["qpos"][probe], want, hand)
    rotate_world(kin, layout, psi)
    M = Rot.from_euler("z", psi, degrees=True).as_matrix()
    M4 = np.eye(4)
    M4[:3, :3] = M
    check_fk_oracle(model, kin, want, probe, before, M4, "yaw", hand)
    return psi, dstat, tstat


def _prepare(job: dict, ds: str, hand: str, emb: str, clip: str, res: dict) -> tuple:
    """Load the delivered retarget tree, re-derive parts, merge duplicates, align it (H2O tilt · yaw) and audit it
    → (clip context c, segments, objects, scene roots, qpos layout, nq, yaw, dt)."""
    from dexengine.retarget.quality import audit_clip

    src = _dest("retarget", ds, hand=hand, emb=emb, clip=clip)
    ti, kin, rollout, T, nq0, roots = _load_delivered(src)
    scene_parts = _rederive_parts(roots, ds, hand, clip)
    layout = qpos_layout_from_root(roots["scene.xml"])
    if sum(e["width"] for e in layout) != nq0:
        raise ValueError(f"qpos layout Σwidth != nq {nq0}")
    objs = build_objects(ds, ti, layout, kin)
    merged, layout, objs = _merge_duplicates(ds, ti, kin, roots, layout, objs)
    nq = int(kin["qpos"].shape[1])
    want, hqd = _hand_layout(layout, emb, hand)
    xml_text = ET.tostring(roots["scene.xml"], encoding="unicode")
    model = load_model(xml_text, ds, hand)
    if model.nq != nq:
        raise ValueError(f"merged XML nq {model.nq} != qpos {nq}")
    dt = 1.0 / float(kin["frequency"]) if "frequency" in kin and float(kin["frequency"]) > 0 else float(ti["ref_dt"])
    probe = np.linspace(0, T - 1, 3).astype(int)
    mano = np.load(data_root() / dl.render("mano", ds, emb=emb, clip=clip) / "0/trajectory_keypoints.npz")
    mano = {k: mano[k] for k in mano.files}
    T_mano = int(job["seg"]["T"])
    off = mano_to_kin_offset(T_mano, T)
    h2o_tilt, mano, tilt_version = _h2o_tilt(ds, clip, model, kin, layout, mano, probe, want, hand, res)
    psi, dstat, tstat = _yaw_align(ds, clip, job, mano, want, model, kin, layout, probe, hand)
    kin["qvel"] = recompute_qvel(model, kin["qpos"], dt)
    quality = audit_clip(src)
    sha = dl.asset_hash(str(_dest("assets_robot", ds, hand=hand)))
    segs = segments_for(job["seg"], T, off)
    c = {
        "ds": ds,
        "hand": hand,
        "emb": emb,
        "clip": clip,
        "job": job,
        "src": src,
        "ti": ti,
        "kin": kin,
        "rollout": rollout,
        "T": T,
        "nq0": nq0,
        "scene_parts": scene_parts,
        "merged": merged,
        "want": want,
        "hqd": hqd,
        "T_mano": T_mano,
        "off": off,
        "h2o_tilt": h2o_tilt,
        "tilt_version": tilt_version,
        "psi": psi,
        "dstat": dstat,
        "tstat": tstat,
        "quality": quality,
        "sha": sha,
    }
    return c, segs, objs, roots, layout, nq, psi, dt


def _adapt_segment(
    c: dict, sg: dict, objs: list, roots: dict, layout: list, nq: int, psi: float, dt: float, res: dict
) -> None:
    """One segment of the clip → one base tree (skipped when an up-to-date tree exists and the job allows reuse)."""
    ds, hand, emb, clip, job, kin, T, want, hqd = (
        c[k] for k in ("ds", "hand", "emb", "clip", "job", "kin", "T", "want", "hqd")
    )
    out_clip = clip if sg["n"] == 1 else SEG_FMT.format(clip=clip, idx=sg["index"] + 1)
    dst = _dest("base", ds, hand=hand, emb=emb, clip=out_clip)
    if job["skip_existing"] and not tree_stale_reason(dst):  # reuse test = the same function as the stage-1 column 2
        res["skipped"].append(out_clip)
        return
    a, b, Ts, kin_s, dropped_k, sups, sup_dup, roots_s, layout_s, objs_base, nq_s = _segment_arrays(
        ds, hand, out_clip, sg, kin, T, objs, roots, layout, nq, psi, dt
    )
    objs_s, slots, shared, jumps = _segment_roles(ds, emb, want, hqd, objs_base, kin_s, res)
    _write_segment(
        c,
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
    )


def adapt_clip(job: dict) -> dict:
    ds, hand, emb, clip = job["ds"], job["hand"], job["emb"], job["clip"]
    res = {
        "clip": clip,
        "emb": emb,
        "out": [],
        "skipped": [],
        "err": None,
        "merged": 0,
        "art_groups": 0,
        "supplemented": 0,
    }
    try:
        import mujoco  # noqa: F401  (a missing mujoco fails this clip here, like the steps below)

        c, segs, objs, roots, layout, nq, psi, dt = _prepare(job, ds, hand, emb, clip, res)
        for sg in segs:
            _adapt_segment(c, sg, objs, roots, layout, nq, psi, dt, res)
    except Exception as e:  # noqa: BLE001
        res["err"] = f"{type(e).__name__}: {e}"
        for t in _dest("base", ds, hand=hand, emb=emb).glob(TT.tree_tmp_glob(clip)):
            shutil.rmtree(t, ignore_errors=True)
    return res
