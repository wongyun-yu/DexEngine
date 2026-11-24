"""E5 kinematic replay baseline (kin_data.prep): one clip's tree — the source tree (prep_clip) and its
placement-augmentation variants (prep_variants)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from experiments.policies.kin_data.prep_shadow import own_dir, place_base, set_env
from experiments.policies.kin_data.prep_tree import (
    ARM,
    HAND,
    MARK,
    SIDES,
    drop_tree,
    finish_tree,
    inside,
    match_inputs,
)


def _source_e5_kin(a, troot: Path, tree: str) -> tuple[dict, dict, str]:
    """Source tree task_info and e5_kin record; dies unless built with this placement and offsets table."""
    import hashlib  # noqa: PLC0415

    ti0 = json.loads((troot / tree / "task_info.json").read_text())
    e0 = ti0.get("e5_kin")
    offs_sha = hashlib.sha256(Path(a.offsets).read_bytes()).hexdigest()[:16]
    if not e0 or e0.get("placement") != a.placement or e0.get("offsets_sha256") != offs_sha:
        raise SystemExit(
            f"{MARK} {tree}: the source kinematic tree is not of this placement · offsets table (prepare "
            "the source first)"
        )
    return ti0, e0, offs_sha


def _source_sides(root: Path, ds: str, emb: str, clip: str, ti0: dict, tree: str):
    """Augmentation source of the clip, its active/idle hands (checked against the placement record) and robot z."""
    from dexengine.record.augment import place_src as APS  # noqa: PLC0415

    src = APS.Src(root, HAND, ds, emb, clip, arm=ARM, aerial=str(ti0.get("scene") or "") == "aerial")
    ad0 = src.adopted
    act, idle = list(src.active), list(src.idle)
    if act != list(ad0.get("active_sides", act)):
        raise SystemExit(f"{MARK} {tree}: active hands {act} ≠ placement record {ad0.get('active_sides')}")
    robot_z = float(ad0["robot_z_off"])
    return src, act, idle, robot_z


def _copy_ref_variant(rvr: Path, ds: str, combo: str, emb: str, troot: Path, srcv: str) -> dict:
    """--ref-var-root (default off): variant = the variant reference our RL followed through RSI, as is.

    No make_variant or IK — the tree and placement record are copied from the reference root (the record pieces' aug
    shadows gathered in the canonical root layout, chosen by piece id). Arm and wrist caches are copied from the same
    root by finish_tree → ref_arm. Missing = death.
    """
    from dexengine.core import paths as P  # noqa: PLC0415

    name = srcv.split(".")[0]
    vs, vp = (
        rvr / "data" / ds / "trees" / combo / emb / name,
        rvr / f"precompute/{ds}/robots/{combo}/placement/{name}{P.PLACEMENT_SFX}",
    )
    ok = (vs / "0" / "trajectory_kinematic.npz").is_file() and vp.is_file()
    if not ok or (troot / name).exists():
        raise SystemExit(
            f"{MARK} --ref-var-root: reference variant {vs} · placement {vp} missing or {name} already in the shadow"
        )
    shutil.copytree(vs, troot / name, symlinks=False)
    pd = P.placement_dir(combo, ds)
    pd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vp, pd / f"{name}{P.PLACEMENT_SFX}")
    rec = {"status": "written", "name": name, "pivot_tree": None, "desk_last_ok": None}
    print(
        f"{MARK} [KIN_REFVAR] {name} ← {vs} (reference variant copied · no make_variant)",  # applied marker
        flush=True,
    )
    return rec


def _finish_variant(
    a,
    combo: str,
    ds: str,
    emb: str,
    clip: str,
    troot: Path,
    name: str,
    rdir: Path,
    robot_z: float,
    act: list,
    idle: list,
    rvr: Path | None,
) -> tuple[dict, dict]:
    """Arm IK and reach of one variant tree (or the SPIDER rewrite when --spider-root is on)."""
    act_v, idle_v, seed_v = match_inputs(a, ds, name, act, idle)
    ik, reach = finish_tree(
        combo,
        ds,
        troot,
        name,
        rdir,
        robot_z,
        act_v,
        idle_v,
        seed_v,
        (rvr or Path(a.main)) if getattr(a, "ref_arm", False) else None,
    )
    if getattr(a, "spider_root", ""):
        # --spider-root/--spider-arm (default off; the E5 third source): rewrite the copied variant tree with SPIDER
        # hands and objects;
        # arm = SPIDER A.6, labels = SPIDER ctrl (owned by experiments/policies/spider_arm/data.py, markers
        # [SPIDER_*])
        from experiments.policies.spider_arm import data as SE  # noqa: PLC0415

        ik, reach = SE.build_variant(a, combo, ds, emb, clip, name, troot, rdir, robot_z)
    return ik, reach


def _variant_info(
    j: int, dx: float, dy: float, yaw: float, srcv: str, rec: dict, rvr: Path | None, dsha: str, tree: str
) -> dict:
    """The e5_kin.variant record of one drawn variant."""
    from experiments.policies.kin_data import variants as KV  # noqa: PLC0415

    vinfo = {
        "k": j,
        "dx_scene": dx,
        "dy_scene": dy,
        "yaw_deg": yaw,
        "source_variant": srcv,
        "pivot_tree": rec["pivot_tree"],
        "desk_last_ok": bool(rec["desk_last_ok"]) if rvr is None else None,
        "dist_rule": KV.RULE,
        "dist_points_sha256": dsha,
        "parent_tree": tree,
        "fn": (f"ref_var_root copy {rvr}" if rvr is not None else "aug_place_gen.make_variant(desk_check=False)"),
    }
    return vinfo


def _write_variant_info(
    combo: str, ds: str, troot: Path, name: str, e0: dict, reach: dict, ik: dict, vinfo: dict
) -> None:
    """Write e5_kin (source record + this variant's reach, IK and draw) into the variant's task_info and placement."""
    from dexengine.core import paths as P  # noqa: PLC0415

    ti = json.loads((troot / name / "task_info.json").read_text())
    ti["e5_kin"] = dict(
        e0,
        reach=reach,
        variant=vinfo,
        ik={s: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()} for s, d in ik.items()},
    )
    (troot / name / "task_info.json").write_text(json.dumps(ti, indent=1, ensure_ascii=False))
    ad_p = P.placement_dir(combo, ds) / f"{name}{P.PLACEMENT_SFX}"
    ad = json.loads(ad_p.read_text())
    ad["e5_kin"] = dict(ad.get("e5_kin") or {}, variant=vinfo)
    ad_p.write_text(json.dumps(ad, indent=1, ensure_ascii=False))


def _draw_row(
    j: int, name: str, reach: dict, vinfo: dict, dx: float, dy: float, yaw: float, srcv: str, ik: dict
) -> dict:
    """Result-row entry of one draw."""
    d = {
        "j": j,
        "name": name,
        "reach": reach["reach"],
        "reach_per_side": reach["per_side"],
        "passes": reach["passes"],
        "n_jump": reach["n_jump"],
        "max_dq_ratio": reach["max_dq_ratio"],
        "desk_last_ok": vinfo["desk_last_ok"],
        "dx_scene": dx,
        "dy_scene": dy,
        "yaw_deg": yaw,
        "source_variant": srcv,
        "ik": {
            s: {"pos_mm_mean": round(v["pos_mm_mean"], 1), "fallback_frames": v["fallback_frames"]}
            for s, v in ik.items()
        },
    }
    return d


def prep_variants(a, ds: str, emb: str, clip: str, dist: dict) -> dict:
    """Placement-augmentation variants: drawn from our data distribution (variants.draws, seed 42) and built on the
    source kinematic tree with the canonical generator place_gen.make_variant (desk_check=False — the rule is reach
    only; the desk result is recorded as desk_last_ok), then measured with the source's finish (finish_tree). Pass =
    kept, fail = deleted; stops when PER_CLIP (augment policy) pass or MAX_DRAWS were drawn (shortfall = result row
    short). Placement pipeline keeps no filter — PER_CLIP are kept in draw order and reach is only recorded."""
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.record.augment import bake as AB  # noqa: PLC0415 — owner of the variant base alias (alias_base)
    from dexengine.record.augment import place_gen as AG  # noqa: PLC0415
    from dexengine.record.augment import place_src as APS
    from experiments.policies.kin_data import variants as KV  # noqa: PLC0415

    fl = set_env(ds, emb)
    combo = f"{ARM}_{HAND}{fl['variant']}"
    tree, root = P.tree_name(clip), Path(a.root)
    troot, rdir = P.trees_dir(combo, ds, emb), P.robots_dir(combo, ds)
    ti0, e0, offs_sha = _source_e5_kin(a, troot, tree)
    src, act, idle, robot_z = _source_sides(root, ds, emb, clip, ti0, tree)
    mcv, need = (
        AG._load_mcv(),
        KV.need(dist, ds, clip),
    )  # distribution mode = PER_CLIP · exact mode = our listed variant count
    dsha = hashlib.sha256(json.dumps(dist["datasets"][ds]["points"]).encode()).hexdigest()[:16]
    kept, draws = [], []
    rvr = Path(a.ref_var_root) if getattr(a, "ref_var_root", "") else None
    for j, dx, dy, yaw, srcv in KV.draws(dist, ds, clip):
        if rvr is not None:
            rec = _copy_ref_variant(rvr, ds, combo, emb, troot, srcv)
        else:
            rec = APS.make_variant(
                src, mcv, root, j, np.array([dx, dy]), yaw, src.kin, "trajectory_kinematic", desk_check=False
            )
        if rec.get("status") != "written":
            raise SystemExit(f"{MARK} {tree} draw {j}: make_variant {rec.get('status')} {rec.get('why')}")
        name = rec["name"]
        if KV.is_exact(dist) and name != srcv.split(".")[0]:  # exact = the same tree name as our scene (pair proof)
            raise SystemExit(f"{MARK} {tree} exact placement {j}: tree {name} ≠ our scene {srcv}")
        # the variant's base = the source's base (rigid transform) — the contact sidecar and base tree that the
        # idle-arm support stamping (stamp_scene) looks up by tree name are linked relatively into the shadow by the
        # canonical bake's owner (bake.alias_base), not copied
        alias = AB.alias_base(src, root, name)
        ik, reach = _finish_variant(a, combo, ds, emb, clip, troot, name, rdir, robot_z, act, idle, rvr)
        vinfo = _variant_info(j, dx, dy, yaw, srcv, rec, rvr, dsha, tree)
        _write_variant_info(combo, ds, troot, name, e0, reach, ik, vinfo)
        d = _draw_row(j, name, reach, vinfo, dx, dy, yaw, srcv, ik)
        if reach["passes"] or a.placement == "pipeline":  # pipeline = no filter (reach only recorded)
            kept.append(name)
        else:
            d["dropped"] = drop_tree(root, combo, ds, troot, name)
            d["alias_left"] = alias
        draws.append(d)
        if len(kept) >= need:
            break
    return {
        "mode": "variants",
        "ds": ds,
        "emb": emb,
        "clip": clip,
        "tree": tree,
        "combo": combo,
        "active": act,
        "idle": idle,
        "kept": kept,
        "n_kept": len(kept),
        "need": need,
        "short": need - len(kept),
        "n_draws": len(draws),
        "draws": draws,
        "offsets_sha256": offs_sha,
        "placement": a.placement,
        "dist_points_sha256": dsha,
        "desk_last_fail_kept": sum(1 for x in draws if x["passes"] and not x["desk_last_ok"]),
    }


def _load_offsets(a, ds: str) -> dict:
    """The offsets table; dies unless it holds the dataset and matches --placement."""
    from experiments.policies.kin_data import offsets as KO  # noqa: PLC0415 — owner of the placement rule

    offs = KO.load(a.offsets)
    if ds not in offs["datasets"]:
        raise SystemExit(f"{MARK} {ds}: not in the offsets table ({a.offsets})")
    if a.placement not in KO.PLACEMENTS or a.placement != offs.get("placement"):
        raise SystemExit(
            f"{MARK} --placement {a.placement!r} ≠ offsets table {offs.get('placement')} — known: {KO.PLACEMENTS}"
        )
    return offs


def _placement_marker(a, tree: str, cp: dict, act, o, yaw, pivot, psrc) -> tuple[str, str]:
    """Print the applied placement marker; returns the target-centre name and the placement source text."""
    if psrc is None:
        cname = f"c_{'L' if act[0] == 'left' else 'R'}" if len(act) == 1 else "c_robot"  # the centre used
        print(
            f"{MARK} [KIN_PLACE] {tree} placement {a.placement} · centre {cname} {cp['target_center_scene']} · s_clip "
            f"{cp['start_center_scene']} · o_clip {o} · tree shift {cp['tree_shift_xy']} · ψ {yaw} · pivot {pivot[:2]}",
            flush=True,
        )  # applied marker
        src_txt = (
            "e5_kin clip move (start wrist centre → c_robot) + ψ_ds — placement search · no judge (kin_prep · "
            "kin_offsets)"
        )
    else:
        cname = psrc
        print(
            f"{MARK} [KIN_PLACE] {tree} placement {a.placement} · placement_src={psrc} · record "
            f"{cp['record_sha256']} · "
            f"tree shift {cp['tree_shift_xy']} · ψ {yaw} · pivot {pivot[:2]} · robot_z_off {cp['robot_z_off']}",
            flush=True,
        )  # applied marker
        src_txt = f"e5_kin pipeline — {psrc}({cp['record']}) as is · no judge (kin_prep · kin_offsets)"
    return cname, src_txt


def _own_layers(root: Path, main: Path, combo: str, ds: str, emb: str) -> tuple[Path, Path]:
    """Make the written layers real folders inside the shadow; returns the robots and placement folders."""
    from dexengine.core import paths as P  # noqa: PLC0415

    for rel in (
        f"precompute/{ds}/robots/{combo}",
        f"data/{ds}/trees/{combo}",
    ):  # empty layer, never mixed with our trees and caches
        if not (root / rel).is_dir() or not inside(root, root / rel):
            own_dir(root, main, rel)
    rdir = P.robots_dir(combo, ds)
    pdir = P.placement_dir(combo, ds)
    if not inside(root, rdir) or not inside(root, P.trees_dir(combo, ds, emb).parent):
        raise SystemExit(f"{MARK} output location {rdir} is outside the shadow ({root}) — never written to the base")
    pdir.mkdir(parents=True, exist_ok=True)
    return rdir, pdir


def _adopted_record(a, tree, cp, offs, o, dx_t, dy_t, yaw, pivot, robot_z, src_txt, act, idle, rule, psrc) -> dict:
    """Placement record (adopted.json form) of the kinematic tree — written in place of the judge."""
    adopted = {
        "sequence": tree,
        "dx": round(dx_t, 6),
        "dy": round(dy_t, 6),
        "yaw": yaw,
        "pivot": pivot,
        "robot_z_off": robot_z,
        "anchor": 0,
        "valid": True,
        "source": src_txt,
        "active_sides": list(act),
        "idle_sides": list(idle),
        "active_sides_source": f"{rule}@e5_kin",
        "e5_kin": {
            "rule": offs["rule"],
            "placement": a.placement,
            "placement_src": psrc,
            "offset_scene": o,
            "yaw": yaw,
            "pivot": pivot,
            "start_center_scene": cp["start_center_scene"],
            "record": cp.get("record"),
            "record_sha256": cp.get("record_sha256"),
            "offsets_file": str(a.offsets),
        },
    }
    return adopted


def _source_tree(a, root: Path, main: Path, troot: Path, tree: str, combo, ds, emb, clip, act, idle, adopted: dict):
    """Bake the source tree, or copy our tree (--our-trees; then its active/idle hands update adopted).

    Returns the active and idle hands.
    """
    from dexengine.placement.bake import chain as SR  # noqa: PLC0415

    sp = (
        f"bake:--combo,{combo},--base,{clip},--processed_root,{root / 'data' / ds / 'base'},--dataset,{ds},"
        f"--emb,{emb},--src-layer,base"
    )
    if getattr(a, "our_trees", False):
        # --our-trees (default off): copy our trees (the base's data/<ds>/trees — retiming and f0 repair layers as they
        # are) instead of baking
        ours = main / "data" / ds / "trees" / combo / emb / tree
        if not (ours / "0" / "trajectory_kinematic.npz").is_file() or (troot / tree).exists():
            raise SystemExit(f"{MARK} --our-trees: our tree {ours} missing or {tree} already in the shadow")
        shutil.copytree(ours, troot / tree, symlinks=False)
        print(
            f"{MARK} [KIN_OURTREE] {tree} ← {ours} (retiming · f0 layer as is · no bake)", flush=True
        )  # applied marker
        to = json.loads(
            (troot / tree / "task_info.json").read_text()
        )  # our tree's active/idle (our idle-hand rule) also decides the placement record
        if to.get("active_sides") and list(to["active_sides"]) != list(act):
            print(
                f"{MARK} [KIN_OURTREE] {tree} active hands = our tree {to['active_sides']} (base rule {list(act)})",
                flush=True,
            )
            act, idle = list(to["active_sides"]), [x for x in SIDES if x not in to["active_sides"]]
            adopted.update(active_sides=list(act), idle_sides=list(idle), active_sides_source="our_tree@e5_kin")
    else:
        rc, _ex = SR.run_one(sp)
        if rc:
            raise RuntimeError(f"step failed rc={rc}: bake")
    return act, idle


def _robot_z(psrc, troot: Path, tree: str, offs: dict, o, cp: dict) -> float:
    """Robot z offset of the placement.

    The env adds scene_z_offset (desk = DESK_HEIGHT − desk_z, aerial = z_off) to the tree z, so the robot is moved back
    by that much: the robot's world z = capture z + DESK_HEIGHT + o_z (the z of the clip move). z_def = root z of the
    robot default pose (canonical TACO layer — offsets.robot_default_z reads dexengine.placement.search_policy).
    """
    from dexengine.placement.coords import core as CO  # noqa: PLC0415
    from experiments.policies.kin_data import offsets as KO  # noqa: PLC0415 — owner of the placement rule

    if psrc is None:
        robot_z = round(
            KO.robot_default_z() + float(CO.scene_z_offset(troot / tree)) - (float(offs["desk_height"]) + float(o[2])),
            5,
        )
    else:  # pipeline: robot_z_off of our stage 1 (or opt.py) as is — not measured again
        robot_z = round(float(cp["robot_z_off"]), 5)
    return robot_z


def _write_e5_kin(a, troot, tree, reach, offs, psrc, cp, o, dx_t, dy_t, yaw, pivot, cname, robot_z, ik, idle):
    """Write task_info.e5_kin of the source tree; returns the offsets table sha and the task_info."""
    import hashlib  # noqa: PLC0415

    offs_sha = hashlib.sha256(Path(a.offsets).read_bytes()).hexdigest()[:16]
    ti = json.loads((troot / tree / "task_info.json").read_text())
    ti["e5_kin"] = {
        "reach": reach,
        "offsets_sha256": offs_sha,
        "rule": offs["rule"],  # rule text = the one placement table
        "placement": a.placement,
        "placement_src": psrc,
        "record": cp.get("record"),
        "record_sha256": cp.get("record_sha256"),
        "offset_scene": o,
        "tree_shift_xy": [dx_t, dy_t],
        "yaw": yaw,
        "pivot": pivot,
        "start_center_scene": cp["start_center_scene"],
        "target_center": cname,
        "target_center_scene": cp["target_center_scene"],
        "robot_z_off": robot_z,
        "ik": {s: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()} for s, d in ik.items()},
        "idle_sides": list(idle),
        "code": "experiments/policies/kin_data/prep.py",
    }
    (troot / tree / "task_info.json").write_text(json.dumps(ti, indent=1, ensure_ascii=False))
    return offs_sha, ti


def prep_clip(a, ds: str, emb: str, clip: str) -> dict:
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.core import qpos_layout as QL  # noqa: PLC0415
    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415

    fl = set_env(ds, emb)
    combo = f"{ARM}_{HAND}{fl['variant']}"
    tree = P.tree_name(clip)
    root, main = Path(a.root), Path(a.main)
    src = place_base(root, main, Path(a.inputs), ds, emb, clip)
    ti0 = json.loads((src / "task_info.json").read_text())
    act, idle, rule = IA.active_idle_for_tree(ti0.get("case"), QL.read(str(src), nq=None).hands)
    from experiments.policies.kin_data import offsets as KO  # noqa: PLC0415 — owner of the placement rule

    offs = _load_offsets(a, ds)
    cp = KO.clip_placement(offs, Path(a.inputs), ds, clip, list(act), shadow=root)  # clip placement (owner offsets.py)
    o, (dx_t, dy_t), yaw, pivot = cp["offset_scene"], cp["tree_shift_xy"], cp["yaw"], cp["pivot"]
    psrc = cp.get("placement_src")  # pipeline = adopted | kin_opt · clipside = None
    cname, src_txt = _placement_marker(a, tree, cp, act, o, yaw, pivot, psrc)
    robot_z = 0.0  # measured again with the scene z offset after the bake (below; adopted is rewritten with it)
    rdir, pdir = _own_layers(root, main, combo, ds, emb)
    adopted = _adopted_record(a, tree, cp, offs, o, dx_t, dy_t, yaw, pivot, robot_z, src_txt, act, idle, rule, psrc)
    ad_p = pdir / f"{tree}{P.PLACEMENT_SFX}"
    ad_p.write_text(json.dumps(adopted, indent=1, ensure_ascii=False))
    troot = P.trees_dir(combo, ds, emb)
    act, idle = _source_tree(a, root, main, troot, tree, combo, ds, emb, clip, act, idle, adopted)
    robot_z = _robot_z(psrc, troot, tree, offs, o, cp)
    adopted["robot_z_off"] = robot_z
    ad_p.write_text(json.dumps(adopted, indent=1, ensure_ascii=False))
    act, idle, seed = match_inputs(a, ds, tree, act, idle)
    ik, reach = finish_tree(
        combo, ds, troot, tree, rdir, robot_z, act, idle, seed, main if getattr(a, "ref_arm", False) else None
    )
    offs_sha, ti = _write_e5_kin(
        a, troot, tree, reach, offs, psrc, cp, o, dx_t, dy_t, yaw, pivot, cname, robot_z, ik, idle
    )
    return {
        "ds": ds,
        "emb": emb,
        "clip": clip,
        "tree": tree,
        "combo": combo,
        "active": list(act),
        "idle": list(idle),
        "ik": ik,
        "reach": reach["reach"],
        "reach_per_side": reach["per_side"],
        "reach_pass": reach["passes"],
        "n_jump": reach["n_jump"],
        "max_dq_ratio": reach["max_dq_ratio"],
        "offsets_sha256": offs_sha,
        "placement": a.placement,
        "placement_src": psrc,
        "scene": ti.get("scene") or "table",
    }
