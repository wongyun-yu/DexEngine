"""Scene layout of one episode: the owner reads (desk, robot base, head camera chain, trees and shadows) and the
per-episode line of `meta/scene_layout.jsonl` (episode_line); scene_layout.py syncs and checks the sidecar."""

from __future__ import annotations

import json
import math
import os
from functools import cache, lru_cache
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core import stdio
from dexengine.core.errors import input_missing

MARK = "[SCENE_LAYOUT]"


SCHEMA = 1


FRAME = "render_env"  # render scene env origin, z up, m, rad, floor z 0


METHODS = ("none", "torso", "world_shift", "desk_stretch")


SCENES = ("desk", "aerial")  # episode framing.scene


DESK_ORACLE_TOL_M = 1e-3


TORSO_ORACLE_TOL_M = 1e-5


CAM_ORACLE_TOL_M = 1e-3


OBJ_SIDE_PREFIX, EXTRA_PREFIX = "object_", "extra_obj_"


ROUND = 6


def die(msg: str):
    raise SystemExit(f"{MARK} ✗ {msg}")


# ---- owner reads -----------------------------------------------------------------------------------------------
@cache
def _desk(ds: str) -> tuple:
    from dexengine.placement.coords import core as C

    dp = C.desk_placement()
    x, hx, hy = C.desk_top_for(ds)
    return float(x), float(hx), float(hy), float(dp.DESK_TOP_THICK), float(dp.DESK_TOP_CENTER_Z) + dp.DESK_TOP_THICK / 2


@lru_cache(maxsize=4)
def _base_pos(arm: str) -> tuple:
    from dexengine.core.arm_generations import ARM_GENERATIONS

    gens = ARM_GENERATIONS.get(arm)
    if not gens:
        die(input_missing(f"no generation table for arm {arm!r} (core.arm_generations.ARM_GENERATIONS)"))
    pos = {tuple(float(v) for v in g.base_pos) for g in gens.values()}
    if len(pos) != 1:
        die(f"arm {arm} generations have different base_pos {sorted(pos)} — the render generation must be explicit")
    return next(iter(pos))


@lru_cache(maxsize=1)
def _chain():
    from dexengine.record.render import body as CB

    ch = CB.chain()
    return ch, CB.fk(ch, ch["q0"])


@lru_cache(maxsize=1)
def _head():
    """(torso end -> ZED parent link chain [(name, type, xyz, rpy, axis)], left-eye position) from the arm URDF."""
    import xml.etree.ElementTree as ET

    from dexengine.robots.usd import head_camera as HC

    ch, _ = _chain()
    rig = HC.zedm_rig()
    by_child = {j.find("child").get("link"): j for j in ET.parse(ch["urdf"]).getroot().iter("joint")}
    seq, link, top = [], rig["parent"], ch["seq"][-1]["child"]
    while link != top:
        j = by_child.get(link)
        if j is None:
            die(input_missing(f"the arm URDF has no chain {rig['parent']} -> {top}"))
        o, a = j.find("origin"), j.find("axis")
        seq.append(
            (
                j.get("name"),
                j.get("type"),
                [float(v) for v in o.get("xyz").split()],
                [float(v) for v in (o.get("rpy") or "0 0 0").split()],
                [float(v) for v in a.get("xyz").split()] if a is not None else None,
            )
        )
        link = j.find("parent").get("link")
    return list(reversed(seq)), [float(v) for v in rig["left_pos"]]


def _cam_left(base_xyz, torso_q: dict, head_rad: dict):
    import numpy as np

    from dexengine.record.render import body as CB

    ch, _ = _chain()
    T = np.eye(4)
    T[:3, 3] = base_xyz
    T = T @ CB.fk(ch, torso_q)[ch["seq"][-1]["child"]]
    seq, left = _head()
    for name, kind, xyz, rpy, axis in seq:
        M = np.eye(4)
        R = CB._rpy(*rpy)
        if kind != "fixed":
            if name not in head_rad:
                raise KeyError(name)
            R = R @ CB._rot(axis, head_rad[name])
        M[:3, :3], M[:3, 3] = R, xyz
        T = T @ M
    return (T @ np.r_[left, 1.0])[:3]


def _emb(side: str) -> str:
    from dexengine.record.store import form as CF

    if side == CF.BOTH:
        return "bimanual"
    if side not in CF.sides():
        die(f"line side {side!r} — {CF.BOTH} | {list(CF.sides())}")
    return side


def _tree_at(tree: str) -> dict | None:
    """Tree folder -> {discs, main, extras, objects}, or None without task_info."""

    from dexengine.data.objects import extra_slots as XS

    tree = Path(tree)
    if not (tree / "task_info.json").is_file():
        return None
    lay = P.qpos_layout.read(str(tree))
    ti = json.loads((tree / "task_info.json").read_text())
    stamp = lay.aerial if getattr(lay, "scene", "table") == "aerial" else lay.desk_support
    with stdio.capture():  # the rule's marker lines are not this tool's output
        rows, _routed = XS.slot_rows(str(tree), ti)
    return {
        "discs": list((stamp or {}).get("discs") or []),
        "main": {s: lay.main_object_id(s) for s in ("right", "left")},
        "extras": [r["name"] for r in rows],
        "objects": {o.get("name"): {k: o.get(k) for k in ("obj_id", "name", "role")} for o in ti.get("objects") or []},
    }


def _tree_rel(arm: str, hand: str, ds: str, side: str, clip: str) -> str:
    """Tree path relative to a data root (the data root and a piece's shadow root use the same relative layout)."""
    from dexengine.record.augment import scene as AA

    emb = _emb(side)
    return P.data_layout.render("trees", ds, combo=AA.tree_combo(arm, hand, ds, emb), emb=emb, tree=P.tree_name(clip))


@lru_cache(maxsize=4096)
def _tree(arm: str, hand: str, ds: str, side: str, clip: str) -> tuple:
    """Original line: (tree data | None, tree path | reason), from this machine's data root."""
    from dexengine.record.augment import scene as AA

    emb = _emb(side)
    tree = Path(P.layer_path("trees", ds, combo=AA.tree_combo(arm, hand, ds, emb), emb=emb, tree=P.tree_name(clip)))
    d = _tree_at(str(tree))
    return (None, f"no tree {tree}") if d is None else (d, str(tree))


@cache
def _shadow_rel() -> str:
    """Shadow root path inside a piece output (the '<path>/*' line of record.clean.CARRY)."""
    from dexengine.record import clean as CC

    g = [x.removesuffix("/*") for x in CC.CARRY if x.endswith("shadow/*")]
    if len(g) != 1:
        die(f"record.clean.CARRY has {len(g)} shadow root lines — need one")
    return g[0]


def aug_variant(rec: dict) -> bool:
    """Aug round and the clip is a placement variant (paths.variant_source): the line reads a variant tree."""
    if rec.get("round") != "aug":
        return False
    c = P.clip_name(rec["clip"])
    return P.variant_source(c) != c


def _variant_candidates(arm: str, hand: str, ds: str, rec: dict) -> list[str] | None:
    """Candidate variant tree paths of an aug line (piece archive roots in order); None without piece_id."""
    pid = rec.get("piece_id")
    if not pid or "/" in str(pid):
        return None
    rel = _tree_rel(arm, hand, ds, rec["side"], rec["clip"])
    return [
        os.path.join(r, "jobs", hand, ds, str(pid), _shadow_rel(), rel)
        for r in P.data_layout.collect_job_archive_roots("aug")
    ]


def _head_rad(hj: dict) -> dict:
    """head_joints -> {joint: rad}: render-pinned value, else the middle of the rollout range."""
    out = {}
    for n, r in (hj or {}).items():
        if r.get("source") == "render_pinned":
            out[n] = math.radians(float(r["render_deg"]))
        else:
            lo, hi = r["rollout_state_deg"]
            out[n] = math.radians((float(lo) + float(hi)) / 2)
    return out


def _line_on_tree(rec, arm, hand, ds, shift, r, dz, scene, m, desk, src, link, cerr, bx, by, bz, tz, q):
    """The tree part (support disc tops, object assets) checked against the render, then the sidecar line."""
    from dexengine.record.scene import supports as SUP

    tr, where = rec["vtree"] if aug_variant(rec) else _tree(arm, hand, ds, rec["side"], rec["clip"])
    if tr is None:
        return None, where
    discs = rec["discs"]
    if len(discs) != rec["n_on"]:
        return None, f"render discs {len(discs)} != standing discs in line supports {rec['n_on']}"
    sup = []
    for d in discs:
        k = int(str(d["slot"]).rsplit("_", 1)[1])
        if k >= len(tr["discs"]):
            return None, f"disc {d['slot']} has no stamp ({len(tr['discs'])} stamps)"
        s, c = tr["discs"][k], d["center"]
        top = float(s["z"]) + shift
        if max(abs(float(s["x"]) - c[0]), abs(float(s["y"]) - c[1])) > SUP.SUPPORT_POS_TOL_M or top <= c[2]:
            return None, f"disc oracle stamp ({float(s['x']):.4f},{float(s['y']):.4f},{top:.4f}) <-> render {c}"
        sup.append({"slot": d["slot"], "top_z_m": round(top, r), "height_m": round(2 * (top - float(c[2])), r)})
    objs = {}
    for col, on in sorted((rec["object_present"] or {}).items()):
        if not on:
            continue
        if col.startswith(EXTRA_PREFIX):
            k = int(col[len(EXTRA_PREFIX) :])
            if k >= len(tr["extras"]):
                return None, f"{col} stands but the tree has {len(tr['extras'])} extra slot rows"
            o = tr["objects"].get(tr["extras"][k]) or {}
        elif col.startswith(OBJ_SIDE_PREFIX) and col[len(OBJ_SIDE_PREFIX) :] in tr["main"]:
            oid = tr["main"][col[len(OBJ_SIDE_PREFIX) :]]
            o = next((v for v in tr["objects"].values() if v.get("obj_id") == oid), {}) if oid else {}
        else:
            die(f"{ds}/{rec['clip']}: unknown object column {col!r}")
        if not o.get("obj_id"):
            return None, f"the tree does not know the asset of {col}"
        objs[col] = {"obj_id": o["obj_id"], "name": o.get("name"), "role": o.get("role")}
    return {
        "episode_index": rec["i"],
        "clip": rec["clip"],
        "root_z_offset_m": dz,
        "schema": SCHEMA,
        "frame": FRAME,
        "scene": scene,
        "body_method": m,
        "robot": {
            "base_xyz_m": [round(bx, r), round(by, r), round(bz, r)],
            "base_yaw_rad": 0.0,
            "torso_joints_rad": {k: round(v, 9) for k, v in q.items()},
            "torso_q_source": src,
            "torso_link": link,
            "torso_link_z_m": round(bz + tz, r),
        },
        "desk": desk,
        "supports": sup or "none",
        "objects": objs,
        "oracle": {"cam_left_err_m": round(cerr, 7)},
    }, ""


# ---- one episode ---------------------------------------------------------------------------------------------
def episode_line(rec: dict, arm: str, hand: str, ds: str) -> tuple[dict | None, str]:
    """Extracted record -> (sidecar line, '') | (None, reason it is missing)."""
    from dexengine.record.render import body as CB
    from dexengine.record.render import desk_stretch as DS
    from dexengine.record.store import meta as CM

    b, scene = rec["body"], rec["scene"]
    m = b.get("method")
    if m not in METHODS:
        die(f"{ds}/{rec['clip']} ep {rec['i']}: body method {m!r} — {METHODS}")
    if scene not in SCENES:
        die(f"{ds}/{rec['clip']} ep {rec['i']}: scene {scene!r} — {SCENES}")
    dz = float(b["root_z_offset_m"])
    shift = CM.world_shift_m(b)
    r = ROUND
    if scene == "aerial":
        desk = "none"
    else:
        x, hx, hy, thick, top0 = _desk(ds)
        top = float(DS.desk_after({"top_z": top0}, b)["top_z"])
        if m == "desk_stretch" and abs(float(b["visual"]["collider"]["top_m"]) - top) > DESK_ORACLE_TOL_M:
            return None, f"desk oracle render {float(b['visual']['collider']['top_m']):.4f} != {top:.4f}"
        desk = {
            "center_xy_m": [round(x, r), 0.0],
            "yaw_rad": 0.0,
            "size_xyz_m": [round(2 * hx, r), round(2 * hy, r), round(thick, r)],
            "top_z_m": round(top, r),
        }
    bx, by, bz = _base_pos(arm)
    ch, T0 = _chain()
    link = ch["seq"][-1]["child"]
    if m == "torso":
        q, src = {k: float(v) for k, v in b["torso_rad"].items()}, "solved"
        canon = b.get("torso_canon_rad") or {}
        if set(canon) != set(ch["q0"]) or max(abs(float(canon[k]) - ch["q0"][k]) for k in ch["q0"]) > 1e-9:
            return None, "torso_canon_rad != canonical pose (the chain changed after the render)"
        bx += float(b["base_shift_m"][0])
    else:
        q, src = dict(ch["q0"]), "canon"
    if set(q) != set(ch["q0"]):
        die(f"{ds}/{rec['clip']}: torso joints {sorted(q)} != chain {sorted(ch['q0'])}")
    tz = float(CB.fk(ch, q)[link][2, 3])
    if m == "torso" and abs((tz - float(T0[link][2, 3])) - dz) > TORSO_ORACLE_TOL_M:
        return None, f"torso oracle dz {tz - float(T0[link][2, 3]):+.6f} != dz {dz:+.6f}"
    try:
        cam = _cam_left((bx, by, bz), q, _head_rad(rec["head_joints"]))
    except KeyError as e:
        return None, f"head_joints lacks {e}"
    rec_cam = rec["cam_left"]
    cerr = max(abs(float(a) - float(c)) for a, c in zip(cam, rec_cam, strict=False))
    if cerr > CAM_ORACLE_TOL_M:
        return None, f"camera oracle {cerr * 1000:.2f} mm"
    # tree (support disc tops, object assets): variant lines use their variant tree, others the original tree
    return _line_on_tree(rec, arm, hand, ds, shift, r, dz, scene, m, desk, src, link, cerr, bx, by, bz, tz, q)
