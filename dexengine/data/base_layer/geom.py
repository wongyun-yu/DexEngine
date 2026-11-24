"""Geometry and scene-file helpers of the base adapter: world transform, duplicate objects, palm FK oracle, qpos
layout, array slicing, hand direction of a clip."""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P  # noqa: E402
from dexengine.data.base_layer.common import _dest, dl  # noqa: F401

JOINT_W = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
DUP_POS_M, DUP_ANG_DEG = 0.005, 2.0


# ── Dataset hand direction (the yaw alignment reference) ─────────────────
def _circ_mean_deg(a: np.ndarray) -> tuple[float, float]:
    r = np.exp(1j * np.radians(a))
    m = r.mean()
    return float(np.degrees(np.angle(m))), float(np.degrees(np.sqrt(-2 * np.log(max(abs(m), 1e-9)))))


def clip_hand_dir_deg(mano: dict, sides) -> float:
    """Circular mean of one clip's hand direction (wrist→fingertip xy angle) — the same definition
    as hand_dir_stats, at clip scope."""
    angs = []
    for s in sides:
        if f"qpos_finger_{s}" not in mano:
            continue
        tips = mano[f"qpos_finger_{s}"][:, :, :3]
        wr = mano[f"qpos_wrist_{s}"][:, :3]
        if not np.isfinite(tips).all() or np.abs(tips).max() == 0:
            continue
        v = (tips.mean(1) - wr)[:, :2]
        v = v[np.linalg.norm(v, axis=1) > 1e-3]
        if len(v):
            angs.append(np.degrees(np.arctan2(v[:, 1], v[:, 0])))
    if not angs:
        raise RuntimeError("no frame to measure the clip hand direction")
    return _circ_mean_deg(np.concatenate(angs))[0]


# ── XML → qpos layout ───────────────────────────────────────────────────────────
def _hand_qpos_dim(hand: str) -> int | None:
    """Hand qpos width declared by HandSpec. None when unreadable (the guard is skipped; no value is invented)."""
    try:
        return int(P.load_registry("hand_specs").HAND_REGISTRY[P.hand_id(hand)].hand_qpos_dim)
    except (ImportError, KeyError, AttributeError, ValueError, TypeError) as e:
        print(f"[HAND_QPOS_DIM] {hand}: unreadable ({type(e).__name__}: {e}) — width guard skipped", flush=True)
        return None


def qpos_layout_from_root(root) -> list[dict]:
    wb = root.find("worldbody")
    out = []
    q = 0
    for body in wb.findall("body"):
        name = body.get("name", "")
        js = [(j.get("name", ""), j.get("type", "hinge")) for j in body.iter("joint")] + [
            (j.get("name", ""), "free") for j in body.iter("freejoint")
        ]
        if not js:
            continue
        w = sum(JOINT_W[t] for _, t in js)
        # hands are picked **by structure, not by name**: vendor link names (`_palm_link` · `_link_mount` …) differ per
        # hand, and picking by name finds **zero** hands on another hand (the hand layout check then sees []).
        # Rule (applies to hands never seen before): under the PATH A rig contract a hand carries its 6-DoF floating
        # base as **six joints** (`{side}_pos_*` · `{side}_rot_*`), so it has no free joint. Only objects attach with a
        # freejoint (width 7). Hence "root bodies without a free joint" = hands.
        if not any(t == "free" for _, t in js):
            out.append({"name": name, "start": q, "width": w, "kind": "hand", "side": name.split("_")[0]})
            q += w
            continue
        pos = q
        for jn, t in js:
            if t == "free":
                role = "extra" if name.startswith("extra_object") else "main"
                out.append(
                    {
                        "name": name,
                        "start": pos,
                        "width": 7,
                        "kind": "object",
                        "side": name.split("_")[0] if role == "main" else None,
                        "role": role,
                    }
                )
                pos += 7
            else:
                out.append(
                    {"name": jn, "start": pos, "width": JOINT_W[t], "kind": "joint", "object": name, "joint_type": t}
                )
                pos += JOINT_W[t]
        q = pos
    return out


def _objects_layer_rel(ds: str, retarget_path: str | None):
    if not retarget_path:
        return None, None
    key = "/assets/objects/"
    if key not in retarget_path:
        raise ValueError(f"unexpected object path form: {retarget_path!r}")
    rel = retarget_path.split(key, 1)[1].strip("/")
    p = _dest("assets_objects", ds) / rel
    if not p.is_dir():
        raise FileNotFoundError(f"object asset missing: {p}")
    return rel.split("/")[0], dl.object_ref(
        ds, rel
    )  # stored = logical name (dexengine.core.layout); bytes unchanged by physical moves


def build_objects(ds: str, ti: dict, layout: list[dict], kin: dict) -> list[dict]:
    objs = []
    blocks = [e for e in layout if e["kind"] == "object"]
    for i, e in enumerate(blocks):
        if e["role"] == "extra":
            k = int(e["name"].split("_")[-1])
            ex = (ti.get("extra_objects") or [])[k]
            oid, mesh = _objects_layer_rel(ds, ex.get("mesh_dir"))
            _, conv = _objects_layer_rel(ds, ex.get("convex_dir"))
            objs.append(
                {
                    "name": e["name"],
                    "role": "extra",
                    "side": None,
                    "obj_id": ex.get("obj_id", oid),
                    "mesh_dir": mesh,
                    "convex_dir": conv,
                    "spider_mesh_dir": ex.get("mesh_dir"),
                    "spider_convex_dir": ex.get("convex_dir"),
                    "qpos_start": e["start"],
                    "qpos_width": 7,
                    "slot": i,
                    "npz_key": f"extra_obj_{k}_qpos" if f"extra_obj_{k}_qpos" in kin else None,
                    "articulation": None,
                }
            )
            continue
        side = e["side"]
        oid, mesh = _objects_layer_rel(ds, ti.get(f"{side}_object_mesh_dir"))
        _, conv = _objects_layer_rel(ds, ti.get(f"{side}_object_convex_dir"))
        if mesh is None:
            raise ValueError(f"scene has {e['name']} but task_info.{side}_object_mesh_dir is empty")
        art = None
        ao = ti.get("articulated_object")
        if ao and ao.get("side") == side:
            hinge = [j for j in layout if j["kind"] == "joint" and j["object"] == e["name"]]
            if len(hinge) != 1:
                raise ValueError(f"articulated object {e['name']}: {len(hinge)} XML joints (must be 1)")
            key = f"qpos_obj_arti_{side}"
            if key not in kin:
                raise KeyError(f"articulated object but {key} is not in the npz")
            col = kin["qpos"][:, hinge[0]["start"]]
            sep = kin[key][:, 0]
            art = {
                "type": hinge[0]["joint_type"],
                "joint": hinge[0]["name"],
                "axis": ao.get("joint_axis"),
                "pos": ao.get("joint_pos"),
                "range": ao.get("joint_range"),
                "qpos_index": hinge[0]["start"],
                "npz_key": key,
                "parts": {"base": ao.get("bottom_mesh"), "moving": ao.get("top_mesh")},
                "max_abs_diff_vs_qpos": float(np.abs(col - sep).max()),
            }
        objs.append(
            {
                "name": e["name"],
                "role": "main",
                "side": side,
                "obj_id": oid,
                "mesh_dir": mesh,
                "convex_dir": conv,
                "spider_mesh_dir": ti.get(f"{side}_object_mesh_dir"),
                "spider_convex_dir": ti.get(f"{side}_object_convex_dir"),
                "qpos_start": e["start"],
                "qpos_width": 7,
                "slot": i,
                "npz_key": None,
                "articulation": art,
            }
        )
    return objs


# ── ① Merge duplicate objects ────────────────────────────────────────────────
def _quat_angle_deg(qa, qb):
    r = Rot.from_quat(qa[:, [1, 2, 3, 0]]) * Rot.from_quat(qb[:, [1, 2, 3, 0]]).inv()
    return np.degrees(r.magnitude())


def find_duplicate(objs: list[dict], kin: dict):
    """The first pair where the same mesh is in two bodies with the same pose sequence (keep, drop).
    main is kept, and right_object is kept in preference to left_object (in hocap the main object is
    also in extra)."""

    def rank(o):
        return (0 if o["role"] == "main" else 1, 0 if o["name"] == "right_object" else 1, o["slot"])

    cand = sorted([o for o in objs if not o["articulation"]], key=rank)
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            same_id = re.sub(r"^(right|left)_", "", a["obj_id"]) == re.sub(
                r"^(right|left)_", "", b["obj_id"]
            )  # oakink2: the same part appears twice, as right_/left_ mesh directories
            if a["mesh_dir"] != b["mesh_dir"] and not same_id:
                continue
            qa = kin["qpos"][:, a["qpos_start"] : a["qpos_start"] + 7]
            qb = kin["qpos"][:, b["qpos_start"] : b["qpos_start"] + 7]
            if (
                np.abs(qa[:, :3] - qb[:, :3]).max() < DUP_POS_M
                and _quat_angle_deg(qa[:, 3:], qb[:, 3:]).max() < DUP_ANG_DEG
            ):
                return a, b
    return None


def remove_body_from_xml(root, body_name: str) -> dict:
    """Remove <body name=body_name> from worldbody, and the pair/sensor/equality/mesh that reference
    it. Returns the removed names."""
    parent = {c: p for p in root.iter() for c in p}
    wb = root.find("worldbody")
    body = next((b for b in wb.findall("body") if b.get("name") == body_name), None)
    if body is None:
        raise KeyError(f"body {body_name} not in the XML")
    names = {el.get("name") for el in body.iter() if el.get("name")}
    meshes = {el.get("mesh") for el in body.iter("geom") if el.get("mesh")}
    wb.remove(body)
    removed = {"body": body_name, "elements": len(names), "meshes": 0, "pairs": 0, "sensors": 0, "equality": 0}
    other_mesh_refs = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for tag, key in (("pair", "pairs"), ("sensor", "sensors"), ("equality", "equality")):
        for el in list(root.iter(tag) if tag != "pair" else root.iter("pair")):
            attrs = el.attrib
            if tag == "pair" and (attrs.get("geom1") in names or attrs.get("geom2") in names):
                parent[el].remove(el)
                removed[key] += 1
        if tag in ("sensor", "equality"):
            sec = root.find(tag)
            if sec is not None:
                for el in list(sec):
                    if any(v in names for v in el.attrib.values()):
                        sec.remove(el)
                        removed[key] += 1
    asset = root.find("asset")
    if asset is not None:
        for m in list(asset.findall("mesh")):
            if m.get("name") in meshes and m.get("name") not in other_mesh_refs:
                asset.remove(m)
                removed["meshes"] += 1
    return removed


# ── ② yaw alignment ───────────────────────────────────────────────────────────────
def transform_world(kin: dict, layout: list[dict], M: np.ndarray, keys=("qpos", "qpos_rollout")) -> None:
    """World rigid transform — M is a constant (4,4) or a per-frame (T,4,4). Applied as p' = R p + t,
    R' = R·R_old to the hand root (slide xyz + rotation XYZ Euler = world position · world rotation;
    the palm body quat in scene.xml turns the axes onto world x/y/z) and to the objects (free joint,
    quaternion wxyz). The caller confirms it with the palm FK oracle (shared by yaw and the h2o
    correction)."""
    M = np.asarray(M, float)
    for key in keys:
        if key not in kin:
            continue
        q = kin[key]
        T = q.shape[0]
        Mt = np.broadcast_to(M, (T, 4, 4)) if M.ndim == 2 else M
        if Mt.shape[0] != T:
            raise ValueError(f"transform_world: M frame count {Mt.shape[0]} != qpos T {T} ({key})")
        Rm = Rot.from_matrix(Mt[:, :3, :3])
        Rmat = Mt[:, :3, :3]
        tm = Mt[:, :3, 3]
        for e in layout:
            s = e["start"]
            if e["kind"] == "hand":
                q[:, s : s + 3] = np.einsum("nij,nj->ni", Rmat, q[:, s : s + 3]) + tm
                R = Rot.from_euler("XYZ", q[:, s + 3 : s + 6])
                e3 = (Rm * R).as_euler("XYZ")
                q[:, s + 3 : s + 6] = np.unwrap(e3, axis=0)
            elif e["kind"] == "object":
                q[:, s : s + 3] = np.einsum("nij,nj->ni", Rmat, q[:, s : s + 3]) + tm
                R = Rot.from_quat(q[:, [s + 4, s + 5, s + 6, s + 3]])
                xyzw = (Rm * R).as_quat()
                q[:, s + 3 : s + 7] = xyzw[:, [3, 0, 1, 2]]
    for e in layout:  # hocap: extra_obj_k_qpos is a copy of the qpos block
        if e["kind"] == "object" and e["role"] == "extra":
            k = e["name"].split("_")[-1]
            if f"extra_obj_{k}_qpos" in kin:
                kin[f"extra_obj_{k}_qpos"] = kin["qpos"][:, e["start"] : e["start"] + 7].copy()


def rotate_world(kin: dict, layout: list[dict], psi_deg: float) -> None:
    if abs(psi_deg) < 1e-9:
        return
    M = np.eye(4)
    M[:3, :3] = Rot.from_euler("z", psi_deg, degrees=True).as_matrix()
    transform_world(kin, layout, M)


def check_fk_oracle(
    model, kin: dict, want: list[str], probe: np.ndarray, before, M: np.ndarray, what: str, hand: str
) -> None:
    """Confirm that palm FK(after) == M_k · palm FK(before). M is (4,4) or (T,4,4)."""
    after = palm_fk(model, kin["qpos"][probe], want, hand)
    M = np.asarray(M, float)
    for pi, (bf, af) in enumerate(zip(before, after, strict=True)):
        Mk = M if M.ndim == 2 else M[probe[pi]]
        for (p0, R0), (p1, R1) in zip(bf, af, strict=True):
            if np.abs(Mk[:3, :3] @ p0 + Mk[:3, 3] - p1).max() > 1e-6 or np.abs(Mk[:3, :3] @ R0 - R1).max() > 1e-6:
                raise RuntimeError(f"{what} oracle failed: FK palm pose differs from M·old pose")


def recompute_qvel(model, qpos: np.ndarray, dt: float) -> np.ndarray:
    import mujoco

    T = len(qpos)
    v = np.zeros((T, model.nv))
    buf = np.zeros(model.nv)
    for t in range(T - 1):
        mujoco.mj_differentiatePos(model, buf, dt, qpos[t], qpos[t + 1])
        v[t] = buf
    if T > 1:
        v[-1] = v[-2]
    return v


def load_model(xml_text: str, ds: str, hand: str):
    import mujoco

    from dexengine.data import intake_ingest as IH

    root = str(IH.scene_assets_root(ds, hand).resolve()) + "/"
    return mujoco.MjModel.from_xml_string(
        re.sub(r'(meshdir|texturedir)="[^"]*"', lambda m: f'{m.group(1)}="{root}"', xml_text)
    )


def _palm_body(hand: str) -> str:
    """FK palm body name of this hand, owned by HandSpec.palm_fk_body_right.

    No literal name: palm body names differ per hand (`right_link_mount` etc.), and for an unknown name `mj_name2id`
    returns **-1**, and `d.xpos[-1]` silently gives **the last body**, so the oracle would compare the wrong body.
    """
    try:
        return str(P.load_registry("hand_specs").HAND_REGISTRY[P.hand_id(hand)].palm_fk_body_right)
    except Exception as e:  # noqa: BLE001 — no invented name
        raise RuntimeError(f"could not read palm_fk_body_right of hand {hand!r}: {e}") from e


def palm_fk(model, qpos_rows: np.ndarray, sides: list[str], hand: str):
    import mujoco

    bname = _palm_body(hand)
    d = mujoco.MjData(model)
    out = []
    ids = {}
    for s_ in sides:
        n = bname.replace("right_", f"{s_}_", 1)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid < 0:  # using -1 as is would silently read the last body
            raise RuntimeError(f"FK palm body {n!r} not in the model (hand {hand}) — check HandSpec.palm_fk_body_right")
        ids[s_] = bid
    for q in qpos_rows:
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        out.append([(d.xpos[ids[s_]].copy(), d.xmat[ids[s_]].copy().reshape(3, 3)) for s_ in sides])
    return out


def segments_for(seg_rec: dict | None, T: int, off: int = 0) -> list[dict]:
    if not seg_rec or not seg_rec.get("cuts"):
        return [{"index": 0, "n": 1, "start": 0, "end": T, "idle": False, "cut_source": None}]
    cuts = sorted({min(max(int(c) - off, 1), T - 1) for c in seg_rec["cuts"]})
    bounds = [0] + cuts + [T]
    idle = seg_rec.get("idle") or [False] * (len(bounds) - 1)
    return [
        {
            "index": i,
            "n": len(bounds) - 1,
            "start": a,
            "end": b,
            "idle": bool(idle[i]),
            "cut_source": seg_rec.get("cut_source"),
        }
        for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
    ]


def slice_arrays(arrs: dict, a: int, b: int, T_full: int) -> tuple[dict, list[str]]:
    out = {}
    dropped = []
    for k, v in arrs.items():
        v = np.asarray(v)
        if v.ndim == 0:
            out[k] = v
        elif v.shape[0] == T_full:
            out[k] = v[a:b]
        else:
            dropped.append(f"{k}{list(v.shape)}")
    return out, dropped
