"""Placement augment: the shadow data root and the rigid transform on the desk plane (including the tree array
transform rigid_any; written by augment.place_gen and exported again)."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError, input_missing

MARK = "[AUG_GEN]"


# ── shadow root ─────────────────────────────────────────────
#: shadow layer build lock, beside the shadow root (.<root name>.shadow_dir.lock; inside the root it would mix into
#: the augment pack's move list)
SHADOW_LOCK = ".{}.shadow_dir.lock"


@contextlib.contextmanager
def _locked(shadow: Path):
    """Shadow root lock (SHADOW_LOCK · flock): all three writers of the root (shadow_dir · init_shadow · relink_shadow)
    hold this one. Without it, several clip gens of one spec building the same root at once move each other's identity
    record tmp and two of them create the same sibling symlink."""
    import fcntl

    shadow.parent.mkdir(parents=True, exist_ok=True)
    with open(shadow.parent / SHADOW_LOCK.format(shadow.name), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        yield


def _link(t: Path, ch: Path) -> bool:
    """Sibling symlink t -> ch; True when created. False when a link to the same target already exists (another writer
    made it first). Anything else dies (never silently overwritten or skipped)."""
    try:
        t.symlink_to(ch)
        return True
    except FileExistsError:
        if t.is_symlink() and os.readlink(t) == str(ch):
            return False
        raise SystemExit(
            f"{MARK} ✗ shadow {t} already exists and does not point to {ch} "
            f"({os.readlink(t) if t.is_symlink() else 'a real entry'})"
        ) from None


def shadow_dir(shadow: Path, main: Path, rel: str) -> Path:
    """Makes shadow/rel a real directory (ancestors real directories too; the other siblings are symlinks to main), the
    original entries of rel symlinks. Turning a symlink into a real directory happens inside the shadow root lock
    (_locked), so unlink, mkdir and sibling symlinks of two processes building the same layer first do not cross."""
    with _locked(shadow):
        return _shadow_dir(shadow, main, rel)


def _shadow_dir(shadow: Path, main: Path, rel: str) -> Path:
    parts = Path(rel).parts
    cur_s, cur_m = shadow, main
    for p in parts:
        # cur_s must be a real directory: fill the sibling symlinks
        if not cur_s.is_dir() or cur_s.is_symlink():
            raise SystemExit(f"{MARK} shadow {cur_s} is not a real directory")
        nxt_s, nxt_m = cur_s / p, cur_m / p
        if nxt_s.is_symlink():
            nxt_s.unlink()
        if not nxt_s.exists():
            nxt_s.mkdir()
            if nxt_m.is_dir():
                for ch in nxt_m.iterdir():
                    t = nxt_s / ch.name
                    if not t.exists() and not t.is_symlink():
                        _link(t, ch)
        cur_s, cur_m = nxt_s, nxt_m
    return cur_s


def relink_shadow(shadow: Path, main: Path) -> tuple[int, int]:  # noqa: V103 — used by experiments/policies/dense_render.py
    """Realigns the shadow layers to the data root main: fills missing siblings of every real directory with symlinks
    to main (a shadow received from another machine; the same layer rule as shadow_dir · init_shadow) and removes
    sibling symlinks pointing to entries deleted from main (layer = main + overwritten entries; tidying main's top
    entries leaves broken sibling links). Other entries are not touched. The identity record's base is rewritten to
    this machine's main too (data_layout.write_shadow_record).
    Returns (symlinks created, broken sibling symlinks removed)."""
    with _locked(shadow):
        return _relink_shadow(shadow, main)


def _relink_shadow(shadow: Path, main: Path) -> tuple[int, int]:
    from dexengine.core import paths as P

    P.data_layout.write_shadow_record(str(shadow), str(main), "aug_place")
    n = gone = 0
    for root, dirs, files in os.walk(shadow):
        rp = Path(root)
        rel = rp.relative_to(shadow)
        if rel.parts and rel.parts[0] == "_ik_tmp":
            dirs[:] = []
            continue
        m = main / rel
        for nm in dirs + files:  # sibling symlink = main / rel / name; removed when its target is gone
            t = rp / nm
            if t.is_symlink() and os.readlink(t) == str(m / nm) and not (m / nm).exists() and not (m / nm).is_symlink():
                t.unlink()
                gone += 1
        dirs[:] = [d for d in dirs if not (rp / d).is_symlink()]
        if not m.is_dir():
            continue
        for ch in m.iterdir():
            if rel == Path(".") and ch.name == ".git":
                continue
            t = rp / ch.name
            if not t.exists() and not t.is_symlink():
                n += _link(t, ch)
    return n, gone


def init_shadow(shadow: Path, main: Path) -> None:
    """Shadow root: links main's top entries as symlinks and writes the identity (data_layout.write_shadow_record, use
    aug_place). The data root guard accepts registered shadows only (links only to the base main; writing to base data
    dies)."""
    from dexengine.core import paths as P

    with _locked(shadow):
        shadow.mkdir(parents=True, exist_ok=True)
        P.data_layout.write_shadow_record(str(shadow), str(main), "aug_place")
        for ch in main.iterdir():
            if ch.name == ".git":
                continue
            t = shadow / ch.name
            if not t.exists() and not t.is_symlink():
                _link(t, ch)


# ── transform ────────────────────────────────────────────────
def rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def scene_to_tree_vec(d_scene_xy) -> np.ndarray:
    """Scene (robot forward +x · left +y) plane vector -> tree (capture, +Y forward) vector. scene = Rz(−90°)·tree ->
    tree = Rz(+90°)·scene."""
    v = rz(90.0) @ np.array([d_scene_xy[0], d_scene_xy[1], 0.0])
    return v[:2]


def tree_to_scene_pts(p) -> np.ndarray:
    return np.asarray(p) @ rz(-90.0).T


def quat_left(qz_wxyz, q):
    a1, b1, c1, d1 = qz_wxyz
    a2, b2, c2, d2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ],
        axis=-1,
    )


def qz_of(yaw_deg: float) -> np.ndarray:
    h = np.deg2rad(yaw_deg) / 2
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


class Tf:
    def __init__(self, dx_t, dy_t, yaw, pivot_t):
        self.R = rz(yaw)
        self.piv = np.array([pivot_t[0], pivot_t[1], 0.0])
        self.d = np.array([dx_t, dy_t, 0.0])
        self.qz = qz_of(yaw)

    def p(self, x):
        x = np.asarray(x, float)
        return (x - self.piv) @ self.R.T + self.piv + self.d

    def v(self, x):
        return np.asarray(x, float) @ self.R.T

    def q(self, q):
        return quat_left(self.qz, np.asarray(q, float))

    def pose7(self, p7) -> list:
        """One tree pose row [x, y, z, qw, qx, qy, qz] -> variant pose (the same transform of position and quaternion
        as the object block of rigid_any)."""
        p7 = np.asarray(p7, float)
        w = p7[3:7] / max(float(np.linalg.norm(p7[3:7])), 1e-12)
        return [float(v) for v in (*self.p(p7[:3]), *self.q(w))]


def variant_tf(d_scene, yaw: float, pivot_t) -> Tf:
    """Tree coordinate transform of a variant (scene plane shift d_scene m · yaw deg · tree rotation centre pivot_t);
    the side that builds variant trees (augment.place_src make_variant) and the side that reads their record
    (record_tf) make it with the same function."""
    d_t = scene_to_tree_vec(d_scene)
    return Tf(float(d_t[0]), float(d_t[1]), float(yaw), pivot_t)


def record_tf(ti: dict) -> Tf | None:
    """aug_place record in a variant tree's task_info (dx_scene · dy_scene · yaw_deg · pivot_tree) -> that variant's Tf
    · None without a record (original tree). A record with missing fields dies (read as identity the objects of the
    variant scene would stand at the original place)."""
    ap = ti.get("aug_place")
    if ap is None:
        return None
    miss = [k for k in ("dx_scene", "dy_scene", "yaw_deg", "pivot_tree") if ap.get(k) is None]
    if miss:
        raise InputMissingError(f"{MARK} aug_place record lacks {miss}: variant transform unknown")
    return variant_tf((ap["dx_scene"], ap["dy_scene"]), ap["yaw_deg"], ap["pivot_tree"])


_MAPS: dict = {}


def _base_maps(mcv, src, starts: list) -> dict:
    """Hand block addresses -> BaseMap (the hand root convention of the tree scene); both hands = the canonical owner
    retime_variants.maps_for · one-hand trees: maps_for assumes two blocks, so only the present blocks of the same
    scene model. Once per tree."""
    key = (str(src.tree), tuple(starts))
    if key not in _MAPS:
        if starts == [0, src.H]:
            m = mcv.maps_for(src.tree, src.H)
            _MAPS[key] = {0: m[0], src.H: m[1]}
        else:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(src.tree / "scene.xml"))
            _MAPS[key] = {b: mcv.BaseMap(model, b) for b in starts}
    return _MAPS[key]


def rigid_any(mcv, src, qpos, qvel, cp, dx_t, dy_t, yaw, pivot_t):
    """Applies the same transform as retime_variants.rigid_pl over **only the present hands** of the tree layout (for
    one-hand trees; rigid_pl assumes two hand blocks). Two-hand trees call rigid_pl as it is (the same function; the
    second implementation is the one-hand branch only). Identity (dx = dy = yaw = 0 exactly, the canonical cell pv00)
    returns the input as it is: through the transform 1 ulp (4.4e-16) stays in the quaternion columns and the cross
    repair grows it to 7.8e-3 rad on the arm (measured on a drawer clip, when the canonical cell pv00 was still baked
    through this transform; pv00 is now a copy of the canonical clip, place_src canon_variant)."""
    if dx_t == 0.0 and dy_t == 0.0 and yaw == 0.0:
        return qpos.copy(), (None if qvel is None else qvel.copy()), (None if cp is None else cp.copy())
    starts = sorted(src.lay.hands.values())
    bm = _base_maps(mcv, src, starts)
    if starts == [0, src.H]:
        return mcv.rigid_pl(
            qpos, qvel, cp, src.H, dx_t, dy_t, yaw, [pivot_t[0], pivot_t[1], 0.0], (bm[0], bm[src.H]), layout=src.lay
        )
    from scipy.spatial.transform import Rotation as Rot

    Rz = Rot.from_euler("z", np.deg2rad(yaw))
    Rzm = Rz.as_matrix()
    tf = Tf(dx_t, dy_t, yaw, pivot_t)
    q = qpos.copy()
    for lo in src.lay.object_starts():
        q[:, lo : lo + 3] = tf.p(q[:, lo : lo + 3])
        w = q[:, lo + 3 : lo + 7]
        w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
        q[:, lo + 3 : lo + 7] = tf.q(w)
    for b in starts:
        p_w, R_w = bm[b].to_world(qpos[:, b : b + 6])
        q[:, b : b + 6] = bm[b].from_world(tf.p(p_w), Rz * R_w)
    v = None
    if qvel is not None:
        v = qvel.copy()
        for b in starts:
            v[:, b : b + 3] = v[:, b : b + 3] @ (bm[b].Ainv @ Rzm @ bm[b].A).T
        for blk in src.lay.qvel_blocks():
            if blk[0] == "object":
                lo = blk[3]
                v[:, lo : lo + 3] = v[:, lo : lo + 3] @ Rzm.T
                v[:, lo + 3 : lo + 6] = v[:, lo + 3 : lo + 6] @ Rzm.T
    c = None
    if cp is not None:
        c = tf.p(cp.reshape(len(cp), -1, 3)).reshape(cp.shape)
    return q, v, c


ARTIC_MARK = "[AUG_ARTIC]"
RIGID_TOL = 1e-9  # oracle: the same rigid transform · part relative poses
JOINT_TYPES = ("hinge", "slide")


def die(msg: str):
    raise SystemExit(f"{ARTIC_MARK} {msg}")


# ── tree place ──────────────────────────────────────────────
def tree_combo(arm: str, hand: str, ds: str, emb: str) -> str:
    """(arm, hand, dataset, emb) -> tree combo. Tail = the dataset axis variant (owner dexengine.core.datasets)."""
    from dexengine.core import paths as P
    from dexengine.data import rules as RR

    try:
        var = RR.ledger_axis_variant(RR.ledger_key(ds, emb))
    except KeyError as e:
        die(input_missing(f"cannot resolve the tree tail: dataset {ds!r} emb {emb!r}: {e}"))
    return P.combo_id(arm, hand, var)


def check_tree_combo(ti: dict, combo: str, tree) -> None:
    """When the tree stamped a combo (task_info.combo) it must equal the resolved combo."""
    got = ti.get("combo")
    if got is not None and str(got) != combo:
        die(f"tree stamp combo={got!r} != resolved combo {combo!r}: {tree}")


# ── articulated statement ─────────────────────────────────────────────
def artic_spec(lay, kin: dict, ti: dict, tree) -> list:
    """Joint records -> [{name, col, type, axis, pos, root}] (rigid tree = []). Mismatches die. Joint value =
    qpos[:, col]."""
    recs = lay.artic_records()
    if not recs:
        return []
    q = kin["qpos"]
    kinds = {int(s): k for s, k, _w in lay.blocks()}
    roots = {str(o.get("name")): int(o["qpos_start"]) for o in ti.get("objects") or [] if "qpos_start" in o}
    out = []
    for r in recs:
        col, typ = int(r["col"]), str(r.get("type") or "")
        if not 0 <= col < q.shape[1] or kinds.get(col) != "joint":
            die(f"joint {r.get('name')} column {col} is not in the joint block of qpos width {q.shape[1]}: {tree}")
        if typ not in JOINT_TYPES:
            die(f"joint {r.get('name')} type {typ!r} unknown (known {JOINT_TYPES}): {tree}")
        if not np.isfinite(q[:, col]).all():
            die(f"joint {r.get('name')} qpos[:, {col}] has non-finite values: {tree}")
        if r.get("name") not in roots or roots[r["name"]] not in lay.object_starts():
            die(f"joint {r.get('name')}: object root block not found in objects[].qpos_start: {tree}")
        axis, pos = np.asarray(r.get("axis"), float), np.asarray(r.get("pos"), float)
        if axis.shape != (3,) or pos.shape != (3,) or np.linalg.norm(axis) < 1e-9:
            die(f"joint {r.get('name')} axis/pos statement missing {r.get('axis')}/{r.get('pos')}: {tree}")
        out.append(
            {
                "name": r["name"],
                "col": col,
                "type": typ,
                "axis": axis / np.linalg.norm(axis),
                "pos": pos,
                "root": roots[r["name"]],
            }
        )
    sup = (ti.get("aerial_support") or {}).get("supports") or []
    if sup:  # support objects: the generator does not move them; the rule moves the support as a whole, so it dies
        die(
            f"articulated tree with {len(sup)} aerial_support.supports: supports the generator does not move (outside "
            f"the rule): {tree}"
        )
    return out


# ── oracle ───────────────────────────────────────────────
def _qmat(w: np.ndarray) -> np.ndarray:
    """(T,4) wxyz → (T,3,3)."""
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    a, b, c, d = w.T
    return np.stack(
        [
            np.stack([1 - 2 * (c * c + d * d), 2 * (b * c - a * d), 2 * (b * d + a * c)], -1),
            np.stack([2 * (b * c + a * d), 1 - 2 * (b * b + d * d), 2 * (c * d - a * b)], -1),
            np.stack([2 * (b * d - a * c), 2 * (c * d + a * b), 1 - 2 * (b * b + c * c)], -1),
        ],
        1,
    )


def _rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _joint(axis, pos, typ, th):
    """Joint angle/displacement th (T,) -> pose of the child (moving part) relative to the parent (root) (R (T,3,3),
    p (T,3))."""
    T = len(th)
    if typ == "slide":
        return np.broadcast_to(np.eye(3), (T, 3, 3)).copy(), th[:, None] * axis[None]
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    R = np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]
    return R, pos[None] - R @ pos  # the rotation axis passes through pos: p_child = pos − R·pos


def part_poses(q: np.ndarray, rec: dict):
    """qpos -> (root R, root p, part R, part p) world (tree) poses; part = root ∘ joint."""
    lo = rec["root"]
    Rb, pb = _qmat(q[:, lo + 3 : lo + 7]), q[:, lo : lo + 3]
    Rj, pj = _joint(rec["axis"], rec["pos"], rec["type"], q[:, rec["col"]])
    return Rb, pb, Rb @ Rj, pb + np.einsum("tij,tj->ti", Rb, pj)


def oracle(spec: list, src_kin: dict, var_kin: dict, src_ikr: dict, var_ikr: dict, src_ti: dict, var_ti: dict) -> dict:
    """Articulated check values of one variant + ok. Takes the original arrays and the variant arrays (read back from
    the file)."""
    ap = var_ti["aug_place"]
    yaw, d = float(ap["yaw_deg"]), np.array([ap["dx_scene"], ap["dy_scene"], 0.0])
    piv = tree_to_scene_pts(np.array([ap["pivot_tree"][0], ap["pivot_tree"][1], 0.0]))
    R = _rz(yaw)
    tf_s = lambda x: (np.asarray(x, float) - piv) @ R.T + piv + d  # noqa: E731  (scene coordinate rigid transform, one record)
    o = {
        "hinge_bytes": True,
        "rel_pos": 0.0,
        "rel_rot": 0.0,
        "root_pos": 0.0,
        "root_rot": 0.0,
        "disc": 0.0,
        "n_disc": 0,
    }
    for rec in spec:
        c = rec["col"]
        for nm, a, b in (("qpos", src_kin, var_kin), ("qpos_rollout", src_kin, var_kin), ("ikr", src_ikr, var_ikr)):
            k = "qpos" if nm == "ikr" else nm
            if k in a and np.ascontiguousarray(a[k][:, c]).tobytes() != np.ascontiguousarray(b[k][:, c]).tobytes():
                o["hinge_bytes"] = False
        Rb0, pb0, Rm0, pm0 = part_poses(src_kin["qpos"], rec)
        Rb1, pb1, Rm1, pm1 = part_poses(var_kin["qpos"], rec)
        rel0 = (np.swapaxes(Rb0, 1, 2) @ Rm0, np.einsum("tji,tj->ti", Rb0, pm0 - pb0))
        rel1 = (np.swapaxes(Rb1, 1, 2) @ Rm1, np.einsum("tji,tj->ti", Rb1, pm1 - pb1))
        o["rel_rot"] = max(o["rel_rot"], float(np.abs(rel0[0] - rel1[0]).max()))
        o["rel_pos"] = max(o["rel_pos"], float(np.abs(rel0[1] - rel1[1]).max()))
        # root: in scene coordinates equal to the record transform (position) · rotation difference = Rz(yaw) (z
        # rotation in both tree and scene, the same value)
        e_root = np.abs(tree_to_scene_pts(pb1) - tf_s(tree_to_scene_pts(pb0)))
        o["root_pos"] = max(o["root_pos"], float(e_root.max()))
        o["root_rot"] = max(o["root_rot"], float(np.abs(Rb1 @ np.swapaxes(Rb0, 1, 2) - R[None]).max()))
    d0 = (src_ti.get("aerial_support") or {}).get("discs") or []
    d1 = (var_ti.get("aerial_support") or {}).get("discs") or []
    if len(d0) != len(d1):
        o["disc"] = float("inf")
    for a, b in zip(d0, d1, strict=False):
        exp = tf_s([a["x"], a["y"], 0.0])
        o["disc"] = max(
            o["disc"],
            float(np.hypot(b["x"] - exp[0], b["y"] - exp[1])),
            *(abs(float(a[k]) - float(b[k])) for k in ("r", "z") if k in a),
        )
    o["n_disc"] = len(d0)
    o["ok"] = bool(
        o["hinge_bytes"] and max(o["rel_pos"], o["rel_rot"], o["root_pos"], o["root_rot"], o["disc"]) <= RIGID_TOL
    )
    return o
