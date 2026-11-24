"""Source reading and variant making of the placement augment generation (record.augment.place_gen runs them; its main
is the CLI): Src reads one clip's canonical tree and placement, rel_layers names its layer folders, make_variant
builds one variant tree (rigid move, desk check at the last frame), add_roster_rows adds variants to the shadow
roster. Owns the generation marker MARK. Part of the augment cache fingerprint (AUG_CODE_FILES)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.record.augment import scene as AA
from dexengine.record.augment.scene import rigid_any, rz, scene_to_tree_vec, shadow_dir, tree_to_scene_pts, variant_tf

MARK = "[AUG_GEN]"


DESK_MARGIN = 0.02


# ── source reading ────────────────────────────────────────────────
class Src:
    def __init__(self, main: Path, hand: str, ds: str, emb: str, clip: str, arm: str, aerial: bool = False):
        os.environ.setdefault("DEXENGINE_ARM", arm)
        self.main, self.hand, self.ds, self.emb, self.arm = main, hand, ds, emb, arm
        self.combo = AA.tree_combo(
            arm, hand, ds, emb
        )  # tail = dataset axis variant (aerial scene != tail; arctic is _artic)
        self.spec = P.hand_spec_dir(hand)
        self.tree_name = P.tree_name(clip)
        self.trees_rel, self.wrist_rel, self.ik_rel, self.pl_rel = rel_layers(ds, self.combo, emb)
        self.tree = main / self.trees_rel / self.tree_name
        if not self.tree.is_dir():
            raise InputMissingError(f"{MARK} tree missing {self.tree}")
        self.ti = json.loads((self.tree / "task_info.json").read_text())
        AA.check_tree_combo(self.ti, self.combo, self.tree)
        qpos_layout = P.qpos_layout
        self.kin = dict(np.load(self.tree / "0/trajectory_kinematic.npz", allow_pickle=True))
        self.lay = qpos_layout.read(str(self.tree), nq=self.kin["qpos"].shape[1])
        self.H = int(self.lay.hand_qpos_dim)
        self.artic = AA.artic_spec(self.lay, self.kin, self.ti, self.tree)  # articulated statement (rigid = [])
        self.active = [s for s in ("right", "left") if self.lay.is_active(s) and s in self.lay.hands]
        self.idle = [s for s in ("right", "left") if s in self.lay.hands and s not in self.active]
        self.adopted = json.loads((main / self.pl_rel / f"{self.tree_name}{P.PLACEMENT_SFX}").read_text())
        self.aerial = str(self.ti.get("scene") or "") == "aerial"
        if aerial != self.aerial:
            raise SystemExit(f"{MARK} --aerial={aerial} but tree scene={self.ti.get('scene')!r}: {self.tree}")

    def object_xy_frame0(self) -> np.ndarray:
        q = self.kin["qpos"]
        pts = [q[0, s : s + 3] for s in self.lay.object_starts()]
        return np.mean(np.array(pts), axis=0)

    def object_tracks(self) -> list:
        q = self.kin["qpos"]
        return [q[:, s : s + 3] for s in self.lay.object_starts()]


def desk_box_scene(tree):
    """The tree's desk box (scene coordinates); owner coords.desk_bounds_scene_for (the tree dataset's top · margin
    0)."""
    from dexengine.placement.coords import core as coords

    (x0, x1), (y0, y1) = coords.desk_bounds_scene_for(tree, margin=0.0)
    return x0, x1, y0, y1


def rel_layers(ds: str, combo: str, emb: str):
    """Paths relative to the data root (tree, wrist, arm IK, placement) from the layer path owner paths (no hand
    joining). The shadow root has the same relative paths."""
    m = P.main_root()
    rel = lambda x: str(x.relative_to(m))  # noqa: E731
    sides = ("right", "left")
    return (
        rel(P.trees_dir(combo, ds, emb)),
        {s: rel(P.wrist_dir(combo, s, ds)) for s in sides},
        {s: rel(P.ik_dir(combo, s, ds)) for s in sides},
        rel(P.placement_dir(combo, ds)),
    )


def add_roster_rows(src: Src, shadow: Path, names: list) -> list:
    """Adds variant rows to the list source (overfit/hands/<hand>/ledger/status_final_*.tsv): the original clip row
    copied with only the name changed. The shadow's ledger files become real files (copies). Every ledger file with the
    original row."""
    import fcntl

    rel = f"overfit/hands/{src.hand}/ledger"
    d = shadow_dir(shadow, src.main, rel)
    base = src.tree_name.removesuffix(P.TREE_SFX)
    done = []
    with open(shadow / ".roster.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for f in sorted((src.main / rel).glob("status_final_*.tsv")):
            txt = f.read_text()
            rows = [ln for ln in txt.splitlines() if ln.split("\t", 1)[0] in (base, src.tree_name)]
            if not rows:
                continue
            t = d / f.name
            cur = t.read_text() if (t.exists() and not t.is_symlink()) else txt
            have = {ln.split("\t", 1)[0] for ln in cur.splitlines()}
            add = []
            for n in names:
                vb = n.removesuffix(P.TREE_SFX)
                if vb not in have:
                    add.append(vb + "\t" + rows[0].split("\t", 1)[1])
            if t.is_symlink():
                t.unlink()
            t.write_text(cur.rstrip("\n") + "\n" + "".join(x + "\n" for x in add))
            done.append(f"{f.name}+{len(add)}")
    return done


def _desk_last(src: Src, tf, pivot_t, yaw: float, d_scene) -> tuple[list, list | None]:
    """Desk check at the last frame (frame 0 on the desk was already checked by the judge desk_ok_final) -> (points
    that left, moved support discs | None). On/off = coords.desk_check_on(scene) alone (aerial = off). The support
    discs of aerial clips (aerial_support.discs, already scene coordinates) move with the same transform."""
    from dexengine.placement.coords import core as coords

    bad = []
    if coords.desk_check_on(coords.scene_mode(src.tree)):
        x0, x1, y0, y1 = desk_box_scene(src.tree)

        def inside(q) -> bool:
            return x0 + DESK_MARGIN <= q[0] <= x1 - DESK_MARGIN and y0 + DESK_MARGIN <= q[1] <= y1 - DESK_MARGIN

        for tr in src.object_tracks():
            f = len(tr) - 1
            # the original is outside the desk box too (ends held): not checked
            if not inside(tree_to_scene_pts(tr[f])):
                continue
            ps = tree_to_scene_pts(tf.p(tr[f]))
            if not inside(ps):
                bad.append((f, [round(float(ps[0]), 3), round(float(ps[1]), 3)]))
    discs_new = None
    if src.aerial:
        rz_ = rz(yaw)
        ps_ = tree_to_scene_pts(np.array([pivot_t[0], pivot_t[1], 0.0]))
        c = np.array([ps_[0], ps_[1], 0.0])
        discs_new = []
        for dd in (src.ti.get("aerial_support") or {}).get("discs") or []:
            nxy = rz_ @ (np.array([dd["x"], dd["y"], 0.0]) - c) + c + np.array([d_scene[0], d_scene[1], 0.0])
            discs_new.append(dict(dd, x=float(nxy[0]), y=float(nxy[1])))
    return bad, discs_new


def _write_variant_tree(shadow, src, name, discs_new, rec, yaw, base_name, out, z):
    """Write the variant tree (task_info with aug_place, kinematic and IK-rollout npz, artic files) and its placement
    file."""
    tdir = shadow_dir(shadow, src.main, src.trees_rel) / name
    if tdir.exists():
        shutil.rmtree(tdir)
    (tdir / "0").mkdir(parents=True)
    for fn in ("scene.xml", "scene_eq.xml"):
        if (src.tree / fn).exists():
            (tdir / fn).symlink_to(src.tree / fn)
    for fn in ("F0_REPAIR.json",):
        if (src.tree / fn).exists():
            shutil.copy2(src.tree / fn, tdir / fn)
    ti = dict(src.ti)
    if discs_new is not None:
        ti["aerial_support"] = dict(ti["aerial_support"], discs=discs_new)
    ti["aug_place"] = {
        "src_tree": src.tree_name,
        "dx_scene": rec["dx"],
        "dy_scene": rec["dy"],
        "yaw_deg": yaw,
        "pivot_tree": rec["pivot_tree"],
        "fn": "make_combo_variants.rigid_pl",
        "idle_kept": src.idle,
        "base": base_name,
        "arm": "aug_bake",
    }
    (tdir / "task_info.json").write_text(json.dumps(ti, indent=1, ensure_ascii=False))
    np.savez(tdir / "0/trajectory_kinematic.npz", **out)
    # the loader finds clips by trajectory_ikrollout.npz; retiming rewrites it from qpos_rollout (written ahead, used
    #   only when present)
    np.savez(tdir / "0/trajectory_ikrollout.npz", qpos=out.get("qpos_rollout", out["qpos"]))
    # articulated: hinge bytes · part relative poses · root = support transform, from the arrays before retiming (a
    #   mismatch dies)
    if src.artic:
        o = AA.oracle(
            src.artic,
            z,
            out,
            {"qpos": z.get("qpos_rollout", z["qpos"])},
            {"qpos": out.get("qpos_rollout", out["qpos"])},
            src.ti,
            ti,
        )
        if not o["ok"]:
            raise SystemExit(f"{MARK} articulated oracle failed {name}: {o}")
        rec["artic"] = o
    # placement
    pd = shadow_dir(shadow, src.main, src.pl_rel)
    ad = dict(src.adopted)
    ad["sequence"] = name
    ad["aug_place"] = ti["aug_place"]
    (pd / f"{name}{P.PLACEMENT_SFX}").write_text(json.dumps(ad, indent=1))
    rec.update(status="written")
    return rec


def make_variant(
    src: Src, mcv, shadow: Path, k: int, d_scene, yaw: float, base: dict, base_name: str, desk_check: bool = True
) -> dict:
    """Writes one variant's bake inputs (rigid move of the base the IK saw, placement file) into the shadow root; arm
    cache, wrist, body and retiming are built by augment.bake (bake steps). -> record dict (status written | reject).
    desk_check=False (E5 kinematics; the rule is reach only): the last-frame desk check does not reject, its result
    goes to rec desk_last_ok only. The aerial support disc move (_desk_last's second value) still happens. Default
    True = as before."""
    name = P.tree_name(P.variant_name(src.tree_name.removesuffix(P.TREE_SFX), k))
    pivot_t = src.object_xy_frame0()
    d_t = scene_to_tree_vec(d_scene)
    tf = variant_tf(d_scene, yaw, pivot_t)
    rec = {
        "k": k,
        "name": name,
        "clip": src.tree_name.removesuffix(P.TREE_SFX),
        "dx": float(d_scene[0]),
        "dy": float(d_scene[1]),
        "yaw": float(yaw),
        "r": float(np.hypot(*d_scene)),
        "pivot_tree": [float(pivot_t[0]), float(pivot_t[1])],
    }
    bad, discs_new = _desk_last(src, tf, pivot_t, yaw, d_scene)
    if bad and desk_check:
        rec.update(status="reject", why="desk_last", detail=str(bad[:2]))
        return rec
    if not desk_check:
        rec["desk_last_ok"] = not bad
    z = base
    q_new, v_new, cp_new = rigid_any(
        mcv, src, z["qpos"], z.get("qvel"), z.get("contact_pos"), d_t[0], d_t[1], yaw, pivot_t
    )
    out = dict(z)
    out["qpos"] = q_new.astype(z["qpos"].dtype)
    if "qpos_rollout" in z:
        qr, _, _ = rigid_any(mcv, src, z["qpos_rollout"], None, None, d_t[0], d_t[1], yaw, pivot_t)
        out["qpos_rollout"] = qr.astype(z["qpos_rollout"].dtype)
    if v_new is not None:
        out["qvel"] = v_new.astype(z["qvel"].dtype)
    if cp_new is not None:
        out["contact_pos"] = cp_new.astype(z["contact_pos"].dtype)
    for s in src.idle:  # idle hands = robot rest pose -> original as it is
        a, b = src.lay.hand_slice(s)
        out["qpos"][:, a:b] = z["qpos"][:, a:b]
        if "qpos_rollout" in z:
            out["qpos_rollout"][:, a:b] = z["qpos_rollout"][:, a:b]
        if "qvel" in z:
            out["qvel"][:, a:b] = z["qvel"][:, a:b]
    return _write_variant_tree(shadow, src, name, discs_new, rec, yaw, base_name, out, z)
