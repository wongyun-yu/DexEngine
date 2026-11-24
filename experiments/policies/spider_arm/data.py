#!/usr/bin/env python3
"""E5 third data source — SPIDER (arXiv 2511.09484, floating-hand physics retargeter) + an arm attached afterwards by
SPIDER's own rule (paper Sec. A.6). Built exactly like the dense kinematic-replay rows
(experiments.policies.kin_data.prep --our-trees --ref-arm
--ref-var-root) except for three swaps; this module owns the swaps, experiments.policies.kin_data.prep calls it behind
--spider-root/--spider-arm.

  (a) hand + wrist joints = SPIDER simulated qpos (`trajectory_mjwp.npz`, W windows × S sim steps at SIM_DT) resampled
  to the
      reference's 50 Hz frames: frame 0 = SPIDER's initial state = its input reference frame 0 (run_mjwp sets
      mj_data.qpos = qpos_ref[0]); frame k ≥ 1 = flat sim index i with time[i] == k / 50 (time[i] = (i + 1)·SIM_DT, so
      i = 2k − 1) — asserted on the time array, not assumed.
  (b) object poses = SPIDER simulated object qpos (same columns), through the SAME two rigid transforms the kin variant
  got:
      bake placement `dexengine.placement.retime_variants.rigid_pl(orig adopted dx dy yaw pivot)` then the variant
      `dexengine.record.augment.scene.rigid_any(scene_to_tree_vec(aug_place.dx/dy_scene), aug_place.yaw_deg,
      aug_place.pivot_tree)`. NOT applied (they are our pipeline's repairs, not part of a SPIDER baseline): xr cross
      repair, retiming. KEPT as kin: the D16 idle side (task_info idle_sides) — the render env has no robot side and no
      objects there, so SPIDER drives exactly the tree's ACTIVE sides; an idle side (a hand SPIDER did not simulate, or
      a D16 idle hand of a
      bimanual clip that SPIDER did simulate) keeps the kin tree's constant idle block and idle arm.
  (c) arm = A.6 rollout DLS IK (constants block A6 below): per placement the palm trajectory is re-expressed in the
  robot
      frame (same target rule as experiments.policies.kin_data.prep_tree.kin_ik), the first waypoint starts from the
      NEAREST healthy-table configuration
      (400 iterations), every later one from the previous solution (50 iterations); ACCEPTED iff every frame tracks
      within
      3 cm / 0.15 rad and the proximal joints stay within 0.5 rad (L∞) of the healthy table. Kept either way
      (first-author
      decision: every variant stays in the training data — the flag is recorded, not used as a filter).
  (d) labels: fingers + wrist = SPIDER ctrl (position actuators — mapped actuator → joint column through SPIDER's own
      scene.xml) at the same flat index as the state of that row; arm = DLS IK of the commanded palm, each frame
      warm-started
      from the same frame's achieved-arm solution (50 iterations) so label and state stay on one IK branch. Written to
      `<tree>/0/spider_cmd.npz`; experiments.policies.kin_data.record_kin.record_kin puts it into joint_target when
      KIN_CMD_TARGETS=spider_cmd.

Markers: [SPIDER_SRC] [SPIDER_XFORM] [SPIDER_A6] [SPIDER_CMD] [SPIDER_LIMITS] — one line per tree each.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.policies.spider_arm.arm_targets import (
    A6,
    a6_rollout,
    arm_ctx,
    cmd_ik,
    targets,
    write_arm_cache,
)
from experiments.policies.spider_arm.data_limits import limits_main
from experiments.policies.spider_arm.data_tree import write_spider_tree
from experiments.policies.spider_arm.spider_load import CMD_FILE, MARK, load_spider


# ── (b) layout + the two rigid
# transforms
# ─────────────────────────────────────────────────────────────────────────────
def _layout(tree: Path, nq: int):
    from dexengine.core.paths import qpos_layout as QL  # noqa: PLC0415

    return QL.read(str(tree), nq=nq, use_cache=False)


def base_to_tree(qb: np.ndarray, lay_b, lay_t, fill: np.ndarray) -> tuple[np.ndarray, list]:
    """Base-layout columns → tree-layout columns (hand blocks by side, objects in order, articulation columns in
    order).
    Tree columns with no base source (a hand the base has not) come from `fill` (T_fill × nq_t, must be constant in
    time)."""
    H = int(lay_t.hand_qpos_dim)
    if int(lay_b.hand_qpos_dim) != H:
        raise SystemExit(f"{MARK} hand width base {lay_b.hand_qpos_dim} ≠ tree {H}")
    m = []
    for s, o in lay_b.hands.items():
        if s not in lay_t.hands:
            raise SystemExit(f"{MARK} base hand {s} is not in the tree")
        m += [(o + j, lay_t.hands[s] + j) for j in range(H)]
    ob, ot = lay_b.object_starts(), lay_t.object_starts()
    if len(ob) != len(ot):
        raise SystemExit(f"{MARK} object count base {len(ob)} ≠ tree {len(ot)}")
    m += [(b + j, c + j) for b, c in zip(ob, ot, strict=True) for j in range(7)]
    ab, at = lay_b.artic_cols(), lay_t.artic_cols()
    if len(ab) != len(at):
        raise SystemExit(f"{MARK} joint columns base {len(ab)} ≠ tree {len(at)}")
    m += list(zip(ab, at, strict=True))
    src = [a for a, _ in m]
    dst = [b for _, b in m]
    if sorted(src) != list(range(qb.shape[1])) or len(set(dst)) != len(dst):
        raise SystemExit(
            f"{MARK} base→tree column map is not a bijection (base {qb.shape[1]} columns · map {len(src)})"
        )
    miss = [c for c in range(int(lay_t.nq)) if c not in set(dst)]
    T = len(qb)
    out = np.empty((T, int(lay_t.nq)))
    out[:, dst] = qb[:, src]
    if miss:
        f = np.asarray(fill, np.float64)[:, miss]
        if np.abs(f - f[0]).max() > 0:
            raise SystemExit(
                f"{MARK} columns to fill {miss[:3]}… are not constant in the kin tree (must be the idle-hand block)"
            )
        out[:, miss] = f[0][None, :]
    return out, miss


def place_and_vary(
    q_t: np.ndarray, orig_tree: Path, base_dir: Path, lay_b, adopted: dict, aug: dict, lay_t, keep_cols: list
) -> np.ndarray:
    """bake placement (rigid_pl · the orig tree's adopted dx dy yaw pivot) → variant (rigid_any · aug_place).
    Mirrors the bake's arguments (dexengine.placement.bake.combo._placed_arrays): two-hand base → maps + layout of the
    base clip; one-hand base (the bake injects the missing block) → maps of the orig tree scene (it carries the injected
    hand), layout None. keep_cols = tree columns that are NOT SPIDER's (kin idle block) — restored after the
    transforms."""
    from dexengine.placement import retime_variants as mcv  # noqa: PLC0415
    from dexengine.record.augment import scene as AS  # noqa: PLC0415

    H = int(lay_t.hand_qpos_dim)
    if sorted(lay_t.hands) != ["left", "right"]:
        raise SystemExit(
            f"{MARK} tree {orig_tree.name} hand blocks {sorted(lay_t.hands)} — not the baked form (both blocks)"
        )
    if len(lay_b.hands) == 2:
        if int(lay_b.nq) != int(lay_t.nq):
            raise SystemExit(f"{MARK} two-hand base nq {lay_b.nq} ≠ tree {lay_t.nq}")
        maps, lay_pl = mcv.maps_for(base_dir, H), lay_b
    else:
        maps, lay_pl = mcv.maps_for(orig_tree, H), None
    q1, _, _ = mcv.rigid_pl(
        q_t,
        None,
        None,
        H,
        float(adopted["dx"]),
        float(adopted["dy"]),
        float(adopted["yaw"]),
        adopted["pivot"],
        maps,
        layout=lay_pl,
    )
    d_t = AS.scene_to_tree_vec([float(aug["dx_scene"]), float(aug["dy_scene"])])
    # = dexengine.record.augment.place_gen.Src
    shim = SimpleNamespace(lay=_layout(orig_tree, int(lay_t.nq)), H=H, tree=Path(orig_tree))
    q2, _, _ = AS.rigid_any(
        mcv,
        shim,
        q1,
        None,
        None,
        float(d_t[0]),
        float(d_t[1]),
        float(aug["yaw_deg"]),
        [float(x) for x in aug["pivot_tree"]],
    )
    if keep_cols:
        q2[:, keep_cols] = q_t[:, keep_cols]
    return q2


# ── palm FK (commanded wrist) — validated against the official body → wrist stages on the state
# trajectory ──────────
def palm_poses(tree: Path, q: np.ndarray, side: str, hand: str) -> tuple[np.ndarray, np.ndarray]:
    import mujoco  # noqa: PLC0415

    from dexengine.placement import arm_ik as AIK  # noqa: PLC0415

    spec = AIK._load_spec_module("hand_specs").HAND_REGISTRY[hand]
    body = spec.palm_fk_body_right.replace("right_", f"{side}_", 1)
    m = mujoco.MjModel.from_xml_path(str(Path(tree) / "scene.xml"))
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise SystemExit(f"{MARK} palm body {body} missing ({tree})")
    P = np.zeros((len(q), 3))
    Qw = np.zeros((len(q), 4))
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        P[t], Qw[t] = d.xpos[bid], d.xquat[bid]
    Qw *= np.where(Qw[:, :1] < 0, -1.0, 1.0)
    return P, Qw


# ── one variant
# tree
# ──────────────────────────────────────────────────────────────────────────────────────────────────


def _load_variant(a, combo: str, ds: str, emb: str, clip: str, name: str, troot: Path):
    """SPIDER's rows of the clip, our variant tree `name`, its source tree and the adopted placement (checked)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    main = Path(a.main)
    base_dir = main / f"data/{ds}/base/wuji2/{emb}/{clip}"
    lmf = Path(a.spider_root).parent / "layout_meta.json"  # owner of changed/identity per clip (input step)
    if not lmf.is_file():
        raise SystemExit(f"{MARK} {lmf} missing — no SPIDER column map statement")
    sp = load_spider(Path(a.spider_root) / ds / "wuji2" / emb / clip, base_dir, json.loads(lmf.read_text()))
    tdir = troot / name
    orig = troot / P.tree_name(clip)
    ti = json.loads((tdir / "task_info.json").read_text())
    aug = ti.get("aug_place") or {}
    if aug.get("src_tree") != orig.name:
        raise SystemExit(f"{MARK} {name}: aug_place.src_tree {aug.get('src_tree')} ≠ {orig.name}")
    adopted = json.loads((P.placement_dir(combo, ds) / f"{orig.name}{P.PLACEMENT_SFX}").read_text())
    ti0 = json.loads((orig / "task_info.json").read_text())
    for k_ad, k_ti in (("dx", 0), ("dy", 1)):
        if abs(float(adopted[k_ad]) - float(ti0["xy_shift_traj"][k_ti])) > 1e-9:
            raise SystemExit(f"{MARK} {orig.name}: adopted {k_ad} ≠ task_info xy_shift_traj")
    zk = dict(np.load(tdir / "0/trajectory_kinematic.npz", allow_pickle=True))
    return base_dir, sp, tdir, orig, ti, aug, adopted, ti0, zk


def _spider_on_tree(name: str, sp: dict, base_dir: Path, tdir: Path, zk: dict, ti: dict, ti0: dict):
    """SPIDER state and command on the tree's qpos layout; a D16 idle hand keeps the kin tree's constant block."""
    lay_t = _layout(tdir, zk["qpos"].shape[1])
    lay_b = _layout(base_dir, sp["q"].shape[1])
    q_t, miss = base_to_tree(sp["q"], lay_b, lay_t, zk["qpos"])
    c_t, _ = base_to_tree(sp["cmd"], lay_b, lay_t, zk["qpos"])
    act = list(ti.get("active_sides") or [])
    if not act or any(s not in lay_b.hands for s in act) or act != list(ti0.get("active_sides") or []):
        raise SystemExit(
            f"{MARK} {name}: active hands {act} — SPIDER hands {sorted(lay_b.hands)} · source {ti0.get('active_sides')}"
        )
    sp_sides = [s for s in ("right", "left") if s in act]  # SPIDER drives the active sides (the env has only these)
    other = [s for s in ("right", "left") if s not in sp_sides]  # kin idle block + idle arm (constant)
    H_ = int(lay_t.hand_qpos_dim)
    d16 = [s for s in other if s in lay_b.hands]  # SPIDER simulated it, the kin env keeps it idle (D16)
    if d16:
        cols = [lay_t.hands[s] + j for s in d16 for j in range(H_)]
        f = np.asarray(zk["qpos"], np.float64)[:, cols]
        if np.abs(f - f[0]).max() > 0:
            raise SystemExit(f"{MARK} {name}: the kin tree's D16 idle hand {d16} block is not constant")
        q_t[:, cols] = f[0][None, :]
        c_t[:, cols] = f[0][None, :]
        miss = sorted(set(miss) | set(cols))
    return lay_t, lay_b, q_t, c_t, miss, sp_sides, other, d16


def _a6_side(combo, ds, name, s, tdir, rdir, qv, cv, zoff, robot_z, hand, arm, T):
    """One SPIDER side: A.6 on the state's palm (oracle: palm FK == the canonical wrist step), its arm cache, and the
    command IK. -> (A.6 rollout, command IK, res entry, ik entry)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    c = arm_ctx(s, hand, arm)
    wz = np.load(rdir / f"wrist_poses_{s}" / f"{name}.npz")
    P0, Q0 = palm_poses(tdir, qv, s, hand)  # oracle: my palm FK == the official stages on the state
    dp = np.abs(P0.astype(np.float32) - wz["positions"]).max()
    dq_ = np.abs(np.abs((Q0.astype(np.float32) * wz["quaternions"]).sum(1)) - 1).max()
    if dp > 1e-5 or dq_ > 1e-5:
        raise SystemExit(
            f"{MARK} {name}/{s}: palm FK ≠ canonical wrist step (position {dp:.2e} · quaternion {dq_:.2e})"
        )
    TP, TR = targets(c, wz["positions"].astype(np.float64), wz["quaternions"].astype(np.float64), zoff, robot_z)
    r = a6_rollout(c, TP, TR)
    write_arm_cache(P.ik_dir(combo, s, ds) / f"{name}.npz", c, r, robot_z, zoff)
    Pc, Qc = palm_poses(tdir, cv, s, hand)
    TPc, TRc = targets(c, Pc, Qc, zoff, robot_z)
    rc_ = cmd_ik(c, TPc, TRc, r["q"])
    res_s = {
        "accepted": r["accepted"],
        "pos_max_mm": round(float(r["pe"].max() * 1e3), 1),
        "rot_max_rad": round(float(r["re"].max()), 4),
        "prox_dev_max_rad": round(float(r["dev"].max()), 3),
        "table_index": r["i0"],
        "pos_mean_mm": round(float(r["pe"].mean() * 1e3), 1),
        "cmd_pos_max_mm": round(float(rc_["pe"].max() * 1e3), 1),
        "cmd_minus_state_max_rad": round(float(np.abs(rc_["q"] - r["q"]).max()), 3),
    }
    ik_s = {
        "side": s,
        "T": T,
        "pos_mm_mean": res_s["pos_mean_mm"],
        "pos_mm_max": res_s["pos_max_mm"],
        "rot_deg_mean": float(np.degrees(r["re"].mean())),
        "rot_deg_max": float(np.degrees(r["re"].max())),
        "fallback_frames": 0,
        "a6": res_s,
    }
    print(
        f"{MARK} [SPIDER_A6] {name} {s} accepted {r['accepted']} · position max {res_s['pos_max_mm']} "
        "mm · rotation max "
        f"{res_s['rot_max_rad']} rad · proximal {res_s['prox_dev_max_rad']} rad · table {r['i0']} · "
        "command IK max "
        f"{res_s['cmd_pos_max_mm']} mm",
        flush=True,
    )
    return r, rc_, res_s, ik_s


def _idle_arm_on_spider_time(combo, ds, name, s, T) -> np.ndarray:
    """Arm of a hand SPIDER did not simulate = kin's idle arm (constant) on the SPIDER time axis (cache rewritten)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    src = P.ik_dir(combo, s, ds) / f"{name}.npz"
    zz = dict(np.load(src, allow_pickle=True))
    qa = np.asarray(zz["arm_joint_positions_all"])
    if np.abs(qa - qa[0]).max() > 0:
        raise SystemExit(f"{MARK} {name}/{s}: kin arm cache is not constant — not an idle arm")
    for k in ("arm_joint_positions_all",):
        zz[k] = np.repeat(qa[:1], T, axis=0)
    for k in ("wrist_pos_error_all", "wrist_rot_error_deg_all"):
        if k in zz:
            zz[k] = np.repeat(np.asarray(zz[k])[:1], T, axis=0)
    zz["num_frames"] = np.int32(T)
    tmp = src.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(tmp, **zz)
    os.replace(tmp, src)
    return zz["arm_joint_positions_all"].astype(np.float64)


def build_variant(
    a, combo: str, ds: str, emb: str, clip: str, name: str, troot: Path, rdir: Path, robot_z: float
) -> tuple[dict, dict]:
    """After experiments.policies.kin_data.prep copied our variant tree `name` (kin data): rewrite it with SPIDER
    (a)(b), run the body → wrist stages, solve A.6 (c), write the commanded file (d) and the two flags. → (ik summary,
    reach-like dict)."""
    from dexengine.placement.bake import chain as SR  # noqa: PLC0415
    from dexengine.placement.coords import core as CO  # noqa: PLC0415

    hand = "wuji2"
    arm = "rby1"
    base_dir, sp, tdir, orig, ti, aug, adopted, ti0, zk = _load_variant(a, combo, ds, emb, clip, name, troot)
    lay_t, lay_b, q_t, c_t, miss, sp_sides, other, d16 = _spider_on_tree(name, sp, base_dir, tdir, zk, ti, ti0)
    qv = place_and_vary(q_t, orig, base_dir, lay_b, adopted, aug, lay_t, miss)
    cv = place_and_vary(c_t, orig, base_dir, lay_b, adopted, aug, lay_t, miss)
    T = len(qv)
    print(
        f"{MARK} [SPIDER_XFORM] {name} T {T}(kin {len(zk['qpos'])}) · SPIDER hands {sp_sides} · kin idle block {other}"
        f"(D16 {d16}) · "
        f"placement dx {adopted['dx']} dy {adopted['dy']} yaw {adopted['yaw']} · variant "
        f"{aug['dx_scene']:.4f} {aug['dy_scene']:.4f} "
        f"{aug['yaw_deg']} · no xr, no retime · D16 idle hand kept from kin",
        flush=True,
    )
    write_spider_tree(name, tdir, orig, ti, zk, qv, lay_t, T, sp, sp_sides, other, d16)
    # body → wrist stages on the SPIDER tree (as finish_tree's non-ref path)
    for spc in [f"body:{tdir},-q"] + [
        f"wrist:--data_dir,{troot},--output_dir,{rdir / f'wrist_poses_{s}'},--side,{s},--sequence,{name}"
        for s in ("right", "left")
    ]:
        rc, _ex = SR.run_one(spc)
        if rc:
            raise RuntimeError(f"step failed rc={rc}: {spc.split(':', 1)[0]}")
    zoff = float(CO.scene_z_offset(tdir))
    ik, res, qs, pe_mm, cmd_arm = {}, {}, {}, {}, {}
    for s in sp_sides:
        r, rc_, res[s], ik[s] = _a6_side(combo, ds, name, s, tdir, rdir, qv, cv, zoff, robot_z, hand, arm, T)
        qs[s], pe_mm[s], cmd_arm[s] = r["q"], r["pe"] * 1e3, rc_["q"]
    for s in other:  # arm of a hand SPIDER did not simulate = kin's idle arm (constant) on the SPIDER time axis
        qs[s] = _idle_arm_on_spider_time(combo, ds, name, s, T)
        cmd_arm[s] = qs[s]
    # (d) commanded file
    cmdf = tdir / "0" / CMD_FILE
    np.savez(
        cmdf,
        hand_cmd=cv,
        sides=np.array(sp_sides),
        **{f"arm_cmd_{s}": cmd_arm[s].astype(np.float32) for s in ("right", "left")},
        **{f"arm_joint_names_{s}": np.array(arm_ctx(s, hand, arm)["joints"]) for s in ("right", "left")},
        rule=np.array(
            "row k: fingers+wrist = SPIDER ctrl at the state's sim index; arm = DLS IK of the commanded palm "
            "warm-started from row k's achieved arm (50 it); row 0 = state"
        ),
    )
    print(f"{MARK} [SPIDER_CMD] {name} command file {cmdf.name} T {T} · hands {sp_sides}", flush=True)
    # our arm-limit flag is NOT computed here (final_guard is minutes per tree) —
    # `experiments.policies.spider_arm.data_limits` runs it as a separate
    #   CPU pass over finished roots (render does not need it); the result goes to a jsonl next to the batch, not the
    # tree.
    a6_all = all(res[s]["accepted"] for s in sp_sides)
    ti = json.loads((tdir / "task_info.json").read_text())
    ti["e5_spider"].update(a6=res, a6_accepted=a6_all, a6_constants=A6)
    (tdir / "task_info.json").write_text(json.dumps(ti, indent=1, ensure_ascii=False))
    reach = {
        "rule": "spider_a6 — accepted recorded, not a filter",
        "per_side": {s: res[s]["accepted"] for s in sp_sides},
        "reach": 1.0 if a6_all else 0.0,
        "passes": True,
        "frames": dict.fromkeys(sp_sides, T),
        "reach_min": None,
        "solver": "spider_a6_dls",
        "n_jump": dict.fromkeys(sp_sides),
        "max_dq_ratio": dict.fromkeys(sp_sides),
        "spider": {"a6_accepted": a6_all, "a6": res},
    }
    return ik, reach


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "limits":
        sys.exit(limits_main(sys.argv[2:]))
    raise SystemExit(f"{MARK} usage: python -m experiments.policies.spider_arm.data limits --trees … --out …")
