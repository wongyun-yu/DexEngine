"""Per-trajectory application of the extra-object rule in the Isaac render session (apply, called by
render.episode.load_traj): the rollout overlap gate and the main-object oracle. record.scene.extras owns the rule,
the CPU decision, the markers and the visual-only spawn. Part of the render fingerprint
(record.cache.RENDER_CODE_FILES)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from dexengine.record.scene import extras_rules as OCC
from dexengine.record.scene.extras import (
    MARK,
    ORACLE_TOL_M,
    ROBOT_SAMPLE_M,
    ROLL_CLEAR_M,
    ROLL_FAR_STRIDE,
    ROLL_MEASURE_M,
    ExtrasError,
    by_source,
    decide,
    marker,
    rule,
    scene_points,
    spawn,
    summary,
)

if TYPE_CHECKING:  # the render session imports this module before AppLauncher: numpy inside functions only ([PRE_APP])
    import numpy as np


# ── Isaac render session ────────────────────────────────────────────────────────
def _prim_path(raw_env, name: str) -> str:
    return raw_env.scene[name].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")


def rollout_overlap(raw_env, binds, write, steps, cands: list[dict], pts: dict, origin: np.ndarray) -> None:
    """Nearest distance between the robot surface samples (stage-1 chosen trajectory, written per render frame step)
    and the object surface samples -> row fields clear_min_m, clear_step and (below the threshold) reason
    rollout_overlap."""
    import numpy as np
    from scipy.spatial import cKDTree

    if not cands:
        return
    from dexengine.record.scene import extras_rules as RP

    robot = RP.robot_local_points(raw_env, binds, ROBOT_SAMPLE_M, _prim_path)
    arts = list({id(a): a for a, _, _ in robot}.values())
    trees = {
        r["name"]: (
            cKDTree(pts[r["name"]] + origin),
            pts[r["name"]].min(0) + origin - ROLL_MEASURE_M,
            pts[r["name"]].max(0) + origin + ROLL_MEASURE_M,
        )
        for r in cands
    }
    best = {r["name"]: (np.inf, -1) for r in cands}
    for t in sorted(set(steps)):
        write(t)
        for a in arts:
            a.update(1e-6)
        W = np.concatenate(
            [
                RP.quat_apply(a.data.body_quat_w[0, bi].detach().cpu().numpy().astype(float), P)
                + a.data.body_pos_w[0, bi].detach().cpu().numpy().astype(float)
                for a, bi, P in robot
            ]
        )
        for nm, (tr, lo, hi) in trees.items():
            sel = W[np.all((lo <= W) & (hi >= W), axis=1)]
            d = float(tr.query(sel if len(sel) else W[::ROLL_FAR_STRIDE])[0].min())
            if d < best[nm][0]:
                best[nm] = (d, int(t))
    for r in cands:
        d, t = best[r["name"]]
        r.update(clear_min_m=round(d, 4) if np.isfinite(d) else None, clear_step=t)
        if d < ROLL_CLEAR_M:
            r["reason"] = "rollout_overlap"
    n_pts = sum(len(P) for _, _, P in robot)
    print(
        f"{MARK} rollout overlap gate · robot surface samples {n_pts} · render steps {len(set(steps))} · "
        f"{[(r['name'], r['clear_min_m']) for r in cands]}",
        flush=True,
    )


def main_oracle(binds, tree_dir, ti: dict, z_off: float, origin: np.ndarray) -> float:
    """Per main object, nearest tree f0 scene position to the scene object slots (state of step 0 written); returns the
    max (m). Beyond the tolerance dies."""
    import numpy as np

    from dexengine.placement.coords import core as C

    slots = [
        b.asset.data.root_pos_w[0].detach().cpu().numpy().astype(float) - origin
        for b in binds
        if b.name.startswith("object")
    ]
    mains = [o for o in ti.get("objects") or [] if o.get("role") == "main" and o.get("qpos_start") is not None]
    if not slots or not mains:
        raise ExtrasError(f"{MARK} oracle: scene object slots {len(slots)} · tree main objects {len(mains)}")
    q = np.load(Path(tree_dir) / "0" / "trajectory_kinematic.npz")["qpos"]
    worst = 0.0
    for o in mains:
        st = int(o["qpos_start"])
        xyz, _ = C.support_scene_pose(q[0, st : st + 7], z_off)
        worst = max(worst, min(float(np.linalg.norm(np.asarray(xyz) - s)) for s in slots))
    if worst > ORACLE_TOL_M:
        raise ExtrasError(
            f"{MARK} oracle: main object tree f0 scene position vs scene slot step 0 max "
            f"{worst * 1000:.1f} mm > {ORACLE_TOL_M * 1000:.0f} mm (the scene transform is wrong)"
        )
    return worst


def _place(cands, write, steps, binds, tree_dir, ti, dec, sh, raw_env, origin, shift_z, cam_fn, per, fps):
    """Oracle of the main objects, overlap gate of the candidates, then spawn of the ones that pass."""
    oracle, placed = None, []
    if cands:
        write(steps[0])
        for b in binds:
            b.asset.update(1e-6)
        oracle = main_oracle(binds, tree_dir, ti, dec["z_off"], sh)
        ds = str(ti["dataset_name"])
        pts = {r["name"]: scene_points(ds, dec["objs"][r["name"]], r["pose_render"], dec["z_off"]) for r in cands}
        rollout_overlap(raw_env, binds, write, steps, cands, pts, sh)
        OCC.gate(write, steps, [r for r in cands if not r["reason"]], pts, origin, shift_z, cam_fn, per, fps)
        put = [r for r in cands if not r["reason"]]
        write(steps[0])
        placed = spawn(put, pts, origin, shift_z) if put else []
        for r, pl in zip(put, placed, strict=True):
            r.update(pl)
    return oracle


def apply(raw_env, binds, write, steps, tree_dir, clip: str, shift_z: float, cam_fn, per: dict, fps: float) -> dict:
    """Render session entry (after the cameras are fitted, before the first frame): decision -> oracle -> rollout
    overlap gate -> occlusion gate (extras_rules.gate) -> spawn -> marker. Returns the meta."""
    tree_dir = Path(tree_dir)
    ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    dec = decide(tree_dir, ti)
    rows = dec["rows"]
    origin = raw_env.scene.env_origins[0].detach().cpu().numpy().astype(float)
    sh = origin + (0.0, 0.0, shift_z)  # world shift
    cands = [r for r in rows if not r["reason"]]
    oracle = _place(cands, write, steps, binds, tree_dir, ti, dec, sh, raw_env, origin, shift_z, cam_fn, per, fps)
    put, drop, by = summary(rows)
    print(
        marker(clip, rows) + f" · scene {dec['scene']} · main object oracle "
        f"{'-' if oracle is None else f'{oracle * 1000:.1f}mm'} · no cover, training USD "
        f"{sum(1 for r in put if r.get('train_usd_fallback'))}",
        flush=True,
    )
    OCC.overhang_say(clip, rows)
    for r in put:
        print(
            f"{MARK}   put {r['name']} {r['obj_id']} xyz={[round(v, 4) for v in r['scene_xyz']]} gap="
            f"{r['f0_gap_m']} lowered {r.get('lowered_dz_m')} hand {r.get('hand_min_m')} rollout "
            f"{r.get('clear_min_m')} "
            f"usd={r['usd']}",
            flush=True,
        )
    keep = (
        "name",
        "obj_id",
        "source",
        "rid",
        "reason",
        "move_m",
        "rot_deg",
        "f0_cx",
        "f0_cy",
        "f0_gap_m",
        "hand_min_m",
        "float_action",
        "lowered_dz_m",
        "clear_min_m",
        "clear_step",
        "occ_frames",
        OCC.OVERHANG_FIELD,
    )
    return {
        "rule": rule(),
        "scene": dec["scene"],
        "n_extra": len(rows),
        "main_oracle_m": oracle,
        "by_reason": by,
        "by_source": by_source(rows),
        "render_only": dec["render_only"],
        "placed": [
            {
                k: r.get(k)
                for k in (
                    *keep,
                    "scene_xyz",
                    "scene_wxyz",
                    "usd",
                    "prim",
                    "train_usd_fallback",
                    "spawn_err_m",
                    "bbox_world",
                )
            }
            for r in put
        ],
        "dropped": [{k: r.get(k) for k in keep if k in r} for r in drop],
    }
