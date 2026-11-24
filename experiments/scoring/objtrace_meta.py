"""Per-clip and global metadata of the object-pose trace (split from objtrace_recorder.py): the clip's frames, the
object assets and the trajectory-meta fields the trace copies."""

from __future__ import annotations

import numpy as np

# inverse-chain values the running env does not hold — only the tree files do (not made up here; the scorer reads them
# from <data_dir>/<clip>/)
MISSING_FOR_INVERSE_CHAIN = {
    "retime_src_idx": "<tree>/0/trajectory_kinematic.npz key retime_src_idx [T] float — tree row → row before retiming "
    "(older clips: restored afterwards, residual retime_map_resid)",
    "f0_prepad": "<tree>/task_info.json f0_prepad (else F0_REPAIR.json n_blend_effective · else 0) — "
    "placement coords.tree_f0_pad · the front pad frames hold the first pose",
    "placement_xy_shift_yaw_pivot": "<tree>/task_info.json xy_shift_traj · yaw_deg · pivot_traj — coords.tree_placement"
    " · already baked into the tree qpos (q = (p − pivot)·Rz(yaw)ᵀ + pivot + (dx,dy))",
    "base_frame_yaw_deg": "<tree>/task_info.json frame.yaw_deg — base = "
    "Rz(yaw)·capture(dexengine/data/base_adapter.py)",
    "desk_z": "<tree>/task_info.json support.desk_z (qpos_layout.desk_z_of) — folded into clip_dz (dz = the clip's "
    "world_z_offset − the constant world_z_offset)",
    "ref_dt": "<tree>/task_info.json ref_dt (coords.tree_dt) — tree frame spacing (0.02 s = step_dt)",
    "source_window": "MANO window start (mano_frame.window_start = segment.start + frame_map.offset) · "
    "original TACO frame = origin + round(m·30/50)"
    "(scripts/collect/collect_headsrc.mano_rows · raw_frames)",
    "capture_z_shift": "taco_frames.hand_shift_z(seq) — capture world → scene z constant (per-session exceptions)",
}


def _np(x) -> np.ndarray:
    return x.detach().to("cpu").numpy() if hasattr(x, "detach") else np.asarray(x)


def clip_meta(raw_env) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    """Per-clip inverse-chain values the running loader holds → (arrays [K] · source · not found). Nothing made
    up.
    """
    ld = raw_env.trajectory_manager._loader
    src = {
        "length": ("clip_lengths", "tree frames (the reference frame stops at length−1)"),
        "dz": ("_clip_dz", "z shift m (the loader adds it to the object and wrist z)"),
        "scene": ("_clip_scene", "scene kind (table · aerial)"),
        "dataset": ("clip_datasets", "tree dataset (task_info.dataset_name)"),
        "n_extras": ("clip_n_extras", "number of extra objects"),
    }
    out, doc, missing = {}, {}, dict(MISSING_FOR_INVERSE_CHAIN)
    for k, (attr, what) in src.items():
        v = getattr(ld, attr, None)
        if v is None:
            missing[k] = f"trajectory_manager._loader.{attr} absent (this run)"
            continue
        a = _np(v)
        out[k] = a.astype(str) if a.dtype.kind in "OUS" else a
        doc[k] = f"_loader.{attr} — {what}"
    return out, doc, missing


def global_meta(raw_env) -> dict:
    """Global inverse-chain values the running trajectory_manager and loader hold."""
    tm = raw_env.trajectory_manager
    ld = tm._loader
    g = {
        "data_dir": getattr(ld, "data_dir", None),
        "tree_dir_rule": "<data_dir>/<clip_name> (loader glob <data_dir>/*/0/trajectory_*.npz)",
        "slowdown_factor": float(getattr(tm, "_slowdown_factor", 1.0)),
        "per_side_clock": bool(getattr(tm, "per_side_clock", False)),
        "ref_frame_rule": "clamp(floor(k / slowdown_factor) + start_frame, 0, length − 1) — measured = ref_frame",
        "world_z_offset": getattr(ld, "_world_z_offset", None),
        "desk_height": getattr(tm, "_desk_height", None),
        "rotate_to_x_forward": getattr(tm, "_rotate", None),
        "ref_space_rule": "pos = Rz(−90°)·(tree + [0,0,clip_dz]) + [0,0,desk_height] · quat = rz_quat ⊗ tree"
        "(rotate_to_x_forward · trajectory_manager._rotate_pos_batch · _rotate_quat_batch)",
    }
    for k, attr in (("rz_quat_wxyz", "_rz_quat"), ("rz_mat", "_rz_mat"), ("z_offset", "_z_offset")):
        v = getattr(tm, attr, None)
        if v is not None:
            g[k] = _np(v).astype(float).tolist()
    # the judge's object-tracking termination off (an explicit measurement arm, owner judge.terms): the record says so
    # (on = standard = no field)
    from dexengine.rl.judge import terms as judge_terms

    if judge_terms.objtrack_term()[0] == 0:
        g["judge_objtrack_term"] = "off"
    return g


def object_assets(rec) -> tuple[dict, list[str]]:
    """Each side's object spawn USDs (scene asset cfg) → ({side: entry}, sides not read). Several ∧ random_choice
    False:
    env e's object = usds[e % n] (env setup build_object_spawn_cfg — the loader's clip order)."""
    out, bad = {}, []
    for s in rec.sides:
        a = rec.scene.rigid_objects.get(s["entity"]) or rec.scene.articulations.get(s["entity"])
        sp = getattr(getattr(a, "cfg", None), "spawn", None)
        if hasattr(sp, "assets_cfg"):
            usds, rc = [str(c.usd_path) for c in sp.assets_cfg], bool(getattr(sp, "random_choice", True))
        elif getattr(sp, "usd_path", None):
            usds, rc = [str(sp.usd_path)], False
        else:
            bad.append(s["side"])
            continue
        out[s["side"]] = {
            "entity": s["entity"],
            "usds": usds,
            "random_choice": rc,
            "rule": "env e → usds[e % n]" if not rc else "random (env → usd unknown)",
        }
    return out, bad


#: fields copied from the trajectory file's meta as they are (owner: the recorder's finalize)
FROM_TRAJ_META = (
    "tree",
    "num_envs",
    "step_dt",
    "physics_dt",
    "decimation",
    "frame",
    "time",
    "warmup_batches",
    "sides",
    "judge",
    "env",
)
