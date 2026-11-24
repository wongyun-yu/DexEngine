"""TACO sequence loading for the MANO videos: alignment gates and the per-frame hand and object meshes."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# isort: split
# names and suffixes are owned by dexengine/core/paths.py (no literals)
from dexengine.core.desk import DESK_TOP_SURFACE_Z as DESK_HEIGHT  # owner of the desk top height
from dexengine.core.paths import (
    IK_SIDECAR_SFX,
    raw_from_clip_id,
    taco_source_root,  # noqa: E402  (owner of the TACO source tree place)
)
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402  (owner of the data root, env DEXENGINE_MAIN_ROOT)
)
from dexengine.record.viz.mano_gate import taco_alignment_gate
from dexengine.record.viz.mano_model import RZ_MINUS_90, mano_fk_batch


def _assemble_taco_result(
    _ti_all, z_anchor, sequence, rh_verts, lh_verts, rh_obj_T, lh_obj_T, _tree, T_obj, _yaw, _dxy, _piv
):
    info = _ti_all

    def resolve_mesh(d_str):
        if not d_str:
            return None
        mid = d_str.split("/")[-1]
        p = _data_root() / "data" / "taco" / "meshes" / mid / "visual.obj"
        return str(p) if p.exists() else None

    r_mesh = resolve_mesh(info.get("right_object_mesh_dir"))
    l_mesh = resolve_mesh(info.get("left_object_mesh_dir"))

    # the z_anchor="contact" heuristic is gone (it applied corrections of 0.4-5 cm, larger than the measured desk
    # height spread of ±3 mm, and distorted the source). The single constant above is the exact path; the z_anchor
    # argument is accepted for compatibility, ignored, and warned about.
    if z_anchor not in ("", "legacy", "exact"):
        print(f"  [render_mano] z_anchor={z_anchor!r} is retired: the single-constant path is used", flush=True)

    fps = round(1.0 / float(info.get("ref_dt", 0.02)))

    # alignment is checked in **one place, the canonical loader**; consumers (annotations, trees, judge, render) do not
    # check on their own, so what this misses nobody catches.
    taco_alignment_gate(sequence, rh_verts, lh_verts, rh_obj_T, lh_obj_T, r_mesh, l_mesh, tree=_tree)

    return {
        "right_verts": rh_verts,
        "left_verts": lh_verts,
        "right_obj_T": rh_obj_T,
        "left_obj_T": lh_obj_T,
        "right_obj_mesh_path": r_mesh,
        "left_obj_mesh_path": l_mesh,
        "fps": fps,
        "n_frames": T_obj,
        # placement yaw (deg) applied to hands and objects; the raw frame camera (--cam_frame raw / --cam_preset
        # retarget) gets the same rotation.
        "placement_yaw_deg": _yaw,
        "placement_xy_shift": [float(_dxy[0]), float(_dxy[1])],
        "placement_pivot": [float(_piv[0]), float(_piv[1])],
        # the tree this render read: retarget must be drawn from the same tree's scene (scene.xml) and trajectory for
        # frames and placement to match (dexengine/record/viz/retarget_render.py).
        "tree_dir": str(_tree),
        "sequence": sequence,
    }


def _load_hand_pkl(hp_dir: Path, side: str):
    """Raw TACO MANO of one hand -> (pose [T_raw, 48], trans [T_raw, 3], betas)."""
    with open(hp_dir / f"{side}_hand.pkl", "rb") as f:
        d = pickle.load(f)
    with open(hp_dir / f"{side}_hand_shape.pkl", "rb") as f:
        s = pickle.load(f)
    keys = sorted(d.keys())
    poses = torch.stack([d[k]["hand_pose"] for k in keys])  # (T_raw, 48)
    trans = torch.stack([d[k]["hand_trans"] for k in keys])  # (T_raw, 3)
    return poses, trans, s["hand_shape"]


def _tree_inputs(sequence: str, tree_dir):
    """The tree the render reads (tree_dir, else the TACO source tree) -> (tree, kinematic npz, qpos, T_obj,
    task_info, object column base 2*hand_qpos_dim, f0 pad P, motion frames T_obj - P)."""
    # tree_dir: a combo tree can be rendered directly, with objects, placement and time axis taken from that tree (for
    # SBS with absolute positions matching the robot video). Default = the TACO source tree (taco_source_root).
    _tree = Path(tree_dir) if tree_dir else (taco_source_root() / sequence)
    kin = np.load(_tree / "0" / "trajectory_kinematic.npz")
    qpos = kin["qpos"]
    T_obj = qpos.shape[0]
    _ti_all = json.loads((_tree / "task_info.json").read_text())
    # combo trees have a different hand stride: the object slot column is 2*hand_qpos_dim (legacy 18 -> 36).
    _hqd = int(_ti_all.get("hand_qpos_dim", 18))
    _HH2 = 2 * _hqd
    # f0 pad: v2 trees start with a P-frame hold pad. The raw MANO covers only the motion span, so it is resampled to
    # (T_obj - P) and frame 0 is repeated P times in front (the same convention as the overlay).
    _f0p = _tree / "F0_REPAIR.json"
    _pad = 0
    if _f0p.exists():  # a marker that exists but cannot be read dies (never P = 0)
        _pad = int(json.loads(_f0p.read_text()).get("n_blend_effective", 0) or 0)
    _T_motion = max(1, T_obj - _pad)
    return _tree, kin, qpos, T_obj, _ti_all, _HH2, _pad, _T_motion


def _hand_shift_z(sequence: str, base_seq: str) -> float:
    """The raw-hand z shift of this clip's capture session (taco_frames)."""
    # the z shift is the **single canonical constant** taco_frames.HAND_SHIFT_Z. Official TACO ships hands and
    # objects in one world frame (its visualiser composes them directly); only our object copies are in the desk
    # frame. The shift is the same constant as the data chain (dexengine.data.readers.taco_lift), checked by the desk
    # contact anchor to ±3.3 mm over all 15 sessions. A per-clip inversion (object rest height minus scene
    # object z0) differs per object and floated hands 2-4 cm above objects. The constant is **per session** (capture
    # date): 23 of 24 sessions are 0.5476 ±19 mm, but 20231002 has another world origin at -0.185 (725 mm apart).
    # Only the canonical resolver is called; no copy of the convention.
    from dexengine.data.readers import taco_frames as _TF

    z_shift_r = _TF.hand_shift_z(base_seq)
    if _TF.is_session_exception(base_seq):
        print(
            f"  [render_mano] {sequence}: session {_TF.session_of(base_seq)} exception z "
            f"shift {z_shift_r:+.4f} (default {_TF.HAND_SHIFT_Z:+.4f})",
            flush=True,
        )
    return z_shift_r


def _frame_index(kin, _tree: Path, T_raw: int, T_obj: int, _pad: int, _T_motion: int) -> np.ndarray:
    """Raw MANO frame index for each tree motion frame (the retime map when the tree has one, else uniform)."""
    # Resample raw MANO to the MOTION span (the pad is filled below by repeating frame 0).
    # A retimed tree is a **piecewise non-uniform** warp (dexengine.placement.retime.warp: "stretched N intervals"), so
    # a uniform linspace misaligns hands and objects in stretched spans. When the tree holds the warp map (retime_tau:
    # source frame i -> new time) it aligns by that; otherwise (old trees) it falls back to the uniform approximation.
    _files = getattr(kin, "files", [])
    _sidx = kin["retime_src_idx"] if "retime_src_idx" in _files else None
    if _sidx is not None and len(_sidx) == _T_motion:
        # tree frame -> frame just before retiming (fractional). That frame axis shares the raw MANO time axis, so the
        # raw index follows by length ratio.
        _pre = _tree / "0" / f"trajectory_kinematic{IK_SIDECAR_SFX}"
        _Tpre = int(np.load(_pre)["qpos"].shape[0]) if _pre.exists() else int(np.max(_sidx)) + 1
        _scale = (T_raw - 1) / max(_Tpre - 1, 1)
        idx = np.clip(np.rint(np.asarray(_sidx, dtype=np.float64) * _scale), 0, T_raw - 1).astype(int)
        print(f"[load_taco] retime_src_idx alignment used (raw {T_raw}f · pre_rt {_Tpre}f · tree {_T_motion}f)")
    else:
        idx = np.linspace(0, T_raw - 1, _T_motion).round().astype(int)
    return idx


def _hand_verts(sequence, idx, _pad, T_obj, right, left, z_shift_r, z_shift_l):
    """MANO FK of both hands at the target wrists (raw trans with the z shift down) on the motion frames, the f0 pad
    prepended -> (right verts, left verts). right/left = (raw pose, raw trans, betas)."""
    rh_pose_raw, rh_trans_raw, rh_shape = right
    lh_pose_raw, lh_trans_raw, lh_shape = left
    rh_pose = rh_pose_raw[idx].clone()  # (T_obj, 48) — full raw pose preserved
    lh_pose = lh_pose_raw[idx].clone()

    # Target wrist in retarget/object frame = raw_trans with Z shifted down
    rh_target_wrist = rh_trans_raw[idx].clone().float()
    rh_target_wrist[:, 2] -= z_shift_r
    lh_target_wrist = lh_trans_raw[idx].clone().float()
    lh_target_wrist[:, 2] -= z_shift_l

    rh_verts = mano_fk_batch(rh_pose, rh_target_wrist, rh_shape, is_right=True)
    lh_verts = mano_fk_batch(lh_pose, lh_target_wrist, lh_shape, is_right=False)
    if _pad > 0:
        rh_verts = np.concatenate([np.repeat(rh_verts[0:1], _pad, axis=0), rh_verts], axis=0)
        lh_verts = np.concatenate([np.repeat(lh_verts[0:1], _pad, axis=0), lh_verts], axis=0)
        print(f"  [render_mano] {sequence}: f0 pad {_pad} frames prepended (T={T_obj})", flush=True)
    return rh_verts, lh_verts


def _apply_placement(rh_verts, lh_verts, _yaw: float, _dxy, _piv):
    """Re-apply the rigid placement shift (trajectory frame) to the hand verts. The object tracks in
    trajectory_kinematic.npz are ALREADY shifted, but raw MANO poses are the base placement — so the hand must be
    rigidly transformed by the SAME (dxy, yaw about pivot) to track the shifted objects. Mirrors apply_xy_shift.rt_pts:
    (p - pivot) @ Rz.T + pivot + d. yaw is about +Z, dz=0 → Z unchanged."""
    if _yaw != 0.0 or _dxy[0] != 0.0 or _dxy[1] != 0.0:
        _pivot = np.array([_piv[0], _piv[1], 0.0])
        _Rz = R.from_euler("z", _yaw, degrees=True).as_matrix()
        _d3 = np.array([_dxy[0], _dxy[1], 0.0])

        def _rt(v):
            shp = v.shape
            return (((v.reshape(-1, 3) - _pivot) @ _Rz.T) + _pivot + _d3).reshape(shp)

        rh_verts = _rt(rh_verts)
        lh_verts = _rt(lh_verts)
    return rh_verts, lh_verts


# Build object 4x4 transforms from each side's object block (slot below — side_objects.resolve); wxyz quat
def quat_pos_to_T(arr7):
    T_arr = np.tile(np.eye(4, dtype=np.float64), (arr7.shape[0], 1, 1))
    T_arr[:, :3, 3] = arr7[:, :3]
    # mujoco wxyz → xyzw for scipy
    quat_xyzw = arr7[:, [4, 5, 6, 3]]
    T_arr[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    return T_arr


def _object_transforms(sequence: str, _tree: Path, qpos, _HH2: int, legacy_slot: bool):
    """Per-frame 4x4 transforms of the right and left hand's objects (slots from side_objects.resolve)."""
    # slot <-> side assignment is owned by the side_objects resolver. Taking **trajectories from fixed slots**
    # (right = range(36, 43)) and **meshes from task_info (grasp measurement)** drew the two meshes on each other's
    # trajectories in clips where the two assignments differ (a person holding the tool in the left hand): the hand
    # motion is right, but the right hand appears to hold the kettle and the left the bowl (pour in
    #   some_kettle_bowl__20231019_009).
    # With the standard assignment right = range(36, 43) / left = range(43, 50) it is byte-identical.
    from dexengine.data import side_objects as _SO

    _tree_dir = _tree
    _sides_rm = _SO.resolve(_tree_dir)
    _slot_r, _slot_l = _sides_rm["right"][1], _sides_rm["left"][1]
    # diagnosis only (before/after comparison): legacy_slot=True ignores the resolver and draws the old way (right
    # hand = slot 0, left = slot 1). **A function argument / CLI flag, not an env variable**: env is inherited by child
    # processes and could silently taint the training, evaluation and annotation paths. Default False = resolver =
    # canonical.
    if os.environ.get("MANO_LEGACY_SLOT", ""):
        raise SystemExit(
            "[render_mano] the MANO_LEGACY_SLOT env variable is retired: "
            "use the CLI flag --legacy_slot_diag for diagnosis."
        )
    if legacy_slot:
        _slot_r, _slot_l = 0, 1
        print("  [render_mano] ⚠ --legacy_slot_diag: drawing with the old fixed slots (diagnosis only)", flush=True)
    if _slot_r != 0:
        print(
            f"  [render_mano] {sequence}: slots flipped: right=slot{_slot_r}"
            f"({_sides_rm['right'][0]}) left=slot{_slot_l}({_sides_rm['left'][0]})",
            flush=True,
        )
    rh_obj_T = quat_pos_to_T(qpos[:, _HH2 + 7 * _slot_r : _HH2 + 7 * _slot_r + 7])
    lh_obj_T = quat_pos_to_T(qpos[:, _HH2 + 7 * _slot_l : _HH2 + 7 * _slot_l + 7])
    return rh_obj_T, lh_obj_T


def _to_isaac(rh_verts, lh_verts, rh_obj_T, lh_obj_T):
    """TACO (Z-up) → Isaac world: Rz(-90°) + z+=DESK_HEIGHT, for the hand verts and the object transforms."""
    rh_verts = (RZ_MINUS_90 @ rh_verts.transpose(0, 2, 1)).transpose(0, 2, 1)
    rh_verts[..., 2] += DESK_HEIGHT
    lh_verts = (RZ_MINUS_90 @ lh_verts.transpose(0, 2, 1)).transpose(0, 2, 1)
    lh_verts[..., 2] += DESK_HEIGHT
    T_taco = np.eye(4)
    T_taco[:3, :3] = RZ_MINUS_90
    T_taco[:3, 3] = [0, 0, DESK_HEIGHT]
    rh_obj_T = np.einsum("ij,njk->nik", T_taco, rh_obj_T)
    lh_obj_T = np.einsum("ij,njk->nik", T_taco, lh_obj_T)
    return rh_verts, lh_verts, rh_obj_T, lh_obj_T


def load_taco(sequence: str, tree_dir=None, z_anchor: str = "legacy", legacy_slot: bool = False) -> dict:
    """Load TACO MANO + object data for the given sequence.

    Coordinate alignment: raw TACO Hand_Poses are in TACO's Z-up world
    frame; the object trajectory in trajectory_kinematic.npz is in the
    retarget-processed frame which differs by a CONSTANT Z translation
    (~55 cm; XY axes are identical, as verified by raw hand_trans matching
    the MANO layer's qpos_wrist_right in XY at every frame).

    Approach (mesh-correct, hand wraps object):
      1. Use raw MANO global_orient + raw finger pose + raw betas
         (preserves the natural hand orientation and finger curl)
      2. Subtract the per-sequence Z shift from raw hand_trans to put the
         wrist target in the retarget frame
      3. Compute MANO rest wrist offset J_rest[0] from the betas, and
         set MANO transl = target_wrist - J_rest[0] so FK output joint[0]
         lands exactly at the target wrist position
    This eliminates the ~10cm hand-object offset that the previous version
    showed (caused by treating raw hand_trans as the FK wrist output
    directly, when in fact joint[0] = transl + J_rest[0]).

    Sources:
    - Raw MANO pose + shape: data/taco/raw_unzipped/Hand_Poses/
    - Wrist Z-shift anchor: trajectory_keypoints.npz qpos_wrist_*[0,2]
    - Object 7DOF: trajectory_kinematic.npz qpos — each side's free-joint block
      (2*hand width + 7*slot, slot from side_objects.resolve)
    - Object meshes: data/taco/meshes/<id>/visual.obj
    """
    # Rigid-shift sequences (e.g. "..._shift4") carry the placement transform in their object
    # trajectory + task_info, but raw MANO Hand_Poses only exist for the BASE sequence. Resolve
    # raw poses from the base name; the shift is re-applied to the FK'd verts below (see task_info).
    import re

    # _pl = placement-applied (an earlier placement chain) — same rigid-shift
    # semantics as _shift<k>, so the same base-name resolution applies. _rt (retimed) is
    # NOT stripped here: retimed sequences carry their own resampled sidecars and callers
    # must not silently fall back to the un-retimed raw poses.
    _m = re.search(r"(_shift\d*|_pl)$", sequence)
    base_seq = sequence[: _m.start()] if _m else sequence
    # the raw place is resolved by paths.raw_from_clip_id alone (a lookup table plus a split("_") inversion here could
    # not find the raw folder of normalised task names such as "pour-in-some" <- "pour in some").
    triplet, seq_id = raw_from_clip_id(base_seq)
    hp_dir = _data_root() / "data" / "taco" / "raw_unzipped" / "Hand_Poses" / triplet / seq_id
    rh_pose_raw, rh_trans_raw, rh_shape = _load_hand_pkl(hp_dir, "right")
    lh_pose_raw, lh_trans_raw, lh_shape = _load_hand_pkl(hp_dir, "left")
    T_raw = rh_pose_raw.shape[0]

    _tree, kin, qpos, T_obj, _ti_all, _HH2, _pad, _T_motion = _tree_inputs(sequence, tree_dir)
    z_shift_r = z_shift_l = _hand_shift_z(sequence, base_seq)
    idx = _frame_index(kin, _tree, T_raw, T_obj, _pad, _T_motion)
    rh_verts, lh_verts = _hand_verts(
        sequence,
        idx,
        _pad,
        T_obj,
        (rh_pose_raw, rh_trans_raw, rh_shape),
        (lh_pose_raw, lh_trans_raw, lh_shape),
        z_shift_r,
        z_shift_l,
    )
    _ti = _ti_all  # the tree's own placement record: for a combo tree the combo placement also applies to the hands
    _yaw = float(_ti.get("yaw_deg", 0.0))
    _dxy = _ti.get("xy_shift_traj", [0.0, 0.0])
    _piv = _ti.get("pivot_traj", [0.0, 0.0])
    rh_verts, lh_verts = _apply_placement(rh_verts, lh_verts, _yaw, _dxy, _piv)
    rh_obj_T, lh_obj_T = _object_transforms(sequence, _tree, qpos, _HH2, legacy_slot)
    rh_verts, lh_verts, rh_obj_T, lh_obj_T = _to_isaac(rh_verts, lh_verts, rh_obj_T, lh_obj_T)

    return _assemble_taco_result(
        _ti_all, z_anchor, sequence, rh_verts, lh_verts, rh_obj_T, lh_obj_T, _tree, T_obj, _yaw, _dxy, _piv
    )
