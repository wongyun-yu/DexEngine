"""Fingertip and surface geometry shared by the contact rewards, and the loaders of contact labels and grasp targets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase

#: Returned by a split-out loop step that skips the rest of this iteration (`continue`).
_SKIP = object()

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Per-fingertip OBJECT-filtered contact sensors (one ContactSensor per fingertip body,
# each filtering /Object/base_link). A single multi-body hand sensor CANNOT populate
# force_matrix_w in multi-env (PhysX filter cardinality: expects n_bodies*n_envs filter
# matches, finds n_envs). Ordered thumb, index, middle, ring, little — matching the tip
# order from resolve_tip_body_ids. Names must match the ContactSensorCfg attrs in the
# arm-residual scene cfgs.
OBJ_SENSOR_NAMES = {
    "right": ["right_thumb_obj_s", "right_index_obj_s", "right_middle_obj_s", "right_ring_obj_s", "right_little_obj_s"],
    "left": ["left_thumb_obj_s", "left_index_obj_s", "left_middle_obj_s", "left_ring_obj_s", "left_little_obj_s"],
}


def resolve_tip_body_ids(hand: Articulation, side: str) -> list[int]:
    """Return PhysX body indices of the 5 fingertips for ``side`` hand.

    The names come from the spec-based resolver in naming; a per-hand name list here once failed for any hand missing
    from it ('Could not resolve tip body names').
    """
    from dexengine.rl.tasks.mdp.naming import resolve_tip_body_names

    body_names = hand.data.body_names
    names = resolve_tip_body_names(hand, side)
    missing = [n for n in names if n not in body_names]
    if missing:
        raise ValueError(
            f"side='{side}' fingertip bodies {missing} are not in the articulation. Available: {body_names}"
        )
    return [body_names.index(n) for n in names]


_TIP_OFFSET_LOGGED: set = set()


def resolve_tip_offsets_tensor(hand: Articulation, side: str, device) -> torch.Tensor | None:
    """(5,3) fingertip-site offset tensor for ``hand``/``side``; None when all
    offsets are zero (RH56F1 / dataset hands) so ``tip_sites_w`` can skip the
    quat math entirely (byte-identical body-origin path). Resolved ONCE at
    term init. Logs the resolution once per (side, table)."""
    from dexengine.rl.tasks.mdp.naming import resolve_fingertip_site_offsets

    off = resolve_fingertip_site_offsets(hand, side)
    nonzero = any(v != 0.0 for row in off for v in row)
    key = (side, off)
    if key not in _TIP_OFFSET_LOGGED:
        if nonzero:
            print(
                f"[FingertipSite] side={side}: tip-site offsets (m, tip-body local) "
                f"= {[list(r) for r in off]} — object-distance quantities measure "
                f"from the PHYSICAL fingertip (DP origin + R·offset)"
            )
        else:
            print(
                f"[FingertipSite] side={side}: all-zero offsets — tip BODY ORIGIN (RH56F1/legacy path, byte-identical)"
            )
        _TIP_OFFSET_LOGGED.add(key)
    if not nonzero:
        return None
    return torch.tensor(off, dtype=torch.float32, device=device)


def tip_sites_w(hand: Articulation, tip_body_ids, tip_offsets) -> torch.Tensor:
    """(N, n_tips, 3) WORLD fingertip SITE positions for the given tip bodies.

    site = body_link_pos_w + quat_apply(body_link_quat_w, offset_local).
    ``tip_offsets`` is the (n_tips, 3) tensor from ``resolve_tip_offsets_tensor``
    or None; None returns the body origins EXACTLY (no math executed), keeping
    the RH56F1 path byte-identical."""
    tips_w = hand.data.body_link_pos_w[:, tip_body_ids, :]  # (N,n,3) WORLD
    if tip_offsets is None:
        return tips_w
    from isaaclab.utils.math import quat_apply

    q = hand.data.body_link_quat_w[:, tip_body_ids, :]  # (N,n,4) wxyz WORLD
    N, n = tips_w.shape[0], tips_w.shape[1]
    off = tip_offsets.unsqueeze(0).expand(N, n, 3)
    return tips_w + quat_apply(q.reshape(-1, 4), off.reshape(-1, 3)).reshape(N, n, 3)


class LookaheadWristTarget(ManagerTermBase):
    """Per-side lookahead wrist obs.

    For each ``k`` in ``lookahead_frames`` (default (10, 30)), emit
    delta_pos_k = ref_wrist_pos[t+k] - ref_wrist_pos[t]      (3D, WORLD)
    delta_quat_k = ref_wrist_quat[t+k] * ref_wrist_quat[t]^-1 (4D wxyz, WORLD)

    Output dim per side: 7 * len(lookahead_frames). Default 14.
    Returned as flat (N, 7*K) tensor.

    Slow-down aware: get_wrist_pos_w / get_wrist_quat_w handle the
    effective_frame = (t+k) / slowdown_factor interp internally.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._lookahead_frames = tuple(cfg.params.get("lookahead_frames", (10, 30)))
        self._side = cfg.params.get("side", "right")
        self._traj_mgr = env.trajectory_manager
        self._sanity_done = False

    def __call__(self, env: ManagerBasedEnv, lookahead_frames: tuple = (10, 30), side: str = "right") -> torch.Tensor:
        traj_mgr = env.trajectory_manager
        # frame_ids: WORLD time, per-env policy-step counter
        frame_ids = env.episode_length_buf
        cur_pos = traj_mgr.get_wrist_pos_w(frame_ids)  # (N, 3) WORLD
        cur_quat = traj_mgr.get_wrist_quat_w(frame_ids)  # (N, 4) wxyz WORLD
        outs: list[torch.Tensor] = []
        for k in lookahead_frames:
            la_frames = frame_ids + k
            # Clamp to clip end via traj_mgr's internal clip_lengths; if past
            # end, get_wrist_pos_w may saturate. Safe behavior is to clamp to
            # the last available frame (no extrapolation).
            fut_pos = traj_mgr.get_wrist_pos_w(la_frames)  # (N, 3) WORLD
            fut_quat = traj_mgr.get_wrist_quat_w(la_frames)  # (N, 4) wxyz
            outs.append(fut_pos - cur_pos)  # delta (N, 3) WORLD
            outs.append(self._quat_delta(fut_quat, cur_quat))  # delta (N, 4) wxyz
        if not self._sanity_done:
            # SANITY CHECK: at frame 0 of env 0, the delta_pos at k=10
            # should be small (a few cm) since trajectories are ~1 m/s peak.
            try:
                k0 = lookahead_frames[0]
                d_pos = outs[0][0].cpu()
                print(
                    f"[LookaheadWristTarget][sanity] side={side} k={k0} "
                    f"env0 delta_pos_world=({d_pos[0]:+.4f}, {d_pos[1]:+.4f}, "
                    f"{d_pos[2]:+.4f}) m  (expect <0.5m for normal trajectory)"
                )
            except (AttributeError, IndexError, TypeError) as _exc:
                print(f"[LookaheadWristTarget][sanity] skipped: {_exc!r}")
            self._sanity_done = True
        return torch.cat(outs, dim=-1)  # (N, 7 * K)

    @staticmethod
    def _quat_delta(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        """q_delta = q_a * inverse(q_b) for wxyz quats. Both (N, 4)."""
        # inverse of unit quat = conjugate
        q_b_inv = torch.cat([q_b[:, :1], -q_b[:, 1:]], dim=-1)
        return _quat_mul_wxyz(q_a, q_b_inv)


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions. Both (N, 4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


# Module-level cache for the annotation/asset loaders. The contact pilot
# instantiates ~11 ContactMatchBase subclasses (1 reward + 1 approach + diags),
# each of which previously re-read surface_points.npz / contact_<src>.npz /
# trajectory_kinematic.npz from disk. Keyed by (string args, device) so each unique
# asset is loaded ONCE per process; returns the SAME tensor (no semantic change).
# A fresh process (new training/eval run) starts with an empty cache → loads fresh.
_LOADER_CACHE: dict = {}


def surf_pts_for(term, env):
    """(P,3) shared cloud, or (N,P,3) per-env clouds gathered by clip id (--multi_object)."""
    if getattr(term, "_surf_multi", None) is None:
        return term._surf_local
    return term._surf_multi[env.trajectory_manager.clip_ids]


def load_object_surface_points(mesh_dir: str, device: torch.device) -> torch.Tensor:
    """Load (N_pts, 3) surface point cloud from <mesh_dir>/surface_points.npz.
    Points are in OBJECT-LOCAL frame; caller transforms to world via current
    object pose. Returns float32 tensor on ``device``. Cached per (mesh_dir, device)."""
    key = ("surf", os.path.abspath(mesh_dir), str(device))
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]
    p = os.path.join(mesh_dir, "surface_points.npz")
    d = np.load(p)
    out = torch.tensor(d["points"], dtype=torch.float32, device=device)
    _LOADER_CACHE[key] = out
    return out


def tip_to_surface_dist(tips_w: torch.Tensor, obj, surf_local: torch.Tensor) -> torch.Tensor:
    """Min distance from each tip to nearest object-surface point.
    Inputs:
        tips_w: (N, 5, 3) WORLD frame
        obj:    RigidObject with data.root_pos_w / root_quat_w
        surf_local: (n_pts, 3) OBJECT-LOCAL frame
    Returns:
        (N, 5) distances in meters, WORLD frame
    """
    obj_pos_w = obj.data.root_pos_w  # (N, 3)
    obj_quat_w = obj.data.root_quat_w  # (N, 4) wxyz
    # Transform surf_local → world: world = q * local * q^-1 + pos
    # Use the standard Isaac Lab quat_apply
    from isaaclab.utils.math import quat_apply

    # surf_local: (n_pts, 3) → broadcast to (N, n_pts, 3)
    N = tips_w.shape[0]
    if surf_local.dim() == 3:  # (N, P, 3) per-env clouds (--multi_object)
        n_pts = surf_local.shape[1]
        surf_local_b = surf_local
    else:
        n_pts = surf_local.shape[0]
        surf_local_b = surf_local.unsqueeze(0).expand(N, n_pts, 3)
    q_b = obj_quat_w.unsqueeze(1).expand(N, n_pts, 4)
    surf_rotated = quat_apply(q_b.reshape(-1, 4), surf_local_b.reshape(-1, 3)).reshape(N, n_pts, 3)
    surf_world = surf_rotated + obj_pos_w.unsqueeze(1)  # (N, n_pts, 3) WORLD
    # Pairwise distances (N, 5, n_pts)
    diffs = tips_w.unsqueeze(2) - surf_world.unsqueeze(1)
    dists = diffs.norm(dim=-1)
    return dists.min(dim=-1).values  # (N, 5)


def _load_fingertip_contact_label(data_dir: str, source: str, traj_mgr, device: torch.device) -> torch.Tensor:
    """Load the new per-finger FINGERTIP-region contact annotation per clip.

    Indexes per ``traj_mgr.clip_ids`` / ``traj_mgr.start_frames`` so the
    returned tensor is aligned with the live env clip assignment exactly.

    Reads ``<data_dir>/<seq>/0/contact_<source>.npz`` field ``contact`` of
    shape (T,5,2) bool; keeps the fingertip region ``contact[:, :, 0]``.

    Returns:
        (num_clips, max_length, 5) bool tensor on ``device``. Missing clips
        are all-False (the reward is simply 0 for them).
    """
    _ck = ("fingertip_label", os.path.abspath(data_dir), source, str(device))
    if _ck in _LOADER_CACHE:
        return _LOADER_CACHE[_ck]
    loader = traj_mgr._loader
    seq_names = loader.sequence_names
    num_clips = loader.num_clips
    max_length = loader.max_length
    all_contact = np.zeros((num_clips, max_length, 5), dtype=bool)
    n_loaded = 0
    n_missing = 0
    for ci, seq_name in enumerate(seq_names):
        p = os.path.join(data_dir, seq_name, "0", f"contact_{source}.npz")
        if not os.path.exists(p):
            n_missing += 1
            continue
        d = np.load(p)
        if "contact" not in d.files:
            n_missing += 1
            continue
        c = np.asarray(d["contact"]).astype(bool)  # (T,5,2)
        if c.ndim != 3 or c.shape[1] != 5 or c.shape[2] < 1:
            n_missing += 1
            continue
        c_tip = c[:, :, 0]  # (T,5) fingertip region
        n = min(c_tip.shape[0], max_length)
        all_contact[ci, :n, :] = c_tip[:n, :]
        n_loaded += 1
    out = torch.tensor(all_contact, device=device, dtype=torch.bool)
    print(
        f"[RewardContactMatch] Loaded fingertip contact_{source} for "
        f"{n_loaded}/{num_clips} clips (missing: {n_missing}); "
        f"shape={tuple(out.shape)}"
    )
    # A contact reward that is on without labels is 0, and training runs "with a dead reward" (all labels 0 once passed
    # an A/B on a new combo tree set). The recipe always turns this reward on, so missing labels are never passed over.
    if n_missing:
        raise ValueError(
            f"[RewardContactMatch] contact_{source} labels missing {n_missing}/{num_clips} — "
            "the contact reward would train dead at 0. Re-annotate the tree contacts first"
        )
    if n_loaded > 0:
        per_tip = out.float().mean(dim=(0, 1)).cpu().tolist()
        print(f"[RewardContactMatch][sanity] per-tip ref-contact frac (thumb..pinky)={[f'{v:.2f}' for v in per_tip]}")
    _LOADER_CACHE[_ck] = out
    return out


def _load_grasp_targets(data_dir: str, source: str, traj_mgr, device: torch.device, side: str = "right"):
    """Per-finger GT contact target (object-local) from the annotation ``witness_obj``.

    For each clip and finger, the GT contact point on the OBJECT is ``witness_obj``
    (region 0 = fingertip), stored SCENE-frame in ``<seq>/0/contact_<source>.npz``.
    We convert it to OBJECT-LOCAL using the clip's object pose (``trajectory_kinematic.npz``
    qpos free-joint block, scene-frame), then take the MEDIAN over the finger's
    contact window → one stable object-local point per (clip, finger). Object-local
    is transform-invariant and matches the mesh frame of ``surface_points.npz``
    (verified: witness lands 0.09-0.15 cm from the surface cloud), so at runtime
    ``target = obj_pos_w + R(obj_quat_w)·c_local`` is correct under the standard
    object pose convention used by reset / reward / termination.

    ``side`` selects WHICH object's free-joint block to convert against. The bimanual
    qpos layout has the right object's 7-wide free-joint block first and the left
    object's second (e.g. TACO: right_obj@[36, 43), left_obj@[43, 50)). side='right'
    (default) takes the first block (byte-identical to the prior scan-and-break);
    side='left' takes the second. Using the wrong object frame silently corrupts every
    left spatial-target reward (grasp-target/approach/pregrasp/opposition/aperture/
    closure-potential).

    Returns:
        targets (num_clips, 5, 3) object-local contact point per finger (float32, device),
        will_contact (num_clips, 5) bool: finger has any contact frame in the clip.
    """
    return _scan_grasp_targets(data_dir, source, traj_mgr, device, side)


def _scan_grasp_targets(data_dir, source, traj_mgr, device, side):
    """Scan every clip's contact sidecar into the per-finger targets (the uncached path)."""
    _gk = ("grasp_targets", os.path.abspath(data_dir), source, side, str(device))
    if _gk in _LOADER_CACHE:
        return _LOADER_CACHE[_gk]
    from scipy.spatial.transform import Rotation as _Rot

    loader = traj_mgr._loader
    seq_names = loader.sequence_names
    num_clips = loader.num_clips
    targets = np.zeros((num_clips, 5, 3), dtype=np.float32)
    will = np.zeros((num_clips, 5), dtype=bool)
    first_contact = np.full((num_clips,), -1, dtype=np.int64)  # first frame ANY finger contacts
    ref_start = np.full((num_clips, 5), -1, dtype=np.int64)  # per-finger first ref-contact frame
    ref_end = np.full((num_clips, 5), -1, dtype=np.int64)  # per-finger last ref-contact frame
    n_loaded = 0
    for ci, seq in enumerate(seq_names):
        cpath = os.path.join(data_dir, seq, "0", f"contact_{source}.npz")
        tpath = os.path.join(data_dir, seq, "0", "trajectory_kinematic.npz")
        if not (os.path.exists(cpath) and os.path.exists(tpath)):
            continue
        cm = np.load(cpath)
        if "witness_obj" not in cm.files or "contact" not in cm.files:
            continue
        wit = np.asarray(cm["witness_obj"])[:, :, 0, :]  # (T,5,3) scene-frame contact point
        con = np.asarray(cm["contact"])[:, :, 0].astype(bool)  # (T,5) fingertip-region contact
        _anyc = np.where(con.any(axis=1))[0]
        if len(_anyc):
            first_contact[ci] = int(_anyc[0])
        qpos = np.load(tpath)["qpos"]  # (T, nq)
        oslice = _clip_object_slice(data_dir, side, cm, qpos, seq)
        if oslice is _SKIP:
            continue
        opos = qpos[:, oslice : oslice + 3]
        oquat_wxyz = qpos[:, oslice + 3 : oslice + 7]
        oquat_wxyz = oquat_wxyz / np.clip(np.linalg.norm(oquat_wxyz, axis=1, keepdims=True), 1e-8, None)
        R_obj = _Rot.from_quat(oquat_wxyz[:, [1, 2, 3, 0]]).as_matrix()  # wxyz->xyzw (T,3,3)
        for fi in range(5):
            cw = np.where(con[:, fi])[0]
            if len(cw) == 0:
                continue
            will[ci, fi] = True
            # object-local: R_obj^T (witness - obj_pos), median over contact frames
            loc = np.einsum("tij,tj->ti", np.transpose(R_obj[cw], (0, 2, 1)), wit[cw, fi] - opos[cw])  # (n,3)
            targets[ci, fi] = np.median(loc, axis=0)
            ref_start[ci, fi] = int(cw.min())
            ref_end[ci, fi] = int(cw.max())
        n_loaded += 1
    print(
        f"[GraspTarget] loaded witness_obj GT targets for {n_loaded}/{num_clips} clips; "
        f"will-contact fingers/clip(mean)={will.sum(axis=1).mean():.2f}; "
        f"first_contact(env0-clip0)={int(first_contact[0])}"
    )
    out = (
        torch.tensor(targets, device=device),
        torch.tensor(will, device=device),
        torch.tensor(first_contact, device=device),
        torch.tensor(ref_start, device=device),
        torch.tensor(ref_end, device=device),
    )
    _LOADER_CACHE[_gk] = out
    return out


def _clip_object_slice(data_dir, side, cm, qpos, seq):
    """The clip's object column for this side, checked against the tree layout and the contact sidecar.

    Returns ``_SKIP`` (the caller skips the clip) when the tree, the sidecar or the quaternion block does not fit.
    """
    # Object free-joint slice, derived from the loader's AUTHORITATIVE layout (no
    # hardcoded column index): both hands span [0 : 2*hand_qpos_dim], then the right
    # primary object's 7-wide block (pos3 + quat4 wxyz), then the left's. This mirrors
    # TacoTrajectoryLoader.get_frame exactly (right_obj@_HH, left_obj@_HH+7) and
    # generalizes across hand morphologies (RH56F1 _H=18, SharpA _H=28).
    # The tree states the object columns. A second copy of the loader's arithmetic (2·hand_qpos_dim + 7·slot) used
    # to live here: it gave the same value on TACO but splits from the loader on the base layer (one-hand trees,
    # shared objects, extras). The address owner is the qpos_layout reader alone; this **only reads** it.
    from dexengine.rl.tasks.mdp.trajectory.loader_clips import qpos_layout_reader as _ql_reader

    try:
        _lay = _ql_reader().read(os.path.join(data_dir, seq))
    except Exception as _e:  # noqa: BLE001 — an unreadable tree skips that clip only
        print(f"[GraspTarget] WARNING: clip {seq} layout unreadable ({type(_e).__name__}: {_e}) — skipped", flush=True)
        return _SKIP
    _sl = _lay.main_slice(side)
    if _sl is None:
        print(f"[GraspTarget] WARNING: clip {seq} side={side} has no main object in the tree — skipped", flush=True)
        return _SKIP
    oslice = _sl[0]
    # Check that the contact sidecar was computed for **this hand's object**. Otherwise the clip's grasp reward
    # would
    # target an object the hand never touches, so it is not used silently.
    if "tool_mesh_id" in cm.files:
        # This hand's object identity = the layout reader (objects[] role=main; legacy <side>_object_mesh_dir).
        # Reading
        # the legacy key directly could disagree with objects[] on base trees (shared objects, reassignment).
        _want = str(_lay.main_object_id(side) or "")
        if _want:
            _got = str(cm["tool_mesh_id"])
            if _got != _want:
                print(
                    f"[GraspTarget] WARNING: clip {seq} side={side} contact sidecar is "
                    f"for object {_got} but this hand's object is {_want} — this clip's grasp target "
                    f"is unused until the contacts are re-annotated. "
                    f"Fix: re-annotate the MANO contacts of '{os.path.join(data_dir, seq)}' "
                    f"(that tree's _pl name, side {side}, overwrite)"
                )
                return _SKIP
    # Sanity: the selected block must be a unit-norm free-joint quat across all frames.
    # Fails loud (skips clip) if the layout assumption is violated (e.g. extra scene
    # objects shift the primary blocks) rather than silently using the wrong frame.
    if oslice + 7 > qpos.shape[1]:
        # Not dropped silently: one-hand trees could vanish through a quiet path here.
        print(
            f"[GraspTarget] WARNING: clip {seq} side={side} object block {oslice}:{oslice + 7} exceeds "
            f"the qpos width {qpos.shape[1]} — skipped",
            flush=True,
        )
        return _SKIP
    _qn = np.linalg.norm(qpos[:, oslice + 3 : oslice + 7], axis=1)
    # The block must be a free-joint quat (norm ~1). Accept a valid-but-slightly-
    # DENORMALIZED quat (e.g. brush_brush_bowl__20231005_194's right object dips to
    # ~0.98 from the kinematic retargeting pipeline) and renormalize it below; only a
    # GROSS deviation (e.g. ~0.5 = a wrong/quat-first block from a layout shift) is a
    # real mismatch. The old strict <1e-2 tolerance silently SKIPPED the brush clip,
    # leaving will_contact all-False -> the right-hand closure/approach/grasp rewards
    # were dead for the brush (policy pressed the desk-supported brush without grasping).
    if not (np.all(np.abs(_qn - 1.0) < 0.1) and not np.allclose(qpos[:, oslice + 3], 0.0)):
        print(
            f"[GraspTarget] WARNING: clip {seq} side={side} expected object free-joint "
            f"block at col {oslice} but quat norm off by >0.1 (min={_qn.min():.3f}, "
            f"max={_qn.max():.3f}) — skipping (real layout mismatch)."
        )
        return _SKIP
    return oslice
