"""Object keypoint tracking reward and its per-keypoint diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg

from dexengine.core.desk import DESK_TOP_SURFACE_Z

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from dexengine.rl.tasks.mdp.rewards.task_base import _TaskRewardBase


def _kp_missing_msg(missing: list[str], n_dirs: int) -> str:
    """Keypoints are a declared training asset (train_ready_util.OBJ_ASSETS): a requested keypoint
    reward whose asset is absent is an error, never a silent no-op. KEYPOINT=0 turns the reward off."""
    head = ", ".join(missing[:3]) + (" …" if len(missing) > 3 else "")
    return (
        f"[ObjKeypoint] keypoints_fps2.npy missing for {len(missing)}/{n_dirs} object mesh dirs ({head}) — "
        "the keypoint reward was requested (KEYPOINT=1). Restore the assets in this tree "
        "(check: python3 -m dexengine.placement.train_ready) or launch with KEYPOINT=0."
    )


def _load_kp_multi(mesh_dir: str, device):
    """--multi_object: comma-list mesh_dir (LOADER clip order) -> per-clip keypoint table.
    Returns None for a single dir (legacy path unchanged), else a dict:
      table (C, 2, 3) object-local keypoints (zeros where absent),
      valid (C,) bool — clip has keypoints_fps2.npy,
      far   (C,) long — per-clip farthest-from-surface-centroid keypoint index."""
    import os

    import numpy as np

    dirs = [d for d in str(mesh_dir).split(",") if d]
    if len(dirs) <= 1:
        return None
    C = len(dirs)
    missing = [d for d in dirs if not os.path.exists(os.path.join(d, "keypoints_fps2.npy"))]
    if missing:
        raise FileNotFoundError(_kp_missing_msg(missing, len(dirs)))
    table = torch.zeros(C, 2, 3, dtype=torch.float32, device=device)
    valid = torch.zeros(C, dtype=torch.bool, device=device)
    far = torch.zeros(C, dtype=torch.long, device=device)
    for i, d in enumerate(dirs):
        p = os.path.join(d, "keypoints_fps2.npy")
        arr = np.load(p).astype(np.float32)[:2]
        table[i, : arr.shape[0]] = torch.tensor(arr, device=device)
        valid[i] = True
        sp = np.load(os.path.join(d, "surface_points.npz"))["points"].astype(np.float64)
        far[i] = int(np.argmax(np.linalg.norm(arr.astype(np.float64) - sp.mean(0), axis=1)))
    return {"table": table, "valid": valid, "far": far}


def _kp_for_envs(multi, traj_mgr, kp_index: int, kp_farthest: int):
    """Per-env keypoint (N,3) + validity (N,) gathered from the per-clip table."""
    clip = traj_mgr.clip_ids
    idx = multi["far"][clip] if kp_farthest else torch.full_like(clip, int(kp_index))
    return multi["table"][clip, idx], multi["valid"][clip]


class RewardObjKeypoint(_TaskRewardBase):
    """Object keypoint position tracking: exp(-k * ||ref_kp_world - sim_kp_world||).

    For each FPS keypoint i stored in keypoints_fps2.npy (object-local coords, float32 (2,3)):
      p_sim_i = obj_pos + R(obj_quat) · kp_local_i
      p_ref_i = ref_obj_pos + R(ref_obj_quat) · kp_local_i
      reward_i = exp(-k * ||p_ref_i - p_sim_i||)

    Weight = W/2 of obj_pos weight for each term (ramped identically by ObjFollowSchedule).
    k=80 matches obj_pos. Centroid tracking remains in RewardObjPos — do not duplicate it here.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._kps: torch.Tensor | None = None  # lazily loaded (K, 3) float32
        self._kps_multi = None  # --multi_object per-clip table (dict)
        self._kps_loaded = False  # load once, on the first call

    def _load_kps(self, mesh_dir: str, device: torch.device) -> torch.Tensor:
        """Load object-local FPS keypoints (K, 3). A missing keypoints_fps2.npy is an error —
        see _kp_missing_msg (every training object must have them, train_ready_util.OBJ_ASSETS)."""
        import os

        import numpy as np

        if not mesh_dir:
            raise ValueError(
                f"[RewardObjKeypoint:{self._side}] mesh_dir param is required (there is no knife-120 fallback default)"
            )
        path = os.path.join(mesh_dir, "keypoints_fps2.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(_kp_missing_msg([mesh_dir], 1))
        arr = np.load(path).astype(np.float32)  # (K, 3)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        kp_index: int = 0,
        kp_farthest: int = 0,
        k: float = 80.0,
        side: str = "right",
        mesh_dir: str = "",
        # consumed via self._object_key (set in _TaskRewardBase); declared so IsaacLab param-validation passes
        object_key: str = "object_right",
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        if not self._kps_loaded:
            self._kps_multi = _load_kp_multi(mesh_dir, env.device)
            if self._kps_multi is None:
                self._kps = self._load_kps(mesh_dir, env.device)
            self._kps_loaded = True

        N = env.num_envs
        if self._kps_multi is None and kp_index >= self._kps.shape[0]:
            raise IndexError(
                f"[RewardObjKeypoint:{self._side}] kp_index {kp_index} but {mesh_dir}/keypoints_fps2.npy "
                f"holds {self._kps.shape[0]} keypoints"
            )

        # SIDE-AWARE: read THIS hand's object (object_key='object_left' for the left term),
        # not the hardcoded right object. self._object_key from _TaskRewardBase.
        obj = env.scene[self._object_key]
        origins = env.scene.env_origins
        obj_pos = obj.data.root_pos_w - origins  # (N, 3) env-origin-subtracted
        obj_quat = obj.data.root_quat_w  # (N, 4) wxyz

        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_pos = ref_state[f"{self._side}_obj_pos"]  # (N, 3)
        ref_quat = ref_state[f"{self._side}_obj_quat"]  # (N, 4) wxyz

        if self._kps_multi is not None:  # --multi_object: per-clip keypoint (+farthest)
            kp_exp, _kp_valid = _kp_for_envs(self._kps_multi, self._traj_mgr, kp_index, kp_farthest)
        else:
            _kp_valid = None
            kp = self._kps[kp_index]  # (3,)
            kp_exp = kp.unsqueeze(0).expand(N, 3)  # (N, 3)

        p_sim = obj_pos + quat_apply(obj_quat, kp_exp)
        p_ref = ref_pos + quat_apply(ref_quat, kp_exp)

        err = torch.norm(p_ref - p_sim, dim=-1)  # (N,)

        # Stash for diagnostics / W&B custom logging — keyed by (side, kp_index) so the LEFT
        # term does not clobber the RIGHT term's values (was keyed by kp_index only).
        if not hasattr(env, "_obj_kp_err"):
            env._obj_kp_err = {}
        env._obj_kp_err[(self._side, kp_index)] = err.detach()

        rew = torch.exp(-k * err)
        if not hasattr(env, "_obj_kp_reward"):
            env._obj_kp_reward = {}
        env._obj_kp_reward[(self._side, kp_index)] = rew.detach()

        if _kp_valid is not None:  # multi-object: zero envs whose clip has no keypoints
            rew = rew * _kp_valid.float()
        return rew


class RewardObjKeypointDiag(_TaskRewardBase):
    """Zero-weight diagnostic: object keypoint tracking error (cm) for W&B logging.

    Mirrors the ObjectTrackingSplitDiag pattern: registered at weight 1e-9, rescaled
    by 1e9 in the W&B logger so the dashboard shows true-scale values.

    component="err_cm"  -> per-keypoint error in cm (kp_index selects which one)
    component="mean"    -> mean of kp0 and kp1 errors in cm (kp_index ignored)
    component="tip_z_cm" -> SIM keypoint z above the desk surface (cm). Knife-specific tip-height
                            metric (use kp_index = FARTHEST keypoint = the tip): center-based
                            completion/lift are blind to tip drag because the reference lifts the
                            tip by ROTATING the knife.
    component="tip_drag" -> 1.0 while the REF tip is airborne (> desk_z + ref_airborne_m) but the
                            SIM tip is still at desk level (< desk_z + drag_below_m), else 0.
                            Episode mean == fraction of ref-airborne time spent dragging (per env).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._kps: torch.Tensor | None = None  # lazily loaded (2, 3) float32
        self._kps_multi = None  # --multi_object per-clip table (dict)
        self._kps_loaded = False
        self._kp_farthest = 0  # stashed from __call__ (params)

    def _load_kps(self, mesh_dir: str, device: torch.device) -> torch.Tensor:
        import os

        import numpy as np

        if not mesh_dir:
            raise ValueError(
                "[RewardObjKeypointDiag] mesh_dir param is required (there is no knife-120 fallback default)"
            )
        path = os.path.join(mesh_dir, "keypoints_fps2.npy")
        arr = np.load(path).astype(np.float32)  # (2, 3)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def _kp_exp(self, env, kp_index: int, mesh_dir: str, N: int) -> torch.Tensor:
        """(N,3) keypoint: shared (single mesh) or per-clip gather (--multi_object)."""
        if not self._kps_loaded:
            self._kps_multi = _load_kp_multi(mesh_dir, env.device)
            if self._kps_multi is None:
                self._kps = self._load_kps(mesh_dir, env.device)
            self._kps_loaded = True
        if self._kps_multi is not None:
            kp_exp, _ = _kp_for_envs(self._kps_multi, self._traj_mgr, kp_index, self._kp_farthest)
            return kp_exp
        return self._kps[kp_index].unsqueeze(0).expand(N, 3)

    def _compute_err(
        self,
        env: ManagerBasedEnv,
        kp_index: int,
        mesh_dir: str,
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        # SIDE-AWARE: left diag reads object_left (was hardcoded scene['object'] = right object).
        obj = env.scene[self._object_key]
        origins = env.scene.env_origins
        obj_pos = obj.data.root_pos_w - origins
        obj_quat = obj.data.root_quat_w

        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_pos = ref_state[f"{self._side}_obj_pos"]
        ref_quat = ref_state[f"{self._side}_obj_quat"]

        N = obj_pos.shape[0]
        kp_exp = self._kp_exp(env, kp_index, mesh_dir, N)

        p_sim = obj_pos + quat_apply(obj_quat, kp_exp)
        p_ref = ref_pos + quat_apply(ref_quat, kp_exp)

        return torch.norm(p_ref - p_sim, dim=-1)  # (N,) in metres

    def _tip_z(
        self,
        env: ManagerBasedEnv,
        kp_index: int,
        mesh_dir: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(z_sim, z_ref) of keypoint *kp_index* in world z (== env-local z; env origins have z=0)."""
        from isaaclab.utils.math import quat_apply

        obj = env.scene[self._object_key]
        origins = env.scene.env_origins
        obj_pos = obj.data.root_pos_w - origins
        obj_quat = obj.data.root_quat_w

        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_pos = ref_state[f"{self._side}_obj_pos"]
        ref_quat = ref_state[f"{self._side}_obj_quat"]

        N = obj_pos.shape[0]
        kp_exp = self._kp_exp(env, kp_index, mesh_dir, N)
        z_sim = (obj_pos + quat_apply(obj_quat, kp_exp))[:, 2]
        z_ref = (ref_pos + quat_apply(ref_quat, kp_exp))[:, 2]
        return z_sim, z_ref

    def __call__(
        self,
        env: ManagerBasedEnv,
        component: str = "err_cm",
        kp_index: int = 0,
        kp_farthest: int = 0,
        side: str = "right",
        mesh_dir: str = "",
        desk_z: float = DESK_TOP_SURFACE_Z,  # tabletop top surface z (DESK_HEIGHT, cfg.constants)
        ref_airborne_m: float = 0.05,  # ref tip counts as airborne above desk_z + this
        drag_below_m: float = 0.02,  # sim tip counts as dragging below desk_z + this
        object_key: str = "object_right",  # consumed via self._object_key; declared for IsaacLab param-validation
    ) -> torch.Tensor:
        self._kp_farthest = int(kp_farthest)
        if component == "mean":
            err0 = self._compute_err(env, 0, mesh_dir)
            err1 = self._compute_err(env, 1, mesh_dir)
            return ((err0 + err1) * 0.5) * 100.0  # cm
        elif component == "tip_z_cm":
            z_sim, _ = self._tip_z(env, kp_index, mesh_dir)
            return (z_sim - desk_z) * 100.0  # cm above desk
        elif component == "tip_drag":
            z_sim, z_ref = self._tip_z(env, kp_index, mesh_dir)
            return ((z_ref > desk_z + ref_airborne_m) & (z_sim < desk_z + drag_below_m)).float()
        else:  # "err_cm"
            return self._compute_err(env, kp_index, mesh_dir) * 100.0  # cm
