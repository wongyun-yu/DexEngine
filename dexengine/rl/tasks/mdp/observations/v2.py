# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation V2 — coordinate-unified, de-duplicated, compressed obs (input redesign step 1).

Summary of the input audit:

Frames
  - Proprio (q/dq) in the FLOATING-ROOT frame (robot base @ desk height, fixed Rz(±90));
    desk-height info is carried by q_wrist_pos.z as before.
  - ALL object / reference-tracking / contact-geometry quantities in the VIRTUAL-WRIST
    frame → orientation-equivariance (same grasp state ⇒ same obs regardless of scene yaw).
  - ``gravity_dir`` expressed in the wrist frame restores the absolute-orientation cue
    that wrist-framing removes (needed to reason about gravity during lifts).

Rotations
  - 6D everywhere (first two rotation-matrix columns, Zhou et al. 2019). Unique — no
    quaternion double-cover shortest-path sign hacks; no euler discontinuity; no sin/cos.

Removed vs V1 (449/hand)
  - BPS zeros(128, dead), sin/cos(24), privileged↔target obj-state dup(10),
    gt_tips_distance(5, ⊂ obj_to_joints), tip-force magnitudes(5, =‖xyz‖),
    palm zeroed pos(3), obj_com_rel(3, ≡ obj_pos_rel), delta_obj_ang_vel(3, was
    −cur because ref was a zeros placeholder), constant weight(1, DR-blind),
    obj_to_joints(13, root-distance scalars — superseded by per-tip surface geometry),
    13-body target stream(117 → palm + 5 tips).

Fixed vs V1
  - ref object angular velocity: finite-diff of ref quats (was zeros placeholder).
  - object mass: PhysX masses via ``physx_props.read`` (bit-identical cache of
    ``root_physx_view.get_masses()``) — reflects per-reset mass DR (was ``default_mass`` = constant).
  - per-tip surface geometry: vec + normal + signed distance all included (V1 shipped
    normal only — direction without range).

Layout
  - HandObservationV2  (96): proprio 30 + wrist tracking 21 + tip tracking 45.
  - ObjectStateV2      (87): obj kinematics 15 + ref tracking 18 + surface 35 + privileged 19.
  - With ReferenceBaseAction(12) + last_action(12/hand): per-hand policy obs = 207 (was 449).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)

from dexengine.rl.tasks.mdp.observations.hand import HandObservation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Reference trajectory rate (ref_dt = 0.02 s). Same constant ObjectTarget used for the
# finite-difference reference velocity; kept as one shared name here.
REF_FPS = 50.0


def rot6d_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """6D rotation representation: first two COLUMNS of R (Zhou et al. 2019). (…,4)→(…,6)."""
    m = matrix_from_quat(quat_wxyz)  # (..., 3, 3)
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def q_rot_vecs(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of K vectors per env by one quat per env: q (N,4), v (N,K,3) → (N,K,3).

    ``.contiguous()`` before the flatten is REQUIRED: quat_apply is TorchScript and
    ``.view``s its inputs — expanded (non-contiguous) tensors crash on torch 2.7
    (a bimanual dexengine.rl.tasks.mdp.surface crash).
    """
    N, K = v.shape[0], v.shape[1]
    q_e = q.unsqueeze(1).expand(N, K, 4).contiguous().reshape(-1, 4)
    return quat_apply(q_e, v.contiguous().reshape(-1, 3)).reshape(N, K, 3)


class HandObservationV2(HandObservation):
    """(39+2F+9T)-dim per-hand proprio + reference tracking (frame-unified; F = finger
    joints, T = tips — e.g. wuji F=20,T=5 → 124).

    Layout:
      proprio (18+2F):
        q_wrist_pos_root (3), q_wrist_rot6d_root (6), q_fingers (F)
        dq_wrist_lin_root (3), dq_wrist_ang_root (3), dq_fingers (F)
        gravity_dir_wrist (3)
      wrist tracking, wrist frame (21):
        delta_wrist_pos (3), delta_wrist_rot6d (6)
        ref_wrist_vel (3), delta_wrist_vel (3)
        ref_wrist_ang_vel (3), delta_wrist_ang_vel (3)
      tip tracking, wrist frame (9T):
        delta_tip_pos (T×3), ref_tip_vel (T×3), delta_tip_vel (T×3)

    Reuses HandObservation.__init__ (joint/palm/body resolution, link7→virtual-wrist
    offsets, DEXENGINE_RIGHT_OBS_ZERO gating); only __call__ is replaced.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not self._is_fullrobot:
            raise RuntimeError(
                "[HandObservationV2] full-robot mode required — the floating-hand path is "
                "legacy and not supported by obs V2."
            )
        if not self._has_body_orientations:
            raise RuntimeError(
                "[HandObservationV2] body_positions.npz has no orientations — regenerate "
                "the reference (wrist quats/ang-vels are required)."
            )
        _fk = getattr(env.trajectory_manager, "fk_finger_indices", None)
        self._tip_indices = list(_fk["tips_list"]) if _fk else [4, 6, 8, 10, 12]
        self._grav_dir = torch.tensor([0.0, 0.0, -1.0], device=env.device)
        _F = self._n_indep - 6
        _W = 18 + 2 * _F + 21 + 9 * len(self._tip_indices)
        print(
            f"[HandObservationV2:{self._hand_name}] {_W}-dim "
            f"(proprio {18 + 2 * _F} + wrist 21 + tips {len(self._tip_indices) * 9}); frames: proprio=root, "
            f"tracking=wrist; rot=6D"
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        hand_name: str = "right_hand",
        link7_to_wrist_offset: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        asset = env.scene[self._hand_name]
        N = env.num_envs

        # ---- virtual wrist pose/vel from link7 (identical math to HandObservation) ----
        link7_pos_w = asset.data.body_link_pos_w[:, self._palm_body_id, :]
        link7_quat_w = asset.data.body_link_quat_w[:, self._palm_body_id, :]
        link7_lin_vel = asset.data.body_link_lin_vel_w[:, self._palm_body_id, :]
        link7_ang_vel = asset.data.body_link_ang_vel_w[:, self._palm_body_id, :]
        r_offset = self._link7_to_wrist_rot.unsqueeze(0).expand(N, -1).contiguous()
        vw_quat = quat_mul(link7_quat_w, r_offset)
        link7_rot = matrix_from_quat(link7_quat_w)
        t_col = self._link7_to_wrist_trans.view(1, 3, 1).expand(N, -1, -1)
        offset_w = torch.bmm(link7_rot, t_col).squeeze(-1)
        vw_pos = link7_pos_w + offset_w
        vw_lin_vel = link7_lin_vel + torch.cross(link7_ang_vel, offset_w, dim=-1)
        vw_ang_vel = link7_ang_vel
        vw_conj = quat_conjugate(vw_quat)

        return self._proprio_features(env, N, asset, vw_ang_vel, vw_conj, vw_lin_vel, vw_pos, vw_quat)

    def _proprio_features(self, env, N, asset, vw_ang_vel, vw_conj, vw_lin_vel, vw_pos, vw_quat):
        """Proprio (30): root-local q/dq + wrist-frame gravity."""
        root_q = self._floating_root_quat.unsqueeze(0).expand(N, -1).contiguous()
        root_conj = quat_conjugate(root_q)
        floating_root_pos = env.scene.env_origins.clone()
        floating_root_pos[:, 2] += self._floating_root_z_offset
        q_wrist_pos = quat_apply(root_conj, vw_pos - floating_root_pos)  # (N,3)
        q_wrist_rot6d = rot6d_from_quat(quat_mul(root_conj, vw_quat))  # (N,6)
        q_fingers = asset.data.joint_pos[:, self._indep_joint_ids]  # (N,6)
        dq_wrist_lin = quat_apply(root_conj, vw_lin_vel)  # (N,3)
        dq_wrist_ang = quat_apply(root_conj, vw_ang_vel)  # (N,3)
        dq_fingers = asset.data.joint_vel[:, self._indep_joint_ids]  # (N,6)
        gravity_wf = quat_apply(vw_conj, self._grav_dir.unsqueeze(0).expand(N, -1).contiguous())

        # ---- reference wrist tracking (21), wrist frame ----
        ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)
        cur_wrist_pos_l = vw_pos - env.scene.env_origins
        ref_palm_pos = ref_body[f"{self._side}_body_pos"][:, 0, :]
        ref_palm_vel = ref_body[f"{self._side}_body_vel"][:, 0, :]
        ref_palm_quat = ref_body[f"{self._side}_body_quat"][:, 0, :]
        ref_palm_ang = ref_body[f"{self._side}_body_ang_vel"][:, 0, :]
        delta_wrist_pos = quat_apply(vw_conj, ref_palm_pos - cur_wrist_pos_l)
        delta_wrist_rot6d = rot6d_from_quat(quat_mul(vw_conj, ref_palm_quat))
        ref_wrist_vel_wf = quat_apply(vw_conj, ref_palm_vel)
        delta_wrist_vel_wf = quat_apply(vw_conj, ref_palm_vel - vw_lin_vel)
        ref_wrist_ang_wf = quat_apply(vw_conj, ref_palm_ang)
        delta_wrist_ang_wf = quat_apply(vw_conj, ref_palm_ang - vw_ang_vel)

        # ---- tip tracking (45), wrist frame ----
        tip_ids = self._body_ids_tensor[self._tip_indices]
        cur_tip_pos = asset.data.body_link_pos_w[:, tip_ids, :] - env.scene.env_origins.unsqueeze(1)
        cur_tip_vel = asset.data.body_link_lin_vel_w[:, tip_ids, :]
        ref_tip_pos = ref_body[f"{self._side}_body_pos"][:, self._tip_indices, :]
        ref_tip_vel = ref_body[f"{self._side}_body_vel"][:, self._tip_indices, :]
        delta_tip_pos_wf = q_rot_vecs(vw_conj, ref_tip_pos - cur_tip_pos).reshape(N, -1)
        ref_tip_vel_wf = q_rot_vecs(vw_conj, ref_tip_vel).reshape(N, -1)
        delta_tip_vel_wf = q_rot_vecs(vw_conj, ref_tip_vel - cur_tip_vel).reshape(N, -1)

        out = torch.cat(
            [
                q_wrist_pos,
                q_wrist_rot6d,
                q_fingers,  # 15
                dq_wrist_lin,
                dq_wrist_ang,
                dq_fingers,  # 12
                gravity_wf,  # 3
                delta_wrist_pos,
                delta_wrist_rot6d,  # 9
                ref_wrist_vel_wf,
                delta_wrist_vel_wf,  # 6
                ref_wrist_ang_wf,
                delta_wrist_ang_wf,  # 6
                delta_tip_pos_wf,
                ref_tip_vel_wf,
                delta_tip_vel_wf,  # 45
            ],
            dim=-1,
        )  # (N, 96)

        return out
