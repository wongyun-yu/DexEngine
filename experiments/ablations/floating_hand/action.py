"""Inner action term of the floating hand: a WRIST WRENCH (or virtual-joint targets) in place of the arm IK; the fingers
follow the arm body's path.

The outer term (``ResidualArmIKAction``) builds the wrist WORLD TARGET from the reference wrist + residual exactly as on
the arm body. The arm body passes that target to IK; this term receives it through ``set_wrist_target_w`` and, every
physics step, applies ``body.wrist_wrench`` (SE(3) spring-damper · FlashCHORD gains · load observer ``LoadObserver``)
to the hand root body's centre of mass as an external wrench — or, with the joint drive (``DEXENGINE_FH_WRIST=joint``),
sets the six virtual wrist joint targets. Fingers, couplings and recording are the parent's
(``ArmIKFingersPositionAction._apply_fingers``). Values and rules live in body.py; this file is the Isaac wiring.
Isaac Lab only (imports torch and the arm action term).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from dexengine.core.arms import active_hand_name
from dexengine.rl.tasks.mdp.actions.arm_ik import ArmIKFingersPositionAction
from experiments.ablations.floating_hand import body as _fh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: diagnostic env flags of an earlier version of this term (probe lines, injected disturbance, model mass scale, Schur
#: inertia). They were never set by any launcher and are gone; a run that still sets one dies instead of running without
#: it.
REMOVED_DIAG_ENVS = (
    "DEXENGINE_FH_PROBE",
    "DEXENGINE_FH_DISTURB",
    "DEXENGINE_FH_MODEL_SCALE",
    "DEXENGINE_FH_MODEL_INERTIA",
)


class FloatingWristFingersAction(ArmIKFingersPositionAction):
    """No arm joints · hand root = free root body = wrist. Wrist = external wrench (or virtual joints), fingers =
    position PD (parent)."""

    EXPECTED_ARM_JOINTS = 0
    #: tells the outer term (ResidualArmIKAction) to pass a world target instead of an IK command
    takes_world_wrist_target = True
    #: physics steps between the saturation / gimbal statistics lines
    _STAT_EVERY = 2000

    def _check_virtual_joints(self, want):
        """Joint drive: the spawned virtual joints, their limits and armature equal the rule (fix_base anchor)."""
        if not self._asset.is_fixed_base:
            raise RuntimeError(
                f"{_fh.MARK} {self._side}: {_fh.WRIST_DRIVE_ENV}=joint but the articulation is not "
                "fix_base — the virtual-joint body (root = world-fixed anchor) was not spawned."
            )
        self._vj_ids, vj_names = self._asset.find_joints(_fh.vj_joint_names(self._side), preserve_order=True)
        if list(vj_names) != _fh.vj_joint_names(self._side):
            raise RuntimeError(f"{_fh.MARK} {self._side}: virtual joints {vj_names} ≠ {_fh.vj_joint_names(self._side)}")
        if (
            self._asset.body_names[self._ee_body_idx] != want
            or self._asset.body_names[0] != _fh.vj_link_names(self._side)[0]
        ):
            raise RuntimeError(
                f"{_fh.MARK} {self._side}: wrist body is not '{want}' or the root anchor is not "
                f"'{_fh.vj_link_names(self._side)[0]}' — ee="
                f"{self._asset.body_names[self._ee_body_idx]!r} · body "
                f"0={self._asset.body_names[0]!r}"
            )
        # spawned virtual joint limits = the rule (body.VJ_*) — the source URDF check
        # (dexengine.rl.tasks.scene.asset_checks) has no virtual joints
        lim = self._asset.data.joint_pos_limits[0, self._vj_ids].cpu()
        want = torch.tensor(
            [[-_fh.VJ_LIN_LIMIT_M, _fh.VJ_LIN_LIMIT_M]] * 3
            + [
                [-_fh.VJ_ROT_OUTER_LIMIT, _fh.VJ_ROT_OUTER_LIMIT],
                [-_fh.VJ_ROT_MID_LIMIT, _fh.VJ_ROT_MID_LIMIT],
                [-_fh.VJ_ROT_OUTER_LIMIT, _fh.VJ_ROT_OUTER_LIMIT],
            ]
        )
        if float((lim - want).abs().max()) > 1e-3:
            raise RuntimeError(
                f"{_fh.MARK} {self._side}: spawned virtual joint limits {lim.tolist()} ≠ rule {want.tolist()}"
            )
        # spawned virtual joint armature = the rule (body.vj_armature) — without it the light-link defect returns
        arm_s = self._asset.data.joint_armature[0, self._vj_ids].cpu()
        arm_w = torch.tensor([_fh.vj_armature((self._side,))[n] for n in _fh.vj_joint_names(self._side)])
        if float((arm_s - arm_w).abs().max()) > 1e-6:
            raise RuntimeError(
                f"{_fh.MARK} {self._side}: spawned virtual joint armature {arm_s.tolist()} ≠ rule "
                f"{arm_w.tolist()} ({_fh.VJ_ARMATURE_RULE})"
            )
        # parsed by the floating job proofs (fh_proof) and its training-physics check
        print(f"{_fh.MARK} {self._side} {_fh.vj_armature_mark()} · spawn values check passed", flush=True)
        return vj_names, want

    def _setup_anchor(self):
        """Joint drive: prismatic limit (anchor rule) and the gimbal / step counters."""
        self._lin_lim = _fh.vj_lin_limit()
        if _fh.anchor_rule():
            n_env = self._asset.num_instances
            lims = torch.tensor([-self._lin_lim, self._lin_lim], device=self.device).repeat(n_env, 3, 1)
            self._asset.write_joint_position_limit_to_sim(lims, joint_ids=list(self._vj_ids[:3]))
            got = self._asset.data.joint_pos_limits[:, self._vj_ids[:3]]
            if float((got - lims).abs().max()) > 1e-5:
                raise RuntimeError(
                    f"{_fh.MARK} {self._side}: prismatic limit write failed — read {got[0].tolist()} ≠ ±{self._lin_lim}"
                )
            print(
                f"[FH_ANCHOR] {self._side} on rule={_fh.anchor_rule()} · prismatic limit ±{self._lin_lim:g} m "
                "written · read back and checked",
                flush=True,
            )
        self._n_gimbal = 0
        self._n_steps_j = 0

    def _check_free_root(self, want):
        """Wrench drive: the articulation is a free root whose body 0 is the wrist."""
        if self._asset.is_fixed_base:
            raise RuntimeError(
                f"{_fh.MARK} {self._side}: the articulation is fix_base — the floating-hand USD is "
                "baked with fix_base False (repository bake rule). Another asset is spawned."
            )
        if self._asset.body_names[self._ee_body_idx] != want or self._ee_body_idx != 0:
            raise RuntimeError(
                f"{_fh.MARK} {self._side}: the wrist body is not the root '{want}' (body 0) — "
                f"ee={self._asset.body_names[self._ee_body_idx]!r} (idx {self._ee_body_idx})."
            )

    def _init_wrench_state(self, env):
        """Controller state: masses, inertia, targets, root ids, load observer."""
        d = self._asset.data
        self._mass = d.default_mass.to(self.device)  # (N,B)
        self._inertia_b = d.default_inertia.to(self.device).view(self.num_envs, -1, 3, 3)
        self._dt = float(env.physics_dt)
        self._tgt_pos_w = d.body_link_pos_w[:, self._hand_idx].clone()
        self._tgt_quat_w = d.body_link_quat_w[:, self._hand_idx].clone()
        self._no_arm = torch.zeros(self.num_envs, 0, device=self.device)
        # root body index as one device tensor (a list [0] makes the composer build and copy a device array every step)
        self._root_ids = torch.full((1,), self._hand_idx, dtype=torch.int32, device=self.device)
        self._n_apply = 0
        self._sat_f = 0.0
        m_tot = float(self._mass[0].sum())
        # load observer (body.LoadObserver) · controller mass and inertia = the root body's own
        # (MASS_RULE · DAMP_INERTIA_RULE — lower bounds of the effective inertia = the observer's convergence
        # condition)
        self._obs = _fh.LoadObserver(self.num_envs, self.device)
        self._m_root = self._mass[:, 0]
        return m_tot

    def __init__(self, cfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        _fh.require_supported(active_hand_name())
        gone = [f"{k}={os.environ[k]}" for k in REMOVED_DIAG_ENVS if os.environ.get(k, "") not in ("", "0")]
        if gone:
            raise RuntimeError(f"{_fh.MARK} removed diagnostic flags are set: {gone} — unset them")
        #: joint drive (DEXENGINE_FH_WRIST=joint, off by default): the wrist = 6 virtual joints driven implicitly by
        #: PhysX — no wrench, no load observer
        self._joint = _fh.joint_drive()
        want = _fh.root_body(active_hand_name(), self._side)
        if self._joint:
            vj_names, want = self._check_virtual_joints(want)
            # anchor rule (body.anchor_rule, off by default): on → write the prismatic limits ±VJ_ANCHOR_LIN_LIMIT_M
            # at runtime (URDF and USD unchanged) · read back · marker ('on', parsed by fh_proof)
            self._setup_anchor()
        else:
            self._check_free_root(want)
        #: hand root body index — 0 by default (free root) · joint drive = below the virtual chain (the anchor is 0)
        self._hand_idx = self._ee_body_idx if self._joint else 0
        if (
            float(self._t_offset.abs().max()) > 0.0
            or float((self._r_offset_fwd_wxyz - self._r_offset_fwd_wxyz.new_tensor([1.0, 0, 0, 0])).abs().max()) > 0.0
        ):
            raise RuntimeError(
                f"{_fh.MARK} {self._side}: the wrist offset is not the identity (t={self._t_offset.tolist()} "
                f"r={self._r_offset_fwd_wxyz.tolist()}) — "
                f"the mounts of ({_fh.FLOATING_ARM}, hand) are the identity."
            )
        m_tot = self._init_wrench_state(env)
        print(
            f"{_fh.describe()} · side={self._side} root='{want}' m={m_tot:.3f} kg "
            f"(m·g {m_tot * 9.81:.2f} N) dt={self._dt:g} s · gain source: {_fh.WRIST_GAINS.source}",
            flush=True,
        )
        if self._joint:
            print(
                f"[FH_WRIST] joint side={self._side} · virtual joints {list(vj_names)} (index {list(self._vj_ids)}) · "
                f"hand root body {self._hand_idx} · wrench and load observer off · target = anchor-frame translation + "
                "Y·X·Z angles (nearest branch)",
                flush=True,
            )

    # -- called by the outer term ------------------------------------------------------------------------------------
    def set_wrist_target_w(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Wrist world target (reference + residual — built by the outer term as on the arm body)."""
        self._tgt_pos_w[:] = pos_w
        self._tgt_quat_w[:] = quat_w

    # -- every physics step ------------------------------------------------------------------------------------------
    def _apply_arm(self) -> torch.Tensor:
        if self._joint:
            return self._apply_arm_joint()
        d = self._asset.data
        pos = d.body_link_pos_w[:, 0]
        quat = d.body_link_quat_w[:, 0]
        # linear mass · rotational inertia = the root body's own (MASS_RULE · DAMP_INERTIA_RULE) — the whole inertia
        # diverges; a lower bound is what lets the load observer converge
        inertia_w = _fh.damping_inertia_w(quat, self._inertia_b[:, 0])
        v, w = d.body_link_lin_vel_w[:, 0], d.body_link_ang_vel_w[:, 0]
        dF, dT = self._obs.estimate(v, w, inertia_w, self._m_root, self._dt)
        F, tau_c = _fh.wrist_wrench(
            pos,
            quat,
            v,
            w,
            self._tgt_pos_w,
            self._tgt_quat_w,
            inertia_w,
            self._m_root,
            self._dt,
            d_force=dF,
            d_torque=dT,
        )
        self._obs.commit(v, w, F, tau_c)
        tau = tau_c  # applied at the centre of mass (wrist_wrench) — no lever arm torque
        # link axes with is_global=False — world axes would be turned by the pose the composer cached at reset
        F_b, tau_b = _fh.wrench_to_link_frame(F, tau_c, quat)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=F_b.unsqueeze(1), torques=tau_b.unsqueeze(1), body_ids=self._root_ids, is_global=False
        )
        self._n_apply += 1
        g = _fh.WRIST_GAINS
        # accumulated on the device — a float() every physics step would sync GPU → CPU each step (read at stat lines
        # only)
        self._sat_f = self._sat_f + (F.abs() >= g.lin_force_max - 1e-6).any(-1).float().mean()
        if self._n_apply % self._STAT_EVERY == 0:
            print(
                f"{_fh.MARK} {self._side} wrench {self._n_apply} steps · mean share of envs at the force limit "
                f"{float(self._sat_f) / self._STAT_EVERY:.3f} · |F| p50 {float(F.norm(dim=-1).median()):.2f} N · "
                f"|τ| p50 {float(tau.norm(dim=-1).median()):.3f} N·m",
                flush=True,
            )
            self._sat_f = 0.0
        return self._no_arm

    def _apply_arm_joint(self) -> torch.Tensor:
        """Joint drive: hand-root world target → virtual joint targets (anchor frame) — PhysX solves the implicit PD
        in
        the same step with the true inertia."""
        d = self._asset.data
        q_cur = d.joint_pos[:, self._vj_ids]
        tgt = _fh.vj_targets(
            d.body_link_pos_w[:, 0], d.body_link_quat_w[:, 0], self._tgt_pos_w, self._tgt_quat_w, q_cur, self._lin_lim
        )
        self._asset.set_joint_position_target(tgt, joint_ids=self._vj_ids)
        # steps near the gimbal (middle joint |X| > 75°) — accumulated on the device, read at stat lines only
        self._n_gimbal = self._n_gimbal + (tgt[:, 4].abs() > _fh.VJ_GIMBAL_WARN).float().sum()
        self._n_steps_j += 1
        if self._n_steps_j % self._STAT_EVERY == 0:
            print(
                f"[FH_WRIST] {self._side} {self._n_steps_j} physics steps · env-steps near the gimbal (|X| > 75°) "
                f"{int(self._n_gimbal)} · virtual joint |target − current| p50 linear "
                f"{float((tgt[:, :3] - q_cur[:, :3]).norm(dim=-1).median()) * 1000:.1f} mm · rotation "
                f"{float((tgt[:, 3:] - q_cur[:, 3:]).norm(dim=-1).median()) * 57.29578:.2f}°",
                flush=True,
            )
            self._n_gimbal = 0
        return self._no_arm

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._obs.reset(ids)  # the reset wrote the velocity — no load from the last step's record
        # until the next process_actions the target is where the reset put the root, so the physics steps right after a
        # reset do not pull toward a stale target. With the joint drive the anchor (body 0) = the reset pose (joints 0).
        self._tgt_pos_w[ids] = self._asset.data.body_link_pos_w[ids, 0]
        self._tgt_quat_w[ids] = self._asset.data.body_link_quat_w[ids, 0]
