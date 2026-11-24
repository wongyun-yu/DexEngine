"""IK-target side of the residual arm action: the IK target and finger command composed from the residual (or the
absolute action), then smoothed into the combined action buffer."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


class ResidualTargetMixin:
    """IK target and finger command of ResidualArmIKAction (the residual side is ResidualProcessMixin)."""

    def _process_ik_target_and_command(
        self,
        N,
        _pos_cap,
        _rot_cap_deg,
        abs_finger,
        arm_res,
        base,
        base_pos_delta,
        base_rot_quat,
        cur_pos_local,
        cur_quat_local,
        floating_root_pos,
        inner,
        residual_used,
        root_q,
        rs_finger,
        rs_wrist,
        target_pos_local,
        target_quat_local,
        wrist_pos_w,
        wrist_quat_w,
    ):
        """IK target EMA, the world -> link7 -> root conversion and the IK command (statements unchanged)."""
        return self._compose_combined_actions(
            N,
            _pos_cap,
            _rot_cap_deg,
            abs_finger,
            arm_res,
            base,
            base_pos_delta,
            base_rot_quat,
            cur_pos_local,
            cur_quat_local,
            floating_root_pos,
            inner,
            residual_used,
            root_q,
            rs_finger,
            rs_wrist,
            target_pos_local,
            target_quat_local,
            wrist_pos_w,
            wrist_quat_w,
        )

    def _compose_combined_actions(
        self,
        N,
        _pos_cap,
        _rot_cap_deg,
        abs_finger,
        arm_res,
        base,
        base_pos_delta,
        base_rot_quat,
        cur_pos_local,
        cur_quat_local,
        floating_root_pos,
        inner,
        residual_used,
        root_q,
        rs_finger,
        rs_wrist,
        target_pos_local,
        target_quat_local,
        wrist_pos_w,
        wrist_quat_w,
    ):
        """Compose the IK target and finger command into the combined action buffer."""
        abs_finger, target_pos_local, target_quat_local = self._absolute_or_residual_targets(
            N,
            _pos_cap,
            _rot_cap_deg,
            abs_finger,
            base,
            base_pos_delta,
            base_rot_quat,
            cur_pos_local,
            cur_quat_local,
            residual_used,
            rs_finger,
            target_pos_local,
            target_quat_local,
        )

        combined_finger = self._smooth_ik_target_and_fingers(
            abs_finger,
            base,
            floating_root_pos,
            inner,
            residual_used,
            root_q,
            rs_finger,
            target_pos_local,
            target_quat_local,
            wrist_pos_w,
            wrist_quat_w,
        )

        # Store combined for logging / monitoring
        if arm_res is not None:
            self._combined_actions[:, 0 : self._nw] = arm_res * self._arm_res_cap
            self._combined_actions[:, self._nw :] = combined_finger
            return
        self._combined_actions[:, 0:6] = base[:, 0:6] + residual_used[:, 0:6] * rs_wrist
        self._combined_actions[:, 6:] = combined_finger

    def _absolute_or_residual_targets(
        self,
        N,
        _pos_cap,
        _rot_cap_deg,
        abs_finger,
        base,
        base_pos_delta,
        base_rot_quat,
        cur_pos_local,
        cur_quat_local,
        residual_used,
        rs_finger,
        target_pos_local,
        target_quat_local,
    ):
        """Wrist and finger targets in absolute-action mode, or from the residual."""
        if self._action_abs:  # full-motion action (see __init__): the target ignores the residual path above
            import math

            box = self._abs_box  # built in __init__ (round trip checked there)
            ref_pos_local = cur_pos_local + base_pos_delta
            ref_quat_local = math_utils.quat_mul(cur_quat_local, base_rot_quat)
            ac = residual_used.clamp(-1.0, 1.0)
            abs_pos = box["anchor"] + ac[:, 0:3] * box["half"]
            abs_quat = math_utils.quat_mul(
                box["quat"].unsqueeze(0).expand(N, -1), self._quat_from_rotvec(ac[:, 3:6] * math.pi)
            )
            abs_finger = ac[:, 6:]
            rot_cap = _rot_cap_deg * 0.017453292519943295
            imp = torch.empty_like(residual_used)
            imp[:, 0:3] = (box["anchor"] + residual_used[:, 0:3] * box["half"] - ref_pos_local) / _pos_cap
            dq = math_utils.quat_mul(math_utils.quat_conjugate(ref_quat_local), abs_quat)
            imp[:, 3:6] = (
                math_utils.axis_angle_from_quat(dq) + (residual_used[:, 3:6] - ac[:, 3:6]) * math.pi
            ) / rot_cap
            imp[:, 6:] = (residual_used[:, 6:] - base[:, 6:]) / rs_finger
            if self._parked is not None:  # a parked side holds its reference (implied residual 0)
                pk = self._parked.unsqueeze(-1)
                abs_pos = torch.where(pk, ref_pos_local, abs_pos)
                abs_quat = torch.where(pk, ref_quat_local, abs_quat)
                abs_finger = torch.where(pk, base[:, 6:], abs_finger)
                imp = torch.where(pk, torch.zeros_like(imp), imp)
            self._implied_prev[:] = self._implied_cur
            self._implied_cur[:] = imp
            self._raw_actions[:] = imp
            target_pos_local, target_quat_local = abs_pos, abs_quat
            if not self._abs_logged:  # proof the full-motion path ran: the implied residual of the first action
                self._abs_logged = True
                m = imp.abs().mean(0)
                print(
                    f"[ACTION_ABS] first step side={self._side_rb} N={N} mean|implied| "
                    f"wrist_pos={float(m[0:3].mean()):.3f} "
                    f"wrist_rot={float(m[3:6].mean()):.3f} finger={float(m[6:].mean()):.3f} (cap units)",
                    flush=True,
                )
        return abs_finger, target_pos_local, target_quat_local

    def _smooth_ik_target_and_fingers(
        self,
        abs_finger,
        base,
        floating_root_pos,
        inner,
        residual_used,
        root_q,
        rs_finger,
        target_pos_local,
        target_quat_local,
        wrist_pos_w,
        wrist_quat_w,
    ):
        """IK target EMA (disabled by default in current arm IK config)."""
        if inner._ik_target_ema < 1.0:
            ema = inner._ik_target_ema
            target_pos_local = ema * target_pos_local + (1.0 - ema) * inner._prev_wrist_pos_local
            prev_q = inner._prev_wrist_quat_local
            dot = (target_quat_local * prev_q).sum(dim=-1, keepdim=True)
            prev_q = torch.where(dot < 0, -prev_q, prev_q)
            target_quat_local = ema * target_quat_local + (1.0 - ema) * prev_q
            target_quat_local = target_quat_local / target_quat_local.norm(dim=-1, keepdim=True)
        inner._prev_wrist_pos_local[:] = target_pos_local
        inner._prev_wrist_quat_local[:] = target_quat_local

        # ============================================================
        # Convert to world frame → link7 → robot root frame → IK command
        # ============================================================
        wrist_pos_w = floating_root_pos + math_utils.quat_apply(root_q, target_pos_local)
        wrist_quat_w = math_utils.quat_mul(root_q, target_quat_local)

        if getattr(inner, "takes_world_wrist_target", False):
            # Floating hand: hands the same world target to the wrist wrench term (no IK command).
            inner.set_wrist_target_w(wrist_pos_w, wrist_quat_w)
        else:
            link7_pos_w_t, link7_quat_w_t = inner._wrist_to_link7_batch(wrist_pos_w, wrist_quat_w)

            root_pos_w = inner._asset.data.root_pos_w
            root_quat_w = inner._asset.data.root_quat_w
            link7_pos_b, link7_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, link7_pos_w_t, link7_quat_w_t
            )

            inner._ik_commands[:, :3] = link7_pos_b
            inner._ik_commands[:, 3:7] = link7_quat_b
            ee_pos_b, ee_quat_b = inner._compute_ee_pose_in_root()
            inner._ik_controller.set_command(inner._ik_commands, ee_pos_b, ee_quat_b)

        # ============================================================
        # Fingers: base + residual*rs_finger (affine ManipTrans scaling, order-safe)
        # Clamp combined to [-1, 1] so the inner term maps into valid joint limits.
        # ============================================================
        combined_finger = (
            torch.clamp(base[:, 6:] + residual_used[:, 6:] * rs_finger, -1.0, 1.0) if abs_finger is None else abs_finger
        )

        # ------------------------------------------------------------
        # Pre-contact finger override: force fingers to "open" action
        # until the palm approaches the object. Once triggered per-env,
        # the override releases permanently until the next reset.
        # ------------------------------------------------------------
        if self._pre_contact_enabled:
            palm_pos_w = inner._asset.data.body_pos_w[:, inner._ee_body_idx]  # (N, 3)
            obj_pos_w = self._env.scene[self._obj_scene_key].data.root_pos_w  # (N, 3)
            palm_to_obj = torch.norm(palm_pos_w - obj_pos_w, dim=-1)  # (N,)

            # Latch: still-active envs remain active only while palm is far
            still_far = palm_to_obj > self._pre_contact_release_dist
            self._pre_contact_active &= still_far

            if self._pre_contact_active.any():
                active_mask = self._pre_contact_active.unsqueeze(-1)  # (N, 1)
                open_val = torch.full_like(combined_finger, self._pre_contact_open_action)
                combined_finger = torch.where(active_mask, open_val, combined_finger)

        inner._processed_actions[:, 6:] = combined_finger
        return combined_finger
