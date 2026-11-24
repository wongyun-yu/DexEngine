"""Per-step state of the residual action: effort anneal, reference frames, parking and reset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    pass


class ResidualStateMixin:
    """Methods of :class:`ResidualArmIKAction` (per-step state of the residual action: effort anneal, reference frames,
    parking and reset)."""

    def _init_effort_anneal(self) -> None:
        """Prepare the ramp from the declared build cap (cfg.finger_effort_build) to the hardware cap
        (cfg.finger_effort_hw, owner = HandSpec). Fails when the build cap is not above the hardware cap (a curriculum
        with nothing to lower is a setup error). The initial PhysX cap must be the build cap when the builder loaded
        it, the hardware cap when it did not (the eval path's default); anything else fails."""
        import re as _re

        inner = self._inner
        ids = [int(i) for i in inner._finger_joint_ids]
        names = list(inner._finger_joint_names)
        hw = self.cfg.finger_effort_hw
        build, applied = self.cfg.finger_effort_build, self.cfg.finger_effort_build_applied
        if hw is None or build is None or applied is None:
            raise RuntimeError(
                "[EFFORT_ANNEAL] finger_effort_hw·finger_effort_build·finger_effort_build_applied "
                "has an empty value "
                "— the action contract (apply_action_contract) fills them"
            )

        def _cap(n):
            if not isinstance(hw, dict):
                return float(hw)
            hit = [float(v) for k, v in hw.items() if _re.search(k, n)]
            if len(hit) != 1:
                raise RuntimeError(f"[EFFORT_ANNEAL] {n}: {len(hit)} hardware cap regexes match (must be one)")
            return hit[0]

        phys = inner._asset.data.joint_effort_limits[0, ids].clone()
        start = torch.full_like(phys, float(build))
        end = torch.tensor([_cap(n) for n in names], device=phys.device, dtype=phys.dtype)
        if not bool((start > end).all()):
            raise RuntimeError(
                f"[EFFORT_ANNEAL] the build cap must be above the hardware cap: "
                f"start={start.tolist()} hw={end.tolist()}"
            )
        expect = start if applied else end
        if not torch.allclose(phys, expect, rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                f"[EFFORT_ANNEAL] initial PhysX cap {phys.tolist()} ≠ expected "
                f"{'build' if applied else 'hardware'} cap {expect.tolist()} — "
                "the builder loaded a cap other than the declared one"
            )
        self._eff_anneal = {
            "ids": ids,
            "start": start,
            "end": end,
            "p0": float(self.cfg.finger_effort_anneal_start),
            "w": None,
            "band": None,
        }
        print(
            f"[EFFORT_ANNEAL] side={self._side_rb} joints={len(ids)} "
            f"start={start.min().item():.3f}~{start.max().item():.3f} → "
            f"hw={end.min().item():.3f}~{end.max().item():.3f} N·m @progress {self._eff_anneal['p0']}→1.0 "
            f"(eval = hardware cap) · build cap "
            f"{'loaded' if applied else 'not loaded (PhysX = hardware cap, progress 1.0 path)'}",
            flush=True,
        )

    def _effort_anneal_step(self) -> None:
        """schedule_progress -> ramp weight w (0 before p0, 1 at 1.0) -> writes the cap to PhysX. Writes only when w
        moved by more than 0.01 or newly reaches an end (0 or 1), entering and leaving frozen evaluation included.
        Progress 0 is never read as 1 (no `or 1.0`)."""
        a = self._eff_anneal
        p = getattr(getattr(self._env, "trajectory_manager", None), "schedule_progress", None)
        p = 1.0 if p is None else float(p)
        w = 0.0 if p <= a["p0"] else min(1.0, (p - a["p0"]) / (1.0 - a["p0"]))
        last = a["w"]
        if last is not None and abs(w - last) < 0.01 and not (w in (0.0, 1.0) and w != last):
            return
        cap = a["start"] + (a["end"] - a["start"]) * w
        asset = self._inner._asset
        asset.write_joint_effort_limit_to_sim(
            cap.unsqueeze(0).expand(asset.num_instances, -1).contiguous(), joint_ids=a["ids"]
        )
        a["w"] = w
        band = int(w * 10)
        if band != a["band"]:
            a["band"] = band
            print(
                f"[EFFORT_ANNEAL] side={self._side_rb} progress={p:.3f} w={w:.2f} "
                f"cap={cap.min().item():.3f}~{cap.max().item():.3f} N·m",
                flush=True,
            )

    def legacy_dims(self, dims):
        """Map a penalty's legacy (wrist 0..6 | finger 6..6+n) slice onto this term's layout. Identity unless
        DEXENGINE_ARM_JOINT_RESIDUAL: any slice inside the wrist block -> the whole arm block (0, nw); a finger slice
        (a, b) with a >= 6 -> (a - 6 + nw, b - 6 + nw). A slice across the boundary dies (no such penalty exists)."""
        if not getattr(self, "_arm_joint_res", False) or dims is None:
            return dims
        a, b = int(dims[0]), int(dims[1])
        if b <= 6:
            out = (0, self._nw)
        elif a >= 6:
            out = (a - 6 + self._nw, b - 6 + self._nw)
        else:
            raise RuntimeError(f"[ARM_JOINT_RESIDUAL] penalty dims {dims} cross the wrist/finger boundary")
        if not self._arm_dims_logged:
            self._arm_dims_logged = True
            print(
                f"[ARM_JOINT_RESIDUAL] penalty dims side={self._side_rb}: legacy {tuple(dims)} -> {out} "
                f"(wrist block -> arm block 0:{self._nw}, fingers shifted by {self._nw - 6})",
                flush=True,
            )
        return out

    def _arm_ref_q(self, frame_steps: torch.Tensor) -> torch.Tensor:
        """Reference arm joints (N, nw) at policy-step frames `frame_steps`, from the all-frame arm IK cache of this
        side
        (column order = env._arm_ik_joint_ids_<side>). Dies when the cache or the ids are missing."""
        env, sd = self._env, self._side_rb
        cache_all = getattr(env, f"_arm_ik_cache_all_{sd}", None)
        cache_len_t = getattr(env, f"_arm_ik_cache_all_len_{sd}", None)
        ids = getattr(env, f"_arm_ik_joint_ids_{sd}", None)
        if cache_all is None or cache_len_t is None or not ids or cache_all.shape[-1] != self._nw:
            raise RuntimeError(
                f"[ARM_JOINT_RESIDUAL] side={sd} needs the all-frame arm IK cache "
                f"(_arm_ik_cache_all_{sd}: {None if cache_all is None else tuple(cache_all.shape)}, "
                f"ids {ids}, arm dims {self._nw})"
            )
        if not getattr(self, "_arm_ids_checked", False):  # cache columns = env ids = this term's own arm joints
            own = [int(i) for i in self._inner._arm_joint_ids]
            if [int(i) for i in ids] != own:
                raise RuntimeError(
                    f"[ARM_JOINT_RESIDUAL] side={sd} env arm ids {list(ids)} != this term's arm joint "
                    f"ids {own} (asset {self._inner._asset.cfg.prim_path})"
                )
            self._arm_ids_checked = True
            print(
                f"[ARM_JOINT_RESIDUAL] ids side={sd} {own} = {[self._inner._asset.joint_names[i] for i in own]}",
                flush=True,
            )
        tm = self._traj_mgr
        clip_ids = tm.clip_ids
        f = torch.minimum(tm._abs_frames(frame_steps), cache_len_t[clip_ids] - 1).clamp(min=0)
        return cache_all[clip_ids, f]

    def arm_joint_obs(self) -> torch.Tensor:
        """(N, 2·nw) policy observation under DEXENGINE_ARM_JOINT_RESIDUAL: arm joints minus the reference arm joints of
        the current frame, then arm joint velocities. Zeros before the arm cache exists (observation-manager shape probe
        at env construction); after the first action step a missing cache dies in _arm_ref_q."""
        N = self.num_envs
        ids = getattr(self._env, f"_arm_ik_joint_ids_{self._side_rb}", None)
        if self._arm_q_target is None and (
            ids is None or getattr(self._env, f"_arm_ik_cache_all_{self._side_rb}", None) is None
        ):
            return torch.zeros(N, 2 * self._nw, device=self.device)
        robot = self._inner._asset
        q = robot.data.joint_pos[:, ids]
        qd = robot.data.joint_vel[:, ids]
        return torch.cat([q - self._arm_ref_q(self._env.episode_length_buf), qd], dim=-1)

    def _park_still(self, *refs: torch.Tensor) -> tuple:
        """A parked env's reference does not move: zero the feed-forward reference derivatives (velocity,
        acceleration)."""
        if self._parked is None:
            return refs
        keep = (~self._parked).unsqueeze(-1).to(refs[0].dtype)
        return tuple(r * keep for r in refs)

    def _wrist_frame_ids(self) -> torch.Tensor:
        """Wrist reference frame (policy steps) = episode clock + lead (0 for parked envs).

        WRIST LOOKAHEAD: +k trajectory frames in POLICY-STEP units (x slowdown). get_wrist_* does NOT clamp per-clip
        (it indexes the padded table directly — unlike _abs_frames), so clamp HERE in policy-step units:
        floor(step/sd)+start <= clip_len-1. This also makes the last k frames all target the final frame."""
        frame_ids = self._env.episode_length_buf
        self.parked_seen = self._parked
        if self._wrist_lookahead > 0:
            _sd_la = float(getattr(self._traj_mgr, "_slowdown_factor", 1.0) or 1.0)
            _max_step_la = (
                (self._traj_mgr.clip_lengths - self._traj_mgr.start_frames - 1).clamp(min=0).float() * _sd_la
            ).long()
            frame_ids = torch.minimum(frame_ids + self._lead(int(round(self._wrist_lookahead * _sd_la))), _max_step_la)
        return frame_ids

    def _virtual_wrist_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World pose of the actual virtual wrist (link7 + wrist offset) (pos (N,3), quat wxyz (N,4))."""
        inner = self._inner
        link7_pos_w = inner._asset.data.body_pos_w[:, inner._ee_body_idx]
        link7_quat_w = inner._asset.data.body_quat_w[:, inner._ee_body_idx]
        N = link7_pos_w.shape[0]
        r_offset = inner._r_offset_fwd_wxyz.unsqueeze(0).expand(N, -1)
        link7_rot = math_utils.matrix_from_quat(link7_quat_w)
        t_col = inner._link7_to_wrist_trans.view(1, 3, 1).expand(N, -1, -1)
        return link7_pos_w + torch.bmm(link7_rot, t_col).squeeze(-1), math_utils.quat_mul(link7_quat_w, r_offset)

    def parked_wrist_offset(self, ids: torch.Tensor) -> torch.Tensor:
        """Actual virtual wrist of env ids − reference wrist at frame 0 (clock 0 = the parked target), (k, 3) [m]:
        the comparison of dexengine.rl.judge.park's [PARK]."""
        pos, _ = self._virtual_wrist_w()
        tm = self._env.trajectory_manager
        ref = tm.get_wrist_pos_w(torch.zeros_like(self._env.episode_length_buf), side=self._side_rb)
        return pos[ids] - ref[ids] - self._env.scene.env_origins[ids]

    def apply_actions(self) -> None:
        """Delegate to inner action term for simulation writes."""
        self._inner.apply_actions()
        if self._arm_joint_res and self._arm_q_target is not None:  # arm joint target replaces the IK solution
            self._inner._asset.set_joint_position_target(
                self._arm_q_target, joint_ids=getattr(self._env, f"_arm_ik_joint_ids_{self._side_rb}")
            )
        if self._arm_ff:
            self._apply_arm_feedforward()
        self._accum_arm_jerk()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action buffers and delegate to inner action term."""
        # Reset inner action term (handles arm targets, finger targets)
        self._inner.reset(env_ids)

        # Reset our own buffers
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._implied_cur[:] = 0.0
            self._implied_prev[:] = 0.0
            self._smoothed_residual_actions[:] = 0.0
            self._combined_actions[:] = 0.0
            self._base_action_buffer[:] = 0.0
            if self._pre_contact_enabled:
                self._pre_contact_active[:] = True
        else:
            env_ids_t = (
                torch.tensor(env_ids, device=self.device, dtype=torch.long)
                if not isinstance(env_ids, torch.Tensor)
                else env_ids
            )
            self._raw_actions[env_ids_t] = 0.0
            self._implied_cur[env_ids_t] = 0.0
            self._implied_prev[env_ids_t] = 0.0
            self._smoothed_residual_actions[env_ids_t] = 0.0
            self._combined_actions[env_ids_t] = 0.0
            self._base_action_buffer[env_ids_t] = 0.0
            if self._pre_contact_enabled:
                self._pre_contact_active[env_ids_t] = True
