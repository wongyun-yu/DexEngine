"""Object access of the env: reference frames, held objects, the pinned pose and the hand-object contact force."""

from __future__ import annotations

import torch

from dexengine.rl.swallow import note_once


class ObjectPinMixin:
    """Methods of :class:`ArmResidualTrainEnv` (object access of the env: reference frames, held objects, the pinned
    pose and the hand-object contact force)."""

    def reference_frames(self, ids: torch.Tensor) -> torch.Tensor:
        """Current reference frame of `ids` = start_frames + episode_length_buf, clamped to the loaded clip length.
        clip_ids are global indices, so a render/eval that loaded a clip subset can exceed clip_lengths; indexing with
        them would die on a CUDA device-side assert at an unrelated line, so they clamp to the last clip."""
        tm = self.trajectory_manager
        fr = (tm.start_frames[ids] + self.episode_length_buf[ids]).clamp(min=0)
        ci = tm.clip_ids[ids].clamp(max=int(tm.clip_lengths.numel()) - 1)
        return torch.minimum(fr, tm.clip_lengths[ci] - 1)

    def held_object_assets(self) -> list:
        """(side, object asset) list: the single owner for the reference pose pin (SPAWN_HOLD, judge park) and the
        judge free-fall gate. An idle side (rule D16) has no object and is left out."""
        from dexengine.rl.tasks.scene.collision_bodies import scene_object_key as _sok

        shared = bool(getattr(self.cfg, "shared_object", False))
        idle = str(getattr(self.cfg, "idle_side", "") or "")
        out = []
        for side, key in (
            ("right", "object_right" if idle != "right" else None),
            ("left", _sok(self.scene.keys(), "left", shared, idle)),
        ):
            if key is None:
                continue
            obj = self.scene.get(key) if hasattr(self.scene, "get") else None
            if obj is None:
                try:
                    obj = self.scene[key]
                except KeyError as _exc:
                    note_once("ENV_OBJECTS scene key", _exc)
                    continue
            out.append((side, obj))
        return out

    def object_pin_pose(
        self, ids: torch.Tensor, frames: torch.Tensor, side: str, obj, ref: dict | None = None
    ) -> tuple:
        """(pos env-local · quat wxyz · lift · lift per source): the pinned object pose. Env window onto the owner
        `object_pin.pin_pose` (the judge park oracle dexengine.rl.judge.park calls it with the env only)."""
        from dexengine.rl.tasks.scene.object_pin import pin_pose as _pin_pose

        return _pin_pose(self, ids, frames, side, obj, ref)

    def object_support_gap(self, ids: torch.Tensor, frames: torch.Tensor, side: str, obj, ref: dict | None = None):
        """(n,) m: support gap of the pinned object pose. Env window onto the owner `object_pin.support_gap`
        (judge gate D calls it with the env only)."""
        from dexengine.rl.tasks.scene.object_pin import support_gap as _support_gap

        return _support_gap(self, ids, frames, side, obj, ref)

    def hand_object_contact_force(self, side: str):
        """PhysX filtered contact force between side `side`'s object and both hands' links (per hand (n_envs,) max
        force [N]) plus the own fingertip sensor contact: the judge park's 'start touch' measure. Env window onto the
        owner `hand_object_contact.measure` (read only; training is unchanged)."""
        from dexengine.rl.tasks.mdp.hand_object_contact import measure as _measure

        return _measure(self, side)

    def object_pin_joints(self, ids: torch.Tensor, side: str, obj, ref: dict) -> int:
        """Articulated object joint state = (reference θ, ω 0): env window onto the owner `spawn_hold.pin_joints` (the
        same function as reset). Returns the number of joints written."""
        from dexengine.rl.tasks.scene.spawn_hold import pin_joints as _pin_joints

        return _pin_joints(self, ids, side, obj, ref)

    def pin_objects_to_reference(self, ids: torch.Tensor, *, joints: bool) -> dict:
        """Write the objects of `ids` at the pinned pose of the current reference frame (zero velocity): a kinematic
        approximation. Pinned pose = the reference pose lifted by its penetration into the static supports (desk,
        support discs living at that frame); owner `object_pin.pin_pose` (the same function as reset, the PSS snap and
        the judge park oracle). `joints` (explicit, no default): True = also write the articulated objects' joint θ·ω
        from the same frame's reference (ω 0); owner `spawn_hold.pin_joints` (the same function as reset; why = that
        module's header). Callers: SPAWN_HOLD right after reset (DEXENGINE_OBJ_SPAWN_HOLD · joints=True), and parking
        finished envs between judge batches (DEXENGINE_EVAL_PARK · joints=False: PARK gates C·D measure with the root
        pinned). Returns {side: joints written} (empty dict when joints=False)."""
        frames = self.reference_frames(ids)
        ref = self.trajectory_manager.get_reference_state_for_envs(ids, frames)
        wrote = {}
        for side, obj in self.held_object_assets():
            pos, quat, _, _ = self.object_pin_pose(ids, frames, side, obj, ref)
            rs = torch.zeros(len(ids), 13, device=self.device)
            rs[:, :3] = pos + self.scene.env_origins[ids]
            rs[:, 3:7] = quat
            obj.write_root_state_to_sim(rs, ids)
            if joints:
                wrote[side] = self.object_pin_joints(ids, side, obj, ref)
        return wrote
