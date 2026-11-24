"""Taco trajectory loader for Isaac Lab replay.

Loads dataset NPZ trajectories and provides vectorized access for parallel environments.
All trajectories are pre-padded to max length and stacked into GPU tensors.

**This file does not define the input qpos layout.** Its owner is `dexengine/core/qpos_layout.py` and the facts come
from the tree's `task_info.json`: base-layer trees write **absolute columns** in `qpos_layout`/`objects[].qpos_start`,
and for legacy trees (no qpos_layout key) the same reader's legacy branch reproduces the old arithmetic
(2H + 7·n_extra + 7·slot) literally. A `[0, 18) / [36, 43) / [43, 50)` table here held only for one hand (18) and one
dataset (TACO); on the base trees one-hand trees (left hand at column 0), shared objects (one object block) and extra
positions all break it.

The **output** normalised tensor keeps the fixed layout below (the only thing this file owns):
  [0, 18)   right hand
  [18, 36)  left hand
  [36, 43)  right primary object
  [43, 50)  left primary object
  [50, 50+7*n_extra_slots)  extra-object slots (slot k = the clip's k-th slot column)

Extra-object slots (dexengine/data/objects/extra_slots.py owns the rule and K): slot k of a clip carries the qpos block
at the
clip's k-th column in `extra_slot_cols[seq]`; a clip with fewer slots leaves the rest hidden at z=-100.
"""

from __future__ import annotations

import os
import re
from glob import glob

import numpy as np
import torch

from dexengine.core.desk import DESK_TOP_SURFACE_Z
from dexengine.rl.tasks.mdp.trajectory.loader_clips import load_clips


class TacoTrajectoryLoader:
    """Loads and manages the clip trajectories (any dataset) for parallel replay."""

    def __init__(
        self,
        data_dir: str,
        device: str | torch.device = "cpu",
        subject_filter: str | None = None,
        scene_filter: str | None = None,
        sequence_filter: str | None = None,
        n_extra_slots: int = 0,
        extra_slot_cols: dict | None = None,
        extra_slot_static: dict | None = None,
        hand_qpos_dim: int = 18,
        world_z_offset: float = DESK_TOP_SURFACE_Z,
        expect_idle_side: str = "",
    ):
        """Initialize the trajectory loader.

        Args:
            data_dir: Path to the processed Taco dataset directory
                (e.g., .../taco/trees/rby1_wuji/bimanual/ — the trees layer)
            device: Device to store tensors on.
            subject_filter: Optional subject filter matching ``__s{N}__`` pattern
                in the sequence name (e.g., "s0", "s11").
            scene_filter: Optional scene filter matching ``scene_XX`` prefix
                (e.g., "scene_01", "scene_02").
            sequence_filter: Optional substring filter on full sequence name.
            n_extra_slots: extra-object slot count K (0 = no slots — the normalized tensor is byte-identical).
            extra_slot_cols: {seq_name: [qpos column of slot 0, slot 1, ...]} — every loaded clip must be a key when
                K > 0 (an empty list = a clip without slots). Columns must be the tree's non-main object blocks.
            hand_qpos_dim: Per-hand qpos width in the NPZ layout. 18 for RH56F1
                (6 wrist + 12 finger, default), 28 for SharpA Wave (6 + 22).
            expect_idle_side: the run's D16 idle side ("" | "right" | "left"). When set, every clip's tree must declare
                exactly that side idle (qpos_layout.Layout.idle_sides) — a clip that disagrees raises (the env has no
                object prim on that side). "" = no check (the two-object run, byte-identical).
        """
        self.device = torch.device(device)
        self.data_dir = data_dir

        filtered_paths = self._discover_paths(data_dir, subject_filter, scene_filter, sequence_filter)

        self._n_unique_extras = int(n_extra_slots)
        self._extra_slot_cols = dict(extra_slot_cols or {})
        self._extra_slot_static = dict(extra_slot_static or {})  # {seq: [static per slot?]}; static = pinned at frame 0
        if self._n_unique_extras and len(self._extra_slot_cols) == 0:
            raise ValueError("n_extra_slots > 0 but extra_slot_cols is empty — the slot columns are unknown")
        self._hand_qpos_dim = int(hand_qpos_dim)
        _H = self._hand_qpos_dim  # per-hand stride (18 RH56F1, 28 SharpA)
        _HH = 2 * _H  # both hands

        self._normalized_dims(world_z_offset, expect_idle_side, _H, _HH, filtered_paths)

    def _normalized_dims(self, world_z_offset, expect_idle_side, _H, _HH, filtered_paths):
        """Normalized dim: 2*H (hands) + 14 (primary objects) + 7*n_unique_extras Articulated objects (rule D1): the
        **hinge/slide angle** rides on the reference. An articulated clip (arctic, oakink2) has two parts and one
        joint, but the reference used to carry position and orientation only, so the training scene could only place
        a single rigid body. The two columns go **at the end**, keeping the front slice (_HH+14+7n) byte-identical.
        """
        self._artic_off = _HH + 14 + 7 * self._n_unique_extras
        self._norm_dim = self._artic_off + 2
        raw_clips = self._clip_z_shifts(world_z_offset, expect_idle_side, _H, filtered_paths)
        padded = np.zeros((self.num_clips, self.max_length, self._norm_dim), dtype=np.float32)
        for i, clip in enumerate(raw_clips):
            T = clip.shape[0]
            padded[i, :T] = clip
            if self.max_length > T:
                padded[i, T:] = clip[-1]

        self.qpos_all = torch.tensor(padded, device=self.device)

        # Store per-clip extra counts as tensor
        self.clip_n_extras = torch.tensor(self._clip_n_extras, dtype=torch.long, device=self.device)

        print(
            f"[TacoTrajectoryLoader] Loaded {self.num_clips} clips, "
            f"max_length={self.max_length}, "
            f"length range=[{self.clip_lengths.min().item()}, {self.clip_lengths.max().item()}], "
            f"n_extra_slots={self._n_unique_extras}"
        )

    def _clip_z_shifts(self, world_z_offset, expect_idle_side, _H, filtered_paths):
        """Per-clip z shift: the tree keeps the capture z and the consumer (dexengine.rl.tasks.mdp.trajectory.manager)
        adds a **constant** world_z_offset. The clip's true offset follows the judge and bake rule
        (qpos_layout.Layout.world_z_offset: desk scene = DESK_HEIGHT − support.desk_z, aerial scene = the baked
        aerial_support.z_off), so the difference is added **per clip** to the wrist and object z here; no value is made
        up here. Trees with support z 0 (TACO staging) have dz 0 and the arrays stay byte-identical.
        """
        self._world_z_offset = float(world_z_offset)
        self._clip_dz: list[float] = []  # dz applied per clip (loader order)
        self._clip_scene: list[str] = []  # scene per clip (for the marker)
        self.clip_datasets: list[str] = []  # tree dataset per clip (task_info.dataset_name); read by the desk check
        self._n_aerial_scene = 0  # number of clips with scene=aerial (for the marker)

        _layout_src, clip_has_hand, raw_clips = self._load_clips(expect_idle_side, _H, filtered_paths)

        if expect_idle_side:  # marker: the idle-side check really ran
            print(
                f"[IDLE_SIDE] loader: clips {len(raw_clips)}/{len(filtered_paths)} have tree idle side = "
                f"{expect_idle_side} "
                f"(that side's object block is a unit pose 0 — the env has no object on that side)",
                flush=True,
            )
        if self._n_aerial_scene:  # marker: the aerial tree z shift really applied
            _nz = [d for d, sc in zip(self._clip_dz, self._clip_scene, strict=True) if sc == "aerial"]
            print(
                f"[AERIAL] aerial tree z shift applied (rule D20): aerial clips "
                f"{self._n_aerial_scene}/{len(filtered_paths)} · "
                f"dz(z_off−{self._world_z_offset:.2f}) min {min(_nz):+.4f} max {max(_nz):+.4f} m "
                f"(wrist, main object, extra z; joint columns excluded)",
                flush=True,
            )
        _tab = [d for d, sc in zip(self._clip_dz, self._clip_scene, strict=True) if sc != "aerial"]
        _tnz = [
            d for d in _tab if d != 0.0
        ]  # marker: the desk tree z shift (the judge's coords.scene_z_offset rule) applied
        print(
            f"[DESK_Z] desk tree per-clip dz (= −support.desk_z): n_nonzero={len(_tnz)}/{len(_tab)}"
            + (f" · min {min(_tnz):+.4f} max {max(_tnz):+.4f} m" if _tnz else "")
            + f" · aerial {self._n_aerial_scene} (owner qpos_layout.Layout.world_z_offset)",
            flush=True,
        )

        # Completion is declared by count only. Two layout generations under one root mean two conditions: fail.
        print(
            f"[qpos_layout] source={_layout_src} · trees {sum(_layout_src.values())}/{len(filtered_paths)}", flush=True
        )
        if len(_layout_src) > 1:
            raise ValueError(
                f"layout generations mixed in one data_dir: {_layout_src} — a run with two conditions is invalid."
            )
        self.layout_source = next(iter(_layout_src), "none")
        self.clip_has_hand = (
            torch.tensor(clip_has_hand, dtype=torch.bool, device=self.device)
            if clip_has_hand
            else torch.zeros((0, 2), dtype=torch.bool, device=self.device)
        )

        self.num_clips = len(raw_clips)
        self.clip_lengths = torch.tensor([c.shape[0] for c in raw_clips], dtype=torch.long, device=self.device)
        self.max_length = int(self.clip_lengths.max().item())

        # Pre-pad and stack into single tensor
        tensor_gb = self.num_clips * self.max_length * self._norm_dim * 4 / (1024**3)
        print(
            f"[TacoTrajectoryLoader] Allocating trajectory tensor: "
            f"{self.num_clips} × {self.max_length} × {self._norm_dim} = {tensor_gb:.2f} GB"
        )
        if tensor_gb > 8.0:
            print(
                f"[TacoTrajectoryLoader] WARNING: Large dataset ({tensor_gb:.1f} GB). "
                f"Consider using --scene filter to reduce data size."
            )
        return raw_clips

    def _discover_paths(self, data_dir, subject_filter, scene_filter, sequence_filter):
        """Trajectory files under data_dir, filtered to the requested sequences."""
        # Discover all trajectory files
        pattern = os.path.join(data_dir, "*/0/trajectory_ikrollout.npz")
        all_paths = sorted(glob(pattern))

        if not all_paths:
            raise FileNotFoundError(f"No trajectory files found in {data_dir}")

        # Apply filters
        filtered_paths = []
        for p in all_paths:
            seq_name = p.split("/")[-3]

            if subject_filter:
                if not re.search(rf"__{re.escape(subject_filter)}__", seq_name):
                    continue
            if scene_filter:
                if not seq_name.startswith(scene_filter):
                    continue
            if sequence_filter:
                # Comma-list => explicit multi-clip set (EXACT seq_name membership) for Track B
                # single-policy multi-clip training. Single token => EXACT match
                # (a substring match would silently load EVERY placement variant
                # sharing the prefix — e.g. "..._189_shift" pulled shift+shift2+shift3+shift4
                # into ONE training run (4-clip multiclip-from-scratch, the configuration
                # heritage flagged as failing without warm-start). Multi-clip now requires
                # the explicit comma-list; a non-existent exact name fails loudly below.
                _seq_toks = [t.strip() for t in sequence_filter.split(",") if t.strip()]
                if len(_seq_toks) > 1:
                    if seq_name not in _seq_toks:
                        continue
                elif seq_name != sequence_filter:
                    continue

            filtered_paths.append(p)

        if not filtered_paths:
            raise FileNotFoundError(
                f"No trajectories match filters: subject={subject_filter}, "
                f"scene={scene_filter}, sequence={sequence_filter}"
            )
        return filtered_paths

    _load_clips = load_clips

    def _fill_slots(self, seq_name, qpos, extras, extra_cols, D, dz) -> None:
        """Slot k = the pose of this clip's k-th slot column (static slots pinned at frame 0) + dz. Column None = an
        empty
        cell of the clip (the axis alignment of a one-env render, env_setup.extra_slots._extra_slot_axis): the hidden
        default stays. A clip outside the roster, an unknown column or a static-mark count mismatch fails."""
        if seq_name not in self._extra_slot_cols:
            raise ValueError(
                f"{seq_name}: slots on (K={self._n_unique_extras}) but extra_slot_cols has no entry for this clip"
            )
        cols = [None if c is None else int(c) for c in self._extra_slot_cols[seq_name]]
        if len(cols) > self._n_unique_extras or not {c for c in cols if c is not None} <= set(extra_cols):
            raise ValueError(
                f"{seq_name}: slot columns {cols} — above K={self._n_unique_extras} or outside the extra "
                f"blocks {extra_cols}"
            )
        stat = list(self._extra_slot_static.get(seq_name) or [False] * len(cols))
        if len(stat) != len(cols):
            raise ValueError(f"{seq_name}: {len(stat)} static marks ≠ {len(cols)} slot columns")
        for k, c in enumerate(cols):
            if c is None:
                continue
            if c + 7 > D:
                raise ValueError(f"{seq_name}: slot {k} columns {c}:{c + 7} exceed the qpos width {D}")
            extras[:, 7 * k : 7 * k + 7] = qpos[:1, c : c + 7] if stat[k] else qpos[:, c : c + 7]
            extras[:, 7 * k + 2] += dz

    @property
    def n_extra_slots(self) -> int:
        """Extra-object slot count K (0 = off)."""
        return self._n_unique_extras

    def sample_clips(self, num_envs: int) -> torch.Tensor:
        """Randomly assign clips to environments."""
        return torch.randint(0, self.num_clips, (num_envs,), device=self.device)

    def get_clip_lengths(self, clip_ids: torch.Tensor) -> torch.Tensor:
        """Get lengths of specified clips."""
        return self.clip_lengths[clip_ids]

    def clip_dz_of(self, clip_idx: int) -> float:
        """z shift (m) applied to clip `clip_idx` (loader order): desk scene −support.desk_z, aerial scene z_off −
        world_z_offset. The consumer (dexengine.rl.tasks.mdp.trajectory.manager) adds **the same value** to the
        body_positions and wrist_poses caches so the references lie on one plane. A clip number the loader did not read
        raises IndexError (no silent 0)."""
        return float(self._clip_dz[clip_idx])

    def get_frame(
        self, clip_ids: torch.Tensor, frame_ids: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Get trajectory data for given clips and frames.
        8-tuple (rule D1: the last item is the joint columns (N,2)); all 3 callers (trajectory manager) unpack 8.

        Args:
            clip_ids: Clip indices of shape (N,).
            frame_ids: Frame indices of shape (N,).

        Returns:
            Tuple of:
            - right_hand_qpos: (N, H) right hand joint positions (H = hand_qpos_dim)
            - left_hand_qpos: (N, H) left hand joint positions
            - right_obj_pos: (N, 3) right object position XYZ
            - right_obj_quat: (N, 4) right object quaternion (w,x,y,z)
            - left_obj_pos: (N, 3) left object position XYZ
            - left_obj_quat: (N, 4) left object quaternion (w,x,y,z)
            - extra_obj_poses: (N, n_extra_slots, 7) extra-object slot poses (pos+quat each)
        """
        clamped_frames = torch.clamp(frame_ids, 0, self.max_length - 1)
        qpos = self.qpos_all[clip_ids, clamped_frames]  # (N, norm_dim)

        _H = self._hand_qpos_dim
        _HH = 2 * _H
        right_hand_qpos = qpos[:, 0:_H]
        left_hand_qpos = qpos[:, _H:_HH]
        right_obj_pos = qpos[:, _HH : _HH + 3]
        right_obj_quat = qpos[:, _HH + 3 : _HH + 7]
        left_obj_pos = qpos[:, _HH + 7 : _HH + 10]
        left_obj_quat = qpos[:, _HH + 10 : _HH + 14]

        # Extra objects: (N, 7*max_extras) -> (N, max_extras, 7)
        extras_flat = qpos[:, _HH + 14 : self._artic_off]
        artic_ang = qpos[:, self._artic_off : self._artic_off + 2]  # (N, 2) = [right, left]
        N = clip_ids.shape[0]
        extra_obj_poses = extras_flat.reshape(N, self._n_unique_extras, 7)

        return (
            right_hand_qpos,
            left_hand_qpos,
            right_obj_pos,
            right_obj_quat,
            left_obj_pos,
            left_obj_quat,
            extra_obj_poses,
            artic_ang,
        )
