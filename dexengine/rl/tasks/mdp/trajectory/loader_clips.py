"""Clip loading of TacoTrajectoryLoader (bound as its ``_load_clips`` method): per clip the object / hand blocks, the
layout and desk-z checks and the per-clip metadata; and the lazy qpos-layout reader it uses."""

from __future__ import annotations

import os

import numpy as np

from dexengine.retarget.quat_bounds import require_unit

_QL = None  # qpos_layout module (lazy)


def qpos_layout_reader():
    """Single owner of the qpos address table. Reads the same file where the package cannot be imported (scripts)."""
    global _QL
    if _QL is None:
        from dexengine.core import qpos_layout as _m

        _QL = _m
    return _QL


def load_clips(self, expect_idle_side, _H, filtered_paths):
    """Load every filtered clip: object / hand blocks, layout and desk-z checks, per-clip metadata."""
    # Load and normalize all clips
    raw_clips = []
    self.sequence_names = []
    self._clip_n_extras = []  # per-clip extra object count (local)
    clip_has_hand = []  # per clip (right hand present, left hand present); a missing hand is zero-filled
    _layout_src: dict[str, int] = {}
    _layout_warned = {"extra": False, "drop": False}

    for idx, p in enumerate(filtered_paths):
        if (idx + 1) % 100 == 0 or (idx + 1) == len(filtered_paths):
            print(f"  Loading clips: {idx + 1}/{len(filtered_paths)}")
        seq_name = p.split("/")[-3]
        seq_dir = os.path.dirname(p)

        # Always prefer kinematic trajectory (ground truth from MANO retargeting)
        kin_path = os.path.join(seq_dir, "trajectory_kinematic.npz")
        if os.path.exists(kin_path):
            npz_path = kin_path
        else:
            npz_path = p  # fallback to ikrollout
            print(f"  WARNING: No kinematic trajectory for {seq_name}, falling back to ikrollout")

        npz = np.load(npz_path, allow_pickle=True)
        qpos = npz["qpos"]
        T, D = qpos.shape

        # The tree states the address table. For legacy trees the reader's legacy branch reproduces the old
        # arithmetic, and the npz `n_extra_objects` counts in the address **only for legacy**: on the base layer
        # some trees disagree with their layout (hocap 40/269 measured) and the absolute columns are the facts, so
        # it is read for comparison only.
        n_extra_npz = int(npz["n_extra_objects"]) if "n_extra_objects" in npz else 0
        _ql = qpos_layout_reader()
        lay = _ql.read(os.path.dirname(seq_dir), expect_hand_qpos_dim=_H)
        if lay.source != "task_info.layout":
            lay = _ql.with_npz_extras(lay, n_extra_npz)
        elif n_extra_npz and n_extra_npz != len(lay.extras) and not _layout_warned["extra"]:
            print(
                f"[TacoTrajectoryLoader] ⚠ npz n_extra_objects={n_extra_npz} differs from the layout's extra "
                f"{len(lay.extras)} ({seq_name}) — addresses follow the layout",
                flush=True,
            )
            _layout_warned["extra"] = True
        _layout_src[lay.source] = _layout_src.get(lay.source, 0) + 1
        _load_clip_blocks(
            self, expect_idle_side, _H, D, T, _layout_warned, clip_has_hand, lay, npz, qpos, raw_clips, seq_name
        )
    return _layout_src, clip_has_hand, raw_clips


def _load_clip_blocks(
    self, expect_idle_side, _H, D, T, _layout_warned, clip_has_hand, lay, npz, qpos, raw_clips, seq_name
):
    """Idle-side check, hand and object blocks and extra objects of one clip."""
    n_extra = len(lay.extras)
    if expect_idle_side and tuple(lay.idle_sides()) != (expect_idle_side,):
        raise ValueError(
            f"{seq_name}: this run declares the idle side (rule D16) = {expect_idle_side}, but the "
            f"tree's idle side is {list(lay.idle_sides()) or 'none (both active)'} — run property "
            "and clip differ. The list bypassed the entry's (rl/cli.py --sequence_dir) uniformity check"
        )

    # Hand blocks: "column 0 = right hand" is false on the base layer (a left-only tree has the left hand at 0).
    # A missing hand is zero-filled and **flagged** with a mask; another column is never read as a hand.

    right_hand, _has_r = _hand_block(lay, qpos, T, D, _H, seq_name, "right")
    left_hand, _has_l = _hand_block(lay, qpos, T, D, _H, seq_name, "left")
    clip_has_hand.append((_has_r, _has_l))
    _dz, left_hand, right_hand = _clip_z_shift_owner(self, _has_l, _has_r, lay, left_hand, right_hand)

    # Extract primary objects using correct offset.
    # The slot<->side assignment reads **the value the tree recorded**. The default is right->0 / left->1;
    # only clips where the person holds the tool in the left hand (recorded
    # right_object_slot=1) swap. Hard-coding it here once spawned meshes by grasp while reading trajectories by
    # slot, with hands and objects in different places.
    # Main objects: **absolute columns**, not slot arithmetic. A shared object (one block, both hands) has both
    # sides read the same columns (this once zero-filled as "no left object" silently, on 4,573 base trees).
    # An inactive side (rule D16, one-hand) has its hand injected at an idle pose and **no main object is
    # normal**; the env spawns no object there either (`[OBJECTS] … left=none`, source objects[]). An active
    # side without an object is still a defect and fails; the tree's active_sides is the one basis.

    right_obj = _obj_block(lay, qpos, T, D, seq_name, "right")
    left_obj = _obj_block(lay, qpos, T, D, seq_name, "left")
    if _dz != 0.0:
        right_obj = right_obj.copy()
        right_obj[:, 2] += _dz
        left_obj = left_obj.copy()
        left_obj[:, 2] += _dz

    _place_extra_objects(
        self,
        D,
        T,
        _dz,
        _layout_warned,
        lay,
        left_hand,
        left_obj,
        n_extra,
        npz,
        qpos,
        raw_clips,
        right_hand,
        right_obj,
        seq_name,
    )


def _clip_z_shift_owner(self, _has_l, _has_r, lay, left_hand, right_hand):
    """Per-clip z shift: the rule's owner is qpos_layout (desk support, aerial z_off). No statement fails."""
    _dz = lay.world_z_offset(self._world_z_offset) - self._world_z_offset
    if lay.scene == "aerial":
        self._n_aerial_scene += 1
    if _dz != 0.0:
        if _has_r:
            right_hand = right_hand.copy()
            right_hand[:, 2] += _dz
        if _has_l:
            left_hand = left_hand.copy()
            left_hand[:, 2] += _dz
    self._clip_dz.append(_dz)
    self._clip_scene.append(lay.scene)
    self.clip_datasets.append(lay.dataset)
    return _dz, left_hand, right_hand


def _place_extra_objects(
    self,
    D,
    T,
    _dz,
    _layout_warned,
    lay,
    left_hand,
    left_obj,
    n_extra,
    npz,
    qpos,
    raw_clips,
    right_hand,
    right_obj,
    seq_name,
):
    """Extract extra objects and place at GLOBAL indices."""
    extras = np.zeros((T, 7 * self._n_unique_extras), dtype=np.float32)
    # Default hidden position for all unused extras: z=-100
    for gi in range(self._n_unique_extras):
        extras[:, gi * 7 + 2] = -100.0  # z = -100 (hidden underground)
        extras[:, gi * 7 + 3] = 1.0  # quat w = 1 (identity)

    _ex_sl = lay.extra_slices()
    # Slot k = this clip's k-th slot column (extra_slot_cols; rule owner dexengine/data/objects/extra_slots.py).
    # The columns must be the reader's **absolute columns** (non-main object blocks, lay.extras). With slots on,
    # a clip outside the roster or an unknown column fails.
    if self._n_unique_extras:
        self._fill_slots(seq_name, qpos, extras, lay.extras, D, _dz)
    elif _ex_sl and not _layout_warned["drop"]:
        # A meaningful input is never swallowed without a consumer: this run drops the extras.
        print(
            f"[TacoTrajectoryLoader] ⚠ this run drops {len(_ex_sl)} extra objects "
            f"(slots off — n_extra_slots=0 · DEXENGINE_EXTRA_SLOTS)",
            flush=True,
        )
        _layout_warned["drop"] = True

    npz.close()

    # Articulated angle (rule D1): the layout owns the column. 0 for a non-articulated clip (the scene uses 0).
    artic = np.zeros((T, 2), dtype=np.float32)
    for _k, _side in ((0, "right"), (1, "left")):
        _a = (lay.artic or {}).get(_side)
        if not _a:
            continue
        _col = int(_a["col"])
        if _col >= qpos.shape[1]:  # a layout that disagrees with the data is never left at 0 silently
            raise ValueError(
                f"{seq_name}: artic column {_col} is outside the qpos width {qpos.shape[1]} — another generation"
            )
        artic[:, _k] = qpos[:, _col].astype(np.float32)

    # Build normalized clip
    clip = np.concatenate(
        [
            right_hand.astype(np.float32),
            left_hand.astype(np.float32),
            right_obj.astype(np.float32),
            left_obj.astype(np.float32),
            extras,
            artic,
        ],
        axis=1,
    )

    raw_clips.append(clip)
    self.sequence_names.append(seq_name)
    self._clip_n_extras.append(n_extra)


def _hand_block(lay, qpos, T, D, _H, seq_name, side):
    """One hand block of the clip qpos (zero-filled and flagged False when the tree has no such hand)."""
    sl = lay.hand_slice(side)
    if sl is None:
        return np.zeros((T, _H), dtype=qpos.dtype), False
    a, b = sl
    if b > D:
        raise ValueError(f"{seq_name}: {side} hand block {a}:{b} exceeds the qpos width {D}")
    return qpos[:, a:b], True


def _obj_block(lay, qpos, T, D, seq_name, side):
    """The main-object block of one side (identity pose when that side is idle; an active side without one fails)."""
    sl = lay.main_slice(side)
    if sl is None:
        if lay.is_active(side):
            raise ValueError(
                f"{seq_name}: no main object for the {side} hand in the tree (objects[] role=main). "
                f"This tree declares {side} active (active_sides={list(lay.active_sides)}). "
                "Not zero-filled: the clip is not a training target or its assignment is missing."
            )
        z = np.zeros((T, 7), dtype=qpos.dtype)
        z[:, 3] = 1.0
        return z
    a, b = sl
    if b > D:
        raise ValueError(f"{seq_name}: {side} object block {a}:{b} exceeds the qpos width {D}")
    require_unit(qpos[:, a + 3 : b], f"{seq_name} {side} main object")  # rewards and observations read it as a rotation
    return qpos[:, a:b]
