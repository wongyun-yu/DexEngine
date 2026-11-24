"""Configuration of the trajectory manager, the virtual-wrist joint order and the fingertip index resolver."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.utils import configclass

from dexengine.core.desk import DESK_TOP_SURFACE_Z
from dexengine.rl.tasks.mdp.naming import SHARPA_FINGER_JOINT_SUFFIXES


def virtual_wrist_order(side: str) -> list[str]:
    """Names of the 6 virtual wrist columns: a convention every hand shares (not a per-hand fact)."""
    return [f"{side}_pos_x", f"{side}_pos_y", f"{side}_pos_z", f"{side}_rot_x", f"{side}_rot_y", f"{side}_rot_z"]


# Dataset MuJoCo depth-first joint order (18 joints per hand)
DATASET_RIGHT_JOINT_ORDER = [
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_rot_x",
    "right_rot_y",
    "right_rot_z",
    "right_thumb_proximal_yaw_joint",
    "right_thumb_proximal_pitch_joint",
    "right_thumb_intermediate_joint",
    "right_thumb_distal_joint",
    "right_index_proximal_joint",
    "right_index_intermediate_joint",
    "right_middle_proximal_joint",
    "right_middle_intermediate_joint",
    "right_ring_proximal_joint",
    "right_ring_intermediate_joint",
    "right_pinky_proximal_joint",
    "right_pinky_intermediate_joint",
]
DATASET_LEFT_JOINT_ORDER = [name.replace("right_", "left_") for name in DATASET_RIGHT_JOINT_ORDER]
SHARPA_RIGHT_JOINT_ORDER = [
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_rot_x",
    "right_rot_y",
    "right_rot_z",
] + [f"right_{sfx}" for sfx in SHARPA_FINGER_JOINT_SUFFIXES]
SHARPA_LEFT_JOINT_ORDER = [name.replace("right_", "left_") for name in SHARPA_RIGHT_JOINT_ORDER]
SHARPA_RIGHT_FINGER_ORDER = SHARPA_RIGHT_JOINT_ORDER[6:]  # 22 finger joints
SHARPA_LEFT_FINGER_ORDER = [name.replace("right_", "left_") for name in SHARPA_RIGHT_FINGER_ORDER]
# Root quaternion matching the replay pipeline (Rule 13: coordinate frame alignment).
# With root=y_180_x_m90, the double rotation (root * converter_baked) = identity,
# so PhysX body_link_pos_w directly equals MuJoCo FK positions.
# No pre-rotation is needed on reference body positions.
ROOT_QUAT_WXYZ = torch.tensor([0.0, 0.0, 0.7071068, -0.7071068])  # read at _base_root_quat init (non-URDF-hand branch)
# Dataset FK body name -> Isaac Lab body name mapping.
# For dataset hand USD: only palm body differs (right_hand_base → right_wrist_rot_z).
# (the legacy hand/full-robot body-name tables of the
# removed arm were deleted with it — registered hands name their merged bodies through
# ``fk_body_aliases`` / ``fk_palm_isaac_body_*`` from the specs.)
_DATASET_TO_ISAAC_BODY_MAP = {
    "right_hand_base": "right_wrist_rot_z",
    "left_hand_base": "left_wrist_rot_z",
}


def _resolve_fk_finger_indices(body_names: list[str], tip_body_names: tuple[str, ...] = ()) -> dict:
    """Resolve per-finger FK body indices from an FK body-name list by name.

    Convention (matches both rig FK exports): body 0 = palm/hand base, then
    finger bodies grouped per finger in proximal->distal order. Returns
    ``{"palm": 0, "tips": {finger: idx}, "tips_list": [...],
    "level1": [...], "level2": [...]}`` where level1 = first (proximal) body of
    each finger, tips = last (distal) body, level2 = everything between.

    Used for non-13-body layouts (e.g. SharpA, 23 bodies); the legacy 13-body
    RH56F1 layout keeps its hardcoded index tables (drift invariant).
    """
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    if tip_body_names:
        # When the spec gives the fingertip bodies, split by **structure**, not by English finger names (Wuji has
        # finger1..finger5, and a name search failed outright). Convention: body 0 = palm, then each finger proximal ->
        # distal, contiguous, so each fingertip index ends its finger's range.
        if len(tip_body_names) != len(fingers):
            raise ValueError(
                f"{len(tip_body_names)} fingertip bodies — the FK group labels assume "
                f"{len(fingers)} ({', '.join(fingers)})."
            )
        missing = [t for t in tip_body_names if t not in body_names]
        if missing:
            raise ValueError(f"fingertip bodies {missing} are not in the FK list: {body_names}")
        tip_idx = [body_names.index(t) for t in tip_body_names]
        if tip_idx != sorted(tip_idx):
            raise ValueError(
                f"fingertip indices {tip_idx} are not ascending — the convention that FK bodies are contiguous per "
                "finger broke. Check that hand's FK export."
            )
        groups = {}
        prev = 0
        for f, ti in zip(fingers, tip_idx, strict=True):
            groups[f] = list(range(prev + 1, ti + 1))
            prev = ti
    else:
        groups: dict[str, list[int]] = {f: [] for f in fingers}
        for i, name in enumerate(body_names):
            for f in fingers:
                if f"_{f}_" in name or name.endswith(f"_{f}"):
                    groups[f].append(i)
                    break
        missing = [f for f in fingers if not groups[f]]
        if missing:
            raise ValueError(f"Cannot resolve FK finger groups (missing {missing}) from {body_names}")
    tips = {f: groups[f][-1] for f in fingers}
    level1 = [groups[f][0] for f in fingers]
    level2 = [i for f in fingers for i in groups[f][1:-1]]
    return {
        "palm": 0,
        "tips": tips,
        "tips_list": [tips[f] for f in fingers],
        "level1": level1,
        "level2": level2,
    }


#: Absolute tolerance of the full table <-> old path comparison; above it the init fails.
REF_TABLE_ATOL = 1e-6
#: Stash a reference-state call leaves for rebuilding observations (finger-only mode); the table path leaves the same
# names and values.
_REF_STASH = ("_last_rh_wrist_qpos", "_last_lh_wrist_qpos", "_last_rh_finger_dataset", "_last_lh_finger_dataset")


def ref_table_requested() -> bool:
    """The precomputed reference-state table is always built (checked against the cache path at init)."""
    return True


@configclass
class TrajectoryManagerCfg:
    """Configuration for the trajectory manager."""

    data_dir: str = MISSING
    """Path to the processed Taco dataset directory."""

    idle_side: str = ""
    """D16 idle side of this run ("" | "right" | "left") — written by env_setup.sides.apply_idle_side from the entry's
    run statement. The loader checks every clip against it (the tree's Layout.idle_sides must equal it). "" = the
    two-object bimanual run (no check, byte-identical)."""

    hand_qpos_dim: int = MISSING
    """Per-hand qpos width in the trajectory NPZ layout (6 wrist + fingers; e.g.
    28 = SharpA 6+22, 26 = Wuji 6+20) — from the hand spec, no default. Width is NOT an
    identity — two hands can share it — so never branch on this number alone;
    key on ``hand_name``."""

    hand_name: str = MISSING
    """Which hand the NPZ layout belongs to — the hand spec's name, no default. Width is
    NOT an identity: two hands can share a qpos width while meaning different things."""

    dataset_finger_joint_names: tuple[str, ...] = ()
    """RIGHT-side finger joint names in **dataset qpos order** (the 6 virtual
    wrist columns come first and are named by convention, so only the fingers
    need declaring). Empty = the legacy dataset/SharpA name probes below.

    Why: without this value the joint-order candidates were five name-keyed lists, and a newly attached hand was in
    none of them, so env creation died with `Cannot match joints`. The order is that hand's vendor URDF file order,
    which the HandSpec already carries: pass it from the spec instead of writing it again here."""

    dataset_coupled_finger_names: tuple[str, ...] = ()
    """Of ``dataset_finger_joint_names``, the joints DRIVEN by another (mimic).
    The rest are the independent DOFs the policy commands."""

    fk_palm_isaac_body_right: str = ""
    fk_palm_isaac_body_left: str = ""
    """Isaac body the hand's palm got MERGED into — the arm EE link. FK body 0
    is the palm by export convention, so this needs no name matching and works
    for any hand on any arm (rby1: EE is link_right_arm_6)."""

    fk_tip_body_names: tuple[str, ...] = ()
    """RIGHT-side fingertip FK body names, thumb->pinky. Lets the FK finger
    grouping split by STRUCTURE instead of scanning for the English words
    thumb/index/middle/ring/pinky — Wuji's bodies are finger1..finger5 and that
    scan failed outright."""

    fk_body_aliases: tuple[tuple[str, str], ...] = ()
    """(MuJoCo FK body name -> Isaac body name) pairs for bodies the URDF
    importer MERGED away. The palm/hand-base is always one of these: it hangs
    off the arm EE by a FIXED joint, so Isaac fuses it into the EE link and the
    name disappears from the articulation.

    Why: this alias lived in three name tables per hand (dataset, legacy hand, legacy full-body), so Wuji died with
    'right_palm_link not found'. Which body merges where follows from two spec values (the hand palm name, the arm EE
    name), so it is passed here."""

    sequence_filter: str | None = None
    """Optional sequence name filter."""

    subject_filter: str | None = None
    """Optional subject filter."""

    scene_filter: str | None = None
    """Optional scene filter."""

    pose_corrections: bool = False
    """Load and apply initial pose corrections if available. Enable via --pose_corrections."""

    wrist_poses_dir_right: str | None = None
    """Directory containing pre-dumped wrist pose .npz files.
    Required for arm-IK training (arm IK needs wrist world poses, not joint values).
    Each .npz must match a sequence name and contain 'positions' (N,3) and 'quaternions' (N,4 wxyz)
    in the replay convention (+Y front). Rz(-90°) + traj_offset_z transform is applied at load."""

    wrist_poses_dir_left: str | None = None
    """Bimanual: LEFT-hand wrist pose .npz directory (same format as wrist_poses_dir_right).
    If None, the manager auto-derives the sibling '<wrist_poses_dir_right>_left' and loads it IF present.
    The left table is queried ONLY by get_wrist_pos_w/get_wrist_quat_w(side='left'), which only the
    left ResidualArmIKAction calls — so it is inert (never read) in single-hand mode."""

    traj_offset_z: float = 0.0
    """Table height offset for wrist poses (default 0.0, use 0.76 for the desk setup)."""

    rotate_to_x_forward: bool = False
    """If True, rotate trajectory data from +Y forward to +X forward via Rz(-90°).
    Applied to: hand root orientation, object poses, body positions.
    Wrist joint values are unchanged (root-relative local frame).
    Also adds desk_height Z offset to hand root and object positions."""

    desk_height: float = DESK_TOP_SURFACE_Z
    """Desk surface height for Z offset when rotate_to_x_forward=True."""

    urdf_hand: bool = False
    """If True, the hand USD was converted from URDF (no baked rotation).
    MJCF-converted hand (default=False) has a baked rotation cancelled by y_180_x_m90 root quat.
    URDF-converted hand (True) has no baked rotation, so identity root quat is used."""

    static_clip_assignment: bool = False
    """Fix env->clip = env_idx % num_clips for the whole run (multi-object mode: the
    spawned per-env mesh cannot change, so clips must not resample at reset). Uniform
    clip coverage comes from num_envs >> num_clips; RSI still randomizes start frames."""

    arm_ik_poses_dir_right: str | None = None
    """Directory containing pre-computed arm IK joint positions.
    Each .npz must match a sequence name and contain 'arm_joint_positions' (7,) float32.
    If set, reset_arm_to_trajectory uses direct teleport instead of single-step IK.
    If None, falls back to auto-discovery from wrist_poses_dir_right sibling path."""

    preprocessed_dir: str | None = None
    """Deprecated — analytical transform now handles the dataset->URDF-hand wrist conversion.
    Kept for backward compatibility; ignored if set."""

    n_extra_slots: int = 0
    """Extra-object slot count K — written by env_setup.extra_slots.build_extra_slots (DEXENGINE_EXTRA_SLOTS=1) from the
    rule owner dexengine/data/objects/extra_slots.py. 0 = no slots (the loader tensor is byte-identical)."""

    extra_slot_cols: dict | None = None
    """{seq_name: [qpos column of slot 0, ...]} for every loaded clip when n_extra_slots > 0 (same writer)."""

    extra_slot_static: dict | None = None
    """{seq_name: [static? per slot]} — a static slot holds its frame-0 pose (same writer · rule STATIC_SPAWN)."""

    # ------------------------------------------------------------------
    # Runtime slow-down + reference contact (added for the
    # contact-aware lag compensation Phase 1 experiments).
    # Both default to no-op so existing callers stay byte-identical.
    # ------------------------------------------------------------------
    slowdown_factor: float = 1.0
    """Time-warp factor for trajectory playback. 1.0 = no slow-down (default).
    >1.0 stretches the trajectory in time: at step k, effective_frame = k / slowdown_factor.
    Linear interp for positions/fingers, slerp for quaternions. Episode length
    auto-scales accordingly. Lookahead obs uses the SAME slowdown_factor for
    consistency (see ``get_wrist_pos_w`` / ``get_wrist_quat_w``)."""
