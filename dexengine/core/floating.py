"""Floating hand body identity: the armless arm entry ``floating`` of ``dexengine.core.arms``.

The release registry knows the floating arm (``arms.FLOATING_ARM``), so any process that resolves a robot (mounts,
robot selection, launch checks, bakes) must be able to answer the identity questions about it without Isaac Lab and
without the experiment code: which hands have a floating body, its root body (= end effector), its mounts, which
arm's data world it reads, and which merged asset a wrist drive mode loads. The physics of the floating body (wrist
wrench, virtual-joint drive, anchors, reset event, inner action) lives in ``experiments.ablations.floating_hand`` and
reaches the RL task through ``dexengine.rl.tasks.hooks``.

Environment: ``DEXENGINE_FH_WRIST`` ('' = wrench, the default | 'joint') selects the merged asset of the
virtual-joint body; it is read in every process that resolves the floating robot files (launcher, bundle builder,
Isaac process), so it stays an environment variable.

Stdlib only. The import of ``dexengine.robots.selection`` stays inside ``source_urdf``: selection imports this module.
"""

from __future__ import annotations

import functools
import os
import pathlib
import xml.etree.ElementTree as ET

from dexengine.core.arms import FLOATING_ARM

#: run-log marker: a run whose log lacks it is not a floating-hand run
MARK = "[FLOATING_HAND]"
#: The arm whose DATA world the floating hand reads — the trees (data/<ds>/trees/<arm>_<hand>/) and the wrist tables
#: (precompute/…/wrist_poses_*) are combo products (placement dx·dy·yaw·dz baked for the arm's reach). The floating hand
#: reads the rby1 world so that the same clips keep the same objects and the same wrist reference; only the body
#: changes.
DATA_WORLD_ARM = "rby1"
#: hands with a floating body (mounts and baked assets exist); any other hand dies. To add one: bake its per-side URDF
#: and USD, register (FLOATING_ARM, hand) in the mounts, then add it here.
SUPPORTED_HANDS = ("sharpa_wave", "wuji2")

WRIST_DRIVE_ENV = "DEXENGINE_FH_WRIST"
WRIST_DRIVES = ("wrench", "joint")
#: stem suffix of the virtual-joint body's merged asset (``<stem>_vj/<stem>_vj.<ext>``)
VJ_STEM_SUFFIX = "_vj"


def is_floating(arm_name: str) -> bool:
    """Is this arm name the floating hand?"""
    return arm_name == FLOATING_ARM


def data_world_arm(arm_name: str) -> str:
    """The arm whose data world (trees · wrist tables · sequence folder identity) this arm reads: DATA_WORLD_ARM for the
    floating hand, itself otherwise."""
    return DATA_WORLD_ARM if is_floating(arm_name) else arm_name


def source_urdf(hand: str) -> pathlib.Path:
    """The floating body's source = the DATA_WORLD_ARM combo URDF of the sim body (arms.sim_body_variant — with the
    hand's limit variant suffix, e.g. wuji2 ``_truncated``). Path rule owner:
    dexengine.robots.selection.combo_urdf_path (dies when missing). The per-side URDF generator and root_body read
    it."""
    from dexengine.core.arms import sim_body_variant  # noqa: PLC0415
    from dexengine.robots.selection import combo_urdf_path  # noqa: PLC0415 — selection imports this module

    return combo_urdf_path(DATA_WORLD_ARM, hand, sim_body_variant(DATA_WORLD_ARM, hand))


@functools.cache
def root_body(hand: str, side: str) -> str:
    """The hand root body (free root of the floating hand = wrist frame = EE), by rule: the child link of
    ``{side}_mount_joint`` in the source combo URDF. Dies unless there is exactly one such joint and it is fixed.
    This link is the wrist frame because the mount joint's origin = the mounts' ee_to_wrist, so with the arm removed
    EE → wrist is the identity (identity_mounts). The bake's merge_fixed_joints keeps the root name."""
    if side not in ("right", "left"):
        raise ValueError(f"{MARK} side {side!r} — right|left")
    src = source_urdf(hand)
    joints = [j for j in ET.parse(src).getroot().findall("joint") if j.get("name") == f"{side}_mount_joint"]
    if len(joints) != 1 or joints[0].get("type") != "fixed" or joints[0].find("child") is None:
        raise RuntimeError(
            f"{MARK} {src.name}: {side}_mount_joint is not exactly one fixed joint"
            f" ({len(joints)}) — the hand root cannot be found by rule"
        )
    return joints[0].find("child").get("link")


def identity_mounts(mount_cls) -> dict:
    """Mounts of (FLOATING_ARM, hand) for dexengine.core.mounts.MOUNT_REGISTRY — the IDENTITY for every supported hand.
    The hand root (``{side}_hand_flange``) is the wrist: the combo URDF's mount joint child sits at the wrist frame
    (EE ⊕ ee_to_wrist), so with the arm removed EE → wrist is the identity by definition. The class comes in as an
    argument to avoid a mounts ↔ this-module import cycle."""
    return {
        (FLOATING_ARM, h): mount_cls(
            arm=FLOATING_ARM,
            hand=h,
            ee_to_wrist_offset=(0.0, 0.0, 0.0),
            ee_to_wrist_rot_right=(1.0, 0.0, 0.0, 0.0),
            ee_to_wrist_rot_left=(1.0, 0.0, 0.0, 0.0),
            provisional=False,
            provenance="floating hand: EE = hand root = wrist frame (combo URDF mount_joint child) — identity",
        )
        for h in SUPPORTED_HANDS
    }


def wrist_drive() -> str:
    """The wrist drive of this process — unset = 'wrench' (default). An unknown value dies."""
    raw = os.environ.get(WRIST_DRIVE_ENV, "")
    if raw == "":
        return "wrench"
    if raw not in WRIST_DRIVES:
        raise RuntimeError(f"{MARK} {WRIST_DRIVE_ENV}={raw!r} — only {'|'.join(WRIST_DRIVES)} (default unset = wrench)")
    return raw


def joint_drive() -> bool:
    """Is the wrist drive the virtual-joint drive?"""
    return wrist_drive() == "joint"


def vj_canon_path(canon) -> pathlib.Path:
    """The default body's merged asset path (``<dir>/<stem>/<stem>.<ext>``) → the virtual-joint body's
    (``<dir>/<stem>_vj/<stem>_vj.<ext>``). The split pair names follow from it by split_artic's rule."""
    p = pathlib.Path(canon)
    st = p.stem + VJ_STEM_SUFFIX
    return p.parent.parent / st / f"{st}{p.suffix}"


def mode_canon_path(canon) -> pathlib.Path:
    """The merged asset path for this process's wrist drive — the default (wrench) is unchanged."""
    return vj_canon_path(canon) if joint_drive() else pathlib.Path(canon)
