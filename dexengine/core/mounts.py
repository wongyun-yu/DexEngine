"""How a HAND is bolted onto an ARM — the geometry that belongs to neither alone.

WHY THIS FILE EXISTS
    ``HandSpec.link7_to_wrist_offset`` used to hold this. Its own docstring gave
    the game away: *"link7 -> rig-floating-base translation in link7 frame"*.
    ``link7`` was the first arm's EE link, so the value is not a property of the
    hand — it is a property of the (arm, hand) PAIR. With one arm that distinction
    costs nothing. With two arms a hand-keyed field has nowhere to put the second
    arm's number: you would either overwrite one pair's offset when mounting the
    hand on the other arm, or clone the whole hand spec. Both break the
    single-source rule.

WHAT BELONGS HERE
    Only the mount transform: where the hand's floating-base frame sits relative
    to the arm's end-effector link, and how it is rotated. Nothing about the hand
    itself (that is HandSpec) and nothing about the arm itself (ArmSpec), and as
    always nothing about control policy. Three owners meet in it: the arm's flange
    (``ArmSpec.flange``), the hand's mounting face (``HandSpec.mount_face_to_root``)
    and the pair's adapter (the mount file's ``translation.stack.adapter_thickness``)
    — or one measured vendor transform (``translation.measured``).

WHAT "VIRTUAL WRIST" MEANS HERE
    The retargeting emits a floating base per hand; that frame is what the
    policy tracks. Which physical body it coincides with is hand-specific -
    the merged flange body for SharpA, the palm link for Wuji - so the translation
    differs per hand even on the same arm. That is exactly why the value is
    pair-keyed rather than arm-keyed.
"""

from __future__ import annotations

from dataclasses import dataclass

from dexengine.core import embodiment as _E
from dexengine.core.arms import ARM_REGISTRY
from dexengine.core.floating import identity_mounts


@dataclass(frozen=True)
class RenderVisualPart:
    """A part bolted between the arm and the hand on the real robot that is drawn **on the render body only** — a mesh
    with a material, never a collider, a mass or a joint (the training body, the judge and the bake never see it, so
    adding or removing one changes no training condition). It belongs to the (arm, hand) pair for the same reason the
    mount transform does — an adapter plate is neither the arm's nor the hand's.

    Frames: the part frame sits at ``ee_to_part`` in the arm END-EFFECTOR link frame (the frame ``MountSpec`` uses),
    on every side of the arm; the mesh is drawn at ``part_to_mesh`` in the part frame, its file's vertices in metres at
    scale 1. Quaternions wxyz. The one reader is ``dexengine/robots/usd/render_parts.py`` (authored by
    ``dexengine/robots/usd/render_body.py`` into the render body · allowed by the render-plate parity
    ``render_plate.parity`` · written by ``dexengine/robots/urdf/merge.py`` as visual-only fixed links into the render
    URDF ``<combo>_render.urdf`` — never into the merged URDF the bake, IK, judge and placement read).
    ``sources`` is the record of where each value was read — (field, citation) — a field, not a comment, so the check
    and the marker can print it."""

    name: str
    """Prim name under the EE link's ``visuals`` (and the marker's word)."""

    mesh: str
    """Mesh file, repository-relative with an explicit first part: ``source/…`` (a tracked asset of the code tree) or a
    data-layer top (``data/…`` — the vendor stages, read from the data root); any other prefix dies
    (``render_visual_parts.mesh_file``)."""

    mesh_sha256: str
    """sha256 of the mesh file — a different file dies (the pose is only valid for these vertices)."""

    ee_to_part_pos: tuple[float, float, float]
    ee_to_part_quat: tuple[float, float, float, float]
    part_to_mesh_pos: tuple[float, float, float]
    part_to_mesh_quat: tuple[float, float, float, float]
    color: tuple[float, float, float]
    metallic: float
    roughness: float
    sources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MountSpec:
    """One (arm, hand) mounting.

    Translation is expressed in the arm END-EFFECTOR link frame (``ArmSpec.ee_body_name``). Rotations are forward
    quaternions (wxyz) ``q`` such that ``wrist_quat = ee_quat (x) q``; the inverse is the conjugate.
    """

    arm: str
    hand: str

    ee_to_wrist_offset: tuple[float, float, float]
    """EE-link -> virtual-wrist translation, metres, in the EE-link frame."""

    ee_to_wrist_rot_right: tuple[float, float, float, float]
    """EE-link -> virtual-wrist rotation (wxyz) for the RIGHT arm."""

    ee_to_wrist_rot_left: tuple[float, float, float, float]
    """Same for the LEFT arm. Not derivable from the right one in general — a
    mirrored arm negates the yaw, but an arm whose left flange is NOT the mirror
    of its right would not, so both are stated."""

    provisional: bool = False
    """True while no physical adapter exists and the numbers are COMPUTED
    (triad alignment / vendor attach frame), not measured from real hardware.
    A provisional mount is good enough to train and compare, and every result
    produced on one must say so — re-measure and re-train when the adapter is
    designed."""

    provenance: str = ""
    """One line saying where the numbers came from (script + date), so the day
    a value looks wrong nobody has to guess how it was made."""

    assumptions: tuple[str, ...] = ()
    """The **reasons** for ``provisional``, as items — one line per value that is assumed rather than measured (what ·
    how much · who decided). A field rather than a comment because comments never reach run logs or checkers. When an
    assumption is replaced by a measurement, delete its item; when none are left, clear ``provisional`` too."""

    render_visual_parts: tuple[RenderVisualPart, ...] = ()
    """Mount parts drawn only on the render body (and the render URDF ``<combo>_render.urdf``) (``RenderVisualPart``)
    — empty means the render body is the plain one and there is no render URDF.
    The mount transform (the three fields above) does not depend on this field — a part never moves the hand, it only
    fills the visible gap (render body only · training conditions unchanged)."""

    ee_to_wrist_offset_left: tuple[float, float, float] | None = None
    """LEFT arm EE-link -> virtual-wrist translation. ``None`` = shared ``ee_to_wrist_offset`` (unchanged hands).
    Read only through ``ee_to_wrist_offset_for(side)``."""

    def ee_to_wrist_offset_for(self, side: str) -> tuple[float, float, float]:
        """EE-link -> virtual-wrist translation of a side (``"right"`` | ``"left"``) — the one reader of the
        translation."""
        if side == "right":
            return tuple(self.ee_to_wrist_offset)
        if side == "left":
            return tuple(
                self.ee_to_wrist_offset if self.ee_to_wrist_offset_left is None else self.ee_to_wrist_offset_left
            )
        raise ValueError(f"MountSpec({self.arm},{self.hand}).ee_to_wrist_offset_for: side={side!r} — right|left only")


def parent_link_pose(offset, quat, joint_xyz):
    """EE → the vendor link one fixed joint ABOVE the hand root a mount places (joint origin ``joint_xyz``, rpy 0):
    the root sits at (``offset``, ``quat``) in the EE frame, so its parent sits at ``offset − R(quat)·joint_xyz`` with
    the same rotation (the loader's ``ee_to_part: hand_mount_face``). Pure Python (this module stays import-light)."""
    w, qx, qy, qz = quat
    vx, vy, vz = joint_xyz
    tx, ty, tz = 2 * (qy * vz - qz * vy), 2 * (qz * vx - qx * vz), 2 * (qx * vy - qy * vx)  # t = 2 q×v
    rv = (vx + w * tx + qy * tz - qz * ty, vy + w * ty + qz * tx - qx * tz, vz + w * tz + qx * ty - qy * tx)
    return tuple(o - r for o, r in zip(offset, rv, strict=True)), tuple(quat)


#: Registered mounts: one spec file per (arm, hand) (configs/robots/mounts/<arm>+<hand>.yaml — the loader composes
#: the translation from the arm's flange, the mount's adapter and the hand's mount_face_to_root, or takes a measured
#: one; every hand file has at least one mount file, or the load dies), plus the floating hand's identity mounts
#: (dexengine.core.floating owns those values).
MOUNT_REGISTRY: dict[tuple[str, str], MountSpec | None] = {
    **_E.default_catalog().mounts,
    **identity_mounts(MountSpec),
}


def _require_arm(arm: str | None, who: str) -> str:
    """The arm is an identity input — no default.

    ``default_arm()`` answered "the only
    registered arm" and raised once there were two. With the legacy arm gone there
    is one again, so it would have **silently re-armed** and answered ``"rby1"``
    for callers that never said which arm they have. Deleted instead; every live
    caller already passes ``arm``.
    """
    if not arm:
        raise ValueError(
            f"dexengine.core.mounts.{who}: arm is required (no default) — pass arm=... "
            f"explicitly. Registered arms: {sorted(k for k, v in ARM_REGISTRY.items() if v is not None)}"
        )
    return arm


def mount_for(hand: str, arm: str | None = None) -> MountSpec:
    """Return the mount for ``(arm, hand)``, or raise saying what is missing."""
    arm = _require_arm(arm, "mount_for")
    if (arm, hand) not in MOUNT_REGISTRY:
        known = sorted(f"{a}+{h}" for a, h in MOUNT_REGISTRY)
        raise ValueError(
            f"dexengine.core.mounts.mount_for: no entry for arm={arm!r} hand={hand!r}. "
            f"Known pairs: {known}. Add one to MOUNT_REGISTRY after MEASURING the "
            "adapter geometry — do not copy another pair's number."
        )
    spec = MOUNT_REGISTRY[(arm, hand)]
    if spec is None:
        raise ValueError(
            f"dexengine.core.mounts.mount_for: arm={arm!r} hand={hand!r} has no mount adapter "
            "yet. The hand spec exists but the geometry that bolts it to this arm "
            "does not, so there is nothing to return."
        )
    return spec
