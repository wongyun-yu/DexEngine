"""Static description of one arm variant (`ArmSpec`): joint naming and DoF, the per-class velocity / effort caps
(vendor limits) and the end-effector body the IK servos — plus the arm registry and the arm / hand of the running
process (`active_arm_name`, `active_hand_name`, `active_arm_spec`).

Hardware facts and the arm's own defaults (drive gains, rest pose, link groups); IK damping, retiming safety and
render settings live in the shared configs. Body generations (which baked USD / URDF variant a hand runs on) live in
`dexengine.core.arm_generations` and are re-exported here."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from dexengine.core import embodiment as _E
from dexengine.core.arm_generations import (  # noqa: F401 — re-exported
    ARM_CANON_GEN,
    ARM_GENERATIONS,
    ARM_RENDER_GEN,
    ARM_SIM_COLLIDER_REV,
    HAND_LIMIT_SUFFIX,
    RETIRED_PLATE_HINT,
    TRUNCATED_HAND_LIMITS,
    RobotGeneration,
    SimColliderRev,
    canon_variant,
    generation_of,
    hand_bodies,
    plate_variant,
    render_body_variant,
    retired_plate_why,
    sim_body_variant,
    sim_hand_limits,
    sim_usd_variant,
    urdf_variant,
)


@dataclass(frozen=True)
class ArmJointClass:
    """One group of arm joints that shares a vendor limit.

    ``suffix`` is the regex tail appended to the joint stem, e.g. ``left_arm_`` (Isaac matches with
    fullmatch semantics), e.g. ``"[0-2]"``. ``indices`` lists the joint numbers (in the vendor's index base) the
    suffix covers, so callers that need a flat per-joint array (retiming) can expand the same
    source of truth the actuator config uses.
    """

    suffix: str
    indices: tuple[int, ...]
    velocity_limit: float
    """rad/s, vendor URDF."""
    effort_limit: float
    """N·m, vendor URDF."""


@dataclass(frozen=True)
class ArmFlange:
    """The arm's tool attach frame, in the EE-link frame — the arm's part of every hand mount on it (``MountSpec``
    ``translation.stack``): the hand stack (adapter + the hand's ``mount_face_to_root``) is laid along ``approach``
    starting at ``xyz``."""

    xyz: tuple[float, float, float]
    """Attach frame origin (m) in the EE-link frame."""
    approach: tuple[float, float, float]
    """Unit direction, in the EE-link frame, the hand stack points along (away from the arm)."""


@dataclass(frozen=True)
class ExtraLink:
    """A link the skeleton builder adds to the vendor body (e.g. a head camera): its URDF ``<link>`` text (``{mesh}``
    is filled with the mesh path relative to the output directory) fixed to ``parent`` by joint ``joint``."""

    link_xml: str
    """Repository-relative file holding the ``<link>`` element."""
    mesh: str
    """Repository-relative mesh file the link references (``{mesh}``)."""
    joint: str
    parent: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class ArmSkeleton:
    """How the full-robot merge (``robots.urdf.merge.skeleton_build``) cuts the arm body out of its vendor URDF: the
    subtree at ``root_link`` without the links below each EE link, without ``drop_links`` and their joints, with the
    joints named ``fixed_joint_prefix*`` welded at ``posture`` (rad; a joint not listed is welded at 0), the
    ``extra_links`` added, and the root fixed to a ``world`` link by ``world_joint``."""

    vendor_urdf: str
    """Repository-relative vendor URDF."""
    root_link: str
    drop_links: tuple[str, ...]
    fixed_joint_prefix: str
    posture: dict[str, float]
    world_joint: str
    extra_links: tuple[ExtraLink, ...]


@dataclass(frozen=True)
class JointDrive:
    """Default PD drive of a joint group (Isaac ImplicitActuator): ``kp`` [N·m/rad], ``kd`` [N·m·s/rad]."""

    kp: float
    kd: float


@dataclass(frozen=True)
class HeadDrive(JointDrive):
    """The head group's drive, its effort [N·m] / velocity [rad/s] caps and the pitch joint's (the last head joint)
    target [deg] — not driven by the policy, a PD target holds it (the head camera pitches with it)."""

    effort_limit: float
    velocity_limit: float
    pitch_deg: float


#: Keys of ArmSpec.limb_joints.
LIMB_POINTS = ("shoulder", "elbow", "wrist")


@dataclass(frozen=True)
class ArmSpec:
    """Static description of one arm (the arm spec file configs/robots/arms/<arm>.yaml, read by
    dexengine.core.embodiment)."""

    name: str
    """Identifier, e.g. ``"rby1"``."""

    joint_prefix: str
    """Joint names are ``f"{joint_prefix}_{side}_joint{n}"``."""

    n_joints: int
    """DoF per arm."""

    sides: tuple[str, ...]
    """Side tokens in the joint names, in the order the asset lists them."""

    joint_classes: tuple[ArmJointClass, ...]
    """Vendor limit groups. Must cover 1..n_joints exactly once."""

    ee_body: str
    """Bare name of the body the arm IK servos, without side or prefix.

    ⚠ NOT a usable body name on its own: ``f"{side}_{ee_body}"`` does not produce RB-Y1's
    ``link_right_arm_6``.

    Use ``ee_body_name(side)``, which composes it through ``body_name_fmt``.
    """

    joint_name_fmt: str
    """How this vendor spells a joint name. Placeholders: ``{prefix}`` ``{side}`` ``{index}``.
    Required — no default (a default spelling would be inherited silently by a new spec).

    WHY THIS IS A FIELD: vendors spell the same concept differently. RB-Y1 spells it
    ``right_arm_0`` — side FIRST, no prefix, a different separator word, and
    0-based. A hardcoded spelling could describe exactly one arm.

    The part before ``{index}`` is the "stem" every regex/filter is built from,
    so a fmt whose index is not last will not work — asserted in __post_init__.
    """

    index_base: int
    """First joint index in the vendor's numbering (RB-Y1: 0). Required."""

    body_name_fmt: str
    """How this vendor spells a LINK name. Same placeholders as joint_name_fmt. Required.

    A separate field because the two are not the same rule even within one
    vendor: RB-Y1 changes the word order entirely (``right_arm_0`` vs
    ``link_right_arm_0``). Deriving links from the joint fmt would break.
    """

    ee_body_index: int
    """Link index of the end effector (RB-Y1 ``arm_6``). Required.

    Not ``n_joints``: the link and joint numberings need not line up (an arm may
    have link0..link7 for joint1..joint7).
    """

    base_pos: tuple[float, float, float] | None = None
    """World spawn position. ``None`` = the value the scene holds (ROBOT_DEFAULT_POS).

    Why per arm: a short arm must be **raised on a pedestal** to reach the desk. RB-Y1 is tall on its own
    wheels; raising it would overlap the bent torso and correct twice — z=0 is canonical.
    """

    head_camera_prim_relpath: str = ""
    """Robot-relative path of the ZED hardware prim on the head. "" = no head camera.

    The sim camera spawns as a **child** of this prim, so it moves with the camera hardware.
    The importer merges fixed links into their parents, so the path does not simply join URDF
    link names — record the value **checked by opening the built USD**.
    """

    floating: bool = False
    """Floating hand only: a body with no arm whose hand root is a **free-root body**.
    False = a normal arm. True → ``n_joints`` is 0 and the EE = the hand root body, whose name comes from a rule,
    not a table: ``floating.root_body(hand, side)`` (the child of the mount joint in each hand's canonical
    combo URDF — it differs per hand: SharpA ``{side}_hand_flange`` · wuji2 ``{side}_wrist``).
    A caller asking for arm joint regexes dies loudly."""

    head_joint_names: tuple[str, ...] = ()
    """Non-arm joints that ride along in this arm's skeleton (e.g. a 2-DOF
    head). The combo articulation builder gives them their own actuator group;
    an arm without one leaves this empty and gets no head group — a hardcoded
    head-group access would KeyError on such an arm, which is why this is spec data.
    """

    flange: ArmFlange | None = None
    """The tool attach frame (``ArmFlange``). None = no flange stated (no mount of this arm can use the stack rule)."""

    skeleton: ArmSkeleton | None = None
    """How the full-robot merge builds this arm's body (``ArmSkeleton``). None = not buildable (the floating hand)."""

    torso_ruler: bool = False
    """True = the arm has torso colliders (``arms.ARM_SIM_COLLIDER_REV`` torso lists) and placement guards the hands
    against them; a missing list then dies. False = the body has no torso to guard (marker [TORSO_RULER] off)."""

    torso_link_prefixes: tuple[str, ...] = ()
    """Body links outside the two arms that form the placement TORSO group (``is_torso_link``): the skeleton root link
    and every link whose name starts with one of these."""

    idle_qpos: dict[str, tuple[float, ...]] = field(default_factory=dict)
    """Rest pose of an idle arm (the inactive side of a one-hand clip), per side, in joint order (rad) — read through
    ``placement.collision.idle_arm``. Empty = no rest pose stated (an idle side then dies)."""

    reach_clearance_links: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """Reach-map desk clearance classes (``placement.reach_map.build``), arm link indices: ``arm`` = held above the desk
    top (elbow to forearm) · ``hand`` = may go lower together with the hand (wrist). Other links are in neither."""

    arm_drive: JointDrive | None = None
    """Default PD drive of the arm joints (launches may override kp/kd: --arm_kp/--arm_kd). None = no arm joints."""

    head_drive: HeadDrive | None = None
    """Drive and pitch target of the ``head_joint_names`` group; None exactly when there is no head joint."""

    limb_joints: dict[str, int] = field(default_factory=dict)
    """Arm joints (this numbering) whose frame origins are the shoulder, elbow and wrist points (LIMB_POINTS) of the
    placement judge's natural posture (upper arm and forearm lengths). Empty only without arm joints."""

    port_base_pos: tuple[float, float, float] | None = None
    """Base position the ported (v2-era) placements were adopted at; the judge shifts them by the base move. None = no
    ported placements."""

    generation_pitch_joint: str | None = None
    """Skeleton joint whose origin pitch is the body generation fingerprint (computed by the loader: the posture joint
    bound to ``{generation: torso3_pitch_rad}``). None = no such joint (no generation axis)."""

    def __post_init__(self) -> None:
        for fld, val in (("joint_name_fmt", self.joint_name_fmt), ("body_name_fmt", self.body_name_fmt)):
            if not val.endswith("{index}"):
                raise ValueError(
                    f"{self.name}: {fld} must end with '{{index}}' — every regex and "
                    f"name filter here is built from the part before it. Got {val!r}"
                )
        lo = self.index_base
        expect = list(range(lo, lo + self.n_joints))
        covered = sorted(i for c in self.joint_classes for i in c.indices)
        if covered != expect:
            raise ValueError(
                f"{self.name}: joint_classes cover {covered}, expected "
                f"{expect[0]}..{expect[-1]} (index_base={lo}, n_joints={self.n_joints})"
            )
        bad = {s: len(q) for s, q in self.idle_qpos.items() if s not in self.sides or len(q) != self.n_joints}
        if bad:
            raise ValueError(f"{self.name}: idle_qpos {bad} — one {self.n_joints}-value pose per side of {self.sides}")
        if self.reach_clearance_links and set(self.reach_clearance_links) != {"arm", "hand"}:
            raise ValueError(
                f"{self.name}: reach_clearance_links keys {sorted(self.reach_clearance_links)} ≠ arm, hand"
            )
        if (self.arm_drive is None) == (self.n_joints > 0) or (self.head_drive is None) == bool(self.head_joint_names):
            raise ValueError(f"{self.name}: arm_drive / head_drive are stated exactly when arm / head joints exist")
        limbs = self.limb_joints
        if self.n_joints and (
            set(limbs) != set(LIMB_POINTS) or not all(lo <= k <= self.index_hi for k in limbs.values())
        ):
            raise ValueError(
                f"{self.name}: limb_joints {limbs} — keys {LIMB_POINTS}, each a joint {lo}..{self.index_hi}"
            )

    # --- naming, in ONE place ---------------------------------------------------------

    def _stem(self, side: str) -> str:
        """Everything before the joint index, e.g. ``left_arm_``."""
        return self.joint_name_fmt.format(prefix=self.joint_prefix, side=side, index="")

    @property
    def index_hi(self) -> int:
        """Last joint index (inclusive)."""
        return self.index_base + self.n_joints - 1

    def body_name(self, side: str, index: int) -> str:
        """Full link name for ``index`` on ``side``, in this vendor's spelling."""
        return self.body_name_fmt.format(prefix=self.joint_prefix, side=side, index=index)

    def body_side(self, link: str) -> str | None:
        """The side whose arm ``link`` belongs to (its name starts with that side's body stem, e.g.
        ``link_right_arm_``); None for any other link."""
        for side in self.sides:
            if link.startswith(self.body_name(side, "")):
                return side
        return None

    def is_torso_link(self, link: str) -> bool:
        """Whether ``link`` is in the TORSO group: the skeleton root link or a ``torso_link_prefixes`` name."""
        root = self.skeleton.root_link if self.skeleton is not None else None
        return link == root or link.startswith(self.torso_link_prefixes)

    @property
    def is_floating(self) -> bool:
        """Whether this body has no arm and its hand root is a free-root body (``floating``)."""
        return bool(self.floating)

    def ee_body_name(self, side: str, hand: str | None = None) -> str:
        """Full link name of the end effector -- what the IK actually servos.
        For a floating hand EE = hand root = a rule (floating.root_body — per hand): a caller that knows the hand
        (the stage-4 bake reads it from the combo name) passes ``hand``; a run env uses this process's DEXENGINE_HAND
        (dies when unset). A normal arm ignores ``hand``."""
        if self.is_floating:
            from dexengine.core.floating import root_body  # noqa: PLC0415 — floating imports this module

            return root_body(hand if hand is not None else active_hand_name(), side)
        return self.body_name(side, self.ee_body_index)

    # --- derived views: one source, several shapes ------------------------------------

    def joint_expr(self, side: str) -> str:
        """Regex matching all joints of one arm, e.g. ``left_arm_[0-6]``."""
        if self.is_floating:
            raise ValueError(
                f"{self.name}: a floating hand has no arm joints — a caller asking for joint_expr "
                "does not know the floating-hand branch yet (use arm_joint_exprs(side) or add it)."
            )
        return f"{self._stem(side)}[{self.index_base}-{self.index_hi}]"

    def arm_joint_exprs(self, side: str) -> list[str]:
        """Arm joint regexes of one side — ``[joint_expr(side)]`` for a normal arm,
        ``[]`` for a floating hand (find_joints accepts zero)."""
        return [] if self.is_floating else [self.joint_expr(side)]

    def joint_names_expr(self) -> list[str]:
        """``joint_names_expr`` for an ImplicitActuatorCfg covering every arm joint."""
        if self.is_floating:
            raise ValueError(
                f"{self.name}: a floating hand has no arm actuators — make_combo_articulation_cfg must skip this arm."
            )
        return [self.joint_expr(s) for s in self.sides]

    def joint_prefix_for(self, side: str) -> str:
        """``left_arm_`` — for ``name.startswith(...)`` filters."""
        return self._stem(side)

    def joint_names(self, side: str) -> list[str]:
        """Individual joint names in index order, e.g. ``left_arm_0..6``.

        Some call sites need explicit names rather than a regex (deploy ordering,
        per-joint lookups). Same source as :meth:`joint_expr`.
        """
        return [
            self.joint_name_fmt.format(prefix=self.joint_prefix, side=side, index=i)
            for i in range(self.index_base, self.index_base + self.n_joints)
        ]

    def any_joint_expr(self, side: str) -> str:
        """``left_arm_.*`` — matches every joint of one arm regardless of count.

        Used where a config wants "all arm joints" without pinning the DoF, e.g. the
        asset's default ``joint_pos`` map.
        """
        return f"{self.joint_prefix_for(side)}.*"

    def limit_map(self, which: str) -> dict[str, float]:
        """Per-class regex -> limit, for ``velocity_limit_sim`` / ``effort_limit_sim``.

        ``which`` is ``"velocity"`` or ``"effort"``. Key order follows sides-within-class,
        matching how the asset config was written by hand.
        """
        if which not in ("velocity", "effort"):
            raise ValueError(f"which must be 'velocity' or 'effort', got {which!r}")
        out: dict[str, float] = {}
        for c in self.joint_classes:
            val = c.velocity_limit if which == "velocity" else c.effort_limit
            for side in self.sides:
                out[f"{self._stem(side)}{c.suffix}"] = val
        return out

    def per_joint_limits(self, which: str) -> list[float]:
        """Flat length-``n_joints`` list ordered by joint index.

        This is the shape ``dexengine/placement/retime/warp.py`` needs (its ``VLIM``).
        Same numbers as :meth:`limit_map`, so the reference is retimed against exactly the
        caps the simulated actuator enforces.
        """
        vals: dict[int, float] = {}
        for c in self.joint_classes:
            v = c.velocity_limit if which == "velocity" else c.effort_limit
            for i in c.indices:
                vals[i] = v
        return [vals[i] for i in range(self.index_base, self.index_base + self.n_joints)]


# ===========================================================================
# Floating hand (the floating-hand rows of the paper's tables and the kinematic baseline). No arm: each hand's
# root (the child of the mount joint in the canonical combo URDF — floating.root_body) is a free-root body, and
# the wrist is moved by an external wrench (the physics is an experiment plugin, see dexengine.rl.tasks.hooks).
# The joint/body name shapes are placeholders that only satisfy the __post_init__ contract (end with ``{index}``) —
# a floating hand has no such joints, and a caller asking for arm joint regexes dies loudly in joint_expr. There is
# no generation axis, collider revision or render body (absent from the tables = "").
#: The floating-hand value of ``DEXENGINE_ARM`` — owned by this constant (core.floating · core.paths read it).
FLOATING_ARM = "floating"


def _floating_spec() -> ArmSpec:
    return ArmSpec(
        name=FLOATING_ARM,
        joint_prefix=FLOATING_ARM,
        n_joints=0,
        sides=("left", "right"),
        joint_classes=(),
        ee_body="hand_root",  # placeholder — the EE name comes from ee_body_name → floating.root_body
        base_pos=None,
        joint_name_fmt="{side}_floating_none_{index}",
        index_base=0,
        body_name_fmt="{side}_floating_none_{index}",
        ee_body_index=0,
        floating=True,
    )


FLOATING_SPEC = _floating_spec()

#: Registered arms: one spec file each (configs/robots/arms/<arm>.yaml — rby1 is sim-trainable, not deployable: this
#: repo has no deploy bridge), plus the floating hand, which enters only through DEXENGINE_ARM=floating.
ARM_REGISTRY: dict[str, ArmSpec | None] = {**_E.default_catalog().arms, FLOATING_SPEC.name: FLOATING_SPEC}


# ===========================================================================
# "Which arm and hand is this process?" — the **single accessor**
#
# Why: when several modules each answer this with their own default, a diagnostic path imported without a launcher
# can point at two different robots within one process. Joint naming differs per arm, so that split silently
# becomes an empty joint list — a diagnostic that is "on but shows nothing".
#
# Rule: **this file owns the answer.** Other files do not restate the value; they call the functions below.
ARM_ENV = "DEXENGINE_ARM"
HAND_ENV = "DEXENGINE_HAND"
# The active arm and hand have no default: a silent default for an identity input sends a run without a stated
# hand to one hand while later stages of the same pipeline use the stated one. The launcher owns the value; the
# code dies when it is missing.
CANON_ARM: str | None = None  # no default — the launcher states it with DEXENGINE_ARM
CANON_HAND: str | None = None  # no default — the launcher states it with DEXENGINE_HAND
_ACTIVE_SAID: set[str] = set()


def _active_name(env: str, what: str) -> str:
    """The value of `env`. Unset or empty **dies loudly** (no silent default).

    No module calls these two functions at top level (every call is inside a function), so dying leaves
    imports intact and limits the failure to paths that actually use the value.
    """
    v = os.environ.get(env, "").strip()
    if not v:
        raise RuntimeError(
            f"[{what}] {env} is not set — {what} is **stated at run time** (no default). "
            f"The launcher promotes --arm/--hand from EXTRA to env; "
            f"when calling directly, pass {env}=<value>."
        )
    if env not in _ACTIVE_SAID:
        print(f"[{what}] {env}={v}", file=sys.stderr)
        _ACTIVE_SAID.add(env)
    return v


def active_arm_name() -> str:
    """The name of the arm this process handles. Dies when unset."""
    return _active_name(ARM_ENV, "ARM")


def active_hand_name() -> str:
    """The name of the hand this process handles. Dies when unset."""
    return _active_name(HAND_ENV, "HAND")


def active_arm_spec() -> ArmSpec:
    """The ArmSpec of the arm this process handles. An unregistered name **dies loudly**.

    A silent fallback such as `ARM_REGISTRY.get(env) or <default arm>` would hide a typo or an unregistered name
    as a run on the default arm. A caller that needs a substitute uses ARM_REGISTRY directly.
    """
    name = active_arm_name()
    spec = ARM_REGISTRY.get(name)
    if spec is None:
        raise RuntimeError(
            f"arms.active_arm_spec: {ARM_ENV}={name!r} is not in ARM_REGISTRY"
            f" (registered: {sorted(k for k, v in ARM_REGISTRY.items() if v is not None)}). "
            "A stub does not enter the registry until it passes a smoke test."
        )
    return spec
