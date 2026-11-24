"""Shared naming resolution utilities for dataset and robot hand USDs.

Handles two naming mismatches:
1. dataset vs robot body/joint naming conventions
2. Placeholder left hand (right-hand USD re-used) where all names are ``right_*``

All resolvers follow the pattern: try ``{side}_*`` first, fall back to ``right_*``.
"""

from __future__ import annotations

from dexengine.core.hands import HAND_REGISTRY as _HAND_REGISTRY
from dexengine.core.hands import SHARPA_WAVE_RIGHT as _SHARPA
from dexengine.core.hands import fingertip_site_offsets_for as _fingertip_site_offsets_for
from dexengine.core.mounts import mount_for as _mount_for

# ==========================================================================
# Joint / body name tables
# SINGLE SOURCE OF TRUTH: every HAND-SPECIFIC table below is DERIVED from its ``HandSpec``, so one hardware fact
# never lives in two files (adding a hand would mean editing both, and forgetting one is silent).
# third hand would have meant editing both, and forgetting one is silent.
#
#   Add a hand  ->  fill a HandSpec. Do NOT add tables here.
#
# What is NOT derived, and why:
#   * the virtual-wrist 6 DOF  -- a naming CONVENTION of the floating-hand
#     rig, identical for every hand, so it belongs to no spec.
#   * the DATASET (rig MJCF) tables -- a different naming AXIS for the same
#     hardware, not a different hand. HandSpec describes the USD/sim side only.
# ==========================================================================

# Virtual-wrist DOFs of the floating-hand rig (hand-independent by construction).
_VIRTUAL_WRIST_RIGHT = [
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_rot_x",
    "right_rot_y",
    "right_rot_z",
]


def _tip_suffixes(spec) -> list[str]:
    """``("right_hand_thumb_4", ...)`` -> ``["hand_thumb_4", ...]``."""
    return [n.split("right_", 1)[1] for n in spec.tip_body_names_right]


# --- dataset (rig MJCF) naming: a separate AXIS, not a separate hand ----
_RIGHT_INDEPENDENT_JOINTS_DATASET = _VIRTUAL_WRIST_RIGHT + [
    "right_thumb_proximal_yaw_joint",
    "right_thumb_proximal_pitch_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
]
_DATASET_TIP_SUFFIXES = [
    "thumb_distal",
    "index_intermediate",
    "middle_intermediate",
    "ring_intermediate",
    "pinky_intermediate",
]

# --- SharpA Wave (22-DoF fully actuated) — DERIVED from HandSpec -----------
# Joint order = rig MJCF / vendor URDF qpos order.
# Exported: dexengine.rl.tasks.mdp.trajectory.manager imports SHARPA_FINGER_JOINT_SUFFIXES.
SHARPA_FINGER_JOINT_SUFFIXES = [n.split("right_", 1)[1] for n in _SHARPA.finger_joint_names_right]

# Fingertip SITE offsets in the tip-BODY (*_DP) local frame, metres.
# Source: vendor URDF (SharpA package sharpa-urdf-usd-xml/wave_01/{right,left}_
# sharpa_wave/*_with_flange.urdf; the same chain is in the merged training URDF):
#   DP -> elastomer fixed joint: xyz (0,0,0), rpy (pi/2, -pi/2, 0)
#   elastomer -> *_fingertip fixed joint: xyz (0, -d, 0), rpy (pi/2, pi/2, 0)
# Composing (URDF rpy = extrinsic XYZ, R = Ry(-pi/2)Rx(pi/2)) maps the
# elastomer-frame (0, -d, 0) to (+d, 0, 0) in the DP frame — the physical
# fingertip lies d metres along DP-local +x. Same numbers and same sign for
# the LEFT SharpA URDF (verified; no mirroring). d matches
# core.hands.SHARPA_WAVE_RIGHT.fingertip_site_offsets:
#   thumb d = 0.027611 (URDF 0.0276109346321216)
#   fingers d = 0.026009 (URDF 0.02600925742625xx)
#
# There is no SharpA-specific constant here any more. The values are
# read out of whichever HandSpec the asset actually carries, by _site_offsets_of
# below -- so a newly ported hand that fills the field gets its numbers USED.
# Until then this field was SharpA-only in practice: every other hand fell into a
# hardcoded all-zero branch, and a porter filling it in a new spec would have seen
# it silently ignored.


# ==========================================================================
# Known-hardware gate — the ONE place that answers "is this a hand we know?"
#
# Each resolver below keeps its own lookup order (dataset naming, per-side names, placeholder-left fallback: all
# deliberate, see their docstrings). Only the LAST step goes through this gate: instead of
# silently handing an unrecognised hand RH56F1's numbers, they ask this gate
# and raise. Detection uses each spec's OWN declared joints, so porting a hand
# means adding a HandSpec — no name table to update here.
# ==========================================================================

# The hands the resolvers know, DERIVED from HAND_REGISTRY.
#
# ONE list, not two. This used to be a hand-maintained tuple here PLUS a separate
# per-hand candidate list spelled out inside three resolvers, so porting a hand
# meant four edits in lockstep, which was the real per-hand cost.
# Registering the spec in HAND_REGISTRY is now the whole
# change, and a hand deliberately left unregistered (a draft spec) stays invisible
# to the resolvers, which is what keeps an unregistered hand failing fast.
_KNOWN_HAND_SPECS = tuple(s for s in _HAND_REGISTRY.values() if s is not None)

# Detection keys on each spec's FIRST finger joint, so those must be distinct.
# Vendors have never collided -- each name carries its own vendor's convention --
# but a collision would silently hand one hand another's geometry, and "silently
# wrong geometry" is the exact failure the gate below exists to stop. Checked at
# import so a bad registration cannot reach a training run.
_FIRST_JOINTS = [s.finger_joint_names_right[0] for s in _KNOWN_HAND_SPECS]
if len(set(_FIRST_JOINTS)) != len(_FIRST_JOINTS):
    raise ValueError(
        "naming: two registered hands share a first finger joint name "
        f"({_FIRST_JOINTS}). Hand detection keys on that name, so it must be "
        "unique per hand. Rename in the HandSpec to match the vendor URDF."
    )
del _FIRST_JOINTS


# ==========================================================================
# Probe tables — the candidate namings each resolver tries, in order.
#
# ORDER: dataset (rig MJCF naming) first, then each registered hand's USD naming in registry order.
# ==========================================================================

# Independent joints: the 12-DoF form (virtual wrist + fingers) is tried before
# the fingers-only form, because the full robot drops the virtual wrist to arm IK
# and would otherwise match the shorter table first. Dataset naming has no
# fingers-only form -- it only ever appears on the floating hand.
_INDEP_PROBES_FULL = (_RIGHT_INDEPENDENT_JOINTS_DATASET,) + tuple(
    _VIRTUAL_WRIST_RIGHT + list(_s.finger_joint_names_right) for _s in _KNOWN_HAND_SPECS
)
_INDEP_PROBES_FINGERS = tuple(list(_s.finger_joint_names_right) for _s in _KNOWN_HAND_SPECS)

# Fingertip bodies: suffixes, joined to a side prefix at call time.
_TIP_SUFFIX_PROBES = (_DATASET_TIP_SUFFIXES,) + tuple(_tip_suffixes(_s) for _s in _KNOWN_HAND_SPECS)


def _for_side(names, side: str) -> list[str]:
    """Re-prefix a right-hand name table for ``side``."""
    return [n.replace("right_", f"{side}_", 1) for n in names]


def _hand_joints(spec, side: str) -> list[str]:
    """The spec's finger joints, mapped to ``side``."""
    return [n.replace("right_", f"{side}_", 1) for n in spec.finger_joint_names_right]


def _asset_has_known_hand(asset) -> bool:
    """True if ANY known hand is present on EITHER side (or in dataset naming).

    Deliberately side-agnostic: a right-only floating hand queried for "left",
    and the placeholder-left build, are legitimate configurations that the
    resolvers handle by their own fallbacks. Only genuinely unknown hardware
    should trip the gate.
    """
    joint_set = set(asset.data.joint_names)
    for spec in _KNOWN_HAND_SPECS:
        for side in ("right", "left"):
            names = _hand_joints(spec, side)
            if names and names[0] in joint_set:
                return True
    for side in ("right", "left"):  # dataset (rig MJCF) naming = RH56F1 hardware
        if any(n.replace("right_", f"{side}_", 1) in joint_set for n in _RIGHT_INDEPENDENT_JOINTS_DATASET[6:]):
            return True
    return False


def _require_known_hand(asset, side: str, what: str) -> None:
    """Raise unless the asset carries a hand we have a HandSpec for."""
    if _asset_has_known_hand(asset):
        return
    raise ValueError(
        f"naming.{what}: no known hand found on this asset (side='{side}'). "
        f"Known: {[s.name for s in _KNOWN_HAND_SPECS]}. Joints seen: "
        f"{sorted(set(asset.data.joint_names))[:6]}...\n"
        "Porting a hand? Fill a HandSpec (dexengine/core/hands) and register it in "
        "HAND_REGISTRY. This raises ON PURPOSE: falling back to another hand's "
        "joint names and mount offset here would give silently wrong "
        "geometry."
    )


def _spec_on_side(asset, side: str):
    """The HandSpec whose finger joints appear on ``side``, or None.

    PER-SIDE ONLY, deliberately with no cross-side fallback. On the mixed full
    robot (right SharpA + left RH56F1) a fallback would hand the left RH56F1
    hand SharpA's geometry. Returning None here lets each caller apply its own
    documented default for "this side carries no hand I recognise".
    """
    joint_set = set(asset.data.joint_names)
    for spec in _KNOWN_HAND_SPECS:  # most specific first
        names = _hand_joints(spec, side)
        if names and names[0] in joint_set:
            return spec
    return None


def _site_offsets_of(spec, side: str) -> tuple[tuple[float, float, float], ...]:
    """A spec's fingertip-SITE offsets for ``side`` as ``(n_tips, 3)`` metres, tip-local.

    The table's shapes, the empty-table default (tip BODY ORIGIN) and the per-side rule live in ONE place,
    ``core.hands.fingertip_site_offsets_for``, shared with ``dexengine/robots/hand_rig.py``. Sides matter: the XHand
    Pro left thumb pad sits on −y of its distal link where the right one sits on +y, so reusing the right value on
    the left puts the pad 62.5 mm off.
    """
    return _fingertip_site_offsets_for(spec, side)


def resolve_fingertip_site_offsets(asset, side: str) -> tuple[tuple[float, float, float], ...]:
    """Per-finger fingertip-SITE offset in the tip-BODY local frame ((n, 3), m).

    Read from the HandSpec the asset actually carries on ``side`` (see
    ``_site_offsets_of`` for the accepted table shapes). SharpA Wave is the case
    that needs it: its tip bodies are the ``*_DP`` distal phalanges whose ORIGINS
    sit ~2.6 cm proximal of the physical fingertip, so object-distance quantities
    would otherwise measure from the wrong point.

    A side carrying no recognised hand returns all-zero offsets = tip BODY
    ORIGIN, which is the historical RH56F1 behaviour, after the known-hardware
    gate confirms the asset is a hand we know at all.
    """
    spec = _spec_on_side(asset, side)
    if spec is not None:
        return _site_offsets_of(spec, side)
    _require_known_hand(asset, side, "resolve_fingertip_site_offsets")
    return ((0.0, 0.0, 0.0),) * 5


# link7 -> virtual-wrist TRANSLATION (in link7 frame) of the SharpA hand on the FIRST arm
# this repository trained (its EE link was ``link7``). The ROTATION is NOT here and is NOT
# side-symmetric (right Rz(-90deg), mirrored left Rz(+90deg); resolve_ee_to_wrist_rot).
# SharpA: link7->flange = 0.0655. FRAME-NAMING FIX (audit
# 11/12f): the rig's SharpA floating base is at the FLANGE (the vendor merged
# flange/wrist/C_MC into one body NAMED right_hand_C_MC whose ORIGIN is the flange), so the
# virtual wrist is the FLANGE, NOT true C_MC; the prior 0.0850 (link7->C_MC) over-extended the
# wrist by 2.95 cm.
#
# these were DERIVED from MOUNT_REGISTRY's entry for that first arm + SharpA. The arm left the repository, so the
# entry is gone and the values are literal copies of what the derivation evaluated (repr bit-identical). Names are
# kept (``link7_*`` helper/npz names are a contract). Nothing live reads them for control: every registered pair
# resolves its mount through ``resolve_link7_to_wrist_offset`` / ``resolve_ee_to_wrist_rot`` from the active (arm,
# hand).
LINK7_TO_WRIST_OFFSET_SHARPA = (0.0, 0.0, 0.0655)


def _active_arm_name() -> str:
    """Name of the arm this process builds. Unset = the canonical rby1 (the same default as robots.selection).

    Does not call robots.selection.resolve_robot_selection(): this module is evaluated at import and that function
    raises on an unregistered combo, which would take the normal path down too. dexengine.core.arms.active_arm_name()
    does not look at the registry, so it never raises, and the canonical default still comes from one place.
    """
    from dexengine.core.arms import active_arm_name

    return active_arm_name()


def resolve_link7_to_wrist_offset(asset, side: str) -> tuple[float, float, float]:
    """Resolve the fixed link7->virtual-wrist translation for ``side`` hand.

    The virtual wrist is the rig floating-base frame (hand_base for RH56F1, C_MC for SharpA), so the translation
    differs per hand: it is looked up in MOUNT_REGISTRY for whichever hand ``side`` actually carries. A hand without
    a spec fails here even past the known-hardware gate (no fallback to another hand's mount).
    """
    spec = _spec_on_side(asset, side)
    if spec is not None:
        return tuple(_mount_for(spec.name, _active_arm_name()).ee_to_wrist_offset_for(side))
    _require_known_hand(asset, side, "resolve_link7_to_wrist_offset")
    raise ValueError(
        f"naming.resolve_link7_to_wrist_offset: no HandSpec found for the hand on side={side!r} — "
        "register this (arm, hand) in MOUNT_REGISTRY."
    )


def resolve_ee_to_wrist_rot(asset, side: str) -> tuple[float, float, float, float]:
    """EE (flange) -> wrist (rig palm) mounting **rotation** (wxyz) of the active (arm, hand) pair.

    Same rule as the translation (resolve_link7_to_wrist_offset): find the spec of the hand that side carries and use
    the pair's MOUNT_REGISTRY value. The registered values match the IK cache's back-solved M on every combo and side
    (<= 0.24°); the rby1 mount adds roll 180° (sharpa) / ±170.5° (wuji). A side without a spec fails here even past
    the known-hardware gate (no fallback to another arm's mount).
    """
    spec = _spec_on_side(asset, side)
    if spec is not None:
        m = _mount_for(spec.name, _active_arm_name())
        return tuple(m.ee_to_wrist_rot_left if side == "left" else m.ee_to_wrist_rot_right)
    _require_known_hand(asset, side, "resolve_ee_to_wrist_rot")
    raise ValueError(
        f"naming.resolve_ee_to_wrist_rot: no HandSpec found for the hand on side={side!r} — "
        "register this (arm, hand) in MOUNT_REGISTRY."
    )


# ==========================================================================
# Resolution functions
# ==========================================================================


def is_arm_ee_body(name: str) -> bool:
    """Is this body the **active arm's EE link**, i.e. full-robot mode?

    The palm hangs off the EE through a fixed joint and the importer merges it into the EE, so a palm body named like
    the EE means full-robot mode (a hard-coded `"link7" in name` would reject every RB-Y1 body, EE=link_*_arm_6).
    """
    from dexengine.core.arms import ARM_REGISTRY as _ARM_REGISTRY

    _arm = _ARM_REGISTRY.get(_active_arm_name())
    return bool(_arm) and name in (_arm.ee_body_name("right"), _arm.ee_body_name("left"))


#: Palm point of the floating hand at the same physical point as the arm body. Consumers: the palm rewards and
#: contact-aware diagnostics (pregrasp_palm, diag_palm_dist). Values: unset (default, byte-identical) | arm_ee.
#: On the arm body the palm body is the arm EE link (the hand root is merged into it through a fixed joint), whose
# origin
#: sits one mount translation from the wrist towards the arm (rby1 x wuji2: 15.5 cm). The floating hand measures the
# hand
#: root (the wrist): same name, different signal (pregrasp_palm measured 2.2x the arm body's). arm_ee measures the arm
# EE
#: origin as wrist + R_wrist · (−R_m^T t) (mount = the data-world arm's registered one).
#: The env read (DEXENGINE_FH_PALM_POINT) and its value check belong to the floating-hand plugin (palm_point_mode); the
#: offset math stays here. Entry for the rewards: hooks.floating_palm_point.


def _quat_conj_rotate(q_wxyz, v):
    """R(q)^T · v (q = unit quaternion, wxyz). Pure Python (this module is imported without torch)."""
    w, x, y, z = (float(c) for c in q_wxyz)
    x, y, z = -x, -y, -z  # conjugate
    vx, vy, vz = (float(c) for c in v)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))


def fh_palm_point_offset(asset, side: str):
    """The palm-point offset of ``DEXENGINE_FH_PALM_POINT`` (palm body frame · m) — None when off.
    The mode comes from the floating-hand plugin (``palm_point_mode()``: '' | 'arm_ee', it dies on any other value and
    on
    the arm body); without the plugin it is off. The mount is the data-world arm's registered one (MOUNT_REGISTRY)."""
    from dexengine.rl.tasks import hooks

    fh = hooks.floating_hand()
    if fh is None or not fh.palm_point_mode():
        return None
    arm = _active_arm_name()
    spec = _spec_on_side(asset, side)
    if spec is None:
        raise ValueError(f"{fh.PALM_POINT_ENV}: no HandSpec for side={side!r}")
    m = _mount_for(spec.name, fh.data_world_arm(arm))
    t = m.ee_to_wrist_offset_for(side)
    q = m.ee_to_wrist_rot_left if side == "left" else m.ee_to_wrist_rot_right
    return tuple(-c for c in _quat_conj_rotate(q, t))


def resolve_palm_body_id(asset, side: str) -> int:
    """Find palm body ID, handling floating hand, full robot, and placeholder left hand.

    Priority: {side}_wrist_rot_z → right_wrist_rot_z → the active arm's EE link.

    On the full robot the palm hangs off the arm EE through a fixed joint and the importer merges it into the EE, so
    **palm = that arm's EE link** (a hard-coded link7 name would fail RB-Y1 with 'Cannot find palm body'). The arm
    name comes from ARM_REGISTRY.
    """
    from dexengine.core.arms import ARM_REGISTRY as _ARM_REGISTRY

    body_names = asset.data.body_names
    _arm = _ARM_REGISTRY.get(_active_arm_name())
    _ee = [_arm.ee_body_name(side), _arm.ee_body_name("right")] if _arm else []
    for candidate in [
        f"{side}_wrist_rot_z",  # Floating hand (robot/dataset)
        "right_wrist_rot_z",  # Placeholder left hand
        f"{side}_hand_C_MC",  # Floating SharpA hand (rig base body)
        *_ee,  # Full robot: arm end-effector == merged palm
    ]:
        if candidate in body_names:
            return body_names.index(candidate)
    raise ValueError(f"Cannot find palm body for side='{side}' in {body_names}")


def resolve_independent_joints(asset, side: str) -> list[str]:
    """Auto-detect dataset vs robot joint naming and handle placeholder left hand.

    Tries full 12-DOF (6 wrist + 6 finger) first, then finger-only 6-DOF
    (for full robot where wrist is controlled by arm IK).

    Returns:
        List of 6 or 12 joint name strings in canonical order.
    """
    joint_set = set(asset.data.joint_names)

    # Requested side first; then the right_* tables again, which is how a
    # PLACEHOLDER LEFT hand (a right-hand USD mounted on the left) resolves.
    # For side="right" the second pass repeats the first and changes nothing.
    for probe_side in (side, "right"):
        for table in _INDEP_PROBES_FULL:
            names = _for_side(table, probe_side)
            if names[6] in joint_set and names[0] in joint_set:
                return names
        for table in _INDEP_PROBES_FINGERS:
            names = _for_side(table, probe_side)
            if names and names[0] in joint_set:
                return names

    # No last-resort return of another hand's joint names: an unknown hand fails.
    _require_known_hand(asset, side, "resolve_independent_joints")
    raise ValueError(
        f"naming.resolve_independent_joints: no independent joints found for side={side!r}. "
        f"Joints looked at: {sorted(joint_set)[:12]}… — register this hand's probe table in its HandSpec."
    )


def resolve_wrist_joints(asset, side: str, rot_only: bool = False) -> list[str]:
    """Resolve wrist joint names, handling floating hand, full robot, and placeholder.

    Returns empty list for full robot (no virtual wrist joints — arm IK handles wrist).

    Args:
        asset: The articulation asset.
        side: ``"right"`` or ``"left"``.
        rot_only: If True, return only rotation joints (3 DOF). Otherwise all 6.

    Returns:
        List of 0, 3, or 6 wrist joint name strings.
    """
    joint_set = set(asset.data.joint_names)
    if rot_only:
        primary = [f"{side}_rot_x", f"{side}_rot_y", f"{side}_rot_z"]
        fallback = ["right_rot_x", "right_rot_y", "right_rot_z"]
    else:
        primary = [
            f"{side}_pos_x",
            f"{side}_pos_y",
            f"{side}_pos_z",
            f"{side}_rot_x",
            f"{side}_rot_y",
            f"{side}_rot_z",
        ]
        fallback = [
            "right_pos_x",
            "right_pos_y",
            "right_pos_z",
            "right_rot_x",
            "right_rot_y",
            "right_rot_z",
        ]
    if primary[0] in joint_set:
        return primary
    if fallback[0] in joint_set:
        return fallback
    # Full robot: no virtual wrist joints (arm IK handles wrist)
    return []


_TIP_SAID: set = set()


#: The finger order of resolve_tip_body_names (and of the fingertip rewards' FINGER names).
TIP_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def resolve_tip_body_names(asset, side: str) -> list[str]:
    """Resolve fingertip body names for dataset vs robot naming + placeholder left hand.

    Returns:
        List of 5 fingertip body name strings in TIP_FINGERS order.
    """
    body_set = set(asset.data.body_names)

    # Requested side, then right_* again for a PLACEHOLDER LEFT hand.
    # all five present (a thumb name alone can match another family — wuji2 vs dataset)
    for probe_side in (side, "right"):
        for suffixes in _TIP_SUFFIX_PROBES:
            names = [f"{probe_side}_{s}" for s in suffixes]
            if body_set.issuperset(names):
                if (side, *names) not in _TIP_SAID:
                    _TIP_SAID.add((side, *names))
                    print(f"[TIP_BODIES] side={side} {names}", flush=True)
                return names

    raise ValueError(
        f"Cannot resolve fingertip body names for side='{side}'. "
        f"Tried {len(_TIP_SUFFIX_PROBES)} naming families "
        f"({[s.name for s in _KNOWN_HAND_SPECS]} + dataset) on both sides. "
        f"Available bodies: {asset.data.body_names}"
    )
