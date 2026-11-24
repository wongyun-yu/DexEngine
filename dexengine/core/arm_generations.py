"""Robot body generations of an arm: which baked USD / URDF variant a hand runs on in simulation and in
renders, the simulated collider revision, and the hands whose joint limits are truncated.

The rows are the arm spec files' (configs/robots/arms/<arm>.yaml keys ``generations`` and ``collider_rev``, read by
body_sections); this module owns their dataclasses, the rules that derive the other generations from them (collider
revision on top, hand joint limit variants) and the tables of the spec root of this tree."""

from __future__ import annotations

from dataclasses import dataclass

from dexengine.core import embodiment as _E
from dexengine.core.typed_spec import build, fail


# ===========================================================================
# Asset generations — the values that define a robot's "body" are **one set**
# ===========================================================================
# Different bakes of the same robot are called "generations". rby1 generations differ in torso pose: one line of
# the merged URDF, the `torso_3` joint origin pitch, sets the accumulated pitch of the chest link (link_torso_4) to
# +14.04° or 0.00°. The shoulder moves x −9.45 cm · z +1.16 cm and the whole arm hangs off it.
#
# Why it lives in code: when the canonical value sat only in a sourced env file and the code default was another
# generation, paths that sourced it (judge, bake) solved IK on the upright robot while paths that did not (every
# training launcher) spawned the 14°-tilted one. Joint angles are relative to the base, so no guard fired, and the
# fingertips were off by p50 12.1 cm while overfitting failed to converge.
#
# Contract:
#   1. The three values below (variant suffix · base_pos · pose fingerprint) are **one set**; never change one alone.
#   2. **The code default is canonical.** An env value is only an override for a stated exception.
#   3. A mismatched combination must not run silently —
#      `dexengine.robots.selection.assert_generation_coherent()` kills it.
#
# Measured (URDF Pinocchio FK and USD pxr xform, two independent paths, equal to three decimals):
#   variant     chest pitch  right shoulder link_right_arm_0
#   (none)      +14.037°     (0.1023, −0.22, 1.2598)
#   _chest00     −0.000°     (0.0078, −0.22, 1.2714)
#   _chest00pr   −0.000°     same (61 joints · 988 xforms, **0 differences**)
# ===========================================================================
@dataclass(frozen=True)
class RobotGeneration:
    """One set of a robot's "body" — asset variant suffix · base position · torso pose fingerprint.

    ``variant`` is the **same string** in the USD (`data/usd_cache/robots/full_robot_gen/<combo><variant>/`) and the
    URDF (`assets/data/Robots/full_robot_gen/<combo><variant>/`). Sharing one suffix across both axes is the point:
    it is what lets the sim axis and the offline axis be checked against each other.

    ``torso3_pitch_rad`` is the merged URDF's `torso_3` joint origin rpy·y — the **one number a machine can check**
    to prove an asset on disk really is this generation; it is read from the content, not the file name (a directory
    name cannot be verified).
    """

    variant: str
    """Asset directory suffix. "" = a generation without a suffix."""

    base_pos: tuple[float, float, float]
    """Base spawn position that goes with this pose (world, m)."""

    torso3_pitch_rad: float
    """Merged URDF `torso_3` origin rpy·y (rad). Fingerprint that identifies the asset."""

    chest_pitch_deg: float
    """Accumulated chest link pitch (degrees). For humans — 0 = perpendicular to the ground."""

    why: str = ""
    """One line on what this generation is and why it exists."""

    derived_from: str | None = None
    """When set (None = not derived; "" names the unsuffixed generation), this is a **sim-only derivative**
    post-processed from that generation's USD (same body). The URDF (offline axis — judge · bake · IK) is that
    generation's, and identity is checked through the derivation record
    (dexengine.robots.derived_usd.provenance_file)."""

    material_of: str | None = None
    """When set (None = not a repaint), this body generation changes **only the materials** of that generation's
    USD — the body (URDF · kinematics · colliders) is that generation's, so it keeps no URDF copy and reads that
    generation's URDF (urdf_variant)."""

    limits_of: str | None = None
    """When set (None = not a limit copy), this is a sim-only body baked from a copy of that generation's merged
    URDF with **only the hand joint limits** truncated (generator dexengine/robots/urdf/truncate.py · suffix = a
    HAND_LIMIT_SUFFIX value). Kinematics · mass · colliders · arm geometry are that generation's; the URDF is its own
    copy (folder of the same name). HandSpec.sim_finger_limits decides which hands use this body — the offline axis
    (judge · IK · bake = canon_variant) never reads it."""


@dataclass(frozen=True)
class SimColliderRev:
    """The **collider revision** of the robot USD the sim spawns — an axis separate from the body (kinematics ·
    URDF · IK) generation.

    The sim spawns the derived USD ``<combo><body variant><suffix>`` with this revision applied; the offline axis
    (judge · bake) reads the body generation's URDF but builds the arm collision shapes with the same revision. The
    fields below own what changes, and the only shape generator is dexengine.placement.collision.sim_collider.bodies —
    the USD generator (dexengine/robots/usd/collider_rev.py) writes its vertices, the judge
    (dexengine.placement.collision.cross_sets) measures with that shape, and the spawn audit checks the stage.
    Rule: the collider of a listed arm link = the convex hull of that link's URDF ``<visual>`` mesh (reduced to the
    vertex cap).
    """

    suffix: str
    """Suffix appended after the body variant."""

    replace: tuple[int, ...]
    """Arm link numbers whose vendor collider is replaced by the visual mesh convex hull (ArmSpec numbering)."""

    add: tuple[int, ...]
    """Arm link numbers that had no collider and get a visual mesh convex hull added."""

    hull_vertex_cap: int
    """Hull vertex cap = PhysX convex cooking's default vertex limit. Past it PhysX reduces the hull itself and the
    shape becomes unknown, so the generator reduces it deterministically (a subset of the original hull vertices ·
    inner approximation) and records the maximum error."""

    prim_name: str
    """Prim name of the new collider (under the link's ``collisions``)."""

    why: str = ""

    torso_replace: tuple[str, ...] = ()
    """Torso links (URDF link names) whose vendor capsules are replaced by visual mesh convex hulls. Torso joints
    are fixed, so these links merge into one static root rigid body (``world``) in the USD — hull vertices are in root
    coordinates (sim_collider.torso_bodies). How many parts a hull has is measured by the split rule
    (torso_split.torso_parts) — not written here."""

    torso_add: tuple[str, ...] = ()
    """Torso links without a collider that get a convex hull added (same root rigid body)."""

    def torso_prim(self, link: str, part: int) -> str:
        """Prim name of torso link hull part ``part`` (0..) under ``world/collisions`` — the generator and the spawn
        audit both read this one (the audit counts from 0 with no gaps)."""
        return f"{self.prim_name}_{link.removeprefix('link_')}_p{part}"


def _derived_generation(g: RobotGeneration, rev: SimColliderRev) -> RobotGeneration:
    """Derived generation = body generation g with collider revision rev on top — body fields are g's (the one-set
    check still binds)."""
    return RobotGeneration(
        variant=g.variant + rev.suffix,
        base_pos=g.base_pos,
        torso3_pitch_rad=g.torso3_pitch_rad,
        chest_pitch_deg=g.chest_pitch_deg,
        why=f"sim-only derivative: collider revision {rev.suffix} on the USD of {g.variant} — {rev.why}",
        derived_from=g.variant,
    )


#: Suffix of the hand joint limit variant — **this line alone owns the name** (the generator
#: dexengine/robots/urdf/truncate.py reads it here). The variant = a copy of the merged URDF with only the hand joint
#: `<limit lower/upper>` cut by the retargeting hyperextension rule (retarget prior.hyperext_clip_range — the rule
#: clip selection references already follow), plus its USD.
TRUNCATED_HAND_LIMITS = "_truncated"

#: Finger limit kind (each hand picks one in HandSpec.sim_finger_limits) → asset suffix of the sim body. An unknown
#: kind is killed by sim_hand_limits. The offline axis (judge · IK · bake = canon_variant) does not read this axis —
#: its reference qpos is already cut by the same rule, so the vendor limits never bind the judge result.
HAND_LIMIT_SUFFIX: dict[str, str] = {"vendor": "", "truncated": TRUNCATED_HAND_LIMITS}


def _limit_generations(
    canon: RobotGeneration, render: RobotGeneration, rev: SimColliderRev | None, suffixes
) -> tuple[RobotGeneration, ...]:
    """The generations of each hand joint limit suffix: the body (canonical + suffix — its own URDF) · the render body
    (that body's material plate — the suffix goes before the render generation's material tail) · the collider
    revision derivatives of both (an arm with a revision). Body fields (base · pose fingerprint) stay canonical."""
    if not render.variant.startswith(render.material_of) or render.material_of != canon.variant:
        raise ValueError(
            f"render generation {render.variant!r} is not a material plate of canonical {canon.variant!r}"
            " — cannot derive the render body name of a limit variant"
        )
    tail, out = render.variant[len(render.material_of) :], []
    for sfx in sorted(set(suffixes) - {""}):
        body = RobotGeneration(
            variant=canon.variant + sfx,
            base_pos=canon.base_pos,
            torso3_pitch_rad=canon.torso3_pitch_rad,
            chest_pitch_deg=canon.chest_pitch_deg,
            limits_of=canon.variant,
            why=f"sim-only body baked from a copy of the merged URDF of {canon.variant} with only the hand joint "
            f"limits cut ({sfx} — dexengine/robots/urdf/truncate.py). Hands using it = HandSpec.sim_finger_limits.",
        )
        mi = RobotGeneration(
            variant=body.variant + tail,
            base_pos=canon.base_pos,
            torso3_pitch_rad=canon.torso3_pitch_rad,
            chest_pitch_deg=canon.chest_pitch_deg,
            material_of=body.variant,
            why=f"render body of {body.variant} (materials only — same generator as {render.variant}).",
        )
        out += [body, mi] + ([_derived_generation(body, rev), _derived_generation(mi, rev)] if rev is not None else [])
    return tuple(out)


#: Keys of the generations section of an arm spec file: the canonical body generation states these (its derived_from ·
#: material_of · limits_of are None), the render plate states its own (its body fields are the canonical one's).
CANON_KEYS = ("variant", "base_pos", "torso3_pitch_rad", "chest_pitch_deg", "why")
RENDER_KEYS = ("variant", "material_of", "why")


def _stated(raw, keys: tuple, where: str) -> dict:
    if not isinstance(raw, dict) or set(raw) != set(keys):
        got = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        fail(where, f"keys {got} — exactly {list(keys)}")
    return dict(raw)


def body_sections(
    raw: dict, where: str
) -> tuple[tuple[RobotGeneration, RobotGeneration] | None, SimColliderRev | None]:
    """Pop an arm spec file's body sections (``raw`` = the parsed file) → ((canonical, render) generations | None,
    collider revision | None). ``generations`` states the canonical body generation (CANON_KEYS) and its render plate
    (RENDER_KEYS — materials only: material_of is the canonical variant and the body fields are the canonical one's);
    ``collider_rev`` states every SimColliderRev field. Both keys are stated, null = none."""
    for k in ("generations", "collider_rev"):
        if k not in raw:
            fail(where, f"missing key [{k!r}] — null when the arm has none")
    g, r = raw.pop("generations"), raw.pop("collider_rev")
    rev = None if r is None else build(SimColliderRev, r, f"{where}.collider_rev")
    if g is None:
        return None, rev
    g = _stated(g, ("canon", "render"), f"{where}.generations")
    none = {"derived_from": None, "material_of": None, "limits_of": None}
    c = _stated(g["canon"], CANON_KEYS, f"{where}.generations.canon")
    canon = build(RobotGeneration, c | none, f"{where}.generations.canon")
    r = _stated(g["render"], RENDER_KEYS, f"{where}.generations.render")
    if r["material_of"] != canon.variant:
        fail(where, f"generations.render.material_of {r['material_of']!r} ≠ the canonical variant {canon.variant!r}")
    body = {k: getattr(canon, k) for k in ("base_pos", "torso3_pitch_rad", "chest_pitch_deg")}
    render = build(RobotGeneration, r | body | {"derived_from": None, "limits_of": None}, f"{where}.generations.render")
    return (canon, render), rev


def generation_table(canon: RobotGeneration, render: RobotGeneration, rev: SimColliderRev | None) -> dict:
    """Variant suffix → generation of one arm: the two stated body generations, their collider revision derivatives
    and the hand joint limit variants (all rules over the stated rows). A suffix not listed here (an experimental asset
    such as _pose2/_torsofree/_decomp) is "unregistered" and skips the generation check, so experiments are not
    blocked; retired generations are not kept (they would look selectable)."""
    gens = [canon, render]
    if rev is not None:
        gens += [_derived_generation(canon, rev), _derived_generation(render, rev)]
    gens += _limit_generations(canon, render, rev, HAND_LIMIT_SUFFIX.values())
    return {g.variant: g for g in gens}


_BODIES = {arm: body_sections(dict(_E.read_spec(p)), str(p)) for arm, p in _E.spec_files(_E.SPEC_ROOT, "arms").items()}
#: Arm name → sim collider revision. An arm not listed spawns its body generation USD as is.
ARM_SIM_COLLIDER_REV: dict[str, SimColliderRev] = {a: r for a, (_g, r) in _BODIES.items() if r is not None}
#: Arm name → that arm's generation table. An arm without a torso pose has no such axis (not in the table).
ARM_GENERATIONS: dict[str, dict[str, RobotGeneration]] = {
    a: generation_table(*g, r) for a, (g, r) in _BODIES.items() if g is not None
}
#: Arm name → canonical generation (the spec file's generations.canon).
ARM_CANON_GEN: dict[str, RobotGeneration] = {a: g[0] for a, (g, _r) in _BODIES.items() if g is not None}
#: Arm name → render body generation (only the look differs — the body is the canonical generation). Every process
#: spawns this generation with the sim collider revision on top (plate_variant). Read by the plate owner · the USD
#: generators · the render plate checks.
ARM_RENDER_GEN: dict[str, RobotGeneration] = {a: g[1] for a, (g, _r) in _BODIES.items() if g is not None}


def canon_variant(arm: str) -> str:
    """Canonical asset variant suffix of the arm. "" for an arm without a generation axis."""
    g = ARM_CANON_GEN.get(arm)
    return g.variant if g is not None else ""


def sim_hand_limits(arm: str, hand: str) -> str:
    """Sim hand joint limit suffix of (arm, hand) = HAND_LIMIT_SUFFIX[that hand's HandSpec.sim_finger_limits].
    "" for an arm without a body generation axis. An unregistered hand or an unknown kind raises."""
    if arm not in ARM_GENERATIONS:
        return ""
    from dexengine.core.hands import HAND_REGISTRY  # noqa: PLC0415 — the hand registry imports the arm module

    spec = HAND_REGISTRY.get(hand)
    if spec is None:
        raise KeyError(
            f"hand {hand!r} has no spec in dexengine.core.hands.HAND_REGISTRY (registered: {sorted(HAND_REGISTRY)})"
        )
    kind = spec.sim_finger_limits
    if kind not in HAND_LIMIT_SUFFIX:
        raise KeyError(
            f"hand {hand!r} has sim_finger_limits={kind!r} — known kinds are {sorted(HAND_LIMIT_SUFFIX)} "
            "(dexengine.core.arms.HAND_LIMIT_SUFFIX)"
        )
    return HAND_LIMIT_SUFFIX[kind]


def sim_body_variant(arm: str, hand: str) -> str:
    """Body the sim spawns = canonical body generation + the hand's limit suffix (sim_hand_limits · before the
    collider revision — the source of the derived plate)."""
    return canon_variant(arm) + sim_hand_limits(arm, hand)


def render_body_variant(arm: str, hand: str) -> str:
    """Render body = the render body generation (the canonical material plate) with the hand's limit suffix put
    before the material tail. An arm without a render body generation raises — there is no fallback to rendering
    the training plate."""
    g = ARM_RENDER_GEN.get(arm)
    if g is None:
        raise KeyError(f"arm {arm!r} has no render body generation — register it in dexengine.core.arms.ARM_RENDER_GEN")
    lim = sim_hand_limits(arm, hand)
    return g.variant if not lim else g.material_of + lim + g.variant[len(g.material_of) :]


def hand_bodies(arm: str, hand: str) -> tuple[str, str]:
    """The two bodies (arm, hand) spawns (training body · render body) — the sources the collider revision generator
    bakes for this hand. The material plate comes after its training body (the render plate check needs the
    training plate first)."""
    return sim_body_variant(arm, hand), render_body_variant(arm, hand)


def sim_usd_variant(arm: str, hand: str) -> str:
    """The training body's collider-revision derivative = the hand's sim body (sim_body_variant) + the arm's collider
    revision — a bake intermediate: the physics reference the render plate check
    compares the plate with (render_plate.parity), never spawned (plate_variant is). Its last generator reader
    (split_arm.py's default) and its bake (collider_rev via hand_bodies) leave with the re-bake bundle (hyperext ·
    xr_determ) — a change of those generator files stales every derived record, so not before that bake.
    For an arm without a body generation axis (a floating hand) it is that arm's one body (plate_variant)."""
    rev = ARM_SIM_COLLIDER_REV.get(arm)
    return sim_body_variant(arm, hand) + (rev.suffix if rev is not None else "")


def plate_variant(arm: str, hand: str) -> str:
    """The robot plate every process spawns — training, judge, record, render (one plate per combo):
    the hand's render body (render_body_variant — the training body repainted, appearance only) + the
    arm's collider revision. Colliders, masses and joint limits are the training body's (render_plate.parity at the
    bake). An arm without a body generation axis has one body (sim_usd_variant: '' for a floating hand)."""
    if arm not in ARM_RENDER_GEN:
        return sim_usd_variant(arm, hand)
    rev = ARM_SIM_COLLIDER_REV.get(arm)
    return render_body_variant(arm, hand) + (rev.suffix if rev is not None else "")


#: Message tail of a retired plate name: a run recorded with the training plate is re-judged where that plate lives.
RETIRED_PLATE_HINT = "re-judge it with the release that made it — this tree spawns one plate per combo"


def retired_plate_why(arm: str, hand: str, variant: str) -> str:
    """'' unless `variant` names the retired training plate of (arm, hand) (sim_usd_variant, with or without the split
    suffix) — then the reason, ending in RETIRED_PLATE_HINT. The launcher (a recorded switch) and the spawn
    (dexengine.robots.selection) both ask here."""
    if arm not in ARM_RENDER_GEN:
        return ""
    old = sim_usd_variant(arm, hand)
    if variant not in (old, old + "_split"):
        return ""
    return (
        f"{arm}_{hand}{variant} is the training plate, retired by render_unify — every process spawns "
        f"{arm}_{hand}{plate_variant(arm, hand)}. {RETIRED_PLATE_HINT}"
    )


def generation_of(arm: str, variant: str) -> RobotGeneration | None:
    """(arm, variant) → registered generation. None for an unregistered variant (check skipped)."""
    return ARM_GENERATIONS.get(arm, {}).get(variant)


def urdf_variant(arm: str, variant: str) -> str:
    """Variant suffix of the URDF a registered variant reads — a derived plate reads its source body generation's, a
    materials-only body generation reads its origin generation's. An unregistered variant raises (its body cannot be
    proven)."""
    g = generation_of(arm, variant)
    if g is None:
        raise KeyError(
            f"unregistered variant {arm}{variant!r} — cannot pick the URDF (dexengine.core.arms.ARM_GENERATIONS)"
        )
    body = variant if g.derived_from is None else g.derived_from
    origin = generation_of(arm, body).material_of
    return body if origin is None else origin
