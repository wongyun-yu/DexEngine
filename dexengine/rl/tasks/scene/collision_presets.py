"""What a collision preset **means**: the one place that states it.

Why a module of its own: the same statement used to live in two places, `collision_check.EXPECTED` (the runtime
oracle) and `_CP_ROWS` in an older shell launcher (the human banner). A comment said "the two must agree",
but nothing checked it, so fixing one side made the banner lie silently. The banner is how a person judges "did this run
have the right physics", so a false banner is as bad as wrong physics.

This module has **no dependencies** (standard library only), so shells outside Isaac can call it, and
`collision_check` reads it to compare with the measurement. One statement; the comparison stays independent.

From a shell:
    _CP_ROWS=$(python3 <this file> --preset canon_r2r3off --what rows)
    PRESETS=$(python3 <this file> --what list)
"""

from __future__ import annotations

import argparse
import sys

# ── Preset expectation table (1:1 with the recipe's COLL_PRESET; the human banner _CP_ROWS comes from it too) ──
# Values are in their normalize() form: "ON" | "OFF" | "right:OFF left:ON" | "hand:ON arm:OFF".
EXPECTED = {
    # CANON = the real-robot physics. With the split articulation (SPLIT, two pieces) R3 is OFF on the right only (the
    # torso belongs to the right articulation).
    "canon": {
        "R1": "OFF",
        "R2": "ON",
        "R3": "right:OFF left:ON",
        "R4": "ON",
        "R5": "ON",
        "R6": "ON",
        "R7": "ON",
        "R8": "ON",
        "RX": "ON",
    },
    # ABL = floating hand. Only hand<->own object (R5) and object<->desk (R8) remain.
    "abl": {
        "R1": "OFF",
        "R2": "OFF",
        "R3": "OFF",
        "R4": "OFF",
        "R5": "ON",
        "R6": "OFF",
        "R7": "OFF",
        "R8": "ON",
        "RX": "OFF",
    },
    # Separating the deciding factor: CANON with the inter-articulation rows turned off one at a time (filters, all
    # envs). single = the physics the ABL family really trained with.
    "canon_r2off": {
        "R1": "OFF",
        "R2": "OFF",
        "R3": "right:OFF left:ON",
        "R4": "ON",
        "R5": "ON",
        "R6": "ON",
        "R7": "ON",
        "R8": "ON",
        "RX": "ON",
    },
    "canon_r3off": {
        "R1": "OFF",
        "R2": "ON",
        "R3": "OFF",
        "R4": "ON",
        "R5": "ON",
        "R6": "ON",
        "R7": "ON",
        "R8": "ON",
        "RX": "ON",
    },
    "canon_r2r3off": {
        "R1": "OFF",
        "R2": "OFF",
        "R3": "OFF",
        "R4": "ON",
        "R5": "ON",
        "R6": "ON",
        "R7": "ON",
        "R8": "ON",
        "RX": "ON",
    },
    "single": {
        "R1": "OFF",
        "R2": "OFF",
        "R3": "OFF",
        "R4": "ON",
        "R5": "ON",
        "R6": "ON",
        "R7": "ON",
        "R8": "ON",
        "RX": "ON",
    },
    # ABL + hand<->other object ON (abl_ho), + object<->object ON (abl_hoo). The arm stays OFF through ARM_OBJ_NOCOLL.
    "abl_ho": {
        "R1": "OFF",
        "R2": "OFF",
        "R3": "OFF",
        "R4": "OFF",
        "R5": "ON",
        "R6": "hand:ON arm:OFF",
        "R7": "OFF",
        "R8": "ON",
        "RX": "OFF",
    },
    "abl_hoo": {
        "R1": "OFF",
        "R2": "OFF",
        "R3": "OFF",
        "R4": "OFF",
        "R5": "ON",
        "R6": "hand:ON arm:OFF",
        "R7": "ON",
        "R8": "ON",
        "RX": "OFF",
    },
}


# Human hints for the banner only. They say **why**, not the value (ON/OFF).
NOTE = {
    "canon": "(R3: two-piece limit)",
    "single": "(= the physics the ABL family really trained with)",
}

ROW_ORDER = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "RX")

# ── A row independent of presets: robot<->support disc (RD, rule D40: disc-hand and disc-arm collisions are off) ──
# The stamp `collide` (value owner coords.DISC_COLLIDE, stamped per disc by the bake) -> what the RD row means. This
# table
# is the one place of that meaning: authoring (`collision_authoring.author_disc_collision`) reads it to build the
# physics
# and the oracle (`collision_check.check_preset`) reads it to compare with the measurement (one statement, an
# independent comparison). A value missing from the table fails.
DISC_ROW = "RD"
DISC_COLLIDE_RD = {
    "objects": "OFF",  # discs (aerial scene) support objects only: no collision with any robot body
    "pedestal_objects": "OFF",
}  # the pedestal scene's statement (coords.PEDESTAL_OBJECTS, the column under
#                                                 a main object): off with the robot. The judge's arm guard filters
#                                                 placements where the arm passes through a column


def expected_disc(n_slots: int, collide: str | None) -> str:
    """Expected RD value. A scene without stated disc slots (n_slots 0) is '—'. With slots, the rule of the stamp
    `collide`; a missing statement or a value outside the table raises KeyError (no silent default)."""
    if not isinstance(n_slots, int) or n_slots < 0:
        raise ValueError(f"disc slot count {n_slots!r} must be an integer ≥ 0")
    if n_slots == 0:
        return "—"
    if collide not in DISC_COLLIDE_RD:
        raise KeyError(f"{n_slots} disc slots but no RD rule for collide={collide!r} — table {sorted(DISC_COLLIDE_RD)}")
    return DISC_COLLIDE_RD[collide]


# ── The second row of the same stamp: support disc<->object (RO, rule D11″ (d): a disc supports its own object
# only) ──
# A disc collides only with **its own main object** (owner side = coords.disc_owner_side), not with the scene's other
# objects (the other side's main object). Meaning of 'OWN' = every pair (the env clip's real disc body, another object
# body) filtered, own-object pairs not filtered. Authoring and the oracle read this table.
DISC_OBJ_ROW = "RO"
DISC_COLLIDE_RO = {
    "objects": "OWN",
    "pedestal_objects": "OWN",
}  # a pedestal also supports its own main object only; a 'visual only' column
#                                                 (owner static) has prim collision off and is no filter target (writer
# _build_disc_slots)


def expected_disc_obj(n_slots: int, collide: str | None, n_objects: int) -> str:
    """Expected RO value. No disc slots, or fewer than two object entities in the scene (one hand, shared, idle side),
    is '—' (there is no other object). Otherwise the rule of the stamp `collide`; a missing statement or a value outside
    the table raises KeyError."""
    if not isinstance(n_slots, int) or n_slots < 0 or not isinstance(n_objects, int) or n_objects < 0:
        raise ValueError(f"disc slot count {n_slots!r} and object count {n_objects!r} must be integers ≥ 0")
    if n_slots == 0 or n_objects < 2:
        return "—"
    if collide not in DISC_COLLIDE_RO:
        raise KeyError(
            f"{n_slots} disc slots and {n_objects} objects but no RO rule for collide={collide!r} — "
            f"table {sorted(DISC_COLLIDE_RO)}"
        )
    return DISC_COLLIDE_RO[collide]


# ── Support objects (RS: a support object collides with hands, arms and objects; judged by the same
# ruler as the desk) ──
# Support object = a real tracked object that supports another object (support_object in the data-side kind decision
# coords.support_kinds; not a disc). This table owns the meaning per partner: authoring (collision_authoring, support
# slot consumer) builds the physics, the oracle (collision_check RS row) compares with the measurement, and the judge's
# support<->arm term (placement collision.support_surface) measures only when the 'arm' cell is ON, so changing one
# cell changes sim and judge together. Partners: hand = hand bodies (fingers, palm) · arm = every arm link (the arm_6
# wrist housing included; the same link set as the desk ruler) · objects = the scene's objects · desk = the desk (a
# support is a kinematic body pinned at its first-frame pose and never meets the desk). Unknown partners or values fail.
SUPPORT_ROW = "RS"
SUPPORT_COLLIDE = {"hand": "ON", "arm": "ON", "objects": "ON", "desk": "OFF"}
SUPPORT_VALUES = ("ON", "OFF")


def expected_support(n_slots: int, partner: str) -> str:
    """Expected RS value per partner. A scene without support slots (n_slots 0) is '—'. An unknown partner or a table
    value other than ON/OFF fails."""
    if not isinstance(n_slots, int) or n_slots < 0:
        raise ValueError(f"support slot count {n_slots!r} must be an integer ≥ 0")
    if partner not in SUPPORT_COLLIDE:
        raise KeyError(f"no RS rule for support partner {partner!r} — table {sorted(SUPPORT_COLLIDE)}")
    v = SUPPORT_COLLIDE[partner]
    if v not in SUPPORT_VALUES:
        raise ValueError(f"SUPPORT_COLLIDE[{partner!r}]={v!r} — {SUPPORT_VALUES} only")
    return "—" if n_slots == 0 else v


# ── Extra object slots (RE, `env_setup.extra_slots.build_extra_slots`, on with DEXENGINE_EXTRA_SLOTS): kinematic bodies
# that move along the reference.
# hand · arm = the robot (on: the hand must wrap the extra) · objects = the main objects (off: a kinematic body has
# infinite mass and would push a main object into a tracking termination regardless of the policy) · desk is not in the
# table (PhysX makes no contacts between kinematic and static bodies). Unknown partners or values fail.
# Authoring = collision_authoring.author_extra_collision.
EXTRA_ROW = "RE"
EXTRA_COLLIDE = {"hand": "ON", "arm": "ON", "objects": "OFF"}  # moved extras (slot kind moving): follow the reference

# Collision partners of static slots (kinematic bodies pinned at their frame-0 pose): one named table per kind.
# Robot (hand · arm): off for both kinds (collision might get in the way of exploration, so the bodies
# stay in the scene but **do not collide with the robot**; placements where the arm passes through one are filtered by
# the judge's arm guard as now, not blocked by RL collision).
# Main objects (objects):
#   touched  = a static object the hand touched without moving it -> off (the same reason as moved extras; resting is
#              handled by resting discs and pedestals).
#   receiver = a static object that receives -> **on** (only main objects and hand-carried extras <->
#              receivers are on; robot hand and arm <-> receivers stay off). A receiver is a kinematic body standing at
#              its frame-0 pose, so a (dynamic) main object rests on it and cannot pass through it. Hand-carried extras
#              (moving slots) are kinematic too, so PhysX makes no contacts between them and a receiver (both follow the
#              reference); the table has no cell for that pair. The collision shape is the training USD's convex pieces.
# To change a value, edit only that kind's table line (authoring reads the tables; hand and arm switch on or off
# together, authoring has no separate switch and dies on such a table; whether a collider stands also derives from the
# table, extra_has_collider).
EXTRA_TOUCHED_COLLIDE = {"hand": "OFF", "arm": "OFF", "objects": "OFF"}
EXTRA_RECEIVER_COLLIDE = {"hand": "OFF", "arm": "OFF", "objects": "ON"}
# ── All-object collision (training turns on collisions between every object in the scene, from
# the visual aug round on) ──
# One switch, default off = the tables above as they are (byte-identical). On: (1) every slot kind gets objects ON
# (extra_collide overrides the table; robot cells unchanged, scope is between objects only), (2) the static objects the
# render only shows (rows record.scene.extras.decide adds; visual only, render-only source) stand as invisible kinematic
# bodies at the same pose (built by env_setup.objects.build_all_obj_bodies, posed by
# all_obj_collision.place_all_obj_bodies, robot off, measured marker all_obj_collision.author_all_obj_collision).
# Main object <-> main object (R7) is already on through the preset. PhysX makes no contacts between kinematic bodies,
# so turning those pairs on yields no contact (the marker counts 'main object pairs' only).
ALL_OBJ_ENV = "DEXENGINE_ALL_OBJ_COLLISION"
ALL_OBJ_TAG = "[ALL_OBJ_COLL]"
ALL_OBJ_KEY = "all_obj_body_"  # scene entity name prefix (a number follows)
ALL_OBJ_LEAF = "AllObjBody_"  # prim name prefix under the env (a number follows)
ALL_OBJ_POSE_TOL_M = 0.005  # max position error static body <-> visual prim (above it fails; marker 'pose check')
# Exception to "on": slot kinds that stay off with the main objects (tool <-> target: the static object a held main
# object touches; brush_plate's plate is a touched slot). Basis (pilot measurement): with it on, a checkpoint trained
# with it off got stuck on brush <-> plate forces of 20–100 N and the RSI success went from 20/32 to 3/32. Hence
# only tool <-> target touched slots are off with the main objects; the rest on. Robot cells, other kinds and static
# bodies stay on. Marker `[ALL_OBJ_COLL] … exception`.
ALL_OBJ_KEEP_OFF_KINDS = ("touched",)


def all_obj_collision() -> bool:
    """All-object collision switch: empty or 0 = off · 1 = on · any other value fails."""
    from dexengine.rl.tasks import switches  # the script entry (--what) never reads the switch

    v = switches.text(ALL_OBJ_ENV)
    if v not in ("", "0", "1"):
        raise ValueError(f"{ALL_OBJ_TAG} {ALL_OBJ_ENV}={v!r} — 0 | 1 only")
    return v == "1"


#: Slot kind (`kind` of a dexengine/data/objects/extra_slots.py row) -> that kind's collision partner table.
#: Unknown kinds, partners or values fail.
EXTRA_KIND_COLLIDE = {"moving": EXTRA_COLLIDE, "touched": EXTRA_TOUCHED_COLLIDE, "receiver": EXTRA_RECEIVER_COLLIDE}


def slot_robot_off(table: dict) -> bool:
    """The authoring filters every robot body <-> this slot when the table's hand partner is OFF (one rule for the
    authoring and the ledger's check of what it filtered)."""
    return table["hand"] == "OFF"


EXTRA_PARTNERS = ("hand", "arm", "objects")
# Whether the judge's arm guard measures the bodies of fixed extras (static slots: touched or receiving). A **separate
# statement** from the sim collision (the tables above, robot OFF): placements where the arm passes
# through a fixed object are filtered by the judge, not blocked by RL collision.
# Reader = placement collision.support_surface.extra_measured. Values other than ON|OFF fail.
EXTRA_FIXED_JUDGE_ARM = "ON"


def extra_collide(kind: str) -> dict:
    """Collision partner table of an extra slot kind {hand · arm · objects: ON|OFF}.
    An unknown kind, a table with other partners or a value other than ON/OFF fails."""
    if kind not in EXTRA_KIND_COLLIDE:
        raise KeyError(f"no RE rule for extra slot kind {kind!r} — table {sorted(EXTRA_KIND_COLLIDE)}")
    t = EXTRA_KIND_COLLIDE[kind]
    if tuple(sorted(t)) != tuple(sorted(EXTRA_PARTNERS)) or any(v not in ("ON", "OFF") for v in t.values()):
        raise ValueError(f"EXTRA_KIND_COLLIDE[{kind!r}]={t!r} — partners {EXTRA_PARTNERS} · values ON|OFF only")
    return {**t, "objects": "ON"} if all_obj_collision() and kind not in ALL_OBJ_KEEP_OFF_KINDS else dict(t)


def extra_has_collider(kind: str | None) -> bool:
    """Whether a collider stands on an extra slot body: only when that kind's table has an ON partner (derived from the
    table, never from the kind name). A body of an all-OFF kind has nothing to collide with: the tables turn off the
    robot and the main objects, and kinematic bodies make no contacts with the desk (static), supports, discs or other
    slots (kinematic) in the first place. Without a collider the physics is the same, and the per-env collision meshes
    disappear (extra objects stand per env with physics replication off). Turning one table cell ON brings colliders
    back for that kind. None = an empty place absent from the clip (a stand-in body at the hidden pose; nothing stands)
    -> no collider. Readers: the spawn `env_setup.extra_slots.build_extra_slots` and the authoring check
    `collision_authoring.author_extra_collision`."""
    return kind is not None and table_has_collider(extra_collide(kind))


#: Marker of the measured extra-slot colliders (collision_authoring). scripts/rl/judge_sidecar.sh compares the text
#: after the first ')' of this line, counts removed, between the training and the judge log.
COLLIDER_MARK = "[EXTRA_COLL] colliders"


def collider_line(kinds, n: dict, tot: dict) -> str:
    """The COLLIDER_MARK line: per kind the statement (with | without) and measured slots with a collider / slots."""
    return (
        f"{COLLIDER_MARK}(statement collision_presets.extra_has_collider = stage measurement) "
        + " · ".join(
            f"{k} {'with' if extra_has_collider(k) else 'without(table all OFF)'} {n[k]}/{tot[k]}" for k in kinds
        )
        + f" · empty without(empty place) {n['empty']}/{tot['empty']}"
    )


def table_has_collider(table: dict) -> bool:
    """Whether one collision partner table line {hand · arm · objects: ON|OFF} stands a collider: when any partner is ON
    (that body collides with the robot or a main object, the things the policy moves). One rule read by both
    `extra_has_collider` and the training scene comparison ('same scene': a new slot of a collider kind, or a changed
    partner of that kind, means another scene; rl.launch.train_scene)."""
    return "ON" in table.values()


def extra_collide_text() -> str:
    """One line of collision partners per kind (carried by the run log marker): `moving(hand ON · arm ON · objects
    OFF) · touched(…) · …`."""
    return " · ".join(
        f"{k}(" + " · ".join(f"{p} {extra_collide(k)[p]}" for p in EXTRA_PARTNERS) + ")" for k in EXTRA_KIND_COLLIDE
    )


# ── Object<->torso (RT, rule D-T4): the torso collider (the 64-vertex visual hull) and the scene objects
# collide
# (0 filters). A row independent of presets; the audit row (collision_check) reads this statement and compares it with
# the measurement (0 filter pairs).
OBJ_TORSO_ROW = "RT"
OBJ_TORSO_COLLIDE = "ON"

# `legacy` is not a preset but the **explicit choice of "no preset"** (the launcher). Having no expectation table is
# normal, so it gets its name here, and shells never keep a list of their own.
NO_PRESET = "legacy"
PRESETS = tuple(EXPECTED)
VALID = PRESETS + (NO_PRESET,)


def rows(preset: str) -> str:
    """One line for people. The banner and the oracle come from **the same statement**."""
    if preset == NO_PRESET:
        return "no preset (legacy physics) — no expectation check"
    if preset not in EXPECTED:
        raise KeyError(f"unknown preset: {preset!r} (registered: {list(VALID)})")
    e = EXPECTED[preset]
    body = " ".join(f"{r} {e[r]}" for r in ROW_ORDER if r in e)
    note = NOTE.get(preset, "")
    return f"{body} {note}".rstrip()


# ── Cross-repair sets -> training physics (no filtering, plus a record) ──
# For each set in which cross-repair (dexengine/placement/repair/cross_repair.py SET_REPAIR) measures hand-pair
# penetration, the row that decides that pair in the sim. The status ledger reads it to fill only 'remaining
# penetration of the sets on in the training physics'; set names are derived from the preset tables, never picked as
# literals. The ledger checks that these keys are inside the cross-repair report's sets (params.sets).
XR_SET_ROW = {"1_cross": "R2", "2_hand_obj": "R6", "4_torso": "R3"}


def hand_on(value: str) -> bool:
    """Normalised row value -> does the hand collide in that row. A rule, not a table: 'ON' and 'OFF' as they are; for
    the
    '<part>:<value> …' form, the 'hand' part's value when present, and when it splits only by side (right/left), on if
    either side is ON. Any other form fails."""
    if value in ("ON", "OFF"):
        return value == "ON"
    toks = value.split()
    parts = dict(t.split(":", 1) for t in toks if ":" in t)
    if not toks or len(parts) != len(toks) or not set(parts.values()) <= {"ON", "OFF"}:
        raise ValueError(f"row value {value!r} has an unknown form — only 'ON', 'OFF' and '<part>:<ON|OFF> …'")
    if "hand" in parts:
        return parts["hand"] == "ON"
    if set(parts) <= {"right", "left"}:
        return "ON" in parts.values()
    raise ValueError(
        f"row value {value!r} has neither a hand part nor a side split — cannot tell whether the hand is on"
    )


def xr_sets_on(preset: str) -> tuple[str, ...]:
    """Cross-repair sets with the hand on in the training preset (XR_SET_ROW order). A name without a table (legacy
    included) fails."""
    if preset not in EXPECTED:
        raise KeyError(
            f"preset {preset!r} has no expectation table — cannot tell which cross-repair sets are on (registered: "
            f"{list(PRESETS)})"
        )
    return tuple(s for s, r in XR_SET_ROW.items() if hand_on(EXPECTED[preset][r]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="")
    ap.add_argument("--what", default="rows", choices=("rows", "list"))
    a = ap.parse_args()
    if a.what == "list":
        print("|".join(VALID))
        return 0
    if not a.preset:
        print("[collision_presets] --preset is required", file=sys.stderr)
        return 2
    try:
        print(rows(a.preset))
    except KeyError as e:
        print(f"[collision_presets] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
