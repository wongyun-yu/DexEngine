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
#: What each preset row pairs (collision_check measures them; the scene manifest writes them).
ROW_MEANING = {
    "R1": "hand internal · arm internal · hand <-> same arm",
    "R2": "right hand·arm <-> left hand·arm",
    "R3": "arm·hand <-> torso",
    "R4": "robot <-> desk",
    "R5": "hand <-> own object",
    "R6": "hand·arm <-> other object",
    "R7": "object <-> object",
    "R8": "object <-> desk",
    "RX": "arm <-> own object",
}

# ── Support-disc rows RD (robot<->disc) and RO (disc<->object): collision_disc_rows owns them (stamp-driven, not
# preset rows).

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
# Robot (hand · arm): the tables hold OFF, the state of a dataset that opts out of S-robot-static (below), where the
# bodies stay in the scene without colliding with the robot and the judge's arm guard filters placements where the arm
# passes through one. Where S-robot-static is on, `extra_collide` turns hand and arm ON for ROBOT_STATIC_KINDS.
# Main objects (objects):
#   touched  = a static object the hand touched without moving it -> off (the same reason as moved extras; resting is
#              handled by resting discs and pedestals).
#   receiver = a static object that receives -> **on** (main objects and hand-carried extras <-> receivers; the robot
#              cells as stated above). A receiver is a kinematic body standing at
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


# ── Full physics (the delivered scene is physically correct; the training's exploration filters stay in training) ──
# Training keeps the tables above (the switch is off: the kinematic reference is planned collision-free beforehand and
# exploration runs with the robot's static collisions off). The delivered scene (record.scene.manifest) and the
# physical-validity judge (a training-record copy with the switch on) use it: every slot kind collides with the robot
# and the main objects, except per kind FULL_PHYSICS_OBJECTS_OFF (objects; the value is the reason). No robot pair has
# a recorded-overlap exception. Marker `[FULL_PHYSICS]`.
FULL_PHYSICS_ENV = "DEXENGINE_FULL_PHYSICS"
FULL_PHYSICS_TAG = "[FULL_PHYSICS]"
FULL_PHYSICS_OBJECTS_OFF = {
    "touched": "a main-object pair the gate could not measure (S-touch-gate; a measured overlap drops the clip)",
    "carried_attached": "kinematic rider at the simulated carrier pose one policy step late would push its carrier",
}


def full_physics() -> bool:
    """Full-physics switch: 1 = on · 0 = off · any other value fails."""
    from dexengine.rl.tasks import switches

    v = switches.text(FULL_PHYSICS_ENV)
    if v not in ("0", "1"):
        raise ValueError(f"{FULL_PHYSICS_TAG} {FULL_PHYSICS_ENV}={v!r} — 0 | 1 only")
    return v == "1"


# ── S-dyn-pairs: a pair of two bodies dynamic in training (role=main, at most one per side: the R7 pair) that overlaps
# at frame 0 (statement data/objects/dyn_pairs.dyn_pairs). Verdict pair_off = that pair's collision is OFF in training
# and in the delivered scene; no object moves and every other pair keeps its row. Training mechanism (authoring, not
# here): collision groups as the robot-off bodies (collision_groups) — in an env whose clip states pair_off each of the
# two objects leaves the env group for a group of its own listing itself, the env, robot, robot-off and global groups,
# not the other. Not FilteredPairs: robot × slot as FilteredPairs held 22.41 GB GPU heap against 3.76 GB as groups (4.3
# without the slot; 1024 envs, collision_groups doc); one object pair per env as FilteredPairs is not measured.
DYN_PAIR_ROW = "R7"
DYN_PAIR_OFF = "pair_off"
DYN_PAIR_COLLIDE = {"training": "OFF", "delivered": "OFF"}


def dyn_pair_cells(pairs: list) -> list:
    """Collision cells [{row, a, b, delivered, training, why}] (the manifest cell form) of the pairs S-dyn-pairs turns
    off (data/objects/dyn_pairs.scene_pair: [{a, b, f0_m, ...}])."""
    return [
        {
            "row": DYN_PAIR_ROW,
            "a": p["a"],
            "b": p["b"],
            **DYN_PAIR_COLLIDE,
            "why": f"S-dyn-pairs: both dynamic, frame-0 overlap {p['f0_m'] * 1e3:.1f} mm (collision shapes), not moving"
            " together; a recorded visual overlap above the true-shape overlap drops the clip",
        }
        for p in pairs
    ]


#: Slot kind (`kind` of a dexengine/data/objects/extra_slots.py row) -> that kind's collision partner table.
#: Unknown kinds, partners or values fail.
EXTRA_KIND_COLLIDE = {"moving": EXTRA_COLLIDE, "touched": EXTRA_TOUCHED_COLLIDE, "receiver": EXTRA_RECEIVER_COLLIDE}
# ── Scene-rule kinds, kept out of EXTRA_KIND_COLLIDE so the every-kind statements (markers, authoring line) stay
# byte-identical where no clip has such a row (`extra_kinds_in_play`). S-carried (extra_slots.KIND_ATTACHED):
#   carried_attached  kinematic body at the simulated carrier pose × the relative pose: objects OFF (a kinematic body
#                     lagging its dynamic carrier by one policy step would push it); hand · arm as moving.
EXTRA_ATTACHED_COLLIDE = {"hand": "ON", "arm": "ON", "objects": "OFF"}
# S-touch-gate and S-envelope (extra_slots.KIND_TOUCHED_SOLID · KIND_ENVELOPE): still bodies pinned at frame 0 that
# collide with the main objects (which rest on or stop at them); robot cells as for every still extra (above).
#   touched_solid  a touched static whose recorded pairs with the main objects pass the penetration gate (touch_gate);
#                  a touched static that fails it keeps kind touched (objects OFF)
#   envelope       a still extra within the envelope margin of the recorded object sweep (envelope)
EXTRA_TOUCHED_SOLID_COLLIDE = {"hand": "OFF", "arm": "OFF", "objects": "ON"}
EXTRA_ENVELOPE_COLLIDE = {"hand": "OFF", "arm": "OFF", "objects": "ON"}
EXTRA_SCOPED_KIND_COLLIDE = {
    "carried_attached": EXTRA_ATTACHED_COLLIDE,
    "touched_solid": EXTRA_TOUCHED_SOLID_COLLIDE,
    "envelope": EXTRA_ENVELOPE_COLLIDE,
}


# ── S-robot-static (the robot collides with the still extras in training) ──
# Where the rule is on for the run's dataset (scene_rules.ROBOT_RULE, default on, opt-out per dataset profile), hand and
# arm are ON for these kinds; self-collision, the desk, supports and discs keep their own rows. The run's dataset is
# DATASET_ENV (set by the launcher and --dataset; missing fails) unless the caller names one.
ROBOT_STATIC_KINDS = ("touched", "receiver", "touched_solid", "envelope")
DATASET_ENV = "DEXENGINE_DATASET"


def run_dataset() -> str:
    """The dataset of this process's run (DATASET_ENV); unset fails (no default dataset)."""
    from dexengine.rl.tasks import env_flags  # noqa: PLC0415

    ds = env_flags.get(DATASET_ENV).strip()
    if not ds:
        raise ValueError(f"{DATASET_ENV} is not set — the robot cells of the still kinds depend on the run's dataset")
    return ds


def robot_static(ds: str) -> bool:
    """Is S-robot-static on for dataset `ds`?"""
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415 — numpy; this module stays stdlib at import

    return SC.on(SC.ROBOT_RULE, ds)


def slot_robot_off(table: dict) -> bool:
    """The authoring filters every robot body <-> this slot when the table's hand partner is OFF (one rule for the
    authoring and the ledger's check of what it filtered)."""
    return table["hand"] == "OFF"


def extra_kinds_in_play(kinds) -> tuple:
    """The kinds a statement lists: every EXTRA_KIND_COLLIDE kind, plus the scoped kinds that occur in `kinds` (an
    iterable of kind names or nested rows; None cells ignored)."""
    seen = {k for row in kinds for k in (row if isinstance(row, (list, tuple)) else [row]) if k is not None}
    return (*EXTRA_KIND_COLLIDE, *(k for k in EXTRA_SCOPED_KIND_COLLIDE if k in seen))


EXTRA_PARTNERS = ("hand", "arm", "objects")
# Whether the judge's arm guard measures the bodies of fixed extras (static slots: touched or receiving). A **separate
# statement** from the sim collision (the tables above and S-robot-static): placements where the arm passes through a
# fixed object are filtered by the judge.
# Reader = placement collision.support_surface.extra_measured. Values other than ON|OFF fail.
EXTRA_FIXED_JUDGE_ARM = "ON"


def extra_collide(kind: str, full: bool | None = None, ds: str | None = None) -> dict:
    """Collision partner table of an extra slot kind {hand · arm · objects: ON|OFF}; full = the full-physics override
    above (None: the switch); ds = the dataset S-robot-static is read for (None: `run_dataset`). An unknown kind, a
    table with other partners or a value other than ON/OFF fails."""
    if kind not in EXTRA_KIND_COLLIDE and kind not in EXTRA_SCOPED_KIND_COLLIDE:
        raise KeyError(f"no RE rule for extra slot kind {kind!r} — table {sorted(EXTRA_KIND_COLLIDE)}")
    t = EXTRA_KIND_COLLIDE[kind] if kind in EXTRA_KIND_COLLIDE else EXTRA_SCOPED_KIND_COLLIDE[kind]
    if tuple(sorted(t)) != tuple(sorted(EXTRA_PARTNERS)) or any(v not in ("ON", "OFF") for v in t.values()):
        raise ValueError(f"EXTRA_KIND_COLLIDE[{kind!r}]={t!r} — partners {EXTRA_PARTNERS} · values ON|OFF only")
    if full_physics() if full is None else full:
        return {"hand": "ON", "arm": "ON", "objects": "OFF" if kind in FULL_PHYSICS_OBJECTS_OFF else "ON"}
    if kind in ROBOT_STATIC_KINDS and robot_static(run_dataset() if ds is None else ds):
        t = {**t, "hand": "ON", "arm": "ON"}
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


def extra_collide_text(kinds=None) -> str:
    """One line of collision partners per kind (carried by the run log marker):
    `moving(hand ON · arm ON · objects OFF) · touched(…) · …`. kinds = the kinds to list (default every
    EXTRA_KIND_COLLIDE kind; `extra_kinds_in_play` adds the scoped ones)."""
    return " · ".join(
        f"{k}(" + " · ".join(f"{p} {extra_collide(k)[p]}" for p in EXTRA_PARTNERS) + ")"
        for k in (EXTRA_KIND_COLLIDE if kinds is None else kinds)
    )


# ── Object<->torso (RT): the torso collider (the 64-vertex visual hull) and the scene objects
# collide (0 filters). A row independent of presets; the audit row (collision_check) reads this statement and compares
# it with the measurement (0 filter pairs).
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
