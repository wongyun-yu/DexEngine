"""Support-disc collision rows: RD (robot<->support disc) and RO (support disc<->object). They do not depend on the
preset: the disc stamp `collide` (value owner coords.DISC_COLLIDE, stamped per disc by the bake) states them. Authoring
(collision_authoring) builds the physics from these tables and the oracle (collision_preset_check) compares the
measurement with them. Standard library only, like collision_presets (the preset rows).
"""

from __future__ import annotations

# ── A row independent of presets: robot<->support disc (RD, rule D40: disc-hand and disc-arm
# collisions off) ──
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


# ── The second row of the same stamp: support disc<->object (RO, rule D11″ (d): a disc supports its own
# object only) ──
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
