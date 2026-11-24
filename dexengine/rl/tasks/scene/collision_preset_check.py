"""Collision preset check: the declared COLL_PRESET text normalized and compared with the measured row values of
collision_check.audit_env (split from collision_check, which keeps the truth table and the audit)."""

from __future__ import annotations

import re

from dexengine.rl.tasks.scene.collision_presets import (
    DISC_OBJ_ROW,
    DISC_ROW,
    EXPECTED,
    OBJ_TORSO_COLLIDE,
    OBJ_TORSO_ROW,
    SUPPORT_COLLIDE,
    SUPPORT_ROW,
    expected_disc,
    expected_disc_obj,
    expected_support,
)

_PAREN = re.compile(r"\([^)]*\)")


_PAIR = re.compile(r"^(\S+):(\S+) (\S+):(\S+)$")


#: R3 value of a body with no torso-class body at all (audit_env gives it from the body count only). With no pairs it is
#: the same physics as any preset expectation (OFF or 'right:OFF left:ON'); check_preset lets only this R3 value pass
#: (other rows and other values are compared as usual).
VACUOUS_TORSO = "—(no torso)"


VACUOUS_ROWS = ("R3",)


def norm_one(v: str) -> str:
    """'OFF(filter 61)' -> 'OFF'. 'hand:OFF(filter 46) arm:OFF(filter 12)' -> 'OFF' when the values agree and only the
    evidence differs."""
    v = _PAREN.sub("", v).strip()
    m = _PAIR.match(v)
    if m and m.group(2) == m.group(4):
        return m.group(2)
    return v


def normalize(effective: dict[str, str]) -> dict[str, str]:
    """Strip the evidence parentheses into the form of the expectation table (EXPECTED)."""
    return {k: norm_one(v) for k, v in effective.items()}


def check_preset(rows: list[dict], preset: str, disc_collide: str | None = None) -> list[str]:
    """Compare the effective table of **every** audited env with the expectation table. Returns the mismatches as
    strings.

    The preset-independent RD (robot <-> support disc) and RO (disc <-> object) are compared here too: the expectation
    comes from the statements (the audit row's `disc_slots` = the cfg that makes the discs, `objects` = the scene object
    count, `disc_collide` = the env cfg's `support_disc_collide`) through `collision_presets.expected_disc` and
    `expected_disc_obj`, the measurement from the audit row."""
    exp = EXPECTED[preset]
    bad = []
    for r in rows:
        act = normalize(r["effective"])
        env = r["env"].rsplit("/", 1)[-1]
        for k, v in exp.items():
            if k in VACUOUS_ROWS and r["effective"].get(k) == VACUOUS_TORSO:
                continue  # no target (0 torso bodies): no pairs, the same physics as expected
            if act.get(k) != v:
                bad.append(f"{env}.{k}: actual {r['effective'].get(k)!r} ≠ expected {v!r}")
        try:
            want = expected_disc(r["disc_slots"], disc_collide)
        except KeyError as e:
            bad.append(f"{env}.{DISC_ROW}: {e}")
            continue
        if act.get(DISC_ROW) != want:
            bad.append(
                f"{env}.{DISC_ROW}: actual {r['effective'].get(DISC_ROW)!r} ≠ expected {want!r}"
                f"(disc slots {r['disc_slots']} · collide={disc_collide!r})"
            )
        try:
            want_o = expected_disc_obj(r["disc_slots"], disc_collide, r["objects"])
        except KeyError as e:
            bad.append(f"{env}.{DISC_OBJ_ROW}: {e}")
            continue
        if want_o == "ON" and act.get(DISC_OBJ_ROW) not in ("—", "ON") and not r["filtered"].get(DISC_OBJ_ROW, 0):
            # pedestal: 0 filtered disc<->object pairs means on (the same stage as the measured 'own object only · 0
            # filters')
            pass
        elif act.get(DISC_OBJ_ROW) != want_o:
            bad.append(
                f"{env}.{DISC_OBJ_ROW}: actual {r['effective'].get(DISC_OBJ_ROW)!r} ≠ expected {want_o!r}"
                f"(disc slots {r['disc_slots']} · objects {r['objects']} · collide={disc_collide!r})"
            )
        bad += _support_torso_bad(r, act, env)
    return bad


def _support_torso_bad(r: dict, act: dict, env: str) -> list[str]:
    """RS.<partner> (expected expected_support; 0 support slots is '—') and RT (expected OBJ_TORSO_COLLIDE)
    comparison."""
    out = []
    for k in SUPPORT_COLLIDE:
        row, want = f"{SUPPORT_ROW}.{k}", expected_support(r["support_slots"], k)
        if act.get(row) != want:
            out.append(
                f"{env}.{row}: actual {r['effective'].get(row)!r} ≠ expected {want!r} (support slots "
                f"{r['support_slots']})"
            )
    if act.get(OBJ_TORSO_ROW) != OBJ_TORSO_COLLIDE:
        out.append(
            f"{env}.{OBJ_TORSO_ROW}: actual {r['effective'].get(OBJ_TORSO_ROW)!r} ≠ expected {OBJ_TORSO_COLLIDE!r}"
        )
    return out
