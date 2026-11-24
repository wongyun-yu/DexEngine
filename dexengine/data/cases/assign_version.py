"""Key and versions of the hand–object assignment record (`rules.ASSIGN`). No module-level dependencies (only
hand_version reads the hand table in paths, lazily).

Used by hand_object_assign (ASSIGN-1 version) · assign_r (ASSIGN-2 version: rule R and the validity gate) ·
cases.taxonomy (stage 1 step 3, writes the record) · object_roles (step 4, reads it and writes roles) · data.stages
(the step-3 cell counts the hand's version). The assignment modules pull in numpy, so the stage table cannot import
them; only the versions live here. Trees on another version rerun from step 3. Each version covers its code:
ASSIGN_VERSION ← hand_object_assign.py · ASSIGN2_VERSION ← assign_r.py · assign_r_frames.py · assign_r_inputs.py
(rule R and its input reader form one version); bump it when that code changes.

Two versions:
  ASSIGN-1  the hand_object_assign.assign result as is.
  ASSIGN-2  assignment with a validity gate. When every moving object that rule R requires (v3.1 measurement + TACO
            source task objects + riders follow their carrier) is in the ASSIGN-1 scene (main objects), it is valid and
            per_side stays ASSIGN-1 (missing still objects go to gate.static_required, candidates to add at render
            time); if any is missing it is invalid and per_side = R. Each record's `gate` field holds the verdict.
The version in use is one value shared by all hands (ACTIVE), not a per-hand axis. The contact sidecar layer
(precompute/<ds>/contact_mano_base) is shared across hands and its generation includes the assignment version
(dexengine.data.contact.fingerprint.GEN_RULES); per-hand versions would put one hand's step 5 out of generation with
another hand's live base, which stops it (rc 4) or breaks the other hand's consumers. Hands that take a version =
registered hands (paths.HAND_SPEC_DIR); an unregistered hand fails.
"""

ASSIGN_KEY = "ASSIGN"  # key under base task_info `rules`
# surface-point carry · longest run · assisting hand (dropped only when carry time < threshold) · object tags · quality
# inputs
ASSIGN_VERSION = "ASSIGN-1"
ASSIGN2_VERSION = "ASSIGN-2"  # ASSIGN-1 + validity gate of rule R (v3.1 · TACO source task objects · riders)
ASSIGN_VERSIONS = (ASSIGN_VERSION, ASSIGN2_VERSION)

#: Assignment version shared by all hands: (version, expiry condition, owner). ASSIGN-2 has no expiry ("-").
#: Reading a record (record) from a base still on ASSIGN-1 fails with KeyError by design (a half-switched layer is
#: never read silently); such bases rerun from stage 1 step 3.
ACTIVE = (ASSIGN2_VERSION, "-", "-")


def version() -> str:
    """Assignment version shared by all hands (the ACTIVE version); KeyError outside the version vocabulary."""
    v = ACTIVE[0]
    if v not in ASSIGN_VERSIONS:
        raise KeyError(f"assignment version ACTIVE {v!r} ∉ {ASSIGN_VERSIONS}")
    return v


def hand_version(hand: str) -> str:
    """Assignment version of that hand's (spec dir) base = the shared version (version()). KeyError for a hand outside
    the registered spec dirs (paths.HAND_SPEC_DIR). paths is imported lazily, so readers of only the versions and the
    key (dexengine.placement.judge.input_record · dexengine.data.contact.fingerprint) pull in nothing."""
    from dexengine.core import (
        paths as _P,  # noqa: PLC0415 — owner of the hand table (no numpy; the caller has set up the path)
    )

    specs = sorted(set(_P.HAND_SPEC_DIR.values()))
    if hand not in specs:
        raise KeyError(
            f"unregistered hand {hand!r} (paths.HAND_SPEC_DIR spec dirs: {specs}) — "
            "call it by its spec dir (no silent default)"
        )
    return version()


def record(ti: dict, hand: str) -> dict:
    """Assignment record of a base task_info; the single entry point for readers. KeyError when it is missing or not
    on that hand's version (hand_version); no silent default."""
    want = hand_version(hand)
    a = (ti.get("rules") or {}).get(ASSIGN_KEY)
    got = a.get("version") if isinstance(a, dict) else a
    if got != want:
        raise KeyError(
            f"rules.{ASSIGN_KEY} version {got!r} ≠ {want}({hand}) — run stage 1 step 3 (cases/taxonomy.py --write)"
        )
    return a


#: Fields counted by the [ASSIGN] marker of stage 1 step 3: the three gate verdicts (assign_r.gate) · no verdict
#: (ASSIGN-1 record) · no assignment (no objects).
GATE_VERDICTS = ("valid", "valid_static", "invalid")
GATE_NO, GATE_NONE = "-", "none"


def gate_of(rec: dict | None) -> str:
    """[ASSIGN] field of one assignment record: no record GATE_NONE · no verdict GATE_NO · otherwise gate.verdict."""
    if rec is None:
        return GATE_NONE
    return (rec.get("gate") or {}).get("verdict") or GATE_NO


def marker(hand: str, version: str, counts) -> tuple[str, str]:
    """Confirmation line `[ASSIGN] <hand> <version> · …` of stage 1 step 3 and the version–verdict conflict ("" if
    none).
    ASSIGN-2 has a verdict on every record and ASSIGN-1 none; on a conflict the caller fails."""
    line = (
        f"[ASSIGN] {hand} {version} · valid {counts.get('valid', 0)} · static added {counts.get('valid_static', 0)}"
        f" · invalid {counts.get('invalid', 0)} · no verdict {counts.get(GATE_NO, 0)}"
        f" · no assignment {counts.get(GATE_NONE, 0)}"
    )
    gated = sum(counts.get(v, 0) for v in GATE_VERDICTS)
    other = sorted(set(counts) - {*GATE_VERDICTS, GATE_NO, GATE_NONE})
    if other:
        return line, f"unknown verdicts {other}"
    if version == ASSIGN2_VERSION and counts.get(GATE_NO):
        return line, f"{version} but assignments without a verdict {counts[GATE_NO]}"
    if version == ASSIGN_VERSION and gated:
        return line, f"{version} but assignments with a verdict {gated}"
    return line, ""
