"""Vocabulary of the per-clip terminal states — no imports but the dependency-free core.legacy_text.

The status ledger (`ledger.py`) owns this vocabulary and re-exports these names; the values live here so that
readers (the gate, the judge, the selector) can read them without importing `ledger.py` and its dependencies
(`judge.input_record` imports this module, so reading it through `ledger.py` would be circular).

Named subsets exist only where they carry a meaning (element order = STATES order). The gate-table ↔ ledger contract
(gate-table header, verdict vocabulary, the gate cells the ledger carries) and the rule for non-verdict failures that
go to BAKE_ERROR (`measured`, `error_systematic`) live here for the same reason: `gate.core` and `gate.f0_pen` both
call them without importing each other.
"""

from dexengine.core import legacy_text as _LT

#: Terminal states — ADOPTED (placement adopted) · EXHAUSTED (search exhausted = proof that no placement exists) ·
#: UNPROVEN (budget cap, undecided) · GUARD_FAIL (a bake/gate physics guard rejected it) · SOURCE_DEFECT (input
#: defect) · BAKE_ERROR (a bake failure that is not a physics-guard rejection — contract violation, exception,
#: timeout, dead server, failed placement record; it is re-baked; the rule is ledger.bake_fail_status) · HELD (the f0
#: object repair refused a side — not judged; a rule rejection: the reference cannot be made penetration-free within
#: the repair rule, so the trajectory is dropped; stage2 f0_held.txt, owner f0.shadow_status.HELD).
STATES = ("ADOPTED", "EXHAUSTED", "UNPROVEN", "GUARD_FAIL", "SOURCE_DEFECT", "BAKE_ERROR", "HELD")
#: A clip with no terminal state — not an element of STATES. ledger.main() counts such clips under this name and does
#: not write the ledger (rc 2). Readers of the folded ledger treat it as "output lost = judge again".
MISSING = "MISSING"

#: The judge found no placement — search exhausted or budget cap.
PLACEMENT_FAILED = ("EXHAUSTED", "UNPROVEN")
#: Non-adoptions proven by the judge input fingerprint (judge.input_record) — produced by the judge and the bake
#: guard after it. These clips have no tree.
JUDGED_STATUSES = PLACEMENT_FAILED + ("GUARD_FAIL",)
#: Rejected — a non-adoption blocked by a rule, physics or the input. BAKE_ERROR is a code/environment failure
#: (repair, retry, re-bake), not a rejection.
REJECTED = JUDGED_STATUSES + ("SOURCE_DEFECT", "HELD")
#: Gate-table verdict cell → the clip's terminal state (None = does not block). Written by gate.core, read by the
#: ledger; a value outside this table dies. Order = priority: of the two sides the earlier value wins — FAIL (physics
#: rejection, terminal) > ERROR (only undeclared measurement exceptions on that side; not a data verdict, re-baked) >
#: PASS.
GATE_VERDICT_STATE = {"FAIL": "GUARD_FAIL", "ERROR": "BAKE_ERROR", "PASS": None}
#: Per-side post-hoc comparison cells of the gate (hand ↔ own object first-frame burial · hand ↔ support first-frame
#: penetration — recorded, not filtered): depth in mm (one decimal, as measured by f0_pen.frame0_penetration and
#: support_surface.hand_support_mm) · 'N/A' = nothing to measure. Readers read them by clip name only.
GATE_DEPTH_COLUMNS = ("f0_hand_ownobj_mm", "hand_support_mm")
#: Gate-table header (one row per clip and side) — verdict and reason, then GATE_DEPTH_COLUMNS. Readers find columns by
#: header name. New cells go at the end only (shell readers use positions: stage 2 reads clip $1 and side $3).
GATE_COLUMNS = (
    "clip",
    "anchor",
    "side",
    "T",
    "stretch_ratio",
    "r_p99",
    "r_max",
    "err_p99mm",
    "grasp_offset",
    "pen_mm",
    "xr_obj_rel_mm",
    "verdict",
    "reason",
) + GATE_DEPTH_COLUMNS
#: The gate cells the status ledger and the folded ledger carry on adopted rows — source (gate cell, side) and name
#: `<cell>_<side>` (same order).
GATE_DEPTH_SOURCE = tuple((c, s) for c in GATE_DEPTH_COLUMNS for s in ("right", "left"))
GATE_DEPTH_LEDGER = tuple(f"{c}_{s}" for c, s in GATE_DEPTH_SOURCE)
#: Systematic threshold for non-verdict failures that go to BAKE_ERROR (gate unmeasurable, output defect, contact
#: derivation error): when one chunk has at least ERROR_SYSTEMATIC_MIN of them and they are all of the measured clips
#: or at least ERROR_SYSTEMATIC_FRAC of them, it is an environment/code defect, not a clip verdict — the chunk stops
#: and writes no states (a single deterministic failure does not stop it). The rule is error_systematic.
ERROR_SYSTEMATIC_MIN = 2
ERROR_SYSTEMATIC_FRAC = 0.1


#: Reason heads of judge rejection rows (verdict `-`) that prove "no place to put it, before any search" —
#: noconfirm_fold folds them to EXHAUSTED. no_volume = no candidate volume at all · no_place_spot = no grid placement
#: puts the main object (start and placed position) and the required static objects outside the robot footprint and
#: inside the desk top (written by the judge, rule place_rule) · no_receiver_spot = the old name of that rule (read from
#: old ledgers only; nothing writes it).
VOLUME_NONE = "no_volume"
PLACE_NO_SPOT = "no_place_spot"
RECV_NO_PLACE = "no_receiver_spot"
NO_PLACE_HEADS = (VOLUME_NONE, PLACE_NO_SPOT, RECV_NO_PLACE)


def noconfirm_fold(prev: str | None, verdict: str, why: str = "") -> str | None:
    """Fold one judge rejection row (noconfirm_<combo>.tsv: clip · verdict · reason) into the clip's state — the
    status ledger and the judge's `[CLIP_T]` line call the same function. A rejection row never changes ADOPTED.
    EXHAUSTED overrides an earlier value; TRUNCATED → UNPROVEN and `-` + NO_PLACE_HEADS → EXHAUSTED only set an empty
    state. Other rows (anchor failure, …) set nothing. The reason is read only for `-` rows: a head in the old (Korean)
    names is not migrated data and dies (ValueError) instead of setting nothing."""
    if verdict == "-" and _LT.is_legacy(head := why.split("(", 1)[0].split(":", 1)[0].split(" ", 1)[0]):
        raise ValueError(f"reject row reason head {head!r} is a legacy value — {_LT.MIGRATE_HINT}")
    if prev == "ADOPTED":
        return prev
    if verdict == "EXHAUSTED":
        return "EXHAUSTED"
    if verdict == "TRUNCATED":
        return prev or "UNPROVEN"
    if verdict == "-" and why.startswith(NO_PLACE_HEADS):
        return prev or "EXHAUSTED"
    return prev


def error_systematic(bad: int, tried: int) -> bool:
    """Whether `bad` non-verdict failures out of `tried` measured clips are systematic (threshold above)."""
    return bad >= ERROR_SYSTEMATIC_MIN and (bad == tried or bad >= ERROR_SYSTEMATIC_FRAC * tried)


class GateUnmeasured(RuntimeError):
    """A gate scene term could not be measured and the exception is not a declared data property — both sides of the
    clip are ERROR (not a data verdict; GATE_VERDICT_STATE → BAKE_ERROR). The message is the gate reason."""


def measured(n: dict, head: str, data_exc: tuple, fn):
    """Measure one gate scene term (the classification of non-verdict failures; pairs with error_systematic):
    value of ``fn()`` → (value, None) · a declared data property (`data_exc`) → (None, '<head> unmeasurable(…)' — the
    caller drops both sides with that reason) · any other exception → GateUnmeasured. Counts measured, unmeasurable
    and error in ``n`` (read by the gate's chunk_rc). Called by gate.core and f0_pen.gate_reason."""
    try:
        v = fn()
    except Exception as e:  # noqa: BLE001 — classified: a declared data property is a reason, the rest ERROR
        n["unmeasurable"] += 1
        why = f"{head} unmeasurable({type(e).__name__}: {str(e)[:80]})"
        if data_exc and isinstance(e, data_exc):
            return None, why
        n["error"] = n.get("error", 0) + 1
        raise GateUnmeasured(why) from e
    n["measured"] += 1
    return v, None
