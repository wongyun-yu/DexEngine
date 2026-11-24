"""Rule-set version of the final ledgers: retired versions, the steps that changed and the clips they touch,
equivalent versions, and the reasons a ledger row is out of date."""

from __future__ import annotations

import json
import os

from dexengine.core.errors import InputMissingError
from dexengine.core.rules.profiles import DATASET_PROFILE
from dexengine.core.rules.source import _paths

#: Rule-set version name — the **unit of freezing** and the stamp of a judge chunk. stage2 prints the `[RULESET]`
#: marker and records `ruleset=` in run.env, so the version a chunk was judged with is kept. Bump it when any rule
#: (D table), threshold or budget changes. There is **only one** version: every rule lives here (RULES, thresholds,
#: EXHAUST_RETRY), and an older stamp only reads as "not the current stamp" (versions are never ordered).
RULESET_VERSION = "v5-20261012a"

#: **Stamp retirement rule** (single owner, D38) — lists the datasets whose judge input changed for the whole axis
#: (value = reason). Their chunks whose run.env stamp differs from RULESET_VERSION (older or missing) are **always
#: retired** by the fold (reason `ruleset:<stamp|(none)>≠<current>(<why>)`, the rows are kept in ledger_retired) and
#: replaced by a re-judge under the current version. Chunks of unlisted datasets are not judged by their stamp; the
#: fold's structural checks own their validity.
#: The current version changed the judge input of **every** dataset, so the rule covers every registered dataset
#: (DATASET_PROFILE). The reason text below is data: it is written into ledger_retired.tsv; the next fold compares it
#: and rewrites rows whose reason changed (final_ledger.merge_retired counts them as updated).
RULESET_WHY = (
    "search-filter proxy geometry = capsules (arm_proxy.COLL_PROXY — torso · desk filter only · revised arm "
    "hull · fingers = table capsules · palm = surface points · radius = 40 % quantile of the support gap (q40)) · "
    "adopt·bake·gate·final guard·2e = exact guard on the original mesh (arm_proxy.GUARD_KIND · the first "
    "candidate that passes the mesh guard is adopted · the judge's last desk decision = original mesh · capsule "
    "filter loose by the protrusion ε) · fingerprint collider +capsule-q40-<table>+guard-mesh+desk-mesh · "
    "support-kind judgement (support object · support↔arm term · disc to the end · object-object first frame "
    "≤ 10 mm · fingerprint jfp3) · resting-hand fingers = rest pose · finger range = retarget clipping · torso "
    "split · retiming = arm + fingers (HandSpec applied limits — same function for judge prediction · bake · "
    "gate) · judge input overhaul (v5): judge fingerprint jfp2 (stage hash · collider) · strict collision "
    "ruler = sim collider surface (desk · both arms · torso · fingers) · clearance sidecar desk component + "
    "ruler identity · one judge window · desk mask = spawn set · (v4) standard hand–object assignment · piece "
    "window · mid-air support gap · D24 hull generation · judge-guard support↔arm term = supports with the "
    "candidate placement applied (support_surface.placed_supports — judge = gate)"
)
RULESET_RETIRE = dict.fromkeys(DATASET_PROFILE, RULESET_WHY)

#: **Predicates for the clips a version change touches** (D38).
#: Row = (old version, new version, predicate name, reason) — one per version bump.
#: An old-stamp chunk follows the rows from its stamp to the current version (ruleset_chain):
#:   chained      the predicate is measured per row (ruleset_row_reason). If any is true (the change can alter that
#:                clip's judge input or result) **only that row** retires (`ruleset:<old>→<new> touched(…)`);
#:                if all are false the row is carried (note `ruleset:carried(<old>→<current>)`).
#:   not chained  (a version with no predicate, no stamp, an unknown version) the whole chunk retires
#:                (ruleset_retire_reason).
#: A row whose predicate cannot be measured (judge input missing or unreadable) retires — never silently carried.
#: A version bump either adds a row (predicate) here or, by adding none, retires everything.
#: Marker = the fold log line `[RULESET_RETIRE]`. The reason text is data (see above).
RULESET_STEPS = (
    (
        "v5-20261005e",
        "v5-20261012a",
        "judge_supports",
        "judge-guard support↔arm term = supports with the candidate placement applied — only clips whose judge input "
        "(base) has a support object can change verdict (coords.scene_support_poses)",
    ),
)


def _pred_judge_supports(ds: str, emb: str, hand: str, clip: str) -> bool:
    """Does the judge input (stage2's base place dexengine.core.paths.ref_base_dir/<clip>) hold a support object
    (objects[] role "support")? This is exactly the input of the support↔arm term (support_surface.scene_supports
    builds supports with this constructor). A missing or unreadable judge input raises (→ retire)."""
    import numpy as _np  # noqa: PLC0415

    _P = _paths()
    from dexengine.placement.coords import core as _VC  # noqa: PLC0415 — owner of the support constructor

    d = _P.ref_base_dir(ds, hand, emb) / clip
    ti = json.loads((d / "task_info.json").read_text(encoding="utf-8"))
    q = _np.load(d / "0" / "trajectory_kinematic.npz")["qpos"]
    return bool(_VC.scene_support_poses(ti, q))


#: Predicate name → function(ds, emb, hand, clip) → True = touched. A RULESET_STEPS name missing here fails at
#: import.
RULESET_PREDICATES = {"judge_supports": _pred_judge_supports}
if _miss := [st[2] for st in RULESET_STEPS if st[2] not in RULESET_PREDICATES]:
    raise KeyError(
        f"RULESET_STEPS predicates {_miss} are not in RULESET_PREDICATES — "
        "the clips a version change touches cannot be measured"
    )


def ruleset_chain(stamp: str | None) -> list | None:
    """RULESET_STEPS rows chaining an old stamp to the current version (RULESET_VERSION) — [] when current, None
    when they do not chain (unregistered version, no stamp, a cycle)."""
    if stamp == RULESET_VERSION:
        return []
    by_old = {st[0]: st for st in RULESET_STEPS}
    out, cur, seen = [], stamp, set()
    while cur != RULESET_VERSION:
        st = by_old.get(cur)
        if st is None or cur in seen:
            return None
        seen.add(cur)
        out.append(st)
        cur = st[1]
    return out


#: D37: an UNPROVEN clip that stopped **because of the budget** at the canonical budget (--exhaust-max 1024; candidates
#: ≤ factor × budget, or reached the assembly stage) is re-judged **once** with budget × factor, from the cell where
#: it stopped. This table is the policy (from, default factor, round cap). Stage 2 runs the round itself after its
#: status ledger (scripts/stage2/_ground_lib.sh ground_d37_round · owner of targets · record · resume table ·
#: retry tag = dexengine/placement/judge/retry.py): the targets go to a sibling retry chunk run with
#: `--retry-budget <record> --exhaust-resume <table>` (run.env `retry_budget=` · `exh_resume=`; marker
#: `[RETRY_BUDGET] r1 exhaust_max=<budget> n=<targets>`). Still UNPROVEN at that budget = terminal.
EXHAUST_RETRY = {"from": 1024, "factor": 2, "max_rounds": 1}

#: **Owner of the D37 retry campaign's paths and markers** (its readers take them from here).
#: The retry queue (`<campaign>/queue_retry/{manifest.tsv,<axis>.txt,chunks/}`) = the set the last round **actually
#: dispatched**; `unproven_retry__<hand>.tsv` = the list drawn now. Taken at different times, the counts differ — do
#: not mix them. The axis hand's shared campaign id is not a constant; `axis_campaign()` measures it from the ledger.
RETRY_QUEUE_DIRNAME = "queue_retry"  # inside the campaign directory
RETRY_RUN_MARKER = "[RETRY_BUDGET]"  # applied marker that stage2 prints into run.log


def campaign_dir():
    """Folder of the running campaign = env `CAMPAIGN_DIR`. **No default.**

    With a default folder, a new hand launched without a folder would write into the axis hand's folder. Places that
    need the axis hand's folder (shared dashboard, shared audit) call `axis_campaign_dir()` — a different question.
    """
    from pathlib import Path as _P

    env = os.environ.get("CAMPAIGN_DIR", "").strip()
    if not env:
        raise InputMissingError(
            "[campaign_dir] CAMPAIGN_DIR is not set — the caller names the campaign folder (no "
            "default). For the axis hand's shared folder use axis_campaign_dir()"
        )
    return _P(env)


def retry_queue_dir(camp=None):
    """D37 retry dispatch queue. It may not exist yet (nothing sent) — the caller checks.
    With `camp`, the queue of that campaign folder (the shared dashboard passes `axis_campaign_dir()`), else
    `campaign_dir()`."""
    return (camp or campaign_dir()) / RETRY_QUEUE_DIRNAME


#: **Equivalent stamps** (D38) — the reason a non-current stamp is **not re-judged**, written in code. A stamp not
#: listed here raises instead of passing silently: whoever bumps the version must choose, for every old stamp,
#: "equivalent (a line here)" or "retired (RULESET_RETIRE)".
RULESET_EQUIVALENT: dict = {
    "v1-20260921": "judge rules unchanged (every threshold of the candidates, case taxonomy, coords, judge, dz "
    "policy and canon env is the same). The only differences are ① the training gate T3 (after "
    "judging) ② the taco source layer (retired via RULESET_RETIRE) ③ D36/D37 (structural checks, "
    "retry) — none changes a judge result",
    None: "chunk from before stamps existed. It only lacks a stamp; the judge rules are the same — the fold's "
    "structural checks own its validity (REGEN_EVENTS, REGEN_EVENTS_UNITS, JUDGE_INPUT_EVENTS, "
    "units_replaced, axis_sets)",
}


def ruleset_note(ds: str, stamp: str | None) -> str:
    """**Stamp note** for a ledger row — empty for the current stamp, `ruleset:equivalent(<stamp|none>)` for an
    equivalent one.

    Why a note: retiring every old or unstamped row would mean a mass re-judge, but leaving them unmarked leaves rows
    that sit there **for no stated reason**. So the status is kept (not retired) and the row says **why it stays** —
    RULESET_EQUIVALENT owns the reasons. Retired rows (RULESET_RETIRE) are never folded, so they get "" here."""
    if stamp == RULESET_VERSION or ruleset_retire_reason(ds, stamp):
        return ""
    if ds in RULESET_RETIRE and ruleset_chain(stamp):  # carried: predicate says untouched (touched rows retired)
        return f"ruleset:carried({stamp}→{RULESET_VERSION})"
    if stamp not in RULESET_EQUIVALENT:
        raise KeyError(
            f"unknown rule-set stamp {stamp!r} ({ds}) — add an equivalence reason to RULESET_EQUIVALENT "
            f"or a retirement to RULESET_RETIRE. It is never let through silently"
        )
    return f"ruleset:equivalent({stamp or 'none'})"


def ruleset_retire_reason(ds: str, stamp: str | None) -> str | None:
    """Reason string when the chunk's stamp (run.env `ruleset=`, None when absent) falls under the **stamp retirement
    rule**, else None — the fold's single entry.

    Only two conditions: ① the dataset is in RULESET_RETIRE and ② stamp ≠ RULESET_VERSION. Versions are never
    ordered (there is one). Every non-current stamp (old, unknown, none) means the same — "not judged under the
    current version" — and the reason records that stamp as is."""
    why = RULESET_RETIRE.get(ds)
    if why is None or stamp == RULESET_VERSION:
        return None
    if ruleset_chain(stamp) is not None:  # chained by predicates — per row, not per chunk (ruleset_row_reason)
        return None
    return f"ruleset:{stamp or '(none)'}≠{RULESET_VERSION}({why})"


def ruleset_row_reason(ds: str, emb: str, hand: str, clip: str, stamp: str | None) -> str | None:
    """Retirement reason of **one row** of a predicate-chained old-stamp chunk (None = carried) — the fold's single
    entry (called only for chunks where ruleset_retire_reason is None). Each row's predicate is measured: true gives
    `ruleset:<old>→<new> touched(<why>)`, unmeasurable gives `ruleset:<old>→<new> not measured(…)` (
    retired, never silently carried). A stamp that does not chain gets the whole-chunk reason."""
    if RULESET_RETIRE.get(ds) is None or stamp == RULESET_VERSION:
        return None
    chain = ruleset_chain(stamp)
    if chain is None:
        return ruleset_retire_reason(ds, stamp)
    for old, new, pred, why in chain:
        try:
            hit = RULESET_PREDICATES[pred](ds, emb, hand, clip)
        except Exception as e:  # noqa: BLE001 — an unmeasured row retires (never counted as carried)
            return f"ruleset:{old}→{new} not measured({pred}: {type(e).__name__}: {str(e)[:120]})"
        if hit:
            return f"ruleset:{old}→{new} touched({why})"
    return None
