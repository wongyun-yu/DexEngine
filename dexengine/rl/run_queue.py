"""The RL campaign queue (dexengine/rl/campaign.py reads it): what blocks a launch (BLOCKERS · SUPERSEDED), the
order tokens, the queue from the manifests, the judge path, and completion. Two views: manifest-level (block_reason,
a manifest alone: run_inputs) and campaign-level (campaign_block_reason · campaign_plan_queue: the runner and its
tick). No I/O.
"""

from __future__ import annotations

import collections

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.rl.cell_state import bkey

#: What stands between a bundle class and a launch: (class, text) rows. class = that class's bundles wait until
#: --allow-class names it; None = the whole campaign (--launch refuses). Empty: nothing blocks a launch.
BLOCKERS: tuple = ()


#: Campaign-scoped supersession (campaign, dataset, by, text): campaign `by` (a manifest of regenerated data under a
#: new campaign id) replaces that campaign's bundles of that dataset. They leave the campaign's queue for good (no
#: --allow-class lifts it: accounting, not a block) and the campaign completes without them; a superseded bundle with
#: a launch record raises (a result already run is never erased). A row goes in only when `by`'s manifest exists.
SUPERSEDED: tuple = ()
SUPERSEDED_PREFIX = "superseded → "


def superseded_by(b: dict, campaign: str) -> str:
    """The campaign replacing bundle `b` of `campaign` ("" = none). No campaign, a bundle without a dataset, or two rows
    for one (campaign, dataset) raise."""
    if not campaign:
        raise ValueError("no campaign id for a campaign-scoped decision")
    hit = [by for c, d, by, _w in SUPERSEDED if c == campaign and d == b["dataset"]]
    if len(hit) > 1:
        raise ValueError(f"SUPERSEDED has {len(hit)} rows for ({campaign}, {b['dataset']})")
    return hit[0] if hit else ""


def collect_state(b: dict, campaign: str, state: str, run: str) -> str:
    """The state column of the campaign's result table (acceptance.collect): a superseded bundle is 'superseded:<by>'
    (manifest accounting, outside the six bundle states cell_state.STATES) and raises if it has a launch record
    (run ≠ '-'); every other bundle keeps its state."""
    by = superseded_by(b, campaign)
    if by and run != "-":
        raise ValueError(
            f"{b.get('hand')}/{b['id']}: superseded bundle (→ {by}) has a launch record {run} — a result is "
            "never erased"
        )
    return f"superseded:{by}" if by else state


def allow_tokens() -> set[str]:
    """The tokens --allow-class accepts: the classes BLOCKERS blocks now (an allowance nothing blocks dies)."""
    return {k for k, _w in BLOCKERS if k}


def parse_order(order: str, have: set[str], defer: set[str]) -> list[str]:
    """Order tokens, checked: each is a registered hand id or @artic (anything else dies: a misspelt hand would
    silently drop a whole hand); a hand without a manifest dies unless deferred explicitly."""
    toks = [t for t in order.split(",") if t]
    known = set(P.HAND_SPEC_DIR)
    bad = [t for t in toks if t != "@artic" and t not in known]
    if bad:
        raise SystemExit(f"[campaign] unknown --order tokens {bad} — registered hand ids {sorted(known)} or @artic")
    if len(set(toks)) != len(toks):
        raise SystemExit(f"[campaign] a token appears twice in --order: {toks}")
    miss = [t for t in toks if t != "@artic" and t not in have and t not in defer]
    if miss:
        raise SystemExit(f"[campaign] hands without a manifest {miss} — name them with --defer-hand to defer them")
    stray = sorted(defer - set(toks))
    if stray:
        raise InputMissingError(f"[campaign] --defer-hand {stray} is not in --order")
    return toks


def order_bundles(mans: dict[str, dict], toks: list[str]) -> list[dict]:
    """The campaign queue from checked order tokens: each hand's bundles in order, its articulated bundles only after
    an @artic token. Deferred hands (no manifest) contribute none."""
    out, seen = [], []
    for tok in toks:
        if tok == "@artic":
            out += [dict(b, hand=h) for h in seen if h in mans for b in mans[h]["bundles"] if b["klass"] == "artic"]
            seen = [h for h in seen if h not in mans] + ["@"]
            continue
        after = "@" in seen
        out += [dict(b, hand=tok) for b in mans.get(tok, {}).get("bundles", []) if after or b["klass"] != "artic"]
        seen.append(tok)
    return out


def judge_supported(b: dict) -> bool:
    """The judge sidecar's input contract: SIDE=both|right|left (two-hand and one-hand runs; judge_sidecar.sh refuses
    anything else)."""
    return b["side"] in ("both", "right", "left")


#: Why a bundle without a judge path is not queued (judge_supported decides it).
NO_JUDGE = "no judge path — the judge sidecar takes SIDE=both|right|left only (two-hand and one-hand runs)"


def block_reason(b: dict, allow: set[str], no_judge_ok: bool = False) -> str:
    """Why a bundle may not be queued ("" = it may): a class blocker not allowed, or no judge path (a bundle that can
    train but never reach done; only with --allow-no-judge). A bundle without a dataset is a malformed manifest row
    and raises KeyError."""
    if "dataset" not in b:
        raise KeyError(f"bundle {b.get('id')} has no dataset — malformed manifest row")
    for k, why in BLOCKERS:
        if k == b["klass"] and k not in allow:
            return why
    return "" if judge_supported(b) or no_judge_ok else NO_JUDGE


def campaign_block_reason(b: dict, allow: set[str], campaign: str, no_judge_ok: bool = False) -> str:
    """block_reason inside a campaign: a superseded bundle first (SUPERSEDED_PREFIX + the replacing campaign; no token
    lifts it), then the manifest-level reasons."""
    by = superseded_by(b, campaign)
    if by:
        why = next(w for c, d, x, w in SUPERSEDED if (c, d, x) == (campaign, b["dataset"], by))
        return f"{SUPERSEDED_PREFIX}{by} — {why}"
    return block_reason(b, allow, no_judge_ok)


def campaign_plan_queue(
    bundles: list[dict], done: set[tuple[str, str]], allow: set[str], campaign: str, no_judge_ok: bool = False
) -> list[dict]:
    """The launch queue inside a campaign (campaign_block_reason)."""
    return [b for b in bundles if bkey(b) not in done and not campaign_block_reason(b, allow, campaign, no_judge_ok)]


def campaign_view(
    mans: dict, bundles: list[dict], states: dict, allow: set[str], campaign: str, no_judge_ok: bool = False
) -> tuple[set, list[dict], collections.Counter]:
    """The runner's start view of one campaign: the set check over the manifests first (complete: a state outside them
    or a launched superseded bundle raises) → (done keys, the queue in order, blocked reasons counted)."""
    keys = [(h, b["id"]) for h, m in mans.items() for b in m["bundles"]]
    complete(keys, states, {(h, b["id"]) for h, m in mans.items() for b in m["bundles"] if superseded_by(b, campaign)})
    done = {k for k, s in states.items() if s == "done"}
    return (
        done,
        campaign_plan_queue(bundles, done, allow, campaign, no_judge_ok),
        collections.Counter(w for b in bundles if (w := campaign_block_reason(b, allow, campaign, no_judge_ok))),
    )


def complete(keys: list[tuple[str, str]], states: dict[tuple[str, str], str], superseded: set[tuple[str, str]]) -> bool:
    """Campaign complete = every manifest bundle (hand, id) done or superseded (a set match: done ∪ superseded = keys).
    A state outside the manifests is an error; a superseded bundle that was launched (a state other than queued)
    raises, because its result would vanish from the campaign's accounting."""
    extra = sorted(set(states) - set(keys))
    if extra:
        raise ValueError(f"states outside the manifests {len(extra)}: {extra[:3]}")
    if stray := sorted(superseded - set(keys)):
        raise ValueError(f"superseded bundles outside the manifests {len(stray)}: {stray[:3]}")
    ran = sorted(k for k in superseded if states.get(k, "queued") != "queued")
    if ran:
        raise ValueError(
            f"{len(ran)} superseded bundles have a launch record — a result already run is never erased: {ran[:3]}"
        )
    return all(states.get(k) == "done" for k in keys if k not in superseded)
