"""The rule list (`RULES`): each rule's owner, scope (datasets, scenes) and thresholds read from the owning module."""

from __future__ import annotations

import os

from dexengine.core.rules.cards import RULES
from dexengine.core.rules.profiles import ALL, DATASET_PROFILE, Profile
from dexengine.core.rules.reader_calls import READER_CALL_SCOPE, reader_scope
from dexengine.core.rules.source import REPO, SELF_PATH, _sym
from dexengine.core.rules.spec import Rule  # noqa: F401 — re-exported

#: Scope read from a **dataset property**: `profile:<field>=<value>` = the datasets whose DATASET_PROFILE field has
#: that value.
PROFILE_SCOPE = "profile:"
#: **Scene-rule scope**: every dataset except those whose `Profile.scene_rules_off` names this rule's id (with a
#: reason). It may be empty (every dataset opted out), and empty means **no dataset** (the opposite of `datasets=()`,
#: which means all). Query such a rule with `covers` / `rule_on` only — `rule_datasets` refuses it, since its callers
#: read an empty tuple as 'all datasets'.
SCENE_SCOPE = "default:scene_rules"


def is_scene_rule(r) -> bool:
    """Is this a scene rule (scope `SCENE_SCOPE`: on by default, off per dataset only by an explicit opt-out)?"""
    return r.scope_from == SCENE_SCOPE


def _scene_scope(r) -> tuple:
    """The datasets a scene rule is on for (sorted, may be empty): all but those that opt out. Every opt-out entry of
    every dataset must name a scene rule and carry a reason — an unknown id or a blank reason raises (a typo must not
    leave a dataset silently ON, and an opt-out without a reason is a silent fallback)."""
    ids = {x.id for x in RULES if is_scene_rule(x)}
    off = {(d, s): w for d, p in DATASET_PROFILE.items() for s, w in dict(p.scene_rules_off).items()}
    bad = sorted(k for k in off if k[1] not in ids)
    blank = sorted(k for k, w in off.items() if not (isinstance(w, str) and w.strip()))
    if bad or blank:
        raise ValueError(
            f"DATASET_PROFILE scene_rules_off: unknown rules {bad} (known {sorted(ids)}) · without a reason {blank}"
        )
    return tuple(sorted(d for d in DATASET_PROFILE if (d, r.id) not in off))


def covers(r, dataset: str) -> bool:
    """The one predicate 'does rule r apply to this dataset': a scene rule to every dataset that does not opt out
    (empty = none), any other rule to its datasets (empty = all). Readers of `r.datasets` (applies, the judge input
    record)
    go through this. A rule with `scope_from` that `resolve()` has not filled yet is read through `scope_of` (its
    resolved scope is never empty), so an unresolved rule never reads as 'all datasets'."""
    if is_scene_rule(r):
        return dataset in scope_of(r)
    ds = r.datasets or (scope_of(r) if r.scope_from else ())
    return not ds or dataset in ds


def rule_on(rid: str, dataset: str) -> bool:
    """Is the rule `rid` on for this dataset (`covers`)? The query for scene rules. Unknown rule or dataset raises."""
    if dataset not in ALL:
        raise KeyError(f"dexengine.core.rules: unknown dataset {dataset!r} (known {list(ALL)})")
    for r in RULES:
        if r.id == rid:
            return covers(r, dataset)
    raise KeyError(f"dexengine.core.rules: unknown rule {rid!r}")


def _blank_reasons(path: str, name: str, v: dict) -> list:
    """Names in a scope dict that carry no reason. A reason is a non-empty string, a dataset profile (a statement
    for every dataset) or a non-literal expression (an alias into dexengine.placement.search_policy.DZ_CELLS — the value
    states that dataset). A literal None, number, bool or empty string is not a reason. For a dict in another file,
    whether a value is a literal is read from the source syntax (this tells a key-only dict's None apart from a literal
    None)."""

    def blank(x) -> bool:
        return not (isinstance(x, Profile) or (isinstance(x, str) and x.strip()))

    if path == SELF_PATH:
        return sorted(k for k, x in v.items() if blank(x))
    import ast

    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    # an annotated assignment (`X: dict[...] = {…}`) is the same declaration; _sym reads both, so read both here too
    node = next(
        n.value
        for n in tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and isinstance(n.value, ast.Dict)
        and any(isinstance(t, ast.Name) and t.id == name for t in getattr(n, "targets", [getattr(n, "target", None)]))
    )
    return sorted(
        k.value
        for k, x in zip(node.keys, node.values, strict=True)
        if isinstance(k, ast.Constant) and isinstance(x, ast.Constant) and blank(x.value)
    )


def _reader_call_scope(r) -> tuple:
    """The datasets whose reader calls the rule's owner function (`READER_CALL_SCOPE`, reader_calls). None, or a
    dataset outside the profiles, raises: an empty scope would read as 'all datasets'."""
    hit = reader_scope(r.owner)
    bad = sorted(set(hit) - set(ALL))
    if not hit or bad:
        raise ValueError(
            f"{r.id}: scope {r.scope_from!r} over owner {r.owner} — no reader calls it, or readers of datasets "
            f"without a profile call it {bad}"
        )
    return hit


def scope_of(r) -> tuple:
    """The single reader of a rule's scope (`scope_from`) → sorted dataset tuple. There are four sources: a
    dataset property (`profile:<field>=<value>`), a per-dataset statement with a reason (dict keys), the scene-rule
    scope (`SCENE_SCOPE` — every dataset that does not opt out; empty = none), or the reader-call scope
    (`READER_CALL_SCOPE` — the datasets whose reader calls the rule's owner function). **A tuple
    constant that just lists names raises** — it is a lookup table, not a rule: a new dataset could not enter by its
    property, only by having its name written in. A missing field or symbol also raises (never a silent
    'all datasets')."""
    src = r.scope_from
    if src == SCENE_SCOPE:  # the third source: a scene rule (may be empty = no dataset)
        return _scene_scope(r)
    if src == READER_CALL_SCOPE:
        return _reader_call_scope(r)
    if src.startswith(PROFILE_SCOPE):
        key, _, want = src[len(PROFILE_SCOPE) :].partition("=")
        if not want or key not in Profile.__dataclass_fields__:
            raise ValueError(f"{r.id}: scope {src!r} — Profile field {key!r} does not exist or the value is empty")
        hit = tuple(sorted(d for d, p in DATASET_PROFILE.items() if str(getattr(p, key)) == want))
        if not hit:  # a value typo (incl. `true` on a bool field) must not become "all datasets"
            raise ValueError(f"{r.id}: no dataset falls in scope {src!r} — value typo? (bool fields take True|False)")
        return hit
    path, name = src.split(":")
    if path == SELF_PATH:
        from dexengine.core import datasets as _self  # noqa: PLC0415 — the registry facade (imports this module)

        v, ok = getattr(_self, name, None), hasattr(_self, name)
    else:
        v, ok = _sym(src)
    if not ok or not isinstance(v, dict):
        raise ValueError(
            f"{r.id}: scope {src!r} is {'missing' if not ok else type(v).__name__} — only a dataset"
            f" property ({PROFILE_SCOPE}<field>=<value>) or a per-dataset reason dict is accepted"
            f" (no name tuples)"
        )
    bad = sorted(set(v) - set(ALL)) or (["(empty dict)"] if not v else [])
    blank = _blank_reasons(path, name, v)
    if bad or blank:  # unknown names · names without a reason (= a name tuple)
        raise ValueError(f"{r.id}: scope {src!r} — unknown datasets {bad} · datasets with an empty reason {blank}")
    return tuple(sorted(v))


def rule_datasets(rid: str) -> tuple:
    """Rule → datasets it applies to (empty tuple = all datasets), so consumers never re-list datasets.
    An unknown rule raises KeyError."""
    for r in RULES:
        if r.id == rid:
            if is_scene_rule(r):  # its empty tuple means none, the callers here read it as all
                raise ValueError(f"dexengine.core.rules: {rid} is a scene rule — query it with rule_on(rid, dataset)")
            return scope_of(r) if r.scope_from else tuple(r.datasets)
    raise KeyError(f"dexengine.core.rules: unknown rule {rid!r}")


def rule_scenes(rid: str) -> tuple:
    """Rule → scenes it applies to. The consumer (dexengine.data.candidates) calls this per first-stage rule.
    An unknown rule raises KeyError."""
    for r in RULES:
        if r.id == rid:
            return r.scenes
    raise KeyError(f"dexengine.core.rules: unknown rule {rid!r} — add it to RULES to declare the scenes it applies to")


def resolve() -> list:
    """Fill each rule's scope·thresholds by reading them from the owning module (the registry copies no values)."""
    for r in RULES:
        if r.scope_from:
            r.datasets = scope_of(r)
        r.resolved = {}
        for label, ref in r.thresholds:
            v, _ = _sym(ref)
            r.resolved[label] = v
    return RULES


def applies(rule_id: str, dataset: str) -> bool:
    for r in resolve():
        if r.id == rule_id:
            return covers(r, dataset)
    raise KeyError(f"unknown rule {rule_id}")
