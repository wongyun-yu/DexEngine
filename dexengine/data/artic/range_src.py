"""**Source records** of the articulation spec: the rule that measures OakInk2 joint ranges and canonical statements
once from hand-independent sources and fixes them.

Rule: the motion range of an articulated object is measured once from the hand-independent source recording and fixed.
  source record  tree statements measured from the part poses of base trees not merged yet (before 2b) (converter
                 per_tree_record · per_tree_rest; stage dexengine.data.artic.spec_collect._tree_arts). Object poses do
                 not depend on the hand (delivered objects = the same source · supplements = the shared INTAKE).
  name           <emb>/<tree>, without the hand. The same name must be the same record for every hand's tree
                 (one_record; a difference fails · no '@hand' copies). The articulation of an already merged tree is the
                 spec's consumer copy, not an input.
  home           the source record section SRC of the spec file (artic_ranges.json):
                 {"oakink2/<asset>": {"obj_id", "trees": {name: statement}}}.
  measure        only when the section is missing (first time) or with --remeasure (explicit): measure → derive
                 (statement · window).
  fixed value    published assets follow the keep rule keeps_published (keep when the published statement fits the
                 sources within the band about as well as a re-derived statement, KEEP_FIT_DEFICIT); kept → window =
                 published ∪ source (never shrinks), otherwise re-derived from the sources. The difference between
                 fixed value and source (mm) is printed per asset on the [ARTIC_SRC] line.
  verify         otherwise the records measured from the hand's trees are only compared with the fixed values
                 (verify_src; a different name or content fails · entries unchanged). Building more hands, changing
                 hand order or losing one hand's tree therefore never moves ranges or statements.
Marker [ARTIC_SRC] src=… mode=measure|verify (the physical range marker [ARTIC_RANGE] belongs to
dexengine.data.artic.limits).
"""

from __future__ import annotations

import json

from dexengine.data.artic import (
    ranges as AR,  # noqa: E402 — owner of the asset spec rule (asset_spec · statement · FK oracle)
)

SRC_MARK = "[ARTIC_SRC]"
SRC = "src"
#: Max rounds matching frozen-tree statements ↔ spec windows (pair-substitute records read a neighbouring asset's
#: window)
REST_ROUNDS = 4
_JOINT = ("type", "axis", "pos", "child_pose")


def rec_key(art: dict) -> str:
    """Content key of a tree record: statement + range (per-hand fields such as qpos addresses, fit residuals and canon
    records are left out)."""
    return json.dumps({**AR.statement(art), "range": art.get("range")}, sort_keys=True)


def one_record(table: dict, key: tuple, oid: str, name: str, art: dict, hand: str, what: str) -> None:
    """Add one record to table[key] = (object id, {name: art}); the name carries no hand. An existing record with the
    same
    name but different content (rec_key) fails (a 'source' that differs per hand is not a source)."""
    have = table.setdefault(key, (oid, {}))[1]
    if name in have and rec_key(have[name]) != rec_key(art):
        raise SystemExit(
            f"{SRC_MARK} ✗ {key[0]}/{key[1]} {name}: {what} record differs between hands ({hand} ≠ earlier hand) — "
            "object poses must not depend on the hand"
        )
    have.setdefault(name, art)


def joint_same(a: dict, b: dict) -> bool:
    """Whether two statements are the same joint: kind · axis · anchor · child_pose · part ids (fixed attachment poses
    are medoids and may differ)."""

    def ids(x):
        return {k: (x.get("parts") or {}).get(k) for k in ("base_id", "moving_id")}

    return all(a.get(k) == b.get(k) for k in _JOINT) and ids(a) == ids(b)


#: Keep rule for published statements (keep the old statement as the fixed value when it fits the current source records
#: within the D1-g thresholds): keep while the published statement fits at most this many fewer source trees within the
#: band than the statement re-derived from the sources (tree reproducing the most). The band (threshold) belongs to
#: oakink2.artic_fit.band (hinge D1-g · slide rail play), used through _relate.
KEEP_FIT_DEFICIT = 0
_RANGE_TOL = 1e-9  # numerical slack for the window-shrink check (rad · m)


def keeps_published(prev: dict, rl: list[tuple[str, dict]]) -> dict:
    """The one keep rule: {keep, n_prev (source trees the published statement fits within the band), n_new (the same for
    the re-derived statement), n, ref_mm · ref_deg (published canonical clip's source record ↔ published statement; None
    without that record)}."""
    n_prev = sum(AR._relate(prev, a)["fits"] for _c, a in rl)
    n_new = sum(r["fits"] for r in AR.canonical(rl)[2].values())
    ref = dict(rl).get(prev["ref_clip"])
    d = AR._relate(prev, ref) if ref is not None else None
    return {
        "keep": n_prev >= n_new - KEEP_FIT_DEFICIT,
        "n_prev": n_prev,
        "n_new": n_new,
        "n": len(rl),
        "ref_mm": None if d is None else d["pos_err_m"] * 1e3,
        "ref_deg": None if d is None else d["rot_err_deg"],
    }


def derive(published: dict, src: dict, skip) -> tuple[dict, dict]:
    """Source records → (spec entries, {asset: keep verdict}). A published asset kept by keeps_published: statement =
    the
    published statement (the 1e-6 canonical check compares against that fixed value, asset_spec fixed_canon) · window =
    published window ∪ sources (never shrinks; shrinking fails); otherwise re-derived from the sources (asset_spec: tree
    reproducing the most, window = sources). New assets are re-derived too. Skipped when skip(key, statement) gives a
    reason."""
    out, why = {}, {}
    for (ds, asset), (oid, recs) in src.items():
        k = f"{ds}/{asset}"
        rl = sorted(recs.items())
        if skip(k, rl[0][1]):
            continue
        prev = published.get(k)
        why[k] = keeps_published(prev, rl) if prev is not None else None
        if why[k] is None or not why[k]["keep"]:
            out[k] = AR.asset_spec(ds, oid, asset, rl)
            continue
        out[k] = AR.asset_spec(ds, oid, asset, rl, prev, fixed_canon=True)
        if out[k]["lo"] > prev["lo"] + _RANGE_TOL or out[k]["hi"] < prev["hi"] - _RANGE_TOL:
            raise SystemExit(
                f"{SRC_MARK} ✗ {k}: window of a kept asset shrank ({prev['lo']:.6f}, {prev['hi']:.6f}) → "
                f"({out[k]['lo']:.6f}, {out[k]['hi']:.6f}) — the fixed range is published ∪ source"
            )
    return out, why


def _why_line(k: str, w: dict | None, e: dict) -> str:
    """[ARTIC_SRC] line for one asset: keep verdict · fixed value vs source difference (mm · canonical clip · max over
    fitting trees)."""
    if w is None:
        return f"{SRC_MARK} {k}: new asset · max difference over fitting trees {e['pos_err_max_m'] * 1e3:.2f} mm"
    ref = "-" if w["ref_mm"] is None else f"{w['ref_mm']:.2f} mm · {w['ref_deg']:.2f}°"
    head = "kept (published statement)" if w["keep"] else "re-derived"
    return (
        f"{SRC_MARK} {k}: {head} · trees in band, published {w['n_prev']}/{w['n']} · "
        f"re-derived statement {w['n_new']}/{w['n']} · source difference of the published canonical clip {ref} "
        "· fixed value ↔ fitting trees max "
        f"{e['pos_err_max_m'] * 1e3:.2f} mm"
    )


def src_doc(src: dict) -> dict:
    """Source record section (file form): {"<ds>/<asset>": {"obj_id", "trees": {name: statement}}} (sorted)."""
    return {
        f"{ds}/{a}": {"obj_id": oid, "trees": dict(sorted(recs.items()))}
        for (ds, a), (oid, recs) in sorted(src.items())
    }


def src_table(doc: dict | None) -> dict:
    return {tuple(k.split("/", 1)): (v["obj_id"], dict(v["trees"])) for k, v in (doc or {}).items()}


def src_union(base: dict, new: dict) -> dict:
    """Merge source records: for the same (asset, name) the new one wins (remeasured), the rest come from both."""
    out = {k: (oid, dict(recs)) for k, (oid, recs) in base.items()}
    for k, (oid, recs) in new.items():
        out.setdefault(k, (oid, {}))[1].update(recs)
    return dict(sorted(out.items()))


def verify_src(stored: dict, now: dict) -> tuple[int, int]:
    """Source records measured now from the hand's trees ↔ the fixed source records. Every record must have the same
    name
    and content; a missing name (a tree the spec does not know) or different content (the source changed) fails
    (remeasuring only via --remeasure). Returns (records compared, fixed records)."""
    bad, n = [], 0
    for key, (_oid, recs) in now.items():
        have = stored.get(key, (None, {}))[1]
        for name, art in recs.items():
            n += 1
            if name not in have:
                bad.append(f"{key[0]}/{key[1]} {name}: source record unknown to the spec")
            elif rec_key(have[name]) != rec_key(art):
                bad.append(f"{key[0]}/{key[1]} {name}: source record differs from the fixed value")
    if bad:
        raise SystemExit(
            f"{SRC_MARK} ✗ source record check failed {len(bad)}/{n}: {bad[:3]} — if the sources changed, "
            "remeasure with dexengine/data/artic/spec_stage.py --remeasure and review the impact (trees to rebuild)"
        )
    return n, sum(len(v[1]) for v in stored.values())


def measure(published: dict, stored: dict, collect, skip) -> tuple[dict, dict, tuple]:
    """Measure the source records: statements of moved trees → derive → statements of frozen trees (with that spec) →
    derive again, until the frozen-tree statements stay the same (REST_ROUNDS). collect = the stage's collect_all.
    Returns (OakInk2 entries, source records, (stats, per, fixed, keep verdicts))."""
    stats, per, fixed, mov = collect({})
    src = src_union(stored, mov)
    oak, why = derive(published, src, skip)
    rest_prev = None
    for _ in range(REST_ROUNDS):
        _st, _per, fx_rest, rest = collect({**published, **oak}, rest_only=True)
        if (
            rest == rest_prev
        ):  # rigid-joint records of static articulated objects are measured only once the spec exists (this round)
            return (
                oak,
                src,
                (
                    stats,
                    per,
                    sorted(
                        {(n, r["root"]): (n, r) for n, r in [*fixed, *fx_rest]}.values(),
                        key=lambda x: (x[0], x[1]["root"]),
                    ),
                    why,
                ),
            )
        rest_prev = rest
        src = src_union(src_union(stored, mov), rest)
        oak, why = derive(
            published, src, skip
        )  # keep verdicts always refer to the published entries (not the spec update)
    raise SystemExit(f"{SRC_MARK} ✗ frozen-tree statements did not settle within {REST_ROUNDS} rounds")


def oak_entries(doc: dict, remeasure: bool, collect, skip) -> dict:
    """Spec document → OakInk2 entries · source records · collection results · mode (verify | measure); prints the
    marker lines (called by update)."""
    published = doc.get("objects") or {}
    stored = src_table(doc.get(SRC))
    mode = "measure" if remeasure or not stored else "verify"
    if mode == "verify":
        stats, per, fixed, now = collect(published)
        n, n_fix = verify_src(stored, now)
        src, oak = stored, {k: v for k, v in published.items() if k.startswith("oakink2/")}
        note = f"verified {n} · fixed records {n_fix}"
    else:
        oak, src, (stats, per, fixed, why) = measure(published, stored if remeasure else {}, collect, skip)
        for k in sorted(why):
            print(_why_line(k, why[k], oak[k]), flush=True)
        kept = sorted(k for k in published if k.startswith("oakink2/") and k not in oak)
        oak.update(
            {k: published[k] for k in kept}
        )  # published assets without a source record stay as they are (windows never shrink)
        note = (
            f"measured records {sum(len(v[1]) for v in src.values())} · published assets without a source record "
            f"{len(kept)}"
        )
    print(
        f"{SRC_MARK} src=source records (part poses of base trees before 2b · hand-independent · one per "
        f"name) mode={mode} · {note} · "
        f"assets {len(src)}",
        flush=True,
    )
    changed = {}
    for k in sorted(set(oak) | {k for k in published if k.startswith("oakink2/")}):
        old, new = published.get(k), oak.get(k)
        if old is None and new is not None:
            changed[k] = "new"
        elif (
            new is not None and new != old
        ):  # statement and window equal, only other fields changed → not a bake target
            if joint_same(new, old) and (new["lo"], new["hi"]) == (old["lo"], old["hi"]):
                continue
            how = "statement kept" if joint_same(new, old) else "statement re-derived"
            changed[k] = f"remeasured({how} · ({old['lo']:.4f}, {old['hi']:.4f}) → ({new['lo']:.4f}, {new['hi']:.4f}))"
    return {"mode": mode, "oak": oak, "src": src, "stats": stats, "per": per, "fixed": fixed, "changed": changed}
