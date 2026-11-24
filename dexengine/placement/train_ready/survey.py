"""The survey loop: per ledger axis, every adopted clip through the 1st-pass rule, T4, T5-base, T5, the static desk
check, T5-desk and the asset check, then every judge result through T5-judge. One loop fills the per-dataset rows,
the re-judge list (STALE) and the per-clip verdicts (CLIPS) — no second calculator.
"""

from __future__ import annotations

import sys

from dexengine.data import rules as RR
from dexengine.placement import final_ledger as FL
from dexengine.placement.train_ready import assets as A
from dexengine.placement.train_ready import judge_fp as JF
from dexengine.placement.train_ready import trees as TR

#: T5 prefetch processes (main's --t5-workers; None = capacity.plan). 1 = no prefetch, one tree at a time in the loop
#: (same result).
T5_WORKERS: int | None = 1

STALE: list[tuple[str, str, str, str]] = []  # T4·T5·T5-judge list (--stale-out): (ds, ledger axis, clip, reason)
#: Per-clip verdicts (`--json-clips`), written in the same loop that counts. verdict ∈ {covered, missing, stale_roles,
#: stale_base, base_unevaluable, stale_setup, setup_unevaluable, stale_static_desk, desk_lite, stale_desk, no_tree,
#: stale_excluded}. The dashboard's "adopted ∖ trainable" clip list reads it.
CLIPS: list[dict] = []


def excluded_now(datasets) -> tuple[set, set]:
    """(dataset, clip) that the current 1st-pass rule (data.candidates) drops on this hand (DEXENGINE_HAND), and the
    variant axes' targets (route). The owner is called on this hand's base — filtering with another hand's reasons
    is wrong for hand-dependent rules (retarget_error …). `datasets` = only those with adoptions in the ledgers.
    The legitimate set differs per axis: on the canonical (desk) axis every 1st-pass reason excludes; on a variant
    (aerial) axis a `route:` reason marks that axis' target. The check is not skipped on variant axes either."""
    from dexengine.data import candidates as TC  # noqa: PLC0415 — the rule and route-prefix owner

    excl, route = set(), set()
    for key, why in A.first_pass_reasons(datasets, RR.hand_dir()).items():
        (route if TC.is_route(why) else excl).add(key)
    return excl, route


def scope_files(scope: tuple[str, ...] | None) -> list:
    """The ledger files this run measures — every file of the hand, or only those of the `scope` datasets — and the
    marker `[TRAIN_READY]` (stderr: stdout may be the one JSON document of --json). Nothing is dropped unsaid: the
    marker names every out-of-scope dataset with the ADOPTED rows left unmeasured, and a scope dataset that has no
    ledger file at all."""
    every = sorted(RR.ledger_files())
    ds_of = {f: RR.ledger_axis(f).split("-")[0] for f in every}
    files = [f for f in every if scope is None or ds_of[f] in scope]
    skipped: dict[str, int] = {}
    for f in every:
        if f not in files:
            skipped[ds_of[f]] = skipped.get(ds_of[f], 0) + len(A.adopted(f))
    absent = [d for d in (scope or ()) if d not in {ds_of[f] for f in files}]
    print(
        f"[TRAIN_READY] scope={','.join(scope) if scope is not None else 'all'}"
        f" checked={sum(len(A.adopted(f)) for f in files)}"
        f" skipped_out_of_scope={','.join(f'{d}:{n}' for d, n in sorted(skipped.items())) or '-'}"
        + (f" no_ledger={','.join(absent)}" if absent else ""),
        file=sys.stderr,
        flush=True,
    )
    return files


def _axis_of(f: str) -> tuple[str, str, bool]:
    """(ledger axis, dataset, variant) of a ledger file. D20/2c variant axes come before the env wiring (disc · root
    z), so they are left out of the trainable total but shown in the table."""
    key = RR.ledger_axis(f)
    return key, key.split("-")[0], key.endswith(("-aerial", "-artic"))


def _prefetch(files: list, excluded: set, route: set) -> None:
    """Fill the T5 and T5-judge tables in T5_WORKERS processes before the loop (decides nothing)."""
    jobs, jobs_j = [], []
    for f in files:
        key, ds, variant = _axis_of(f)
        jobs += [(ds, c, key) for c in A.adopted(f) if not A.first_pass_out(ds, c, variant, excluded, route)]
    TR.prefetch(jobs, T5_WORKERS)
    for f in files:  # T5-judge: the judge's non-adoptions + every adoption (tree or not)
        key, ds, variant = _axis_of(f)
        emb, scene = JF.axis(key)
        jobs_j += [
            (ds, emb, RR.active_hand(), c, scene)
            for c, stt, _j in A.ledger_rows(f)
            if A.t5j_scope(stt) and not A.first_pass_out(ds, c, variant, excluded, route)
        ]
    JF.prefetch(jobs_j, T5_WORKERS)


def _bump(rec: dict, k: str) -> None:
    rec[k] = rec.get(k, 0) + 1


def _desk_verdict(rec: dict, ds: str, key: str, clip: str, clip_row) -> bool:
    """The desk checks of one adopted clip (first verdict wins) → True when one stopped it: a render-added static
    object hangs off the desk (re-judge it) · S-desk-lite, the recorded hand goes under the scene desk top (gated)."""
    if why := TR.stale_static_desk(ds, clip, key):
        _bump(rec, "stale_static_desk")
        STALE.append((ds, key, clip, why))
        clip_row("stale_static_desk", why=why)
        return True
    if why := TR.desk_lite(ds, clip, key):
        _bump(rec, "desk_lite")
        clip_row("desk_lite", why=why)
        return True
    return False


def _adopted_clip(rec: dict, ds: str, key: str, clip: str, variant: bool, stored: str | None, j_emb: str) -> None:
    """One adopted clip through T4 → T5-base → T5 → static desk → desk lite → T5-desk → assets (first verdict wins)."""

    def clip_row(verdict: str, **kw) -> None:
        CLIPS.append(dict(dataset=ds, axis=key, clip=clip, variant=variant, verdict=verdict, missing=[], **kw))

    rec["adopted"] += 1
    objs = TR.tree_objects(ds, clip, key)
    if objs is None:
        rec["no_tree"] += 1
        clip_row("no_tree")
        return
    if why := TR.stale_roles(ds, clip, objs, key):  # T4 — a tree baked with old roles: re-judge it
        _bump(rec, "stale_roles")
        STALE.append((ds, key, clip, why))  # --stale-out uses this list as is; the queue resolves the axis name
        clip_row("stale_roles")
        return
    if b5 := JF.stale_base(ds, j_emb, clip, stored):  # T5-base — judged on an older base
        kind, why = b5
        _bump(rec, "stale_base" if kind == "stale" else "base_unevaluable")
        if kind == "stale":
            STALE.append((ds, key, clip, why))
        clip_row("stale_base" if kind == "stale" else "base_unevaluable", why=why)
        return
    if t5 := TR.stale_setup(ds, clip, key):  # T5 — a tree stamped with an old setup statement (re-judge it)
        kind, why = t5
        if kind == "stale":
            _bump(rec, "stale_setup")
            STALE.append((ds, key, clip, why))
            clip_row("stale_setup", why=why)
        else:
            # The current setup cannot be evaluated (e.g. a base statement the rule rejects): not trainable as
            # "current", not queued either (re-judging the same broken base fails again) — counted, loud.
            _bump(rec, "setup_unevaluable")
            clip_row("setup_unevaluable", why=why)
        return
    if _desk_verdict(rec, ds, key, clip, clip_row):
        return
    if d5 := JF.stale_desk(ds, j_emb, clip, key):  # T5-desk — the tree's desk fields ≠ the current base's
        kind, why = d5
        v = "stale_desk" if kind == "stale" else "desk_unmeasured"
        _bump(rec, v)
        if kind == "stale":  # unmeasured is only counted — the trainable verdict stays (decided by hand)
            STALE.append((ds, key, clip, why))
            clip_row(v, why=why)
            return
    kinds: set[str] = set()  # missing asset kinds — empty means the clip is covered
    for o in objs:
        if not o.get("obj_id"):
            continue
        aid, w = A.object_asset(ds, o)
        w += A.missing_assets(
            ds, aid, articulated=bool(o.get("articulation")), aerial=(TR.tree_scene(ds, clip, key) == "aerial")
        )
        if w:
            rec["missing"].setdefault(aid, set()).update(w)
            kinds.update(w)
    rec["covered"] += not kinds
    CLIPS.append(
        {
            "dataset": ds,
            "axis": key,
            "clip": clip,
            "variant": variant,
            "verdict": "missing" if kinds else "covered",
            "missing": sorted(kinds),
        }
    )


def _judge_rows(rec: dict, f: str, ds: str, key: str, variant: bool, excluded: set, route: set, carried: set) -> None:
    """T5-judge over every judge result of one ledger (non-adoptions · every adoption, tree or not). A clip the
    adoption loop already listed is not listed again (one clip, one line)."""
    j_emb, j_scene = JF.axis(key)
    listed = {(s[1], s[2]) for s in STALE}
    st = JF.T5J_STATS
    for clip, status, stored in A.ledger_rows(f):
        if not A.t5j_scope(status):
            st[f"oos:{status}"] = st.get(f"oos:{status}", 0) + 1
            continue
        if A.first_pass_out(ds, clip, variant, excluded, route):
            st["first_pass_out"] = st.get("first_pass_out", 0) + 1
            continue  # the current rule drops the clip — not a re-judge target (the same rule as the adoption loop)
        if (key, clip) in listed:
            st["already_listed"] = st.get("already_listed", 0) + 1
            continue
        if j5 := JF.stale_judge(ds, j_emb, j_scene, clip, stored, status, carry=(key, clip) in carried):
            kind, why = j5
            if kind == "stale":
                _bump(rec, "judge_unknown" if why == JF.JUDGE_UNKNOWN else "stale_judge")
                STALE.append((ds, key, clip, why))  # one list with T4·T5 — one detector, a reason column
            else:
                _bump(rec, "judge_unevaluable")


def survey(L, scope: tuple[str, ...] | None = None) -> list[dict]:
    """Per dataset: {adopted, covered, no_tree, missing, status, ready, promotable, …}. The ledger file name owns the
    universe. `scope` = the datasets the caller measures (None = every ledger file) — the 1st-pass rules, the
    prefetch and the loop all read `files`, so an out-of-scope dataset's inputs are not opened at all."""
    out: dict[str, dict] = {}
    files = scope_files(scope)
    excluded, route = excluded_now({RR.ledger_axis(f).split("-")[0] for f in files})
    if T5_WORKERS > 1:
        _prefetch(files, excluded, route)
    for f in files:
        key, ds, variant = _axis_of(f)
        # hand-carried rows = the ledger row's note says so; a row this hand judged again has its own note
        carried = {(key, r["clip"]) for r in FL.read_final_rows(f) if A.is_carry_note(r.get("note"))}
        name = key if variant else ds
        rec = out.setdefault(
            name,
            {"dataset": name, "sets": [], "adopted": 0, "covered": 0, "no_tree": 0, "missing": {}, "variant": variant},
        )
        rec["sets"].append(key)
        j_emb, _scene = JF.axis(key)
        stored_of = {c: j for c, stt, j in A.ledger_rows(f) if stt == "ADOPTED"}  # T5-base: adopted fingerprints
        for clip in A.adopted(f):
            if A.first_pass_out(ds, clip, variant, excluded, route):
                _bump(rec, "stale_excluded")
                CLIPS.append(
                    {
                        "dataset": ds,
                        "axis": key,
                        "clip": clip,
                        "variant": variant,
                        "verdict": "stale_excluded",
                        "missing": [],
                    }
                )
                continue
            _adopted_clip(rec, ds, key, clip, variant, stored_of.get(clip), j_emb)
        _judge_rows(rec, f, ds, key, variant, excluded, route, carried)
    rows = []
    for name, rec in sorted(out.items()):
        spec = L.DATASETS.get(name.split("-")[0])
        rec["missing"] = {k: sorted(v) for k, v in sorted(rec["missing"].items())}
        rec["status"] = spec.status if spec else "?"
        # Trainable = registry canon (T2) ∧ every adopted clip has its assets (T1). no_tree (this ledger axis has no
        # tree) is not excluded from the promotion verdict but shown. Variant axes are gated per clip from the tree
        # (`env_aerial` / `env_artic` in `missing`, owners RR.AERIAL_ENV_WIRED / ARTIC_ENV_WIRED).
        rec["ready"] = rec["status"] == "canon" and rec["covered"] == rec["adopted"]
        rec["promotable"] = rec["covered"] == rec["adopted"] and rec["adopted"] > 0 and rec["no_tree"] == 0
        rows.append(rec)
    return rows
