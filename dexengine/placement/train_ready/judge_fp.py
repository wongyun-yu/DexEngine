"""The judge-input fingerprint checks: T5-base (an adoption judged on an older base), T5-desk (the tree's desk fields
against the current base) and T5-judge (every judge result's input fingerprint against the current base and rules).

Why T5-judge: T4 and T5 see baked trees only. Clips the judge dropped (ledger_states.JUDGED_STATUSES) and adoptions
without a tree have nothing to compare, so without it every rule change would mean judging everything again. The
judge leaves an input fingerprint per clip (ledger column `judge_input`, owner judge/input_record.py); here the same
function runs on the current base and rules. Different → the same `--stale-out` list (`T5-judge:<parts>`); no
fingerprint → `T5-judge:unknown` (no silent pass — clips judged before fingerprints existed are judged once more). A
part that cannot be computed now is counted as unevaluable and not queued (a re-judge of the same input hits the same
wall — the same rule as T5 setup_unevaluable).

Why T5-desk: a desk rule change rebuilds the base but not its qpos, so T5-base does not see it, and T5 re-measures
from the tree's own old desk stamp. The fields = the desk fields the judge reads (JFP.DESK_READ); the current value =
the base tree the judge reads (JFP.base_tree). A field the current base has and the tree lacks is counted as
unmeasured and not queued (how to handle it is decided by hand).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

from dexengine.data import rules as RR
from dexengine.placement.judge import input_record as JFP
from dexengine.placement.train_ready import assets as A
from dexengine.placement.train_ready import trees as TR

#: --stale-out reasons of a row without a fingerprint (the re-judge list's 4th column; cli and survey count them).
JUDGE_UNKNOWN = "T5-judge:unknown"
BASE_UNKNOWN = "T5-base:unknown"

T5J_STATS: dict = {}
T5B_STATS: dict = {}
T5D_STATS: dict = {}
_T5J_UNEVAL: dict = {}  # unevaluable reason (part:value) → count; the marker prints the top few


def _base_part(ds: str, emb: str, clip: str) -> str:
    """Cached JFP.base_part — key = the base npz (mtime, size). Rebuilding the base misses."""
    tree = JFP.base_tree(ds, emb, RR.active_hand(), clip)
    try:
        s = os.stat(tree / "0/trajectory_kinematic.npz")
    except FileNotFoundError:
        return ""
    slot, ck = f"base:{tree}", f"{s.st_mtime_ns}:{s.st_size}"
    table = TR.cache_load()
    if (table.get(slot) or {}).get("key") == ck:
        T5B_STATS["cache_hit"] = T5B_STATS.get("cache_hit", 0) + 1
        return table[slot]["base"]
    cur = JFP.base_part(ds, emb, RR.active_hand(), clip)
    if not cur.startswith(JFP.ERR):
        table[slot] = {"key": ck, "base": cur}
        TR.T5_CACHE["dirty"] = True
    return cur


def stale_base(ds: str, emb: str, clip: str, stored: str | None) -> tuple[str, str] | None:
    """T5-base for one row (adopted + tree) → None (same · no current base) · ("stale", reason) · ("unevaluable", …)."""
    st = T5B_STATS
    cur = _base_part(ds, emb, clip)
    if not cur:
        st["no_base"] = st.get("no_base", 0) + 1
        return None
    st["compared"] = st.get("compared", 0) + 1
    if cur.startswith(JFP.ERR):
        st["unevaluable"] = st.get("unevaluable", 0) + 1
        return ("unevaluable", f"T5-base not evaluable:{cur}")
    p = JFP.parse(stored)
    if p is None:
        st["unknown"] = st.get("unknown", 0) + 1
        return ("stale", BASE_UNKNOWN)
    if p["base"] == cur:
        st["same"] = st.get("same", 0) + 1
        return None
    st["stale"] = st.get("stale", 0) + 1
    return ("stale", "T5-base")


def t5b_marker() -> None:
    """Applied marker: T5-base really ran."""
    s = T5B_STATS
    print(
        f"[T5-base] adopted+tree base fingerprints compared {s.get('compared', 0)} · same {s.get('same', 0)}"
        f" · stale {s.get('stale', 0)}"
        f" · unknown {s.get('unknown', 0)} · not evaluable {s.get('unevaluable', 0)}"
        f" · no current base {s.get('no_base', 0)}"
        f" · cache hits {s.get('cache_hit', 0)}",
        file=sys.stderr,
        flush=True,
    )


def stale_desk(ds: str, emb: str, clip: str, key: str) -> tuple[str, str] | None:
    """T5-desk for one row (adopted + tree) → None (same · no current base) · ("stale", "T5-desk:<fields>") ·
    ("unmeasured", "T5-desk unmeasured:<fields>"). One differing field makes it stale (its unmeasured fields are
    counted)."""
    st = T5D_STATS.setdefault(ds, {})
    bp = JFP.base_tree(ds, emb, RR.active_hand(), clip) / "task_info.json"
    if not bp.is_file():
        st["no_base"] = st.get("no_base", 0) + 1
        return None
    bsup = json.loads(bp.read_text()).get("support") or {}
    tsup = (TR.tree_ti(ds, clip, key) or {}).get("support") or {}
    st["compared"] = st.get("compared", 0) + 1
    gone = [k for k in JFP.DESK_READ if k in bsup and k not in tsup]
    diff = [k for k in JFP.DESK_READ if k not in gone and ((k in bsup) != (k in tsup) or bsup.get(k) != tsup.get(k))]
    for k in [f"diff:{k}" for k in diff] + [f"unmeasured:{k}" for k in gone]:
        st[k] = st.get(k, 0) + 1
    kind = "stale" if diff else "unmeasured" if gone else "same"
    st[kind] = st.get(kind, 0) + 1
    if diff:
        return ("stale", "T5-desk:" + ",".join(diff))
    return ("unmeasured", "T5-desk unmeasured:" + ",".join(gone)) if gone else None


def t5d_marker() -> None:
    """Applied marker: T5-desk really ran — one line per dataset (per-field differences · unmeasured)."""
    for ds, s in sorted(T5D_STATS.items()) or [("-", {})]:
        by = " ".join(f"{k}={v}" for k, v in sorted(s.items()) if ":" in k) or "-"
        print(
            f"[T5-desk] {ds} adopted+tree desk fields compared {s.get('compared', 0)} · same {s.get('same', 0)}"
            f" · stale {s.get('stale', 0)} · unmeasured {s.get('unmeasured', 0)}"
            f" · no current base {s.get('no_base', 0)}"
            f" · fields {by}",
            file=sys.stderr,
            flush=True,
        )


def _code_fp() -> str:
    if TR.T5_CACHE.get("jfp_code") is None:
        h = hashlib.sha1()
        for p in A.code_sources():
            h.update(pathlib.Path(p).read_bytes())
        TR.T5_CACHE["jfp_code"] = h.hexdigest()
    return TR.T5_CACHE["jfp_code"]


def _key(ds: str, emb: str, hand: str, clip: str, scene: str) -> tuple[str, str]:
    """(cache slot, input key) — slot = `jfp|<base tree>|<scene>`, key = code bytes + (mtime, size) of every data
    input (JFP.inputs — the files setup · sidecar fingerprint · base trajectory open). Any change re-measures."""
    fp = [TR.file_key(p, TR.coords().input_path(ds, p)) for p in JFP.inputs(ds, emb, hand, clip)]  # the opened file
    slot = f"jfp|{JFP.base_tree(ds, emb, hand, clip)}|{scene}"
    return slot, hashlib.sha1(json.dumps([_code_fp(), scene, fp]).encode()).hexdigest()


def compute(job: tuple) -> dict:
    """(ds, emb, hand, clip, scene) → clip parts. Top level (a worker process calls it); part failures are in-band."""
    ds, emb, hand, clip, scene = job
    return JFP.clip_parts(ds, emb, hand, clip, scene)


def _parts(ds: str, emb: str, hand: str, clip: str, scene: str) -> dict:
    slot, ck = _key(ds, emb, hand, clip, scene)
    table = TR.cache_load()
    hit = table.get(slot)
    if hit is not None and hit.get("key") == ck:
        if slot not in TR.T5_CACHE["fresh"]:
            T5J_STATS["cache_hit"] = T5J_STATS.get("cache_hit", 0) + 1
        return hit["parts"]
    parts = compute((ds, emb, hand, clip, scene))
    table[slot] = {"parts": parts, "key": ck}
    TR.T5_CACHE["dirty"] = True
    TR.T5_CACHE["fresh"].add(slot)
    return parts


def prefetch(jobs: list, workers: int | None) -> None:
    """Measure the cache-missing clip parts in a fork pool sized by capacity (`workers` = an explicit option; 1 = no
    prefetch) — no verdicts (the loop reads the table)."""
    if (workers is not None and workers <= 1) or not jobs:
        return
    t0 = time.time()
    table = TR.cache_load()
    todo = []
    for j in jobs:
        slot, ck = _key(*j)
        if (table.get(slot) or {}).get("key") != ck:
            todo.append((j, slot, ck))
    if todo:
        import multiprocessing as mp  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from dexengine.core import capacity  # noqa: PLC0415

        step = "dexengine.placement.train_ready.judge_fp"  # capacity.NEEDS key (this module)
        with (
            capacity.sized(step, len(todo), workers) as cap,
            ProcessPoolExecutor(
                cap.workers, mp_context=mp.get_context("fork"), initializer=capacity.worker_init, initargs=cap.initargs
            ) as ex,
        ):
            for (_j, slot, ck), parts in zip(todo, ex.map(compute, [t[0] for t in todo], chunksize=8), strict=True):
                table[slot] = {"parts": parts, "key": ck}
                TR.T5_CACHE["dirty"] = True
                TR.T5_CACHE["fresh"].add(slot)
    print(
        f"[T5-judge] precomputed {len(todo)}/{len(jobs)} clips · pool: the [CPU] line · {time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def axis(key: str) -> tuple[str, str]:
    """Ledger axis → (emb, scene); the registry reads the axis (ledger_axis_parts · f0_axis, the same as T5)."""
    ds, emb, _v = RR.ledger_axis_parts(key)
    return emb, RR.f0_axis(ds, RR.ledger_axis_variant(key))["scene"]


def stale_judge(
    ds: str, emb: str, scene: str, clip: str, stored: str, status: str, carry: bool = False
) -> tuple[str, str] | None:
    """T5-judge for one row → None (same) · ("stale", reason) · ("unevaluable", reason — the clip parts cannot be
    measured now). Rule hashes are compared only up to the stages the row's status passed (JFP.decided_stages); an old
    stamp's rs · rules difference is read by the ruleset-change predicate (JFP.diff_ruleset). When a run part (arm
    shape · rules · budget) cannot be measured, JFP.canon_conditions dies with the reason — counting it would empty the
    re-judge list silently, since every row hits the same wall."""
    st = T5J_STATS
    cur = JFP.encode(_parts(ds, emb, RR.active_hand(), clip, scene), JFP.canon_conditions(ds, scene, RR.active_arm()))
    st["compared"] = st.get("compared", 0) + 1
    verdict, comps = JFP.diff_ruleset(stored, cur, status, (ds, emb, RR.hand_dir(), clip))
    if carry and verdict == "stale":
        # A hand-carried row: the source hand judged the input, so the base part (the hand's qpos hash) always
        # differs — only that part leaves the comparison. Any other differing part is still stale (re-judge).
        comps = [k for k in comps if k.split(".")[0] != "base"]
        st["carry_base_only"] = st.get("carry_base_only", 0) + (not comps)
        verdict = "stale" if comps else "same"
    if verdict == "same":
        st["same"] = st.get("same", 0) + 1
        return None
    if verdict == "unevaluable":
        st["unevaluable"] = st.get("unevaluable", 0) + 1
        p = JFP.parse(cur) or {}
        for k in comps:
            _T5J_UNEVAL[f"{k}={p.get(k)}"] = _T5J_UNEVAL.get(f"{k}={p.get(k)}", 0) + 1
        return ("unevaluable", "T5-judge not evaluable:" + ",".join(f"{k}={p.get(k)}" for k in comps))
    if verdict == "unknown":
        st["unknown"] = st.get("unknown", 0) + 1
        return ("stale", JUDGE_UNKNOWN)
    st["stale"] = st.get("stale", 0) + 1
    for k in comps:
        st[f"diff:{k}"] = st.get(f"diff:{k}", 0) + 1
    return ("stale", "T5-judge:" + ",".join(comps))


def t5j_marker(t0: float) -> None:
    """Applied marker: T5-judge really ran — its own line, apart from the T5 line."""
    s = T5J_STATS
    diffs = " ".join(f"{k[5:]}={v}" for k, v in sorted(s.items()) if k.startswith("diff:"))
    oos = " ".join(f"{k[4:]}={v}" for k, v in sorted(s.items()) if k.startswith("oos:"))
    top = " · ".join(f"{k} {v}" for k, v in sorted(_T5J_UNEVAL.items(), key=lambda kv: -kv[1])[:4])
    carry = (
        " · hand carry-over rows differing in base only (same) " + str(s["carry_base_only"])
        if "carry_base_only" in s
        else ""
    )
    print(
        f"[T5-judge] judge inputs compared {s.get('compared', 0)} · same {s.get('same', 0)} · stale {s.get('stale', 0)}"
        f"({diffs or '-'}) · unknown {s.get('unknown', 0)} · not evaluable {s.get('unevaluable', 0)}"
        f"{' (' + top + ')' if top else ''} · first pass out {s.get('first_pass_out', 0)}"
        f" · already listed {s.get('already_listed', 0)} (clips the adoption loop above added — one line per clip)"
        f" · states out of scope {oos or '-'}"
        f" · cache hits {s.get('cache_hit', 0)}/{s.get('compared', 0)}"
        f"{carry}"
        f" · {time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )
