"""Tree lookups and the tree checks of one adopted clip: T4 (roles · active sides), the static-object desk check,
and T5 (the stamped setup statement against the current rule), with the T5 verdict cache.

Why T5: when an initial-setup rule (D26′ desk support · D20 aerial support · piece window · representative frame)
changes, T4 alone does not re-judge the tree, and a person would have to queue it by hand on every rule change. T5
recomputes the tree's stamped setup with the current rule and the current base on the same tree and compares. The
formula is not here — it calls coords.setup_stale (the same compute_* the bake stamps with).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pathlib
import sys
import time

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.placement.train_ready import assets as A

_TREE_SCENE: dict[tuple[str, str, str], str] = {}  # (ds, clip, ledger axis) -> tree scene, filled by tree_objects
_TREE_TI: dict[tuple[str, str, str], dict | None] = {}  # (ds, clip, ledger axis) -> task_info | None, read once

T5_STATS: dict = {}
T5_CACHE: dict = {"path": None, "data": None, "rule_fp": None, "dirty": False, "fresh": set()}


def coords():
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — heavy; loaded on first use

    return C


def t5_tree(ds: str, clip: str, key: str):
    """The baked tree of this ledger axis — combo from the axis variant (RR.ledger_axis_variant), emb from the axis
    (RR.ledger_axis_parts). The only tree locator here: T4 (tree_ti) and T5 both read it, because a hot3d clip has
    table + `_aerial` trees and an all-variant glob reads the other axis' tree. The path is the layer owner's
    (paths.layer_path "trees"). Returns (path | None, n_hits)."""
    _ds, emb, _v = RR.ledger_axis_parts(key)
    combo = RR.combo("", RR.ledger_axis_variant(key))
    p = pathlib.Path(P.layer_path("trees", ds, hand=RR.active_hand(), emb=emb, combo=combo)) / P.tree_name(clip)
    return (p, 1) if (p / "task_info.json").is_file() else (None, 0)


def tree_ti(ds: str, clip: str, key: str) -> dict | None:
    """The baked tree task_info of this ledger axis (None without a tree) — located by t5_tree alone. Globbing
    every canonical combo × emb and reading the first hit would, for a clip with desk and aerial trees (hot3d), read
    the other axis' tree, or an orphan tree that axis had dropped, and raise T4 on old roles."""
    if (ds, clip, key) not in _TREE_TI:
        tree, _nh = t5_tree(ds, clip, key)
        _TREE_TI[(ds, clip, key)] = json.loads((tree / "task_info.json").read_text()) if tree is not None else None
    return _TREE_TI[(ds, clip, key)]


def tree_objects(ds: str, clip: str, key: str) -> list[dict] | None:
    ti = tree_ti(ds, clip, key)
    if ti is None:
        return None
    _TREE_SCENE[(ds, clip, key)] = str(ti.get("scene") or "table")  # the tree's own scene (coords.scene_mode)
    return ti.get("objects") or []


def tree_scene(ds: str, clip: str, key: str) -> str | None:
    return _TREE_SCENE.get((ds, clip, key))


def base_task_info(ds: str, clip: str) -> tuple[dict, str] | None:
    """(task_info, tree folder) of the current base tree — the folder feeds the layout reader (qpos_layout)."""
    hits = glob.glob(os.path.join(str(P.main_root()), "data", ds, "base", RR.hand_dir(), "*", clip, "task_info.json"))
    if not hits:
        return None
    if len(hits) > 1:  # the clip is in two base embs: which one is current is unknown — no first-hit pick
        raise ValueError(f"{ds}/{clip}: {len(hits)} base trees — {sorted(hits)}")
    with open(hits[0]) as f:
        return json.load(f), os.path.dirname(hits[0])


def stale_static_desk(ds: str, clip: str, key: str) -> str:
    """Static-object desk: the reason when, on this axis' baked tree, a static object the render adds (base record
    gate=valid_static) hangs off the desk (static_desk.stale_reason owns the rule, thresholds and geometry), else ''.
    Floating is not a reason (a re-judge cannot fix it). The base record is read first — without valid_static the
    tree npz and surface samples are not opened (most of the thousands of adoptions)."""
    from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415 — put-down rule owner
    from dexengine.placement.scene import static_desk as SD  # noqa: PLC0415 — static-object desk rule owner

    tree, _nh = t5_tree(ds, clip, key)
    bt = base_task_info(ds, clip)
    if tree is None or bt is None:
        return ""
    if why := PR.stale_reason(tree, tree_ti(ds, clip, key)):  # the adopted placement breaks the put-down rule
        return why
    verdict, req = SD.required(bt[0])
    if verdict != "valid_static" or not req:
        return ""
    return SD.stale_reason(SD.assess(tree, ti=tree_ti(ds, clip, key), base_ti=bt[0]))


def stale_roles(ds: str, clip: str, tree_objs: list[dict] | None, key: str) -> str:
    """T4 — a tree whose roles differ from the current base layer was baked on an old decision (the reward target,
    contact derivation and spawn set are old); the adoption ledger does not know it. Returns "T4" (roles), "T4-b"
    (same roles, different active sides) or "" (current) — the `--stale-out` reason column."""
    if tree_objs is None or not any("role" in o for o in tree_objs):
        return ""  # no tree, or a layout without roles (taco = V1) — nothing to compare
    hit = base_task_info(ds, clip)
    if hit is None:  # no base layer on this axis (taco = V1) — no verdict
        return ""
    bti, bdir = hit
    if A.roles_of(bti) != A.roles_of({"objects": tree_objs}):
        return "T4"
    # T4-b: same roles, different active sides — the main object of a hand frozen idle leaves the reference and the
    # episode always terminates (0 % completion). The tree's active_sides is the judge's choice, so it is compared
    # with what the judge would pick on the current base (judged_active), not with the raw case.active.
    t_act = (tree_ti(ds, clip, key) or {}).get("active_sides")
    if t_act is None or (bti.get("case") or {}).get("active") is None:  # either side unstated → no comparison
        return ""
    return "T4-b" if sorted(t_act) != sorted(A.judged_active(bdir, bti)) else ""


def _rule_fp() -> str:
    """Bytes of the setup rule code (assets.rule_sources). The verdict is built in coords only; this package only
    picks the tree and base (their paths are in the key) and the scene (in the key)."""
    if T5_CACHE["rule_fp"] is None:
        h = hashlib.sha1()
        for p in A.rule_sources():
            h.update(pathlib.Path(p).read_bytes())
        T5_CACHE["rule_fp"] = h.hexdigest()
    return T5_CACHE["rule_fp"]


def file_key(name, path) -> tuple:
    """(declared name, mtime_ns, size) of one input file — (name, None) when it is missing (a cache-key entry)."""
    try:
        s = os.stat(path)
    except FileNotFoundError:
        return (str(name), None)
    return (str(name), s.st_mtime_ns, s.st_size)


def _input_key(tree, base_p, statement: dict, scene: str) -> str:
    """Cache key = rule code bytes + (mtime, size) of every data input the computation reads (coords.setup_inputs —
    tree task_info · npz · f0 marker · contact sidecars · MANO keypoints · meshes) + the base task_info + the scene."""
    ds = statement.get("dataset_name")
    # the opened file (object ref → dexengine.core.layout); key = declared name
    fp = [file_key(p, coords().input_path(ds, p)) for p in coords().setup_inputs(tree, statement) + [base_p]]
    return hashlib.sha1(json.dumps([_rule_fp(), scene, fp]).encode()).hexdigest()


def cache_load() -> dict:
    """The verdict table of this run — the file cache when `--t5-cache` gave one, else memory only."""
    if T5_CACHE["data"] is None:
        T5_CACHE["data"] = {}
        if T5_CACHE["path"] is not None:
            try:
                T5_CACHE["data"] = json.loads(pathlib.Path(T5_CACHE["path"]).read_text())
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as e:  # a broken cache is only a miss — say so, never trust it
                print(
                    f"[T5] cache read failed ({type(e).__name__}) → emptied, measured again: {T5_CACHE['path']}",
                    file=sys.stderr,
                )
    return T5_CACHE["data"]


def cache_save() -> None:
    if T5_CACHE["path"] is None or not T5_CACHE["dirty"]:
        return
    p = pathlib.Path(T5_CACHE["path"])
    try:
        if T5_CACHE.get("guard"):
            RR._hand_root_mod().assert_writable(p, RR.active_hand(), by="train_ready")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(T5_CACHE["data"]))
        os.replace(tmp, p)  # a reader never sees half a file
    except Exception as e:  # noqa: BLE001 — the cache is only speed; the result is already right
        print(
            f"[T5] cache save failed ({type(e).__name__}: {str(e)[:160]}) — the next run measures again",
            file=sys.stderr,
        )


def _prepare(ds: str, clip: str, key: str) -> dict:
    """What T5 compares for one adopted clip — {tree, statement, scene, ck} or {skip: why[, err]}.
    statement = coords.setup_statement(tree, current base); scene = the axis scene (RR.f0_axis, the owner the judge's
    f0 and the bake read); ck = the input-fingerprint key."""
    tree, nh = t5_tree(ds, clip, key)
    if tree is None:
        if nh == 0:
            return {"skip": "no_axis_tree"}
        return {"skip": "tree_ambiguous", "err": f"{nh} trees (tree_glob) — which one is current is unknown"}
    _ds, emb, _v = RR.ledger_axis_parts(key)
    base_p = pathlib.Path(P.main_root()) / "data" / ds / "base" / RR.hand_dir() / emb / clip / "task_info.json"
    if not base_p.is_file():
        return {"skip": "no_base"}  # an axis without a base (taco = V1): no current statement, no verdict (as T4)
    try:
        statement = coords().setup_statement(
            json.loads((tree / "task_info.json").read_text()), json.loads(base_p.read_text())
        )
    except ValueError as e:  # the base cannot state the current setup — counted as unevaluable
        return {"skip": "statement_error", "err": str(e)[:200]}
    scene = RR.f0_axis(ds, RR.ledger_axis_variant(key))["scene"]
    return {"tree": tree, "statement": statement, "scene": scene, "ck": _input_key(tree, base_p, statement, scene)}


def compute(job: tuple) -> dict:
    """(tree, statement, scene) → {diff, stamp, err}. Top level so a worker process can run it."""
    tree, statement, scene = job
    try:
        r = coords().setup_stale(tree, statement, scene)
        return {"diff": r["diff"], "stamp": r["stamp"], "err": None}
    except Exception as e:  # noqa: BLE001 — one tree's unevaluable verdict is counted as a reason
        return {"diff": [], "stamp": "", "err": f"{type(e).__name__}: {str(e)[:200]}"}


def _store(tree, ck: str, res: dict) -> None:
    cache_load()[str(tree)] = dict(res, key=ck)
    T5_CACHE["dirty"] = True
    T5_CACHE["fresh"].add(str(tree))  # measured in this run — not counted as a hit
    T5_STATS["computed"] = T5_STATS.get("computed", 0) + 1
    if T5_STATS["computed"] % 200 == 0:
        cache_save()  # save midway: a cut first run (tens of minutes) keeps what it measured


def prefetch(jobs: list, workers: int | None) -> None:
    """Compute the cache misses among `jobs` [(ds, clip, key)] in a fork pool sized by capacity (`workers` = an
    explicit option; 1 = no prefetch) before the survey loop. Decides nothing — it only fills the table the loop
    reads."""
    if workers is not None and workers <= 1:
        return
    t0 = time.time()
    table = cache_load()
    todo = []
    for ds, clip, key in jobs:
        pr = _prepare(ds, clip, key)
        if "tree" in pr and (table.get(str(pr["tree"])) or {}).get("key") != pr["ck"]:
            todo.append(pr)
    if todo:
        import multiprocessing as mp  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from dexengine.core import capacity  # noqa: PLC0415

        step = "dexengine.placement.train_ready.trees"  # capacity.NEEDS key (this module)
        with (
            capacity.sized(step, len(todo), workers) as cap,
            ProcessPoolExecutor(
                cap.workers, mp_context=mp.get_context("fork"), initializer=capacity.worker_init, initargs=cap.initargs
            ) as ex,
        ):
            args = [(pr["tree"], pr["statement"], pr["scene"]) for pr in todo]
            for pr, res in zip(todo, ex.map(compute, args, chunksize=4), strict=True):
                _store(pr["tree"], pr["ck"], res)
    print(
        f"[T5] precomputed {len(todo)}/{len(jobs)} trees · pool: the [CPU] line · {time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def stale_setup(ds: str, clip: str, key: str) -> tuple[str, str] | None:
    """T5 for one adopted clip → None (current / not judgeable here) · ("stale", reason) · ("error", message).
    The verdict is coords.setup_stale's; a verdict already in the table under the same input key is reused."""
    st = T5_STATS
    pr = _prepare(ds, clip, key)
    if "tree" not in pr:
        st[pr["skip"]] = st.get(pr["skip"], 0) + 1
        return ("error", pr["err"]) if pr.get("err") else None
    st["compared"] = st.get("compared", 0) + 1
    res = cache_load().get(str(pr["tree"]))
    if res is None or res.get("key") != pr["ck"]:
        res = compute((pr["tree"], pr["statement"], pr["scene"]))
        _store(pr["tree"], pr["ck"], res)
    elif str(pr["tree"]) not in T5_CACHE["fresh"]:
        st["cache_hit"] = st.get("cache_hit", 0) + 1
    if res.get("stamp") == "absent_old_bake":
        st["desk_stamp_absent"] = st.get("desk_stamp_absent", 0) + 1
    if res.get("err"):
        st["error"] = st.get("error", 0) + 1
        return ("error", res["err"])
    if res["diff"]:
        st["stale"] = st.get("stale", 0) + 1
        return ("stale", "T5:" + ";".join(res["diff"]))
    return None


def t5_marker(t0: float) -> None:
    """Applied marker: T5 really ran — one line with the cache path and hits."""
    s = T5_STATS
    print(
        f"[T5] setups compared {s.get('compared', 0)} · stale {s.get('stale', 0)} · not evaluable {s.get('error', 0)}"
        f" · old desk trees without the stamp (compared as 0 pedestals) {s.get('desk_stamp_absent', 0)}"
        f" · no axis tree {s.get('no_axis_tree', 0)} · no base {s.get('no_base', 0)}"
        f" · cache {T5_CACHE['path'] or 'off'} hits {s.get('cache_hit', 0)}/{s.get('compared', 0)}"
        f" · {time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )
