"""Single owner of the aerial-scene support gap rule (rule id = RULE_ID).

Gap = a run of frames in which the main object is neither touched by a hand (coords.frame_toucher) nor held up by its
own support disc (coords.compute_aerial_support). Runs are split by **position**: head (attached to the window start) ·
mid · tail (attached to the window end) · whole. First-round candidates (dexengine.data.candidates) drop aerial trees
with a head, mid or whole gap; tails are cut by D41 (segment.tail.tail_main). No dataset names: the rule is the same for
any aerial scene (rule card RULE_ID · scenes=aerial).

Measured once: stage 1 step 5b (this file's main, after the main-object sidecar 5) writes the marker `rules.<key>` per
base tree and first-round candidates read only that marker (articulated contact is a per-frame mesh distance, and
remeasuring at every candidate selection would take tens of minutes per dataset). A missing marker or one that disagrees
with the current tree (version · scene · main objects · reading) makes candidate selection fail loudly. D36 pieces do
not receive the marker (segment_split.PIECE_RULES_COPY) and measure their own window again.

Support = something the simulator builds holds the main object up (coords.stands_supported): standing on its own disc
(on_own_disc: bottom within Z_STAB of the disc top, bottom xy centre within the disc radius) or on a support object
(on_support_object, the same geometry as D42). The disc stands from reset to the end (D40 (b), as the simulator builds
it), so only 'standing on it' counts as support, not 'a disc exists'. Contact, disc, support object and Z_STAB belong to
coords; no thresholds are rebuilt here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GAP_KINDS = ("head", "mid", "tail", "whole")
#: Gap positions excluded from first-round candidates; tails are cut by D41 (remaining tails are counted by the D41
#: marker).
EXCLUDE_KINDS = ("whole", "head", "mid")
#: Position order of the summary line (main): excluded positions (in that order), then the rest (GAP_KINDS order).
#: Derived from the position sets.
SUMMARY_KINDS = EXCLUDE_KINDS + tuple(k for k in GAP_KINDS if k not in EXCLUDE_KINDS)
RULE_ID = RULE_KEY = "C640"
#: Marker version; the stage-1 5b column counts this version (a new version is remeasured). Bump it when this file or
#: the parts of the gap definition outside it change (coords contact · disc · support stands_supported · Z_STAB).
VERSION = "C640-2"  # 2: one reading (standing on the disc or on a support object; D40 (b) disc until the end · D42)
STEP = "dexengine.data.support.gap"  # capacity.NEEDS key (this module)
#: Marker missing or disagreeing with the current tree; returned by marker_kind (callers collect and fail loudly).
MISSING = "missing"


def main_names(ti: dict) -> list:
    """Objects whose support is measured = main objects (role=main, with qpos_start), by name (order of appearance, no
    duplicates). Also used by the D41 tail cut."""
    return list(
        dict.fromkeys(
            str(o["name"])
            for o in ti.get("objects") or []
            if o.get("role") == "main" and o.get("qpos_start") is not None
        )
    )


def runs(mask) -> list:
    """Runs [(i, j)] of true frames (both ends inclusive)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist(), strict=True))


def kind_of(i: int, j: int, T: int) -> str:
    """Position of run (i, j): whether it touches the start or end of the window [0, T)."""
    if i == 0 and j == T - 1:
        return "whole"
    return "head" if i == 0 else "tail" if j == T - 1 else "mid"


def tree_gaps(tree_dir, ti: dict, q) -> dict:
    """Gaps over the whole tree window: {"T": T, "gaps": [(object, position, f0, f1)]}. Frames = this tree's frames
    (base
    tree = base frames, contact is the identity map). Without main objects an empty list (first-round candidates drop
    those earlier). Without sidecar or disc coords fails; this never reads as 'supported' instead."""
    from dexengine.placement.coords import (
        core as C,  # noqa: PLC0415 — owner of contact, disc and support (stands_supported) (placement layer)
    )

    T = len(q)
    out = {"T": T, "gaps": []}
    names = main_names(ti)
    if not names:
        return out
    discs = C.compute_aerial_support(tree_dir, ti=ti, q=q)["discs"]
    placed, _ = C._placers(tree_dir, ti, q)
    sv = C.support_object_verts(tree_dir, ti, q)
    objs = {str(o["name"]): o for o in ti.get("objects") or [] if o.get("qpos_start") is not None}
    for name in names:
        touch = C.frame_toucher(tree_dir, ti, q, name, lambda b: b)
        mine = [d for d in discs if str(d.get("obj")) == name]
        free = np.zeros(T, bool)
        carry = bool(mine) or bool(sv)
        for f in range(T):  # contact is measured only on frames not on a support (articulated contact is expensive)
            free[f] = not (carry and C.stands_supported(placed(objs[name], f), mine, sv)) and not touch(f)
        out["gaps"] += [(name, kind_of(i, j, T), i, j) for i, j in runs(free)]
    return out


def exclude_kind(gaps: dict) -> str | None:
    """First-round excluded position: the first gap position in EXCLUDE_KINDS (whole > head > mid), else None."""
    kinds = {k for _, k, _, _ in gaps["gaps"]}
    return next((k for k in EXCLUDE_KINDS if k in kinds), None)


def reason(kind: str) -> str:
    """First-round exclusion reason string (read by the ledger and dashboards)."""
    if kind not in EXCLUDE_KINDS:
        raise ValueError(f"[C640] excluded position {kind!r} — only {EXCLUDE_KINDS}")
    return f"support_gap({RULE_ID}):{kind}"


def marker(drop: dict) -> str:
    """First-round candidate run marker: rule version and exclusions per position (from dexengine.data.candidates'
    reason counter drop)."""
    return f"[C640] version={VERSION} " + " ".join(f"{k}={drop.get(reason(k), 0)}" for k in EXCLUDE_KINDS)


def summary(recs: list) -> str:
    """Position fields `position=tree count` of the 5b summary line (SUMMARY_KINDS order); one position per tree.
    Excluded positions come straight from the marker's exclude (written by record, owned by exclude_kind) and are not
    re-derived from gaps. Only trees without an exclusion have their remaining position (tail) counted from the gaps,
    i.e. trees with only a tail gap. Trees without gaps and desk trees (no gaps) fall into no field. Unknown gap
    positions or an exclude outside the excluded positions fail."""
    n = dict.fromkeys(SUMMARY_KINDS, 0)
    for r in recs:
        kinds = {g[1] for g in r.get("gaps") or []}
        if kinds - set(SUMMARY_KINDS):
            raise ValueError(
                f"[{RULE_ID}] unknown gap position {sorted(kinds - set(SUMMARY_KINDS))} — only {SUMMARY_KINDS}"
            )
        if r.get("exclude") not in (None, *EXCLUDE_KINDS):
            raise ValueError(f"[{RULE_ID}] marker exclude {r.get('exclude')!r} — only None · {EXCLUDE_KINDS}")
        k = r.get("exclude") or next((k for k in SUMMARY_KINDS[len(EXCLUDE_KINDS) :] if k in kinds), None)
        if k:
            n[k] += 1
    return " ".join(f"{k}={v}" for k, v in n.items())


def record(tree_dir, ti: dict, q, scene: str) -> dict:
    """Marker `rules.<key>` of one tree: version · scene · main objects · excluded position (+ gaps when aerial). Not
    aerial or no main objects: no gaps to measure (exclude None, q is not read)."""
    rec = {"version": VERSION, "scene": scene, "mains": main_names(ti), "exclude": None}
    if scene == "aerial" and rec["mains"]:
        g = tree_gaps(tree_dir, ti, q)
        rec["gaps"] = [[n, k, i, j] for n, k, i, j in g["gaps"]]
        rec["exclude"] = exclude_kind(g)
    return rec


def marker_kind(ti: dict, scene: str) -> str | None:
    """First-round candidate decision: the marker's excluded position (None = nothing to drop). MISSING when the marker
    is absent or disagrees with the current tree (version · scene · main objects; reruns of cases 3 and roles 4 change
    scene and main objects)."""
    rec = (ti.get("rules") or {}).get(RULE_KEY)
    want = [VERSION, scene, main_names(ti)]
    if not isinstance(rec, dict) or [rec.get(k) for k in ("version", "scene", "mains")] != want:
        return MISSING
    return rec.get("exclude")


def _base_trees(ds: str, spec_dir: str, ls=None) -> list:
    """Tree folders of this hand's base layer (all embs); the 5b step and its column (freshness) count the same list.
    `ls(layer root)` = the children to look at (default: the folder's entries; a caller may pass a bounded
    listing)."""
    from dexengine.core import paths as P  # noqa: PLC0415 — owner of layer paths

    out = []
    for emb in P.data_layout.EMBODIMENTS:
        root = P.layer_path("base", ds, hand=spec_dir, emb=emb)
        kids = ls(root) if ls is not None else (sorted(root.iterdir()) if root.is_dir() else [])
        out += [d for d in kids if (d / "task_info.json").is_file()]
    return out


def tree_fresh(ds: str, spec_dir: str, tree: Path) -> bool:
    """Whether one tree's marker matches the current tree (version · scene · main objects); `freshness` (5b column) and
    piece todo (stage1.status.tree_todo) call this one check. Unreadable = stale."""
    from dexengine.data import candidate_rules as TC  # noqa: PLC0415 — owner of the scene

    try:
        ti = json.loads((Path(tree) / "task_info.json").read_text())
        return marker_kind(ti, TC.scene_of(ds, ti)) != MISSING
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def freshness(ds: str, spec_dir: str) -> tuple[int, int]:
    """(base trees whose marker matches the current tree, total); owner of the data.stages 5b column (per-tree check =
    tree_fresh). Any stale tree (pieces, roles or cases rerun) = empty column."""
    ts = _base_trees(ds, spec_dir)
    return sum(tree_fresh(ds, spec_dir, d) for d in ts), len(ts)


def _tree_job(args) -> tuple:
    """(tree name, marker | None, error | None, written). The scene's owner is candidate_rules.scene_of alone."""
    from dexengine.data import (
        candidate_rules as TC,  # noqa: PLC0415 — owner of the scene (same definition as first-round candidates)
    )
    from dexengine.data.tree_io import (
        write_json_if_changed,  # noqa: PLC0415 — base task_info writer (skips equal content)
    )

    tree_dir, ds, write = args
    p = Path(tree_dir) / "task_info.json"
    ti = json.loads(p.read_text())
    scene = TC.scene_of(ds, ti)
    try:
        q = (
            np.load(Path(tree_dir) / "0/trajectory_kinematic.npz")["qpos"]
            if scene == "aerial" and main_names(ti)
            else None
        )
        rec = record(tree_dir, ti, q, scene)
    except Exception as e:  # noqa: BLE001 — an unmeasurable tree gets no marker (candidate selection then fails on the missing marker)
        return Path(tree_dir).name, None, f"{type(e).__name__}: {e}"[:300], False
    if not write:
        return Path(tree_dir).name, rec, None, False
    ti.setdefault("rules", {})[RULE_KEY] = rec
    return Path(tree_dir).name, rec, None, write_json_if_changed(p, ti)


def main() -> int:
    """Stage 1 step 5b: write the marker to every base tree of this hand for one dataset. rc 0 · 2 (unmeasurable trees)
    · 3 (no trees)."""
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from dexengine.core import capacity  # noqa: PLC0415
    from dexengine.core import paths as P  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="aerial support gap marker rules.C640, stage 1 step 5b")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hand", required=True, help="hand id (no default)")
    ap.add_argument("--write", action="store_true", help="otherwise count only")
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    a = ap.parse_args()
    trees = [str(d) for d in _base_trees(a.dataset, P.hand_spec_dir(a.hand))]
    if not trees:
        print(f"[C640] ✗ {a.dataset} hand={a.hand}: 0 base trees", file=sys.stderr)
        return 3
    with (
        capacity.sized(STEP, len(trees), a.jobs) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        res = list(ex.map(_tree_job, [(t, a.dataset, a.write) for t in trees], chunksize=4))
    errs = [(n, e) for n, _r, e, _w in res if e]
    recs = [r for _n, r, _e, _w in res if r]
    print(
        f"[C640] {a.dataset} hand={a.hand} version={VERSION} trees {len(trees)} · aerial "
        f"{sum(r['scene'] == 'aerial' for r in recs)} · "
        + summary(recs)
        + f" · errors {len(errs)} · rewritten {sum(w for *_x, w in res)}"
        + ("" if a.write else " (count only)"),
        flush=True,
    )
    if errs:
        print(
            f"[C640] ✗ unmeasurable trees {len(errs)} — no marker written (candidate selection will fail). "
            f"e.g. {errs[0][0]}: {errs[0][1]}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.support.gap")
    raise SystemExit(main())
