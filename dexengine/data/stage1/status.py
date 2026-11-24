"""Progress of every stage-1 step per dataset for the active hand, and the first empty step to run next.

``status`` prints the table (and fills ``STATUS_ROWS`` for ``--json``); ``tree_todo`` applies the same per-cell
judgement to single trees (which steps a tree still needs).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data import rules as RR
from dexengine.data.stage1 import plan as PL
from dexengine.data.stage1 import table as T

#: Rows of the last `status` call (the same values as the printed table) for `--json`.
STATUS_ROWS: dict[str, dict[str, int | str]] = {}
#: Reach map sides that exist but are not usable, with the reason and how to lock them (printed after the table).
POOL_NOTES: dict[str, str] = {}


def _pool_ready(root: Path, combo: str, side: str) -> bool:
    """Is one map side usable by the judge (meta + sidecar current)? A side that exists but is not gets a note with
    the lock form (reach_map.store LOCK_ENV) — used as is only when named there."""
    from dexengine.placement.reach_map import clearance as _CLR  # noqa: PLC0415
    from dexengine.placement.reach_map import store as _PM  # noqa: PLC0415

    d = root / f"{combo}_{side}"
    if not (d / "meta.npz").is_file():
        return False
    why = _CLR.sidecar_problem(d, RR.active_arm())
    if why:
        POOL_NOTES[d.name] = f"{why} — to use it as is: {_PM.LOCK_ENV}={_PM.pool_id(d)} (else rebuild)"
    return why is None


def _sidecar_cell(ds: str, spec_dir: str, layer: str) -> tuple[int, int]:
    """(clip, side) sidecars whose fingerprint matches this hand's base, expected (clip, side)."""
    from dexengine.data.contact import fingerprint as _SF  # noqa: PLC0415

    c = _SF.summarize(_SF.scan(P.main_root(), ds, spec_dir, (layer,)), layer)
    return c["fresh"], c["total"]


def _input_fp_cell(ds: str, spec_dir: str, owner: str) -> tuple[int, int]:
    """(base trees whose input fingerprint matches the current inputs, all base trees) from `owner.freshness`."""
    fn = getattr(importlib.import_module(owner), "freshness", None)
    if fn is None:
        raise InputMissingError(f"[hand_pipeline] {owner}.freshness is missing -- cannot count '{T._INPUT_FP}{owner}'")
    return fn(ds, spec_dir)


def _frame_current(ds: str, emb: str, clip: Path) -> bool:
    """Was one retarget output solved under the current MANO frame rule? Compares the ``source_frame`` of the retarget
    and mano task_info through ``source_frames.frame_key``; a missing or unreadable record is not current."""
    from dexengine.data.readers import source_frames as sf  # noqa: PLC0415

    try:
        mano = json.loads((P.layer_path("mano", ds, emb=emb) / clip.name / "task_info.json").read_text())
        mine = json.loads((clip / "task_info.json").read_text())
    except (OSError, ValueError, KeyError):
        return False
    return sf.frame_key(mine.get("source_frame")) == sf.frame_key(mano.get("source_frame"))


def _stage1_cover(ds: str, hand: str) -> tuple[int, int]:
    """(parent takes retargeted under the current MANO frame rule, parent takes the base layer needs). With no base
    layer yet the denominator is every retarget output, so a finished retarget does not read as the first gap."""
    need, have, made = set(), set(), set()
    for emb in T.embs(ds):
        for layer in ("base", "retarget"):
            try:
                d = P.layer_path(layer, ds, hand=hand, emb=emb)
            except (KeyError, ValueError):
                continue
            for c in d.glob("*"):
                if not c.is_dir():
                    continue
                if layer == "base":
                    need.add(P.parent_clip(c.name))
                else:
                    made.add(c.name)
                    if _frame_current(ds, emb, c):
                        have.add(c.name)
    if not need:
        return len(have), len(made)
    return len(need & have), len(need)


def _has_bytes(f: Path, needle: bytes) -> bool:
    """Does the file contain `needle` (unreadable = no)?"""
    try:
        return needle in f.read_bytes()
    except OSError:
        return False


def tree_todo(ds: str, spec_dir: str, prefixes, trees) -> dict:
    """{tree: [names of the steps among `prefixes` whose output is missing or stale for that tree]}, with the same
    judgement as the step's cell. Steps outside their rule scope are never to-do. A cell without a per-tree judgement
    or an unknown step id is fatal."""
    trees = [Path(t) for t in trees]
    rows = [g for g in T.STAGES if T.step_id(g) in set(prefixes)]
    lost = sorted(set(prefixes) - {T.step_id(g) for g in rows})
    if lost:
        raise SystemExit(f"[hand_pipeline] tree_todo: unknown steps {lost} (known: {[g[0] for g in T.STAGES]})")
    out: dict = {t: [] for t in trees}
    for name, pat, _slot, cmd in rows:
        if isinstance(cmd, T.Run) and not T.applies(cmd, ds):
            continue
        if pat.startswith(T._SIDECAR_FP):
            from dexengine.data.contact import fingerprint as _SF  # noqa: PLC0415

            scan = _SF.scan(P.main_root(), ds, spec_dir, (pat[len(T._SIDECAR_FP) :],), clips={t.name for t in trees})
            bad = {(r["emb"], r["clip"]) for r in scan if r["verdict"] != "fresh"}
            hit = [t for t in trees if (t.parent.name, t.name) in bad]
        elif pat.startswith(T._INPUT_FP):
            fn = getattr(importlib.import_module(pat[len(T._INPUT_FP) :]), "tree_fresh", None)
            if fn is None:
                raise InputMissingError(
                    f"[hand_pipeline] {pat[len(T._INPUT_FP) :]}.tree_fresh is missing -- cannot judge {name!r} per tree"
                )
            hit = [t for t in trees if not fn(ds, spec_dir, t)]
        elif pat.startswith(T._KEYED):
            needle = T.cell_needle(pat, spec_dir)
            hit = [t for t in trees if not _has_bytes(t / "task_info.json", needle)]
        else:
            raise InputMissingError(f"[hand_pipeline] step {name!r}: cell {pat!r} has no per-tree judgement")
        for t in hit:
            out[t].append(name)
    return out


def _baked_trees(ds: str, combo: str) -> set:
    """Baked tree paths of a dataset over every embodiment and combo variant root."""
    # Aerial and articulated trees are baked under the variant combo roots (combo + variant suffix).
    seen = set()
    for emb in T.embs(ds):
        for v in RR.COMBO_VARIANTS:
            try:
                d = P.layer_path("trees", ds, combo=combo + v, emb=emb)
            except (KeyError, ValueError):
                continue
            seen.update(str(x) for x in d.glob("*" + P.TREE_SFX))
    return seen


def _cell(ds: str, hd: str, combo: str, stage, f0_gaps: dict) -> tuple[str, bool]:
    """(the printed cell, is it a gap) of one step on one dataset."""
    name, pat, _slot, cmd = stage
    if pat.startswith(T._SIDECAR_FP):
        ok, need = _sidecar_cell(ds, hd, pat[len(T._SIDECAR_FP) :])
        return f"{ok}/{need}".rjust(13), ok != need
    if pat.startswith(T._INPUT_FP):
        ok, need = _input_fp_cell(ds, hd, pat[len(T._INPUT_FP) :])
        return f"{ok}/{need}".rjust(13), ok != need
    if pat == T._COVER:
        have, need = _stage1_cover(ds, hd)
        return f"{have}/{need:<7}"[:13].rjust(13), have == 0
    if pat == T._UNI:
        from dexengine.placement import uni_layout as _UL  # noqa: PLC0415

        n = 2 - len(set(_UL.templates_missing(hd)) | set(_UL.templates_stale(hd)))
        return f"{n}/2".rjust(13), n < 2
    if pat == T._POOL:
        root = P.pool_root()
        n = sum(_pool_ready(root, combo, side) for side in ("right", "left"))
        return f"{n}/2".rjust(13), n < 2
    if pat == T._TREES:
        seen = _baked_trees(ds, combo)
        return f"{len(seen):>13}", not seen
    if isinstance(cmd, T.Run) and not T.applies(cmd, ds):
        return f"{'-':>13}", False
    if pat == T._F0_JUDGED:
        f0c = PL.f0_cells(ds, hd)
        ok = sum(o["judged"] for o in f0c.values())
        need = max(sum(o["need"] for o in f0c.values()), 1)  # expecting 0 clips is not complete
        if ok != need:
            f0_gaps[ds] = [f"emb={e} {PL.f0_gap_text(o)}" for e, o in f0c.items() if o["judged"] != o["need"]] or [
                "the base layer has no clips (0 expected)"
            ]
        return f"{ok}/{need}".rjust(13), ok != need
    if pat.startswith(T._KEYED):
        needle = T.cell_needle(pat, hd)
        n = tot = 0
        for f in (P.main_root() / "data" / ds / "base" / hd).glob("*/*/task_info.json"):
            tot += 1
            n += _has_bytes(f, needle)
        if T.every_tree(pat, cmd):
            need = max(tot, 1)
            return f"{n}/{need}".rjust(13), n != need
        return f"{n:>13}", n == 0
    raise SystemExit(f"[hand_pipeline] step {name!r}: cannot count cell {pat!r}")


def status(datasets: list[str]) -> int:
    """Print the progress table of the active hand over `datasets` and the first empty step per dataset."""
    hand = RR.ACTIVE_HAND
    try:
        hd = RR.hand_dir()
        combo = RR.combo()
    except KeyError as e:
        print(f"✗ hand axis: {e}")
        print("  -> register the hand first (python3 dexengine/data/stages.py --register --arm <arm>).")
        return 2
    print(f"hand {hand} · spec dir {hd} · combo {combo}")
    print(f"interpreter: retarget/cases/render = {T.py('retarget-py')}")
    print()
    print(f"{'dataset':11}" + "".join(f"{s[0][:11]:>13}" for s in T.STAGES))
    print(
        f"{'':11}{'(parent takes retargeted/needed)':>13}"[:40]
        + "   <- only step 1 counts takes; the others count segments."
    )
    first_gap, f0_gaps = {}, {}
    for ds in datasets:
        cells = []
        for stage in T.STAGES:
            text, gap = _cell(ds, hd, combo, stage, f0_gaps)
            cells.append(text)
            if gap:
                first_gap.setdefault(ds, stage[0])
        print(f"{ds:11}" + "".join(cells))
        STATUS_ROWS[ds] = {g[0]: c.strip() for g, c in zip(T.STAGES, cells, strict=True)}
        STATUS_ROWS[ds]["first_gap"] = first_gap.get(ds, "")
    print()
    for ds, lines in f0_gaps.items():
        print(f"7 f0 missing clips {ds}: " + " | ".join(lines))
    for side_dir, note in POOL_NOTES.items():
        print(f"reach map {side_dir} not ready: {note}")
    if first_gap:
        print("First empty step (run from here) -- † = per combo (the same for every dataset):")
        for ds, st in sorted(first_gap.items()):
            cmd = T.cmd_of(next(g for g in T.STAGES if g[0] == st))
            print(f"  {ds:11} {st:14} <-\n      {cmd}")
    else:
        print("Every dataset has outputs up to step 8.")
    # Rules whose thresholds are calibrated to hand geometry: reusing another hand's number does not fail, it silently
    # answers differently.
    print()
    print("\n".join(RR.hand_dependent_report(hand)))
    return 0
