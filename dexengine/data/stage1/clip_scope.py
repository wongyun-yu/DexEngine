#!/usr/bin/env python3
"""Stage-1 explicit scope (DEXENGINE_STAGE1_CLIPS): rebuilds only the listed trees and their families in one dataset.

Why: for a repair where only some trees of a dataset should change (retraining only the trees whose rule version
moved), running stage 1 as is rebuilds everything column 2 calls stale (base_layer.stale.tree_stale_reason), and the
scenes of trees that will not be retrained change under their checkpoints.
So the caller states the scope, this module turns it into step commands, and after writing it proves **by
measurement** that no tree outside the scope changed (set difference of before/after sha).

Rules:
  family    every base tree whose source take (paths.parent_clip) equals a listed tree's: take · `_segNN` · pieces.
            Pieces are cut from their parent (segment.split), so a changed parent changes its sibling pieces too; writes
                allowed = the whole family.
  base      base is rebuilt only for takes with at least one stale tree in the family (tree_stale_reason ≠ "").
            With `--rebuild` (DEXENGINE_STAGE1_CLIPS_REBUILD=1), every take of the family regardless of staleness: for
            applying a code repair without a version bump to the listed trees only (the stale rule counts them as
                current).
  convert   2b only for family trees that are not pieces (take · `_segNN`); pieces are re-cut from the converted parent
      (piece upkeep).
  command   when a step argv's script takes a clip input (CLIP_ARGS), that input is **replaced** by the scope value
            (replaced if present, appended otherwise). Steps that take none (sidecars · roles · quality · wrist cache …)
            see the whole dataset but rewrite only trees whose input fingerprint changed; the gate below catches a
                broken promise.
  gate      compares per-tree content sha of the base layer (every dataset · hand · emb) before and after writing:
            changed · added · removed trees ⊆ family. Any tree outside gives rc 8.
  names     the list is a hand-written statement, so after the registry re-cuts pieces it holds old piece names (`_tN`).
            A piece name missing from base resolves to its parent's (only `_tN` stripped) current registry pieces that
            exist in base, else to the parent tree (resolve_names · resolution line · resolved=<n>). Unresolvable names
                fail.
  replan    piece upkeep (quarantining pieces outside the registry) runs **after** the plan; the driver then runs
            `plan --replan` to set family to the current family and widen allowed to old ∪ new (replan_write · marker
                `[STAGE1_CLIPS] replan`).
Markers: [STAGE1_CLIPS] (plan) · [STAGE1_CLIPS_ARGV] (command) · [STAGE1_CLIPS_GATE] (gate).
A run without the markers is invalid.

Usage (called by the stage-1 driver):
  python -m dexengine.data.stage1.clip_scope plan --ds <ds> --hand <hand> --clips <file> --out <dir> [--rebuild]
      [--replan]
  python -m dexengine.data.stage1.clip_scope snap --ds <ds> --hand <hand> --out <tsv>
  python -m dexengine.data.stage1.clip_scope gate --before <tsv> --after <tsv> --scope <dir>
  python -m dexengine.data.stage1.clip_scope argv --line <one shlex line> --scope <dir>   → the rewritten line · SKIP
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from multiprocessing import Pool
from pathlib import Path

from dexengine.core import capacity  # noqa: E402
from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import input_missing

TAG = "[STAGE1_CLIPS]"
STEP = "dexengine.data.stage1.clip_scope"  # capacity.NEEDS key (the snapshot pool)
PIECE_RX = re.compile(r"_t\d+$")
#: Step scripts that take a clip input → (flag, form comma|file, unit base|convert|family). Units:
#:   base = takes to rebuild (with a stale tree in the family) · convert = family trees that are not pieces · family =
#:   the whole family
#: Keys are the basenames of the step table's argv (stage1.table.STAGES; tests/data/test_registries pins that every key
#: names a step script that declares the flag — an unknown key would silently run the whole dataset).
CLIP_ARGS = {
    "base_adapter.py": ("--clips", "comma", "base"),
    "artic_convert.py": ("--clips", "comma", "convert"),
    "wrist_desk_stage.py": ("--clips", "file", "family"),
    "taxonomy.py": ("--clips", "comma", "family"),
    "prejudge.py": ("--clips", "file", "family"),
}
UNITS = ("base", "convert", "family")


def die(msg: str, rc: int = 2) -> None:
    print(f"{TAG} ✗ {msg}", file=sys.stderr, flush=True)
    raise SystemExit(rc)


def trees_of(ds: str, hand: str) -> dict[str, Path]:
    """{emb/tree: path}: every tree of the base layer (folders with task_info.json; layer location from
    paths.layer_path)."""
    out = {}
    for emb in sorted(P.data_layout.EMBODIMENTS):
        d = P.layer_path("base", ds, hand=P.hand_spec_dir(hand), emb=emb)
        for t in sorted(d.iterdir()) if d.is_dir() else ():
            if t.is_dir() and (t / "task_info.json").is_file():
                out[f"{emb}/{t.name}"] = t
    return out


def registries(ds: str) -> dict[str, dict]:
    """{emb: registry parents}: the hand-independent D36 registry (paths.units_registry · producer segment.split).
    Missing embs give an empty table."""
    out = {}
    for emb in sorted(P.data_layout.EMBODIMENTS):
        f = P.units_registry(ds, emb)
        out[emb] = (json.loads(f.read_text(encoding="utf-8")).get("parents") or {}) if f.is_file() else {}
    return out


def resolve_core(names: list[str], trees: dict[str, Path], regs: dict[str, dict]) -> tuple[list[str], dict, list]:
    """List names → names present in base; **the only owner of the old piece name resolution rule** (the scope list
    (this file) and the rejudge list call this one function). The list is a hand-written statement,
    so after the registry changes (pieces re-cut, `_tN` gone) it holds old piece names; trusting names literally would
    make the scope and rejudge lists fail in turn.
      A base tree stays as is.
      Otherwise, **only for the D36 piece form (`_tN`)**, it resolves to the parent that produced the piece (the name
      with only `_tN` stripped; `_segNN` stays, it is the registry key):
        the current registry's pieces of that parent that exist in base → else the parent tree itself (a parent re-cut
            into one piece).
      Anything else (not piece-shaped · neither parent nor piece in base) is unresolved; the caller decides what to do.
    Returns (names in base, in input order without duplicates, {old name: [new names]}, [unresolved names])."""
    have: dict[str, set] = {}
    for k in trees:
        emb, name = k.split("/", 1)
        have.setdefault(emb, set()).add(name)
    every = set().union(*have.values()) if have else set()
    out, moved, dead = [], {}, []
    for n in names:
        new = []
        if n in every:
            new = [n]
        elif PIECE_RX.search(n):
            par = PIECE_RX.sub("", n)
            new = sorted(
                {u["name"] for emb, hs in have.items() for u in (regs.get(emb) or {}).get(par, ()) if u["name"] in hs}
            ) or ([par] if par in every else [])
            if new:
                moved[n] = new
        if not new:
            dead.append(n)
        out += [x for x in new if x not in out]
    return out, moved, dead


def resolve_names(names: list[str], trees: dict[str, Path], regs: dict[str, dict]) -> tuple[list[str], dict]:
    """Name resolution of the scope list (resolve_core); fails when any name stays unresolved (no invented scope).
    Returns (resolved names, {old name: [new names]})."""
    out, moved, dead = resolve_core(names, trees, regs)
    if dead:
        die(
            f"{len(dead)} listed trees are not in the base layer and resolve through neither the registry nor "
            f"base (e.g. {dead[:3]}) — "
            "no invented scope"
        )
    return out, moved


def resolve_listed(ds: str, hand: str, names: list[str]) -> tuple[dict, list]:
    """Reads from the data root and calls resolve_core. Returns ({old name: [new names]}, [unresolved names])."""
    _out, moved, dead = resolve_core(names, trees_of(ds, hand), registries(ds))
    return moved, dead


def resolve_pairs(pairs: set, hand: str, tag: str) -> set:
    """Rejudge list {(dataset, name)} with old piece names resolved to current
    names; unresolved names stay (if they reach no candidate, the end check there fails). Marker `<tag> resolved
    <ds>:<old> → <new>`."""
    out = set(pairs)
    for ds in sorted({d for d, _c in pairs}):
        moved, _dead = resolve_listed(ds, hand, sorted(c for d, c in pairs if d == ds))
        for old, new in sorted(moved.items()):
            out.discard((ds, old))
            out.update((ds, n) for n in new)
            print(
                f"{tag} resolved {ds}:{old} → {' '.join(new)} (piece name missing from base — hand {hand}'s "
                "current registry · "
                "base)",
                flush=True,
            )
    return out


def family_plan(names: list[str], trees: dict[str, Path], stale, rebuild: bool = False) -> dict:
    """List → family · base · convert units. `stale(path) -> str` is the stale reason (empty = current). Missing names
    fail.
    `rebuild` = every take of the family is a base unit regardless of staleness (the stale rule is not called)."""
    have = {k.split("/", 1)[1] for k in trees}
    miss = [n for n in names if n not in have]
    if miss:
        die(input_missing(f"{len(miss)} listed trees are not in the base layer (e.g. {miss[:3]}) — no invented scope"))
    fams = {P.parent_clip(n) for n in names}
    members = sorted(k for k in trees if P.parent_clip(k.split("/", 1)[1]) in fams)
    base = sorted({P.parent_clip(k.split("/", 1)[1]) for k in members if rebuild or stale(trees[k])})
    convert = sorted({k.split("/", 1)[1] for k in members if not PIECE_RX.search(k.split("/", 1)[1])})
    family = sorted({k.split("/", 1)[1] for k in members})
    return {
        "list": sorted(set(names)),
        "families": sorted(fams),
        "members": members,
        "base": base,
        "convert": convert,
        "family": family,
    }


def replan_write(out: Path, plan: dict) -> dict:
    """Write rule of re-planning the scope **after** piece upkeep (the driver's step that quarantines pieces outside
    the registry). The plan was measured before upkeep, so pieces about to be quarantined stay in the family list (then
    2e fails with 'list > base').
      family = the current family (clip input of the later steps 2d … 7) · base · convert = unchanged (2 · 2b already
          ran before this point)
      allowed = old ∪ new; the gate's before-snapshot predates upkeep, so pieces just quarantined must be inside the
          scope too
    Returns {"family": (old count, new count), "allowed": (old count, new count), "gone": [names that left the
    family]}."""
    old_fam = [ln for ln in (out / "family.txt").read_text(encoding="utf-8").splitlines() if ln]
    old_alw = [ln for ln in (out / "allowed.txt").read_text(encoding="utf-8").splitlines() if ln]
    alw = sorted(set(old_alw) | set(plan["members"]))
    (out / "family.txt").write_text("".join(f"{n}\n" for n in plan["family"]))
    (out / "allowed.txt").write_text("".join(f"{k}\n" for k in alw))
    return {
        "family": (len(old_fam), len(plan["family"])),
        "allowed": (len(old_alw), len(alw)),
        "gone": sorted(set(old_fam) - set(plan["family"])),
    }


def cmd_plan(a) -> int:
    listed = P.read_clip_list(a.clips, dataset=a.ds, label="DEXENGINE_STAGE1_CLIPS", quiet=True)
    if not listed:
        die(f"scope list is empty: {a.clips}")
    trees = trees_of(a.ds, a.hand)
    names, moved = resolve_names(listed, trees, registries(a.ds))
    for old, new in sorted(moved.items()):
        print(
            f"{TAG} resolved {old} → {' '.join(new)} (piece name missing from base — current registry · base)",
            flush=True,
        )
    from dexengine.data.base_layer import stale as BS  # noqa: PLC0415 — owner of the stale rule (planning only)

    plan = family_plan(names, trees, BS.tree_stale_reason, a.rebuild)
    out = Path(a.out)
    sha = hashlib.sha256("\n".join(plan["list"]).encode()).hexdigest()[:12]
    if a.replan:
        if not (out / "plan.json").is_file():
            die(input_missing(f"--replan without an earlier plan ({out}/plan.json) — no scope to re-plan"))
        r = replan_write(out, plan)
        print(
            f"{TAG} replan ds={a.ds} hand={a.hand} family={r['family'][0]}→{r['family'][1]} "
            f"allowed={r['allowed'][0]}→{r['allowed'][1]} (old ∪ new) gone={len(r['gone'])} resolved={len(moved)} "
            f"sha={sha}" + (f" e.g. {r['gone'][:3]}" if r["gone"] else ""),
            flush=True,
        )
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for unit in UNITS:
        (out / f"{unit}.txt").write_text("".join(f"{n}\n" for n in plan[unit]))
    (out / "allowed.txt").write_text("".join(f"{k}\n" for k in plan["members"]))
    (out / "plan.json").write_text(
        json.dumps(
            dict(
                ds=a.ds,
                hand=a.hand,
                clips=os.path.abspath(a.clips),
                sha=sha,
                rebuild=bool(a.rebuild),
                resolved=len(moved),
                **{k: len(v) for k, v in plan.items()},
            ),
            ensure_ascii=False,
        )
    )
    print(
        f"{TAG} ds={a.ds} hand={a.hand} list={len(plan['list'])} families={len(plan['families'])} "
        f"members={len(plan['members'])} base={len(plan['base'])} convert={len(plan['convert'])} "
        f"rebuild={int(a.rebuild)} sha={sha} "
        f"resolved={len(moved)} "
        f"file={os.path.abspath(a.clips)}",
        flush=True,
    )
    return 0


def tree_digest(path: str) -> str:
    """Content sha of a tree folder: relative paths and all bytes (sorted)."""
    h = hashlib.sha256()
    root = Path(path)
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(root)).encode() + b"\0")
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
    return h.hexdigest()


def cmd_snap(a) -> int:
    trees = trees_of(a.ds, a.hand)
    keys = sorted(trees)
    opt = None if a.jobs is None else int(a.jobs)
    with capacity.sized(STEP, len(keys), opt) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        digs = pool.map(tree_digest, [str(trees[k]) for k in keys], chunksize=32)
    Path(a.out).write_text("".join(f"{k}\t{d}\n" for k, d in zip(keys, digs, strict=True)))
    print(f"{TAG} snap ds={a.ds} hand={a.hand} trees={len(keys)} → {a.out}", flush=True)
    return 0


def read_snap(f: str) -> dict[str, str]:
    return dict(line.split("\t") for line in Path(f).read_text().splitlines() if line.strip())


def gate(before: dict, after: dict, allowed: set) -> dict:
    """Changed · added · removed trees and those outside the scope."""
    changed = {k for k in before.keys() & after.keys() if before[k] != after[k]}
    added, removed = set(after) - set(before), set(before) - set(after)
    touched = changed | added | removed
    return {
        "changed": sorted(changed),
        "added": sorted(added),
        "removed": sorted(removed),
        "outside": sorted(touched - allowed),
        "touched": len(touched),
    }


def cmd_gate(a) -> int:
    allowed = {x.strip() for x in (Path(a.scope) / "allowed.txt").read_text().splitlines() if x.strip()}
    g = gate(read_snap(a.before), read_snap(a.after), allowed)
    print(
        f"{TAG}_GATE changed={len(g['changed'])} added={len(g['added'])} removed={len(g['removed'])} "
        f"allowed={len(allowed)} outside={len(g['outside'])}",
        flush=True,
    )
    if g["outside"]:
        print(
            f"{TAG}_GATE ✗ {len(g['outside'])} trees outside the scope changed (e.g. {g['outside'][:10]})",
            file=sys.stderr,
        )
        return 8
    return 0


def scoped_argv(argv: list[str], scope: Path) -> list[str] | None:
    """Step argv → scoped argv. For scripts taking a clip input, the input is replaced by the scope value (appended if
    absent) · None when that unit is empty (nothing to run). Other scripts stay as they are."""
    script = next((Path(x).name for x in argv if x.endswith(".py")), "")
    if script not in CLIP_ARGS:
        return list(argv)
    flag, form, unit = CLIP_ARGS[script]
    names = [x.strip() for x in (scope / f"{unit}.txt").read_text().splitlines() if x.strip()]
    if not names:
        return None
    val = ",".join(names) if form == "comma" else str(scope / f"{unit}.txt")
    out = list(argv)
    if flag in out:
        i = out.index(flag)
        if i + 1 >= len(out):
            die(input_missing(f"{script}: no value after {flag}"))
        out[i + 1] = val
    else:
        out += [flag, val]
    return out


def cmd_argv(a) -> int:
    argv = shlex.split(a.line)
    new = scoped_argv(argv, Path(a.scope))
    script = next((Path(x).name for x in argv if x.endswith(".py")), "?")
    if new is None:
        print(f"{TAG}_ARGV {script} scope unit is empty — skipped", file=sys.stderr, flush=True)
        print("SKIP")
        return 0
    how = "scope" if script in CLIP_ARGS else "whole dataset (input fingerprint; the gate measures the scope)"
    print(f"{TAG}_ARGV {script} {how}", file=sys.stderr, flush=True)
    print(shlex.join(new))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    for flag in ("--ds", "--hand", "--clips", "--out"):
        p.add_argument(flag, required=True)
    p.add_argument(
        "--rebuild", action="store_true", help="every take of the family as a base unit regardless of staleness"
    )
    p.add_argument(
        "--replan",
        action="store_true",
        help="re-plan after piece upkeep: only family becomes current · allowed = old ∪ new · base · convert unchanged",
    )
    s = sub.add_parser("snap")
    for flag in ("--ds", "--hand", "--out"):
        s.add_argument(flag, required=True)
    s.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    g = sub.add_parser("gate")
    for flag in ("--before", "--after", "--scope"):
        g.add_argument(flag, required=True)
    v = sub.add_parser("argv")
    for flag in ("--line", "--scope"):
        v.add_argument(flag, required=True)
    a = ap.parse_args()
    return {"plan": cmd_plan, "snap": cmd_snap, "gate": cmd_gate, "argv": cmd_argv}[a.cmd](a)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.stage1.clip_scope")
    raise SystemExit(main())
