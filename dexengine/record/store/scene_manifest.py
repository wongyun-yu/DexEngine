#!/usr/bin/env python3
"""Scene manifest sidecar `meta/scene_manifest.jsonl` of a final dataset: one row per live episode, keyed by its card,
holding the scene statement of the tree the episode was recorded from (record/scene/manifest.build, the one builder).
This module is the one writer of the sidecar.

Card = hand and dataset of the final, the episode's round, its source clip (merge_rows.src_clip), its clip with the
variant suffix (`variant`) and its look name; (variant, look) is merge_rows.card, disjoint within a final. Live
episodes = the final's episodes minus the superseded ones (merge_check.superseded_indices).
Tree of an episode: the variant tree of an aug variant (scene_layout.variant_trees: the piece archive roots), else the
original tree on this machine's data root (scene_episode._tree), as the scene layout sidecar resolves them.
Trajectory of an episode: its data parquet in the final and the final's info.json (manifest.traj); the statement's
self-penetration is measured on its executed joints, once per distinct joint table (the looks of one rollout share it).
sync = build the rows of the live episodes that have none (every row with --all) and rewrite the sidecar under
  piece.final_lock; one scene per tree, shared by the looks of that tree. A missing tree leaves its rows missing (rc 1).
check = `check_scene_manifest`: a missing sidecar, a missing, extra or repeated card, unknown keys, a scene of another
  dataset or tree, or a statement the scene oracle refuses (manifest.problems on the card's trajectory: its
  self-penetration is measured again, and a delivered trajectory over the line fails) fails (rc 1). Run it beside the
  final's `merge verify`; a final made before this sidecar has none and fails it.
Marker [SCENE_MANIFEST].
Usage: python -m dexengine.record.store.scene_manifest sync|check --final <final> --hand <hand> --dataset <ds>
       [--arm rby1 (sync)] [--all] [--dry]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dexengine.core import config
from dexengine.core import paths as P
from dexengine.record.scene import manifest as MF
from dexengine.record.store import merge_check as MC
from dexengine.record.store import merge_rows as MR

SIDECAR = "scene_manifest.jsonl"  # under meta/
MARK = "[SCENE_MANIFEST]"
CARD_KEYS = ("hand", "dataset", "round", "clip", "variant", "look")
ROW_KEYS = ("card", "scene")


def die(msg: str):
    raise SystemExit(f"{MARK} ✗ {msg}")


def card(hand: str, ds: str, e: dict) -> dict:
    """The card of episode line `e` of the final of `hand` x `ds` (module doc)."""
    variant, look = MR.card(e)
    return {
        "hand": hand,
        "dataset": ds,
        "round": e.get("round"),
        "clip": MR.src_clip(e),
        "variant": variant,
        "look": look,
    }


def _key(c: dict) -> tuple:
    return tuple(c[k] for k in CARD_KEYS)


def live_episodes(final: Path) -> list:
    """Episode lines of the final that are not superseded."""
    gone = MC.superseded_indices(final)
    return [e for e in MC.jsonl(final / MC.EPISODES) if int(e["episode_index"]) not in gone]


def row_line(row: dict) -> str:
    """One sidecar line: sorted keys, compact separators (the same row gives the same bytes)."""
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _sidecar(final: Path) -> Path:
    return Path(final) / MC.META / SIDECAR


def _tree_paths(arm: str, hand: str, ds: str, eps: list) -> dict:
    """{episode_index: (tree path or None, the path or the reason it is missing)}."""
    from dexengine.record.store import scene_episode as SE  # noqa: PLC0415
    from dexengine.record.store import scene_layout as SL  # noqa: PLC0415

    recs = [
        {"i": e["episode_index"], "round": e.get("round"), "clip": e["clip"], "side": e["side"]}
        | {"piece_id": e.get("piece_id")}
        for e in eps
    ]
    SL.variant_trees(arm, recs, hand, ds)
    out = {}
    for r in recs:
        tr, where = r["vtree"] if SE.aug_variant(r) else SE._tree(arm, hand, ds, r["side"], r["clip"])
        out[r["i"]] = (None if tr is None else where, where)
    return out


def trajectories(final: Path, eps: list) -> dict:
    """{episode_index: (manifest.traj of the episode, its joint table key)} of episodes of the final."""
    from dexengine.placement.collision import self_pen as SP  # noqa: PLC0415

    info = MC.rjson(final / MC.INFO)
    fam = MC.families(info)[MC.DATA]
    out = {}
    for e in eps:
        pq = MC.ep_path(final, info, fam, e["episode_index"])
        out[e["episode_index"]] = (MF.traj(pq, info), SP.table_key(SP.read_joints(pq)))
    return out


def build_rows(arm: str, hand: str, ds: str, final: Path, eps: list) -> tuple[list, dict]:
    """(rows of the episodes whose tree is found, {reason: count} of the others); one scene per tree and joint table."""
    paths, trajs = _tree_paths(arm, hand, ds, eps), trajectories(final, eps)
    scenes, rows, why = {}, [], {}
    for e in eps:
        tree, where = paths[e["episode_index"]]
        if tree is None:
            why[where[:90]] = why.get(where[:90], 0) + 1
            continue
        t, key = trajs[e["episode_index"]]
        if (tree, key) not in scenes:
            scenes[tree, key] = MF.build(tree, t)
        rows.append({"card": card(hand, ds, e), "scene": scenes[tree, key]})
    return rows, why


def write_sidecar(final: Path, hand: str, ds: str, new: list) -> int:
    """Merge rows `new` into the sidecar (a new row replaces the row of its card), keep the rows of live cards only,
    and rewrite it in episode order under piece.final_lock. Returns the number of rows written."""
    from dexengine.record import piece as PC  # noqa: PLC0415 — owner of the final lock

    with PC.final_lock(final):
        order = [_key(card(hand, ds, e)) for e in live_episodes(final)]
        have = {_key(r["card"]): r for r in MC.jsonl(_sidecar(final))} | {_key(r["card"]): r for r in new}
        body = "".join(row_line(have[k]) for k in order if k in have)
        p = _sidecar(final)
        tmp = p.with_name(p.name + f".part.{os.getpid()}")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, p)
    return sum(k in have for k in order)


def row_problems(row, hand: str, ds: str, seen: dict, trajs: dict) -> list:
    """Problems of one row: keys, card of this final, scene of the card's dataset and tree, scene oracle on the card's
    trajectory (`trajs` {card key: (manifest.traj, joint table key)}; cached per scene text and joint table in `seen`,
    the looks of one rollout share it)."""
    if not isinstance(row, dict) or set(row) != set(ROW_KEYS):
        return [f"row keys {sorted(row) if isinstance(row, dict) else type(row).__name__} != {list(ROW_KEYS)}"]
    c, s = row["card"], row["scene"]
    if not isinstance(c, dict) or set(c) != set(CARD_KEYS):
        return [f"card keys {sorted(c) if isinstance(c, dict) else c!r} != {list(CARD_KEYS)}"]
    if not isinstance(s, dict):
        return [f"{c}: scene is not an object"]
    bad = [f"card of another final {c['hand']}/{c['dataset']}"] * ((c["hand"], c["dataset"]) != (hand, ds))
    bad += [f"scene dataset {s.get('dataset')!r} != {ds!r}"] * (s.get("dataset") != ds)
    bad += [f"scene tree {s.get('tree')!r} != {P.tree_name(str(c['variant']))!r}"] * (
        s.get("tree") != P.tree_name(str(c["variant"]))
    )
    t, tk = trajs.get(_key(c), (None, None))
    key = (row_line(s), tk)
    if key not in seen:
        seen[key] = MF.problems(s, t)
    where = "/".join(str(c[k]) for k in CARD_KEYS[2:])
    return [f"{where}: {b}" for b in bad + seen[key]]


def check_scene_manifest(final: Path, hand: str, ds: str) -> list:
    """The oracle of the sidecar of one final (module doc): [] when it holds, else one line per problem."""
    p = _sidecar(final)
    if not p.is_file():
        return [f"{final}: no meta/{SIDECAR}"]
    eps = live_episodes(final)
    by = trajectories(final, eps)
    trajs = {_key(card(hand, ds, e)): by[e["episode_index"]] for e in eps}
    rows, bad, seen, keys = MC.jsonl(p), [], {}, []
    for r in rows:
        bad += row_problems(r, hand, ds, seen, trajs)
        if isinstance(r, dict) and isinstance(r.get("card"), dict) and set(r["card"]) == set(CARD_KEYS):
            keys.append(_key(r["card"]))
    live = [_key(card(hand, ds, e)) for e in eps]
    bad += [f"repeated cards {len(keys) - len(set(keys))}"] * (len(keys) != len(set(keys)))
    bad += [f"missing rows {len(m)}: {sorted(m)[:2]}" for m in [set(live) - set(keys)] if m]
    return bad + [f"rows without a live episode {len(x)}: {sorted(x)[:2]}" for x in [set(keys) - set(live)] if x]


def sync(a) -> int:
    final = Path(a.final)
    eps = live_episodes(final)
    have = set() if a.all else {_key(r["card"]) for r in MC.jsonl(_sidecar(final))}
    todo = [e for e in eps if _key(card(a.hand, a.dataset, e)) not in have]
    rows, why = build_rows(a.arm, a.hand, a.dataset, final, todo)
    got = None if a.dry else write_sidecar(final, a.hand, a.dataset, rows)
    for w, n in sorted(why.items(), key=lambda kv: -kv[1])[:10]:
        print(f"{MARK} missing {n} × {w}", flush=True)
    print(
        f"{MARK} {a.dataset}/{a.hand} live episodes {len(eps)} · new rows {len(rows)} · missing "
        f"{len(todo) - len(rows)} · rows after write {'- (dry run, nothing written)' if a.dry else got} · {final}",
        flush=True,
    )
    return 0 if len(todo) == len(rows) and (a.dry or got == len(eps)) else 1


def check(a) -> int:
    bad = check_scene_manifest(Path(a.final), a.hand, a.dataset)
    for b in bad[:20]:
        print(f"{MARK} ✗ {b}", flush=True)
    print(f"{MARK} end check {a.dataset}/{a.hand} problems {len(bad)} · {a.final}", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("sync", "check"))
    ap.add_argument("--final", required=True, help="the final dataset folder (data/<ds>/collect/<combo>)")
    ap.add_argument("--hand", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", default="", help="sync: the arm of the trees")
    ap.add_argument("--all", action="store_true", help="sync: rebuild every row")
    ap.add_argument("--dry", action="store_true", help="sync: build only, write nothing")
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    if not (Path(a.final) / MC.EPISODES).is_file():
        die(f"{a.final}: no {MC.EPISODES}")
    if a.cmd == "sync" and not a.arm:
        die("sync needs --arm (the trees are named by arm and hand)")
    return sync(a) if a.cmd == "sync" else check(a)


if __name__ == "__main__":
    sys.exit(main())
