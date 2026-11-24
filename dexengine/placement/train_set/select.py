#!/usr/bin/env python3
"""Training-set selection — the one generator of training bundles.

Partition (--partition-prefix — the campaign bundles; rule PACK_RULE "object-family-v2"):
  1. Pools = one run key each (train_ready_util.run_key = tree root (ledger axis) + run group: idle side · shared
     object · articulated main object · scene). The training entry calls the same function and refuses a list with
     two keys, so every bundle built here passes the entry's run check. Every pool is split — nothing is dropped.
  2. Object key per clip (`target_object`, declared per dataset in datasets.DATASET_PROFILE[ds].obj_key); a
     two-object key with fewer than REKEY_MIN clips in its pool is re-keyed to its member object. Shape family per
     key (`key_shape` — similar objects). Both are read by object_shape.py.
  3. Bundles per pool = ceil(pool / n), all <= n; an object 0 or >= CO_MIN per bundle, several objects per bundle, a
     short object with its family, dups (a clip borrowed from its primary bundle) only where a short object still
     lacks CO_MIN similar clips — the rule is bundle_pack.py (PACK_RULE). Deterministic.
  4. The primaries must equal the input universe, each once, and every dup must name its primary bundle — it raises
     otherwise.

Single set (--out): n clips with at least k per target object, a per-object cap mirroring the remaining corpus,
  round robin (`select`); --n 0 = the whole universe. The universe must be ONE run key (--run-key names one when
  there are several — an explicit choice, never a silent largest-group pick).

Names: input and output are both the contract clip_id (a tree suffix `_pl_rt` is stripped if present, and
re-attached when --tree is given for the output).

Training reads a list, not a view: `SEQ_DIR=<combo tree root> DEXENGINE_SEQ_LIST=<list>` — the training entry's roster
  rule re-checks the list against its roster owner and trains exactly sorted(list). The list files are the product.
View (--view-root, opt-in — a link farm for tools that want a folder): for every list, build a symlink view
  `<view-root>/<set>/<combo>/<tree> → the real tree (absolute path)`. A missing target tree raises, and a link count
  != list count raises. The view is derived: a rebuild wipes and re-creates it (a non-symlink entry makes the tool
  die without touching it).

Usage (split one adopted universe into bundles of at most 64):
  python3 -m dexengine.placement.train_set.select --status-ledger <chunk>/status_ledger.tsv \\
      --dataset taco --emb bimanual --combo rby1_sharpa_wave --n 64 --partition-prefix data/taco/views/sample64
  Single set: --out <list> --n 64 --k 4 --combo <combo> [--run-key <key>|largest]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import ARM_IDS, HAND_SPEC_DIR, qpos_layout, read_clip_list, split_combo, tree_name, trees_dir
from dexengine.data import rules as RR
from dexengine.placement import train_ready_util as TRU
from dexengine.placement.train_set.bundle_pack import PACK_RULE, pack, rekey

# shape_files and target_object_file are re-exported (bundle.run_inputs reads them as ST.*)
from dexengine.placement.train_set.object_shape import (  # noqa: F401
    key_shape,
    obj_key_marker,
    shape_files,
    target_object,
    target_object_file,
)

HELP = (
    "Training-set selection — at least k clips of the same target object per set, without exhausting objects "
    "for the later sets."
)


def _round_robin(k, order, out, n, taken, take):
    """Pass 1: round robin up to k per object; pass 2: fill up to each cap by round robin (until n)."""
    # Pass 1: round robin up to k per object
    for _round in range(k):
        for o in order:
            if len(out) >= n:
                break
            if len(taken[o]) < k:
                take(o)
    # Pass 2: fill up to cap by round robin
    progressed = True
    while len(out) < n and progressed:
        progressed = False
        for o in order:
            if len(out) >= n:
                break
            if take(o):
                progressed = True


def select(
    universe: list[str], exclude: set[str], n: int, k: int, dataset: str, emb: str
) -> tuple[list[str], dict, dict]:
    """One set of n clips of `universe` minus `exclude`: at least k per eligible target object, round robin, a
    per-object cap proportional to the remaining corpus → (clips, taken count per object, remaining per object)."""
    pool = [c for c in universe if c not in exclude]
    by_obj: dict[str, list[str]] = defaultdict(list)
    for c in pool:
        by_obj[target_object(c, dataset, emb)].append(c)
    for v in by_obj.values():
        v.sort()
    eligible = [o for o in by_obj if len(by_obj[o]) >= k]
    if not eligible:
        raise InputMissingError(f"[select] no object has {k} or more clips left ({len(pool)} clips left)")
    # Per-object cap = **share of the remaining corpus** (at least k). A uniform cap exhausts the
    # small objects first, leaving the tail sets with one object (plate) only. With a proportional cap
    # every set mirrors the remaining distribution, so the mix holds to the end. If the total is n or
    # more but the sum of the caps falls short, raise the caps by 1 until it is covered.
    tot = sum(len(by_obj[o]) for o in eligible)
    cap = {o: max(k, round(n * len(by_obj[o]) / tot)) for o in eligible}
    while sum(min(cap[o], len(by_obj[o])) for o in eligible) < min(n, tot):
        for o in eligible:
            cap[o] += 1
    order = sorted(eligible, key=lambda o: (-len(by_obj[o]), o))
    taken: dict[str, list[str]] = {o: [] for o in order}
    out: list[str] = []

    def take(o: str) -> bool:
        if len(taken[o]) >= cap[o] or len(taken[o]) >= len(by_obj[o]):
            return False
        c = by_obj[o][len(taken[o])]
        taken[o].append(c)
        out.append(c)
        return True

    _round_robin(k, order, out, n, taken, take)
    remain = {o: len(by_obj[o]) - len(taken.get(o, [])) for o in by_obj}
    return out, {o: len(v) for o, v in taken.items() if v}, remain


def axis_of(combo: str, dataset: str, emb: str) -> str:
    """The ledger axis of a combo tree root — the root part of the run key (train_ready_util.run_key)."""
    try:
        _arm, hand, variant = split_combo(combo, ARM_IDS, HAND_SPEC_DIR)
    except ValueError as e:
        raise SystemExit(f"[select] {e}") from e
    return RR.ledger_key(dataset, emb, variant)


def run_pools(clips: list[str], combo: str, dataset: str, emb: str) -> dict[str, list[str]]:
    """Pools = one run key each (train_ready_util.run_key — the function the training entry calls). Reads every
    clip's tree (qpos_layout, the owner of the run statements); a missing tree raises."""
    axis, root = axis_of(combo, dataset, emb), trees_dir(combo, dataset, emb)
    pools: dict[str, list[str]] = defaultdict(list)
    for c in clips:
        d = root / tree_name(c)
        if not (d / "task_info.json").is_file():
            raise InputMissingError(f"[run_key] tree task_info missing: {d}")
        pools[TRU.run_key(axis, TRU.run_group(qpos_layout.read(str(d))))].append(c)
    return dict(pools)


def partition(pools: dict[str, list[str]], obj_key, shape, n: int, block: int) -> list[dict]:
    """Every pool, packed (bundle_pack.pack) — one dict per bundle: run_key · clips (the run's list: primaries + dups,
    sorted) · prim (primaries) · dups {clip: origin = index of its primary bundle in this list} · keys {clip: object
    key} · families {key: family} · objects {key: clips of it in the run}. `shape` = key_shape(dataset). The
    primaries must equal the input, each once, and every dup must name its primary bundle — otherwise it raises."""
    family, near = shape
    out = []
    for rk in sorted(pools):
        keys = rekey({c: obj_key(c) for c in pools[rk]})
        fam = {k: family(k) for k in sorted(set(keys.values()))}
        try:
            packed = pack(keys, fam, n, block, near=near)
        except ValueError as e:
            raise SystemExit(f"[partition] {rk}: {e}") from e
        base = len(out)
        for b in packed:
            run = sorted(list(b["clips"]) + list(b["dups"]))
            out.append(
                {
                    "run_key": rk,
                    "clips": run,
                    "prim": list(b["clips"]),
                    "dups": {c: base + o for c, o in b["dups"].items()},
                    "keys": {c: keys[c] for c in run},
                    "families": {keys[c]: fam[keys[c]] for c in run},
                    "objects": dict(collections.Counter(keys[c] for c in run)),
                }
            )
    got = [c for b in out for c in b["prim"]]
    want = [c for v in pools.values() for c in v]
    home = {c: i for i, b in enumerate(out) for c in b["prim"]}
    if len(got) != len(set(got)) or set(got) != set(want):
        raise SystemExit(
            f"[partition] sum mismatch: primary clips {len(got)} (unique {len(set(got))}) != pool {len(want)}"
        )
    bad = [(i, c) for i, b in enumerate(out) for c, o in b["dups"].items() if home.get(c) != o or c in b["prim"]]
    if bad:
        raise SystemExit(f"[partition] dup origins differ from the primary bundles {bad[:3]}")
    return out


def adopted_from_ledger(ledger: Path) -> list[str]:
    """ADOPTED clips out of the status ledger (`clip<TAB>status`). A value outside the known terminal
    states raises."""
    from dexengine.placement.ledger import STATES, read_status_ledger  # noqa: PLC0415

    try:
        rows = read_status_ledger(ledger)
    except ValueError as e:
        raise SystemExit(f"[select] {e}") from None
    if not rows:
        raise SystemExit(f"[select] not a status ledger (clip/status columns required): {ledger}")
    bad = sorted({r["status"] for r in rows} - set(STATES))  # an unknown state dies (BAKE_ERROR is in the vocabulary)
    if bad:
        raise SystemExit(f"[select] unknown states {bad} in status ledger: {ledger}")
    out = [r["clip"] for r in rows if r["status"] == "ADOPTED"]
    print(f"[select] status ledger {ledger}: {len(rows)} rows · ADOPTED {len(out)}", flush=True)
    return out


def build_view(view_root: Path, set_name: str, combo: str, clips: list[str], dataset: str, emb: str) -> Path:
    """`<view-root>/<set>/<combo>/<tree>` → symlink to the real tree's absolute path. No silent shortfall.

    Real root = paths.trees_dir(combo, dataset, emb) = the trees layer `data/<ds>/trees/<combo>/<emb>` (every dataset).
    """
    root = trees_dir(combo, dataset, emb).resolve()
    if not root.is_dir():
        raise InputMissingError(f"[view] combo tree root missing: {root}")
    view = view_root / set_name / combo
    if view.exists():
        bad = [q for q in view.iterdir() if not q.is_symlink()]
        if bad:
            raise SystemExit(f"[view] {view} has {len(bad)} non-symlink entries — not a derived view, left untouched")
        for q in view.iterdir():
            q.unlink()
    view.mkdir(parents=True, exist_ok=True)
    for c in clips:
        t = root / tree_name(c)
        if not t.is_dir():
            raise InputMissingError(f"[view] tree missing: {t}")
        (view / t.name).symlink_to(t)
    n = sum(1 for q in view.iterdir() if q.is_symlink() and q.exists())
    if n != len(clips):
        raise SystemExit(f"[view] {view}: valid links {n} != list {len(clips)}")
    return view


def one_pool(pools: dict[str, list[str]], want: str) -> list[str]:
    """The universe of ONE run (single-set modes). One run key → it. Several → `want` names one key, or `largest`
    (an explicit choice — the other pools are printed, never silently dropped); without `want` it raises."""
    desc = " · ".join(f"{k}:{len(v)}" for k, v in sorted(pools.items(), key=lambda kv: -len(kv[1])))
    if want and want != "largest" and want not in pools:  # a named key is checked whatever the pool count
        raise InputMissingError(f"[RUN_GROUP] --run-key {want!r} is not in the universe — present: {desc}")
    if len(pools) == 1:
        ((rk, clips),) = pools.items()
        print(f"[RUN_GROUP] single run key {rk} · {len(clips)}", flush=True)
        return clips
    if not want:
        raise SystemExit(
            f"[RUN_GROUP] the universe has {len(pools)} run keys ({desc}) — one run = one key. Name one "
            "with --run-key <key>|largest, or split every pool with --partition-prefix"
        )
    rk = max(pools, key=lambda k: (len(pools[k]), k)) if want == "largest" else want
    print(
        f"[RUN_GROUP] {desc} → chosen {rk} {len(pools[rk])} "
        f"(the other {sum(map(len, pools.values())) - len(pools[rk])} "
        f"clips are outside this list — --partition-prefix splits every pool)",
        flush=True,
    )
    return pools[rk]


def numbered(pre: Path) -> list[Path]:
    """Every `<prefix>_NN*` product of one partition prefix (lists and their .meta.json) — the glob the consumers
    use."""
    return sorted(pre.parent.glob(f"{pre.name}_[0-9][0-9]*")) if pre.parent.is_dir() else []


def write_partition(a, pool: list[str], src: dict) -> int:
    """--partition-prefix: every run-key pool packed into sets of at most n; <prefix>_NN.txt + .meta.json each and
    <prefix>_manifest.tsv. The union is re-checked against the pool before anything is written."""
    axis = axis_of(a.combo, a.dataset, a.emb)
    pools = run_pools(pool, a.combo, a.dataset, a.emb)
    sets = partition(pools, lambda c: target_object(c, a.dataset, a.emb), key_shape(a.dataset), a.n, a.block)
    pre = Path(a.partition_prefix)
    have = sorted(pre.parent.glob(f"{pre.name}_[0-9][0-9]*.txt")) if pre.parent.is_dir() else []
    if have and not a.overwrite:
        raise SystemExit(
            f"[partition] {len(have)} {pre}_NN.txt files exist already (e.g. {have[0].name}) — frozen lists "
            "are not overwritten. Give a new name or pass --overwrite"
        )
    if have:  # --overwrite = the whole old generation goes: a higher-numbered old list would outlive it
        old = [f for f in numbered(pre) if f.suffix in (".txt", ".json")]
        bad = [f for f in old if f.is_symlink() or not f.is_file()]
        if bad:
            raise SystemExit(
                f"[partition] {len(bad)} old list entries are not regular files (e.g. {bad[0]}) — left untouched"
            )
        for f in old:
            f.unlink()
        print(f"[partition] --overwrite: removed {len(old)} old-generation {pre.name}_NN files", flush=True)
    pre.parent.mkdir(parents=True, exist_ok=True)
    with open(str(pre) + "_manifest.tsv", "w", encoding="utf-8") as mf:
        mf.write("set\tcount\tmain\tdup\trun_key\tper_object\n")
        for i, b in enumerate(sets, 1):
            f = Path(f"{pre}_{i:02d}.txt")
            f.write_text("".join((tree_name(c) if a.tree else c) + "\n" for c in b["clips"]), encoding="utf-8")
            meta = {
                "set": pre.name,
                "bundle": f.stem,
                "axis": axis,
                "combo": a.combo,
                "run_key": b["run_key"],
                "n": len(b["clips"]),
                "n_primary": len(b["prim"]),
                "dups": {c: f"{pre.name}_{o + 1:02d}" for c, o in b["dups"].items()},
                "objects": b["objects"],
                "families": b["families"],
                "rule": PACK_RULE,
                "n_max": a.n,
                "block": a.block,
                "source": src,
            }
            f.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            mf.write(
                f"{f.stem}\t{len(b['clips'])}\t{len(b['prim'])}\t{len(b['dups'])}\t{b['run_key']}\t"
                + " ".join(f"{o}:{c}" for o, c in sorted(b["objects"].items(), key=lambda x: -x[1]))
                + "\n"
            )
    now = sorted(f for f in numbered(pre) if f.suffix == ".txt")
    if len(now) != len(sets):
        raise SystemExit(
            f"[partition] {len(now)} {pre}_NN.txt files ≠ {len(sets)} new sets — an old generation is mixed in"
        )
    nv = 0
    if a.view_root:
        for i, b in enumerate(sets, 1):
            build_view(Path(a.view_root), f"{pre.name}_{i:02d}", a.combo, b["clips"], a.dataset, a.emb)
            nv += len(b["clips"])
        print(f"[partition] {len(sets)} views · {nv} links (= universe {len(pool)})", flush=True)
    nd, sz = sum(len(b["dups"]) for b in sets), [len(b["clips"]) for b in sets]
    print(
        f"[partition] universe {len(pool)} · run keys {len(pools)} → {len(sets)} sets (size {min(sz)}–{max(sz)}) · "
        f"primary {sum(len(b['prim']) for b in sets)} + dups {nd} · rule {PACK_RULE} block {a.block}",
        flush=True,
    )
    return 0


def write_single(a, universe: list[str], exclude: set[str]) -> int:
    """--out: one set (n=0 → the whole single-run universe) of the run-key pool --run-key names."""
    pool = one_pool(run_pools([c for c in universe if c not in exclude], a.combo, a.dataset, a.emb), a.run_key)
    if a.n == 0:
        out = sorted(pool)
        counts = dict(sorted(collections.Counter(target_object(c, a.dataset, a.emb) for c in out).items()))
        remain = {}
        a.n = len(out)
    else:
        out, counts, remain = select(pool, set(), a.n, a.k, a.dataset, a.emb)
    if len(out) != a.n:
        print(
            f"[select] ★ filled only {len(out)} of the requested {a.n} — per-object cap / too few eligible. "
            f"Lower k or n (a short set is never written).",
            file=sys.stderr,
        )
        return 2
    Path(a.out).parent.mkdir(
        parents=True, exist_ok=True
    )  # a new dataset's views layer (data/<ds>/views) may not exist yet
    Path(a.out).write_text("".join((tree_name(c) if a.tree else c) + "\n" for c in out), encoding="utf-8")
    if a.view_root:
        v = build_view(Path(a.view_root), Path(a.out).stem, a.combo, out, a.dataset, a.emb)
        print(f"[select] view {v} · {len(out)} links", flush=True)
    under = [o for o, c in counts.items() if c < a.k]
    print(
        f"[select] {len(out)}/{a.n} · {len(counts)} objects · per object "
        + " ".join(f"{o}:{c}" for o, c in sorted(counts.items(), key=lambda x: -x[1]))
        + f" · universe left {sum(remain.values())}"
        + (f" · ★below k {under}" if under else ""),
        flush=True,
    )
    return 0


def parse_args():
    """The CLI parser and the parsed arguments."""
    ap = argparse.ArgumentParser(description=HELP.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--universe", default="", help="candidate clip list (one per line; tree names allowed)")
    g.add_argument(
        "--status-ledger",
        default="",
        help="status ledger (<chunk>/status_ledger.tsv) — the clips with status ADOPTED are the universe",
    )
    ap.add_argument("--exclude", default="", help="list of clips already used (removed)")
    ap.add_argument(
        "--n",
        type=int,
        required=True,
        help="set size (the cap when partitioning — bundles = ceil(pool/n), balanced). 0 = the whole "
        "universe of one run (--out)",
    )
    ap.add_argument("--k", type=int, default=4, help="--out single set: minimum clips per object (default 4)")
    ap.add_argument(
        "--block",
        type=int,
        default=8,
        help="partition: object block size (big objects are cut into pieces of this size, default 8)",
    )
    ap.add_argument("--out", default="", help="single-set output (or --partition-prefix)")
    ap.add_argument(
        "--partition-prefix",
        default="",
        help="split the universe per run-key pool by the object rule into <prefix>_NN.txt (+.meta.json) "
        "and <prefix>_manifest.tsv",
    )
    ap.add_argument(
        "--run-key",
        default="",
        help="--out single set: name one run key when the universe has several (<key> | largest); otherwise it dies",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="partition: overwrite lists with the same prefix (default = refuse)"
    )
    ap.add_argument("--tree", action="store_true", help="write tree names (_pl_rt)")
    ap.add_argument(
        "--view-root",
        default="",
        help="(opt-in link farm) build a symlink view <view-root>/<set>/<combo>/ per list. Training runs "
        "without views: tree root + DEXENGINE_SEQ_LIST=<list>",
    )
    ap.add_argument(
        "--combo",
        required=True,
        help="combo (+variant) — tree root = paths.trees_dir(combo, --dataset, "
        "--emb). Required: the run key (train_ready_util.run_key) is read from the trees",
    )
    ap.add_argument("--dataset", required=True, help="dataset id (no default)")
    ap.add_argument("--emb", required=True, help="embodiment (no default) — the emb part of the tree root")
    return ap, ap.parse_args()


def main() -> int:
    """Select one set (--out) or partition the universe (--partition-prefix)."""
    ap, a = parse_args()
    if a.status_ledger:
        universe = read_clip_list(
            adopted_from_ledger(Path(a.status_ledger)), dataset=a.dataset, label="select:status-ledger"
        )
    else:
        universe = read_clip_list(a.universe, dataset=a.dataset, label="select:universe")
    exclude = set(read_clip_list(a.exclude, dataset=a.dataset, label="select:exclude")) if a.exclude else set()
    print(obj_key_marker(a.dataset, a.emb, RR.hand_dir()), flush=True)
    if a.partition_prefix:
        srcf = Path(a.status_ledger or a.universe)
        src = {"path": str(srcf), "sha256": hashlib.sha256(srcf.read_bytes()).hexdigest()}
        return write_partition(a, [c for c in universe if c not in exclude], src)
    if not a.out:
        ap.error("--out or --partition-prefix is required")
    return write_single(a, universe, exclude)


if __name__ == "__main__":
    raise SystemExit(main())
