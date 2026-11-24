#!/usr/bin/env python3
"""Per-hand RL bundle — the processed data an RL-only user (train · judge · render) unpacks at a data root.

What goes in is not stated here; it is read from the statement of what a campaign bundle reads (run_inputs.py):
  set = run_inputs.manifest_level(man)       every folded ledger of the hand (a test bundle: the selected bundles'
                                             axes only)
      ∪ run_inputs.dataset_level(man, sel)   dataset-level inputs (coords.dataset_files — OakInk2 part tree)
      ∪ Stater.bundle(b) per selected bundle training entry · judge (same entry) · verify inputs — trees · wrist/arm
                                             IK caches · objects · robot USD · roster ledger · layer-root records
      ∪ Stater.render(b) per selected bundle render plate USD · tree F0_REPAIR.json (when present)
Every path is relative to the data root (DEXENGINE_MAIN_ROOT — paths.main_root) and is packed at that same relative
place, so the receiver unpacks at its data root and points DEXENGINE_MAIN_ROOT there (no symlinks).

Build (`--hand --manifest --out`): the hand must be the manifest's (no default); every stated file must be a regular
file under the data root, else it dies naming them and writes nothing; the data root must pass the robot gate the
receiver's entry runs first (gate_check — a stale derived USD record would ship a bundle that dies at spawn).
Completion is a count: the trees in the set (reader entry:layout) = the trees of the selected bundles' clips derived
from the manifest (paths.trees_dir · tree_name — set difference 0 both ways), and for the whole manifest that count =
its primary clips (totals.target). The archive is then read back as a stream: its members = the set, each member's
size = the size stated at packing, the source files unchanged since (size · mtime), and SHA256SUMS is written from
the archive bytes; a full set's axis ledgers = the manifest's sha256 (before and after packing). Any difference dies
(`[RL_BUNDLE] FAIL`).
Written to <out>/rl_bundle_<hand>_<manifest fingerprint 12>[_test]/:
  README.md       contents · unpack place · checksums · how to check (readme.py)
  bundle.json     hand · arm · manifest fingerprint · bundles · counts by kind/reader · trees · bytes · parts · git
  FILES.tsv       path · kind · reader · bytes (relative to the data root)
  SHA256SUMS      sha256 of every file, relative to the data root (`cd <root> && sha256sum -c`)
  PARTS.sha256    sha256 of every archive part (`sha256sum -c` in the bundle folder)
  manifest/<hand>/manifest.json + lists/<id>.{txt,tsv}   the campaign manifest (DEXENGINE_SEQ_LIST = lists/<id>.txt)
  data.tar.zst.part-NNN   tar (relative paths, dereferenced) | zstd, split at --part-size
Check (`--check <bundle folder> --root <data root>`, receiver side): every SHA256SUMS path present and equal, count =
bundle.json files, trees = bundle.json trees → `[RL_BUNDLE_CHECK] PASS …` rc 0, else FAIL rc 1 (names the first ones).
Plan (`--plan`): the set and its sizes only (FILES.tsv + plan.json) — no archive.

Usage:
  DEXENGINE_ARM=rby1 python3 -m dexengine.placement.bundle.rl_bundle --hand <hand> \\
      --manifest <dir>/<hand>/manifest.json --out <dir> [--bundle <id> …] [--part-size 4G] [--plan]
  python3 -m dexengine.placement.bundle.rl_bundle --check <bundle folder> --root <data root>
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import contextlib
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.core.hashing import sha256_file
from dexengine.placement.bundle import readme as RM
from dexengine.placement.bundle import run_inputs as RI
from dexengine.placement.bundle.archive import (  # noqa: F401
    ARCHIVE,
    CHUNK,
    TAG,
    ZSTD_LEVEL,
    die,
    pack,
    read_archive,
    sh,
)

FORMAT = 1
CHECK_TAG = "[RL_BUNDLE_CHECK]"
TREE_READER = "entry:layout"  # run_inputs: one tree task_info.json per clip of a bundle
HASH_STEP = "dexengine.placement.bundle.rl_bundle"  # capacity.NEEDS key (the input hash threads)


def load_manifest(path: Path, hand: str) -> dict:
    """The campaign manifest at `path`; its hand must be `hand` and its arm DEXENGINE_ARM, else it dies."""
    man = json.loads(path.read_text(encoding="utf-8"))
    if man.get("hand") != hand:
        die(
            f"--hand {hand} ≠ manifest hand {man.get('hand')!r} ({path}) — the hand is an explicit input and must "
            "match the manifest"
        )
    arm = os.environ.get("DEXENGINE_ARM")
    if arm != man.get("arm"):
        die(
            f"DEXENGINE_ARM={arm!r} ≠ manifest arm {man.get('arm')!r} — set it so the statement (run_inputs) is "
            "derived with the launch's arm"
        )
    return man


def select(man: dict, ids: list[str]) -> list[dict]:
    """The bundles packed: all (no --bundle), or exactly the named ones (unknown or repeated id dies)."""
    if not ids:
        return list(man["bundles"])
    byid = {b["id"]: b for b in man["bundles"]}
    bad = sorted({i for i in ids if i not in byid} | {i for i, n in collections.Counter(ids).items() if n > 1})
    if bad:
        die("bundles not in the manifest or given twice", bad)
    return [byid[i] for i in ids]


def statement(man: dict, bundles: list[dict], full: bool) -> dict[str, tuple[str, str]]:
    """{relative path: (kind, reader)} — run_inputs' statement only (module doc). First reader wins on a shared path.
    full = the whole manifest (every axis ledger); a test bundle states only its bundles' axis ledgers."""
    st = RI.Stater(man)
    out = dict(RI.manifest_level(man, None if full else {b["axis"] for b in bundles}))
    for r, kr in RI.dataset_level(man, bundles).items():  # dataset-level inputs (outside the bundle digests)
        out.setdefault(r, kr)
    for b in bundles:
        for part in (st.bundle(b), st.render(b)):
            for r, kr in part.items():
                out.setdefault(r, kr)
    return dict(sorted(out.items()))


def expected_trees(bundles: list[dict]) -> set[str]:
    """The tree folders of the selected bundles' clips, derived from the manifest alone (an independent derivation of
    what the statement's entry:layout items must cover)."""
    root = P.main_root()
    return {
        os.path.relpath(P.trees_dir(b["combo"], b["dataset"], b["emb"]) / P.tree_name(c), root)
        for b in bundles
        for c in b["clips"]
    }


def count_check(stmt: dict, man: dict, bundles: list[dict], full: bool) -> int:
    """Trees in the statement = trees derived from the manifest (and = totals.target for a full set) → their count."""
    got = {os.path.dirname(r) for r, (_k, rd) in stmt.items() if rd == TREE_READER}
    want = expected_trees(bundles)
    if got != want:
        die(
            f"tree set difference — statement only {len(got - want)} · manifest only {len(want - got)}",
            sorted(got ^ want),
        )
    if full:
        prim = {c for b in bundles for c in b["clips"] if c not in b["dups"]}
        if not (len(got) == len(prim) == man["totals"]["target"]):
            die(f"trees {len(got)} · primary {len(prim)} · manifest target {man['totals']['target']} differ")
    print(
        f"{TAG} trees {len(got)} = manifest clips {len(want)} · difference 0"
        + (f" · primary = target {man['totals']['target']}" if full else " · test bundle (partial)"),
        flush=True,
    )
    return len(got)


#: The robot gate, run in a child interpreter: one GATE_OK line per USD variant that passes.
GATE_CODE = """
import os, sys
from dexengine.robots import selection as rs
arm, hand, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
for v in variants:
    os.environ["DEXENGINE_COMBO_USD_VARIANT"] = v
    rs.assert_generation_coherent(arm, hand, rs.USD_CACHE_DIR)
    print(f"GATE_OK {v}", flush=True)
"""


def gate_check(man: dict, bundles: list[dict]) -> list[str]:
    """The robot gate the receiver's entry runs first (selection.assert_generation_coherent — a derived USD's record ↔
    its source md5 ↔ this tree's generator ↔ this tree's arm collider digest) for every robot variant the bundle
    carries: the table's USD variant of each selected bundle's launch env (run_inputs.Stater.switches) and the plate
    (arms.plate_variant — the one every process spawns; a launch env naming another one is caught here too). Run in a
    child under the retarget python (paths.require_retarget_py — the digest needs its mesh libraries) on this data
    root. A root that fails is not packed."""
    from dexengine.core import arms as AS  # noqa: PLC0415

    st = RI.Stater(man)
    vs = sorted(
        {st.switches(b)["DEXENGINE_COMBO_USD_VARIANT"] for b in bundles} | {AS.plate_variant(man["arm"], man["hand"])}
    )
    r = subprocess.run(
        [P.require_retarget_py(), "-c", GATE_CODE, man["arm"], man["hand"], *vs],
        cwd=P.WT,
        capture_output=True,
        text=True,
        env=dict(os.environ, DEXENGINE_MAIN_ROOT=str(P.main_root())),
    )
    ok = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("GATE_OK ")]
    if r.returncode != 0 or ok != vs:
        die(
            f"this data root fails the robot gate (assert_generation_coherent) — passed {ok} / {vs} · the receiver's "
            "entry would die before spawn (a stale derived USD: run its bake command first)",
            list((r.stderr or "").strip().splitlines()[-6:]),
        )
    print(f"{TAG} robot gate passed {vs}", flush=True)
    return vs


def stat_all(root: Path, rels) -> dict[str, tuple[int, int]]:
    """{rel: (size, mtime_ns)} of the source files (followed — what the readers open). A missing or non-regular
    stated file dies with all of them named (nothing written)."""
    out, miss = {}, []
    for r in rels:
        try:
            s = (root / r).stat()
        except FileNotFoundError:
            miss.append(r)
            continue
        if not (root / r).is_file():
            miss.append(f"{r} (not a regular file)")
            continue
        out[r] = (s.st_size, s.st_mtime_ns)
    if miss:
        die(input_missing(f"{len(miss)} stated inputs are missing under data root {root} — not packing"), miss)
    return out


def counts(stmt: dict, size: dict) -> dict:
    """File and byte counts by kind, file counts by reader."""
    by_kind, by_reader = collections.Counter(), collections.Counter()
    b_kind = collections.Counter()
    for r, (k, rd) in stmt.items():
        by_kind[k] += 1
        by_reader[rd] += 1
        b_kind[k] += size[r][0]
    return {
        "kinds": {k: {"files": by_kind[k], "bytes": b_kind[k]} for k in sorted(by_kind)},
        "readers": dict(sorted(by_reader.items())),
    }


def git_state() -> dict:
    """Commit, branch and dirty flag of the code tree."""

    def g(*a):
        return subprocess.run(["git", "-C", str(P.WT), *a], capture_output=True, text=True).stdout.strip()

    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(g("status", "--porcelain", "--untracked-files=no")),
    }


def write_files_tsv(path: Path, stmt: dict, size: dict) -> None:
    """FILES.tsv — path · kind · reader · bytes."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("path\tkind\treader\tbytes\n")
        f.writelines(f"{r}\t{k}\t{rd}\t{size[r][0]}\n" for r, (k, rd) in stmt.items())


def name(hand: str, fingerprint: str, full: bool) -> str:
    """Bundle folder name: rl_bundle_<hand>_<fingerprint 12> · a --bundle (test) pack → …_test."""
    return f"rl_bundle_{hand}_{fingerprint[:12]}" + ("" if full else "_test")


@contextlib.contextmanager
def gone_on_fail(out: Path):
    """The folder the caller made (new or empty before) is removed when the body dies · success leaves it."""
    try:
        yield out
    except BaseException:
        if out.is_dir():
            shutil.rmtree(out)
            print(f"{TAG} failed — removed the half-built {out.name}", flush=True)
        raise


def disagree(man: dict, root: Path, sha_of: dict[str, str] | None = None) -> list[str]:
    """The manifest's axis source files (axes[].files) whose sha256 in the folder is not the one the manifest states
    — a ledger folded between the cut and the pack. sha_of = {path under root: sha256} of the packed archive; None =
    the files under root now (the check before packing)."""
    bad = []
    for a in man["axes"]:
        for f in a["files"]:
            try:
                rel = Path(f["path"]).relative_to(root).as_posix()
            except ValueError:
                bad.append(f"{a['axis']}: {f['path']} — outside data root {root}")
                continue
            if sha_of is None:
                got = hashlib.sha256((root / rel).read_bytes()).hexdigest() if (root / rel).is_file() else None
            else:
                got = sha_of.get(rel)
            if got != f["sha256"]:
                bad.append(f"{a['axis']}: {rel} — manifest {f['sha256'][:10]} · folder {(got or 'missing')[:10]}")
    return bad


def agree(man: dict, root: Path, sha_of: dict[str, str] | None, full: bool) -> None:
    """disagree → die (before packing over the files now · after, over the archive read back) — a full set only (a
    test set is not checked against its manifest)."""
    if full and (bad := disagree(man, root, sha_of)):
        die("folder ledgers ≠ manifest — a ledger was folded after the cut · cut again from the current sources", bad)


def copy_manifest(man_path: Path, man: dict, bundles: list[dict], dst: Path) -> None:
    """manifest.json + every list file of the selected bundles, at <dst>/<hand>/."""
    src = man_path.parent
    d = dst / man["hand"]
    (d / "lists").mkdir(parents=True)
    shutil.copy2(man_path, d / "manifest.json")
    miss = [f for b in bundles for f in (b["list"], b["table"]) if not (src / f).is_file()]
    if miss:
        die(input_missing(f"manifest list files missing ({src})"), miss)
    for b in bundles:
        for f in (b["list"], b["table"]):
            shutil.copy2(src / f, d / f)


def build(a) -> int:
    """Build (or plan) one bundle folder (module doc)."""
    man = load_manifest(a.manifest, a.hand)
    bundles = select(man, a.bundle)
    full = not a.bundle
    root = P.main_root()
    print(
        f"{TAG} hand {a.hand} · manifest {a.manifest} · bundles {len(bundles)}/{len(man['bundles'])} · "
        f"data root {root}",
        flush=True,
    )
    stmt = statement(man, bundles, full)
    n_trees = count_check(stmt, man, bundles, full)
    size = stat_all(root, stmt)
    cnt = counts(stmt, size)
    total = sum(s for s, _m in size.values())
    gate = [] if a.plan else gate_check(man, bundles)  # a plan only sizes; a bundle must pass the robot gate
    nm = name(a.hand, man["fingerprint"], full)
    out = a.out / nm
    if out.exists() and any(out.iterdir()):
        die(f"{out} exists — not overwriting (remove it and run again)")
    out.mkdir(parents=True, exist_ok=True)
    with gone_on_fail(out):
        return _fill(a, man, bundles, root, stmt, size, cnt, total, n_trees, gate, nm, full, out)


def _fill(a, man, bundles, root, stmt, size, cnt, total, n_trees, gate, name, full, out) -> int:
    write_files_tsv(out / "FILES.tsv", stmt, size)
    meta = {
        "format": FORMAT,
        "name": name,
        "hand": a.hand,
        "arm": man["arm"],
        "scope": "full" if full else "test",
        "manifest": {
            "path": str(a.manifest),
            "fingerprint": man["fingerprint"],
            "sha256": sha256_file(a.manifest),
            "bundles": len(man["bundles"]),
            "target": man["totals"]["target"],
        },
        "bundles": [b["id"] for b in bundles],
        "files": len(stmt),
        "bytes": total,
        "trees": n_trees,
        "counts": cnt,
        "robot_gate": gate,
        "data_root": str(root),
        "git": git_state(),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if a.plan:
        (out / "plan.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{TAG} PLAN files {len(stmt):,} · {RM.gib(total)} · trees {n_trees:,} → {out}", flush=True)
        for k, v in cnt["kinds"].items():
            print(f"  {k}\t{v['files']}\t{RM.gib(v['bytes'])}")
        return 0
    rels = list(stmt)
    agree(man, root, None, full)
    parts = pack(root, rels, out, a.part_size)
    print(
        f"{TAG} {len(parts)} parts · {RM.gib(sum(p.stat().st_size for p in parts))} — reading back to compare",
        flush=True,
    )
    arc = read_archive(parts)
    if set(arc) != set(rels):
        die(
            f"archive members ≠ statement — archive only {len(set(arc) - set(rels))} · statement only "
            f"{len(set(rels) - set(arc))}",
            sorted(set(arc) ^ set(rels)),
        )
    bad = [r for r in rels if arc[r][0] != size[r][0]]
    bad += [f"{r} (changed while packing)" for r, s in stat_all(root, rels).items() if s != size[r]]
    if bad:
        die("sizes or modification times changed while packing — not packing data that is moving", bad)
    agree(man, root, {r: arc[r][1] for r in rels}, full)
    (out / "SHA256SUMS").write_text("".join(f"{arc[r][1]}  {r}\n" for r in rels), encoding="utf-8")
    copy_manifest(a.manifest, man, bundles, out / "manifest")
    meta["parts"] = [{"name": p.name, "bytes": p.stat().st_size} for p in parts]
    meta["part_size"] = a.part_size
    meta["compressor"] = f"zstd -{ZSTD_LEVEL}"
    (out / "PARTS.sha256").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in parts), encoding="utf-8")
    (out / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "README.md").write_text(RM.readme(meta, man, bundles), encoding="utf-8")
    print(
        f"{TAG} DONE files {len(arc):,} = statement {len(rels):,} · difference 0 · trees {n_trees:,} · "
        f"{RM.gib(total)} → {out}",
        flush=True,
    )
    return 0


def check(bdir: Path, root: Path) -> int:
    """Receiver side: SHA256SUMS paths present and equal under `root`, count = bundle.json, trees = bundle.json."""
    meta = json.loads((bdir / "bundle.json").read_text(encoding="utf-8"))
    rows = []
    for ln in (bdir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (.+)", ln)
        if not m:
            raise SystemExit(f"{CHECK_TAG} FAIL SHA256SUMS line form: {ln!r}")
        rows.append((m.group(2), m.group(1)))
    tsv = [ln.split("\t") for ln in (bdir / "FILES.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    trees = {os.path.dirname(r[0]) for r in tsv if r[2] == TREE_READER}
    miss, diff = [], []

    def one(row):
        rel, want = row
        p = root / rel
        if not p.is_file():
            return rel, "missing"
        return rel, "" if sha256_file(p) == want else "differs"

    from dexengine.core import capacity  # noqa: PLC0415

    with capacity.sized(HASH_STEP, len(rows)) as cap, cf.ThreadPoolExecutor(cap.workers) as ex:
        for rel, why in ex.map(one, rows):
            (miss if why == "missing" else diff if why == "differs" else []).append(rel)
    n_ok = len(rows) - len(miss) - len(diff)
    probs = []
    if len(rows) != meta["files"] or len(tsv) != meta["files"]:
        probs.append(f"list counts SHA256SUMS {len(rows)} · FILES.tsv {len(tsv)} ≠ bundle.json {meta['files']}")
    if len(trees) != meta["trees"]:
        probs.append(f"trees {len(trees)} ≠ bundle.json {meta['trees']}")
    if miss or diff or probs:
        print(
            f"{CHECK_TAG} FAIL {meta['name']} — ok {n_ok}/{meta['files']} · missing {len(miss)} · differs {len(diff)}"
            + "".join(f" · {x}" for x in probs)
        )
        for rel in miss[:20]:
            print(f"  missing {rel}")
        for rel in diff[:20]:
            print(f"  differs {rel}")
        return 1
    print(
        f"{CHECK_TAG} PASS {meta['name']} — files {n_ok}/{meta['files']} sha256 equal · trees {len(trees)} · "
        f"root {root}"
    )
    return 0


def main(argv=None) -> int:
    """CLI: build / --plan / --check (module doc)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hand")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--bundle", action="append", default=[], help="test bundle: only these (repeatable)")
    ap.add_argument("--part-size", default="4G", help="part size (split -b form · default 4G)")
    ap.add_argument("--plan", action="store_true", help="the set and its sizes only (no archive)")
    ap.add_argument("--check", type=Path, help="receiver check: bundle folder")
    ap.add_argument("--root", type=Path, help="data root of --check")
    a = ap.parse_args(argv)
    if a.check:
        if not a.root or a.hand or a.manifest or a.out or a.bundle or a.plan:
            ap.error("--check takes --root only")
        return check(a.check, a.root)
    if not (a.hand and a.manifest and a.out) or a.root:
        ap.error("--hand · --manifest · --out are required (--root is for --check only)")
    if not re.fullmatch(r"[1-9][0-9]*[KMGT]?", a.part_size):
        ap.error(f"--part-size {a.part_size!r} — e.g. 4G · 512M")
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
