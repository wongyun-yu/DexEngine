#!/usr/bin/env python3
"""Tree record of the record output: the clip folder (<record out>/seq1/<clip>/) records 'which tree, its fingerprint,
its state', and the render finds the tree through that record. No shortcut (symlink): a symlink inside the data root
fails hygiene, and when the tree is rebuilt a shortcut silently follows the new tree.

  state field (only the two STATES; anything else or missing dies):
      verified     the fingerprint is that record's tree (recorded at record time, or at migration when the tree is
                   older than the shortcut).
      unverifiable the tree cannot be confirmed (at migration the tree was rebuilt after the shortcut, or the tree is
                   missing): the old link path link_target is kept as text · no fingerprint · reason why. A render that
                   reads it dies.
  form of the tree root (moves between machines): field root_rel = path relative to the data root
      (paths.main_root) (rel_of: the data layer data/<ds>/… or a data layer inside a registered shadow root
      (DATA_SHADOW.json); outside dies). The reader (record_root) joins it again to this machine's data root. The old
      field root (the writing machine's absolute path) is read by stripping the data root prefix into the same relative
      form: inside this machine's root that relative path, otherwise from the first data layer field (data_tops heads
      + registered datasets; inside that shadow when the tail before it is a registered shadow root on this machine);
      when it cannot be stripped it dies.
  write(d, root, clip)   record <d>/tree_ref.json = {clip, root_rel, sha, state=verified} and a one-clip list
                         <d>/clip.txt (the render entry's DEXENGINE_SEQ_LIST; SEQ_DIR = tree root). Fingerprint =
                         tree_sha (the same fingerprint as the generation turnover ledger). An existing record stays
                         (first record = that record's tree; one --out = one record). Only an old-form shortcut
                         <d>/<clip> without a record dies (migrate first).
  resolve(d) -> (root, list)   no record, a state other than verified (unverifiable or unknown), no fingerprint, or the
                         current tree fingerprint != the record dies (never follows silently).
  meta_tree(seq, clip) -> (tree | None, reason)   the clip tree of the record meta env.sequence_dir; when not there
                         (an earlier job's scratch, another machine) joined again to this machine's data root by the
                         record_root rule (read by the form, geometry and pre-job checks).
  gate(tf, clips)        pre-render gate: for a record with receiving-object slots the clips' trees are resolved with
                         meta_tree; unresolved dies
                         ([TREE_GATE]).
CLI (old-form migration; when no record runs; counts only by default):
  --migrate <record out roots …> [--apply]   writes a record per old shortcut (with the old link path link_target). When
      the tree was rebuilt after the shortcut was made (task_info time > shortcut time) or the tree is missing, state
      unverifiable (a reason instead of a fingerprint): that clip's render dies. Shortcuts are not removed. Counts per
      state are printed.
  --unlink <record out roots …> --ledger <tsv> [--apply]   removes a shortcut only when it has a record (verified or
      unverifiable) pointing at the same tree; when any shortcut has no record (or an unknown state or another tree),
      none is removed. The ledger lines (time, shortcut, target, state, fingerprint, reason) are all written before
      removal. Only symlinks are removed (videos, records and trees stay).
Marker [COLLECT_TREE].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import input_missing

DL = P.data_layout
# tree root = path relative to the data root (new form) · old-form field = "root" (absolute)
ROOT_REL = "root_rel"

NAME, LIST, SEQ = "tree_ref.json", "clip.txt", "seq1"
VERIFIED, UNVERIFIABLE = "verified", "unverifiable"
STATES = (VERIFIED, UNVERIFIABLE)  # every value of the state field; anything else dies
MARK = "[COLLECT_TREE]"
TIME_FMT = "%Y-%m-%dT%H:%M:%S"


def tree_sha(root: Path, clip: str) -> str:
    """sha256 of <root>/<tree name>/task_info.json: the clip's fingerprint at that tree root."""
    f = Path(root) / P.tree_name(clip) / "task_info.json"
    if not f.is_file():
        raise ValueError(f"{f}: no tree task_info.json, cannot fingerprint this clip")
    return hashlib.sha256(f.read_bytes()).hexdigest()


def _die(msg: str) -> None:
    raise SystemExit(f"{MARK} ✗ {msg}")


def _base(clip: str) -> str:
    return clip.removesuffix(P.TREE_SFX)


def _at(t: float | None = None) -> str:
    return time.strftime(TIME_FMT, time.localtime(t))


def _layer_at(parts: tuple[str, ...]) -> int | None:
    """First place among the relative parts where a data layer head (data_tops fields + registered datasets) appears;
    None if none."""
    tops = DL.data_tops()
    return next((i for i in range(len(parts) - 1) if parts[i] in tops and parts[i + 1] in DL.DATASETS), None)


def rel_of(root: Path) -> str:
    """Writer side: the tree root relative to the data root (paths.main_root). Rule: the root is inside the data root
    and its relative parts are a data layer (a data layer head first), or the part before the data layer head is a
    registered shadow root (SHADOW_FILE: the aug/shadow of a record work piece). Otherwise it dies (no record that
    cannot be resolved after a machine move)."""
    m = P.main_root().resolve()
    for r in (Path(os.path.abspath(root)), Path(root).resolve()):
        try:
            parts = r.relative_to(m).parts
            break
        except ValueError:
            continue
    else:
        _die(
            input_missing(
                f"tree root {root} is outside the data root {m}: it cannot be joined again on another machine"
            )
        )
    i = _layer_at(parts)
    if i is None or (i > 0 and not (m.joinpath(*parts[:i]) / DL.SHADOW_FILE).is_file()):
        _die(
            f"tree root {root} is outside the data layer (under the data root {'/'.join(parts)}; a data layer head "
            "must come first or sit right under a registered shadow root)"
        )
    return "/".join(parts)


def record_root(doc: dict, rec: Path) -> Path:
    """Reader side: join the tree root of record `rec` again to this machine's data root (new form root_rel, old form
    root with the prefix stripped). When it cannot be stripped or the relative form is outside the rule it dies; it
    never silently becomes 'no tree' (when the tree really is missing the fingerprint check says so)."""
    m = P.main_root().resolve()
    if ROOT_REL in doc:
        rel = Path(doc[ROOT_REL])
        if rel.is_absolute() or ".." in rel.parts or _layer_at(rel.parts) is None:
            _die(f"{rec}: root_rel {doc[ROOT_REL]!r} is not a data-layer path relative to the data root")
        return m / rel
    if "root" not in doc:
        _die(input_missing(f"{rec}: no tree root field (root_rel, root)"))
    old = Path(os.path.abspath(doc["root"]))
    try:
        return m / old.relative_to(m)
    except ValueError:
        i = _layer_at(old.parts)
        if i is None:
            _die(
                input_missing(
                    f"{rec}: the data root prefix cannot be stripped from the old-form tree root {old} (no "
                    "data layer head data_tops + registered dataset field): this tree cannot be found on "
                    "this machine"
                )
            )
        # when the part before the data layer is a shadow root: the registered shadow at the same relative place on this
        #   machine (longest tail)
        for j in range(1, i):
            sh = m.joinpath(*old.parts[j:i])
            if (sh / DL.SHADOW_FILE).is_file():
                return sh.joinpath(*old.parts[i:])
        return m.joinpath(*old.parts[i:])


def _put(d: Path, doc: dict) -> None:
    body = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    for name, text in ((NAME, body), (LIST, _base(doc["clip"]) + "\n")):
        tmp = d / f".{name}.tmp"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, d / name)


def write(d: Path, root: Path, clip: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    if not (d / NAME).is_file():
        if (d / clip).is_symlink():
            _die(f"{d / clip}: old-form shortcut without a record; migrate first (tree_ref.py --migrate)")
        try:
            sha = tree_sha(Path(root), _base(clip))
        except ValueError as e:
            _die(f"{clip}: {e}")
        _put(d, {"clip": clip, ROOT_REL: rel_of(root), "sha": sha, "state": VERIFIED, "at": _at()})
    return d


def sha_of(d: Path) -> str:
    """The record's tree fingerprint (verified only; unverifiable or missing fingerprints die). Read by the piece's
    evidence field (collect.tree_sha)."""
    doc = json.loads((d / NAME).read_text(encoding="utf-8"))
    if doc.get("state") != VERIFIED or not doc.get("sha"):
        _die(
            f"{d / NAME}: state {doc.get('state')!r} · fingerprint {doc.get('sha')!r}: only verified fingerprints are "
            f"carried"
        )
    return doc["sha"]


def resolve(d: Path) -> tuple[Path, Path]:
    rec = d / NAME
    if not rec.is_file():
        _die(input_missing(f"{rec} is missing: which tree this clip was recorded with is unknown"))
    doc = json.loads(rec.read_text(encoding="utf-8"))
    root, clip, sha, state = record_root(doc, rec), doc["clip"], doc.get("sha"), doc.get("state")
    if state == UNVERIFIABLE:
        _die(
            f"{rec}: tree unverifiable: old shortcut {doc.get('link_target')} · {doc.get('why') or 'no reason'} "
            "(no render from this record; record it again)"
        )
    if state != VERIFIED:
        _die(f"{rec}: state field {state!r} is not in {STATES}")
    if not sha:
        _die(input_missing(f"{rec}: state verified but no fingerprint"))
    try:
        now = tree_sha(root, _base(clip))
    except ValueError as e:
        _die(f"{rec}: {e}")
    if now != sha:
        _die(
            f"{rec}: current tree fingerprint {now[:12]} != record {sha[:12]}: the tree was rebuilt after the record "
            f"(record it again)"
        )
    lst = d / LIST
    if not lst.is_file() or lst.read_text(encoding="utf-8").split() != [_base(clip)]:
        _die(f"{lst}: the one-clip list differs from the record ({_base(clip)})")
    print(f"{MARK} {clip} tree root {root} · fingerprint {sha[:12]} equal", flush=True)
    return root, lst


def meta_tree(seq: str | None, clip: str) -> tuple[Path | None, str]:
    """Record meta env.sequence_dir (tree root: old form absolute, new form data-root relative root_rel) + clip ->
    (this machine's clip tree | None, reason). When the absolute path holds a tree (task_info.json) it is used as is
    (same job, same machine). Otherwise (scratch of an earlier job that rendered the piece, another machine) it is
    joined again to this machine's data root by the record_root rule (no second rule); a rejoin prints the marker
    '{MARK} rejoined'. When the rejoined place has no tree either, or it cannot be joined: None (the caller dies
    loudly)."""
    if not seq:
        return None, "no sequence_dir"
    p = Path(seq)
    if p.is_absolute() and (p / clip / "task_info.json").is_file():
        return p / clip, ""
    try:
        r = record_root({"root": seq} if p.is_absolute() else {ROOT_REL: seq}, Path("meta env.sequence_dir"))
    except SystemExit as e:
        return None, str(e)
    if not (r / clip / "task_info.json").is_file():
        return None, f"no task_info.json in the rejoined {r / clip} either"
    print(f"{MARK} rejoined {seq} -> {r} ({clip})", flush=True)
    return r / clip, ""


def gate(tf, clips, pfx: str | None = None):
    """Pre-render gate: when the stage-1 record tf (a piece an earlier job left, seeds included) has receiving-object
    slot entities, the clips' trees are resolved by the same rule as the form stage (meta_tree). Unresolved dies before
    the render (1-2 h per clip) · marker [TREE_GATE] · returns tf. pfx = slot name head (default owner
    store.form.support_obj_prefix)."""
    if pfx is None:
        from dexengine.record.store import form as CF

        pfx = CF.support_obj_prefix()
    if not any(str(e.get("name", "")).startswith(pfx) for e in tf.entities):
        return tf
    seq = (tf.meta.get("env") or {}).get("sequence_dir")
    got = {c: meta_tree(seq, c) for c in clips}
    bad = [f"{c} · {why}" for c, (t, why) in got.items() if t is None]
    if bad:
        _die(
            f"[TREE_GATE] trees of {len(bad)}/{len(clips)} receiving-slot clips cannot be resolved on this machine "
            f"(record {seq}): stopped before the render · {bad[:2]}"
        )
    print(f"{MARK} [TREE_GATE] receiving-object slot record · {len(clips)} clip trees resolved", flush=True)
    return tf


def _target(link: Path) -> Path:
    return link.parent / os.readlink(link)


def old_links(roots: list[Path]) -> list[Path]:
    """Every old-form shortcut <out>/seq1/<clip>/<clip> (walked without following symlinks)."""
    out = []
    for r in roots:
        for top, dirs, _files in os.walk(r):
            if Path(top).name == SEQ:
                out += [Path(top) / c / c for c in sorted(dirs) if (Path(top) / c / c).is_symlink()]
                dirs[:] = []
    return sorted(out)


def migrate_doc(link: Path) -> dict:
    root, clip = _target(link).parent, link.name
    ti, t_link = root / P.tree_name(clip) / "task_info.json", link.lstat().st_mtime
    doc = {
        "clip": clip,
        ROOT_REL: rel_of(root),
        "sha": None,
        "state": UNVERIFIABLE,
        "at": _at(),
        "migrated_from": str(link),
        "link_target": str(_target(link)),
        "link_at": _at(t_link),
    }
    if not ti.is_file():
        doc["why"] = f"no tree ({ti})"  # stored in tree_ref.json (data)
    elif ti.stat().st_mtime > t_link:
        # data (stored in tree_ref.json)
        doc["why"] = f"tree rebuilt after collection (task_info {_at(ti.stat().st_mtime)} > link {doc['link_at']})"
    else:
        doc.update(sha=tree_sha(root, _base(clip)), state=VERIFIED)
    return doc


def migrate(roots: list[Path], apply: bool) -> int:
    n = done = 0
    by = dict.fromkeys(STATES, 0)
    for link in old_links(roots):
        n += 1
        if (link.parent / NAME).is_file():
            done += 1
            continue
        doc = migrate_doc(link)
        by[doc["state"]] += 1
        if apply:
            _put(link.parent, doc)
    print(
        f"{MARK} migrate shortcuts {n} · already recorded {done} · new records {n - done} (verified "
        f"{by[VERIFIED]} · unverifiable {by[UNVERIFIABLE]}) · "
        f"{'written' if apply else 'counted only (--apply writes)'}",
        flush=True,
    )
    return 0


def unlink(roots: list[Path], ledger: Path, apply: bool) -> int:
    bad, rows = [], []
    for link in old_links(roots):
        rec = link.parent / NAME
        doc = json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else None
        same = doc is not None and record_root(doc, rec).resolve() == _target(link).parent.resolve()
        if doc is None or doc.get("state") not in STATES or not same or doc["clip"] != link.name:
            bad.append(str(link))
            continue
        rows.append((_at(), str(link), str(_target(link)), doc["state"], doc.get("sha") or "-", doc.get("why") or ""))
    if bad:
        print(
            f"{MARK} ✗ {len(bad)} shortcuts without a record, with an unknown state or pointing at another tree: "
            f"none removed: {bad[:3]}"
        )
        return 1
    by = {s: sum(r[3] == s for r in rows) for s in STATES}
    print(
        f"{MARK} unlink shortcuts {len(rows)} (verified {by[VERIFIED]} · unverifiable {by[UNVERIFIABLE]}) · "
        f"ledger {ledger} · {'removed' if apply else 'counted only (--apply removes)'}"
    )
    if apply and rows:
        new = not ledger.is_file()
        with ledger.open("a", encoding="utf-8") as f:
            if new:
                f.write("at\tlink\ttarget\tstate\tsha\twhy\n")
            f.writelines("\t".join(r) + "\n" for r in rows)
            f.flush()
            os.fsync(f.fileno())
        for _t, link, *_ in rows:
            if not Path(link).is_symlink():  # symlinks only; stop when it changed after the ledger
                _die(f"{link} is no longer a shortcut: not removed (check the remaining lines of ledger {ledger})")
            os.unlink(link)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--migrate", nargs="+", type=Path, help="record out roots: old shortcuts to records")
    g.add_argument("--unlink", nargs="+", type=Path, help="record out roots: remove only shortcuts with a record")
    ap.add_argument("--ledger", type=Path, help="--unlink: ledger written before removing (tsv)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    if a.migrate:
        return migrate(a.migrate, a.apply)
    if not a.ledger:
        ap.error("--unlink needs --ledger <tsv> (the ledger is written first)")
    return unlink(a.unlink, a.ledger, a.apply)


if __name__ == "__main__":
    sys.exit(main())
