#!/usr/bin/env python3
"""Append a recorded piece (<piece>/dataset, written by store.store) to the final LeRobot dataset of a hand x dataset
(CPU, pyarrow).

- Lock: piece.final_lock (flock on the dataset folder's parent, shared with the driver's merge step and with retext).
  merge_piece does not take it, it only proves it is held (merge_check.require_lock); the CLI takes it. Paths, forms
  and checks: store.merge_check.
- Input checks happen before any write. Idempotent via meta/pieces.jsonl: the same id and hash is skipped; the same id
  with a different hash, or a clip that is already present, is an error. Two per-clip exceptions keep the final copy
  and record clip_skipped in the pieces line: an augmented same-input fallback (SAME_FALLBACK_RULE) and an original
  re-render from the same input (ORIG_DUP_RULE: same spec base, policy, card and look version). An allow-listed aug
  clip takes only new placement variants (store.merge_topup, pieces line topup · topup_dropped). Other overlaps die.
- Robot self-penetration (J-self-pen, store.merge_self_pen) and the both-dynamic pairs' delivered visual overlap
  (S-dyn-pairs drop (b), store.merge_dyn_pairs): measure_piece measures the piece before the lock; merge_piece checks
  the piece is unchanged and leaves out every trajectory over either line (pieces line self_pen_dropped ·
  dyn_pair_dropped; a piece with nothing left still gets its line).
- Numbering: episodes continue at total_episodes; chunk = index // chunks_size; the three index columns of each parquet
  are rewritten (task_index from the piece's task texts); episodes_stats lines are renumbered; every other file is
  hard-linked (copied on EXDEV; one copy per source inode). Each line gets piece_id, src_episode_index and piece_tasks
  (the piece's task texts = source labels; retext replaces them with instructions later).
- Append-only metadata (cost independent of the final's size): episodes, pieces, episodes_stats and superseded are only
  appended to. The journal (.merge.journal, version 2) records the four files' previous lengths and tail
  fingerprints plus hard links to the previous versions of the two small metas (info, tasks); after the post-check
  (tails only) passes, journal.json is unlinked (= commit). On failure the files are truncated back and the merge
  dies; a leftover journal is undone by the next call. Duplicate and supersede decisions use the index
  (meta/.merge_index.json: cards and key sets, merge_check.index_open); if it disagrees with the final, it is rebuilt
  from a full read ([MERGE_INDEX]). Pieces are never deleted here (the caller's job).
Callers: measured = measure_piece(<piece>/dataset) outside the lock, then merge_piece(final, piece, id, measured)
under piece.final_lock (the CLI and piece.merge_step do both).
Usage: python -m dexengine.record.store.merge append --final <final> --piece <piece>/dataset --piece-id <id>
       python -m dexengine.record.store.merge verify|status --final <final>    # verify rc 0 = pass
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import shutil
import time
from pathlib import Path

from dexengine.core import config
from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.record import driver_check as PC
from dexengine.record import episodes as EPI
from dexengine.record import piece as CP
from dexengine.record.store import form as FM
from dexengine.record.store import lerobot as CL
from dexengine.record.store import merge_check as MC
from dexengine.record.store import merge_dyn_pairs as DPM
from dexengine.record.store import merge_overlap as MO
from dexengine.record.store import merge_rows as MR
from dexengine.record.store import merge_self_pen as SPM
from dexengine.record.store import merge_topup as TU
from dexengine.record.store import store as ST
from dexengine.record.store.merge_check import DATA, INFO, JFILE, JOURNAL, META, die, jsonl, line, rjson, say
from dexengine.record.store.merge_index import (
    INDEX,
    append_bytes,
    index_add,
    index_empty,
    index_eps,
    index_open,
    read_head,
)
from dexengine.record.store.parquet import read_table

EPISODES, TASKS, SOURCE, PIECES, STATS = MC.EPISODES, MC.TASKS, MC.SOURCE, MC.PIECES, MC.STATS
SUPERSEDED = MC.SUPERSEDED


def read_piece(piece: Path) -> dict:
    """Read a piece and check it before any write (a mismatch is an error)."""
    if any(not (piece / m).is_file() for m in MC.PIECE_META):
        die(f"piece {piece}: some of the metas {[Path(m).name for m in MC.PIECE_META]} are missing — not a store piece")
    info, src, eps, trows = rjson(piece / INFO), rjson(piece / SOURCE), jsonl(piece / EPISODES), jsonl(piece / TASKS)
    tasks, c, stats = {int(r["task_index"]): r["task"] for r in trows}, info.get("collect") or {}, jsonl(piece / STATS)
    want = {P.tree_name(x) for cell in src.get("cells") or [] for x in cell.get("clips") or []}
    srcs = [PC.source_of(e, want) for e in eps]
    extra = sorted({p.name for p in (piece / META).iterdir()} - {Path(m).name for m in MC.PIECE_META})
    bad = [
        m
        for b, m in (
            (extra, f"unknown meta files {extra}"),
            (len(tasks) != len(trows), "duplicate task indices"),
            ([e["episode_index"] for e in eps] != list(range(len(eps))), "episode indices != 0..n-1"),
            (c.get("video_keys") != list(ST.VIDEO_KEYS.values()), "video_keys != ST.VIDEO_KEYS"),
            (FM.FORM_KEY not in c, f"no info.collect.{FM.FORM_KEY} (piece stored before the hand x dataset form)"),
            (not want or set(srcs) - want, f"source clips not in the cells' clips {sorted(set(srcs) - want)[:3]}"),
            (MR.old_tree_aug(eps), f"aug piece without the off-screen rule ({EPI.ORIG_FOV_RULE}) record — old tree"),
        )
        if b
    ]
    bad += MC.stats_problems(info, eps, stats) if not bad else []
    fam, feats, files, i0 = MC.families(info), CL.parquet_columns(info), [], 0
    for e in eps if not bad else []:
        files.append({n: MC.ep_path(piece, info, f, e["episode_index"]) for n, f in fam.items()})
        miss = [n for n, p in files[-1].items() if not p.is_file()]
        bad += (
            [f"ep{e['episode_index']}: missing files {miss[:3]}"]
            if miss
            else MC.parquet_problems(files[-1][DATA], e, e["episode_index"], i0, feats, tasks)
        )
        i0 += e["length"]
    known = {p for fs in files for p in fs.values()}
    other = [
        str(p)
        for p in piece.rglob("*")
        if not bad and not p.is_dir() and p.parts[len(piece.parts)] != META and p not in known
    ]
    if bad or other:
        die(
            f"piece {piece} failed the input check: "
            f"{(bad + [f'files outside the form {len(other)}: {other[:2]}'] * bool(other))[:4]}"
        )
    h = MC.piece_meta_sha(lambda m: (piece / m).read_bytes())
    return {
        "info": info,
        "eps": eps,
        "tasks": tasks,
        "src": src,
        "srcs": srcs,
        "files": files,
        "stats": stats,
        "sha": h,
    }


def _journal(final, piece_id, files):
    """Version-2 journal: previous lengths and tail fingerprints of the four append-only metas, hard links to the
    previous info and tasks (new versions are renamed in); journal.json is written last."""
    (jd := final / JOURNAL / "prev").mkdir(parents=True)
    replace = {m: (final / m).is_file() for m in MC.REPLACE}
    for m in (m for m, ok in replace.items() if ok):
        os.link(final / m, jd / Path(m).name)
    PC.write_atomic(
        jd.parent / JFILE,
        json.dumps(
            {
                "version": 2,
                "piece_id": piece_id,
                "files": files,
                "append": {m: MC.file_fp(final / m) for m in MC.APPEND},
                "replace": replace,
            },
            ensure_ascii=False,
        ),
    )


def _prune(p, final):
    for d in (d for d in p.parents if final in d.parents):
        with contextlib.suppress(OSError):
            d.rmdir()


def _restore(jd, final, m, present):
    """Restore the previous version from its hard link (idempotent)."""
    if present and (prev := jd / "prev" / Path(m).name).is_file():
        os.replace(prev, final / m)
    elif present and not (final / m).is_file():
        die(input_missing(f"undo: no previous version of {m} — inspect {jd} by hand"))
    elif not present:
        (final / m).unlink(missing_ok=True)


def undo(final: Path) -> dict | None:
    """Undo per the journal (idempotent); returns the journal or None. Append-only metas are truncated to their
    previous length (a different tail fingerprint is an error: someone else wrote)."""
    jd = final / JOURNAL
    j = rjson(jd / JFILE) if (jd / JFILE).is_file() else None
    for rel in (j or {}).get("files", []):
        (final / rel).unlink(missing_ok=True)
        _prune(final / rel, final)
    if j:
        if j.get("version") != 2:
            die(f"undo: journal version {j.get('version')} — only version 2 is supported; inspect {jd} by hand")
        for m, fp in j["append"].items():
            if fp is None:
                (final / m).unlink(missing_ok=True)
                continue
            if not (final / m).is_file() or (final / m).stat().st_size < fp[0]:
                die(f"undo: {m} is shorter than the journal length {fp[0]} — inspect {jd} by hand")
            os.truncate(final / m, fp[0])
            if MC.tail_sha(final / m, fp[0]) != fp[1]:
                die(f"undo: {m} tail fingerprint != journal — inspect {jd} by hand")
        for m, present in j["replace"].items():
            _restore(jd, final, m, present)
    for t in (final / META).glob("*.tmp*") if j and (final / META).is_dir() else []:
        t.unlink()
    _prune(final / META / "x", final)
    shutil.rmtree(jd, ignore_errors=True)
    return j


def _renumber(table, ep, index0, tmap):
    """Rewrite the three index columns (episode_index = ep, index = index0 + row, task_index = tmap[old]); column
    order and types are kept."""
    import numpy as np
    import pyarrow as pa

    n = table.num_rows
    for name, v in (
        ("episode_index", np.full(n, ep)),
        ("index", index0 + np.arange(n)),
        ("task_index", np.array([tmap[int(x)] for x in table.column("task_index").to_numpy()], dtype=np.int64)),
    ):
        f = table.schema.field(i := table.schema.get_field_index(name))
        table = table.set_column(i, f, pa.array(v, type=f.type))
    return table


def _link(src, dst):
    os.link(src, dst)


def _place(src, dst, seen):
    """Hard link (copy on EXDEV); a source inode already placed is linked to its first destination."""
    key = (src.stat().st_dev, src.stat().st_ino)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        _link(seen.get(key, src), dst)
    except OSError as ex:
        if ex.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)
    seen.setdefault(key, dst)


def plan(final, cur, pc, piece_id) -> dict:
    """New indices, task table (setdefault order), files to place and new lines."""
    info = cur["info"] if cur else pc["info"]
    n0, f, ft = (info["total_episodes"], info["total_frames"], dict(cur["tasks"])) if cur else (0, 0, {})
    fam, f0, files, lines = MC.families(info), f, [], []
    for k, e in enumerate(pc["eps"]):
        for s in e["tasks"]:
            ft.setdefault(s, len(ft))
        for name, fm in fam.items():
            if (dst := MC.ep_path(final, info, fm, n0 + k)).exists() or dst.is_symlink():
                die(f"the final already has {dst.relative_to(final)} — a file outside the metadata")
            files.append({"name": name, "src": pc["files"][k][name], "dst": dst, "ep": n0 + k, "index0": f})
        lines.append(
            dict(
                e,
                episode_index=n0 + k,
                piece_id=piece_id,
                src_episode_index=e["episode_index"],
                piece_tasks=list(e["tasks"]),
            )
        )
        f += e["length"]
    return {
        "n0": n0,
        "f0": f0,
        "n": n0 + len(lines),
        "frames": f,
        "tasks": ft,
        "files": files,
        "lines": lines,
        "tmap": {i: ft[s] for i, s in pc["tasks"].items() if s in ft},
    }


def write(final, cur, pc, pl, piece_id):
    """Place the files and append / replace the metadata for one planned piece."""
    import pyarrow.parquet as pq

    seen = {}
    for it in pl["files"]:
        if it["name"] != DATA:
            _place(it["src"], it["dst"], seen)
            continue
        it["dst"].parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_renumber(read_table(it["src"]), it["ep"], it["index0"], pl["tmap"]), it["dst"])
    (final / META).mkdir(exist_ok=True)
    info = json.loads(json.dumps(cur["info"] if cur else pc["info"]))
    for k in (k for k in FM.UNION_FIELDS if cur and k in info["collect"]):  # per-side fields: union (store.form)
        info["collect"][k] = FM.union_field(k, [cur["info"]["collect"][k], pc["info"]["collect"][k]])
    for k in (k for k in info["collect"] if k.endswith(MC.COUNTS)):  # summed (the first piece from 0)
        info["collect"][k] = {x: v + (info["collect"][k][x] if cur else 0) for x, v in pc["info"]["collect"][k].items()}
    n = pl["n"]
    info.update(CL.totals(n, pl["frames"], len(pl["tasks"]), info["chunks_size"], len(info["collect"]["video_keys"])))
    rec = dict(
        piece_id=piece_id,
        piece_sha256=pc["sha"],
        piece_path=str(pc["path"]),
        round=pl["round"],
        episodes=[pl["n0"], n],
        frames=[pl["f0"], pl["frames"]],
        clips=sorted(set(pc["srcs"])),
        episode_clips=sorted({e["clip"] for e in pc["eps"]}),
        merged_at=PC.now(),
        collect=pc["src"],
        **({"clip_skipped": pl["clip_skipped"]} if pl.get("clip_skipped") else {}),
        **({"self_pen_dropped": pl["self_pen_dropped"]} if pl.get("self_pen_dropped") else {}),
        **({"dyn_pair_dropped": pl["dyn_pair_dropped"]} if pl.get("dyn_pair_dropped") else {}),
        **({k: pl[k] for k in ("topup", "topup_dropped")} if pl.get("topup") is not None else {}),
    )
    stats = [dict(s, episode_index=pl["n0"] + k) for k, s in enumerate(pc["stats"])]
    pl["text"], pl["rec"] = {}, rec
    for m, text in (
        (EPISODES, "".join(map(line, pl["lines"]))),
        (PIECES, line(rec)),
        (STATS, "".join(map(line, stats))),
        (SUPERSEDED, "".join(map(line, pl.get("superseded") or []))),
    ):
        if text or m != SUPERSEDED:
            append_bytes(final / m, pl["text"].setdefault(m, text))  # append-only: earlier bytes untouched
    PC.write_atomic(final / TASKS, "".join(line({"task_index": v, "task": k}) for k, v in pl["tasks"].items()))
    PC.write_atomic(final / INFO, json.dumps(info, indent=1, ensure_ascii=False, default=float))


def _say_plan(topped, piece_id, tadd, tdrop, odup, pc, skipped, sup, sup2):
    """One marker line per merge decision (top-up, skipped clips, superseded rows)."""
    if topped:
        say(
            f"{TU.MARK} {piece_id} — added {sum(map(len, tadd.values()))} new variants to {len(tadd)} final clips · "
            f"skipped {len(tdrop)} (already in the final · over the target) · recorded in the pieces line topup · "
            "topup_dropped"
        )
    if odup:
        say(
            f"[ORIG_DUP_CLIP_SKIP] {piece_id} — skipped {len(odup)} same-input clips already in the final "
            f"{[x['clip'] for x in odup][:4]} (same spec base, policy, card, look version; final copy kept) · "
            f"merging the other {len(pc['eps'])} episodes · recorded in the pieces line clip_skipped"
        )
    if skipped and not odup:
        say(
            f"[AUG_DUP_CLIP_SKIP] {piece_id} — skipped same-input clips {[x['clip'] for x in skipped]} (new and final "
            f"rows both original-fallback row 0, same card) · merging the other {len(pc['eps'])} episodes · "
            "recorded in the pieces line clip_skipped"
        )
    if sup:
        say(
            f"[ORIG_FOV] superseded {len(sup)} episodes · {len({x['src_clip'] for x in sup})} clips — row 0 of "
            f"{piece_id}"
        )
    if sup2:
        say(
            f"[AUG_ROW0_REPLACE] superseded {len(sup2)} episodes · clips {sorted({x['src_clip'] for x in sup2})[:4]} "
            f"— old original-fallback row 0 -> variant row 0 of {piece_id} · old rows only marked superseded"
        )


def _write_journaled(final, piece_id, pl, cur, pc):
    """Write under the journal, post-check, undo on any failure; commit = journal file unlink."""
    _journal(final, piece_id, [str(it["dst"].relative_to(final)) for it in pl["files"]])
    try:
        write(final, cur, pc, pl, piece_id)
        bad = MC.post_problems(final, cur, pc, pl, piece_id)
    except BaseException:
        undo(final)
        raise
    if bad:
        undo(final)
        die(f"post-check failed {len(bad)} — this piece was undone: {bad[:4]}")
    (final / JOURNAL / JFILE).unlink()  # commit = this unlink (dying during the rmtree below is not undone)
    shutil.rmtree(final / JOURNAL)


def _finish(idx, pl, final, piece_id, skipped, t0):
    """Index after commit and the summary line."""
    idx = index_add(idx, pl["lines"], [pl["rec"]], pl["superseded"])  # index after commit (rebuilt if we die here)
    idx.update(total_episodes=pl["n"], fp={m: MC.file_fp(final / m) for m in MC.APPEND})
    PC.write_atomic(final / INDEX, json.dumps(idx, ensure_ascii=False))
    out = {
        "skipped": False,
        "piece_id": piece_id,
        "episodes": [pl["n0"], pl["n"]],
        "total_episodes": pl["n"],
        "superseded": len(pl["superseded"]),
        "clip_skipped": len(skipped),
        "self_pen_dropped": len(pl["self_pen_dropped"]),
        "dyn_pair_dropped": len(pl["dyn_pair_dropped"]),
        "seconds": round(time.monotonic() - t0, 2),
    }
    say(f"DONE {piece_id} — episodes {out['episodes']} · {out['seconds']} s · {final}")
    return out


def measure_piece(piece_dataset: Path) -> dict:
    """The pre-lock step of a merge (call it before taking piece.final_lock): read and check the piece, measure its
    trajectories (J-self-pen, S-dyn-pairs output) and fingerprint the bytes measured."""
    pc = read_piece(Path(piece_dataset))
    return {
        "piece": str(Path(piece_dataset).resolve()),
        "fp": SPM.fingerprint(pc),
        "trajectories": SPM.measure(pc),
        "dyn_pairs": DPM.measure(pc),
    }


def merge_piece(final_dir: Path, piece_dataset: Path, piece_id: str, measured: dict) -> dict:
    """Append one piece to the final (see the module docstring; call under piece.final_lock) -> summary dict.
    `measured` = measure_piece of the same piece (taken before the lock)."""
    final, piece, t0 = Path(final_dir), Path(piece_dataset), time.monotonic()
    if not piece_id or piece_id != " ".join(piece_id.split()):
        die(f"piece_id {piece_id!r} — must be non-empty without spaces or newlines")
    pc = dict(read_piece(piece), path=piece.resolve())  # a bad piece never creates the final folder
    SPM.same_piece(pc, measured, piece)
    rnd = MC.piece_round(pc["eps"])
    pc, sp_drop = SPM.plan(pc, piece_id, measured["trajectories"])  # J-self-pen: over the line never reaches the final
    pc, dp_drop = DPM.plan(pc, piece_id, measured["dyn_pairs"])  # S-dyn-pairs (b): over its line never reaches it
    final.mkdir(parents=True, exist_ok=True)
    MC.require_lock(final)
    if (final / JOURNAL).exists() and (j := undo(final)):
        say(f"recovered — undid the interrupted merge of {j['piece_id']}")
    MC.check_names(final, cur := read_head(final), pc["info"])
    idx = index_open(final, cur["info"]) if cur else index_empty()
    if hit := idx["pieces"].get(piece_id):
        if hit[0] != pc["sha"]:
            die(input_missing(f"piece_id {piece_id} exists with a different meta hash"))
        say(f"SKIP {piece_id} — already merged (episodes {hit[1:]})")
        return {"skipped": True, "piece_id": piece_id, "episodes": hit[1:]}
    if cur and (diff := MC.shape_diff(cur["info"], pc["info"])):
        die(f"piece form != final form — differing fields {diff[:8]}")
    old, gone = {tuple(k) for k in idx["keys"]}, set(idx["gone"])
    sup = MR.supersede_plan(index_eps(idx), gone, pc["eps"], piece_id) if cur else []  # off-screen originals
    out_ = gone | {x["episode_index"] for x in sup}
    dupk, sup2 = (MC.round_keys(rnd, pc["srcs"]) & old) if cur else set(), []
    skipped, tadd, tdrop, topped = [], {}, [], False
    sup2, out_, dupk, skipped, pc, tadd, tdrop, topped = MO.aug_overlap(
        dupk, rnd, out_, piece_id, final, pc, skipped, sup2, tadd, tdrop, topped
    )
    odup, owhy, pc, dupk = MO.orig_overlap(dupk, rnd, out_, final, pc)
    if cur and (
        dup := dupk | ({MR.card(e) for e in pc["eps"]} & {(c, lk) for c, lk, i in idx["cards"] if i not in out_})
    ):
        die(
            input_missing(
                f"(round, clip) keys or cards already in the final {len(dup)}: {sorted(dup)[:3]}"
                + (f" · original not the same input {owhy[:3]}" if owhy else "")
            )
        )
    skipped += odup
    pl = dict(
        plan(final, cur, pc, piece_id),
        round=rnd,
        superseded=sup + sup2,
        index=idx,
        aug_fb={x["episode_index"]: x["src_clip"] for x in sup2},
        clip_skipped=skipped,
        self_pen_dropped=sp_drop,
        dyn_pair_dropped=dp_drop,
        **({"topup": tadd, "topup_dropped": tdrop} if topped else {}),
    )
    _say_plan(topped, piece_id, tadd, tdrop, odup, pc, skipped, sup, sup2)
    _write_journaled(final, piece_id, pl, cur, pc)
    return _finish(idx, pl, final, piece_id, skipped, t0)


def status(final_dir: Path) -> dict:
    """Totals, piece count, last piece and journal presence."""
    final = Path(final_dir)
    cur = MC.read_final(final) if final.is_dir() else None
    out = {"final": str(final), "dataset": cur is not None, "journal": (final / JOURNAL).exists()}
    if cur:
        out.update(
            {k: cur["info"][f"total_{k}"] for k in ("episodes", "frames", "tasks", "chunks")},
            pieces=len(cur["pieces"]),
            last_piece=(cur["pieces"] or [{}])[-1].get("piece_id"),
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="merge recorded pieces into a final dataset")
    config.add_data_root_arg(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("append", "verify", "status"):
        s = sub.add_parser(name)
        s.add_argument("--final", type=Path, required=True)
        if name == "append":
            s.add_argument("--piece", type=Path, required=True, help="<piece>/dataset")
            s.add_argument("--piece-id", required=True, help="piece name (idempotency key)")
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    if a.cmd == "status":
        r = status(a.final)
    else:
        measured = measure_piece(a.piece) if a.cmd == "append" else None  # outside the lock
        with CP.final_lock(a.final):
            r = merge_piece(a.final, a.piece, a.piece_id, measured) if a.cmd == "append" else MC.verify_final(a.final)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    return 0 if r.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
