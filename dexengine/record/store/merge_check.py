"""Paths, forms and checks of the piece merge (store.merge): path patterns, lock, post-check and full verify.

Names = meta + the first part of every info path pattern + lock and journal. Same form = every info field (except
total_*, chunks_size; features including their order) and every collect field (except the summed *_counts) are
equal. Post-check: episodes and clips = previous ⊔ piece, previous lines unchanged, totals, files, parquet (columns =
store.lerobot.parquet_columns), episodes_stats lines, shared-file layout (inode partition), video and depth frames
(driver_check.frame_problems; file paths = store.tasks.episode_file).
Supersession (off-screen original rule, episodes.ORIG_FOV_RULE): when an aug piece brings a clip that drew row 0
(traj_choice.row0), that clip's old orig episodes are recorded as superseded in meta/superseded.jsonl (append only;
files stay). Card disjointness and the piece-chain check exclude superseded episodes; each superseded line is checked
(that episode is orig, its source clip drew row 0 in that aug piece). Readers drop superseded_indices.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from dexengine.core.errors import input_missing
from dexengine.record import driver_check as PC
from dexengine.record.store import form as FM
from dexengine.record.store import lerobot as CL
from dexengine.record.store import merge_topup as TU  # TU = top-up keys
from dexengine.record.store import tasks as CT
from dexengine.record.store.merge_rows import (
    AUG_ROW0_RULE,
    card,
    fallback_row0,
    live_clips,
    row0_clips,
    src_clip,
    variant_row0_clips,
)
from dexengine.record.store.parquet import read_table

MARK = "[COLLECT_MERGE]"
META = "meta"  # LeRobot v2.1 meta folder
INFO, EPISODES, TASKS, SOURCE, PIECES, SUPERSEDED = (
    f"{META}/{n}"
    for n in ("info.json", "episodes.jsonl", "tasks.jsonl", "collect.json", "pieces.jsonl", "superseded.jsonl")
)
STATS = CL.STATS
PIECE_META, FINAL_META = (INFO, EPISODES, TASKS, SOURCE, STATS), (INFO, EPISODES, TASKS, PIECES, STATS, SUPERSEDED)
JOURNAL, JFILE = ".merge.journal", "journal.json"
FRAME_STEP = "dexengine.record.store.merge_check"  # capacity.NEEDS key (ffprobe threads)
COUNTS, DATA = "_counts", "data"
# Append-only metas (only appended to) and the two small metas replaced as a whole (journal keeps hard links).
APPEND, REPLACE = (EPISODES, PIECES, STATS, SUPERSEDED), (INFO, TASKS)
TAIL_B = 1 << 16  # tail fingerprint window (bytes), shared by journal, index and post-check


def die(msg):
    raise SystemExit(f"{MARK} ✗ {msg}")


def piece_meta_sha(read) -> str:
    """Piece content fingerprint = sha256 over (name + \\0 + bytes) of every PIECE_META file; read(rel) -> bytes."""
    h = hashlib.sha256()
    for m in PIECE_META:
        h.update(m.encode() + b"\0" + read(m))
    return h.hexdigest()


def say(msg):
    print(f"{MARK} {msg}", flush=True)


def jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def rjson(p):
    return json.loads(p.read_text())


def line(o):
    return json.dumps(o, ensure_ascii=False, default=float) + "\n"


def _canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, default=float)


def families(info: dict) -> dict[str, tuple[str, dict]]:
    """File families: data_path, video_path x video_keys, and every collect *_path with an {episode_index} field."""
    c = info.get("collect") or {}
    out = {DATA: (info["data_path"], {})} | {
        f"v:{k}": (info["video_path"], {"video_key": k}) for k in c.get("video_keys") or []
    }
    return out | {k: (v, {}) for k, v in sorted(c.items()) if k.endswith("_path") and "{episode_index" in str(v)}


def ep_path(ds, info, fam, i) -> Path:
    """Episode i's file of family fam (store.tasks.episode_file on the info path pattern)."""
    return CT.episode_file(Path(ds), info, fam[0], i, **fam[1])


def allowed_names(info):
    return {META, JOURNAL} | {t.split("/", 1)[0] for t, _ in families(info).values()}


def shape_diff(a: dict, b: dict) -> list[str]:
    """Names of the form differences between two infos."""
    skip = ("chunks_size", "collect", "splits")  # totals; collect is compared below
    out = [
        k
        for k in sorted(set(a) | set(b))
        if not k.startswith("total_") and k not in skip and _canon(a.get(k)) != _canon(b.get(k))
    ]
    out += ["features(order)"] * (list(a.get("features") or {}) != list(b.get("features") or {}))
    ca, cb = a.get("collect") or {}, b.get("collect") or {}
    out += [f"collect.{k}(present)" for k in FM.UNION_FIELDS if (k in ca) != (k in cb)]  # union fields: presence
    out += FM.form_diff(a, b)  # form = structural fingerprint and version
    return out + [
        f"collect.{k}"
        for k in sorted(set(ca) | set(cb))
        if k not in (*FM.UNION_FIELDS, FM.FORM_KEY)
        and (
            sorted(ca.get(k) or {}) != sorted(cb.get(k) or {})
            if k.endswith(COUNTS)
            else _canon(ca.get(k)) != _canon(cb.get(k))
        )
    ]


def parquet_problems(p, e, ei, index0, feats, tasks) -> list[str]:
    """Columns = feats (in order), rows = length, episode_index, contiguous index, task texts = the line's tasks."""
    import numpy as np
    import pyarrow.parquet as pq

    if (names := pq.read_schema(p).names) != feats:
        return [f"ep{ei}: columns != features (extra {sorted(set(names) - set(feats))[:2]})"]
    t = read_table(p, columns=["episode_index", "index", "task_index"])
    epi, idx, ti = (t.column(k).to_numpy() for k in ("episode_index", "index", "task_index"))
    got = {tasks.get(int(x)) for x in set(ti.tolist())}
    return [
        f"ep{ei}: {m}"
        for b, m in (
            (t.num_rows != e["length"], f"rows {t.num_rows} != length {e['length']}"),
            (not (epi == ei).all(), f"episode_index {sorted(set(epi.tolist()))[:2]} != {ei}"),
            (not np.array_equal(idx, index0 + np.arange(t.num_rows)), f"index != {index0}.. contiguous"),
            (got != set(e["tasks"]), f"task_index texts {sorted(map(str, got))[:2]} != line tasks {e['tasks'][:2]}"),
        )
        if b
    ]


def stats_problems(info, eps, stats) -> list[str]:
    """episodes_stats lines: index = line, features = features - store.form.stats_skip, numeric count = length."""
    if [s["episode_index"] for s in stats] != [e["episode_index"] for e in eps]:
        return ["episodes_stats indices != episodes indices"]
    bad = []
    for e, s in zip(eps, stats, strict=True):
        ks, absent = set(s["stats"]), set(FM.stats_skip(e))
        want = {n for n, f in info["features"].items()} - absent
        if ks != want or any(s["stats"][n]["count"] != [e["length"]] for n in ks if n in CL.parquet_columns(info)):
            bad.append(f"ep{e['episode_index']}: stats features / count mismatch (missing {sorted(want - ks)[:2]})")
    return bad


def tail_sha(p: Path, size: int) -> str:
    """Tail fingerprint = sha256 of bytes [size - TAIL_B, size) (never reads the whole file)."""
    with open(p, "rb") as f:
        f.seek(max(0, size - TAIL_B))
        return hashlib.sha256(f.read(size - max(0, size - TAIL_B))).hexdigest()


def file_fp(p: Path) -> list | None:
    """Fingerprint of one append-only meta = [length in bytes, tail fingerprint], or None when absent."""
    return [s := p.stat().st_size, tail_sha(p, s)] if p.is_file() else None


def read_tail(p: Path, size0: int) -> list[dict]:
    """Read only the lines after size0 bytes (the tail appended now)."""
    with open(p, "rb") as f:
        f.seek(size0)
        return [json.loads(x) for x in f.read().decode("utf-8").splitlines() if x.strip()]


def read_final(final: Path) -> dict | None:
    """The final's metadata, or None without info (first piece)."""
    return (
        {
            "info": rjson(final / INFO),
            "eps": jsonl(final / EPISODES),
            "pieces": jsonl(final / PIECES),
            "tasks": {r["task"]: int(r["task_index"]) for r in jsonl(final / TASKS)},
            "superseded": jsonl(final / SUPERSEDED) if (final / SUPERSEDED).is_file() else [],
        }
        if (final / INFO).is_file()
        else None
    )


def superseded_indices(final: Path) -> set:
    """Superseded episode indices (readers drop them)."""
    return {int(r["episode_index"]) for r in jsonl(final / SUPERSEDED)} if (final / SUPERSEDED).is_file() else set()


def check_names(final, cur, pinfo):
    """The final folder holds only LeRobot names (and no files without info)."""
    names = {p.name for p in final.iterdir()}
    if bad := names - allowed_names(cur["info"] if cur else pinfo):
        die(f"{final}: names outside the LeRobot layout {sorted(bad)[:6]}")
    if cur is None and (
        stale := [p for n in names - {JOURNAL} for p in [final / n, *(final / n).rglob("*")] if not p.is_dir()]
    ):
        die(f"{final}: no {INFO} but {len(stale)} files (e.g. {stale[0]})")


def require_lock(final):
    """Die unless the final lock (piece.final_lock: flock on the dataset folder's parent) is held. Only peeks."""
    fd = os.open(Path(final).parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    finally:
        os.close(fd)
    die(f"called outside the {Path(final).parent} lock (piece.final_lock) — write only under the lock")


def frame_check(final, info, eps, workers=None) -> list[str]:
    """Video and depth frames (driver_check.frame_problems on info paths), episodes split across a thread pool sized by
    capacity (`workers` = an explicit option)."""
    from concurrent.futures import ThreadPoolExecutor

    from dexengine.core import capacity

    with capacity.sized(FRAME_STEP, len(eps), workers) as cap, ThreadPoolExecutor(cap.workers) as ex:
        n = cap.workers
        groups = [g for g in (eps[k::n] for k in range(n)) if g]
        return [x for r in ex.map(lambda g: PC.frame_problems(Path(final), g, info), groups) for x in r]


def share_problems(pairs) -> list[str]:
    """Shared-file layout: the source inode partition equals the final inode partition."""
    m = {((a.stat().st_dev, a.stat().st_ino), (b.stat().st_dev, b.stat().st_ino)) for a, b in pairs}
    na, nb = len({x for x, _ in m}), len({y for _, y in m})
    return [] if na == nb == len(m) else [f"shared layout differs: source inodes {na} · final {nb}"]


def piece_round(eps) -> str:
    """The piece's round: its episodes' round field must have exactly one value. A clip has one piece per round, so
    overlap keys are (round, clip)."""
    rs = {e.get("round") for e in eps}
    if len(rs) != 1 or None in rs:
        raise SystemExit(f"[COLLECT_MERGE] ✗ piece episode rounds {sorted(map(str, rs))} — must be exactly one")
    return rs.pop()


def round_keys(rnd: str, clips) -> set:
    return {(rnd, c) for c in clips}


def meta_problems(final):
    """Metadata consistency: indices, totals, tasks, stats lines, piece range chain, disjoint clips, the self-pen drop
    records (a line without episodes only when its every trajectory was dropped)."""
    from dexengine.record.store import merge_dyn_pairs as DPM  # noqa: PLC0415 — it imports this module
    from dexengine.record.store import merge_self_pen as SPM  # noqa: PLC0415 — it imports this module

    info, eps, trows, pieces = rjson(final / INFO), jsonl(final / EPISODES), jsonl(final / TASKS), jsonl(final / PIECES)
    n, tset = len(eps), {r["task"] for r in trows}
    want = CL.totals(
        n, sum(e["length"] for e in eps), len(trows), info["chunks_size"], len(info["collect"]["video_keys"])
    )
    off = {k: (info.get(k), v) for k, v in want.items() if info.get(k) != v}
    bad = [
        m
        for b, m in (
            ([e["episode_index"] for e in eps] != list(range(n)), "episode indices != 0..N-1"),
            (off, f"totals != measured {off}"),
            ([r["task_index"] for r in trows] != list(range(len(trows))) or len(tset) != len(trows), "duplicate tasks"),
        )
        if b
    ]
    bad += stats_problems(info, eps, jsonl(final / STATS))
    sup_rows = jsonl(final / SUPERSEDED) if (final / SUPERSEDED).is_file() else []
    gone = {int(x["episode_index"]) for x in sup_rows}
    bad += ["superseded lines overlap"] * (len(gone) != len(sup_rows))
    for x in sup_rows:  # check each superseded line
        i, by = int(x["episode_index"]), [r for r in pieces if r["piece_id"] == x.get("by_piece")]
        e = eps[i] if 0 <= i < n else {}
        bye = eps[by[0]["episodes"][0] : by[0]["episodes"][1]] if by else []
        aug2 = x.get("rule") == AUG_ROW0_RULE  # aug -> aug: old = fallback row 0, new piece = variant row 0
        if (
            (not fallback_row0(e) if aug2 else e.get("round") != "orig")
            or src_clip(e) != x.get("src_clip")
            or not by
            or by[0]["round"] != "aug"
            or x["src_clip"] not in (variant_row0_clips(bye) if aug2 else row0_clips(bye))
        ):
            bad.append(
                f"superseded line ep{i}: not orig, other source clip, or not row 0 of the superseding piece "
                f"({x.get('by_piece')})"
            )
    edge, seen = 0, set()
    for r in pieces:
        a, b = r["episodes"]
        ids = [(e.get("piece_id"), e.get("src_episode_index"), e["clip"]) for e in eps[a:b]]
        keys = round_keys(r["round"], TU.key_clips(r, live_clips(r, eps, gone))) | {
            card(e) for e in eps[a:b] if e["episode_index"] not in gone
        }
        if (
            a != edge
            or [x[:2] for x in ids] != [(r["piece_id"], k) for k in range(b - a)]
            or seen & keys
            or {x[2] for x in ids} != set(r["episode_clips"])
            or (
                {e.get("round") for e in eps[a:b]} != {r["round"]}
                and not (a == b and (r.get("self_pen_dropped") or r.get("dyn_pair_dropped")))
            )
        ):
            bad.append(f"piece {r['piece_id']} {r['episodes']}: range / lines / clips / round mismatch")
        edge, seen = b, seen | keys
        bad += SPM.line_problems(r) + DPM.line_problems(r)
    bad += TU.problems(pieces, eps, gone)
    return bad + [f"piece range end {edge} != episodes {n}"] * (edge != n), info, eps, pieces


def episode_problems(final, info, eps, i0: int) -> list[str]:
    """Per-episode family files and parquet (i0 = index start of the first episode = sum of earlier lengths)."""
    fam, feats = families(info), CL.parquet_columns(info)
    tasks = {r["task_index"]: r["task"] for r in jsonl(final / TASKS)}
    bad = []
    for e in eps:
        i = e["episode_index"]
        miss = [n for n, f in fam.items() if not ep_path(final, info, f, i).is_file()]
        bad += (
            [f"ep{i}: missing files {miss[:3]}"]
            if miss
            else parquet_problems(ep_path(final, info, fam[DATA], i), e, i, i0, feats, tasks)
        )
        i0 += e["length"]
    return bad


def _read_tails(j, final, bad, tail, pl):
    """Tail (appended lines) of each appended file; a changed earlier part is a problem."""
    for m, fp0 in j["append"].items():
        size0, p = (fp0 or [0])[0], final / m
        if fp0 and (not p.is_file() or p.stat().st_size < size0 or tail_sha(p, size0) != fp0[1]):
            bad.append(f"{m}: earlier lines changed (length / tail fingerprint)")
            tail[m] = []
            continue
        tail[m] = read_tail(p, size0) if p.is_file() else []
        if p.is_file() and p.stat().st_size - size0 != len(pl["text"].get(m, "").encode("utf-8")):
            bad.append(f"{m}: tail length != appended bytes")


def _line_problems(new, n0, n, pc, piece_id, off, trows, old_t, stats, prow, idx, rnd, srow, cards, have, pl, bad):
    """Index, task, stats, pieces-line, card and superseded-plan checks of the appended lines."""
    bad += [
        m
        for b, m in (
            ([e["episode_index"] for e in new] != list(range(n0, n)), "new episode indices != n0..n-1"),
            (
                [(e.get("piece_id"), e.get("src_episode_index")) for e in new]
                != [(piece_id, k) for k in range(len(pc["eps"]))],
                "new episodes != piece",
            ),
            (off, f"totals != measured {off}"),
            (
                [r["task_index"] for r in trows] != list(range(len(trows)))
                or len({r["task"] for r in trows}) != len(trows),
                "duplicate tasks",
            ),
            ([(r["task"], int(r["task_index"])) for r in trows[: len(old_t)]] != old_t, "earlier task lines changed"),
            ({t for e in new for t in e["tasks"]} - {r["task"] for r in trows}, "new episode tasks not in tasks"),
            ([s["stats"] for s in stats] != [s["stats"] for s in pc["stats"]], "new stats lines != piece stats"),
            (
                len(prow) != 1
                or prow[0]["piece_id"] != piece_id
                or piece_id in idx["pieces"]
                or prow[0]["episodes"] != [n0, n]
                or set(prow[0]["clips"]) != set(pc["srcs"])
                or prow[0]["round"] != rnd
                or set(prow[0]["episode_clips"]) != {e["clip"] for e in new}
                or (
                    {e.get("round") for e in new} != {rnd}
                    and not (not new and (pl.get("self_pen_dropped") or pl.get("dyn_pair_dropped")))
                )
                or round_keys(rnd, TU.key_clips(prow[0], pc["srcs"]))
                & {tuple(k) for k in idx["keys"]}
                - {("aug", x["src_clip"]) for x in srow if x.get("rule") == AUG_ROW0_RULE},
                "last pieces line != this piece",
            ),
            (len(set(cards)) != len(cards) or set(cards) & have, "new cards overlap (index minus superseded)"),
            (
                [x["episode_index"] for x in srow] != [x["episode_index"] for x in pl.get("superseded") or []],
                "superseded tail != plan",
            ),
        )
        if b
    ]
    return bad


def _superseded_problems(srow, pl, orig, piece_id, rnd, idx, new, bad):
    """Superseded tail lines: each is row 0 of an original/fallback the superseding piece replaces."""
    for x in srow:  # superseded check (tail only)
        i = int(x["episode_index"])
        aug2 = x.get("rule") == AUG_ROW0_RULE  # old row = the fallback row 0 found by the plan (pl aug_fb)
        if (
            (pl.get("aug_fb", {}).get(i) if aug2 else orig.get(i)) != x.get("src_clip")
            or x.get("by_piece") != piece_id
            or rnd != "aug"
            or i in idx["gone"]
            or x["src_clip"] not in (variant_row0_clips(new) if aug2 else row0_clips(new))
        ):
            bad.append(
                f"superseded line ep{i}: not orig, other source clip, or not row 0 of the superseding piece "
                f"({x.get('by_piece')})"
            )


def post_problems(final, cur, pc, pl, piece_id) -> list[str]:
    """Post-check of an append (reads only the appended tails; any problem undoes the merge). Earlier bytes = the
    journal's lengths and tail fingerprints; tails = the bytes meant to be appended; new indices n0..n-1, pieces
    line, stats, superseded lines (against the index originals), totals = previous + piece, tasks, disjoint cards
    (index minus superseded), new episode files, parquet, shared layout, frames. The whole final is checked by
    verify (meta_problems)."""
    j, idx, n0, n = rjson(final / JOURNAL / JFILE), pl["index"], pl["n0"], pl["n"]
    bad, tail = [], {}
    _read_tails(j, final, bad, tail, pl)
    new, stats, prow, srow = (tail[m] for m in (EPISODES, STATS, PIECES, SUPERSEDED))
    info, trows, rnd = rjson(final / INFO), jsonl(final / TASKS), pl["round"]
    want = CL.totals(
        n, pl["f0"] + sum(e["length"] for e in new), len(trows), info["chunks_size"], len(info["collect"]["video_keys"])
    )
    off = {k: (info.get(k), v) for k, v in want.items() if info.get(k) != v}
    old_t = list((cur or {}).get("tasks", {}).items())
    out_ = set(idx["gone"]) | {int(x["episode_index"]) for x in srow}
    have = {(c, lk) for c, lk, i in idx["cards"] if i not in out_}
    orig = {i: s for i, c, s in idx["orig"]}
    cards = [card(e) for e in new]
    bad = _line_problems(new, n0, n, pc, piece_id, off, trows, old_t, stats, prow, idx, rnd, srow, cards, have, pl, bad)
    bad += stats_problems(info, new, stats)
    _superseded_problems(srow, pl, orig, piece_id, rnd, idx, new, bad)
    bad += ["superseded lines overlap"] * (len({int(x["episode_index"]) for x in srow}) != len(srow))
    bad += episode_problems(final, info, new, pl["f0"])
    for name in (n for n in families(info) if n != DATA):
        bad += share_problems([(it["src"], it["dst"]) for it in pl["files"] if it["name"] == name])
    return bad + (frame_check(final, info, new) if new else [])


def verify_final(final_dir: Path, frames: bool = True) -> dict:
    """Check the whole final (under final_lock): journal, names, metadata, every episode file, parquet, files outside
    the layout, frames."""
    final, t0 = Path(final_dir), time.monotonic()
    require_lock(final)
    if not (final / INFO).is_file():
        die(input_missing(f"final {final}: no {INFO}"))
    bad, info, eps, pieces = meta_problems(final)
    bad += [f"journal {JOURNAL} left over (the next append undoes it)"] * (final / JOURNAL).exists()
    bad += [
        f"names outside the layout {sorted(x)}" for x in [{p.name for p in final.iterdir()} - allowed_names(info)] if x
    ]
    bad += episode_problems(final, info, eps, 0)
    fam = families(info)
    want = {ep_path(final, info, f, e["episode_index"]) for e in eps for f in fam.values()}
    other = [
        p
        for n in {t.split("/", 1)[0] for t, _ in fam.values()}
        for p in (final / n).rglob("*")
        if not p.is_dir() and p not in want
    ]
    bad += [f"files outside the metadata {len(other)}: {other[:2]}"] * bool(other)
    bad += frame_check(final, info, eps) if frames and eps else []
    return {
        "ok": not bad,
        "problems": bad[:20],
        "n_problems": len(bad),
        "episodes": len(eps),
        "pieces": len(pieces),
        "seconds": round(time.monotonic() - t0, 2),
    }
