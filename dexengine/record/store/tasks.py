"""Task text of the recorded datasets: source labels, instructions (DEXENGINE_COLLECT_TEXTS), meta/tasks and
meta/episodes writing, and re-attaching texts to a stored dataset (retext).

The store (store.store.write_episode / store) and retext share these functions: this module owns task numbering and
the text fields of an episode line. CPU, retarget venv (pyarrow only for retext).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dexengine.record.store import lerobot as CL
from dexengine.record.store.parquet import read_table
from dexengine.record.text import text_table as TT


def episode_file(ds: Path, info: dict, tmpl: str, i: int, **kw) -> Path:
    """Episode i's file of dataset `ds` from an info.json path pattern (data_path, video_path, collect.depth_path,
    collect.force_path, ...). LeRobot v2.1: episode_chunk = i // chunks_size (chunks_size is required). The same rule
    serves a piece (one chunk) and the merged final (several chunks)."""
    return ds / tmpl.format(episode_chunk=int(i) // int(info["chunks_size"]), episode_index=int(i), **kw)


def task_text(clip: str) -> dict:
    """Source-label task: the source part of the clip name (before the first '__') as words; the split suffixes
    (_t<k>, _seg<k>, placement suffix) are kept separately."""
    head, _, tail = clip.partition("__")
    parts = tail.split("_")
    seg = [p for p in parts if p[:1] == "t" and p[1:].isdigit() or p.startswith("seg")]
    return {"task": head.replace("_", " "), "source_label": head, "segment": seg, "clip": clip}


def episode_task(clip: str, source_clip: str | None) -> tuple[str, dict | None, dict]:
    """Episode task: the instruction when a text table is set (a placement variant uses its source clip's text),
    else the source label. Returns (task, instruction fields | None, source label)."""
    tt = task_text(clip)
    key = TT.clip_key(source_clip or clip)
    if key in HELD_NOW.setdefault("hold", held()) and key not in (TT.store_texts() or {}):
        HELD_NOW.setdefault("hit", set()).add(key)  # held clip: source label, counted in [TEXT_HOLD]
        return tt["task"], None, tt
    tx = TT.episode_text(clip, source_clip)  # None without a text table
    return (tx["instruction"] if tx else tt["task"]), tx, tt


HELD_NOW: dict = {}


def held() -> set:
    """Held clips: the hold list next to the text table (text.store.hold_of), or an empty set without a table."""
    p = TT.texts_path()
    if not p:
        return set()
    from dexengine.record.text import store as CS

    return CS.hold_of(Path(p))


def episode_subtasks(parquet: Path, forces: Path, vfi, tx: dict) -> list[dict]:
    """Episode phase intervals (record.phases.from_files, measured on this episode's forces and object motion) +
    the table's phase texts -> [{phase, start_step, end_step, start_frame, end_frame, text}] (frames index the
    video; no cut files). text = None when the table has no text for that phase."""
    from dexengine.record import phases as PH

    ph, _dt = PH.from_files(parquet, forces)
    by = {x["phase"]: {k: x[k] for k in ("instruction", "paraphrases")} for x in tx.get("subtasks") or []}
    return [
        {
            "phase": p["phase"],
            "start_step": p["i0"],
            "end_step": p["i1"],
            "start_frame": int(vfi[p["i0"]]),
            "end_frame": int(vfi[p["i1"]]),
            "text": by.get(p["phase"]),
        }
        for p in ph
    ]


def write_atomic(p: Path, text: str) -> None:
    """Write a temp file in the same folder and os.replace it, so readers never see a half-written file."""
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def write_task_meta(out: Path, eps: list[dict], tasks: dict) -> None:
    """Write meta/episodes.jsonl and meta/tasks.jsonl (each replaced as a whole)."""
    (out / "meta").mkdir(exist_ok=True)
    write_atomic(
        out / "meta/episodes.jsonl", "".join(json.dumps(e, ensure_ascii=False, default=float) + "\n" for e in eps)
    )
    write_atomic(
        out / "meta/tasks.jsonl",
        "".join(json.dumps({"task_index": v, "task": k}, ensure_ascii=False) + "\n" for k, v in tasks.items()),
    )


def same_column(a, b) -> bool:
    """Are two pyarrow columns equal, treating NaN as equal (equals() is False for NaN columns even after an
    unchanged rewrite)? List columns must also have equal offsets."""
    import pyarrow as pa

    if a.equals(b):
        return True
    a, b = a.combine_chunks(), b.combine_chunks()
    if a.type != b.type or len(a) != len(b) or a.null_count != b.null_count:
        return False
    while pa.types.is_list(a.type) or pa.types.is_large_list(a.type) or pa.types.is_fixed_size_list(a.type):
        if not pa.types.is_fixed_size_list(a.type) and not a.offsets.equals(b.offsets):
            return False
        a, b = a.flatten(), b.flatten()
    if pa.types.is_floating(a.type):
        return np.array_equal(a.to_numpy(zero_copy_only=False), b.to_numpy(zero_copy_only=False), equal_nan=True)
    return a.equals(b)


def _stage_task_columns(eps, found, tasks, out, info, stats, restat, staged):
    """Per episode: task fields of the line, and a staged parquet (+ stats) where task_index differs; returns restat."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for e, (task, tx, tt) in zip(eps, found, strict=True):
        ti = tasks.setdefault(task, len(tasks))
        e["tasks"] = [task]
        for k in ("text", "source_label", "subtasks"):
            e.pop(k, None)
        p = episode_file(out, info, info["data_path"], e["episode_index"])
        if tx:
            e.update(text={k: v for k, v in tx.items() if k != "subtasks"}, source_label=tt["source_label"])
        if tx and tx.get("subtasks"):  # stage texts: same rule as the store
            e["subtasks"] = episode_subtasks(
                p,
                episode_file(out, info, info["collect"]["force_path"], e["episode_index"]),
                read_table(p, columns=["video_frame_index"]).column(0).to_numpy(),
                tx,
            )
        col = read_table(p, columns=["task_index"]).column(0).to_numpy()
        if len(col) != int(e["length"]):
            raise SystemExit(f"[COLLECT_STORE] retext {p.name} rows {len(col)} != length {e['length']}")
        st = stats[e["episode_index"]]["stats"] if stats else None
        if (col == ti).all():
            if st is not None and (ts := CL.numeric_stats(p, info["features"], ["task_index"])["task_index"]) != st.get(
                "task_index"
            ):  # parquet right but stats stale (a previous run died first)
                st["task_index"], restat = ts, restat + 1
            continue
        t = read_table(p)
        new = t.set_column(
            t.schema.get_field_index("task_index"), "task_index", pa.array(np.full(len(col), ti, dtype=col.dtype))
        )
        tmp = p.with_name(p.name + ".retext.tmp")
        pq.write_table(new, tmp)
        staged.append((tmp, p))
        chk = read_table(tmp)
        if (
            chk.num_rows != t.num_rows
            or chk.column_names != t.column_names
            or not all(same_column(chk.column(c), t.column(c)) for c in t.column_names if c != "task_index")
        ):
            raise SystemExit(f"[COLLECT_STORE] retext {p.name} other columns changed — not written")
        if st is not None:  # stats of the changed column, same function as the store
            st["task_index"] = CL.numeric_stats(tmp, info["features"], ["task_index"])["task_index"]
            restat += 1
    return restat


def retext(out: Path, lock=None) -> dict:
    """Re-attach instructions to a stored dataset (when the text table is filled after the store, or edited).

    Only these change: meta/tasks.jsonl, episodes.jsonl (tasks, text, source_label, subtasks), info.json total_tasks,
    the parquet task_index column (all other columns are checked unchanged) and its episodes_stats entry.
    Stable numbering: existing task indices are kept and new tasks are appended (same rule as the merge), so only
    episodes whose task changed or whose column disagrees with meta are rewritten (unused old task rows remain).
    lock (the final-folder lock piece.final_lock, or None): new parquets (<parquet>.retext.tmp) are built and checked
    outside the lock; inside it, if meta is unchanged, they are swapped in (os.replace) and meta is written. If a
    merge changed meta meanwhile, the new files are dropped and retry=True (nothing written). Without a text table
    tasks become source labels. A clip missing from the table dies before anything is written. Idempotent."""
    import contextlib

    snap = {m: (out / "meta" / m).read_bytes() for m in ("episodes.jsonl", "tasks.jsonl", "info.json")}
    eps = [json.loads(x) for x in snap["episodes.jsonl"].decode().splitlines() if x.strip()]
    info = json.loads(snap["info.json"])  # path patterns (data_path, collect.force_path, chunks_size)
    trows = sorted(
        (json.loads(x) for x in snap["tasks.jsonl"].decode().splitlines() if x.strip()),
        key=lambda r: int(r["task_index"]),
    )
    tasks = {r["task"]: int(r["task_index"]) for r in trows}
    if list(tasks.values()) != list(range(len(trows))):
        raise SystemExit(f"[COLLECT_STORE] retext {out} tasks.jsonl indices are not 0..n-1 or tasks repeat")
    HELD_NOW.clear()
    found = [episode_task(e["clip"], (e.get("selection") or {}).get("source_clip")) for e in eps]
    if TT.texts_path():  # held clips are always reported per dataset (also when 0)
        hit = sorted(HELD_NOW.get("hit", ()))
        print(f"[TEXT_HOLD] {len(hit)} {out} {hit[:3]}", flush=True)
    stats = CL.read_stats(out)  # [] when absent; otherwise one line per episode
    if stats and len(stats) != len(eps):
        raise SystemExit(f"[COLLECT_STORE] retext episodes_stats lines {len(stats)} != episodes {len(eps)}")
    staged, restat = [], 0
    try:
        restat = _stage_task_columns(eps, found, tasks, out, info, stats, restat, staged)
        with lock if lock is not None else contextlib.nullcontext():
            if {m: (out / "meta" / m).read_bytes() for m in snap} != snap:
                print(
                    f"[COLLECT_STORE] retext {out} meta changed meanwhile (merge) — dropped {len(staged)} new "
                    "files, retry next time",
                    flush=True,
                )
                return {
                    "episodes": len(eps),
                    "tasks": len(tasks),
                    "changed": 0,
                    "restat": 0,
                    "text_on": None,
                    "retry": True,
                }
            for tmp, p in staged:
                os.replace(tmp, p)
            write_task_meta(out, eps, tasks)
            if restat:
                CL.write_stats(out, stats)
            info["total_tasks"] = len(tasks)
            write_atomic(out / "meta/info.json", json.dumps(info, indent=1, ensure_ascii=False, default=float))
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
    on = TT.store_texts() is not None
    print(
        f"[COLLECT_STORE] retext episodes {len(eps)} · tasks {len(tasks)} · parquet changed {len(staged)} · "
        f"stats {restat} · instructions {'on' if on else 'off (source labels)'} · {out}",
        flush=True,
    )
    return {
        "episodes": len(eps),
        "tasks": len(tasks),
        "changed": len(staged),
        "restat": restat,
        "text_on": on,
        "retry": False,
    }
