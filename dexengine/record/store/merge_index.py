"""Episode index of the final dataset for the piece merge (record.store.merge): the merge index file (INDEX, versioned,
marker IMARK) that lists the final's episodes without re-reading episodes.jsonl, and the append-only writer it shares
with the merge (append_bytes, read_head)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from dexengine.record import driver_check as PC
from dexengine.record.store import merge_check as MC
from dexengine.record.store import merge_rows as MR
from dexengine.record.store.merge_check import INFO, META, die, jsonl, rjson, say

INDEX, INDEX_V, IMARK = f"{META}/.merge_index.json", 1, "[MERGE_INDEX]"  # index file, version, marker


def append_bytes(p: Path, text: str) -> None:
    """Append only (open 'ab', fsync). A file with other hard links (someone else's snapshot) is first copied and
    renamed so the snapshot does not grow."""
    if p.is_file() and (k := p.stat().st_nlink) > 1:
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        shutil.copy2(p, tmp)
        os.replace(tmp, p)
        say(f"unlinked {p.parent.name}/{p.name} ({k} links — snapshot kept, full copy this once)")
    with open(p, "ab") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def read_head(final: Path) -> dict | None:
    """Light read for merging: info and tasks only (the rest comes from the index); None without info."""
    return (
        {"info": rjson(final / INFO), "tasks": {r["task"]: int(r["task_index"]) for r in jsonl(final / MC.TASKS)}}
        if (final / INFO).is_file()
        else None
    )


def index_add(idx: dict, eps, pieces, sup_rows) -> dict:
    """Add to the index: pieces (id -> [meta sha, a, b]; the first line owns it), (round, clip) keys, cards (clip,
    look, index), originals (index, clip, source clip; targets of off-screen supersession) and superseded indices."""
    for r in pieces:
        idx["pieces"].setdefault(r["piece_id"], [r["piece_sha256"], *r["episodes"]])
        idx["keys"] += sorted(map(list, MC.round_keys(r["round"], r["clips"])))
    idx["cards"] += [[*MR.card(e), int(e["episode_index"])] for e in eps]
    idx["orig"] += [[int(e["episode_index"]), e["clip"], MR.src_clip(e)] for e in eps if e.get("round") == "orig"]
    idx["gone"] += [int(x["episode_index"]) for x in sup_rows]
    return idx


def index_empty() -> dict:
    return {"v": INDEX_V, "pieces": {}, "keys": [], "cards": [], "orig": [], "gone": [], "total_episodes": 0, "fp": {}}


def index_open(final: Path, info: dict) -> dict:
    """The index (meta/.merge_index.json). Used as is when its fingerprints (lengths and tails of the four
    append-only metas) and total match the final; otherwise the final is read in full, checked with
    MC.meta_problems and the index is rebuilt (a failed check is an error: never append to a broken final)."""
    fp, why = {m: MC.file_fp(final / m) for m in MC.APPEND}, None
    try:
        idx = rjson(final / INDEX)
    except FileNotFoundError:
        why = "missing"
    except (ValueError, OSError) as ex:
        why = f"read failed {type(ex).__name__}"
    if why is None:
        off = [Path(m).name for m in MC.APPEND if (idx.get("fp") or {}).get(m) != fp[m]]
        why = (
            f"version {idx.get('v')} != {INDEX_V}"
            if idx.get("v") != INDEX_V
            else f"meta fingerprints differ {off}"
            if off
            else f"total {idx.get('total_episodes')} != info {info['total_episodes']}"
            if idx.get("total_episodes") != info["total_episodes"]
            else None
        )
    if why is None:
        say(f"{IMARK} ok — using the index (episodes {idx['total_episodes']} · pieces {len(idx['pieces'])})")
        return idx
    bad, _, eps, pieces = MC.meta_problems(final)
    if bad:
        die(f"{IMARK} full check before rebuilding failed {len(bad)}: {bad[:3]} — inspect the final with verify")
    idx = index_add(index_empty(), eps, pieces, jsonl(final / MC.SUPERSEDED))
    idx.update(total_episodes=len(eps), fp=fp)
    PC.write_atomic(final / INDEX, json.dumps(idx, ensure_ascii=False))
    say(f"{IMARK} rebuilt — {why} · full read episodes {len(eps)} · pieces {len(pieces)}")
    return idx


def index_eps(idx: dict) -> list[dict]:
    """Index originals -> the rows supersede_plan reads (index, clip, round, source clip)."""
    return [
        {"episode_index": i, "clip": c, "round": "orig", "selection": {"source_clip": s}} for i, c, s in idx["orig"]
    ]
