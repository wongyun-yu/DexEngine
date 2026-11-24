"""The final text table the dataset writer reads (store hook): the table path from DEXENGINE_COLLECT_TEXTS, the per-clip
table cached per process, one episode's text and the retext sub-command that loads a table into stored datasets (it
resets that cache)."""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.record.text.api import MARK, die, read_jsonl

# ───────────────────────── store hook ─────────────────────────
#: The text table the dataset writer reads (path of a merged texts.jsonl; empty = off, source labels only). Process
#: boundary: set by the caller of the store (and by cmd_retext for its own process).
TEXTS_ENV = "DEXENGINE_COLLECT_TEXTS"
_TEXTS: dict | None = None


def clip_key(name: str) -> str:
    """Table key = clip name without the tree suffix (an episode row's clip is the tree name `_pl_rt`); the suffix
    rule is owned by paths.clip_name."""
    from dexengine.core import paths as P

    return P.clip_name(name)


def texts_path() -> str:
    """The text table path from TEXTS_ENV ('' = off)."""
    return os.environ.get(TEXTS_ENV, "")


def store_texts() -> dict | None:
    """{clip key: row} when the table is on (None when off); a duplicate clip or a missing file dies. Read once per
    process."""
    global _TEXTS
    p = texts_path()
    if not p:
        return None
    if _TEXTS is None:
        f = Path(p)
        if not f.is_file():
            die(f"{TEXTS_ENV}={p}: no such file")
        by = {}
        for r in read_jsonl(f):
            k = clip_key(r["clip"])
            if k in by:
                die(f"same clip twice in the text table: {r['clip']}")
            by[k] = r
        _TEXTS = by
        models = sorted({r.get("model") for r in by.values()})
        shas = sorted({r.get("prompt_sha") for r in by.values()})
        print(f"{MARK} store on n={len(by)} model={models} prompt={shas} ← {f}", flush=True)
    return _TEXTS


def episode_text(clip: str, source_clip: str | None = None) -> dict | None:
    """The episode's text fields; a placement variant uses its source clip's text (selection.source_clip). None
    when the table is off; a clip missing from the table dies (no silent fallback to the source label)."""
    texts = store_texts()
    if texts is None:
        return None
    key = clip_key(source_clip or clip)
    r = texts.get(key)
    if r is None:
        die(f"clip {key} is not in the text table (rejected at validate): regenerate it or fix the table")
    return {
        k: r.get(k)
        for k in ("instruction", "paraphrases", "short", "detailed", "confidence", "model", "prompt_sha", "subtasks")
    }


def cmd_retext(a) -> int:
    """retext: load --texts (the final table) or --off (back to source labels) into the datasets; sets the table env
    for this process only."""
    global _TEXTS
    if a.texts:
        if not Path(a.texts).is_file():
            die(f"--texts {a.texts} does not exist")
        os.environ[TEXTS_ENV] = str(Path(a.texts).resolve())
    else:
        os.environ.pop(TEXTS_ENV, None)
    _TEXTS = None
    from dexengine.record.store import tasks as CTK

    a.results = [CTK.retext(Path(d), lock=getattr(a, "lock", None)) for d in a.dataset]  # lock held only for swaps
    print(f"{MARK} retext datasets {len(a.dataset)} done", flush=True)
    return 0
