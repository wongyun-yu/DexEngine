"""Retry bookkeeping of the record session launcher (record.session.launch): which classify kinds start a session once
more (RETRY_KINDS), the kept name of a dropped attempt's log (keep_name, discarded), and the copy of a Kit minidump
next to its log (keep_dump)."""

from __future__ import annotations

import re
from pathlib import Path

from dexengine.core import log_markers as LM
from dexengine.core.errors import input_missing

#: kinds that start the same session once more (branches of classify; the others die on the first attempt) — owner
#: core.log_markers (its infrastructure words are these kinds' texts)
RETRY_KINDS = LM.RETRY_KINDS


_DISCARDED = re.compile(r"\.(?:" + "|".join(RETRY_KINDS) + r")\d+\.log$")


def keep_name(log: Path, kind: str, n: int = 1) -> Path:
    """New name of the first attempt's log before the retry: <stem>.<kind><n><suffix> (an unknown kind dies)."""
    if kind not in RETRY_KINDS:
        raise SystemExit(f"[COLLECT_ATTEMPT] ✗ not a retry kind {kind!r}: {RETRY_KINDS}")
    return log.with_name(f"{log.stem}.{kind}{n}{log.suffix}")


def discarded(p: Path) -> bool:  # noqa: V103 — public API for plugins (pod lane)
    """Is this the log of a dropped attempt (a name made by keep_name)?"""
    return bool(_DISCARDED.search(p.name))


#: Kit's crash reporter names the minidump it wrote (breakpad · carb.crashreporter) — the only place its path is stated
_DUMP_LINE = re.compile(r"Wrote dump file '([^']+\.dmp)'")


KIT_DUMP_MARK = "[KIT_DUMP]"


def keep_dump(text: str, log: Path) -> str | None:
    """A session that died with a Kit minidump: copy the dump next to its log as <log stem>.dmp (it lives in the
    job's Kit data dir, which leaves with the scratch — the log dir travels with the piece). Returns the marker line
    or None when the log names no dump. A named dump that is gone → ✗ line (diagnostic · verdict stays)."""
    import shutil

    m = _DUMP_LINE.findall(text)
    if not m:
        return None
    src, dst = Path(m[-1]), log.with_suffix(".dmp")
    if not src.is_file():
        return input_missing(f"{KIT_DUMP_MARK} ✗ the dump the log names is missing {src}")
    shutil.copyfile(src, dst)
    return f"{KIT_DUMP_MARK} kept {dst} ({dst.stat().st_size} B) <- {src}"
