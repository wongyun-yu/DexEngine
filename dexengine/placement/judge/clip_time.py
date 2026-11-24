"""Per-clip time marker of the judge, `[CLIP_T]` — owner of its format, stages and result rule (written once per
clip by judge.core.main).

  [CLIP_T] <clip> anchor/exhaust/guard/confirm <a>/<b>/<g>/<c> other <o> total <s> result <state>
  total        clip start → end wall clock = the second column of judged.tsv (--end-marks)
               = anchor + exhaust + other — sequential stages, they do not overlap.
  guard · confirm  guard and confirm are sub-stages **inside** anchors / exhaustive (they may overlap each
               other), so they are not added to the total. Confirm = candidate checks (dp_accept ·
               track_full_multi). Only the judge process's wall clock is measured: guards prefetched by forked workers
               (guard_pool) count only for the time the judge waited on them.
  result       ADOPTED when adopted, else the clip's reject rows (noconfirm) folded with the status ledger's function
               (ledger_states.noconfirm_fold), MISSING when there are none.
The words are part of the marker format (parsed by `parse`).
"""

from __future__ import annotations

import functools
import re
import time

from dexengine.placement import ledger_states as LS

TAG = "[CLIP_T]"
SEQ = ("anchor", "exhaust")
INNER = ("guard", "confirm")
_acc: dict[str, float] = {}
_on: set[str] = set()
_res: list[str | None] = [None]


def start() -> None:
    """Start of one clip — clear the stage times and the folded result."""
    _acc.clear()
    _on.clear()
    _res[0] = None


def timed(stage: str):
    """Method decorator — when the same stage is entered again inside, only the outer call counts."""
    if stage not in SEQ + INNER:
        raise ValueError(f"{TAG} unknown stage {stage!r} — {SEQ + INNER}")

    def deco(fn):
        @functools.wraps(fn)
        def w(*a, **k):
            if stage in _on:
                return fn(*a, **k)
            _on.add(stage)
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                _acc[stage] = _acc.get(stage, 0.0) + time.perf_counter() - t
                _on.discard(stage)

        return w

    return deco


class Tap:
    """Reject-row file writer that folds this clip's state while writing (line: clip · verdict · reason)."""

    def __init__(self, fh):
        self._fh = fh

    def write(self, line: str):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            _res[0] = LS.noconfirm_fold(_res[0], p[1], p[2] if len(p) > 2 else "")
        return self._fh.write(line)

    def close(self):
        return self._fh.close()


def line(clip: str, total: float, adopted: bool) -> str:
    v = [_acc.get(s, 0.0) for s in SEQ + INNER]
    res = "ADOPTED" if adopted else (_res[0] or LS.MISSING)
    return (
        f"{TAG} {clip} anchor/exhaust/guard/confirm {'/'.join(f'{x:.1f}' for x in v)} other {total - v[0] - v[1]:.1f} "
        f"total {total:.1f} result {res}"
    )


_RE = re.compile(
    re.escape(TAG) + r" (.+?) anchor/exhaust/guard/confirm ([-\d./]+) other (-?[\d.]+) total ([\d.]+) result (\S+)$"
)


def parse(s: str) -> dict | None:
    """One marker line → {clip, stages, other, total, result}; None when it is not a marker."""
    m = _RE.search(s.rstrip("\n"))
    if not m:
        return None
    return {
        "clip": m[1],
        "stages": dict(zip(SEQ + INNER, map(float, m[2].split("/")), strict=False)),
        "other": float(m[3]),
        "total": float(m[4]),
        "result": m[5],
    }
