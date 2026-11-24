"""Proofs that a recipe switch was applied: every batch's logs must carry the switch's marker, else the batch is
invalid.

Each check reads the session logs `<runs_root>/<tag>_*/session.log` (or the DP server log) of one batch and raises
`InvalidBatch` with the reason. Markers come from the session (experiments.sim_eval), the DP server and dexengine.
"""

from __future__ import annotations

import re
from pathlib import Path


class InvalidBatch(SystemExit):
    """A batch whose logs do not prove the switch; its results are not counted."""


def session_logs(runs_root: Path, tag: str) -> list[Path]:
    """Session logs of one batch."""
    return sorted(p / "session.log" for p in runs_root.glob(f"{tag}_*") if (p / "session.log").is_file())


def _text(p: Path) -> str:
    return p.read_text(errors="replace")


def const_state(server_log: Path, tag: str) -> None:
    """DP constant-state normalisation fix: the server prints `[DP_CONST_STATE] dims=` at start."""
    if not server_log.is_file() or not re.search(r"^\[DP_CONST_STATE\] dims=", _text(server_log), re.M):
        raise InvalidBatch(f"batch {tag}: server log has no [DP_CONST_STATE] marker")


def sampler(server_log: Path, tag: str, sampler_steps: str) -> None:
    """DP sampler override: the server prints `[E5_DP_SAMPLER] <S> steps=<N> `."""
    if not server_log.is_file() or not re.search(
        rf"^\[E5_DP_SAMPLER\] {re.escape(sampler_steps)} ", _text(server_log), re.M
    ):
        raise InvalidBatch(f"batch {tag}: server log has no '[E5_DP_SAMPLER] {sampler_steps}' marker")


def texture_overlay(runs_root: Path, tag: str, root: str) -> str:
    """Texture overlay: every session log names the overlay root (`[TEX_OVERLAY] … root=<root> … shells=<n>/…`)."""
    n = shells = 0
    for lg in session_logs(runs_root, tag):
        n += 1
        lines = [x for x in _text(lg).splitlines() if x.startswith("[TEX_OVERLAY]")]
        if not lines:
            raise InvalidBatch(f"batch {tag} {lg.parent.name}: no [TEX_OVERLAY] line")
        if any(f"root={root} " not in x for x in lines):
            raise InvalidBatch(f"batch {tag} {lg.parent.name}: overlay root is not {root}")
        for x in lines:
            m = re.search(r" shells=(\d+)/", x)
            shells += int(m.group(1)) if m else 0
    return f"[TEX_OVERLAY] {tag} sessions {n} · shells spawned {shells} · root {root}"


def spider_start(runs_root: Path, tag: str, start_file: str) -> str:
    """SPIDER start override: every session log carries `[E5_EVAL] [E5_START] source=spider_`."""
    n = 0
    for lg in session_logs(runs_root, tag):
        n += 1
        if not re.search(r"^\[E5_EVAL\] \[E5_START\] source=spider_", _text(lg), re.M):
            raise InvalidBatch(f"batch {tag} {lg.parent.name}: no [E5_START] source=spider_ line")
    return f"[E5_START] {tag} sessions {n} · start file {start_file}"


def place_marker(run_dirs: list[Path], pattern: str) -> list[Path]:
    """P&P place marker: run folders whose session log lacks `[PLACE_MARKER] <pattern>`."""
    return [d for d in run_dirs if f"[PLACE_MARKER] {pattern}" not in _session(d)]


def extra_attach(run_dirs: list[Path]) -> tuple[list[Path], int]:
    """Extra objects resting on the main object: (run folders without an `[EXTRA_ATTACH]` line, scenes that
    attached).
    """
    missing = [d for d in run_dirs if "[EXTRA_ATTACH]" not in _session(d)]
    attached = sum(1 for d in run_dirs if re.search(r"\[EXTRA_ATTACH\] [^ ]+→", _session(d)))
    return missing, attached


def _session(d: Path) -> str:
    try:
        return _text(d / "session.log")
    except OSError:
        return ""


def place_marker_pattern(style: str) -> str:
    """Session-log pattern of a place-marker style (`pad-red` → `style=pad color=red obj=`), validated by its owner."""
    from dexengine.record.render import markers as PM

    v = PM._valid(style, "eval_run")
    s, c = (v.split("-", 1) + [None])[:2]
    return f"style={s}" + (f" color={c}" if c else "") + " obj="
