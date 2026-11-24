"""Outcome tokens of the frame-0 repair worker.

The worker (repair.py, run inside the GPU workers) prints these; prejudge.finish classifies a clip from the worker
output by the tokens alone. Each token and each line that carries one is defined here once, so writer and reader
cannot drift apart. No token may contain the substring SKIP except the two SKIP statuses (finish tests `SKIP in out`).
"""

from __future__ import annotations

#: status prefix of a side abandoned on its eval budget (checked before SKIP: it contains it)
SKIP_BUDGET = "SKIP_BUDGET"
#: status prefix of a side skipped because its penetration is shallow
SKIP = "SKIP"
#: no solution brought the frame-0 overlap to 0 within the hard constraints
OVERLAP_LEFT = "[F0_OVERLAP_LEFT]"
#: the required hold padding is over the cap: the repair is refused
PAD_TOO_BIG = "[F0_PAD_TOO_BIG]"
#: writing the repair in place failed (the hold pad stays)
INPLACE_FAILED = "[F0_INPLACE_FAILED]"
#: summary line: sides left unrepaired (the hold pad stays)
UNREPAIRED = "[F0_UNREPAIRED]"
#: the in-place repair record already exists: the sequence is passed over
ALREADY_REPAIRED = "[F0_ALREADY_REPAIRED]"

PAD_CAP = 60


def pad_too_big(P: int) -> str:
    """Reason of a refused repair whose hold padding P (frames) is over PAD_CAP."""
    return f"{PAD_TOO_BIG} required padding {P} frames > cap {PAD_CAP} — the repair is too large"


def already_repaired_line(seq: str) -> str:
    return f"{seq} — {ALREADY_REPAIRED} in-place repair record exists, skipped (--force repairs again)"


def inplace_failed_line(err: str) -> str:
    return f"  -> {INPLACE_FAILED} in-place write failed: {err} (hold pad kept)"


def unrepaired_line(n: int) -> str:
    return f"[frame0_repair] {UNREPAIRED} {n} unrepaired (hold pad kept):"
