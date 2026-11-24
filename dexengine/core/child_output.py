"""The one reader of a child process's output for a failure line (no imports — any script may use it).

A failure line that quotes only the tail of a child's output drops the child's own marker line (`[TAG] ✗ …`) once
a long path comes before the end — the marker vanishes while the run still fails (a marker must always be printed;
a long dataset path is enough to push it out). Every place that quotes the tail of a marker-printing child's output goes
through `child_failure`; raw tails of children without markers are stated in tests/core/child_tails.tsv
(structural test tests/core/test_child_output.py)."""


def child_failure(text: str | None, n: int = 200) -> str:
    """A child's failure for one log line: every line carrying its own ✗ marker, whole — a marker is never cut;
    a child without a marker line gives its last `n` characters."""
    t = (text or "").strip()
    bad = [ln.strip() for ln in t.splitlines() if "✗" in ln]
    return " | ".join(bad) if bad else t[-n:]
