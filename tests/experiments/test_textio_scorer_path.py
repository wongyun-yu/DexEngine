"""experiments.textio.read_lines equals iterating the open file; the shared scorer still runs by file path."""

import subprocess
import sys
from pathlib import Path

from experiments.textio import read_lines

REPO = Path(__file__).resolve().parents[2]


def test_read_lines_is_file_iteration(tmp_path):
    p = tmp_path / "x.jsonl"
    # universal newlines, a JSON line separator inside a value, a form feed and no final newline
    p.write_bytes('{"a": "x y"}\r\nb\x0cc\rlast'.encode())
    with open(p, encoding="utf-8") as fh:
        want = list(fh)
    assert read_lines(p, encoding="utf-8") == want
    assert want[-1] == "last"


def test_score_runs_by_path_isolated():
    """score.py loads its sibling score_chord.py by path when started as `python -I score.py` (FlashCHORD venv)."""
    r = subprocess.run(
        [sys.executable, "-I", str(REPO / "experiments/scoring/score.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert "usage:" in r.stdout
