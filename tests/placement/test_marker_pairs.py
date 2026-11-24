"""Placement log lines that ops tools parse: the producer's f-string (filled) through the reader's own regex."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _fstring(path: Path, needle: str, fill: dict) -> str:
    """The one f-string of `path` containing `needle`, placeholders replaced by fill[source] (source = ast.unparse)."""
    hits = []
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.JoinedStr):
            text = "".join(
                v.value if isinstance(v, ast.Constant) else "\0" + ast.unparse(v.value) + "\0" for v in n.values
            )
            if needle in text:
                hits.append(re.sub(r"\0([^\0]*)\0", lambda m: fill[m.group(1)], text))
    assert len(hits) == 1, hits
    return hits[0]


def test_aug_cells_done_line_is_what_the_record_augment_counts():
    """placement.judge.aug_cells prints done_marker; record.augment.cells checks the clip count via done_clips."""
    from dexengine.placement.judge import aug_cells as JAC

    log = "\n".join(["[AUG_CELLS] mode=L start", JAC.done_marker("L", 7, 31, 12), JAC.done_marker("D", 3, 9)]) + "\n"
    assert JAC.done_clips(log, "L") == ["7"] and JAC.done_clips(log, "D") == ["3"] and JAC.done_clips(log, "off") == []
    src = (ROOT / "dexengine/record/augment/cells_steps.py").read_text()
    assert "JAC.done_clips(text, mode)" in src  # the reader uses the writer's parser, not its own pattern
