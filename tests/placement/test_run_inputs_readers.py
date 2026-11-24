"""Every reader name the run-inputs statement writes is a READERS key (the bundle table dies on an unknown reader, so a
reader added without its entry breaks every bundle build that meets it)."""

from __future__ import annotations

import re
from pathlib import Path

from dexengine.placement.bundle import run_inputs as RI


def test_every_reader_written_is_declared():
    src = Path(RI.__file__).read_text(encoding="utf-8")
    used = set(re.findall(r'"((?:entry|kit|verify|launch|render):[a-z0-9-]+)"', src))
    assert used and not used - set(RI.READERS), sorted(used - set(RI.READERS))
    assert "kit:scene-rules" in used
