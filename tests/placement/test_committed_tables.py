"""The committed render overlay manifest holds the values its writer emits (English bake marks)."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs/record/render_overlay_manifest.json"
HANGUL = re.compile("[\uac00-\ud7a3\u3131-\u318e]")


def test_overlay_manifest_marks():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))["overlays"]
    assert {v["bake"] for v in m.values()} <= {"applied", "solid", "none"}
    assert not any(HANGUL.search(x) for v in m.values() for x in v["links"].values())
