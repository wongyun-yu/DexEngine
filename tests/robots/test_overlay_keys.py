"""Render-overlay customLayerData keys: one owner (core.artifact_paths), read back by the header reader."""

import ast
from pathlib import Path

from dexengine.core import artifact_paths as AP

REPO = Path(__file__).resolve().parents[2]


def _header(key: str) -> str:
    return (
        "#usda 1.0\n(\n    customLayerData = {\n        dictionary " + key + " = {\n"
        '            string body = "applied mtl"\n        }\n        string generator = "x"\n    }\n)\n'
    )


def test_header_reader_reads_the_owned_key(tmp_path):
    f = tmp_path / "o_render.usd"
    f.write_text(_header(AP.OVERLAY_TEX_KEY), encoding="utf-8")
    assert AP.overlay_bake_mark(str(f)) == ("applied", {"body": "applied mtl"})
    f.write_text(_header("pre_rename_render_textures"), encoding="utf-8")
    assert AP.overlay_bake_mark(str(f)) is None  # no legacy reader: an old overlay must be re-baked


def test_writer_and_readers_import_the_keys():
    """No module spells the key names itself (the writer, the cover check and the header reader share them)."""
    from dexengine.robots.objects import tex_packs

    assert tex_packs.REC_STATUS == AP.OVERLAY_TEX_KEY
    keys = {AP.OVERLAY_TEX_KEY, AP.OVERLAY_TEX_SOURCES_KEY}
    for rel in ("dexengine/robots/objects/render_overlay.py", "dexengine/robots/objects/tex_packs.py"):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not lits & keys, rel
    src = (REPO / "dexengine/robots/objects/render_overlay.py").read_text(encoding="utf-8")
    assert "AP.OVERLAY_TEX_KEY" in src and "AP.OVERLAY_TEX_SOURCES_KEY" in src
