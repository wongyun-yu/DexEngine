"""Articulated USDs live in their own layer, one folder per bake generation (layout.ARTIC_BAKE_GEN): the path owner,
the readers that follow it, the bake's default output, and the pin that ties the generation name to the bake rule."""

import argparse
import ast
import hashlib
import inspect
import io
import tokenize
from pathlib import Path

import pytest

from dexengine.core import artifact_paths as AP
from dexengine.core import layout as L
from dexengine.data.objects import usd_places as UP
from dexengine.robots.objects import part_mass as PM

REPO = Path(__file__).resolve().parents[2]
DS = "oakink2"

#: generation name → fingerprint of the articulated bake rule (part masses · link bodies) it was named for. A rule
#: change fails test_generation_name_follows_the_bake_rule: take a new ARTIC_BAKE_GEN (a new folder — the old bake
#: stays for the code that reads it) and add its row here.
RULE_PINS = {"pmvol1": "bc28425c1b3e"}
RULE_FUNCS = ("create_articulated_object_usd", "_link_masses", "_bake_total", "_part_body")


def _code_text(src: str) -> str:
    """Source text without docstrings, comments and blank lines — raw characters cut by node / token positions, so the
    text is the same under every Python version (ast.dump and ast.unparse are not)."""
    lines = src.splitlines(keepends=True)
    cut = set()  # (row, col) character positions to drop
    for n in ast.walk(ast.parse(src)):
        b = getattr(n, "body", None)
        if isinstance(b, list) and b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant):
            if isinstance(b[0].value.value, str):
                d = b[0]
                for r in range(d.lineno, d.end_lineno + 1):
                    c0 = d.col_offset if r == d.lineno else 0
                    c1 = d.end_col_offset if r == d.end_lineno else len(lines[r - 1])
                    cut.update((r, c) for c in range(c0, c1))
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT:
            cut.update((t.start[0], c) for c in range(t.start[1], t.end[1]))
    kept = ("".join(ch for c, ch in enumerate(ln) if (r, c) not in cut).rstrip() for r, ln in enumerate(lines, 1))
    return "\n".join(k for k in kept if k.strip())


def rule_fingerprint() -> str:
    """sha256 (12 hex) of the code text (no docstrings · comments) of the part_mass rule and the articulated bake
    functions."""
    src = (REPO / "dexengine/robots/objects/usd_build.py").read_text(encoding="utf-8")
    fns = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    parts = [_code_text(inspect.getsource(PM))]
    parts += [_code_text(ast.get_source_segment(src, fns[f])) for f in RULE_FUNCS]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def test_layer_is_one_folder_per_generation():
    assert L.render("usd_artic", DS) == f"data/{DS}/usd_artic/{L.ARTIC_BAKE_GEN}"
    assert L.render("usd", DS) == f"data/{DS}/usd"  # the rigid layer (and the older bake's folders) do not move


def test_generation_name_follows_the_bake_rule():
    got = rule_fingerprint()
    assert L.ARTIC_BAKE_GEN in RULE_PINS, f"no rule pin for generation {L.ARTIC_BAKE_GEN!r}"
    assert RULE_PINS[L.ARTIC_BAKE_GEN] == got, (
        f"the articulated bake rule changed (fingerprint {got}, pinned {RULE_PINS[L.ARTIC_BAKE_GEN]} for "
        f"{L.ARTIC_BAKE_GEN!r}): take a new layout.ARTIC_BAKE_GEN and pin it here — never re-bake into the old folder"
    )


def test_usd_file_articulated_lives_in_the_generation_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    tp = AP.TacoPaths(DS)
    art = Path(tp.usd_file("a__b__hinge", articulated=True))
    assert art == tmp_path / f"data/{DS}/usd_artic/{L.ARTIC_BAKE_GEN}/a__b__hinge/{AP.TacoPaths.ARTIC_USD_NAME}"
    assert Path(tp.usd_file("box")) == tmp_path / f"data/{DS}/usd/box/object.usd"
    # negative: the older bake's place (usd layer) is never the articulated answer
    assert art != tmp_path / f"data/{DS}/usd/a__b__hinge/{AP.TacoPaths.ARTIC_USD_NAME}"
    assert Path(L.cooked_hull_file(str(art))).parent == art.parent  # the cooked dump follows its USD


def _tree(tmp_path):
    usd = tmp_path / f"data/{DS}/usd"
    gen = tmp_path / f"data/{DS}/usd_artic/{L.ARTIC_BAKE_GEN}"
    for f in (usd / "box/object.usd", usd / "old__x__hinge/object_artic.usd", gen / "new__x__hinge/object_artic.usd"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("#usda 1.0\n")
    return usd, gen


def test_object_usds_reads_each_layer_for_its_own_kind(tmp_path, monkeypatch):
    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    _usd, gen = _tree(tmp_path)
    got = UP.object_usds(DS)
    assert [(m, a) for m, a, _f in got] == [("box", False), ("new__x__hinge", True)]  # old-place artic USD ignored
    assert Path(got[1][2]).parent == gen / "new__x__hinge"


def test_object_usds_dies_without_the_usd_layer(tmp_path, monkeypatch):
    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="usd layer"):
        UP.object_usds(DS)


def test_cook_targets_and_orphans_follow_the_layers(tmp_path, monkeypatch, capsys):
    from dexengine.robots.objects import cooked_hull as CH

    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    usd, gen = _tree(tmp_path)
    (usd / "old__x__hinge/object_artic.cooked.npz").write_bytes(b"")  # the older bake's dump: not this code's
    (gen / "gone__y__slide").mkdir()
    (gen / "gone__y__slide/object_artic.cooked.npz").write_bytes(b"")  # a dump whose USD is gone: orphan
    monkeypatch.setattr(CH.OH, "cooked_problem", lambda f: None)
    a = argparse.Namespace(all=False, dataset=DS, ids=None, ids_file=None)
    rows = CH.targets(a)
    assert [(r[1], r[2]) for r in rows] == [("box", False), ("new__x__hinge", True)]
    assert not CH.verify(a, rows)
    out = capsys.readouterr().out
    assert "orphan=1" in out and "gone__y__slide" in out and "old__x__hinge" not in out


def test_bake_default_output_is_the_generation_layer():
    tree = ast.parse((REPO / "dexengine/robots/objects/usd_bake.py").read_text(encoding="utf-8"))
    assigns = {
        t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
    }
    assert ast.unparse(assigns["_OUT_LAYER"]) == "'usd_artic' if args_cli.articulated else 'usd'"
    assert ast.unparse(assigns["_DEFAULT_OUTPUT_DIR"]) == "_layer_root(_OUT_LAYER, args_cli.dataset)"
    fro = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_frozen_output_reason")
    assert "('usd', 'usd_artic')" in ast.unparse(fro)  # a frozen dataset's articulated layer is refused too


def test_train_ready_reads_the_articulated_layer():
    from dexengine.placement.train_ready import assets as A

    assert A.ARTIC_ASSET[1] == "usd_artic"
    src = inspect.getsource(A.missing_assets)
    assert "L.render(layer, ds)" in src and '"data", ds, layer' not in src


def test_layer_walkers_read_the_one_layer_table():
    """The tools that walk object USD layers under another root read usd_places (both layers), not the usd layer
    alone."""
    for f, want in (
        ("dexengine/robots/objects/overlay_manifest.py", "UP.usd_roots(ds)"),
        ("dexengine/robots/objects/cooked_hull.py", "UP.usd_roots(ds)"),
    ):
        src = (REPO / f).read_text(encoding="utf-8")
        assert want in src and 'render("usd", ' not in src, f


def test_overlay_names_follow_the_layer(tmp_path, monkeypatch):
    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    assert UP.usd_name(DS, False, render=True) == "object_render.usd"
    assert UP.usd_name(DS, True, render=True) == "object_artic_render.usd"
    assert [a for a, _r in UP.usd_roots(DS)] == [False, True]


def test_overlay_manifest_scan_lists_both_layers(tmp_path, monkeypatch):
    """The manifest scan lists a rigid overlay and an articulated one (usd_artic layer, its own overlay name); a scan
    of the usd layer alone would miss the articulated one."""
    from dexengine.robots.objects import overlay_manifest as OM

    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    marked = b"#usda 1.0\n(\n    customLayerData = {\n        dictionary dexengine_render_textures = {\n"
    marked += b'            string base_link = "applied(raw)"\n        }\n    }\n)\n'
    wrote = []
    for art, root in UP.usd_roots(DS):
        p = root / ("art_obj" if art else "rigid_obj") / UP.usd_name(DS, art, render=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(marked)
        wrote.append(AP.overlay_manifest_key(str(p)))
    listed = OM.build()["overlays"]
    assert all(k in listed for k in wrote) and len(wrote) == 2, (wrote, sorted(listed))
