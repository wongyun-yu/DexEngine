"""The render overlay manifest in force (artifact_paths.overlay_manifest_rule): a tree with OPS_DIR checks the
canonical git list and dies without it; a tree without it checks the list its overlay bake wrote under the data
root (robots.objects.overlay_manifest.record — the overlays that bake wrote, merged by key); DEXENGINE_OVERLAY_MANIFEST
names a project's own list. A stale shell, an overlay not in the list or a missing list (with overlays spawned) dies.
The tree shape is monkeypatched (OPS_DIR), so these hold in a tree with and without the deployment layer."""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from dexengine.core import artifact_paths as TP

MARKED = b"""#usda 1.0
(
    customLayerData = {
        dictionary dexengine_render_textures = {
            string base_link = "applied(raw)"
        }
    }
    subLayers = [
        @./object.usd@
    ]
)
"""
#: A stale shell (611 B overlays without a material) — the same object, mark 'none'.
SHELL = MARKED.replace(b'"applied(raw)"', b'"none(no texture)"')


def _overlay(root, ds="capture_raw", obj="white_eraser", body=MARKED):
    p = root / "data" / ds / "usd" / obj / f"object{TP.RENDER_USD_SUFFIX}.usd"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


def _manifest(path, overlay, md5=None):
    key = TP.overlay_manifest_key(str(overlay))
    path.write_text(json.dumps({"overlays": {key: {"md5": md5 or hashlib.md5(overlay.read_bytes()).hexdigest()}}}))


@pytest.fixture
def private_tree(tmp_path, monkeypatch):
    """A tree with the operations layer present, the canonical list at a temporary place."""
    (tmp_path / "site_layer").mkdir()
    monkeypatch.setattr(TP, "OPS_DIR", str(tmp_path / "site_layer"))
    monkeypatch.setattr(TP, "OVERLAY_MANIFEST_FILE", str(tmp_path / "site_layer" / "render_overlay_manifest.json"))
    monkeypatch.delenv(TP.OVERLAY_MANIFEST_ENV, raising=False)
    return tmp_path


@pytest.fixture
def public_tree(tmp_path, monkeypatch):
    """A tree without an operations layer; the data root is a temporary folder."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(TP, "OPS_DIR", str(tmp_path / "no_site_layer"))
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(root))
    monkeypatch.delenv(TP.OVERLAY_MANIFEST_ENV, raising=False)
    return root


def test_a_private_tree_checks_the_canonical_list(private_tree):
    assert TP.overlay_manifest_rule() == ("canonical", TP.OVERLAY_MANIFEST_FILE)
    ov = _overlay(private_tree)
    _manifest(private_tree / "site_layer" / "render_overlay_manifest.json", ov)
    tail = TP.overlay_manifest_check([str(ov)], None)
    assert tail.startswith("ok 1/1 · manifest md5 ") and "list" not in tail  # the marker text of before


def test_a_private_tree_without_its_list_dies_and_does_not_switch(private_tree):
    """Deleting the canonical list does not move a tree with the operations layer to the data-root rule."""
    ov = _overlay(private_tree)
    with pytest.raises(FileNotFoundError, match="a tree with the deployment layer"):
        TP.overlay_manifest_check([str(ov)], None)
    with pytest.raises(FileNotFoundError, match="a tree with the deployment layer"):
        TP.overlay_manifest_check([], None)


def test_the_canonical_list_still_refuses_an_overlay_it_does_not_hold(private_tree):
    ov = _overlay(private_tree)
    _manifest(private_tree / "site_layer" / "render_overlay_manifest.json", ov)
    other = _overlay(private_tree, obj="red_cup")
    with pytest.raises(RuntimeError, match="manifest none"):
        TP.overlay_manifest_check([str(ov), str(other)], None)


def test_a_project_list_admits_its_own_overlay_and_names_the_file(tmp_path, monkeypatch):
    ov = _overlay(tmp_path)
    mf = tmp_path / "project_overlays.json"
    _manifest(mf, ov)
    monkeypatch.setenv(TP.OVERLAY_MANIFEST_ENV, str(mf))
    tail = TP.overlay_manifest_check([str(ov)], None)
    assert tail.startswith("ok 1/1") and f"project list {mf}" in tail


def test_a_project_list_still_catches_a_stale_copy(tmp_path, monkeypatch):
    ov = _overlay(tmp_path)
    mf = tmp_path / "project_overlays.json"
    _manifest(mf, ov, md5="0" * 32)
    monkeypatch.setenv(TP.OVERLAY_MANIFEST_ENV, str(mf))
    with pytest.raises(RuntimeError, match="differing from the manifest"):
        TP.overlay_manifest_check([str(ov)], None)


def test_a_stated_missing_project_list_dies(tmp_path, monkeypatch):
    ov = _overlay(tmp_path)
    monkeypatch.setenv(TP.OVERLAY_MANIFEST_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="absent.json"):
        TP.overlay_manifest_check([str(ov)], None)


def test_a_public_tree_checks_what_its_own_bake_recorded(public_tree):
    from dexengine.robots.objects import overlay_manifest as OM

    rule, mf = TP.overlay_manifest_rule()
    assert rule == "data-root" and mf == str(public_tree / TP.DATA_ROOT_OVERLAY_MANIFEST)
    assert TP.overlay_manifest_check([], None).startswith("ok 0/0")  # nothing baked, nothing spawned: no list needed
    ov = _overlay(public_tree)
    with pytest.raises(FileNotFoundError, match="a tree without the deployment layer"):  # an overlay but no list
        TP.overlay_manifest_check([str(ov)], None)
    assert OM.record([str(ov)]).startswith("[OVERLAY_MANIFEST] recorded 1 · list 1 · data-root list")
    tail = TP.overlay_manifest_check([str(ov)], None)
    assert tail.startswith("ok 1/1") and f"data-root list {mf} (no deployment)" in tail
    ov.write_bytes(SHELL)  # (2) a stale shell in place of the baked overlay
    with pytest.raises(RuntimeError, match="differing from the manifest"):
        TP.overlay_manifest_check([str(ov)], None)
    ov.write_bytes(MARKED)
    other = _overlay(public_tree, obj="red_cup")  # (3) an overlay this tree's bake never recorded
    with pytest.raises(RuntimeError, match="manifest none"):
        TP.overlay_manifest_check([str(ov), str(other)], None)
    OM.record([str(other)])  # baked: merged by key, the first entry kept
    assert TP.overlay_manifest_check([str(ov), str(other)], None).startswith("ok 2/2")


def test_a_bake_does_not_write_the_canonical_list(private_tree):
    from dexengine.robots.objects import overlay_manifest as OM

    ov = _overlay(private_tree)
    assert "canonical list not written by a bake" in OM.record([str(ov)])
    assert not (private_tree / "site_layer" / "render_overlay_manifest.json").exists()


def test_an_overlay_without_a_mark_block_is_not_recorded(public_tree):
    from dexengine.robots.objects import overlay_manifest as OM

    ov = _overlay(public_tree, body=b"#usda 1.0\n")
    with pytest.raises(SystemExit, match="no texture mark block"):
        OM.record([str(ov)])


def _bake_main(monkeypatch, root, argv):
    """render_overlay.main() over one object with its bake faked (bake_one writes a marked overlay)."""
    import json as _json
    import sys as _sys

    from dexengine.robots.objects import render_overlay as RO

    usd = root / "data" / "capture_raw" / "usd" / "white_eraser" / "object.usd"
    usd.parent.mkdir(parents=True, exist_ok=True)
    usd.write_bytes(b"#usda 1.0\n")
    spec = root / "spec.json"
    spec.write_text(_json.dumps({"objects": {}}))
    monkeypatch.setattr(RO.RR, "axis_output", lambda _name: str(spec))
    monkeypatch.setattr(RO.OCH, "targets", lambda _a: [("capture_raw", "white_eraser", 0, usd)])
    monkeypatch.setattr(RO.RTP, "setup", lambda _a, _ds: None)

    def bake_one(ds, oid, art, f, specs, scratch, dry, packs=None, overlay_root=None):
        out = TP.render_usd_file(str(f)) if overlay_root is None else TP.overlay_file(str(f), overlay_root)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(MARKED)
        return {"written": out, "links": {"base_link": {"status": "applied(raw)"}}}

    monkeypatch.setattr(RO, "bake_one", bake_one)
    monkeypatch.setattr(RO, "_part_objs", lambda *_a: None)
    monkeypatch.setattr(RO, "mesh_table", lambda *_a: {})
    monkeypatch.setattr(RO, "_write_link_table", lambda *_a: None)
    monkeypatch.setattr(RO, "_write_usd_table", lambda *_a: None)
    monkeypatch.setattr(_sys, "argv", ["render_overlay", "--dataset", "capture_raw", "--out", str(root / "t"), *argv])
    return RO.main(), TP.overlay_manifest_key(str(usd.with_name(f"object{TP.RENDER_USD_SUFFIX}.usd")))


def test_the_bake_entry_records_what_it_wrote_in_a_public_tree(public_tree, monkeypatch, capsys):
    """The CLI itself (render_overlay.main) merges the overlay it wrote into the data-root list; the render passes."""
    rc, key = _bake_main(monkeypatch, public_tree, [])
    assert rc == 0 and "[OVERLAY_MANIFEST] recorded 1 · list 1 · data-root list" in capsys.readouterr().out
    mf = public_tree / TP.DATA_ROOT_OVERLAY_MANIFEST
    assert key in json.loads(mf.read_text())["overlays"]
    ov = public_tree / key
    assert TP.overlay_manifest_check([str(ov)], None).startswith("ok 1/1")


def test_a_preview_bake_writes_no_list(public_tree, monkeypatch, capsys):
    rc, _key = _bake_main(monkeypatch, public_tree, ["--overlay-root", str(public_tree / "preview")])
    assert rc == 0 and "preview root" in capsys.readouterr().out
    assert not (public_tree / TP.DATA_ROOT_OVERLAY_MANIFEST).exists()


def test_the_list_and_its_generator_have_one_place_on_every_line():
    """v0 and v0.1 place the canonical list and its generator the same way (v0.1 = v0 + its changelog): the list under
    configs/record, the generator robots.objects.overlay_manifest, which the list names."""
    from dexengine.core.repo import REPO_ROOT
    from dexengine.robots.objects import overlay_manifest as OM

    rel = os.path.relpath(TP.OVERLAY_MANIFEST_FILE, REPO_ROOT)
    assert rel == "configs/record/render_overlay_manifest.json"
    assert OM.GENERATOR == "dexengine.robots.objects.overlay_manifest"
    with open(TP.OVERLAY_MANIFEST_FILE, encoding="utf-8") as f:
        assert json.load(f)["generator"] == OM.GENERATOR
