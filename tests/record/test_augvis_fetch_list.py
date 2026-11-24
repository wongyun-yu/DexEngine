"""The augvis asset fetch takes the canonical generation's list kept in git as its input (augvis.assets.LIST_DIR):
ids from the list, not from today's live ranking; every file checked against its sha256 row (a differing, missing or
unlisted file dies, a file left on disk by an earlier fetch is checked the same way); a missing or other list dies; the
list's SHA256SUMS and manifest.json are installed as they are. The downloads are replaced by fakes (no network)."""

from __future__ import annotations

import hashlib
import json

import pytest

from dexengine.record.looks.augvis import assets as AA
from dexengine.record.looks.augvis import sources as SR

FILES = {  # the files each fake download writes (relative path → bytes)
    "textures/ph_wood/ph_wood_diff_1k.jpg": b"ph diff",
    "textures/ph_wood/ph_wood_rough_1k.jpg": b"ph rough",
    "textures/ph_wood/ph_wood_nor_gl_1k.jpg": b"ph nor",
    "models/ph_mug/ph_mug_1k.usdc": b"ph mug",
    "hdris/room_2k.hdr": b"hdr",
    "textures/Wood001/Wood001_Color.jpg": b"acg color",
    "models/gso/Cup/model.usda": b"#usda 1.0",
    "models/gso/Cup/texture.jpg": b"gso tex",
    "models/gso/Cup/dims.json": b"[0.1, 0.1, 0.1]",
}


def _write(r, prefix: str) -> None:
    for f, b in FILES.items():
        if f.startswith(prefix):
            (r / f).parent.mkdir(parents=True, exist_ok=True)
            (r / f).write_bytes(b)


@pytest.fixture
def fake_list(tmp_path, monkeypatch):
    """A canonical list of one asset per class, sources.ASSET_GEN naming it, fake downloads, live choice refused."""
    lst = tmp_path / "list"
    lst.mkdir()
    raw = "".join(f"{hashlib.sha256(b).hexdigest()}  {f}\n" for f, b in sorted(FILES.items())).encode()
    man = {
        "textures": {
            "ph_wood": {"src": "polyhaven", "res": "1k"},
            "Wood001": {"src": "ambientcg", "res": "1K-JPG"},
        },
        "models": {"ph_mug": {"src": "polyhaven", "res": "1k"}, "gso:Cup": {"src": "gso"}},
        "hdris": {"room": {"src": "polyhaven"}},
        "acg": {"desk": {"acg_Wood001": {"id": "Wood001", "cat": "Wood"}}},
        "gso_pick": ["Cup"],
        "files": len(FILES),
        "bytes": sum(len(b) for b in FILES.values()),
    }
    (lst / "SHA256SUMS").write_bytes(raw)
    (lst / "manifest.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    mraw = (lst / "manifest.json").read_bytes()
    gen = {
        "name": "augvis_test",
        "sums_sha256": hashlib.sha256(raw).hexdigest(),
        "files": len(FILES),
        "manifest_sha256": hashlib.sha256(mraw).hexdigest(),
        "manifest_from": "0" * 64,
    }
    monkeypatch.setattr(SR, "ASSET_GEN", dict(gen, bytes=man["bytes"]))
    monkeypatch.setattr(AA, "LIST_DIR", lst)
    monkeypatch.setattr(AA.AC, "ph_ids", lambda: [("model", "ph_mug", "1k"), ("texture", "ph_wood", "1k")])
    calls = []

    def fake(prefix):
        def f(r, *ids):
            calls.append((prefix, ids))
            _write(r, prefix)

        return f

    monkeypatch.setattr(AA, "ph_fetch", lambda r, k, pid, res: fake(f"{k}s/{pid}")(r, pid))
    monkeypatch.setattr(AA, "hdri_fetch", fake("hdris/"))
    monkeypatch.setattr(AA, "acg_fetch", fake("textures/Wood001"))
    monkeypatch.setattr(AA, "gso_fetch", fake("models/gso/"))

    def live(*_a):
        raise AssertionError("a live choice was made")

    for name in ("pick_acg", "pick_gso", "pick_hdris"):
        monkeypatch.setattr(SR, name, live)
    monkeypatch.setattr(AA, "get_json", live)
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("AUGVIS_ASSETS", str(root))
    return root, lst, calls


def test_the_list_in_git_is_the_canonical_generation():
    """The list kept in git hashes to sources.ASSET_GEN (840 rows · 864,779,498 B · the translated manifest's sha256);
    its manifest holds no Hangul (all values are English)."""
    import re

    man, sums = AA.canonical_list()
    assert len(sums) == SR.ASSET_GEN["files"] == 840 and man["bytes"] == SR.ASSET_GEN["bytes"] == 864779498
    assert AA.LIST_DIR.name == SR.ASSET_GEN["name"]
    raw = (AA.LIST_DIR / "manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SR.ASSET_GEN["manifest_sha256"]
    assert not re.search("[\uac00-\ud7a3\u3131-\u318e]", raw.decode("utf-8"))


def test_the_list_in_git_names_the_catalog_assets():
    """The fixed Poly Haven tables of the code name the same assets as the list (materialise dies otherwise)."""
    man, _ = AA.canonical_list()
    ph = sorted(
        ("texture" if tab == "textures" else "model", pid, rec["res"])
        for tab in ("textures", "models")
        for pid, rec in man[tab].items()
        if rec["src"] == "polyhaven"
    )
    assert ph == AA.AC.ph_ids()


def test_the_canonical_list_is_fetched_and_installed_as_it_is(fake_list):
    root, lst, calls = fake_list
    AA.materialise()
    assert {p for p, _ids in calls} == {
        "textures/ph_wood",
        "models/ph_mug",
        "hdris/",
        "textures/Wood001",
        "models/gso/",
    }
    for f in ("SHA256SUMS", "manifest.json"):
        assert (root / f).read_bytes() == (lst / f).read_bytes()
    assert AA.verify() == len(FILES)
    assert SR.asset_gen_line(root).startswith("[AUG_VIS] assets canonical augvis_test · files 9")


def test_a_file_that_differs_from_its_row_dies(fake_list, monkeypatch):
    root, _lst, _calls = fake_list

    def changed(r, *_ids):
        _write(r, "models/gso/")
        (r / "models/gso/Cup/texture.jpg").write_bytes(b"another encoder")

    monkeypatch.setattr(AA, "gso_fetch", changed)
    with pytest.raises(SystemExit, match=r"sha256 differs 1 \['models/gso/Cup/texture.jpg'\]"):
        AA.materialise()
    assert not (root / "SHA256SUMS").exists()  # nothing installed


def test_a_missing_or_unlisted_file_dies(fake_list, monkeypatch):
    root, _lst, _calls = fake_list
    (root / "textures/Old002").mkdir(parents=True)
    (root / "textures/Old002/Old002_Color.jpg").write_bytes(b"left by another fetch")
    with pytest.raises(SystemExit, match=r"not listed 1 \['textures/Old002/Old002_Color.jpg'\]"):
        AA.materialise()
    (root / "textures/Old002/Old002_Color.jpg").unlink()
    monkeypatch.setattr(AA, "hdri_fetch", lambda r, pid: None)
    (root / "hdris/room_2k.hdr").unlink()
    with pytest.raises(SystemExit, match=r"missing 1 \['hdris/room_2k.hdr'\]"):
        AA.materialise()


def test_a_stale_file_on_disk_is_checked_like_a_new_one(fake_list):
    """A download that skips an existing file (acg_fetch · gso_fetch · fetch without md5) cannot pass a stale copy."""
    root, _lst, _calls = fake_list
    AA.materialise()
    (root / "textures/ph_wood/ph_wood_rough_1k.jpg").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="sha256 differs 1"):
        AA.verify()


def test_a_manifest_other_than_the_stated_one_dies(fake_list):
    """The list's manifest must be the stated (translated) one; a root holding another manifest fails --verify."""
    root, lst, _calls = fake_list
    AA.materialise()
    (root / "manifest.json").write_text('{"before": "translation"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="is not the canonical list's"):
        AA.verify()
    (lst / "manifest.json").write_text('{"bytes": 1}', encoding="utf-8")
    with pytest.raises(SystemExit, match="is not the canonical augvis_test list"):
        AA.materialise()


def test_a_root_with_another_manifest_is_not_rewritten(fake_list):
    """A root that holds another manifest (the canonical root with its original values) dies before any fetch."""
    root, _lst, calls = fake_list
    (root / "manifest.json").write_text('{"original": "values"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="does not rewrite another root's manifest"):
        AA.materialise()
    assert not calls and (root / "manifest.json").read_text(encoding="utf-8") == '{"original": "values"}'


def test_a_missing_or_other_list_dies(fake_list, monkeypatch):
    _root, lst, _calls = fake_list
    (lst / "SHA256SUMS").write_bytes((lst / "SHA256SUMS").read_bytes() + b"0" * 64 + b"  extra/file\n")
    with pytest.raises(SystemExit, match="is not the canonical augvis_test list"):
        AA.materialise()
    (lst / "SHA256SUMS").unlink()
    with pytest.raises(AA.InputMissingError, match="no live choice"):
        AA.materialise()


def test_a_catalog_that_names_other_assets_dies(fake_list, monkeypatch):
    monkeypatch.setattr(AA.AC, "ph_ids", lambda: [("texture", "ph_wood", "1k")])
    with pytest.raises(SystemExit, match="ph_ids"):
        AA.materialise()
