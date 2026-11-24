"""Robot and hand source files (configs/robots/vendor/): every file a builder reads is listed with its source, the
manufacturer files are not in git, and a listed file whose bytes differ is never written."""

from __future__ import annotations

import hashlib
import subprocess

import pytest

from dexengine.core import hands, paths
from dexengine.robots import vendor


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True, check=True).stdout


def test_manifest_well_formed():
    src, rows = vendor.sources(), vendor.rows()
    assert rows and {r.source for r in rows} <= set(src)
    for name in src:
        assert any(r.source == name for r in rows), f"source {name} lists no file"


def test_every_vendor_document_of_every_hand_is_listed():
    """Structure rule: a HandSpec vendor URDF / MJCF / mesh folder that no source lists is a file nobody can get."""
    dests = {r.dest for r in vendor.rows()}
    folders = {d.rsplit("/", 1)[0] for d in dests}
    for name, hs in hands.HAND_REGISTRY.items():
        if not hs:
            continue
        for side in ("right", "left"):
            for doc in (hs.vendor_urdf, hs.vendor_mjcf):
                if doc:
                    assert doc.format(side=side) in dests, f"{name}: {doc} not listed"
            if hs.vendor_meshes:
                assert hs.vendor_meshes.format(side=side) in folders, f"{name}: {hs.vendor_meshes} not listed"


def test_manufacturer_files_are_not_tracked():
    """A github-sourced destination inside this checkout is ignored and untracked; a repository-sourced file is
    tracked at its source root."""
    src = vendor.sources()
    tracked = set(_git("ls-files", "assets", "configs").splitlines())
    for r in vendor.rows():
        if src[r.source]["kind"] == "github":
            assert r.dest not in tracked, r.dest
            if not r.dest.startswith("data/"):
                rc = subprocess.run(["git", "check-ignore", "-q", r.dest], cwd=paths.REPO_ROOT).returncode
                assert rc == 0, f"{r.dest} is not ignored"
        else:
            assert f"{src[r.source]['root']}/{r.path}" in tracked, r.path


def _local_manifest(tmp_path, data: bytes, listed: bytes):
    vd = tmp_path / "vendor"
    vd.mkdir()
    root = tmp_path / "shipped"
    root.mkdir()
    (root / "a.bin").write_bytes(data)
    (vd / "sources.yaml").write_text(f"local:\n  kind: repository\n  root: {root}\n")
    dest = f"{tmp_path.name}_out/a.bin"
    (vd / "files.tsv").write_text(f"local\t{dest}\ta.bin\t{hashlib.sha256(listed).hexdigest()}\n")
    return vd, dest


def test_fetch_places_and_skips(tmp_path, monkeypatch):
    vd, dest = _local_manifest(tmp_path, b"abc", b"abc")
    monkeypatch.setattr(vendor, "dest_path", lambda r: tmp_path / r.dest)
    assert vendor.fetch(vendor_dir=vd) == {"placed": 1, "present": 0}
    assert (tmp_path / dest).read_bytes() == b"abc"
    assert vendor.fetch(vendor_dir=vd) == {"placed": 0, "present": 1}
    assert vendor.check(vendor_dir=vd) == []


def test_fetch_refuses_other_bytes(tmp_path, monkeypatch):
    """Negative: the source bytes differ from the listed sha256 → nothing is written and the run dies."""
    vd, dest = _local_manifest(tmp_path, b"abc", b"abd")
    monkeypatch.setattr(vendor, "dest_path", lambda r: tmp_path / r.dest)
    with pytest.raises(SystemExit, match="sha256"):
        vendor.fetch(vendor_dir=vd)
    assert not (tmp_path / dest).exists()
    assert vendor.check(vendor_dir=vd) == [f"missing {dest}"]


def test_target_root_refuses_a_data_root(tmp_path, monkeypatch):
    """Negative: a --data-root that holds generated data stops without --allow-data-root; a fresh root is taken."""
    monkeypatch.setenv(paths.data_layout.DATA_ROOT_ENV, str(tmp_path))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert vendor.target_root(str(fresh), False) == fresh.resolve()
    gen = tmp_path / "gen"
    (gen / "data" / "taco" / "trees").mkdir(parents=True)
    with pytest.raises(SystemExit, match="--allow-data-root"):
        vendor.target_root(str(gen), False)
    assert vendor.target_root(str(gen), True) == gen.resolve()


def test_target_root_moves_data_destinations(tmp_path, monkeypatch):
    """A `data/…` destination lands under --data-root even when the ambient data root is another one."""
    other = tmp_path / "other"
    (other / "data" / "x" / "trees").mkdir(parents=True)
    monkeypatch.setenv(paths.data_layout.DATA_ROOT_ENV, str(other))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    vendor.target_root(str(fresh), False)
    row = vendor.Row("s", "data/_stage/q.bin", "q.bin", "0" * 64)
    assert vendor.dest_path(row).is_relative_to(fresh.resolve())


def test_missing_in_checkout_names_unbuilt_outputs(tmp_path, monkeypatch):
    """Negative: a build output that is not on disk is reported (what the test session's start check stops on)."""
    monkeypatch.setattr(vendor, "check", lambda *a, **k: [])
    monkeypatch.setattr(vendor, "built_outputs", lambda hands=None: [paths.REPO_ROOT / "assets" / "never_built.urdf"])
    assert vendor.missing_in_checkout() == ["not built assets/never_built.urdf"]
    monkeypatch.setattr(vendor, "built_outputs", lambda hands=None: [])  # noqa: ARG005
    assert vendor.missing_in_checkout() == []


def _unset_manifest(tmp_path, sha: str = ""):
    vd = tmp_path / "vendor"
    vd.mkdir()
    (vd / "sources.yaml").write_text("blank:\n  kind: github\n  repo: ''\n  commit: ''\n  licence: ''\n")
    (vd / "files.tsv").write_text(f"blank\t{tmp_path.name}_out/a.bin\ta.bin\t{sha}\n")
    return vd


def test_unset_source_dies_on_fetch_and_is_named_by_check(tmp_path, monkeypatch):
    """Negative: a source left empty (no repo and commit) stops fetch loudly — nothing is downloaded
    and there is no fallback — and check names it."""
    vd = _unset_manifest(tmp_path)
    monkeypatch.setattr(vendor, "_download", lambda url: pytest.fail(f"downloaded {url}"))
    with pytest.raises(SystemExit, match="source not set"):
        vendor.fetch(vendor_dir=vd)
    assert vendor.check(vendor_dir=vd)[0].startswith(vendor.UNSET)


def test_half_filled_source_dies(tmp_path):
    """Negative: a repo without a commit (or the reverse) is not 'unset' — it is an error."""
    vd = tmp_path / "vendor"
    vd.mkdir()
    (vd / "sources.yaml").write_text("half:\n  kind: github\n  repo: owner/name\n  commit: ''\n")
    with pytest.raises(SystemExit, match="missing"):
        vendor.sources(vd)


def test_a_filled_source_without_sha256_dies(tmp_path, monkeypatch):
    vd = _unset_manifest(tmp_path)
    (vd / "sources.yaml").write_text("blank:\n  kind: github\n  repo: owner/name\n  commit: abc\n")
    monkeypatch.setattr(vendor, "_download", lambda url: pytest.fail(f"downloaded {url}"))
    with pytest.raises(SystemExit, match="without a sha256"):
        vendor.fetch(vendor_dir=vd)


def test_every_source_is_set():
    """Every github source names its repository and commit and every row its sha256: no hand is left out by an
    unset source (XHand Pro comes from a third-party repository; docs/installation.md, 'XHand Pro')."""
    assert vendor.unset_sources(vendor.sources()) == set() and vendor.unavailable_hands() == {}
    assert all(r.sha256 for r in vendor.rows())


def _github_manifest(tmp_path, content: bytes):
    vd = tmp_path / "vendor"
    vd.mkdir()
    (vd / "sources.yaml").write_text("lfs:\n  kind: github\n  repo: owner/name\n  commit: abc\n")
    dest = f"{tmp_path.name}_out/a.bin"
    (vd / "files.tsv").write_text(f"lfs\t{dest}\ta.bin\t{hashlib.sha256(content).hexdigest()}\n")
    return vd, dest


def _pointer(content: bytes) -> bytes:
    oid = hashlib.sha256(content).hexdigest()
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(content)}\n".encode()


def test_fetch_follows_an_lfs_pointer_to_the_listed_bytes(tmp_path, monkeypatch):
    """A repository that keeps a file in Git LFS serves a pointer at the raw URL: a pointer whose oid is the listed
    sha256 is followed to the LFS object at the same commit, and those bytes are checked and placed."""
    content = b"mesh bytes"
    vd, dest = _github_manifest(tmp_path, content)
    served = {
        "https://raw.githubusercontent.com/owner/name/abc/a.bin": _pointer(content),
        "https://media.githubusercontent.com/media/owner/name/abc/a.bin": content,
    }
    monkeypatch.setattr(vendor, "_download", served.__getitem__)
    monkeypatch.setattr(vendor, "dest_path", lambda r: tmp_path / r.dest)
    assert vendor.fetch(vendor_dir=vd) == {"placed": 1, "present": 0}
    assert (tmp_path / dest).read_bytes() == content


def test_fetch_refuses_an_lfs_pointer_of_other_bytes(tmp_path, monkeypatch):
    """Negative: a pointer whose oid is not the listed sha256 names other bytes — it is not followed, and the pointer
    itself fails the sha256 check (nothing written)."""
    vd, dest = _github_manifest(tmp_path, b"mesh bytes")
    seen = []
    monkeypatch.setattr(vendor, "_download", lambda url: seen.append(url) or _pointer(b"other bytes"))
    monkeypatch.setattr(vendor, "dest_path", lambda r: tmp_path / r.dest)
    with pytest.raises(SystemExit, match="not written"):
        vendor.fetch(vendor_dir=vd)
    assert seen == ["https://raw.githubusercontent.com/owner/name/abc/a.bin"] and not (tmp_path / dest).exists()
    assert vendor.lfs_oid(b"plain bytes") is None
