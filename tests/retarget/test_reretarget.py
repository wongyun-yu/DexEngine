"""Stage-1 reretarget: the count rule, the stage manifest and the stage-link removal guard (no solver, tmp folders)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexengine.retarget import reretarget as RR
from dexengine.retarget import reretarget_stage as RS

SPEC = "dg5f_s"


def _stage(tmp_path):
    src, stage = tmp_path / "data" / "hot3d", tmp_path / "stage"
    p = stage / "processed" / "hot3d"
    (p / "mano" / "right").mkdir(parents=True)
    (src / "assets" / "objects").mkdir(parents=True)
    for c in ("o__c1", "o__c2", "o__c3", "o__c4", "o__c5"):
        (src / "mano" / "right" / c).mkdir(parents=True)
        (src / "mano" / "right" / c / "k.npz").write_bytes(c.encode())
        os.symlink(f"{src}/mano/right/{c}", p / "mano" / "right" / c)
    os.symlink(f"{src}/assets/objects", p / "assets_objects")
    for c in ("o__c1", "o__c2", "o__c4", "o__c9"):
        d = p / SPEC / "right" / c / "0"
        d.mkdir(parents=True)
        (d / "trajectory_kinematic.npz").write_bytes(c.encode())
    log = tmp_path / "log"
    log.mkdir()
    return src, stage, p, SimpleNamespace(spec=SPEC, log=log)


def test_count_rule(tmp_path, monkeypatch, capsys):
    """Missing clips must all be explained; an exclusion of a clip with output does not lower the expectation."""
    _src, stage, p, ctx = _stage(tmp_path)
    monkeypatch.setattr(RR.R, "retarget_excluded", lambda ds, hand=None: {"o__c3": "bad", "o__c4": "stale"})
    assert not RR.count_check(p / "mano/right", stage, "hot3d", "right", "hot3d/right", ctx, None)  # o__c5
    out = capsys.readouterr()
    assert "[reretarget] hot3d/right: input 5 − explained 1 = expected 4 → output 4 OK" in out.out
    assert "    - o__c5" in out.err
    scope = tmp_path / "scope"
    scope.write_text("o__c1\no__c2\no__c3\n")
    assert RR.count_check(p / "mano/right", stage, "hot3d", "right", "hot3d/right", ctx, scope)
    assert "input 3 − explained 1 = expected 2 → output 2 OK" in capsys.readouterr().out


def test_manifest_and_unstage(tmp_path, monkeypatch):
    src, stage, p, _ctx = _stage(tmp_path)
    mf = stage / "processed" / "MANIFEST.sha256"
    mf.write_text("0" * 64 + "  other/f.txt\n" + "1" * 64 + "  hot3d/gone.txt\n")
    RS.write_manifest(stage, "./hot3d")
    lines = mf.read_text().splitlines()
    assert lines[0] == "0" * 64 + "  other/f.txt"  # other datasets' lines stay
    rels = [ln.split("  ", 1)[1] for ln in lines[1:]]
    assert rels == sorted(rels) and "hot3d/gone.txt" not in rels
    assert "hot3d/mano/right/o__c1/k.npz" in rels  # links are followed
    monkeypatch.setattr(RS, "ds_src", lambda ds, hand: src)  # unstage reads ds_src in its own module
    os.symlink("/elsewhere", p / "stray")
    with pytest.raises(SystemExit):
        RS.unstage(stage, "hot3d", SPEC)  # one link outside: none removed
    assert len(RS.stage_links(p)) == 7
    os.unlink(p / "stray")
    assert RS.unstage(stage, "hot3d", SPEC) == 0 and RS.stage_links(p) == []
    assert (p / SPEC / "right" / "o__c1" / "0" / "trajectory_kinematic.npz").is_file()  # outputs stay


def test_run_one_dataset(tmp_path, monkeypatch, capsys):
    """run() on one dataset with the solver stubbed: links, rig copy, manifest, the missing-marker failure, rc 4."""
    src, stage, log, rig = tmp_path / "data" / "hot3d", tmp_path / "stage", tmp_path / "log", tmp_path / "rig"
    for c in ("o__c1", "o__c2"):
        (src / "mano" / "right" / c).mkdir(parents=True)
    (src / "assets" / "objects").mkdir(parents=True)
    rig.mkdir()
    (rig / "right.xml").write_text("<mujoco/>")
    monkeypatch.setenv("DEXENGINE_HAND", "another")
    monkeypatch.setattr(RR.P, "retarget_stage_root", lambda hand: stage)
    monkeypatch.setattr(RR.P, "run_path", lambda layer, hand, **kw: log)
    monkeypatch.setattr(RR.P, "layer_path", lambda *a, **kw: tmp_path / "a" / "b" / "c" / "d")
    monkeypatch.setattr(RR.P, "hand_rig_dir", lambda spec: rig)
    monkeypatch.setattr(RR.P, "retarget_py", lambda: "python")
    monkeypatch.setattr(RR, "ds_src", lambda ds, hand: src)
    monkeypatch.setattr(RR, "stamp_inval", lambda ds: None)
    monkeypatch.setattr(RR.R, "retarget_excluded", lambda ds, hand=None: {})

    def fake_run(cmd, stdout=None, **kw):
        ph = cmd[cmd.index("--phase") + 1]
        d = Path(cmd[cmd.index("--dataset-dir") + 1]) / "processed/hot3d" / SPEC / "right/o__c1/0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "trajectory_kinematic.npz").write_bytes(b"q")
        stdout.write(
            "Found 2 processed clips\n"
            + (f"[SKIP_EXISTING] phase={ph} reuse=0 stale=0 redo=2\n" if ph == "generate-xml" else "")
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(RR.subprocess, "run", fake_run)
    assert RR.main(["hot3d", "2", "--hand", SPEC]) == 4
    out = capsys.readouterr()
    assert os.environ["DEXENGINE_HAND"] == SPEC
    assert f"{RS.SCOPE}hot3d " in out.out
    assert f"[reretarget] hot3d/right: input 2 − explained 0 = expected 2 → output 1 {RS.MISMATCH}" in out.out
    assert "ik: no [SKIP_EXISTING] line" in out.err
    p = stage / "processed" / "hot3d"
    assert sorted(os.readlink(p / "mano/right" / c) for c in ("o__c1", "o__c2")) == [
        f"{src}/mano/right/o__c1",
        f"{src}/mano/right/o__c2",
    ]
    assert (p / "assets/robots" / SPEC / "right.xml").is_file() and (p / ".stage_links").is_file()
    mf = (stage / "processed" / "MANIFEST.sha256").read_text()
    assert "  hot3d/assets/robots/dg5f_s/right.xml\n" in mf
