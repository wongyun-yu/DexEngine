"""From-raw TACO inputs: the shipped-input check (dexengine.core.shipped), the TACO source pack (taco_source), the
union exclusion ledger, the frozen convex parts as a shipped input and the converter's refusal of a hand-made raw
layout. Each rule has a negative test: a missing, changed or extra file dies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dexengine.core import shipped
from dexengine.core.errors import InputMissingError


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A registry with one input `demo` (glob */part/*.obj) whose digest the test fills in."""
    reg = tmp_path / "shipped_inputs.tsv"

    def _write(digest: str, name: str = "demo", glob: str = "*/part/*.obj") -> None:
        reg.write_text("\t".join(shipped.HEAD) + "\n" + f"{name}\tdata/demo\t{glob}\t{digest}\tconsumer\twhy\n")

    _write("pending")
    monkeypatch.setattr(shipped, "REGISTRY", reg)
    return _write


def _demo_root(root: Path) -> Path:
    for i in ("a", "b"):
        (root / i / "part").mkdir(parents=True)
        (root / i / "part" / "0.obj").write_text(f"v {i}\n")
        (root / i / "visual.obj").write_text("written by the pipeline\n")  # outside the glob
    return root


def test_shipped_verified_and_site(tmp_path, registry, capsys):
    root = _demo_root(tmp_path / "demo")
    assert shipped.state("demo", root) == shipped.SITE  # no checksum file = an unverified root
    registry(shipped.write_sums("demo", root))
    assert shipped.members("demo", root) == ["a/part/0.obj", "b/part/0.obj"]
    assert shipped.state("demo", root) == shipped.VERIFIED
    assert "[SHIPPED] demo state=verified" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["change", "remove", "extra", "digest"])
def test_shipped_damage_dies(tmp_path, registry, damage):
    root = _demo_root(tmp_path / "demo")
    registry(shipped.write_sums("demo", root))
    if damage == "change":
        (root / "a" / "part" / "0.obj").write_text("v changed\n")
    elif damage == "remove":
        (root / "b" / "part" / "0.obj").unlink()
    elif damage == "extra":
        (root / "b" / "part" / "1.obj").write_text("v extra\n")
    else:
        registry("0" * 64)
    with pytest.raises(SystemExit):
        shipped.state("demo", root)


def test_shipped_missing_root_dies(tmp_path, registry):
    with pytest.raises(InputMissingError):
        shipped.state("demo", tmp_path / "absent")


def _v1_tree(root: Path, clip: str, ids=("035", "024"), right="035", left="024") -> None:
    d = root / clip
    (d / "0").mkdir(parents=True)
    ti = {
        "source_ref_dt": 1.0 / 30.0,
        "right_object_mesh_dir": f"assets/objects/{right}",
        "left_object_mesh_dir": f"assets/objects/{left}",
    }
    (d / "task_info.json").write_text(json.dumps(ti))
    sides = ("right", "left")  # scene order = right_visual, left_visual (taco_source.read checks it)
    (d / "scene.xml").write_text(
        "".join(f'<mesh name="{s}_visual" file="objects/{i}/visual.obj"/>\n' for s, i in zip(sides, ids, strict=True))
    )
    rng = np.random.default_rng(0)
    np.savez(d / "0" / "trajectory_kinematic.npz", qpos=rng.normal(size=(12, 50)), frequency=np.float64(50.0))


def test_taco_source_pack_reads_like_the_v1_tree(tmp_path, monkeypatch):
    from dexengine.data.readers import taco_source as TS

    v1, pack = tmp_path / "v1", tmp_path / "pack"
    _v1_tree(v1, "brush_brush_bowl__20230919_036")
    _v1_tree(v1, "pour-in-some_bowl_kettle__20231019_013", ids=("199", "191"), right="191", left="199")
    reg = tmp_path / "reg.tsv"
    reg.write_text("\t".join(shipped.HEAD) + "\n" + f"{TS.NAME}\tx\t*/*\tpending\tc\tw\n")
    monkeypatch.setattr(shipped, "REGISTRY", reg)
    digest = TS.build(v1, pack)
    reg.write_text("\t".join(shipped.HEAD) + "\n" + f"{TS.NAME}\tx\t*/*\t{digest}\tc\tw\n")
    assert TS.kind(v1) == TS.V1_TREE and TS.kind(pack) == TS.PACK
    for c in TS.clips(v1):
        a, b = TS.read(v1, TS.V1_TREE, c), TS.read(pack, TS.PACK, c)
        assert a.info == b.info and a.scene_ids == b.scene_ids and a.frequency == b.frequency
        assert a.object_qpos.tobytes() == b.object_qpos.tobytes() and b.object_qpos.shape == (12, 14)
        assert (pack / c / TS.TASK_INFO).read_bytes() == (v1 / c / TS.TASK_INFO).read_bytes()
    # the pack is checked by its reader: a changed track dies
    np.save(pack / TS.clips(pack)[0] / TS.OBJECT_QPOS, np.zeros((12, 14)))
    with pytest.raises(SystemExit):
        TS.kind(pack)


def test_exclusion_ledger_repository_owns_human_lines(tmp_path, monkeypatch, data_root):
    from dexengine.core.rules import ledger as L

    repo = tmp_path / "configs"
    (repo / "datasets" / "taco").mkdir(parents=True)
    (repo / "datasets" / "taco" / "excluded_clips.txt").write_text("# human\nclip_a\trepo reason\nclip_b\trepo b\n")
    monkeypatch.setattr("dexengine.core.repo.CONFIGS_DIR", repo)
    assert L.retarget_excluded("taco") == {"clip_a": "repo reason", "clip_b": "repo b"}  # empty data root
    site = data_root / "precompute" / "taco" / "diag"
    site.mkdir(parents=True)
    (site / "excluded_clips.txt").write_text("clip_a\tsite copy\nclip_c\tD1g:x — generator\n")
    got = L.retarget_excluded("taco")
    assert got == {"clip_a": "repo reason", "clip_b": "repo b", "clip_c": "D1g:x — generator"}
    (site / "excluded_clips.txt").write_text("clip_z\ta human line the repository lacks\n")
    with pytest.raises(SystemExit, match="repository"):
        L.retarget_excluded("taco")


def test_frozen_convex_is_a_shipped_input(tmp_path, monkeypatch, capsys):
    from dexengine.retarget import decompose as D

    objects = tmp_path / "objects"
    (objects / "055" / "convex").mkdir(parents=True)
    (objects / "055" / "convex" / "0.obj").write_text("v 0 0 0\n")
    reg = tmp_path / "reg.tsv"
    reg.write_text("\t".join(shipped.HEAD) + "\n" + "taco_convex\tx\t*/convex/*.obj\tpending\tc\tw\n")
    monkeypatch.setattr(shipped, "REGISTRY", reg)
    monkeypatch.setattr(D, "_FROZEN_CHECKED", set())
    assert D.frozen_convex(str(objects / "055"), "taco") == 1  # an unverified root (no checksum file)
    assert "state=site" in capsys.readouterr().out
    digest = shipped.write_sums("taco_convex", objects)
    reg.write_text("\t".join(shipped.HEAD) + "\n" + f"taco_convex\tx\t*/convex/*.obj\t{digest}\tc\tw\n")
    monkeypatch.setattr(D, "_FROZEN_CHECKED", set())
    assert D.frozen_convex(str(objects / "055"), "taco") == 1
    assert "state=verified" in capsys.readouterr().out
    (objects / "056").mkdir()
    with pytest.raises(RuntimeError, match="shipped input taco_convex"):
        D.frozen_convex(str(objects / "056"), "taco")
    monkeypatch.setattr(D, "_FROZEN_CHECKED", set())
    (objects / "055" / "convex" / "0.obj").write_text("v 1 0 0\n")
    with pytest.raises(SystemExit):
        D.frozen_convex(str(objects / "055"), "taco")


def test_converter_refuses_a_layout_the_lift_did_not_write(tmp_path):
    pytest.importorskip("torch")
    from dexengine.data.readers.taco_convert import _load_grid50

    with pytest.raises(FileNotFoundError, match="lift layout"):
        _load_grid50(tmp_path, "tool", "035")
    np.save(tmp_path / "tool_035.grid50.npy", np.zeros((5, 7)))
    assert tuple(_load_grid50(tmp_path, "tool", "035").shape) == (5, 7)


def test_stage1_expects_the_listed_clips_minus_the_ledger(monkeypatch):
    from dexengine.data import rules as RR
    from dexengine.retarget import run as RUN

    seen = {}

    def fake(ds, hand=None):
        seen["args"] = (ds, hand)
        return {"b__2": "no_contact — test"}

    monkeypatch.setattr(RR, "retarget_excluded", fake)
    robot = RUN.P.hand_spec_dir("wuji2")  # the stage-1 robot name is the hand's spec dir (paths owns it)
    assert RUN.expected_clips(["a__1", "b__2", "c__3"], robot) == ["a__1", "c__3"]
    assert seen["args"] == ("taco", "wuji2")  # hand-scoped lines are read for this hand's id
