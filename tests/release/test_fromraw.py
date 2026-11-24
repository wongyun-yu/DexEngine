"""The from-raw gate root (dexengine.release.fromraw prepare): only the stated downloads are linked, shipped inputs
are copied and verified, and a missing or altered input dies (negative tests)."""

from __future__ import annotations

import json

import pytest

from dexengine.core import shipped
from dexengine.core.errors import InputMissingError
from dexengine.release import fromraw as F


@pytest.fixture
def world(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    (dl / "dl").mkdir(parents=True)
    (dl / "dl" / "subject1_pose_v1_1.tar.gz").write_bytes(b"pose")
    (dl / "dl" / "object.zip").write_bytes(b"zip")
    (dl / "dl" / "pose1").mkdir()
    (dl / "dl" / "pose1" / "derived.txt").write_bytes(b"not a download")  # must not be linked
    mano = tmp_path / "mano"
    mano.mkdir()
    for f in F.MANO_FILES:
        (mano / f).write_bytes(b"m")
    ship = tmp_path / "ship" / "generation"
    (ship / "h2o").mkdir(parents=True)
    (ship / "h2o" / "GENERATION").write_text("generation=1\n")
    reg = tmp_path / "shipped_inputs.tsv"
    reg.write_text(
        "\t".join(shipped.HEAD) + "\n" + "\t".join(["generation", "data", "*/GENERATION", "0" * 64, "x", "y"]) + "\n"
    )
    monkeypatch.setattr(shipped, "REGISTRY", reg)
    digest = shipped.write_sums("generation", ship)
    reg.write_text(
        "\t".join(shipped.HEAD) + "\n" + "\t".join(["generation", "data", "*/GENERATION", digest, "x", "y"]) + "\n"
    )
    return {"dl": dl, "mano": mano, "ship": tmp_path / "ship", "root": tmp_path / "gate"}


def _prep(w, **kw):
    return F.prepare("h2o", w["root"], w["dl"], w["mano"], w["ship"], kw.get("standin"))


def test_root_holds_only_the_stated_downloads(world, capsys):
    rec = _prep(world)
    root = world["root"]
    assert (root / "data/h2o/raw/dl/subject1_pose_v1_1.tar.gz").is_symlink()
    assert not (root / "data/h2o/raw/dl/pose1").exists()
    assert (root / "data/h2o/GENERATION").read_text() == "generation=1\n"
    assert json.loads((root / F.RECORD).read_text())["standin"] is False and rec["copies"] == {"generation": 1}
    assert "[FROMRAW_ROOT] ds=h2o links=4 copies=1" in capsys.readouterr().out


def test_existing_root_dies(world):
    world["root"].mkdir()
    with pytest.raises(SystemExit, match="exists"):
        _prep(world)


def test_missing_download_dies(world):
    (world["dl"] / "dl" / "object.zip").unlink()
    with pytest.raises(InputMissingError, match="object.zip"):
        _prep(world)
    assert not world["root"].exists()  # no partial root


def test_altered_shipped_input_dies(world):
    (world["ship"] / "generation" / "h2o" / "GENERATION").write_text("generation=2\n")
    with pytest.raises(SystemExit):
        _prep(world)


def test_missing_mano_dies(world):
    (world["mano"] / F.MANO_FILES[0]).unlink()
    with pytest.raises(InputMissingError, match="MANO"):
        _prep(world)
    assert not world["root"].exists()  # no partial root


def test_a_dataset_without_stated_downloads_dies(world):
    with pytest.raises(InputMissingError, match="downloads"):
        F.prepare("grab", world["root"], world["dl"], world["mano"], world["ship"], None)


def test_shipped_rows_follow_the_path_rule():
    assert set(F.shipped_rows("taco")) >= {"taco_source", "taco_convex", "taco_meshes", "generation"}
    assert F.shipped_rows("h2o") == ["generation"]


def test_run_dies_for_a_driver_the_gate_does_not_run_yet(tmp_path):
    (tmp_path / F.RECORD).write_text(json.dumps({"dataset": "hot3d", "standin": False}))
    with pytest.raises(SystemExit, match="not yet run by the gate"):
        F.run("hot3d", tmp_path, 1, None)


def test_taco_subset_outside_the_source_dies(tmp_path, monkeypatch):
    from dexengine.core.layout import taco_source_root

    src = tmp_path / taco_source_root()
    (src / "a__1").mkdir(parents=True)
    (tmp_path / F.RECORD).write_text(json.dumps({"dataset": "taco", "standin": False}))
    lst = tmp_path / "list.txt"
    lst.write_text("b__2\n")
    with pytest.raises(SystemExit, match="not in the TACO source"):
        F.run("taco", tmp_path, 1, None, arm="rby1", hand="wuji2", subset=lst)


def test_outputs_lists_made_files_and_copies_not_links(tmp_path, capsys):
    (tmp_path / "data" / "ds" / "made").mkdir(parents=True)
    (tmp_path / "data" / "ds" / "made" / "a.npz").write_bytes(b"x")
    (tmp_path / "data" / "ds" / "copy.json").write_text("{}")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "raw.pkl").write_bytes(b"r")
    (tmp_path / "data" / "ds" / "raw.pkl").symlink_to(tmp_path / "elsewhere" / "raw.pkl")
    (tmp_path / "FROMRAW.json").write_text("{}")
    assert F.outputs(tmp_path) == ["data/ds/copy.json", "data/ds/made/a.npz"]
    assert F.main(["outputs", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.split() == ["data/ds/copy.json", "data/ds/made/a.npz"]
    with pytest.raises(SystemExit, match="no data folder"):
        F.outputs(tmp_path / "elsewhere")
