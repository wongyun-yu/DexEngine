"""An articulated asset's surface points and keypoints live beside its USD in the usd_artic generation folder
(artifact_paths.surface_dir), never in the meshes layer an older code tree reads: the path owner, the writer's whole
write set, and every reader (RL preflight and mesh folders, the bundle statement, train-ready, the spec stage)."""

import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from dexengine.core import artifact_paths as AP
from dexengine.core import layout as L
from tests.core.references_paths import SITE

REPO = Path(__file__).resolve().parents[2]
DS = "oakink2"
ASSET = "a_o02_0001_00001__b_o02_0001_00002__hinge"
#: calls that name a surface file or folder — each must say whether the asset is articulated
SURFACE_CALLS = ("surface_dir", "surface_points_file", "keypoints_file")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(L.DATA_ROOT_ENV, str(tmp_path))
    AP.get.cache_clear()
    yield tmp_path
    AP.get.cache_clear()


def _gen(root: Path) -> Path:
    return root / f"data/{DS}/usd_artic/{L.ARTIC_BAKE_GEN}" / ASSET


def _meshes(root: Path) -> Path:
    return root / f"data/{DS}/meshes" / ASSET


def _plant_old(root: Path) -> Path:
    """The older tree's files in the meshes layer (the place v0.1 must never read or write for an articulated asset)."""
    d = _meshes(root)
    d.mkdir(parents=True)
    np.savez(d / "surface_points.npz", points=np.zeros((4, 3)))
    np.save(d / "keypoints_fps2.npy", np.zeros((2, 3)))
    return d


def test_surface_dir_articulated_is_the_generation_folder(root):
    tp = AP.TacoPaths(DS)
    assert Path(tp.surface_dir(ASSET, True)) == _gen(root)
    assert Path(tp.surface_points_file(ASSET, articulated=True)) == _gen(root) / "surface_points.npz"
    assert Path(tp.keypoints_file(ASSET, articulated=True)) == _gen(root) / "keypoints_fps2.npy"
    assert Path(tp.surface_dir(ASSET, True)) == Path(tp.usd_file(ASSET, articulated=True)).parent  # beside its USD
    assert Path(tp.surface_points_file("box")) == root / f"data/{DS}/meshes/box/surface_points.npz"  # rigid: as before
    assert Path(tp.surface_dir(ASSET, True)) != _meshes(root)  # negative: never the older tree's place


def _calls_without_flag(src: str) -> list[str]:
    """Calls of SURFACE_CALLS that do not state `articulated` (keyword, or the positional slot after the id)."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SURFACE_CALLS:
            slot = 3 if n.func.attr == "keypoints_file" else 2
            if not any(k.arg == "articulated" for k in n.keywords) and len(n.args) < slot:
                out.append(f"{n.func.attr} line {n.lineno}")
    return out


def test_every_surface_call_states_articulated():
    bad = []
    for f in sorted((REPO / "dexengine").rglob("*.py")) + sorted((REPO / SITE).rglob("*.py")):
        if f.name != "artifact_paths.py":
            bad += [f"{f.relative_to(REPO)}: {c}" for c in _calls_without_flag(f.read_text(encoding="utf-8"))]
    assert not bad, f"surface calls that do not say whether the asset is articulated (would read meshes): {bad}"


def test_the_flag_rule_catches_a_bare_call():
    assert _calls_without_flag("t.surface_points_file(oid)\nt.keypoints_file(oid, 2)\n") == [
        "surface_points_file line 1",
        "keypoints_file line 2",
    ]
    assert not _calls_without_flag("t.surface_points_file(oid, art)\nt.keypoints_file(oid, articulated=art)\n")


def test_writer_writes_only_the_generation_folder(root, monkeypatch):
    from dexengine.data.artic import surface as AS

    old = _plant_old(root)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir()}
    spec = {"obj_id": "a_o02_0001_00001", "dataset": DS, "type": "hinge", "parts": {}}
    fake = {
        "points": np.zeros((5, 3), np.float32),
        "normals": np.zeros((5, 3), np.float32),
        "part_id": np.zeros(5, np.int8),
        "joint_type": "hinge",
        "joint_axis": np.array([1.0, 0, 0], np.float32),
        "joint_anchor": np.zeros(3, np.float32),
        "merge_theta": 0.0,
        "keypoints": np.zeros((2, 3), np.float32),
        "n_base": 3,
        "n_moving": 2,
        "area_base": 0.0,
        "area_moving": 0.0,
        "kp_part": "base",
    }
    monkeypatch.setattr(AS, "ROOT", root)
    monkeypatch.setattr(AS, "stated", str)  # records name run-root paths; the tmp root is not one
    monkeypatch.setattr(AS, "LEDGER_FILE", root / "artic_surface_ledger.json")
    monkeypatch.setattr(AS, "_load_specs", lambda ds: {ASSET: spec})
    monkeypatch.setattr(AS, "build_one", lambda *a, **k: fake)
    monkeypatch.setattr(AS, "_ledger_entry", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["surface.py", "--dataset", DS, "--ids", ASSET, "--force"])
    assert AS.main() == 0
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir()} == before  # untouched
    written = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not p.is_relative_to(old))
    gen = str(_gen(root).relative_to(root))
    assert written == ["artic_surface_ledger.clips.json", f"{gen}/keypoints_fps2.npy", f"{gen}/surface_points.npz"]


def test_readers_miss_a_file_that_is_only_in_the_meshes_layer(root, monkeypatch):
    """Half-moved state: the older files exist, the generation folder holds only the USD — every reader says missing."""
    from dexengine.data.artic import spec_stage as SS
    from dexengine.data.objects import hull as OH
    from dexengine.placement.train_ready import assets as A
    from dexengine.rl.cli_entry import seqdir_clips as SD

    _plant_old(root)
    usd = Path(AP.TacoPaths(DS).usd_file(ASSET, articulated=True))
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0\n")
    # train-ready
    miss = A.missing_assets(DS, ASSET, articulated=True)
    assert "surface" in miss and "keypoints" in miss and "usd_artic" not in miss
    # RL preflight
    missing: list[str] = []
    SD._note_missing_clip_inputs(
        "", missing, False, AP.TacoPaths(DS), "right", str(root), {"type": "hinge"}, "", ASSET, "c"
    )
    assert f"c: {_gen(root) / 'surface_points.npz'}" in missing
    assert not [m for m in missing if str(_meshes(root)) in m]
    # spec stage: the surface check is handed the generation folder's file
    seen = []
    monkeypatch.setattr(OH, "cooked_problem", lambda f: "")
    monkeypatch.setattr("dexengine.data.artic.surface.surface_problem", lambda v, p: seen.append(Path(p)) or "x")
    assert SS._unfinished(DS, {f"{DS}/{ASSET}": {"dataset": DS}}) == [ASSET]
    assert seen == [_gen(root) / "surface_points.npz"]
