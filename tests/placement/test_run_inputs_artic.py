"""The run-inputs statement (CD07) names the link model the float-settle statement opens for an articulated main
object at scene build — the tree's scene.xml and every link mesh, welded bodies included — through its one owner,
float_settle.inputs (reader kit:float-settle). The oracle is a measurement, not a second list: the files the
articulated reader itself opens (repair_obj._artic_links, as touch_gate.body_of calls it — scene.xml parse · mesh
checks) must be a subset of what run_inputs.scene_inputs states. Negatives: a link dropped from float_settle.inputs
fails the oracle (restored, it passes); a mesh the reader opens but the statement does not know fails it. A canonical
tree with an articulated main object is measured the same way by the CPU asset path gate
(tests/robots/test_stage_asset_paths.py)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

import pytest

from dexengine.data.objects import float_settle as FS
from dexengine.data.objects import touch_gate as TG
from dexengine.placement.bundle import run_inputs as RI
from dexengine.placement.coords import contacts as CT
from dexengine.retarget.f0 import repair_obj as RO

OBJ = "data/oakink2/assets/objects"


@contextmanager
def _recorded():
    """The files read while the block runs (ElementTree parses · Path.is_file checks), resolved."""
    seen: set[Path] = set()
    parse, is_file = ET.parse, Path.is_file

    def rec_parse(src, *a, **k):
        seen.add(Path(src).resolve())
        return parse(src, *a, **k)

    def rec_is_file(self):
        seen.add(self.resolve())
        return is_file(self)

    ET.parse, Path.is_file = rec_parse, rec_is_file
    try:
        yield seen
    finally:
        ET.parse, Path.is_file = parse, is_file


def artic_reads(tree_dir, ti: dict) -> set[Path]:
    """Measurement: the files the articulated reader opens for every articulated main object the float-settle
    statement measures (TG.main_objects — every side)."""
    out: set[Path] = set()
    for o in TG.main_objects(ti):
        if o.get("articulation"):
            with _recorded() as seen:
                RO._artic_links(o, str(ti["dataset_name"]), Path(tree_dir))
            out |= seen
    return out


def unstated(measured: set[Path], items: list) -> set[Path]:
    """Oracle: measured files the statement's float-settle rows (reader kit:float-settle) do not name."""
    return measured - {Path(it.path).resolve() for it in items if it.reader == "kit:float-settle"}


class _Tpd:
    """The tree-paths stand-in scene_inputs takes: object USD paths (none are asked for without extra objects)."""

    def __init__(self, root: Path):
        self.root = root

    def usd_file(self, oid: str, articulated: bool = False) -> Path:
        return self.root / "usd" / oid / ("object_artic.usd" if articulated else "object.usd")


def _tree(root: Path, idle_side: bool = False, stray: str = "") -> tuple[Path, dict]:
    """A tree with an articulated main object `supp` (base visual.obj · a welded body · a hinged part) and a rigid main
    object `other`, the meshes under the data root's oakink2 objects (absolute meshdir, as a canonical tree);
    `stray` = a mesh the hinged part's body also holds that task_info does not know."""
    assets = root / "data/oakink2/assets"
    for f in ("objects/a/visual.obj", "objects/a/weld.obj", "objects/a/top.obj", "objects/b/visual.obj", stray):
        if f:
            (assets / f).parent.mkdir(parents=True, exist_ok=True)
            (assets / f).write_text("v 0 0 0\n", encoding="utf-8")
    td = root / "tree"
    td.mkdir()
    extra = '<geom mesh="m_stray"/>' if stray else ""
    (td / "scene.xml").write_text(
        f"""<mujoco><compiler meshdir="{assets}/"/>
<asset><mesh name="m_root" file="objects/a/visual.obj"/><mesh name="m_weld" file="objects/a/weld.obj"/>
<mesh name="m_top" file="objects/a/top.obj"/><mesh name="m_other" file="objects/b/visual.obj"/>
{f'<mesh name="m_stray" file="{stray}"/>' if stray else ""}</asset>
<worldbody>
<body name="supp"><geom mesh="m_root"/>
  <body name="supp_weld" pos="0 0 0.1"><geom mesh="m_weld"/></body>
  <body name="supp_top"><joint name="supp_hinge" type="hinge" axis="0 0 1"/><geom mesh="m_top"/>{extra}</body>
</body>
<body name="other"><geom mesh="m_other"/></body>
</worldbody></mujoco>""",
        encoding="utf-8",
    )
    pose = {"pos": [0.0, 0.0, 0.1], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
    parts = {"base": "visual.obj", "moving": "top.obj", "fixed": [{"mesh": "weld.obj", "link": "base", "pose": pose}]}
    supp = {"name": "supp", "obj_id": "a", "role": "main", "side": "right", "mesh_dir": f"{OBJ}/a", "qpos_start": 0}
    supp["articulation"] = {"type": "hinge", "joint": "supp_hinge", "parts": parts}
    other = {"name": "other", "obj_id": "b", "role": "main", "side": "left", "mesh_dir": f"{OBJ}/b", "qpos_start": 8}
    ti = {"dataset_name": "oakink2", "objects": [supp, other], "idle_sides": ["right"] if idle_side else []}
    ti["provenance"] = {"mano_dir": "data/oakink2/mano/bimanual/x"}
    (td / "task_info.json").write_text(json.dumps(ti), encoding="utf-8")
    return td, ti


@pytest.fixture(autouse=True)
def _no_contact_readings(monkeypatch):
    """float_settle.inputs also states the contact readers' files (contacts.read_inputs), which need a base tree; this
    tree has none, and that half of the statement has its own tests."""
    monkeypatch.setattr(CT, "read_inputs", lambda tree_dir, ti: [])


def _items(td: Path, data_root: Path) -> list:
    return RI.scene_inputs(td, _Tpd(data_root))


@pytest.mark.parametrize("idle_side", [False, True])
def test_the_statement_holds_every_file_the_articulated_reader_opens(data_root, idle_side):
    td, ti = _tree(data_root, idle_side)
    measured = artic_reads(td, ti)
    assert sorted(p.relative_to(data_root).as_posix() for p in measured) == [
        f"{OBJ}/a/top.obj",
        f"{OBJ}/a/visual.obj",
        f"{OBJ}/a/weld.obj",
        "tree/scene.xml",
    ]
    assert not unstated(measured, _items(td, data_root))


@pytest.mark.parametrize("dropped", ["scene.xml", "visual.obj", "weld.obj", "top.obj"])
def test_a_link_dropped_from_the_owner_fails_the_oracle(data_root, monkeypatch, dropped):
    """Negative: float_settle.inputs without one of the files fails the oracle on exactly that file; restored, it
    passes."""
    td, ti = _tree(data_root)
    measured = artic_reads(td, ti)
    gone = next(p for p in measured if p.name == dropped and (p.parent.name in ("a", "tree")))
    owner = FS.inputs
    monkeypatch.setattr(FS, "inputs", lambda tree_dir, t: [f for f in owner(tree_dir, t) if Path(f).resolve() != gone])
    assert unstated(measured, _items(td, data_root)) == {gone}
    monkeypatch.setattr(FS, "inputs", owner)
    assert not unstated(measured, _items(td, data_root))


def test_the_oracle_measures_the_reader_not_the_statement(data_root):
    """Negative: a mesh the articulated reader opens that the statement's owner does not know (scene.xml holds it,
    task_info does not) fails the oracle — the measurement comes from the reader, not from the statement."""
    td, ti = _tree(data_root, stray="objects/a/stray.obj")
    measured = artic_reads(td, ti)
    assert unstated(measured, _items(td, data_root)) == {(data_root / OBJ / "a/stray.obj").resolve()}
