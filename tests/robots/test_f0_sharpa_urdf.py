"""The SharpA f0 URDF (repair_cfg.F0_SHARPA_URDF): the bytes the original f0 read outside the head camera link, mesh
paths re-pointed only; the head camera link is the training robot's (the ZED Mini of zed_camera_link.xml).

Pins (measured against the original generation's file, sha256 c64d4710fb4298d3…, with its camera link
removed — the same value on that file and on this one, so only the camera link differs):
- the text outside the camera link with every `filename="…"` emptied hashes to STRIPPED_SHA (so every joint · link ·
  inertial · limit byte of the hands, arms and torso is the original one — torso_3 pitch -0.305 included);
- the mesh files the text outside the camera link resolves to, in order, have the original mesh bytes
  (MESH_MD5_LIST_SHA over their md5 list);
- the camera link carries the training head camera link (assets/robots/head_cameras/zed_camera_link.xml: the same
  origins, mass and inertia) and its mesh is the fetched ZED Mini mesh (the sha256 the vendor list states);
- every mesh resolves to a file in this tree; the resolver is the one place that names the file and dies without it.
"""

import hashlib
import re
import xml.etree.ElementTree as ET

import pytest

from dexengine.core.paths import urdf_mesh
from dexengine.core.repo import REPO_ROOT
from tests.imports import own_module

STRIPPED_SHA = "7adf9cdb302c76f0aed3528285f31933b1de4f34262ba423987abab26cd6b371"
MESH_MD5_LIST_SHA = "97df792276e15ed0dc13d8c58f85f7a0395a7f8a2eb17757bd20e327f7437bdc"
N_MESH_REFS = 138  # outside the camera link (140 in the file: the camera link's visual and collision add two)
CAMERA_LINK = re.compile(r'\n[ \t]*<link name="camera">.*?</link>', re.S)
TRAINING_CAMERA_LINK = REPO_ROOT / "assets/robots/head_cameras/zed_camera_link.xml"
VENDOR_FILES = REPO_ROOT / "configs/robots/vendor/files.tsv"


def _outside_camera(text):
    blocks = CAMERA_LINK.findall(text)
    assert len(blocks) == 1
    return CAMERA_LINK.sub("", text), blocks[0]


def _shape(link_xml):
    """(tag, origin xyz, origin rpy, mass, inertia) of every inertial/visual/collision of one <link> — the mesh name
    is compared separately."""
    root = ET.fromstring(link_xml.strip())
    out = []
    for el in root:
        o = el.find("origin")
        m = el.find("mass")
        i = el.find("inertia")
        out.append(
            (
                el.tag,
                o.get("xyz") if o is not None else None,
                o.get("rpy") if o is not None else None,
                m.get("value") if m is not None else None,
                tuple(sorted(i.attrib.items())) if i is not None else None,
            )
        )
    return out


@pytest.fixture
def RC(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")  # repair_cfg resolves the active hand and arm at import
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair_cfg")


def test_non_mesh_bytes_are_the_original(RC):
    t, _ = _outside_camera(RC.f0_sharpa_urdf("rby1").read_text())
    assert hashlib.sha256(re.sub(r'filename="[^"]+"', 'filename=""', t).encode()).hexdigest() == STRIPPED_SHA


def test_meshes_exist_and_carry_the_original_bytes(RC):
    p = RC.f0_sharpa_urdf("rby1")
    t, _ = _outside_camera(p.read_text())
    refs = re.findall(r'filename="([^"]+)"', t)
    assert len(refs) == N_MESH_REFS
    files = [urdf_mesh(p, f) for f in refs]
    assert all(f.is_file() for f in files)
    md5s = "\n".join(hashlib.md5(f.read_bytes()).hexdigest() for f in files)
    assert hashlib.sha256(md5s.encode()).hexdigest() == MESH_MD5_LIST_SHA


def test_camera_link_is_the_training_head_camera(RC):
    p = RC.f0_sharpa_urdf("rby1")
    _, cam = _outside_camera(p.read_text())
    train = TRAINING_CAMERA_LINK.read_text()
    train = train[train.index("<link") : train.index("</link>") + len("</link>")]
    assert _shape(cam) == _shape(train)
    refs = re.findall(r'filename="([^"]+)"', cam)
    assert len(refs) == 2
    rows = [r.split("\t") for r in VENDOR_FILES.read_text().splitlines() if r.endswith(".stl") or ".stl\t" in r]
    for f in refs:
        q = urdf_mesh(p, f)
        assert q.is_file()
        want = [r[3] for r in rows if len(r) > 3 and r[1].endswith("/" + q.name)]
        assert want and hashlib.sha256(q.read_bytes()).hexdigest() == want[0]


def test_torso_3_is_the_original_pose(RC):
    t = RC.f0_sharpa_urdf("rby1").read_text()
    m = re.search(r'<joint name="torso_3" type="fixed">.*?<origin[^>]*rpy="([^"]+)"', t, re.S)
    assert m and m.group(1).split()[1] == "-0.305000000"


def test_handcfg_reads_it_and_missing_dies(RC, monkeypatch, tmp_path):
    monkeypatch.delenv("DEXENGINE_URDF_OVERRIDE", raising=False)
    hc = RC.HandCfg.__new__(RC.HandCfg)
    hc.name, hc.arm = "sharpa_wave", "rby1"
    assert hc.urdf("right") == RC.f0_sharpa_urdf("rby1") == hc.urdf("left")
    monkeypatch.setattr(RC, "ROBOTS", tmp_path)
    with pytest.raises(SystemExit, match="SharpA f0 URDF missing"):
        RC.f0_sharpa_urdf("rby1")


def test_a_missing_mesh_dies(RC, monkeypatch, tmp_path):
    d = tmp_path / "f0"
    d.mkdir()
    (d / "rby1_sharpa_wave_f0.urdf").write_text(
        '<robot><link name="x"><collision><geometry><mesh filename="nope.stl"/></geometry></collision></link></robot>'
    )
    monkeypatch.setattr(RC, "ROBOTS", tmp_path)
    with pytest.raises(SystemExit, match="mesh files missing"):
        RC.f0_sharpa_urdf("rby1")
