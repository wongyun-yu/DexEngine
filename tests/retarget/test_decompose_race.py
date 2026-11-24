"""Part folders written by clips that share an object (dexengine.retarget.decompose.ensure_convex): two writers at
once both finish with one set of parts, and a writer killed mid-write leaves nothing the next run trips on. CoACD is
replaced by two fixed tetrahedra; the folder handling is the real one."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from dexengine.retarget import decompose as D

REPO = Path(__file__).resolve().parents[2]
V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
FAKE = "D.coacd_parts = lambda v, f: [(V, F), (V + 2.0, F)]\nD.installed_coacd_version = lambda: 'test'\n"


@pytest.fixture
def mesh_dir(tmp_path, monkeypatch) -> str:
    import trimesh

    d = tmp_path / "obj"
    d.mkdir()
    trimesh.Trimesh(V, F).export(d / "visual.obj")
    monkeypatch.setattr(D, "coacd_parts", lambda v, f: [(V, F), (V + 2.0, F)])
    monkeypatch.setattr(D, "installed_coacd_version", lambda: "test")
    return str(d)


def _current(mesh_dir: str) -> bool:
    return D.stale_reason(os.path.join(mesh_dir, "visual.obj"), D.rigid_convex_dir(mesh_dir)) is None


def test_two_clips_sharing_an_object_both_finish(mesh_dir, monkeypatch):
    parts_written, second_done = threading.Event(), threading.Event()
    record = D._write_record

    def slow_record(output_dir, input_file, n):
        parts_written.set()
        second_done.wait(2.0)  # the second clip looks at the folder while the first one's record is not written yet
        record(output_dir, input_file, n)

    monkeypatch.setattr(D, "_write_record", slow_record)
    got = {}

    def first():
        got["first"] = D.decompose_object_dir(mesh_dir)

    def second():
        parts_written.wait(5.0)
        try:
            got["second"] = D.decompose_object_dir(mesh_dir)
        except RuntimeError as e:
            got["second"] = e
        finally:
            second_done.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15.0)
    assert got["first"][:2] == (2, "decomposed")
    assert got["second"] == (2, "reused", "fingerprint match"), got["second"]
    assert _current(mesh_dir)


def test_a_writer_killed_mid_write_leaves_nothing_in_the_way(mesh_dir, tmp_path):
    child = tmp_path / "killed_writer.py"
    child.write_text(
        "import sys, time\nimport numpy as np\nfrom dexengine.retarget import decompose as D\n"
        f"V = np.array({V.tolist()})\nF = np.array({F.tolist()})\n{FAKE}"
        "def stuck(*a):\n    print('PARTS', flush=True)\n    time.sleep(60)\n"
        "D._write_record = stuck\nD.decompose_object_dir(sys.argv[1])\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    with subprocess.Popen([sys.executable, str(child), mesh_dir], stdout=subprocess.PIPE, text=True, env=env) as p:
        assert p.stdout.readline().strip() == "PARTS"  # parts written, record not yet
        p.kill()
        p.wait()
    assert D.decompose_object_dir(mesh_dir)[:2] == (2, "decomposed")
    assert _current(mesh_dir)
    assert sorted(os.listdir(mesh_dir)) == ["convex", "visual.obj"]  # the killed writer's hidden folder is gone


def test_a_folder_without_a_record_is_still_refused(mesh_dir):
    cdir = D.rigid_convex_dir(mesh_dir)
    os.makedirs(cdir)
    Path(cdir, "0.obj").write_text("")  # parts of unknown origin (frozen parts carry no record)
    with pytest.raises(RuntimeError, match="holds stale parts"):
        D.decompose_object_dir(mesh_dir)
