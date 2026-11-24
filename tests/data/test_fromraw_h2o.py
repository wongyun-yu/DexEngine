"""H2O unpack (dexengine.data.readers.h2o_unpack): only the members the file readers use are written, byte for byte;
a second run keeps equal files; other bytes, a missing archive or an archive without such members die."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from dexengine.core.errors import InputMissingError
from dexengine.data.readers import h2o_unpack as U


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_depth_keeps_only_the_plane_frames(tmp_path):
    t = "subject1/h1/0/"
    want = {
        f"{t}cam4/depth/000000.png": b"d0",
        f"{t}cam4/depth/000100.png": b"d100",
        f"{t}cam4/cam_pose/000000.txt": b"p0",
        f"{t}cam4/cam_pose/000100.txt": b"p100",
        f"{t}cam4/cam_intrinsics.txt": b"k",
    }
    skip = {
        f"{t}cam4/depth/000050.png": b"x",
        f"{t}cam4/rgb/000000.png": b"x",
        f"{t}cam1/cam_intrinsics.txt": b"x",
        f"{t}cam4/hand_pose/000000.txt": b"x",
    }
    arc = tmp_path / "subject1_v1_1.tar.gz"
    _tar(arc, {**skip, **want})
    raw = tmp_path / "raw"
    assert U.depth(raw, str(arc)) == len(want)
    assert _files(raw / U.DEPTH_DIR) == want
    assert U.depth(raw, str(arc)) == len(want)  # equal files are kept
    (raw / U.DEPTH_DIR / f"{t}cam4/depth/000000.png").write_bytes(b"other download")
    with pytest.raises(SystemExit, match="other bytes"):
        U.depth(raw, str(arc))
    empty = tmp_path / "empty.tar.gz"
    _tar(empty, skip)
    with pytest.raises(SystemExit, match="no cam4"):
        U.depth(tmp_path / "raw2", str(empty))


def test_poses_unpack_the_file_readers_layout(tmp_path):
    dl = tmp_path / "raw" / "dl"
    dl.mkdir(parents=True)
    with pytest.raises(InputMissingError):
        U.poses(tmp_path / "raw")
    t = "subject2/k1/3/cam4/"
    members = {f"{t}{k}/000001.txt": k.encode() for k in (*U.POSE_KINDS, "hand_pose_mano", "verb_label")}
    _tar(dl / "subject2_pose_v1_1.tar.gz", members)
    with zipfile.ZipFile(dl / U.OBJECT_ZIP, "w") as z:
        z.writestr("object/book/book.obj", "v 0 0 0\n")
    U.poses(tmp_path / "raw")
    got = _files(dl / "pose2")
    assert got == {f"{t}{k}/000001.txt": k.encode() for k in U.POSE_KINDS}
    assert (dl / "objraw" / "object" / "book" / "book.obj").read_text() == "v 0 0 0\n"
