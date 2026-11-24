#!/usr/bin/env python3
"""H2O raw unpack: the folders that the file-reading H2O tools need, written from the official downloads.

The MANO reader (readers/h2o) reads the pose archives in place. Three other readers work on files:
  poses   <raw>/dl/subject<k>_pose_v1_1.tar.gz → <raw>/dl/pose<k>/subject<k>/<scene>/<take>/cam4/<kind>/NNNNNN.txt for
          POSE_KINDS (the frame fit readers/h2o_frame --fit · the scene-object oracle readers/raw/h2o), and
          <raw>/dl/object.zip → <raw>/dl/objraw/object/<name>/… (the oracle's meshes).
  depth   subject<k>_v1_1.tar.gz (the RGB-D release: one gzip stream of ~190 GB per subject, so it is read once, in
          order, from a file or from stdin) → <raw>/sample/subject<k>/<scene>/<take>/cam4/{depth/<f>.png,
          cam_pose/<f>.txt, cam_intrinsics.txt} for f in DEPTH_FRAMES (the desk plane, data/support/h2o.py, fits
          frames 0 and 100). Download it with the official account and stream it in, e.g.
          curl --netrc-file <netrc> https://h2odataset.ethz.ch/data/dataset/subject1_v1_1.tar.gz | \\
            python -m dexengine.data.readers.h2o_unpack depth --archive -
Members are written byte for byte. A file already there with the same bytes is kept; one with other bytes dies (no
silent mix of two downloads). End marker `[H2O_UNPACK] <step> … files=<n>`; a step that writes nothing dies.

Usage: python -m dexengine.data.readers.h2o_unpack (poses | depth --archive <file|->) [--raw <raw layer>]
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

from dexengine.core.errors import InputMissingError

TAG = "[H2O_UNPACK]"
POSE_ARCHIVE = re.compile(r"subject(\d+)_pose_v1_1\.tar\.gz")
#: what the file readers open per take (h2o_frame: cam_pose · hand_pose · obj_pose_rt; readers/raw/h2o: hand_pose ·
#: obj_pose · obj_pose_rt)
POSE_KINDS = ("cam_pose", "hand_pose", "obj_pose", "obj_pose_rt")
POSE_MEMBER = re.compile(r"subject\d+/[^/]+/\d+/cam4/(" + "|".join(POSE_KINDS) + r")/\d+\.txt")
OBJECT_ZIP = "object.zip"
#: the folder of the raw layer the depth frames are unpacked into (read by data/support/h2o.py)
DEPTH_DIR = "sample"
#: the frames the desk plane fits (data/support/h2o.py: frames 0 and 100 of cam4)
DEPTH_FRAMES = ("000000", "000100")
_F = "|".join(DEPTH_FRAMES)
DEPTH_MEMBER = re.compile(
    rf"(?:.*/)?(subject\d+/[^/]+/\d+/cam4/(?:depth/(?:{_F})\.png|cam_pose/(?:{_F})\.txt|cam_intrinsics\.txt))"
)


def default_raw() -> Path:
    from dexengine.core import paths as P  # noqa: PLC0415 — needs the data root

    return P.layer_path("raw", "h2o")


def _put(dst: Path, data: bytes) -> bool:
    """Write `data` at `dst` (True) or keep an equal file (False); other bytes there die."""
    if dst.is_file():
        if dst.read_bytes() != data:
            raise SystemExit(f"{TAG} ✗ {dst} exists with other bytes — remove it or unpack into a fresh raw layer")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return True


def poses(raw: Path) -> int:
    """The pose folders and the object meshes; returns the number of files present for them."""
    dl = raw / "dl"
    archives = sorted(p for p in dl.glob("subject*_pose_v1_1.tar.gz") if POSE_ARCHIVE.fullmatch(p.name))
    if not archives:
        raise InputMissingError(f"{TAG} no subject<k>_pose_v1_1.tar.gz under {dl} (the official pose release)")
    n = 0
    for arc in archives:
        k = POSE_ARCHIVE.fullmatch(arc.name).group(1)
        with tarfile.open(arc, "r:gz") as tf:
            for m in tf:
                if m.isfile() and POSE_MEMBER.fullmatch(m.name):
                    _put(dl / f"pose{k}" / m.name, tf.extractfile(m).read())
                    n += 1
        print(f"{TAG} poses {arc.name} → {dl / f'pose{k}'}", flush=True)
    zp = dl / OBJECT_ZIP
    if not zp.is_file():
        raise InputMissingError(f"{TAG} {zp} missing (the official object meshes)")
    with zipfile.ZipFile(zp) as z:
        for name in z.namelist():
            if name.startswith("object/") and not name.endswith("/"):
                _put(dl / "objraw" / name, z.read(name))
                n += 1
    if n == 0:
        raise SystemExit(f"{TAG} ✗ poses: nothing unpacked from {len(archives)} archives")
    print(f"{TAG} poses files={n}", flush=True)
    return n


def depth(raw: Path, archive: str) -> int:
    """The desk-plane depth frames of one RGB-D subject archive (path or '-' = stdin); returns the files present."""
    out = raw / DEPTH_DIR
    n = 0
    src = sys.stdin.buffer if archive == "-" else open(archive, "rb")  # noqa: SIM115 — closed below
    try:
        with tarfile.open(fileobj=src, mode="r|gz") as tf:
            for m in tf:
                hit = DEPTH_MEMBER.fullmatch(m.name)
                if m.isfile() and hit:
                    _put(out / hit.group(1), tf.extractfile(m).read())
                    n += 1
    finally:
        if src is not sys.stdin.buffer:
            src.close()
    if n == 0:
        raise SystemExit(f"{TAG} ✗ depth: no cam4 depth/cam_pose/intrinsics member in {archive}")
    print(f"{TAG} depth {archive} → {out} files={n}", flush=True)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dexengine.data.readers.h2o_unpack", description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("poses", "depth"))
    ap.add_argument("--archive", default="", help="depth: subject<k>_v1_1.tar.gz, or - for stdin")
    ap.add_argument("--raw", type=Path, default=None, help="the h2o raw layer (default: the data root's)")
    a = ap.parse_args(argv)
    raw = a.raw or default_raw()
    if a.step == "poses":
        poses(raw)
        return 0
    if not a.archive:
        ap.error("depth needs --archive <subject<k>_v1_1.tar.gz | ->")
    depth(raw, a.archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
