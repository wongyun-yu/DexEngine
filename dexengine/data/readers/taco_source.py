#!/usr/bin/env python3
"""TACO source: what the stage-1 lift (taco_lift) and the 4b grasp reference (filtering_refs) read per clip.

WHY
    The TACO object trajectories of the released data come from the frozen V1 generation's trees, whose generator
    no longer exists. They are not a resample of the public TACO `Object_Poses`: at the same timestamps (raw 30 Hz
    frame 3m · V1 50 Hz frame 5m), after the per-session hand shift, positions differ by p50 1.9 / p99 12.2 /
    max 27.9 mm and rotations by p50 0.19 / p99 5.6 / max 41.9 deg (194 tracks, 12,826 frames),
    nearly constant within a track. The kinematic phase passes the V1 object block through verbatim (taco_lift
    `*.grid50.npy`), so the released data is tied to these values: the release ships them (shipped input
    `taco_source`, configs/release/shipped_inputs.tsv).

ONE PLACE, TWO FORMS (location owner dexengine.core.layout.taco_source_root)
    v1_tree  the earlier full trees `<clip>/{task_info.json, scene.xml, 0/trajectory_kinematic.npz}`
    pack     the shipped subset — only what the readers use:
               <clip>/task_info.json    the V1 file, byte for byte (4b reads it whole)
               <clip>/source.json       {"frequency": V1 npz frequency, "scene_object_ids": scene.xml object order}
               <clip>/object_qpos.npy   the V1 qpos object columns (OBJECT_COLS: right object 7 · left object 7, pos +
                                        quat wxyz), float64
             plus its checksum file (dexengine.core.shipped); `kind()` verifies it.
    The form is decided by the checksum file (`pack` = the shipped input verified; `v1_tree` = shipped.state
    `site`). Every reader goes through `read()`, so both forms give the same values (proof: `build` then lift on
    both roots, byte-equal shim).

CLI (retarget venv or any python with numpy):
    python -m dexengine.data.readers.taco_source build --v1-root <V1 trees> --out <pack dir>
    python -m dexengine.data.readers.taco_source clips [--root DIR] --out <list>   # every clip of the source
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexengine.core import shipped
from dexengine.core.errors import InputMissingError
from dexengine.data import side_objects

NAME = "taco_source"
V1_TREE, PACK = "v1_tree", "pack"
#: V1 bimanual qpos object columns: right object = the first 7 (pos3 + quat wxyz), left object = the last 7.
OBJECT_COLS = slice(36, 50)
TASK_INFO, SOURCE_JSON, OBJECT_QPOS = "task_info.json", "source.json", "object_qpos.npy"
V1_NPZ = "0/trajectory_kinematic.npz"


@dataclass(frozen=True)
class Source:
    """One clip of the TACO source."""

    info: dict  # task_info.json
    scene_ids: list  # object mesh ids in scene order (= qpos object slots = scene.xml right_visual, left_visual)
    object_qpos: np.ndarray  # (T50, 14) float64
    frequency: float

    @property
    def slot_ids(self) -> tuple[str, str]:
        """(slot 0, slot 1) mesh ids: the tool and the target (scene.xml right_visual, left_visual)."""
        if len(self.scene_ids) != 2:
            raise SystemExit(f"[TACO_SOURCE] ✗ {len(self.scene_ids)} scene objects, not 2: {self.scene_ids}")
        return str(self.scene_ids[0]), str(self.scene_ids[1])


def default_root() -> Path:
    from dexengine.core.locations.layers import taco_source_root  # noqa: PLC0415 — needs the data root

    return taco_source_root()


def kind(root: Path | None = None) -> str:
    """The form of the source at `root` (module doc); a missing root dies. Prints `[TACO_SOURCE] kind=…`."""
    root = Path(root) if root is not None else default_root()
    st = shipped.state(NAME, root)
    k = PACK if st == shipped.VERIFIED else V1_TREE
    print(f"[TACO_SOURCE] kind={k} root={root}", flush=True)
    return k


@functools.lru_cache(maxsize=4)
def checked(root: Path) -> str:
    """`kind(root)` once per process (readers that open the source per clip — the 4b reference)."""
    return kind(Path(root))


def clips(root: Path) -> list[str]:
    """Clip names of the source (its folders), sorted."""
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


def read(root: Path, k: str, clip: str) -> Source:
    """One clip of the source in form `k` (from `kind`). A missing file dies."""
    d = Path(root) / clip
    ti = d / TASK_INFO
    if not ti.is_file():
        raise InputMissingError(f"[TACO_SOURCE] {clip}: {ti} missing")
    info = json.loads(ti.read_text())
    if k == V1_TREE:
        z = np.load(d / V1_NPZ)
        ids = side_objects.scene_object_ids(d)
        if ids != list(side_objects.scene_visual_ids(d)):  # the pack's slot order states the visuals
            raise SystemExit(f"[TACO_SOURCE] ✗ {clip}: scene order {ids} is not (right_visual, left_visual)")
        return Source(info, ids, z["qpos"][:, OBJECT_COLS], float(z["frequency"]))
    if k == PACK:
        meta = json.loads((d / SOURCE_JSON).read_text())
        return Source(info, list(meta["scene_object_ids"]), np.load(d / OBJECT_QPOS), float(meta["frequency"]))
    raise ValueError(f"[TACO_SOURCE] unknown source kind {k!r}")


def build(v1_root: Path, out: Path) -> str:
    """Write the pack of every clip of the V1 trees into `out` and its checksum file; returns the digest."""
    v1_root, out = Path(v1_root), Path(out)
    names = clips(v1_root)
    for c in names:
        s = read(v1_root, V1_TREE, c)
        o = out / c
        o.mkdir(parents=True, exist_ok=True)
        (o / TASK_INFO).write_bytes((v1_root / c / TASK_INFO).read_bytes())
        (o / SOURCE_JSON).write_text(json.dumps({"frequency": s.frequency, "scene_object_ids": s.scene_ids}) + "\n")
        np.save(o / OBJECT_QPOS, np.ascontiguousarray(s.object_qpos, dtype=np.float64))
    digest = shipped.write_sums(NAME, out)
    print(f"[TACO_SOURCE] pack clips={len(names)} digest={digest} -> {out}", flush=True)
    return digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dexengine.data.readers.taco_source", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="V1 trees → pack")
    b.add_argument("--v1-root", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("clips", help="every clip of the source, one per line")
    c.add_argument("--root", type=Path, default=None)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.v1_root, a.out)
        return 0
    root = a.root or default_root()
    kind(root)
    names = clips(root)
    a.out.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    print(f"[TACO_SOURCE] clips={len(names)} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
