"""The base-tree write of the stage-1 reference repair (2e wrist ↔ desk): new kinematic qpos from the repair's input
(wrist_shift backup chain, tag per repair), the same delta on the ikrollout, body_positions by the FK owner, and the
one undo of that write (also the way back for a tree that holds a 2f repair output, wrist_shift.HAND_CLEAR_TAG). The
backup rules are wrist_shift's.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from dexengine.placement.repair import wrist_shift as WS


def fk_body_positions(tree: Path) -> None:
    """Re-run the FK owner (body_positions.process_sequence, as base_adapter does) on the written qpos in a temp tree
    outside the data layer (the scene's asset dirs made absolute so it compiles there — an interrupted run leaves no
    sibling folder in the base layer) and replace the file atomically inside the tree."""
    import tempfile  # noqa: PLC0415
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    from dexengine.core import stdio  # noqa: PLC0415
    from dexengine.placement.fk import body_positions as BP  # noqa: PLC0415

    tree = Path(tree)
    with tempfile.TemporaryDirectory(prefix=f"ws_fk_{tree.name[:40]}_") as td:
        tmp = Path(td) / tree.name
        (tmp / "0").mkdir(parents=True)
        root = ET.parse(tree / "scene.xml").getroot()
        for comp in root.iter("compiler"):
            for key in ("meshdir", "texturedir", "assetdir"):
                if comp.get(key):
                    comp.set(key, str((tree / comp.get(key)).resolve()) + "/")
        ET.ElementTree(root).write(tmp / "scene.xml")
        shutil.copyfile(tree / "0/trajectory_kinematic.npz", tmp / "0/trajectory_kinematic.npz")
        with stdio.capture():
            ok = BP.process_sequence(str(tmp), verbose=False)
        if not ok or not (tmp / "0/body_positions.npz").is_file():
            raise RuntimeError(f"{tree.name}: body_positions FK failed")
        part = tree / "0/body_positions.npz.part"
        shutil.copyfile(tmp / "0/body_positions.npz", part)
        os.replace(part, tree / "0/body_positions.npz")


def frame_dt(tree: Path, tag: str) -> float:
    """The tree's frame period (s) — the ``frequency`` of the kinematic npz repair ``tag`` reads (none → dies: speeds
    and qvel need it)."""
    with np.load(WS.original(Path(tree) / "0/trajectory_kinematic.npz", tag)) as z:
        if "frequency" not in z.files or not float(z["frequency"]) > 0:
            raise ValueError(f"{Path(tree).name}: trajectory_kinematic.npz has no frequency")
        return 1.0 / float(z["frequency"])


def write_tree(tree: Path, q0: np.ndarray, new: np.ndarray, vel_cols, tag: str) -> None:
    """A base-tree repair's write: the new kinematic qpos (+ qpos_rollout, qvel by base_adapter's forward rule) from
    the repair's input, the same delta on the ikrollout qpos, then body_positions by the FK owner. ``q0`` = the input
    qpos the repair started from."""
    tree = Path(tree)
    kin, ro = tree / "0/trajectory_kinematic.npz", tree / "0/trajectory_ikrollout.npz"
    dt = frame_dt(tree, tag)
    with np.load(WS.source_of(kin, tag)) as z:
        WS.save_npz(kin, WS.apply_delta(z, new, vel_cols, dt, "forward"), tag)
    if ro.is_file():
        with np.load(WS.source_of(ro, tag)) as z:
            if np.asarray(z["qpos"]).shape == new.shape:
                nro = np.asarray(z["qpos"], float) + (new - q0)
                WS.save_npz(ro, WS.apply_delta(z, nro, [], dt, "forward"), tag)
    fk_body_positions(tree)


def undo_tree(tree: Path, tag: str) -> bool:
    """Repair ``tag``'s output on a base tree back to its input — the one undo path (``revert`` of qpos and the
    ikrollout, later repairs of CHAIN first, then the FK owner). False when the tree holds no output of it."""
    tree = Path(tree)
    kin = tree / "0/trajectory_kinematic.npz"
    if not WS.wrote(kin, tag):
        return False
    WS.revert(kin, tag)
    WS.revert(tree / "0/trajectory_ikrollout.npz", tag)
    fk_body_positions(tree)
    return True
