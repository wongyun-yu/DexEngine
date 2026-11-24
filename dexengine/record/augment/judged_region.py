"""Region dump of a judged cell bundle (judge mode D): the dump name and the checked read (missing contract fields or
mismatched shapes die through augment.judged.die). Callers: augment.judged.load, cells_pick, cells_steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexengine.record.augment import policy as AP
from dexengine.record.augment.judged import _obj, die


def clip_of_region(path: Path) -> str:
    name = Path(path).name
    if not (name.startswith("region_") and name.endswith(".npz")):
        die(f"region dump name {name} — not of the form region_<clip>.npz")
    return name[len("region_") : -len(".npz")]


def read_region(path: Path) -> dict:
    """One region dump; missing contract fields or mismatched shapes die."""
    if not Path(path).is_file():
        die(f"region dump missing {path}", missing=True)
    z = np.load(path, allow_pickle=True)
    need = (
        "cells",
        "dz",
        "theta_abs_deg",
        "d_abs",
        "pivot",
        "canon",
        "robot_z_off",
        "grid",
        "judge_commit",
    ) + AP.JUDGE_ID_KEYS
    miss = [k for k in need if k not in z.files]
    if miss:
        die(f"{path}: {miss} missing from the region dump", missing=True)
    R = {
        "cells": np.asarray(z["cells"]),
        "dz": np.asarray(z["dz"], float),
        "theta_abs_deg": np.asarray(z["theta_abs_deg"], float),
        "d_abs": np.asarray(z["d_abs"], float),
        "pivot": np.asarray(z["pivot"], float),
        "canon": _obj(z["canon"]),
        "robot_z_off": float(z["robot_z_off"]),
        "grid": _obj(z["grid"]),
    }
    for k in AP.JUDGE_ID_KEYS + ("judge_commit",):
        R[k] = str(_obj(z[k]))
    n = len(R["cells"])
    if (
        R["cells"].ndim != 2
        or R["cells"].shape[1] != 3
        or not np.issubdtype(R["cells"].dtype, np.integer)
        or R["dz"].shape != (n,)
        or R["theta_abs_deg"].shape != (n,)
        or R["d_abs"].shape != (n, 2)
    ):
        die(
            f"{path}: shapes cells {R['cells'].shape} {R['cells'].dtype} · dz {R['dz'].shape} · "
            f"θ {R['theta_abs_deg'].shape} · "
            f"d {R['d_abs'].shape} — contract (N,3) int · (N,) · (N,) · (N,2)"
        )
    if not isinstance(R["canon"], dict) or not isinstance(R["grid"], dict):
        die(f"{path}: canon · grid is not a dict")
    AP.scale_of(R["grid"])
    return R
