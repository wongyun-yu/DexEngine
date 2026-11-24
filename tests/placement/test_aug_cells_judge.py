"""Judge side of the augmentation cells (placement.judge.aug_cells) against the real judge argument parser (CPU only).

The cases: the guard of
modes D · L reads only flags the judge still has, and the region dump carries the f0 marker key.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest

from dexengine.placement.judge import aug_cells as JAC
from dexengine.placement.judge.args import _build_parser

SRC = Path(JAC.__file__).read_text()


def judge_args(tmp: Path, *extra: str):
    return _build_parser().parse_args(
        ["--arm", "rby1", "--hand", "h", "--dataset", "taco", "--emb", "bimanual", "--out", str(tmp / "out"), *extra]
    )


def test_every_judge_flag_read_by_aug_cells_exists():
    """Every a.<name> the module reads is a destination of the judge parser (a removed flag would crash modes D · L
    at boot with AttributeError)."""
    dests = {x.dest for x in _build_parser()._actions}
    assert set(re.findall(r"\ba\.([a-z_0-9]+)", SRC)) - dests == set()


@pytest.mark.parametrize("mode,batch,dies", [("D", "1", False), ("L", "8", False), ("L", "1", True)])
def test_mode_guard(mode, batch, dies, tmp_path, monkeypatch):
    a = judge_args(tmp_path, "--batch-cells", batch)
    J = types.SimpleNamespace(a=a, combo="rby1_h")
    monkeypatch.setattr(JAC, "axis_combo", lambda *x: ("ax", tmp_path / "ledger.tsv", "rby1_h"))
    monkeypatch.setattr(JAC, "head_commit", lambda: "0" * 40)
    from dexengine.core import paths as P

    monkeypatch.setattr(P, "placement_dir", lambda combo, ds: tmp_path / "pl")
    args = (J, mode, str(tmp_path / "bundle"), "c1" if mode == "L" else "")
    if dies:
        with pytest.raises(SystemExit, match="batched gate"):
            JAC.AugCells(*args)
        return
    aug = JAC.AugCells(*args)
    assert aug.mode == mode and aug.combo == "rby1_h"
    if aug.fh:
        aug.fh.close()


def _dump_ctx(tmp: Path):
    """A judge D context: real cell geometry (Judge._cell_dcap · _pivot_swap), three volume cells on the canonical
    dz layer and one on another layer."""
    import json

    import numpy as np
    from scipy.spatial.transform import Rotation

    from dexengine.core.layout.registry import AXIS_HAND
    from dexengine.placement.judge.core import Judge

    J = types.SimpleNamespace(
        yaws=np.linspace(-45.0, 45.0, 7),
        Wx=6,
        Wy=5,
        step=0.035,
        DZ=[-1, 0, 1],
        Z_OFF=0.74,
        rz=Rotation.from_euler("z", 90, degrees=True),
        _F0ROOT=tmp / "f0",
        a=types.SimpleNamespace(hand=AXIS_HAND, wrist_root="auto", emb="bimanual"),
    )
    J._adopted_pivot = lambda clip, cpiv: np.array([0.31, -0.07])
    J._cell_dcap = Judge._cell_dcap.__get__(J)
    J._pivot_swap = Judge._pivot_swap.__get__(J)
    V = {z: np.zeros((7, 12, 10), bool) for z in J.DZ}
    V[0][3, 6, 5] = V[0][2, 4, 7] = V[0][5, 9, 1] = True
    V[1][0, 0, 0] = True
    ck = types.SimpleNamespace(clip="c", cpiv=np.array([0.12, -0.4, 0.0]), d0cap=np.array([0.01, -0.02, 0.0]), V=V)
    yaw = float(J.yaws[3])
    d, p2 = J._pivot_swap("c", ck.cpiv, J._cell_dcap(ck, 6, 5), yaw)
    cn = {
        "dx": float(d[0]),
        "dy": float(d[1]),
        "yaw_deg": yaw,
        "pivot": [float(p2[0]), float(p2[1])],
        "dz": 0.0,
        "src": "adp",
    }
    aug = object.__new__(JAC.AugCells)
    aug.J, aug.mode, aug.bundle, aug.commit, aug.n_cells = J, "D", tmp / "bundle", "0" * 40, 0
    JAC.dirs(aug.bundle)["region"].mkdir(parents=True)
    mk = JAC.FK.judge_marker(J, "c")
    mk.parent.mkdir(parents=True)

    def mark(**kw):
        mk.write_text(json.dumps({"input_key": "k1", "fp_design": "f0_in_v3", "status": "CLEAN", "P": 0, **kw}))

    return J, ck, cn, aug, mk, mark


def test_region_dump_carries_the_f0_key(tmp_path):
    """Mode D stores the marker key read before the clip; a marker that changes during the dump dies; fields outside
    KEY_FIELDS do not change the key; a judge that does not read f0 dies."""
    import numpy as np

    J, ck, cn, aug, mk, mark = _dump_ctx(tmp_path)
    mark()
    aug._f0k = JAC.FK.key_of(mk)
    aug._dump(ck, "jfp3|x", cn)
    z = np.load(JAC.region_path(aug.bundle, "c"), allow_pickle=False)
    assert str(z["f0_key"]) == aug._f0k == '{"fp_design": "f0_in_v3", "input_key": "k1", "status": "CLEAN"}'
    assert z["cells"].shape == (3, 3) and set(z["dz"]) == {0.0}
    mark(input_key="k2")
    with pytest.raises(SystemExit, match="changed during the judge"):
        aug._dump(ck, "jfp3|x", cn)
    mark(P=9)
    assert JAC.FK.key_of(mk) == aug._f0k
    J.a.wrist_root = ""
    with pytest.raises(SystemExit, match="does not read the f0 shadow"):
        JAC.FK.judge_marker(J, "c")


def test_prejudge_writes_the_marker_f0_key_reads():
    """One owner of the marker name: retarget.f0.prejudge takes MARKER from placement.judge.f0_key."""
    from dexengine.placement.judge import f0_key as FK
    from dexengine.retarget.f0 import prejudge

    assert prejudge.MARKER is FK.MARKER == "F0_PREJUDGE.json"
    assert 'MARKER = "' not in Path(prejudge.__file__).read_text()
