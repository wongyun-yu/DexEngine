"""Rule A (canonical-cell fit, placement.judge.canon_fit) on a fake judge that carries the real Judge origin, pivot-swap
and cell-offset functions (CPU only).

  T1 within the tolerance (residual 0.0014 step, the adopted record's rounding) = unchanged: origin d0 bit-equal to
     Judge._origin_shift and no replacement left behind
  T2 outside the tolerance (residual 0.342 step, a measured h2o clip) = the canonical placement lands on a grid point
  T3 with a pivot-swap term (adopted pivot ≠ clip pivot, yaw ≠ 0) still a grid point; the d0s ↔ d0cap relation holds
  T4 a fit is needed but the yaw is not a yaw grid point → dies
  T5 structure: the Judge's only caller of _origin_shift is _clip_index_cache (what the replacement relies on); the
     tolerance has one home (record.augment.policy only reads it)
  T6 whole-cell shift: a canonical placement one cell outside the window (window x offset −13 at half-width 12) moves
     the origin by whole cells and lands on the window's edge cell
  T7 canonical-cell volume exemption: a canonical cell that passed the desk mask but is out of the volume enters it
     (also when the whole volume is empty) · other cells unchanged · a canonical cell dropped by the desk mask (robot
     spot) is not exempted · the judge's _volume dies only when the volume is empty and the canonical cell was not
     added
  Negative variants of canon_fit: tolerance condition removed → T1 red · tolerance ½ step → T2 red · whole-cell shift
  removed → T6 red.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from dexengine.core.paths import PLACEMENT_SFX, tree_name  # noqa: E402
from dexengine.placement.judge import aug_cells as JAC  # noqa: E402
from dexengine.placement.judge import canon_fit  # noqa: E402
from dexengine.placement.judge.core import Judge  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STEP = 0.035
YAWS = [float(y) for y in np.arange(-90.0, 90.0 + 1e-9, 2.5)]
TARGET = (-0.34, -0.02)


class FakeJ:
    """The Judge's origin · pivot swap · cell offset functions as they are; _clip_index_cache keeps only the line that
    writes the origin."""

    _origin_shift = Judge._origin_shift
    _pivot_swap = Judge._pivot_swap
    _cell_dcap = Judge._cell_dcap
    _adopted_path = Judge._adopted_path
    _adopted_pivot = Judge._adopted_pivot

    def __init__(self, pl_dir: Path):
        from scipy.spatial.transform import Rotation

        self.rz = Rotation.from_euler("z", -90, degrees=True)
        self.D0_SCENE = np.array(TARGET, float)
        self.step, self.yaws, self.Wx, self.Wy = STEP, YAWS, 12, 21
        self.NBx, self.NBy = 2 * self.Wx + 1, 2 * self.Wy + 1
        self.PL_DIR = pl_dir
        self.seen = []

    def _clip_index_cache(self, ck):  # the real one: ck.d0s, d0cap = self._origin_shift(cpiv) … ck.d0cap = d0cap
        ck.d0s, d0cap = self._origin_shift(ck.cpiv)
        ck.d0cap = d0cap
        self.seen.append((ck.d0s.copy(), d0cap.copy()))


def make(tmp: Path, clip: str, cpiv, p2, canon_xy, yaw):
    pl = tmp / clip
    pl.mkdir(parents=True, exist_ok=True)
    (pl / f"{tree_name(clip)}{PLACEMENT_SFX}").write_text(json.dumps({"pivot": [p2[0], p2[1], 0.0]}))
    J = FakeJ(pl)
    A = object.__new__(JAC.AugCells)
    A.J = J
    ck = types.SimpleNamespace(clip=clip, cpiv=np.array([cpiv[0], cpiv[1], 0.0]))
    cn = {"dx": float(canon_xy[0]), "dy": float(canon_xy[1]), "yaw_deg": float(yaw), "dz": -3.5}
    return A, J, ck, cn


def nearest(A, ck, cn) -> float:
    J = A.J
    ky = int(np.argmin([abs((y - cn["yaw_deg"] + 180) % 360 - 180) for y in J.yaws]))
    return min(
        max(abs(float(d[0]) - cn["dx"]), abs(float(d[1]) - cn["dy"]))
        for kx in range(J.NBx)
        for kyy in range(J.NBy)
        for d in [A._geom(ck, (ky, kx, kyy, cn["dz"]))[2]]
    )


def on_grid_canon(J, ck, frac_step: float, yaw: float, p2):
    """Canonical placement frac_step off the d0 (target-point shift) grid — window centre offset + whole cells +
    offset."""
    _d0s, d0cap = Judge._origin_shift(J, ck.cpiv)
    th = np.deg2rad(yaw)
    R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    d_c = d0cap[:2] + (np.eye(2) - R2) @ (ck.cpiv[:2] - np.asarray(p2))
    return d_c + np.array([3 * STEP, -5 * STEP]) + frac_step * STEP


def t1(CF, tmp: Path) -> None:
    A, J, ck, cn = make(tmp, "t1", (0.30, 0.10), (0.30, 0.10), (0, 0), -30.0)
    c = on_grid_canon(J, ck, 0.0014, -30.0, (0.30, 0.10))
    cn["dx"], cn["dy"] = float(c[0]), float(c[1])
    want = Judge._origin_shift(J, ck.cpiv)
    assert CF.fit_origin(J, ck, cn, JAC.die) is None, "T1 within the tolerance but a fit came out"
    CF.index_cache(J, ck, cn, A._geom, JAC.die)
    assert "_origin_shift" not in J.__dict__, "T1 the replaced _origin_shift was left behind"
    got = J.seen[-1]
    assert got[0].tobytes() == want[0].tobytes() and got[1].tobytes() == want[1].tobytes(), (
        f"T1 the origin of a clip within the tolerance changed {got} != {want}"
    )


def t2(CF, tmp: Path) -> None:
    cpiv = (0.008, 0.286)  # a measured h2o clip (adopted pivot ≈ clip pivot)
    A, J, ck, cn = make(tmp, "t2", cpiv, cpiv, (0.21, -0.42), -82.5)
    _d0s, d0cap = Judge._origin_shift(J, ck.cpiv)
    r0 = canon_fit.residual((0.21 - d0cap[0], -0.42 - d0cap[1]), STEP)
    assert 0.33 < max(abs(r0[0]), abs(r0[1])) / STEP < 0.35, f"T2 oracle residual {r0} is not the measured 0.342 step"
    J.seen.clear()
    J._clip_index_cache(ck)
    assert nearest(A, ck, cn) > canon_fit.CANON_MATCH_FRAC * STEP, "T2 already a grid point before the fit"
    CF.index_cache(J, ck, cn, A._geom, JAC.die)
    assert "_origin_shift" not in J.__dict__, "T2 the replaced _origin_shift was left behind"
    e = nearest(A, ck, cn)
    assert e <= 1e-9, f"T2 nearest cell after the fit is {e} m from the canonical placement"


def t3(CF, tmp: Path) -> None:
    cpiv, p2 = (0.12, 0.31), (0.05, 0.27)
    A, J, ck, cn = make(tmp, "t3", cpiv, p2, (0, 0), 37.5)
    c = on_grid_canon(J, ck, 0.27, 37.5, p2)
    cn["dx"], cn["dy"] = float(c[0]), float(c[1])
    CF.index_cache(J, ck, cn, A._geom, JAC.die)
    assert nearest(A, ck, cn) <= 1e-9, "T3 not a grid point with a pivot-swap term"
    assert np.allclose(ck.d0cap[:2], [-ck.d0s[1], ck.d0s[0]], atol=1e-15), "T3 the d0s <-> d0cap relation broke"


def t4(CF, tmp: Path) -> None:
    A, J, ck, cn = make(tmp, "t4", (0.2, 0.2), (0.2, 0.2), (0, 0), 1.3)
    c = on_grid_canon(J, ck, 0.3, 1.3, (0.2, 0.2))
    cn["dx"], cn["dy"] = float(c[0]), float(c[1])
    with pytest.raises(SystemExit, match="yaw grid point"):
        CF.index_cache(J, ck, cn, A._geom, JAC.die)


def t6(CF, tmp: Path) -> None:
    A, J, ck, cn = make(tmp, "t6", (0.25, 0.05), (0.25, 0.05), (0, 0), 32.5)
    _d0s, d0cap = Judge._origin_shift(J, ck.cpiv)
    th = np.deg2rad(32.5)
    R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    d_c = d0cap[:2] + (np.eye(2) - R2) @ (ck.cpiv[:2] - np.asarray((0.25, 0.05)))
    off = J.rz.inv().apply([-13 * STEP, 0.0, 0.0])[:2]  # window x −13 cells (half-width 12) · a grid point
    cn["dx"], cn["dy"] = float(d_c[0] + off[0]), float(d_c[1] + off[1])
    J.seen.clear()
    J._clip_index_cache(ck)
    assert nearest(A, ck, cn) >= 0.99 * STEP, "T6 the canonical placement is inside the window before the fit"
    got = CF.fit_origin(J, ck, cn, JAC.die)
    assert got is not None and tuple(got[3]) == (-1, 0), f"T6 window cell shift {None if got is None else got[3]}"
    CF.index_cache(J, ck, cn, A._geom, JAC.die)
    assert "_origin_shift" not in J.__dict__, "T6 the replaced _origin_shift was left behind"
    assert nearest(A, ck, cn) <= 1e-9, "T6 after the shift the canonical placement is not a window cell"
    moved = d_c + (ck.d0cap[:2] - d0cap[:2])
    w = J.rz.apply([cn["dx"] - moved[0], cn["dy"] - moved[1], 0.0])[:2] / STEP
    assert abs(round(w[0])) == J.Wx and abs(round(w[1])) == 0, f"T6 window offset {w} is not the edge cell"


def t7(CF, tmp: Path) -> None:
    A, J, ck, cn = make(tmp, "t7", (0.30, 0.10), (0.30, 0.10), (0, 0), -30.0)
    c = on_grid_canon(J, ck, 0.0, -30.0, (0.30, 0.10))
    cn["dx"], cn["dy"] = float(c[0]), float(c[1])
    J._clip_index_cache(ck)
    J.DZ = [-3.5, 0.0]
    (ky, kx, kyy), e = A._canon_grid(ck, cn)
    assert e <= 1e-9, f"T7 the canonical placement is not a grid point {e}"
    shape = (len(J.yaws), J.NBx, J.NBy)
    ck.V = {z: np.zeros(shape, bool) for z in J.DZ}
    ck.COV = {z: np.zeros(shape, int) for z in J.DZ}
    ck.n95, ck.desk_m = 68, np.ones(shape, bool)
    assert CF.volume_exempt(J, ck, cn, *A._canon_grid(ck, cn), "[T]") and int(ck.V[-3.5].sum()) == 1, (
        "T7 a clip whose whole volume is empty must get the canonical cell alone"
    )
    ck.V[-3.5][ky, kx, kyy] = False
    ck.V[-3.5][0, 0, 0] = True
    CF.volume_exempt(J, ck, cn, *A._canon_grid(ck, cn), "[T]")
    assert ck.V[-3.5][ky, kx, kyy] and int(ck.V[-3.5].sum()) == 2 and not ck.V[0.0].any(), (
        "T7 only the canonical cell may enter the volume"
    )
    ck.V[-3.5][ky, kx, kyy] = False
    ck.desk_m[ky, kx, kyy] = False
    CF.volume_exempt(J, ck, cn, *A._canon_grid(ck, cn), "[T]")
    assert not ck.V[-3.5][ky, kx, kyy], "T7 a canonical cell dropped by the desk mask (robot spot) was exempted"


CASES = {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T6": t6, "T7": t7}


@pytest.mark.parametrize("name", sorted(CASES))
def test_rule_a(name, tmp_path):
    CASES[name](canon_fit, tmp_path)


@pytest.mark.parametrize("desk_keeps_canon", [True, False])
def test_volume_dies_only_when_empty_and_canon_not_added(desk_keeps_canon, tmp_path):
    """AugCells._volume runs the exemption first: an empty volume with the canonical cell added goes on; with the
    canonical cell masked by the desk (robot spot) it dies."""
    A, J, ck, cn = make(tmp_path, "tv", (0.30, 0.10), (0.30, 0.10), (0, 0), -30.0)
    c = on_grid_canon(J, ck, 0.0, -30.0, (0.30, 0.10))
    cn["dx"], cn["dy"] = float(c[0]), float(c[1])
    J.DZ = [-3.5]
    shape = (len(J.yaws), J.NBx, J.NBy)
    ck.pr, ck.n95, ck.desk_m = None, 68, np.ones(shape, bool)

    def desk_mask(ck_):
        if not desk_keeps_canon:
            ck_.desk_m[:] = False

    def volume(ck_):
        ck_.V, ck_.COV = {-3.5: np.zeros(shape, bool)}, {-3.5: np.zeros(shape, int)}
        return False

    J._clip_desk_mask, J._clip_volume = desk_mask, volume
    if desk_keeps_canon:
        A._volume(ck, cn)
        assert int(ck.V[-3.5].sum()) == 1
    else:
        with pytest.raises(SystemExit, match="no volume"):
            A._volume(ck, cn)


def test_origin_shift_has_one_judge_caller():
    """T5 — the replacement in index_cache relies on _clip_index_cache being the Judge's only caller."""
    callers = set()
    for cls in Judge.__mro__:
        if not cls.__module__.startswith("dexengine."):
            continue
        tree = ast.parse(Path(inspect.getsourcefile(cls)).read_text())
        for f in ast.walk(tree):
            if isinstance(f, ast.FunctionDef) and f.name != "_origin_shift":
                if any(
                    isinstance(n, ast.Attribute) and n.attr == "_origin_shift" and isinstance(n.ctx, ast.Load)
                    for n in ast.walk(f)
                ):
                    callers.add(f.name)
    assert callers == {"_clip_index_cache"}


def test_match_tolerance_has_one_home():
    """T5 — record.augment.policy reads CANON_MATCH_FRAC from canon_fit and never assigns it."""
    from dexengine.record.augment import policy as AP

    body = ast.parse((REPO / "dexengine/record/augment/policy.py").read_text()).body
    assert not any(
        isinstance(n, (ast.Assign, ast.AnnAssign)) and "CANON_MATCH_FRAC" in ast.unparse(n).split("=")[0] for n in body
    )
    assert AP.CANON_MATCH_FRAC == canon_fit.CANON_MATCH_FRAC


def _variant(tmp: Path, old: str, new: str):
    src = Path(canon_fit.__file__).read_text()
    assert src.count(old) == 1, old
    f = tmp / "canon_fit_variant.py"
    f.write_text(src.replace(old, new))
    spec = importlib.util.spec_from_file_location("canon_fit_variant", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NEGATIVE = {
    "condition removed": (("if max(abs(r[0]), abs(r[1])) <= CANON_MATCH_FRAC * float(step):", "if False:"), t1),
    "tolerance half step": (("CANON_MATCH_FRAC = 0.01", "CANON_MATCH_FRAC = 0.5"), t2),
    "whole-cell shift removed": (
        ("s.append(n - (half if n > 0 else -half) if abs(n) > half else 0)", "s.append(0)"),
        t6,
    ),
}


@pytest.mark.parametrize("name", sorted(NEGATIVE))
def test_negative_variants_are_caught(name, tmp_path):
    (old, new), case = NEGATIVE[name]
    with pytest.raises(AssertionError):
        case(_variant(tmp_path, old, new), tmp_path / "case")
