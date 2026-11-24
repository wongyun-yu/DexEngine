"""An arm whose spec declares no torso (ArmSpec.torso_ruler false): the torso record is empty and the torso guard runs
through to a pass (no torso bodies to meet), with the [TORSO_RULER] marker — the same poses fail with the torso on.
The torch filter (torso_torch.torch_torso · torso_filter_sd_t, the search's GPU path, run here on the CPU) reads the
empty record as "nothing to meet" (1e9 on every pose) like the numpy filter, and with the torso on gives the numpy
filter's values (the same poses reach into the torso).

The arm context reads the hand meshes of the data root the suite was started with (read only; the shared conftest gives
every test an empty one) — skipped when that root has no staged hand meshes."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from dexengine.core import arms as AS
from dexengine.placement.collision import arm_proxy as AP
from dexengine.placement.collision import core as CC
from dexengine.placement.collision import torso as TS
from dexengine.robots import vendor as VD

pytest.importorskip("pinocchio")
pytestmark = pytest.mark.skipif(
    "xhand_pro" in VD.unavailable_hands(),
    reason="the example hand xhand_pro is not available (its vendor source is not set)",
)


def _real_data_root() -> Path:
    if os.environ.get("DEXENGINE_MAIN_ROOT"):
        return Path(os.environ["DEXENGINE_MAIN_ROOT"])
    from dexengine.core import paths

    return Path(paths.main_root())


REAL_DATA_ROOT = _real_data_root()


def _ctx(monkeypatch, torso_ruler: bool):
    """(arm context of rby1 + xhand_pro right with the torso on or off, 3 poses — the middle one reaches the torso)."""
    if not (REAL_DATA_ROOT / "data" / "_stage").is_dir():
        pytest.skip(f"no staged hand meshes under {REAL_DATA_ROOT}")
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(REAL_DATA_ROOT))
    monkeypatch.setitem(AS.ARM_REGISTRY, "rby1", dataclasses.replace(AS.ARM_REGISTRY["rby1"], torso_ruler=torso_ruler))
    monkeypatch.setattr(CC, "_CTX_CACHE", {})
    monkeypatch.setattr(TS, "_TORSO_CACHE", {})
    with AP.mesh_ruler():
        ctx = CC.build_arm_ctx("rby1", "xhand_pro", "right")
    q = np.zeros((3, len(ctx["joints"])))
    q[1, 1] = 0.5
    return ctx, q


def _guard(monkeypatch, torso_ruler: bool):
    ctx, q = _ctx(monkeypatch, torso_ruler)
    with AP.mesh_ruler():
        return ctx["torso"], TS.torso_guard(ctx, q, "right", np.zeros((3, ctx["torso"]["hand"]["n"])))


def test_no_torso_record_is_empty_and_the_guard_passes(monkeypatch, capsys):
    tz, (ok, why, d) = _guard(monkeypatch, torso_ruler=False)
    assert tz["bodies"] == [] and tz["torso"] == {} and tz["hand"]["n"] > 0
    assert ok and why == "" and d == pytest.approx(1e9)
    assert "[TORSO_RULER] rby1: no torso" in capsys.readouterr().out


def test_the_same_poses_meet_the_torso_when_it_is_on(monkeypatch):
    tz, (ok, why, _d) = _guard(monkeypatch, torso_ruler=True)
    assert tz["bodies"] and tz["torso"]
    assert not ok and why.startswith("righttorso_surface")


@pytest.mark.parametrize("torso_ruler", [False, True])
def test_the_torch_filter_reads_the_torso_record_like_numpy(monkeypatch, torso_ruler):
    torch = pytest.importorskip("torch")
    from dexengine.placement.collision import geom as CG
    from dexengine.placement.collision import torso_torch as TST

    ctx, q = _ctx(monkeypatch, torso_ruler)
    Ts = CG.arm_link_transforms(ctx, q)
    want = TS.torso_filter_sd(ctx["torso"], Ts)
    got = TST.torso_filter_sd_t(TST.torch_torso(ctx["torso"], "cpu"), torch.tensor(Ts, dtype=torch.float64)).numpy()
    if torso_ruler:
        assert want[1] < 0.0 < want[0] < 1e9 and got == pytest.approx(want, abs=1e-12)
    else:
        assert (want == 1e9).all() and (got == 1e9).all()
