"""Every robot asset a stage opens resolves, for every hand mounted on rby1 — through each stage's own resolver:
f0 (repair_cfg.HandCfg.urdf, both sides) · cross repair and judge (sim_collider.combo_urdf / resolved_urdf) ·
RL spawn (selection.combo_usd_path) · recording (the plate, selection._usd_literal with plate_variant).
A URDF also has every mesh it names resolved by the stages' rule (paths.urdf_mesh). Meshes and USDs that live under
the data root (data/…) are checked when the data root holds them (DEXENGINE_MAIN_ROOT, the canonical machine);
anything in the repository is always checked."""

import os
import re
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.core.mounts import MOUNT_REGISTRY

ARM = "rby1"
DATA_ROOT = Path(os.environ.get("DEXENGINE_MAIN_ROOT", "/nonexistent"))
HAS_DATA = (DATA_ROOT / "data" / "_stage").is_dir()  # vendor files placed (`python -m dexengine.robots.vendor fetch`)
#: built robot USDs and canonical trees are separate layers: a root with only the fetched vendor files has neither
HAS_USD = (DATA_ROOT / "data" / "usd_cache").is_dir()
HANDS = sorted(h for (a, h), m in MOUNT_REGISTRY.items() if a == ARM and m is not None)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """The real data root (captured at import — the suite's autouse fixture points each test at an empty one)."""
    monkeypatch.setenv("DEXENGINE_ARM", ARM)
    if HAS_DATA:
        monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(DATA_ROOT))


def _missing_meshes(urdf: Path) -> tuple[list[str], int]:
    """(missing mesh refs, refs skipped because they live under an absent data root)."""
    miss, skipped = [], 0
    for n in sorted(set(re.findall(r'filename="([^"]+)"', urdf.read_text(encoding="utf-8")))):
        p = Path(P.urdf_mesh(urdf, n))
        if p.is_file():
            continue
        if "data/" in n and not HAS_DATA:
            skipped += 1
            continue
        miss.append(n)
    return miss, skipped


def _urdf_resolves(urdf) -> None:
    urdf = Path(urdf)
    assert urdf.is_file(), f"URDF missing: {urdf}"
    miss, _skipped = _missing_meshes(urdf)
    assert not miss, f"{urdf.name}: {len(miss)} meshes do not resolve, e.g. {miss[:3]}"


def test_hands_are_the_canonical_four():
    assert len(HANDS) == 4, HANDS


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize("side", ["right", "left"])
def test_f0_urdf(hand, side, monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", hand)
    from dexengine.retarget.f0 import repair_cfg as RC

    _urdf_resolves(RC.HandCfg(hand, ARM).urdf(side))


@pytest.mark.parametrize("hand", HANDS)
def test_cross_repair_and_judge_urdf(hand, monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", hand)
    monkeypatch.delenv("DEXENGINE_XR_URDF", raising=False)
    from dexengine.placement.collision import sim_collider as SC

    _urdf_resolves(SC.combo_urdf(ARM, hand))


@pytest.mark.skipif(not HAS_USD, reason="robot USDs live under the data root (DEXENGINE_MAIN_ROOT)")
@pytest.mark.parametrize("hand", HANDS)
def test_rl_and_record_usd(hand, monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", hand)
    from dexengine.core import arms as AR
    from dexengine.robots import selection as SEL

    usd_dir = str(DATA_ROOT / "data" / "usd_cache")
    assert Path(SEL.combo_usd_path(ARM, hand, usd_dir)).is_file()
    assert Path(SEL._usd_literal(ARM, hand, usd_dir, AR.plate_variant(ARM, hand))).is_file()


def test_a_missing_mesh_is_caught(tmp_path):
    """Negative: a URDF naming a mesh that is not there fails the check."""
    u = tmp_path / "x.urdf"
    u.write_text(
        '<robot name="x"><link name="a"><visual><geometry><mesh filename="nope.stl"/></geometry></visual>'
        "</link></robot>",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="do not resolve"):
        _urdf_resolves(u)
