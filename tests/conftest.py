"""Shared pytest setup: every test runs against an empty temporary data root, never the live data tree."""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path

from dexengine.core import blas_pin  # standard library only: the pin is set before any test loads OpenBLAS

blas_pin.pin_env(os.environ)  # the f0 goldens are bytes of the pinned kernel

import pytest  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""

#: The retarget install's pins (pyproject.toml optional-dependencies "retarget"): where the retarget stages run.
#: Measured (tests/retarget f0 golden bits): numpy 2.3.5 · scipy 1.16.3 give the golden bits with warp-lang 1.11.0
#: (the pinned release) and with the recorded dev build alike; the Isaac env (numpy 1.26.4 · scipy 1.17.1 · warp 1.11.0,
#: same inputs) gives other last bits (<= 5e-16). So the guard reads numpy and scipy.
RETARGET_EXTRA = Path(__file__).resolve().parents[1] / "pyproject.toml"
RETARGET_NUMERIC = ("numpy", "scipy")  # the packages whose versions changed the float bits (measured)


def pytest_sessionstart(session):
    """A checkout without the robot files (a fresh clone: manufacturer files are not in git, rigs and merged URDFs are
    built) stops here with the two commands, instead of failing dozens of tests one by one."""
    from dexengine.robots import vendor

    missing = vendor.missing_in_checkout()
    if missing:
        pytest.exit(
            f"{vendor.MARK} {len(missing)} robot file(s) not in this checkout (e.g. {missing[0]}) — run "
            "`python -m dexengine.robots.vendor fetch --data-root <data root>` then "
            "`... build --data-root <data root>` (docs/installation.md: Robot, hand and scene assets)",
            returncode=4,
        )
    session.config._unavailable_hands = vendor.unavailable_hands()
    for hand, why in session.config._unavailable_hands.items():
        print(f"{vendor.MARK} hand {hand} not available ({why}): its parametrized tests are skipped")


def pytest_collection_modifyitems(config, items):
    """A test parametrized with a hand whose vendor source is not set (vendor.unavailable_hands) is skipped with the
    reason; every other hand runs."""
    un = getattr(config, "_unavailable_hands", {})
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        hit = sorted({v for v in params.values() if isinstance(v, str) and v in un})
        if hit:
            item.add_marker(pytest.mark.skip(reason=f"hand {hit[0]} not available: {un[hit[0]]}"))


def _need_retarget_pins(dists) -> None:
    """Skip, naming the versions, unless each of `dists` has its pyproject [retarget] release (a local build tag such
    as +cu128 is not part of the release). A package missing from the pins is an error, not a skip."""
    import tomllib

    deps = tomllib.loads(RETARGET_EXTRA.read_text(encoding="utf-8"))["project"]["optional-dependencies"]["retarget"]
    pins = {n.strip().lower(): v.strip() for n, sep, v in (d.partition("==") for d in deps) if sep}
    off = []
    for dist in dists:
        if dist not in pins:
            raise AssertionError(f"{dist} has no == pin in {RETARGET_EXTRA} [retarget]")
        try:
            have = metadata.version(dist)
        except metadata.PackageNotFoundError:
            have = "absent"
        if have.split("+")[0] != pins[dist]:
            off.append(f"{dist} {have} (pin {pins[dist]})")
    if off:
        pytest.skip(
            f"needs the retarget install (pyproject [retarget]): {' · '.join(off)} — golden bits of that install"
        )


@pytest.fixture
def retarget_venv():
    """The retarget install's numpy and scipy: golden bits of the retarget stages are those of that install."""
    _need_retarget_pins(RETARGET_NUMERIC)


@pytest.fixture
def retarget_torch():
    """retarget_venv plus torch: golden bits of torch CPU arithmetic (the bake's arm IK)."""
    _need_retarget_pins((*RETARGET_NUMERIC, "torch"))


def stage1_pin_off() -> str:
    """'' when every OpenBLAS this interpreter loads is a build the stage-1 pin supports (blas_pin.VERSIONS, the
    retargeting venv), else why a stage-1 entry cannot start here (its pin refuses another build, by design)."""
    import contextlib

    import numpy  # noqa: F401 — load numpy's OpenBLAS to read it

    with contextlib.suppress(ImportError):
        import scipy.linalg  # noqa: F401 — and scipy's

    from dexengine.core import blas_pin as BP

    def pinned(name: str) -> str | None:
        return next((v for k, v in BP.VERSIONS.items() if name.startswith(k)), None)

    off = [f"{n} {v}" for n, _core, v in BP.loaded() if pinned(n) != v]
    if not off:
        return ""
    return (
        "outside the stage-1 pin environment — stage-1 entries run in the retargeting venv "
        f"(loaded {', '.join(off)}; pinned builds blas_pin.VERSIONS)"
    )


@pytest.fixture
def stage1_pin_env():
    """Skip, printing why, a test that starts a stage-1 entry outside the pin's environment (the Isaac env). In the
    retargeting venv it never skips: the test runs there."""
    why = stage1_pin_off()
    if why:
        print(why)
        pytest.skip(why)


@pytest.fixture(autouse=True)
def retarget_py(monkeypatch):
    """The retargeting interpreter of a test is the one running pytest, named explicitly (RETARGET_PY): a code tree
    under test may have no .venv of its own (a worktree, a cluster container), and code that starts the retargeting
    interpreter (the finger-range window of wrist_shift.finger_ranges) must run in the interpreter the gate checks."""
    from dexengine.core.locations import roots

    monkeypatch.setenv(roots.RETARGET_PY_ENV, sys.executable)
    return sys.executable


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    """An empty data root for the test (exported as DEXENGINE_MAIN_ROOT, as `--data-root` would do)."""
    root = tmp_path / "data_root"
    root.mkdir()
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(root))
    for name in (
        "DEXENGINE_SANDBOX_ROOT",
        "DEXENGINE_ARM",
        "DEXENGINE_HAND",
        "DEXENGINE_DATASET",
        "DEXENGINE_EMBODIMENT",
        "DEXENGINE_POOL_LOCK",
    ):
        monkeypatch.delenv(name, raising=False)
    from dexengine.core.layout import roots

    monkeypatch.setattr(roots, "_LAYOUT_CFG", [])
    return root


@pytest.fixture
def run_dataset(monkeypatch):
    """Set the run's dataset (DEXENGINE_DATASET, which the shared setup clears) for code that reads it;
    `robot_static=False` opts that dataset out of S-robot-static (collision_presets.extra_collide)."""

    def _set(ds: str = "hot3d", robot_static: bool = True) -> str:
        import dataclasses

        from dexengine.core import datasets as RR

        monkeypatch.setenv("DEXENGINE_DATASET", ds)
        if not robot_static:
            p = RR.DATASET_PROFILE[ds]
            off = dict(p.scene_rules_off, **{"S-robot-static": "test opt-out"})
            monkeypatch.setitem(RR.DATASET_PROFILE, ds, dataclasses.replace(p, scene_rules_off=off))
        return ds

    return _set
