"""Shared pytest setup: every test runs against an empty temporary data root, never the live data tree."""

from __future__ import annotations

import os

from dexengine.core import blas_pin  # standard library only: the pin is set before any test loads OpenBLAS

blas_pin.pin_env(os.environ)  # the f0 goldens are bytes of the pinned kernel

import pytest  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
