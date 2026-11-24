"""Isolated stage-2 run: root resolution, the test-tag rule and the write-path proof (dexengine.placement.sandbox)."""

from __future__ import annotations

import pytest

from dexengine.core import layout as DL
from dexengine.placement import sandbox as SB


def test_sandbox_root_rules(tmp_path):
    root = SB.resolve("sb", "VAL_x", dry=False, env={}, cwd=str(tmp_path))
    assert root == str((tmp_path / "sb").resolve()) and (tmp_path / "sb").is_dir()
    assert SB.resolve("", "VAL_x", dry=False, env={DL.SANDBOX_ENV: root}) == root  # inherited
    with pytest.raises(SystemExit, match="differs"):  # one input, two values
        SB.resolve(str(tmp_path / "other"), "VAL_x", dry=False, env={DL.SANDBOX_ENV: root})
    with pytest.raises(SystemExit, match="only sandboxed"):  # a test tag
        SB.resolve("", "G1regr_taco_0001", dry=False, env={})
    assert SB.resolve("", "G1regr_taco_0001", dry=True, env={}) == ""  # dry-run writes nothing
    assert SB.prove("", "/r", ["/r/x"]).startswith("[SANDBOX] none")
    assert SB.prove(root, "/r", [f"{root}/precompute/taco"]).startswith(f"[SANDBOX] root={root}")
    with pytest.raises(SystemExit, match="outside the sandbox root"):
        SB.prove(root, "/r", [f"{root}/precompute", "/r/precompute/taco"])
