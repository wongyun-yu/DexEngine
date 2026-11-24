"""The augment bake step list (dexengine.record.augment.bake.step_specs) follows the bake chain's step order
(scripts/stage2/chain_fast.sh)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("idle,xr", [((), False), (("left",), False), ((), True), (("right",), True)])
def test_augment_bake_steps_follow_the_bake_chain_order(idle, xr):
    """augment.bake.step_specs builds the bake chain (scripts/stage2/chain_fast.sh) minus its first `bake` step; the
    step order must equal what _chain_step_names reads from the chain file."""
    import types

    from dexengine.record.augment import bake as AB

    src = types.SimpleNamespace(
        trees_rel="t",
        wrist_rel={"right": "wr", "left": "wl"},
        ik_rel={"right": "ir", "left": "il"},
        arm="rby1",
        hand="h",
    )
    v = {"zoff": 0.0, "pool": "/p", "idle": list(idle), "safety": 1.0, "max_stretch": 2.0, "assets": "/a"}
    got = [s.partition(":")[0] for s in AB.step_specs(src, REPO / "shadow", "clip", {}, xr, v)]
    assert got == _chain_step_names(bool(idle), xr)


def _chain_step_names(idle: bool, xr: bool) -> list[str]:
    """Step name order chain_fast.sh builds, without its first `bake` step (fails when the chain form changes)."""
    import re

    text = (REPO / "scripts/stage2/chain_fast.sh").read_text()

    def arr(var: str) -> list[str]:
        m = re.search(rf"^{var}=\((.*?)\)\s*$", text, re.S | re.M)
        assert m, f"chain_fast.sh: {var}=( … ) not found"
        return re.findall(r'"(\w+):', m.group(1))

    pre, post = arr("PRE"), arr("POST")
    assert pre[:1] == ["bake"], f"chain_fast.sh: first PRE step {pre[:1]} != bake"
    ext = re.search(r'if \[ -n "\$IDLE" \]; then(.*?)\nfi', text, re.S)
    assert ext, "chain_fast.sh: no idle side block"
    idle_steps = re.findall(r'PRE\+=\("(\w+):', ext.group(1))
    xr_steps = re.findall(r'ALL=\("\$\{PRE\[@\]\}" "(\w+):', text)
    assert xr_steps == ["xr"], f"chain_fast.sh: the cross repair step form changed {xr_steps}"
    return pre[1:] + (idle_steps if idle else []) + (xr_steps if xr else []) + post
