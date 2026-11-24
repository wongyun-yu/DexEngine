"""The retargeting prior rules and the hyperextension window that applies them outside the IK."""

from __future__ import annotations

import ast
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from dexengine.core import paths  # noqa: E402
from dexengine.retarget.ik import prior  # noqa: E402

RIGS = ("sharpa_wave", "dg5f_s", "wuji2", "xhand_pro")


def test_prior_owns_the_rules():
    """prior.py owns the rules (no other module of the hand-tracker package defines its public names again); emb_sides
    is a rule, not a table."""
    owned = {
        n
        for n in vars(prior)
        if not n.startswith("_")
        and callable(getattr(prior, n))
        and getattr(getattr(prior, n), "__module__", "") == prior.__name__
    }
    others = [f for f in Path(prior.__file__).parent.glob("*.py") if f.name not in ("prior.py", "hyperext_limits.py")]
    defined = {
        n.name for f in others for n in ast.parse(f.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    assert owned and not (owned & defined)
    assert tuple(prior.emb_sides("bimanual")) == ("right", "left")
    assert tuple(prior.emb_sides("right")) == ("right",)
    with pytest.raises(ValueError, match="unknown embodiment_type"):
        prior.emb_sides("both")


@pytest.mark.parametrize("spec", RIGS)
def test_prior_conf_and_clip_targets(spec):
    path = str(paths.hand_rig_dir(spec) / "retarget_prior.yaml")
    conf, meta = prior.build_prior_conf(path, spec, prior.emb_sides("bimanual"))
    assert prior.prior_marker(meta).startswith("[PRIOR] ")
    if conf is None or not conf["limit_hyperextension"]:
        return
    for joint, sign, neutral in prior.hyperext_clip_targets(conf):
        lo, hi = prior.hyperext_clip_range(-3.0, 3.0, sign, neutral)
        assert -3.0 <= lo <= hi <= 3.0, joint


@pytest.mark.parametrize("hand", ["dg5f_s", "wuji2"])
def test_truncation_window_reproduces_the_committed_ledger(hand, monkeypatch):
    """The window module, run as truncate.py runs it, clips exactly the limits the committed truncated asset records."""
    from dexengine.robots.urdf import truncate as T

    monkeypatch.setenv("RETARGET_PY", sys.executable)
    cdir = paths.combo_dirname(paths.combo_for(hand))
    prov = T.FULL_ROBOT_URDF_DIR / f"{cdir}{T.SUFFIX}" / T.PROV_NAME
    if not prov.is_file():
        pytest.skip(f"no committed truncation ledger for {hand}")
    canon = T.FULL_ROBOT_URDF_DIR / cdir / f"{cdir}.urdf"
    win = T._window(paths.hand_spec_dir(hand), T._limits(ET.parse(canon).getroot()))
    got = {r["joint"]: [r["new_lo"], r["new_hi"]] for r in win["rows"]}
    want = {j["joint"]: j["new_rad"] for j in json.loads(prov.read_text())["joints"]}
    assert got == want
