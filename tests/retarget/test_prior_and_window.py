"""The retargeting prior rules and the hyperextension window that applies them outside the IK."""

from __future__ import annotations

import ast
import json
import math
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
    for joint, sign, neutral, role in prior.hyperext_clip_targets(conf):
        assert role in conf["clip_roles"]
        lo, hi = prior.hyperext_clip_range(-3.0, 3.0, sign, neutral, prior.IK_HYPEREXT_ALLOW_RAD)
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


def test_allow_roles_are_the_clip_roles():
    """The hand spec's allowance roles and the prior's clip roles are one tuple (no second table)."""
    from dexengine.core.hands.spec import HYPEREXT_ROLES

    assert prior._PRIOR_CLIP_ROLE_CHOICES is HYPEREXT_ROLES
    assert prior.IK_HYPEREXT_ALLOW_RAD == 0.0


@pytest.mark.parametrize(
    "lo,hi,sign,neutral",
    [
        (-1.05, 1.57, 1.0, 0.0),
        (0.0, 2.2, 1.0, 0.0),
        (-2.67, 0.0, -1.0, 0.0),
        (-1.57, 1.57, 1.0, 0.0475),
        (0.3, 1.0, 1.0, 0.5),
    ],
)
def test_ik_cut_is_the_cut_at_neutral(lo, hi, sign, neutral):
    """The IK's allowance (0) reproduces the cut at neutral bit for bit — stage-1 outputs do not move."""

    def at_neutral(lo, hi, sign, neutral):
        if sign >= 0:
            n = max(lo, neutral)
            return (n, hi) if n < hi else (lo, hi)
        n = min(hi, neutral)
        return (lo, n) if n > lo else (lo, hi)

    got = prior.hyperext_clip_range(lo, hi, sign, neutral, prior.IK_HYPEREXT_ALLOW_RAD)
    assert got == at_neutral(lo, hi, sign, neutral)


def test_allowance_follows_sign_and_never_widens_the_vendor_range():
    a = math.radians(20.0)
    assert prior.hyperext_clip_range(-1.047, 1.57, 1.0, 0.0, a) == (-a, 1.57)  # flexion +: lower limit at −20°
    assert prior.hyperext_clip_range(-2.67, 0.0, -1.0, 0.0, a) == (-2.67, 0.0)  # flexion −, outer limit 0: unchanged
    assert prior.hyperext_clip_range(-2.67, 1.0, -1.0, 0.0, a) == (-2.67, a)  # flexion −: upper limit at +20°
    assert prior.hyperext_clip_range(-0.1, 1.57, 1.0, 0.0, a) == (-0.1, 1.57)  # vendor limit inside the allowance


def _rig_ranges(spec):
    from dexengine.retarget.ik import hyperext_limits as HL

    return {k: list(v) for k, v in HL._rig_ranges(spec, "bimanual").items()}


@pytest.mark.parametrize("spec", ["wuji2", "dg5f_s"])
def test_ik_window_ignores_the_sim_allowance(spec):
    """Without --sim the window (the reference finger range: f0, placement repairs, gate) cuts at neutral whatever the
    hand spec's sim allowance; with --sim the stated allowance is applied, per role."""
    from dexengine.core.hands import sim_hyperext_allow_deg
    from dexengine.retarget.ik import hyperext_limits as HL

    ranges = _rig_ranges(spec)
    ik = HL.table(spec, "bimanual", ranges)
    assert ik["cut"] == "ik" and all(r["allow"] == 0.0 for r in ik["rows"])
    sim = HL.table(spec, "bimanual", ranges, sim=True)
    allow = sim_hyperext_allow_deg(spec)
    assert sim["sim_allow_deg"] == allow
    roles = [
        role
        for *_, role in prior.hyperext_clip_targets(
            prior.build_prior_conf(
                str(paths.hand_rig_dir(spec) / "retarget_prior.yaml"), spec, prior.emb_sides("bimanual")
            )[0]
        )
    ]
    for r, role in zip(sim["rows"], roles, strict=True):
        assert r["allow"] == math.radians((allow or {}).get(role, 0.0))
    assert "hyperext" not in prior.prior_marker(ik["meta"])  # the IK's marker and meta carry no allowance


def _fake_allow(monkeypatch, value):
    from dexengine.core import hands

    monkeypatch.setattr(hands, "sim_hyperext_allow_deg", lambda hand: value)


def test_sim_allowance_on_a_disabled_prior_dies(monkeypatch):
    """Negative: a hand whose prior is off (no cut) states a sim allowance — a value with no consumer."""
    from dexengine.retarget.ik import hyperext_limits as HL

    _fake_allow(monkeypatch, {"mcp": 20.0, "pip": 0.0, "dip": 0.0})
    with pytest.raises(SystemExit, match="no consumer"):
        HL.table("sharpa_wave", "bimanual", {}, off_ok=True, sim=True)
    _fake_allow(monkeypatch, None)  # control
    assert HL.table("sharpa_wave", "bimanual", {}, off_ok=True, sim=True)["rows"] == []


def test_sim_allowance_on_an_uncut_role_dies(monkeypatch):
    from dexengine.retarget.ik import hyperext_limits as HL

    _fake_allow(monkeypatch, {"mcp": 0.0, "pip": 0.0, "dip": 0.0})  # control: zeros pass
    HL.table("dg5f_s", "bimanual", _rig_ranges("dg5f_s"), sim=True)
    monkeypatch.setattr(HL, "_sim_allow", HL._sim_allow)  # the real check, a prior that cuts only pip/dip
    conf = {"clip_roles": ("pip", "dip"), "limit_hyperextension": True}
    _fake_allow(monkeypatch, {"mcp": 20.0, "pip": 0.0, "dip": 0.0})
    with pytest.raises(SystemExit, match="no consumer"):
        HL._sim_allow("dg5f_s", conf)


@pytest.mark.parametrize(
    "bad",
    [
        {"mcp": 20.0},
        {"mcp": 20.0, "pip": 0.0, "dip": 0.0, "cmc": 0.0},
        {"mcp": -5.0, "pip": 0.0, "dip": 0.0},
        {"mcp": 95.0, "pip": 0.0, "dip": 0.0},
    ],
)
def test_hand_spec_rejects_a_bad_allowance(bad):
    import dataclasses

    from dexengine.core.hands import HAND_REGISTRY

    with pytest.raises(ValueError, match="sim_finger_hyperext_allow_deg"):
        dataclasses.replace(HAND_REGISTRY["wuji2"], sim_finger_hyperext_allow_deg=bad)
