"""Stage-1 admission chain (dexengine.retarget.admit): order, marker checks, rc per failed step, stamps only after
intake, TACO without a stage. The step processes are replaced by a recorder (no data, no retargeting)."""

import pytest

from dexengine.core import paths as P
from dexengine.data.stage1 import plan as SP
from dexengine.retarget import admit as AD
from dexengine.retarget import reretarget_stage as RT

HAND = "wuji2"


def _fake(monkeypatch, out_of):
    calls = []

    def run(name, argv, log, hand):
        calls.append(name)
        assert hand == HAND
        rc, out = out_of(name, argv)
        log.write_text(out)
        return rc, out

    monkeypatch.setattr(AD, "_run", run)
    return calls


def _ok(ds, label=True):
    def out_of(name, argv):
        return {
            "retarget": (0, f"{RT.SCOPE}{ds} list=[{ds}]\n"),
            "manifest": (0, f"{RT.SCOPE}{ds} {RT.MANIFEST_ONLY}\n"),
            "intake": (0, "[INTAKE_SUPERSEDE] label=x moved 0\n" if label else "ok\n"),
            "unstage": (0, f"{RT.SCOPE}{ds} {RT.UNSTAGE} 3 links removed · 0 left\n"),
        }[name]

    return out_of


def test_full_chain_and_stamps(monkeypatch, data_root, tmp_path):
    calls = _fake(monkeypatch, _ok("hot3d"))
    assert AD.admit("hot3d", HAND, tmp_path, retarget=True, label="pre_x", rig="r" * 64, prior="p" * 64) == 0
    assert calls == ["retarget", "manifest", "intake", "unstage"]
    d = P.retarget_stage_root(HAND) / "processed" / "hot3d"
    assert (d / SP.RIG_STAMP_NAME).read_text() == "r" * 64 + "\n"
    assert (d / SP.PRIOR_STAMP_NAME).read_text() == "p" * 64 + "\n"


@pytest.mark.parametrize(
    "bad, want, ran",
    [
        ("retarget", 8, ["retarget"]),  # rc 0 but the scope marker names another dataset
        ("manifest", 5, ["retarget", "manifest"]),
        ("intake", 9, ["retarget", "manifest", "intake"]),  # a label without the supersede marker
        ("unstage", 5, ["retarget", "manifest", "intake", "unstage"]),
    ],
)
def test_a_missing_marker_stops_the_chain(monkeypatch, data_root, tmp_path, bad, want, ran):
    good = _ok("hot3d")
    calls = _fake(monkeypatch, lambda name, argv: (0, "no marker\n") if name == bad else good(name, argv))
    assert AD.admit("hot3d", HAND, tmp_path, retarget=True, label="pre_x", rig="r", prior="p") == want
    assert calls == ran
    stamped = (P.retarget_stage_root(HAND) / "processed" / "hot3d" / SP.RIG_STAMP_NAME).is_file()
    assert stamped == (bad == "unstage")  # stamps come after intake, before unstage


def test_step_rc_passes_through_and_taco_has_no_stage(monkeypatch, data_root, tmp_path):
    calls = _fake(monkeypatch, lambda name, argv: (4, "count mismatch\n"))
    assert AD.admit("hot3d", HAND, tmp_path, retarget=True) == 4 and calls == ["retarget"]
    assert AD.plan("taco", True) == ["retarget", "stamps"] and AD.plan("hot3d", False)[0] == "manifest"
    calls = _fake(monkeypatch, _ok("taco"))
    assert AD.admit("taco", HAND, tmp_path, retarget=True, rig="r", prior="p") == 0 and calls == ["retarget"]
