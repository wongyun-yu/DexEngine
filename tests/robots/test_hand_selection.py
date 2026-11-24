"""A run needs only its own hand's vendor files: `vendor build` reads the selected hands' sources and the shared ones
(vendor.required_sources), and the every-hand code reads the hands whose source is set and whose merged URDF is built
(sim_collider.arm_hands); a run whose own hand is not built still dies where it reads that hand's URDF."""

import pytest

from dexengine.core import hands as HR
from dexengine.placement.collision import sim_collider as SC
from dexengine.robots import vendor


def test_build_reads_the_selected_hand_and_the_shared_sources():
    own = vendor.hand_sources()
    shared = set(vendor.sources()) - set().union(*own.values())
    assert shared and shared.isdisjoint(set(HR.HAND_REGISTRY))  # the arm and the camera, no hand's source
    for hand, mine in own.items():
        need = vendor.required_sources({hand})
        others = set().union(*(s for h, s in own.items() if h != hand)) - mine
        assert need == shared | mine and need.isdisjoint(others), hand
    with pytest.raises(SystemExit, match="unknown hand"):
        vendor.required_sources({"no_such_hand"})


def test_check_limited_to_sources(tmp_path, monkeypatch):
    vd = tmp_path / "vendor"
    vd.mkdir()
    (vd / "sources.yaml").write_text("a:\n  kind: repository\n  root: x\nb:\n  kind: repository\n  root: x\n")
    z = "0" * 64
    (vd / "files.tsv").write_text(f"a\tout/a.bin\ta.bin\t{z}\nb\tout/b.bin\tb.bin\t{z}\n")
    monkeypatch.setattr(vendor, "dest_path", lambda r: tmp_path / r.dest)
    assert [p.split()[-1] for p in vendor.check(vendor_dir=vd)] == ["out/a.bin", "out/b.bin"]
    assert [p.split()[-1] for p in vendor.check(vendor_dir=vd, only={"b"})] == ["out/b.bin"]


def _built_only(monkeypatch, tmp_path, built: set[str]):
    def combo(arm, hand, body_variant=None):
        f = tmp_path / f"{arm}_{hand}.urdf"
        if hand in built:
            f.write_text("<robot/>")
        return f

    monkeypatch.setattr(SC, "combo_urdf", combo)
    monkeypatch.setattr(vendor, "unavailable_hands", lambda vendor_dir=None: {})
    SC._HANDS_SAID.discard("rby1")


def test_an_unbuilt_hand_is_left_out_and_named(monkeypatch, tmp_path, capsys):
    reg = SC.registered_hands("rby1")
    assert len(reg) >= 2
    keep, drop = reg[0], reg[1]
    _built_only(monkeypatch, tmp_path, set(reg) - {drop})
    sel = SC.arm_hands("rby1")
    assert keep in sel and drop not in sel
    assert f"not selected {drop}: NOT BUILT" in capsys.readouterr().out


def test_the_runs_own_unbuilt_hand_dies(monkeypatch, tmp_path):
    """Negative: leaving an unbuilt hand out of the every-hand set does not let a run of that hand pass — the
    resolver of the run's URDF dies."""
    hand = SC.registered_hands("rby1")[0]
    _built_only(monkeypatch, tmp_path, set())
    monkeypatch.delenv("DEXENGINE_XR_URDF", raising=False)
    assert SC.arm_hands("rby1") == []
    read = getattr(SC, "resolved_urdf", None) or (lambda arm, h: SC.digest(arm, [SC.combo_urdf(arm, h)]))
    with pytest.raises(BaseException) as e:  # noqa: PT011,B017 — the reader's own error type; the message is checked
        read("rby1", hand)
    assert f"rby1_{hand}.urdf" in str(e.value)
