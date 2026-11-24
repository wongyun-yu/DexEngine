"""Zero-touched-samples gate of the record oracle (rollout.oracle_gate):
a manipulated object the human touched passes as ORACLE_NONTIP only when the
independent witness (hand_object_contact view) is 0 at the fingertip link columns; fingertip force with sensors at 0
still dies as a wiring fault; unreadable columns or contact die. Mutants of the gate each flip a case."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch

from dexengine.record.rollout import oracle_gate as REC

ns = types.SimpleNamespace
CH = "obj/box"
LINKS = ["right_tip", "right_palm", "left_tip", "left_palm"]


def _sidecars(tree: Path, frames: dict | None) -> None:
    """GT contact sidecars tree/0/contact_mano_<side>.npz with `frames[side]` contact frames (None = no files)."""
    (tree / "0").mkdir(parents=True, exist_ok=True)
    if frames is None:
        return
    for side, n in frames.items():
        c = np.zeros((10, 5), bool)
        c[:n, 0] = True
        np.savez(
            tree / "0" / f"contact_mano_{side}.npz",
            contact=c,
            palm_effective_dist=np.ones(10),
            contact_threshold_m=np.float64(0.01),
        )


def _rec(tmp: Path, frames: dict | None) -> ns:
    """A recorder stand-in with one partner-object channel; the right fingertip is its oracle pair (column 0)."""
    _sidecars(tmp / "c0", frames)
    return ns(
        _oracle=[("right_thumb_obj_s", None, CH, LINKS.index("right_tip"))],
        _chs=[{"name": CH, "names": list(LINKS), "sides": ["right", "left"]}],
        meta_in={},
        clip_names=["c0"],
        traj_mgr=ns(cfg=ns(data_dir=str(tmp))),
    )


def _run(mod, tmp: Path, name: str, wit: list, frames: dict | None) -> str:
    """Gate name, or die:wiring (fingertip force named) / die:other."""
    r = _rec(tmp / name, frames)
    try:
        return mod.oracle_gate(r, {CH: torch.tensor(wit)})
    except (SystemExit, RuntimeError, ValueError) as e:
        return "die:" + ("wiring" if "wiring fault" in str(e) and "right_tip" in str(e) else "other")


PALM = [0.0, 2.5, 1.0, 0.5]  # load on non-fingertip links (and the left tip, which no oracle pair covers)
TIP = [0.7, 2.5, 1.0, 0.5]  # the covered fingertip carries force too
CASES = {
    "bump": (PALM, {"right": 0, "left": 0}, REC.ORACLE_BUMP),
    "palm": (PALM, {"right": 0, "left": 3}, REC.ORACLE_NONTIP),
    "nocar": (PALM, None, "die:other"),
    "tip": (TIP, {"right": 0, "left": 3}, "die:wiring"),
    "zero": ([0.0, 0.0, 0.0, 0.0], {"right": 0, "left": 3}, REC.ORACLE_UNTOUCHED),
}


def test_bump_record_unchanged(tmp_path):
    r = _rec(tmp_path, {"right": 0, "left": 0})
    assert REC.oracle_gate(r, {CH: torch.tensor(PALM)}) == REC.ORACLE_BUMP
    g = r.meta_in["collect"]["oracle_gate"]
    assert g == {"rule": REC.ORACLE_BUMP, "channels": {CH: {"max_n": 2.5, "links": 3, "human_frames": 0}}}


def test_nontip_passes_with_tip_and_load_links(tmp_path, capsys):
    r = _rec(tmp_path, {"right": 0, "left": 3})
    assert REC.oracle_gate(r, {CH: torch.tensor(PALM)}) == REC.ORACLE_NONTIP
    c = r.meta_in["collect"]["oracle_gate"]["channels"][CH]
    assert c["human_frames"] == 3 and c["tip_links"] == ["right_tip"] and c["links"] == 3
    assert c["load_links"] == [["right_palm", 2.5], ["left_tip", 1.0], ["left_palm", 0.5]]
    out = capsys.readouterr().out
    assert REC.ORACLE_NONTIP in out and "right_palm 2.5 N" in out


@pytest.mark.parametrize("name", sorted(CASES))
def test_cases(tmp_path, name):
    wit, frames, want = CASES[name]
    assert _run(REC, tmp_path, name, wit, frames) == want


def test_fingertip_force_dies_naming_the_link(tmp_path):
    r = _rec(tmp_path, {"right": 0, "left": 3})
    with pytest.raises(RuntimeError) as e:
        REC.oracle_gate(r, {CH: torch.tensor(TIP)})
    msg = str(e.value)
    assert msg.startswith(f"{REC.MARK} oracle saw no touched sample — the object force is unproven")
    assert "'right_tip': 0.7" in msg and "wiring fault" in msg


@pytest.mark.parametrize(
    "tips,names,wit",
    [
        ([], ["t", "p"], [0.0, 2.0]),  # no fingertip column
        (None, ["t", "p"], [0.0, 2.0]),  # no oracle pair for the channel
        ([0], ["t"], [0.0, 2.0]),  # names do not match the links
        ([5], ["t", "p"], [0.0, 2.0]),  # column out of range
    ],
)
def test_unreadable_columns_die(tips, names, wit):
    with pytest.raises(RuntimeError, match="cannot read the fingertip link columns"):
        REC.oracle_verdict({"c": wit}, lambda _c: 1, {"c": tips}, {"c": names})


def test_empty_witness_is_not_untouched():
    assert REC.untouched_ok({}) is False


NAN = float("nan")
NAN_CASES = [{"c": [0.0, NAN]}, {"c": [NAN, 0.0]}, {"c": [0.0, 0.0], "d": [0.0, NAN]}]


def _nan_dies(mod, wit) -> bool:
    try:
        mod.oracle_verdict(wit, lambda _c: 0, {}, {})
    except RuntimeError as e:
        return "NaN in the oracle witness" in str(e)
    return False


@pytest.mark.parametrize("wit", NAN_CASES, ids=["not-first", "first", "other-channel"])
def test_nan_witness_dies_before_every_gate(wit):
    """An untouched channel with NaN (not first, first, another channel) dies: no UNTOUCHED or BUMP pass."""
    assert _nan_dies(REC, wit)


def test_nan_in_a_touched_channel_dies_too():
    with pytest.raises(RuntimeError, match="NaN in the oracle witness"):
        REC.oracle_verdict({"c": [0.0, 2.0, NAN]}, lambda _c: 1, {"c": [0]}, {"c": ["t", "p", "q"]})


@pytest.mark.parametrize(
    "old,new",
    [
        ("    return bool(wit) and all(v == 0.0 for v in wit.values())", "    return True"),  # witness check gone
        ("        if frames != 0:", "        if True:"),  # BUMP gate gone
        ("        if frames != 0:", "        if False:"),  # human check gone
        ("    if hot:", "    if False:"),  # fingertip column check gone
        ("    if nan:  #", "    if False:  #"),  # witness NaN check gone
    ],
)
def test_mutants_flip_a_case(tmp_path, old, new):
    src = Path(REC.__file__).read_text()
    assert src.count(old) == 1
    mod = types.ModuleType("oracle_gate_mutant")
    mod.__file__ = REC.__file__
    exec(compile(src.replace(old, new, 1), REC.__file__, "exec"), mod.__dict__)
    got = {k: _run(mod, tmp_path / "m", k, w, f) for k, (w, f, _want) in CASES.items()}
    got["nan"] = all(_nan_dies(mod, w) for w in NAN_CASES)
    assert got != {**{k: want for k, (_w, _f, want) in CASES.items()}, "nan": True}
