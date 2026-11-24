"""The base-tree repair chain (wrist_shift.CHAIN: 2e pre_ws → 2f pre_hc) on one npz file (CPU): a later repair on top
is neither stacked on by an earlier repair's re-run nor taken for an upstream change; a real change of the earlier
output (or of the file upstream) re-backs-up; reverting the earlier repair reverts the later one first."""

from __future__ import annotations

import numpy as np

from dexengine.placement.repair import wrist_desk_fresh as WDF
from dexengine.placement.repair import wrist_shift as WS

WSD, HC = WS.WRIST_DESK_TAG, WS.HAND_CLEAR_TAG


def _q(path):
    with np.load(path) as z:
        return np.asarray(z["qpos"])


def _write(cur, tag, q):
    """One repair's write as the stages do it: read its input through source_of, write, record the post sha."""
    with np.load(WS.source_of(cur, tag)) as z:
        out = {k: z[k] for k in z.files}
    out["qpos"] = np.asarray(q, float)
    WS.save_npz(cur, out, tag)


def _tree(tmp_path):
    (tmp_path / "0").mkdir()
    cur = tmp_path / "0" / "trajectory_kinematic.npz"
    np.savez(cur, qpos=np.zeros((4, 3)))
    return cur


def test_chain_order_is_2e_then_2f():
    assert WS.CHAIN == (WSD, HC) == ("pre_ws", "pre_hc")


def test_an_earlier_repair_reads_its_own_output_under_a_later_one(tmp_path, capsys):
    cur = _tree(tmp_path)
    _write(cur, WSD, np.ones((4, 3)))  # 2e
    _write(cur, HC, np.full((4, 3), 2.0))  # 2f on top of 2e
    assert WS.view(cur, WSD) == cur.with_suffix(f".{HC}.npz")  # the file as 2e left it
    assert (_q(WS.view(cur, WSD)) == 1).all()
    assert WS.original(cur, WSD) == cur.with_suffix(f".{WSD}.npz") and (_q(WS.original(cur, WSD)) == 0).all()
    assert WS.wrote(cur, WSD) and WS.wrote(cur, HC)
    assert (_q(WS.original(cur, HC)) == 1).all()  # 2f's input = 2e's output
    _write(cur, WSD, np.ones((4, 3)))  # 2e re-run, same result: reads its backup, no upstream-change refresh
    assert "upstream change" not in capsys.readouterr().out
    assert WDF.qpos_sha(cur.parent.parent, WSD) == WDF.qpos_sha_file(cur.with_suffix(f".{HC}.npz"))


def test_a_new_earlier_output_makes_the_later_repair_back_up_again(tmp_path, capsys):
    cur = _tree(tmp_path)
    _write(cur, WSD, np.ones((4, 3)))
    _write(cur, HC, np.full((4, 3), 2.0))
    _write(cur, WSD, np.full((4, 3), 3.0))  # 2e changes its answer: 2f's output is gone, its record no longer matches
    assert not WS.wrote(cur, HC) and WS.view(cur, WSD) == cur
    assert (_q(WS.original(cur, HC)) == 3).all()
    _write(cur, HC, np.full((4, 3), 4.0))
    assert "upstream change detected" in capsys.readouterr().out
    assert (_q(cur.with_suffix(f".{HC}.npz")) == 3).all() and (_q(WS.original(cur, WSD)) == 0).all()


def test_reverting_the_earlier_repair_reverts_the_later_one_first(tmp_path):
    cur = _tree(tmp_path)
    _write(cur, WSD, np.ones((4, 3)))
    _write(cur, HC, np.full((4, 3), 2.0))
    assert WS.revert(cur, WSD)
    assert (_q(cur) == 0).all()
    assert not any(cur.with_suffix(f".{t}.{e}").exists() for t in WS.CHAIN for e in ("npz", "meta.json"))
    assert not WS.revert(cur, WSD) and not WS.revert(cur, HC)


def test_an_upstream_change_under_both_is_a_new_original(tmp_path, capsys):
    cur = _tree(tmp_path)
    _write(cur, WSD, np.ones((4, 3)))
    _write(cur, HC, np.full((4, 3), 2.0))
    np.savez(cur, qpos=np.full((4, 3), 9.0))  # the base layer rewrote the file
    assert WS.view(cur, WSD) == cur and WS.original(cur, WSD) == cur
    _write(cur, WSD, np.ones((4, 3)))
    assert "upstream change detected" in capsys.readouterr().out
    assert (_q(cur.with_suffix(f".{WSD}.npz")) == 9).all()


def test_a_tag_outside_the_chain_peels_nothing(tmp_path):
    cur = _tree(tmp_path)
    _write(cur, HC, np.full((4, 3), 2.0))
    assert WS.view(cur, "pre_xr") == cur  # negative: only the chain's later repairs are peeled
    assert WS.view(cur, HC) == cur  # nothing after 2f
