"""Pure helpers of the judge: frame dedup, placement keys, the search policy and the per-clip markers (CPU only)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from dexengine.placement import search_policy as SP  # noqa: E402
from dexengine.placement.judge import clip_time as CT  # noqa: E402
from dexengine.placement.judge import common as JC  # noqa: E402
from dexengine.placement.judge.exhaust import reject_tally  # noqa: E402


def _greedy(q, clr, bnd, dd):
    """Per-frame greedy dedup: keep a candidate unless it is within dd (Chebyshev) of an earlier kept one."""
    cand, off, clrs = [], [], []
    for f in range(len(bnd) - 1):
        qq, cc = q[bnd[f] : bnd[f + 1]], clr[bnd[f] : bnd[f + 1]]
        keep = []
        for i in range(len(qq)):
            if all(np.abs(qq[i] - qq[k]).max() > dd for k in keep):
                keep.append(i)
        off.append((len(cand), len(keep)))
        cand.extend(qq[keep])
        clrs.extend(cc[keep])
    return np.asarray(cand), off, np.asarray(clrs)


def test_dedup_matches_greedy_reference():
    rng = np.random.default_rng(0)
    cnt = np.array([0, 5, 17, 1, 9])
    bnd = np.concatenate([[0], np.cumsum(cnt)])
    q = rng.uniform(-0.4, 0.4, (int(bnd[-1]), 7))
    clr = rng.uniform(0, 1, int(bnd[-1]))
    got = JC._dedup_frames_t(q, clr, bnd, len(cnt), JC.DD12)
    ref = _greedy(q, clr, bnd, JC.DD12)
    assert got[1] == ref[1]
    np.testing.assert_array_equal(got[0], ref[0])
    np.testing.assert_array_equal(got[2], ref[2])


def test_place_key_is_pivot_free():
    # the same rigid motion written about two pivots: q = Rz(p − c) + c + d
    yaw, c1, d1 = 30.0, np.array([0.1, -0.2]), np.array([0.05, 0.02])
    th = np.deg2rad(yaw)
    Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c2 = np.array([-0.3, 0.4])
    d2 = d1 + c1 - Rz @ c1 - (c2 - Rz @ c2)
    assert JC._place_key((*d1, yaw, c1)) == JC._place_key((*d2, yaw, c2))
    assert JC._place_key((*d1, yaw, c1)) != JC._place_key((*d1, yaw + 1.0, c1))


def test_search_policy():
    assert SP.dz_cells_for("taco") == SP.TACO_CANON == "-5.5"
    assert SP.dz_cells_for("taco", "aerial") == SP.AERIAL
    with pytest.raises(SystemExit):
        SP.dz_cells_for("not_a_dataset")
    assert SP.target_for("taco") == "" and SP.target_for("hrdexdb") == "-0.33,-0.15"
    assert SP.norm("-0.18,0.0") == SP.norm("-0.18,0") == "-0.18,0"
    with pytest.raises(SystemExit):
        SP.target_for("hot3d-aerial")
    assert SP.resolve("taco", "table", "", env={}) == ("", "policy")
    with pytest.raises(SystemExit):
        SP.resolve("taco", "table", "0.1,0.2", env={})


def test_clip_time_marker_round_trip():
    CT.start()
    line = CT.line("clip one", 12.5, adopted=True)
    d = CT.parse(line)
    assert d["clip"] == "clip one" and d["total"] == 12.5 and d["result"] == "ADOPTED"
    assert set(d["stages"]) == set(CT.SEQ + CT.INNER)


def test_reject_tally_format():
    assert reject_tally({}) == " · reject_tally -"
    assert (
        reject_tally({"confirm/error": 2, "sparse/coverage x": 5})
        == " · reject_tally sparse/coverage_x=5 confirm/error=2"
    )
