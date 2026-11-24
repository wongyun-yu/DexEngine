"""J-self-pen in the final merge (store.merge_self_pen) on synthetic pieces with a stand-in measurement
(tests.record.merge_fixture.merge: planted {episode position: depth mm}). Needs pyarrow, ffmpeg and zstd.

  a trajectory over the line leaves with all its looks, the other trajectories merge, the pieces line records it and
  verify passes · 15 mm and none planted keep every episode, and the final equals a merge without the stage · a piece
  whose every trajectory is over still gets its pieces line (no episodes) and verify passes; a later piece chains on
  it · a piece changed after it was measured dies with nothing written · the measurement takes no lock · each distinct
  joint table is measured once · verify refuses an edited drop record.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
if not (shutil.which("ffmpeg") and shutil.which("zstd")):
    pytest.skip("ffmpeg and zstd are needed for synthetic pieces", allow_module_level=True)

from dexengine.record.store import merge as CMG  # noqa: E402
from dexengine.record.store import merge_check as MC  # noqa: E402
from dexengine.record.store import merge_self_pen as SPM  # noqa: E402
from tests.record.merge_fixture import Src, dies, make_piece, measure, merge, verify  # noqa: E402

X, Y = "x_take_pl_rt", "y_take_pl_rt"
PV1 = X.replace("_pl_rt", "_pv01_pl_rt")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return Src(tmp_path_factory.mktemp("self_pen_src"))


def _piece(root: Path, src: Src) -> Path:
    """Episodes 0 X (look D) · 1 X_pv01 · 2 X (look r2) · 3 Y (look D): trajectories X (0, 2) · X_pv01 (1) · Y (3)."""
    return make_piece(root, src, [(X, 3, "task x"), (Y, 1, "task y")])


def _final(tmp: Path) -> Path:
    return tmp / "orig" / "data" / "hot3d" / "collect" / "rby1_test"


def test_a_trajectory_over_the_line_leaves_with_all_its_looks(tmp_path, src, capsys):
    fin = _final(tmp_path)
    out = merge(fin, _piece(tmp_path / "p", src), "p1", planted={2: 25.0, 1: 15.0})
    eps = MC.jsonl(fin / MC.EPISODES)
    assert [e["clip"] for e in eps] == [PV1, Y] and out["episodes"] == [0, 2]
    rec = MC.jsonl(fin / MC.PIECES)[-1]
    assert [(d["clip"], d["piece_eps"], d["rule"]) for d in rec["self_pen_dropped"]] == [(X, [0, 2], SPM.RULE)]
    assert rec["self_pen_dropped"][0]["statement"]["verdict"] == "drop" and rec["clips"] == [X, Y]
    assert verify(fin, frames=False)["ok"]
    log = capsys.readouterr().out
    assert f"[SELF_PEN] {X} max 25.00 pair a|b frame 0 -> drop" in log
    assert f"[SELF_PEN] {PV1} max 15.00 pair a|b frame 0 -> keep" in log
    assert "[SELF_PEN_SUM] p1 — trajectories 3 · dropped 1 · episodes kept 2 / 4" in log


def test_a_clean_piece_merges_as_without_the_stage(tmp_path, src, monkeypatch):
    piece = _piece(tmp_path / "p", src)
    merge(fa := _final(tmp_path / "a"), piece, "p1", planted={0: 15.0})
    monkeypatch.setattr(SPM, "plan", lambda pc, pid, got: (pc, []))
    merge(fb := _final(tmp_path / "b"), piece, "p1")
    for m in (MC.EPISODES, MC.INFO, MC.TASKS, MC.STATS):
        assert (fa / m).read_bytes() == (fb / m).read_bytes()
    ra, rb = (json.loads((f / MC.PIECES).read_text()) for f in (fa, fb))
    assert {k: v for k, v in ra.items() if k != "merged_at"} == {k: v for k, v in rb.items() if k != "merged_at"}
    assert "self_pen_dropped" not in ra
    for e in range(4):
        p = f"data/chunk-000/episode_{e:06d}.parquet"
        assert (fa / p).read_bytes() == (fb / p).read_bytes()


def test_a_piece_with_every_trajectory_over_still_gets_its_line(tmp_path, src):
    fin = _final(tmp_path)
    out = merge(fin, _piece(tmp_path / "p", src), "p1", planted=dict.fromkeys(range(4), 30.0))
    assert out["episodes"] == [0, 0] and out["self_pen_dropped"] == 3
    rec = MC.jsonl(fin / MC.PIECES)
    assert [(r["piece_id"], r["episodes"], len(r["self_pen_dropped"])) for r in rec] == [("p1", [0, 0], 3)]
    assert MC.jsonl(fin / MC.EPISODES) == [] and verify(fin, frames=False)["ok"]
    merge(fin, make_piece(tmp_path / "q", src, [("z_take_pl_rt", 1, "task z")]), "p2")
    assert [r["episodes"] for r in MC.jsonl(fin / MC.PIECES)] == [[0, 0], [0, 1]] and verify(fin, frames=False)["ok"]


def test_a_piece_changed_after_it_was_measured_dies(tmp_path, src):
    piece = _piece(tmp_path / "p", src)
    measured = measure(piece)
    import pyarrow.parquet as pq

    f = piece / "data" / "chunk-000" / "episode_000001.parquet"
    t = pq.read_table(f)
    pq.write_table(t.set_column(0, t.schema.field(0), t.column(0)), f, compression="gzip")  # same rows, other bytes
    from dexengine.record import piece as CP

    fin = _final(tmp_path)
    with CP.final_lock(fin):
        msg = dies(CMG.merge_piece, fin, piece, "p1", measured, want="changed after it was measured")
    assert msg and not msg.startswith("<other reason>") and not (fin / MC.EPISODES).exists()


def test_the_measurement_takes_no_lock(tmp_path, src, monkeypatch):
    """measure_piece runs the rulers outside piece.final_lock (merge_piece refuses to run without it): J-self-pen and
    the S-dyn-pairs output check (store.merge_dyn_pairs)."""
    monkeypatch.setattr(MC, "require_lock", lambda final: (_ for _ in ()).throw(AssertionError("lock")))
    assert set(measure(_piece(tmp_path / "p", src))) == {"piece", "fp", "trajectories", "dyn_pairs"}


def test_verify_refuses_an_edited_drop_record(tmp_path, src):
    fin = _final(tmp_path)
    merge(fin, _piece(tmp_path / "p", src), "p1", planted={1: 25.0})
    rows = MC.jsonl(fin / MC.PIECES)
    rows[0]["self_pen_dropped"][0]["clip"] = X  # a clip that is in the final
    (fin / MC.PIECES).write_text("".join(json.dumps(r) + "\n" for r in rows))
    bad = verify(fin, frames=False)["problems"]
    assert any(f"self_pen_dropped clip {X} is in the final" in p for p in bad)


def test_each_distinct_joint_table_is_measured_once(tmp_path, src, monkeypatch):
    calls = []
    real = SPM.SP.read_joints
    monkeypatch.setattr(SPM.SP, "read_joints", lambda p: real(p) * 0)  # every look carries the same table
    monkeypatch.setattr(
        SPM.SP, "measure_joints", lambda combo, names, Q: calls.append(1) or SPM.SP.Result(0.0, "", "", "", -1, len(Q))
    )
    got = SPM.measure(CMG.read_piece(_piece(tmp_path / "p", src)))
    assert len(calls) == 1 and list(got) == [X, PV1, Y] and got[X][1] == [0, 2]
