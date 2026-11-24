"""Merge top-up (store.merge_topup — only new placement variants for a clip already in the final) on synthetic pieces
(cases U1-U7). Needs pyarrow, ffmpeg and zstd (absent: skipped).

  U1 a listed clip X (need 3, one final variant): only the new piece's X variants not in the final are added, in k
     order (2); one already in the final and two over the target are skipped; a new clip Y of the same piece merges as
     usual; verify passes; pieces line topup · topup_dropped
  U2 no allow table · the clip not in the table → dies as already in the final (safety net unchanged)
  U3 no new variant (all already in the final) → dies
  U4 oracle: a pieces line topup edited to a clip without an earlier piece or to an earlier variant → verify fails
  U5 rules: removing key_clips fails a good final · removing problems lets U4 pass (both are needed)
  U6 final = original fallback row 0 + one variant (names 2, need 4): the piece's fallback row is not a top-up target,
     only the 2 new variants are added
  U7 the same name with another placement → dies · a new name with a final variant's placement → dies
  U8 the verify key rule on copies of the U1 final (rule: (round, source clip) key minus the merge-recorded top-up
     clips · plus the card key (episode clip incl. variant, look) · plus the top-up oracle):
     (a) the same card in two pieces fails (the oracle off — the card key alone catches it; with an episode-unique
         card it passes, so the card key is what failed it)
     (b) the same source clip with different variants in two pieces, one of them on an earlier look name, passes
     (c) a card key without the variant (source clip, look): (a) still fails and (b) fails — the variant is
         load-bearing
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
if not (shutil.which("ffmpeg") and shutil.which("zstd")):
    pytest.skip("ffmpeg and zstd are needed for synthetic pieces", allow_module_level=True)

from dexengine.record import episodes as EPI  # noqa: E402
from dexengine.record.store import merge_check as MC  # noqa: E402
from dexengine.record.store import merge_rows as MR  # noqa: E402
from dexengine.record.store import merge_topup as TU  # noqa: E402
from tests.record.merge_fixture import Src, dies, make_piece, merge, verify  # noqa: E402

X, Y, Z, V = "x_take_pl_rt", "y_take_pl_rt", "z_take_pl_rt", "v_take_pl_rt"
DUP = "already in the final"


def place(nm: str) -> dict:
    k = TU.k_of(nm)
    return {"k": k, "dx": 0.01 * k, "dy": -0.01 * k, "yaw": 5.0 * k}


def v(c: str, k: int) -> str:
    return c.replace("_pl_rt", f"_pv{k:02d}_pl_rt")


def aug_piece(root: Path, src: Src, specs: list[tuple[str, list[str]]], moved: dict | None = None) -> Path:
    """An aug piece — specs = [(source clip, names)] · a name twice = two looks · a name equal to the source clip = the
    original fallback row (no source_clip in selection, as in real pieces) · variant placement = place(k), or moved."""
    ds = make_piece(root, src, [(c, len(ns), f"task {c}") for c, ns in specs], round_="aug")
    eps, i = MC.jsonl(ds / MC.EPISODES), 0
    for c, ns in specs:
        for k, nm in enumerate(ns):
            e = eps[i]
            e["clip"] = nm
            e["selection"] = {"clip": nm} if nm == c else {"source_clip": c, **(moved or {}).get(nm, place(nm))}
            e["look"] = {"name": f"L{k}"}
            e["traj_choice"] = {"row0": None, "chosen": [], "orig_fov": {"rule": EPI.ORIG_FOV_RULE, "row0_source": c}}
            i += 1
    (ds / MC.EPISODES).write_text("".join(json.dumps(e) + "\n" for e in eps))
    return ds


def put_allow(fin: Path, rows: list[tuple[str, str, int]] | None) -> None:
    p = TU.allow_path(fin)
    if rows is None:
        p.unlink(missing_ok=True)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\t".join(TU.ALLOW_HEAD) + "\n" + "".join(f"{d}\t{c}\t{n}\n" for d, c, n in rows))


def dies_dup(fn, *a) -> bool:
    got = dies(fn, *a, want=DUP)
    return got is not None and not got.startswith("<other reason>")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """U1 · U2 · U3 on one final (hot3d) — later cases build on the merged state."""
    tmp = tmp_path_factory.mktemp("merge_topup")
    src = Src(tmp / "_src")
    fin = tmp / "aug" / "data" / "hot3d" / "collect" / "rby1_test"
    merge(fin, aug_piece(tmp / "pA", src, [(X, [v(X, 1), v(X, 1)]), (Z, [v(Z, 2)])]), "pA")
    b = aug_piece(
        tmp / "pB", src, [(X, [v(X, 1), v(X, 1), v(X, 9), v(X, 5), v(X, 5), v(X, 7), v(X, 3)]), (Y, [v(Y, 1), v(Y, 1)])]
    )
    out = {"u2": []}
    for rows in (None, [("taco", X, 3)]):
        put_allow(fin, rows)
        out["u2"].append(dies_dup(merge, fin, b, "pB"))
    put_allow(fin, [("hot3d", X, 3)])
    out["res"] = merge(fin, b, "pB")
    put_allow(fin, [("hot3d", X, 4)])
    out["u3"] = dies_dup(merge, fin, aug_piece(tmp / "pC", src, [(X, [v(X, 3), v(X, 5)])]), "pC")
    put_allow(fin, [("hot3d", X, 3)])
    return dict(tmp=tmp, src=src, fin=fin, **out)


def test_allow_path(tmp_path):
    fin = tmp_path / "aug" / "data" / "hot3d" / "collect" / "rby1_test"
    assert TU.allow_path(fin) == tmp_path / "aug" / TU.ALLOW_DIR / "rby1_test.tsv"
    assert TU.allow_path(tmp_path / "fin") is None


def test_u1_only_new_variants_in_k_order(world):
    fin = world["fin"]
    rec = MC.jsonl(fin / MC.PIECES)[-1]
    assert rec.get("topup") == {X: [v(X, 3), v(X, 5)]}
    assert {(d["clip"], d["why"]) for d in rec["topup_dropped"]} == {
        (v(X, 1), TU.WHY_IN_FINAL),
        (v(X, 7), TU.WHY_OVER.format(have=1, need=3)),
        (v(X, 9), TU.WHY_OVER.format(have=1, need=3)),
    }
    assert {d["rule"] for d in rec["topup_dropped"]} == {TU.AUG_TOPUP_DROP_RULE}
    eps = MC.jsonl(fin / MC.EPISODES)
    got = sorted(e["clip"] for e in eps[rec["episodes"][0] : rec["episodes"][1]])
    assert got == sorted([v(X, 3), v(X, 5), v(X, 5), v(Y, 1), v(Y, 1)])
    assert world["res"].get("episodes") == [3, 8]
    vr = verify(fin)
    assert vr["ok"], vr.get("problems")


def test_u2_no_table_or_clip_not_listed_dies(world):
    assert world["u2"] == [True, True]


def test_u3_no_new_variant_dies(world):
    assert world["u3"]


def test_u4_u5_oracle_and_key_rule(world, monkeypatch):
    fin = world["fin"]
    good = (fin / MC.PIECES).read_text()
    rows = [json.loads(x) for x in good.splitlines()]

    def put(top):
        rows[-1]["topup"] = top
        (fin / MC.PIECES).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    try:
        for bad in ({Y: [v(Y, 1)]}, {X: [v(X, 1)]}):
            put(bad)
            assert not verify(fin, frames=False)["ok"]
        (fin / MC.PIECES).write_text(good)
        assert verify(fin, frames=False)["ok"]
        monkeypatch.setattr(TU, "key_clips", lambda r, clips: list(clips))
        assert not verify(fin, frames=False)["ok"], "removing key_clips must fail a good final"
        monkeypatch.undo()
        monkeypatch.setattr(TU, "problems", lambda pieces, eps, gone: [])
        put({X: [v(X, 1)]})
        assert verify(fin, frames=False)["ok"], "without the oracle the overlap with an earlier variant leaks"
    finally:
        (fin / MC.PIECES).write_text(good)


def test_u6_fallback_row_is_not_a_target(world):
    tmp, src = world["tmp"], world["src"]
    f2 = tmp / "aug" / "data" / "taco" / "collect" / "rby1_test"
    merge(f2, aug_piece(tmp / "qA", src, [(V, [V, v(V, 2)])]), "qA")
    put_allow(world["fin"], [("hot3d", X, 3), ("taco", V, 4)])
    merge(f2, aug_piece(tmp / "qB", src, [(V, [V, v(V, 2), v(V, 8), v(V, 4), v(V, 6)])]), "qB")
    rec = MC.jsonl(f2 / MC.PIECES)[-1]
    assert rec.get("topup") == {V: [v(V, 4), v(V, 6)]}
    assert {(d["clip"], d["why"]) for d in rec["topup_dropped"]} == {
        (V, TU.WHY_FALLBACK),
        (v(V, 2), TU.WHY_IN_FINAL),
        (v(V, 8), TU.WHY_OVER.format(have=2, need=4)),
    }
    assert verify(f2)["ok"]


@pytest.mark.parametrize(
    "names,moved,want",
    [("new", {"v10": "v4"}, "same placement, another name"), ("same", {"v4": "v40"}, "another placement")],
)
def test_u7_placement_is_checked_not_only_the_name(names, moved, want, world):
    tmp, src = world["tmp"], world["src"]
    f2 = tmp / "aug" / "data" / "taco" / "collect" / "rby1_test"
    if not (f2 / MC.PIECES).is_file():
        test_u6_fallback_row_is_not_a_target(world)
    put_allow(world["fin"], [("hot3d", X, 3), ("taco", V, 9)])
    ns = [v(V, 10)] if names == "new" else [v(V, 4), v(V, 12)]
    mv = {v(V, int(a[1:])): place(v(V, int(b[1:]))) for a, b in moved.items()}
    p = aug_piece(tmp / f"qC{names}", src, [(V, ns)], moved=mv)
    got = dies(merge, f2, p, "qC", want=want)
    assert got is not None and not got.startswith("<other reason>"), got


def _final_copy(world, name: str, edit) -> Path:
    """A copy of the U1 final (the module's final stays as it is) with edit(piece-B episode rows, piece-B line)."""
    fin = world["tmp"] / name / "aug" / "data" / "hot3d" / "collect" / "rby1_test"
    shutil.copytree(world["fin"], fin)
    eps, pieces = MC.jsonl(fin / MC.EPISODES), MC.jsonl(fin / MC.PIECES)
    a, b = pieces[-1]["episodes"]
    edit(eps[a:b], pieces[-1])
    pieces[-1]["episode_clips"] = sorted({e["clip"] for e in eps[a:b]})
    for f, rows in ((MC.EPISODES, eps), (MC.PIECES, pieces)):
        (fin / f).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return fin


def _row(rows, clip: str) -> dict:
    return next(e for e in rows if e["clip"] == clip)


def test_u8_card_key_rule(world, monkeypatch):
    def same_card(rows, line):  # piece B's X v3 row becomes piece A's first X v1 row (clip and look)
        e = _row(rows, v(X, 3))
        e["clip"], e["look"] = v(X, 1), {"name": "L0"}
        line["topup"] = {X: [v(X, 1), v(X, 5)]}

    def other_variant_same_look(rows, line):  # piece B's X v3 row takes the look name of piece A's X v1 row
        _row(rows, v(X, 3))["look"] = {"name": "L0"}

    fa = _final_copy(world, "u8a", same_card)
    fb = _final_copy(world, "u8b", other_variant_same_look)
    assert verify(fb, frames=False)["ok"], "(b) the same source clip with another variant must pass"
    monkeypatch.setattr(TU, "problems", lambda pieces, eps, gone: [])
    assert not verify(fa, frames=False)["ok"], "(a) the same card in two pieces must fail"
    with monkeypatch.context() as m:
        m.setattr(MC, "card", lambda e: (e["clip"], e["look"]["name"], e["episode_index"]))
        assert verify(fa, frames=False)["ok"], "control: without the card key (a) passes — the card key failed it"
    with monkeypatch.context() as m:
        m.setattr(MC, "card", lambda e: (MR.src_clip(e), e["look"]["name"]))
        assert not verify(fa, frames=False)["ok"], "(c) without the variant (a) must still fail"
        assert not verify(fb, frames=False)["ok"], "(c) without the variant (b) fails — the variant is load-bearing"
