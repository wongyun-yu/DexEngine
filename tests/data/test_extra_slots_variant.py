"""A placement variant takes its source clip's extra-slot classification (data.objects.extra_slots_variant) and the
checkpoint ledger comparison folds variants to their source (rl.env_setup.extra_slots_ledger).
The data-backed cases (the real xhand_pro x taco knife-plate clip and its
yaw +32.5 variant, S1/S3/S4) need a data root; here the classification body is faked so the routing, the placement
rule, the input contract, the ledger fold and the structure are checked on CPU alone."""

from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data.objects import extra_slots as XS
from dexengine.data.objects import extra_slots_variant as XV
from dexengine.rl.env_setup import extra_slots_ledger as EL

REPO = Path(__file__).resolve().parents[2]
CLIP = "knife_plate__20231105_001"
SRC = P.tree_name(CLIP)
VAR = P.tree_name(P.variant_name(CLIP, 32))


def _ti(**kw) -> dict:
    return {"dataset_name": "taco", "objects": [], **kw}


def _layer(root: Path, var_ti: dict | None = None, src: bool = True) -> tuple[Path, Path]:
    """root/<source tree> (unless src=False) and root/<variant tree> with an aug_place record naming the source."""
    s, v = root / SRC, root / VAR
    if src:
        s.mkdir(parents=True)
        (s / "task_info.json").write_text(json.dumps(_ti()))
    v.mkdir(parents=True)
    (v / "task_info.json").write_text(json.dumps(var_ti if var_ti is not None else _ti(aug_place={"src_tree": SRC})))
    return s, v


def _row(name, cls, why, kind=None) -> dict:
    return {"name": name, "cls": cls, "why": why, "kind": kind, "static": cls == XS.STATIC_SPAWN}


# source: a = touched static spawn · b = dropped by the placement rule at the source place · c = visual only
FAKE_SRC = [
    _row("a", XS.STATIC_SPAWN, "touched", XS.KIND_TOUCHED),
    _row("b", XS.STATIC, XS.ROBOT_KEEPOUT),
    _row("c", XS.VISUAL_ONLY, "still"),
]
# the variant measured on its own geometry: a dropped at the variant place · b stands · c a receiver (the yaw defect)
FAKE_OWN = [
    _row("a", XS.STATIC, XS.ROBOT_KEEPOUT),
    _row("b", XS.STATIC_SPAWN, "touched", XS.KIND_TOUCHED),
    _row("c", XS.STATIC_SPAWN, XS.RECEIVER, XS.KIND_RECEIVER),
]


@pytest.fixture
def fake_body(monkeypatch):
    """extra_slots._own_rows faked per tree name (source -> FAKE_SRC, variant -> FAKE_OWN); fresh marker state."""
    monkeypatch.setattr(XV, "_SAID", set())

    def own(tree_dir, ti, q, strict, kinds_only=False):
        return [dict(r) for r in (FAKE_OWN if Path(tree_dir).name == VAR else FAKE_SRC)]

    monkeypatch.setattr(XS, "_own_rows", own)


# ---- names ----------------------------------------------------------------------------------------------------------
def test_names_fold_to_the_source_clip():
    assert XV.source_clip(VAR) == CLIP and XV.source_clip(SRC) == CLIP and XV.source_clip(CLIP) == CLIP
    assert XV.is_variant(VAR) and not XV.is_variant(SRC)


# ---- routing and the placement rule ---------------------------------------------------------------------------------
def test_variant_takes_the_source_classification(tmp_path, fake_body, capsys):
    s, v = _layer(tmp_path)
    got = {r["name"]: (r["cls"], r["why"]) for r in XS.clip_rows(str(v))}
    # a: dropped at the variant place -> dropped · b: dropped only at the source place -> stands in the variant ·
    # c: the receiving answer split with yaw -> the source's (visual only)
    assert got == {
        "a": (XS.STATIC, XS.ROBOT_KEEPOUT),
        "b": (XS.STATIC_SPAWN, "touched"),
        "c": (XS.VISUAL_ONLY, "still"),
    }
    out = capsys.readouterr().out
    assert out.count(XV.MARK) == 1 and "c own STATIC_SPAWN:receiver:receiver -> used VISUAL_ONLY:still:-" in out
    XS.clip_rows(str(v))
    assert XV.MARK not in capsys.readouterr().out  # once per tree
    assert [r["cls"] for r in XS.clip_rows(str(s))] == [r["cls"] for r in FAKE_SRC]  # a source is unchanged


def test_unfaked_variant_without_extras_equals_its_source(tmp_path):
    s, v = _layer(tmp_path)
    assert XS.clip_rows(str(v)) == XS.clip_rows(str(s)) == []


def test_train_scene_clip_now_of_a_variant_equals_its_source(tmp_path, monkeypatch):
    """train_scene.clip_now is what the ledger-less checkpoint branch (_scene_without_ledger) compares: the variant
    reads its source's slots. Negative: with the routing off the yaw-split receiver stands in the variant."""
    from dexengine.rl.launch import train_scene as TS

    slot = {"obj_id": "o", "qpos_start": 7}
    src_rows = [
        dict(_row("a", XS.STATIC_SPAWN, "touched", XS.KIND_TOUCHED), **slot),
        _row("c", XS.VISUAL_ONLY, "still"),
    ]
    own_rows = [src_rows[0], dict(_row("c", XS.STATIC_SPAWN, XS.RECEIVER, XS.KIND_RECEIVER), obj_id="p", qpos_start=14)]
    monkeypatch.setattr(XV, "_SAID", set())
    monkeypatch.setattr(
        XS, "_own_rows", lambda t, *a, **k: [dict(r) for r in (own_rows if Path(t).name == VAR else src_rows)]
    )
    s, v = _layer(tmp_path)
    assert TS.clip_now(str(v)) == TS.clip_now(str(s)) and TS.clip_now(str(s))[0] == [XS.KIND_TOUCHED]
    monkeypatch.setattr(XV, "source_dir", lambda *_a: None)
    assert TS.clip_now(str(v))[0] == [XS.KIND_TOUCHED, XS.KIND_RECEIVER]


# ---- input contract -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "var_ti,src,match",
    [
        (_ti(), True, "aug_place.src_tree None"),  # variant name without an aug_place record
        (_ti(aug_place={"src_tree": "x_pl_rt"}), True, "!= aug_place.src_tree 'x_pl_rt'"),
        (None, False, "no source tree"),  # no sibling source
    ],
)
def test_variant_contract_dies(tmp_path, var_ti, src, match):
    _s, v = _layer(tmp_path, var_ti, src)
    with pytest.raises(ValueError, match=match.replace("(", r"\(").replace(")", r"\)")):
        XS.clip_rows(str(v))


def test_aug_place_on_a_non_variant_dies(tmp_path):
    s, _v = _layer(tmp_path)
    with pytest.raises(ValueError, match="not a variant name but an aug_place record"):
        XS.clip_rows(str(s), _ti(aug_place={"src_tree": SRC}))


def test_variant_of_a_variant_dies(tmp_path):
    d = tmp_path / P.tree_name(P.variant_name(P.variant_name(CLIP, 3), 4))
    with pytest.raises(ValueError, match="is a variant too"):
        XV.source_dir(d, _ti(aug_place={"src_tree": "x"}))


def test_different_extra_objects_die(tmp_path, monkeypatch):
    monkeypatch.setattr(XS, "_own_rows", lambda t, *a, **k: [_row("zz", XS.VISUAL_ONLY, "still")])
    s, v = _layer(tmp_path)
    monkeypatch.setattr(XS, "clip_rows", lambda *a, **k: [_row("a", XS.VISUAL_ONLY, "still")])
    with pytest.raises(ValueError, match="extra objects"):
        XV.rows(str(v), _ti(aug_place={"src_tree": SRC}), None, True, str(s), XS._own_rows)


# ---- ledger fold ----------------------------------------------------------------------------------------------------
SIG = EL.extra_slot_signature([])
OTHER = "0/0/0:ffffffffffff"


def _cmp(tmp_path, clips: dict) -> tuple[SystemExit | None, str]:
    """(the SystemExit the comparison raised | None, its stdout)."""
    led = tmp_path / EL.EXTRA_SLOTS_LEDGER
    led.write_text(f"1\nrule\t{XS.RULE}\nclip\t{SRC}\t{SIG}\n")
    st = {"rule": XS.RULE, "sensitive": {}, "collide": {}, "robot_kinds": [], "drops": {}, "clips": clips}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            EL._ledger_compare(str(led), st, "test")
    except SystemExit as e:
        return e, buf.getvalue()
    return None, buf.getvalue()


def test_ledger_compares_a_variant_with_its_source(tmp_path):
    bad, out = _cmp(tmp_path, {VAR: SIG})
    assert bad is None and "same for 1 clips" in out and "placement variants 1 names -> source clips 1 folded" in out


def test_ledger_variant_with_another_signature_dies(tmp_path):
    bad, _out = _cmp(tmp_path, {VAR: OTHER})
    assert bad is not None and "differ from training" in str(bad)


def test_ledger_old_hole_without_the_fold(tmp_path, monkeypatch):
    """Negative: without the fold a variant with another signature passes as a clip missing from the ledger."""
    monkeypatch.setattr(XV, "source_clip", P.clip_name)
    monkeypatch.setattr(XV, "is_variant", lambda _n: False)
    bad, out = _cmp(tmp_path, {VAR: OTHER})
    assert bad is None and "clips missing from the ledger 1 (not compared)" in out


def test_ledger_variant_without_a_source_row_dies(tmp_path):
    other = P.tree_name(P.variant_name("dust_brush_box__20231005_201", 3))
    bad, _out = _cmp(tmp_path, {other: SIG})
    assert isinstance(bad, InputMissingError) and "have no row in the ledger" in str(bad)


def test_ledger_non_variant_missing_is_counted_as_before(tmp_path):
    bad, out = _cmp(tmp_path, {P.tree_name("dust_brush_box__20231005_201"): SIG, VAR: SIG})
    assert bad is None and "clips missing from the ledger 1 (not compared)" in out


# ---- structure ------------------------------------------------------------------------------------------------------
def calls(rel: str, fn: str, name: str) -> int:
    """Calls `name(...)` / `x.name(...)` inside function fn of module rel (comments and strings do not count)."""
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    f = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == fn)
    return sum(
        1
        for c in ast.walk(f)
        if isinstance(c, ast.Call) and name in (getattr(c.func, "attr", ""), getattr(c.func, "id", ""))
    )


def test_structure():
    assert calls("dexengine/data/objects/extra_slots.py", "clip_rows", "source_dir") == 1
    assert calls("dexengine/rl/env_setup/extra_slots_plan.py", "_plan_clip", "source_dir") == 1
    assert calls("dexengine/rl/env_setup/extra_slots_ledger.py", "_by_clip_name", "source_clip") == 1
    assert calls("dexengine/rl/env_setup/extra_slots_ledger.py", "_ledger_compare", "is_variant") >= 1
    src = (REPO / "dexengine/rl/env_setup/extra_slots_plan.py").read_text()
    assert "zip(plan.seqs, plan.sig_rows, strict=True)" in src
