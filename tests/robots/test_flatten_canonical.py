"""flatten.flatten_canonical — the flattened robot USD's bytes do not depend on the run.

Measured: ``Stage.Flatten`` of the same instanced robot stage gives a different ``Flattened_Prototype_N``
numbering on every run (same input, same output path, ASLR off too), so the baked USD's md5 — the USD_MD5_PINS
identity — changed bake to bake. These tests pin the renumbering: a layer whose prototype numbers are permuted
canonicalises to the same text, the composed stage is unchanged, and counts are kept."""

import re

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from dexengine.robots.usd import flatten as FL  # noqa: E402


def _instanced_stage() -> Usd.Stage:
    """/robot with five instanceable links over three distinct source prims (→ three prototypes)."""
    st = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(st, "/robot")
    st.SetDefaultPrim(st.GetPrimAtPath("/robot"))
    for name, size in (("srcA", 1.0), ("srcB", 2.0), ("srcC", 3.0)):
        UsdGeom.Cube.Define(st, f"/{name}/geom").GetSizeAttr().Set(size)
    for i, src in enumerate(("srcA", "srcB", "srcC", "srcA", "srcB")):
        p = st.DefinePrim(f"/robot/link_{i}/visual", "Xform")
        p.GetReferences().AddInternalReference(f"/{src}")
        p.SetInstanceable(True)
    return st


def _permuted(layer: Sdf.Layer, perm: dict[str, str]) -> Sdf.Layer:
    text = re.sub(
        r"Flattened_Prototype_(\d+)(?!\d)", lambda m: "Flattened_Prototype_" + perm[m.group(1)], layer.ExportToString()
    )
    out = Sdf.Layer.CreateAnonymous(".usda")
    assert out.ImportFromString(text)
    return out


def _composed(layer: Sdf.Layer) -> list[tuple[str, str, object]]:
    st = Usd.Stage.Open(layer)
    rows = []
    for p in st.Traverse(Usd.TraverseInstanceProxies()):
        if not p.GetPath().pathString.startswith("/robot"):
            continue
        size = p.GetAttribute("size").Get() if p.IsA(UsdGeom.Cube) else None
        rows.append((p.GetPath().pathString, p.GetTypeName(), size))
    return sorted(rows)


def test_permuted_numbering_canonicalises_to_the_same_text():
    flat = _instanced_stage().Flatten()
    names = sorted({p.name for p in flat.rootPrims if p.name.startswith(FL._PROTO_PREFIX)})
    assert len(names) == 3, names  # three prototypes: the fixture is instanced as intended
    nums = [n.removeprefix(FL._PROTO_PREFIX) for n in names]
    a = FL.canonical_layer(flat).ExportToString()
    for shift in (1, 2):
        perm = {n: nums[(i + shift) % len(nums)] for i, n in enumerate(nums)}
        b = FL.canonical_layer(_permuted(flat, perm)).ExportToString()
        assert a == b


def test_composition_and_counts_are_kept(capsys):
    flat = _instanced_stage().Flatten()
    canon = FL.canonical_layer(flat)
    assert _composed(canon) == _composed(flat)
    assert sum(map(len, FL._proto_refs(canon).values())) == sum(map(len, FL._proto_refs(flat).values())) == 5
    assert re.search(r"\[FLATTEN_CANON\] protos 3 → 3 · refs 5 → 5", capsys.readouterr().out)


def test_documentation_is_cleared_and_input_untouched():
    flat = _instanced_stage().Flatten()
    flat.documentation = "Generated from Composed Stage of root layer /abs/path/robot.usd"
    canon = FL.canonical_layer(flat)
    assert canon.documentation == ""
    assert flat.documentation.startswith("Generated from")  # the caller's layer is read only


def test_stage3_reflatten_keeps_unreferenced_identical_copies():
    """mimic.py flattens the stage-2 output again: the stage-2 prototypes stay as unreferenced roots next to the new
    ones; two of them may share content (here srcA and srcD are the same cube). That tie cannot change the text, so
    it is kept (measured on the robot: 234 prototype roots, 117 referenced) and the result is still canonical."""
    st = _instanced_stage()
    UsdGeom.Cube.Define(st, "/srcD/geom").GetSizeAttr().Set(1.0)  # same content as srcA
    p = st.DefinePrim("/robot/link_9/visual", "Xform")
    p.GetReferences().AddInternalReference("/srcD")
    p.SetInstanceable(True)
    stage2 = FL.canonical_layer(st.Flatten())
    out = FL.canonical_layer(Usd.Stage.Open(stage2).Flatten())
    names = [q.name for q in out.rootPrims if q.name.startswith(FL._PROTO_PREFIX)]
    assert len(names) == 8 and sum(map(len, FL._proto_refs(out).values())) == 6
    nums = sorted({n.removeprefix(FL._PROTO_PREFIX) for n in names}, key=int)
    perm = {n: nums[(i + 3) % len(nums)] for i, n in enumerate(nums)}
    again = FL.canonical_layer(Usd.Stage.Open(stage2).Flatten())
    assert out.ExportToString() == again.ExportToString()
    assert FL.canonical_layer(_permuted(out, perm)).ExportToString() == out.ExportToString()
