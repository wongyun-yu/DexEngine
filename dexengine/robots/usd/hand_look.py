#!/usr/bin/env python3
"""Hand render look: paint and oracle. The declaration (owner of the values) is ``dexengine.core.hand_looks``.

apply(stage, arm, hand)     paint the hand visual meshes of a render body (render_body.py) by the rules: one new
                            material per rule (`<Looks of the original material>/hand_look_<hand>_<i>`) and the visual
                            binding of every mesh the rule matches (for instancing, the source prototype's mesh) is
                            re-bound to it — the same source mesh matched by two rules is an error. The original
                            materials stay.
problems(stage, arm, hand)  oracle: every hand-link mesh is matched by a rule, and the material values bound to the
                            painted meshes equal the declaration.
The arm names the EE link the rules reach through hand_looks.EE_TOKEN (look_for).
CLI: hand_look.py --usd <render plate usd> [--arm <arm>] [--hand <hand>]  (omitted = from the plate's folder name) · rc
0 = as declared
Marker: [HAND_LOOK] …
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

from pxr import Gf, Pcp, Sdf, Usd, UsdGeom, UsdShade

from dexengine.core import hand_looks as HL
from dexengine.core.errors import InputMissingError

TAG = HL.TAG
TOL = 1e-4
_SIDE = re.compile(r"(^|_)(right|left)(_|$)")


def side_free(link: str) -> str:
    return _SIDE.sub(r"\1{side}\3", link)


def ee_link(arm: str) -> str:
    """The arm's EE link with the side removed (ArmSpec.ee_body_name of side ``{side}``)."""
    from dexengine.core.arms import ARM_REGISTRY  # noqa: PLC0415

    return ARM_REGISTRY[arm].ee_body_name("{side}")


def look_for(arm: str, hand: str):
    """The hand's declared look (hand_looks.HAND_LOOKS) with EE_TOKEN in the link regexes replaced by the arm's EE link;
    None = no declaration."""
    look = HL.HAND_LOOKS.get(hand)
    if look is None:
        return None
    ee = re.escape(ee_link(arm))
    return dataclasses.replace(look, rules=tuple((lre.replace(HL.EE_TOKEN, ee), m, p) for lre, m, p in look.rules))


def _in_scope(link: str, ee: str) -> tuple[bool, bool]:
    """(is it a hand link, is it the EE link) — by the side-stripped name."""
    return link.startswith("{side}_"), link == ee


def _units(stage, ee: str):
    """The visual units of the hand (meshes and GeomSubsets with a binding):
    (prim, side-stripped link, bound material name, material, its binding relationship)."""
    for p in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or "collision" in str(p.GetPath()).lower():
            continue
        seg = str(p.GetPath()).split("/")
        if len(seg) < 3:
            continue
        link = side_free(seg[2])
        hand_l, ee_l = _in_scope(link, ee)
        if not (hand_l or ee_l):
            continue
        units = [p] + [
            s.GetPrim()
            for s in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(p))
            if s.GetPrim().HasRelationship("material:binding")
        ]
        for u in units:
            m, rel = UsdShade.MaterialBindingAPI(u).ComputeBoundMaterial()
            yield u, link, (m.GetPrim().GetName() if m else ""), m, rel


def _rule(look, link: str, mat: str) -> int | None:
    for i, (lre, mre, _paint) in enumerate(look.rules):
        if re.fullmatch(lre, link) and re.fullmatch(mre, mat):
            return i
    return None


def _rule_of(look, hand: str, link: str, mname: str) -> int | None:
    """The oracle's rule number: i for an already painted material (hand_look_<hand>_<i>; None when the declaration
    has no such number), otherwise _rule."""
    own = f"hand_look_{hand}_"
    if mname.startswith(own):
        i = int(mname[len(own) :])
        return i if 0 <= i < len(look.rules) else None
    return _rule(look, link, mname)


def _source(stage, p):
    """Instance proxy → the source prim to edit (the same relative path on the prototype the instance references);
    itself when not instanced."""
    while p.IsInstanceProxy():
        anc = p.GetParent()
        while anc and not anc.IsInstance():
            anc = anc.GetParent()
        tgt = None
        for arc in Usd.PrimCompositionQuery(anc).GetCompositionArcs():
            if arc.GetArcType() == Pcp.ArcTypeReference and arc.GetTargetLayer() == stage.GetRootLayer():
                tgt = arc.GetTargetPrimPath()
                break
        if tgt is None:
            raise InputMissingError(f"{TAG} ✗ no source reference found for instance {anc.GetPath()} — cannot paint")
        p = stage.GetPrimAtPath(tgt.AppendPath(p.GetPath().MakeRelativePath(anc.GetPath())))
        if not p:
            raise InputMissingError(f"{TAG} ✗ source prim missing: {tgt}")
    return p


def _material(stage, scope: Sdf.Path, name: str, paint) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, scope.AppendChild(name))
    sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*paint.diffuse))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(paint.metallic))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(paint.roughness))
    if paint.specular is not None:
        sh.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(paint.specular))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
        o.ConnectToSource(out)
    return mat


def apply(stage, arm: str, hand: str) -> list[str]:
    look = look_for(arm, hand)
    if look is None:
        return [
            f"{TAG} {hand} no declaration — original colours (merged URDF · vendor meshes) kept (hand_looks.HAND_LOOKS)"
        ]
    owner: dict[str, int] = {}
    relname: dict[str, str] = {}
    plan: dict[int, list] = {}
    bad = []
    ee = ee_link(arm)
    for u, link, mname, _m, rel in _units(stage, ee):
        i = _rule(
            look, link, mname
        )  # paint only an unpainted body (render_body's copy of the source) — by the rule regex
        if i is None:
            if _in_scope(link, ee)[0]:
                bad.append(f"{link}:{mname or '(no material)'}")
            continue
        if look.rules[i][2] is None:
            continue
        # re-bind the prim that actually decides the binding (possibly an ancestor's strongerThanDescendants binding)
        # with the same strength
        src = _source(stage, rel.GetPrim() if rel else u)
        relname[str(src.GetPath())] = rel.GetName() if rel else ""
        k = str(src.GetPath())
        if owner.setdefault(k, i) != i:
            raise InputMissingError(
                f"{TAG} ✗ {hand} source mesh {k} is matched by rules {owner[k]} and {i} "
                "(shared prototype) — the rules cannot be split by link"
            )
        plan.setdefault(i, []).append(src)
    if bad:
        raise SystemExit(
            f"{TAG} ✗ {hand}: {len(bad)} hand meshes match no rule (unknown parts) — e.g. "
            f"{sorted(set(bad))[:6]} · add a rule to hand_looks.HAND_LOOKS[{hand!r}]"
        )
    lines = []
    for i, srcs in sorted(plan.items()):
        paint = look.rules[i][2]
        mat = _material(stage, stage.GetDefaultPrim().GetPath().AppendChild("Looks"), f"hand_look_{hand}_{i}", paint)
        uniq = {str(s.GetPath()): s for s in srcs}
        for k, s in uniq.items():
            # Change only the target of the existing binding relationship — strength (bindMaterialAs), applied schemas
            # and everything else stay (render_plate.parity requires the prims outside materials to be equal). A mesh
            # without a binding gets a new one.
            if relname[k]:
                s.GetRelationship(relname[k]).SetTargets([mat.GetPath()])
            else:
                UsdShade.MaterialBindingAPI(s).Bind(mat)
        lines.append(
            f"{TAG} {hand} rule {i} {look.rules[i][1]!r} → colour {paint.diffuse} metallic {paint.metallic} "
            f"roughness {paint.roughness} · units {len(srcs)} · re-bound sources {len(uniq)}"
        )
    if not plan:
        raise SystemExit(
            f"{TAG} ✗ {hand} has a declaration but 0 painted meshes — the rules do not match this plate's names"
        )
    return lines


def _input_value(shader, name):
    i = shader.GetInput(name)
    return i.Get() if i else None


def _shader_vals(m):
    for x in Usd.PrimRange(m.GetPrim()):
        s = UsdShade.Shader(x)
        if s:
            return (
                _input_value(s, "diffuse_color_constant"),
                _input_value(s, "metallic_constant"),
                _input_value(s, "reflection_roughness_constant"),
                _input_value(s, "specular_level"),
            )
    return None, None, None, None


def problems(stage, arm: str, hand: str) -> list[str]:
    look = look_for(arm, hand)
    if look is None:
        return []
    bad, n, ee = [], 0, ee_link(arm)
    for _u, link, mname, m, _rel in _units(stage, ee):
        i = _rule_of(look, hand, link, mname)
        if i is None:
            if _in_scope(link, ee)[0]:
                bad.append(f"{link}:{mname or '(no material)'} no rule")
            continue
        paint = look.rules[i][2]
        if paint is None:
            continue
        n += 1
        c, mt, r, sp = _shader_vals(m) if m else (None, None, None, None)
        ok = (
            c is not None
            and max(abs(float(a) - b) for a, b in zip(c, paint.diffuse, strict=False)) <= TOL
            and mt is not None
            and abs(float(mt) - paint.metallic) <= TOL
            and r is not None
            and abs(float(r) - paint.roughness) <= TOL
            and (paint.specular is None or (sp is not None and abs(float(sp) - paint.specular) <= TOL))
        )
        if not ok:
            bad.append(f"{link}:{mname} paint {c},{mt},{r},{sp} ≠ declared {paint}")
    if n == 0:
        bad.append("0 meshes matched by a paint rule")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--arm", default="")
    ap.add_argument("--hand", default="")
    a = ap.parse_args()
    from dexengine.robots.usd import render_parts as RVP  # noqa: PLC0415

    named = RVP.pair_of(Path(a.usd).parent.name) if not (a.arm and a.hand) else ("", "")
    arm, hand = a.arm or named[0], a.hand or named[1]
    st = Usd.Stage.Open(a.usd)
    bad = problems(st, arm, hand)
    look = HL.HAND_LOOKS.get(hand)
    print(
        f"{TAG} {Path(a.usd).name} · hand {hand} · "
        + (
            "no declaration (nothing compared)"
            if look is None
            else f"mismatches {len(bad)}" + (f" {bad[:4]}" if bad else " ✓")
        ),
        flush=True,
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
