#!/usr/bin/env python3
"""Render-body generator (``<body>mi``): repaint a training body for rendering; kinematics, physics, colliders and
instancing stay as trained.

Palette:
  white parts (material_54_254_255* + material_CAD1EE, NECK mount included) → bright steel
      albedo (1,1,1) · metallic 0.9 · roughness 0.45
  top camera (DefaultMaterial of the camera scope — the camera visual by
      head_camera.camera_visual_names, not by name)                      → full black
  the existing "light black" (material_0_10_10* — shoulder LINK_4/5 · elbow LINK_7/8 ·
      wrist LINK_14/15)                                                  → unified to full black
  green (elastomer) and everything else                                  → unchanged

Three measured traps (why the script looks like this):
  1) The material the white covers bind is not CAD1EE but material_54_254_255_* inside the prototype scope — fixing
     CAD1EE by name changes nothing in the render.
  2) Those materials have no Shader child at all, so the renderer paints its default colour: writing parameters
     anywhere is useless, an OmniPBR shader must be created.
  3) Flattened_Prototype_* are authored in the file, so they can be edited on the composed stage (they are not
     instance proxies); instancing (117 prototypes) is fully kept.
  DefaultMaterial is a separate prim per scope with the same name: it is judged by the set of parts it is actually
  bound to, not by name (so hand parts are not repainted).

Hand look (the manufacturer's official appearance, render body only): after the body paint, ``hand_look.apply``
  paints the hand visual meshes as declared (``dexengine.core.hand_looks.HAND_LOOKS``; a new material per rule,
  visual bindings only). The oracle ``hand_look.problems`` measures the same places. Marker [HAND_LOOK].

Render mount parts (render body only): then ``render_parts.add`` mounts the visual parts the (arm, hand) mount
  declares (``MountSpec.render_visual_parts``) onto the EE link's visual prototype — without collider, mass or joint.
  A pair without declared parts stays byte-identical. Marker [RENDER_VISUAL_PART]. Compared with the training plate by
  ``render_plate.parity`` (only the declared parts are allowed).

Head camera (only the mesh attachment changes): after painting, ``head_camera.swap`` removes the camera link's
  visual child — the one the URDF the body was baked from states (its config.yaml record) and the stage rule finds,
  both the same or the bake stops — and mounts one ZED Mini (zedm.stl at the calibrated mount) onto
  link_head_2's visual prototype — render body only. Marker [RENDER_HEAD_CAM]. Rule and values: head_camera.py;
  comparison with the training plate: render_plate.parity.

Pixel check: |mi − m100| 20.84 → 5.35. A flattened body cannot be used for training (scene creation at NE=2048 stalls
for more than 12 minutes), so keeping instancing is a precondition.

Usage: python3 dexengine/robots/usd/render_body.py --src-dir <body directory>
"""

import argparse
import os
import shutil
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from dexengine.robots.usd import hand_look as HLK
from dexengine.robots.usd import head_camera as HC
from dexengine.robots.usd import render_parts as RVP

WHITE_PREFIXES = ("material_54_254_255",)
WHITE_EXACT = {"material_CAD1EE"}
BLACK_PREFIXES = ("material_0_10_10",)
STEEL = {"diffuse": (1.0, 1.0, 1.0), "metallic": 0.9, "roughness": 0.45}
BLACK = {"diffuse": (0.039, 0.039, 0.039), "metallic": 0.35, "roughness": 0.5}


def _material_parts(st):
    """Material prim path → the parts it is bound to (independent of name clashes)."""
    # material prim path → the set of parts it is bound to (independent of name clashes)
    matparts = {}
    for p in st.TraverseAll():
        if not p.IsA(UsdGeom.Mesh):
            continue
        b = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
        if not b:
            continue
        seg = p.GetPath().pathString.split("/")
        matparts.setdefault(b.GetPrim().GetPath().pathString, set()).add(seg[2] if len(seg) > 2 else seg[-1])
    return matparts


def _paint_materials(st, matparts, a, camera: set):
    """Steel/black OmniPBR paint on the matching materials (``camera`` = the camera visual's names — its
    DefaultMaterial goes black). Returns (made, fixed, fail, n_steel, n_black)."""
    made = fixed = fail = n_steel = n_black = 0
    for m in list(st.TraverseAll()):
        if not m.IsA(UsdShade.Material):
            continue
        nm = m.GetName()
        parts = matparts.get(m.GetPath().pathString, set())
        if nm in WHITE_EXACT or any(nm.startswith(p) for p in WHITE_PREFIXES):
            paint = STEEL
        elif any(nm.startswith(p) for p in BLACK_PREFIXES) or nm == "DefaultMaterial" and parts and parts <= camera:
            paint = BLACK
        else:
            continue
        if m.IsInstanceProxy():
            continue
        ex = None
        for x in Usd.PrimRange(m):
            if x.IsA(UsdShade.Shader):
                ex = x
                break
        try:
            if ex is not None:
                sh = UsdShade.Shader(ex)
            else:
                sh = UsdShade.Shader.Define(st, m.GetPath().AppendChild("Shader"))
                if not sh.GetPrim().IsValid():
                    fail += 1
                    continue
                sh.SetSourceAsset("OmniPBR.mdl", "mdl")
                sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            rough = a.steel_roughness if paint is STEEL else paint["roughness"]
            sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*paint["diffuse"]))
            sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(paint["metallic"]))
            sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
            outp = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
            mat = UsdShade.Material(m)
            mat.CreateSurfaceOutput("mdl").ConnectToSource(outp)
            mat.CreateDisplacementOutput("mdl").ConnectToSource(outp)
            mat.CreateVolumeOutput("mdl").ConnectToSource(outp)
            made += ex is None
            fixed += ex is not None
            if paint is STEEL:
                n_steel += 1
            else:
                n_black += 1
        except Exception:  # noqa: BLE001 — a material that cannot be edited is counted; any count fails the bake
            fail += 1
    return made, fixed, fail, n_steel, n_black


def _apply_render_declarations(base, st, src_usd):
    """Hand look, head camera and render-only mount parts; a declaration the stage violates stops the bake.
    ``src_usd`` = the training body (its bake record names the URDF the camera visual is read from)."""
    arm, hand = RVP.pair_of(base)
    for line in HLK.apply(
        st, arm, hand
    ):  # hand look = the official appearance (declared in hand_looks.HAND_LOOKS · render body only)
        print(line, flush=True)
    bad = HLK.problems(st, arm, hand)
    if bad:
        sys.exit(f"[make_mi] differs from the hand look declaration {bad[:4]} — do not use as canonical")
    print(HC.swap(st, HC.stated_camera_visuals(HC.body_urdf(src_usd), arm)), flush=True)
    bad = HC.problems(st)
    if bad:
        sys.exit(f"[make_mi] head camera rule violated {bad} — do not use as canonical")
    for line in RVP.add(
        st, arm, hand
    ):  # render-only mount parts — a pair without a declaration (dexengine.core.mounts) is left untouched
        print(line, flush=True)
    bad = RVP.problems(st, arm, hand)
    if bad:
        sys.exit(f"[make_mi] render mount part declaration violated {bad} — do not use as canonical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="training body directory (holds <name>.usd)")
    ap.add_argument("--steel-roughness", type=float, default=STEEL["roughness"])
    ap.add_argument(
        "--out-dir", default=None, help="default: <src-dir>mi (the file name follows the same rule: <name>mi.usd)"
    )
    a = ap.parse_args()
    base = os.path.basename(a.src_dir.rstrip("/"))
    src = os.path.join(a.src_dir, base + ".usd")
    outd = a.out_dir or (a.src_dir.rstrip("/") + "mi")
    out = os.path.join(outd, os.path.basename(outd.rstrip("/")) + ".usd")
    if os.path.isdir(outd):
        shutil.rmtree(outd)
    os.makedirs(outd)
    try:
        _make(a, base, src, out)
    except BaseException:  # a stop after the copy leaves no file: the copy is the training body, not a render body
        shutil.rmtree(outd, ignore_errors=True)
        raise


def _make(a, base, src, out):
    """Copy the training body to ``out`` and make it the render body (any stop raises; ``main`` removes ``out``)."""
    shutil.copy2(src, out)  # the .usd only — the source body's provenance metadata (config.yaml etc.) is not copied

    st = Usd.Stage.Open(out)
    matparts = _material_parts(st)

    made, fixed, fail, n_steel, n_black = _paint_materials(st, matparts, a, set(HC.camera_visual_names(st)))
    _apply_render_declarations(base, st, src)
    st.GetRootLayer().Save()
    st2 = Usd.Stage.Open(out)
    protos = len(st2.GetPrototypes())
    print(f"[make_mi] {out}")
    print(
        f"[make_mi]   steel {n_steel} · black {n_black} · new shaders {made} · edited {fixed} · failed {fail} · "
        f"prototypes {protos}"
    )
    if fail or protos == 0:
        sys.exit("[make_mi] failure or lost instancing — do not use as canonical")


if __name__ == "__main__":
    main()
