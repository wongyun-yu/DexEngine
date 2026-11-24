#!/usr/bin/env python3
"""Bake oracle — compare, link by link, the URDF visual colours (stated) with the baked USD mesh material colours
(measured); an independent check of the USD import.

Per link: if the URDF gives that link's visuals a colour (top-level definition or resolved inline value,
``urdf.materials.visual_colors``), the set of material colours bound to the visual meshes under the USD link prim of
that name must contain every such colour; one missing colour fails (rc 1). Links absent from the USD because fixed
joints were merged are only counted (skipped). Visuals without a colour are not compared.
Marker: [VISUAL_COLORS] <usd> · links compared n · skipped m · mismatches k (first few)
Usage: check_colors.py --urdf <merged URDF> --usd <baked USD>
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

from pxr import Usd, UsdGeom, UsdShade

from dexengine.core.errors import input_missing
from dexengine.robots.urdf import materials as UM

TOL = 2e-3


def mesh_color(p) -> tuple[float, ...] | None:
    mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
    if not mat:
        return None
    for sh in Usd.PrimRange(mat.GetPrim()):
        s = UsdShade.Shader(sh)
        if not s:
            continue
        for nm in ("diffuse_color_constant", "diffuseColor", "base_color", "baseColor"):
            i = s.GetInput(nm)
            if i and i.Get() is not None:
                return tuple(float(x) for x in i.Get())
    return None


def usd_link_colors(stage) -> dict[str, list]:
    out: dict[str, list] = {}
    for p in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or "collision" in str(p.GetPath()).lower():
            continue
        parts = str(p.GetPath()).split("/")
        if len(parts) > 2:
            out.setdefault(parts[2], []).append(mesh_color(p))
    return out


def check(urdf: str, usd: str) -> tuple[int, int, list]:
    want = UM.visual_colors(ET.parse(urdf).getroot())
    got = usd_link_colors(Usd.Stage.Open(usd))
    n = skip = 0
    bad = []
    for link, cols in want.items():
        exp = {c[:3] for c in cols if c is not None}
        if not exp:
            continue
        if link not in got:
            skip += 1
            continue
        n += 1
        have = [c[:3] for c in got[link] if c is not None]
        miss = [e for e in exp if not any(max(abs(a - b) for a, b in zip(e, h, strict=False)) <= TOL for h in have)]
        if miss:
            bad.append((link, sorted(miss), sorted({tuple(round(x, 3) for x in h) for h in have}) or ["no material"]))
    return n, skip, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--usd", required=True)
    a = ap.parse_args()
    n, skip, bad = check(a.urdf, a.usd)
    head = (
        f"[VISUAL_COLORS] {a.usd.rsplit('/', 1)[-1]} · links compared {n} · skipped (merged) {skip} · "
        f"mismatches {len(bad)}"
    )
    if bad:
        print(head + " ✗ " + " | ".join(f"{lk} URDF {m} ↔ USD {h}" for lk, m, h in bad[:4]), flush=True)
        return 1
    if n == 0 and skip:
        print(
            head + input_missing(" ✗ no coloured link resolves in the USD — nothing compared is not a pass"), flush=True
        )
        return 1
    if n == 0:
        print(head + " ✓ the URDF has no coloured visual (nothing stated to compare)", flush=True)
        return 0
    print(head + " ✓", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
