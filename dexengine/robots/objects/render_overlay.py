"""Generator of object render overlay USDs: textured render-only layers next to the training object USDs.

Overlay = `<asset>/<stem>_render.usd` (name: artifact_paths.render_usd_file). It loads the training USD (`object.usd`,
`object_artic.usd`) as a sublayer and only adds a UsdPreviewSurface material and face-vertex UVs (`primvars:st`) to the
visual mesh prims (`/Object/<link>/visuals/mesh`). Collision, mass and joints come from the training layer through
composition (there is no copy that could drift), so training USD bytes, cooked dumps (cooked_hull, keyed by the source
USD sha256) and judge input fingerprints do not change. Texture images are copied to `render_tex/` next to the overlay
(so they move with it).

Texture rules: dexengine/robots/textures/rules.py (mesh MTL map_Kd + UV, raw re-intake) + final stage textures/fill.py.
Alignment guard: the texture is loaded only when the training USD's face-corner positions (face and corner order) equal
those of the UV source (the mesh itself, a raw OBJ, or a merged asset's part list -> atlas); otherwise a mismatch
status. Vertex indices are not compared (they differ between trimesh versions). Links whose topology differs from the
raw source (ARCTIC) get the raw UV transferred onto our visual mesh (uv_transfer: shape kept; failing the alignment
gate gives the final-stage uniform colour).
Targets = all object USDs in the dataset's usd layer (enumeration owner: cooked_hull.targets). Parts of articulated
meshes come from the axis hand's articulation spec (datasets.axis_output(ARTIC_SPEC), the same source as the bake);
an articulated USD not in the spec gets the no-spec status. Assets without a texture also get an overlay (sublayer
only, no material); no overlay = not baked yet (rendering then uses the training USD with an `[RENDER_OBJ_USD]` marker).

Texture packs (`--packs`, off by default; off = overlays and tables byte-identical): rules textures.rules.PACK_*; the
uniform policy (GRAB) needs no pack, only per-object colours (colors.object_color); marker `[OBJ_TEX_PACKS]`; adds the
source columns to the table. With packs on, each pack dataset ends with `[OBJ_TEX_COVER] ds=… used=… covered=…
missing=…`: used objects (rules.used_objects) minus covered overlays (rule tex_packs.cover).
Preview `--overlay-root <dir>`: overlays go under that root, not the canonical place (render env OVERLAY_ROOT_ENV).
Overlay list: the overlays a run wrote are merged into the list the render checks (robots.objects.overlay_manifest
record — the data-root list or a project list; the canonical git list is written by its own scan;
a preview root writes no list).

Runs in the retarget venv or Isaac python (no Kit, CPU); arguments: --help.
Markers: per asset ``[RENDER_TEX] <ds>/<id> art=<0|1> <link>=<status> …``; at the end ``[RENDER_TEX_DONE] usd=N
written=W textured=T failed=F``. Tables: `<out>/render_textures.tsv` (per asset and link) and
`<out>/render_textures_table.json` (per dataset: counts of meshes, UV, images and each status, keyed by the recorded
status words). rc 0 = no failures and written overlays = targets. Status words and `why` texts are recorded in the
overlays and tables and keep their exact text.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from dexengine.core import artifact_paths as AP
from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.robots.objects import cooked_hull as OCH
from dexengine.robots.objects import overlay_manifest as OM
from dexengine.robots.objects import tex_packs as RTP
from dexengine.robots.objects import uv_transfer as RUT
from dexengine.robots.objects.render_overlay_tables import write_link_table as _write_link_table
from dexengine.robots.objects.render_overlay_tables import write_usd_table as _write_usd_table
from dexengine.robots.textures import fill as OTF
from dexengine.robots.textures import rules as OT

TP = OCH.TP
#: Asset kind -> links with a visual mesh (prim names of the training bake: rigid base_link, articulated base/moving).
LINKS = {False: ("base_link",), True: ("base_link", "moving_link")}
UNBAKED = "not baked"


def _part_objs(ds: str, oid: str, art: bool, specs: dict) -> dict | None:
    """Link -> (mesh OBJ, raw re-intake object id, part file name). None for an articulated asset not in the spec."""
    t = TP.get(ds)
    if not art:
        return {"base_link": (t.visual_mesh_file(oid), oid, "visual.obj")}
    s = specs.get(f"{ds}/{oid}")
    if s is None:
        return None
    d = t.visual_mesh_dir(s["obj_id"])
    out = {}
    for link, key in (("base_link", "base"), ("moving_link", "moving")):
        f = os.path.normpath(os.path.join(d, s["parts"][key]))
        out[link] = (f, os.path.basename(os.path.dirname(f)), os.path.basename(f))
    return out


def _texture(ds: str, mesh_obj: str, raw_id: str, part: str, scratch: str, packs: dict | None = None) -> dict:
    """Texture rules (textures.rules): the mesh's own material first, then the texture pack (only with `--packs`, rigid
    visual meshes only), then raw re-intake, then the material colour for datasets without raw re-intake, and finally
    uniform colour when the raw has no texture (final stage = textures.fill)."""
    t = OT.find_texture(mesh_obj)
    if t["status"] == OT.OK:
        return {"status": OT.OK, "image": t["image"], "uv_obj": mesh_obj, "why": ""}
    k = RTP.lookup(packs, ds, raw_id, part)
    if k is not None:
        return k
    r = OT.raw_source(
        ds,
        P.data_layout.data_abspath(P.data_layout.render("raw", ds)),
        raw_id,
        part,
        scratch,
        asset_dir=os.path.dirname(mesh_obj),
    )
    if r is None:
        c = OTF.mtl_color(ds, t)
        if c is not None:
            return {
                "status": OT.OK_MTL_COLOR,
                "image": None,
                "uv_obj": None,
                "color": c,
                "why": f"Kd {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}",
            }
        r = {"status": t["status"], "image": None, "raw_obj": None, "why": t.get("why", "")}
    if OTF.no_texture_status(ds, r["status"]):  # the raw has no texture -> uniform colour (why = rule reason)
        return {
            "status": OT.OK_NO_TEX,
            "image": None,
            "uv_obj": None,
            "color": OTF.uniform_rgb(ds, raw_id),
            "why": r["status"],
        }
    return {"status": r["status"], "image": r["image"], "uv_obj": r["raw_obj"], "why": r["why"]}


def _corner_tol(uv_obj) -> float:
    """Corner position tolerance: the merged OBJ's write precision for a part list (fixed_merge.POS_ATOL), else 1e-6."""
    if isinstance(uv_obj, list):
        from dexengine.data.oakink2 import fixed_merge as FM

        return FM.POS_ATOL
    return 1e-6


def _uv_corners(uv_obj):
    """Corner positions (F,3,3) of the UV source and the per-part corner UV list. Source = one OBJ path or
    [(OBJ, pose | None)] (merged asset). Returns a reason string when the UV cannot be read."""
    import trimesh

    srcs = uv_obj if isinstance(uv_obj, list) else [(uv_obj, None)]
    cs, uvs = [], []
    for q, pose in srcs:
        r = trimesh.load(q, force="mesh", process=False)
        ruv = getattr(r.visual, "uv", None)
        if ruv is None or len(ruv) != len(r.vertices):
            return "mismatch:UV load failed"
        V = np.asarray(r.vertices, dtype=np.float64)
        if pose is not None:
            from dexengine.data.oakink2 import fixed_merge as FM

            V = FM.place(V, pose)
        cs.append(V[r.faces])
        uvs.append(np.asarray(ruv, dtype=np.float64)[r.faces])
    return np.concatenate(cs), uvs


def _st(
    mesh_obj: str, uv_obj, pts: np.ndarray | None = None, fvi: np.ndarray | None = None
) -> tuple[np.ndarray | None, str]:
    """Face-vertex UVs (faceVarying), only when the training USD's corner positions (face order, corner order) equal
    those of the UV source. Vertex indices are not compared: even for the same OBJ (v/vt), trimesh versions split
    vertices differently (training USD bake = Isaac env version, this generator = .venv version). Without a training
    USD the mesh OBJ corners are used. When uv_obj is a part list, part UVs are mapped to atlas coordinates
    (uv_transfer.atlas_st)."""
    import trimesh

    same = uv_obj == mesh_obj
    if pts is not None and fvi is not None:
        if fvi.size % 3:
            return None, "mismatch:USD faces not triangles"
        tgt = np.asarray(pts, dtype=np.float64)[np.asarray(fvi).reshape(-1, 3)]
    else:
        m = trimesh.load(mesh_obj, force="mesh", process=False)
        tgt = np.asarray(m.vertices, dtype=np.float64)[m.faces]
    got = _uv_corners(uv_obj)
    if isinstance(got, str):
        return None, got
    cor, uvs = got
    if cor.shape != tgt.shape:
        return None, (
            f"mismatch:USD faces ≠ mesh faces ({len(tgt)} · {len(cor)})"
            if same
            else f"{OT.RAW_MISMATCH}:face count {len(cor)} ≠ {len(tgt)}"
        )
    if len(cor) and float(np.abs(cor - tgt).max()) > _corner_tol(uv_obj):
        return None, "mismatch:USD points ≠ mesh points" if same else f"{OT.RAW_MISMATCH}:face vertex positions"
    if isinstance(uv_obj, list):
        return RUT.atlas_st(uvs, [RUT.mtl_image(q) for q, _ in uv_obj]).astype(np.float32).reshape(-1, 2), ""
    return uvs[0].astype(np.float32).reshape(-1, 2), ""


def write_overlay(train_usd: str, links: dict, overlay_root: str | None = None, sources: dict | None = None) -> str:
    """Write the overlay (temporary file -> rename). links = {link: {status, image, st | color}}: links with image and
    st get a texture material, links with color (uniform pack) a constant-colour material. overlay_root
    (`--overlay-root`) writes artifact_paths.overlay_file, not the canonical place (sublayer = absolute path of the
    training USD). sources (only with `--packs`) = per-link source records."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    out = TP.render_usd_file(train_usd) if overlay_root is None else TP.overlay_file(train_usd, overlay_root)
    d = os.path.dirname(out)
    os.makedirs(d, exist_ok=True)
    tex_dir = os.path.join(d, TP.RENDER_TEX_DIR)
    tmp = os.path.join(d, f".{os.path.basename(out)}.tmp.usda")
    if os.path.exists(tmp):
        os.remove(tmp)
    layer = Sdf.Layer.CreateNew(tmp)
    layer.subLayerPaths.append(
        "./" + os.path.basename(train_usd) if overlay_root is None else os.path.abspath(train_usd)
    )
    layer.defaultPrim = "Object"
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for link, x in links.items():
        if x.get("color") is not None:  # uniform-colour pack
            RTP.uniform_material(stage, link, x["color"])
            continue
        if x.get("st") is None:
            continue
        os.makedirs(tex_dir, exist_ok=True)
        atlas = isinstance(x["image"], list)
        if atlas:  # several part images -> one atlas (_st already gave atlas coordinates)
            from PIL import Image

            name = f"{link}__atlas{len(x['image'])}.png"
            Image.fromarray(RUT.atlas_image(x["image"])).save(os.path.join(tex_dir, name))
        else:
            name = f"{link}__{os.path.basename(x['image'])}"
            shutil.copy2(x["image"], os.path.join(tex_dir, name))
        mp = f"/Object/RenderLooks/{link}"
        mat = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, f"{mp}/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        tx = UsdShade.Shader.Define(stage, f"{mp}/DiffuseTex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{TP.RENDER_TEX_DIR}/{name}")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        wrap = "clamp" if atlas else "repeat"
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        rd = UsdShade.Shader.Define(stage, f"{mp}/StReader")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        prim = stage.OverridePrim(f"/Object/{link}/visuals/mesh")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set(Vt.Vec2fArray.FromNumpy(x["st"]))
    cld = {
        AP.OVERLAY_TEX_KEY: {k: v["status"] for k, v in links.items()},
        "generator": "dexengine/robots/objects/render_overlay.py",
    }
    if sources is not None:  # only with --packs (off = no sources key)
        cld[AP.OVERLAY_TEX_SOURCES_KEY] = sources
    layer.customLayerData = cld
    layer.Save()
    os.replace(tmp, out)  # text layer written as .usda, saved under the .usd name (USD detects the format by content)
    return out


def _usd_mesh(train_usd: str, link: str) -> tuple[np.ndarray, np.ndarray] | None:
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(train_usd)
    p = st.GetPrimAtPath(f"/Object/{link}/visuals/mesh")
    if not p.IsValid():
        return None
    m = UsdGeom.Mesh(p)
    return (
        np.asarray(m.GetPointsAttr().Get(), dtype=np.float64),
        np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64),
    )


def bake_one(
    ds: str,
    oid: str,
    art: bool,
    train_usd: str,
    specs: dict,
    scratch: str,
    dry: bool,
    packs: dict | None = None,
    overlay_root: str | None = None,
) -> dict:
    parts = _part_objs(ds, oid, art, specs)
    if parts is None:
        return {"links": {"*": {"status": "no spec"}}, "written": None}
    return bake_links(ds, train_usd, parts, scratch, dry, packs=packs, overlay_root=overlay_root)


def bake_links(
    ds: str,
    train_usd: str,
    parts: dict,
    scratch: str,
    dry: bool,
    packs: dict | None = None,
    overlay_root: str | None = None,
) -> dict:
    """Per link: rules -> alignment guard -> overlay (entry point that tests call with fixed paths)."""
    links = {}
    for link in parts:
        mesh_obj, raw_id, part = parts[link]
        x = _texture(ds, mesh_obj, raw_id, part, scratch, packs)
        if "pack" in x:
            RTP.pack_link(x, _usd_mesh(train_usd, link) if x["image"] else None)
        elif x["image"]:
            um = _usd_mesh(train_usd, link)
            if um is None:
                x.update(status="mismatch:no visual prim", image=None)
            else:
                st, why = _st(mesh_obj, x["uv_obj"], *um)
                if st is None:
                    x.update(status=why, image=None, cand=x["image"])
                else:
                    x["st"] = st
        links[link] = x
    # Links whose topology differs from the raw source (one shared raw OBJ) -> raw UV transfer (only when it passes the
    # alignment gate; otherwise final-stage uniform colour, raw alignment failed, why = residual)
    topo = [
        k
        for k, x in links.items()
        if x.get("cand") and x["status"].startswith(f"{OT.RAW_MISMATCH}:face ") and not isinstance(x["uv_obj"], list)
    ]  # part lists (merged assets): not transferred
    srcs = {links[k]["uv_obj"] for k in topo}
    if topo and len(srcs) == 1:
        sw, why = RUT.transfer(train_usd, topo, srcs.pop())
        for k in topo:
            x = links[k]
            img = x.pop("cand")
            if sw is not None:
                x.update(status=OT.OK_RAW, image=img, st=sw[k]["st"], why=sw[k]["why"])
            elif (col := OTF.uniform_rgb(ds, parts[k][1])) is not None:  # per-object colour decided only when used
                x.update(status=OT.OK_ALIGN_UNIFORM, color=col, why=why)
            else:
                x.update(status=why)
    sources = None if packs is None else RTP.source_rows(links)
    written = None if dry else write_overlay(train_usd, links, overlay_root=overlay_root, sources=sources)
    return {"links": links, "written": written}


def mesh_table(dss: list[str], rows: list[dict], scratch: str) -> dict:
    """Per-dataset mesh table. Mesh = <id>/visual.obj of assets_objects; a mesh counts under the status of the textured
    overlay of its object folder (articulated part OBJs included) when there is one."""
    used = collections.defaultdict(dict)  # ds -> {object folder: loaded status}
    for r in rows:
        for x in r["links"].values():
            if x.get("st") is not None or x.get("color") is not None:
                used[r["ds"]].setdefault(os.path.dirname(x["mesh_obj"]), x["status"])
    out = {}
    for ds in dss:
        t = TP.get(ds)
        root = t.visual_mesh_root()
        c = collections.Counter()
        for oid in sorted(os.listdir(root)):
            f = os.path.join(root, oid, "visual.obj")
            if not os.path.isfile(f):
                continue
            c["mesh"] += 1
            h = OT.parse_obj(f)
            tx = OT.find_texture(f)
            c["UV"] += h["n_vt"] > 0
            c["image"] += tx["status"] == OT.OK
            x = _texture(ds, f, oid, "visual.obj", scratch)
            if os.path.dirname(f) in used[ds]:
                c[used[ds][os.path.dirname(f)]] += 1  # OK statuses (texture, raw, uniform …)
            elif x["image"]:  # a candidate exists but no loaded overlay: check mesh vs UV source
                st, why = _st(f, x["uv_obj"])
                c[
                    why.split(":")[0]
                    if st is None
                    else (UNBAKED if not os.path.isfile(t.usd_file(oid)) else "mismatch")
                ] += 1
            elif x["status"].startswith(OT.RAW_NEEDED):
                c[OT.RAW_NEEDED] += 1
            elif x["status"].startswith(OT.RAW_MISMATCH):
                c[OT.RAW_MISMATCH] += 1
            else:
                c[OT.NONE] += 1
        out[ds] = dict(c)
    return out


def _parse_args():
    """Command line (targets, output folder, texture-pack options)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--dataset")
    ap.add_argument("--ids")
    ap.add_argument("--ids_file")
    ap.add_argument("--dry-run", action="store_true", help="tables only, no overlays written")
    ap.add_argument("--out", required=True, help="table folder")
    RTP.add_args(ap)
    return ap.parse_args()


def main() -> int:
    a = _parse_args()
    with open(RR.axis_output(RR.ARTIC_SPEC), encoding="utf-8") as fs:
        specs = json.load(fs).get("objects", {})
    tg = OCH.targets(a)
    packs = RTP.setup(a, sorted({t[0] for t in tg}))
    oroot = os.path.abspath(a.overlay_root) if a.overlay_root else None
    if oroot is not None:
        os.makedirs(oroot, exist_ok=True)
        print(f"[RENDER_TEX_OVERLAY_ROOT] {oroot} — overlays at the canonical place are not written", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rows, failed = [], []
    with tempfile.TemporaryDirectory(prefix="render_tex_raw_") as scratch:
        for ds, oid, art, f in tg:
            try:
                r = bake_one(ds, oid, bool(art), str(f), specs, scratch, a.dry_run, packs=packs, overlay_root=oroot)
            except Exception as e:  # noqa: BLE001 — one asset's failure goes in the table; rc 1 at the end
                failed.append(f"{ds}/{oid}:{int(art)} {type(e).__name__}: {e}")
                print(f"[RENDER_TEX] ✗ {failed[-1]}", flush=True)
                continue
            parts = _part_objs(ds, oid, bool(art), specs) or {}
            for link, x in r["links"].items():
                x["mesh_obj"] = parts.get(link, (None,))[0]
            rows.append(dict(ds=ds, oid=oid, art=bool(art), **r))
            print(
                f"[RENDER_TEX] {ds}/{oid} art={int(art)} "
                + " ".join(
                    f"{k}={v['status']}"
                    + (
                        f"[{v['why']}]"
                        if (v.get("st") is not None and v.get("why")) or v["status"] == OT.OK_ALIGN_UNIFORM
                        else ""
                    )
                    for k, v in r["links"].items()
                ),
                flush=True,
            )
        table = mesh_table(sorted({r[0] for r in tg}), rows, scratch)
    _write_link_table(a, packs, rows)
    _write_usd_table(rows, a, table)
    n_w = sum(1 for r in rows if r["written"])
    n_t = sum(
        1 for r in rows if any(x.get("st") is not None or x.get("color") is not None for x in r["links"].values())
    )
    n_spec = sum(1 for r in rows if "*" in r["links"])
    print(
        f"[RENDER_TEX_DONE] usd={len(tg)} written={n_w} textured={n_t} no_spec={n_spec} failed={len(failed)} "
        f"dry={int(a.dry_run)} out={a.out}",
        flush=True,
    )
    if not a.dry_run:  # the overlay list in force takes in what this run wrote (not a preview root)
        print(
            OM.record([r["written"] for r in rows if r["written"]])
            if oroot is None
            else f"[OVERLAY_MANIFEST] preview root {oroot} — no list written",
            flush=True,
        )
    if packs is not None and not a.dry_run:  # used objects minus covered overlays (rule tex_packs.cover)
        for ds in sorted(packs["rules"]):
            print(f"[OBJ_TEX_COVER] {RTP.cover_line(RTP.cover(ds, oroot))}", flush=True)
    ok = not failed and (a.dry_run or n_w + n_spec == len(tg))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
