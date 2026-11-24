"""Texture-pack branch and cover check of the render overlay generator (render_overlay.py).

Pack lookup, the identity guard and source records run only with `--packs` (off: overlays and tables are unchanged).
The constant-colour material (uniform_material: uniform colour, material colour) and the cover check (cover) apply to
every dataset, packs or not. Rules, source kinds and policies belong to textures.rules (PACK_*); this module applies
them to the training USD and handles the uniform material, source records and CLI arguments.

Status and `why` values are recorded in overlays and tables and keep their exact text.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from dexengine.core import artifact_paths
from dexengine.core.errors import InputMissingError
from dexengine.robots.textures import colors as OC
from dexengine.robots.textures import fill as OTF
from dexengine.robots.textures import rules as OT

TP = artifact_paths


def add_args(ap) -> None:
    ap.add_argument(
        "--packs",
        action="store_true",
        help="load texture packs (dexengine.core.layout tex_packs layer; rules textures.rules.PACK_POLICY). "
        "Off by default",
    )
    ap.add_argument(
        "--pack-data-root",
        default="",
        help="read packs from the same template (data/<ds>/tex_packs) under this root instead of the data "
        "root (preview)",
    )
    ap.add_argument(
        "--overlay-root",
        default="",
        help="write overlays under this root (artifact_paths.overlay_file) instead of the canonical place "
        f"(preview); rendering takes the same value from env {TP.OVERLAY_ROOT_ENV}",
    )


def setup(a, dss: list[str]) -> dict | None:
    """Parse `--packs`; None when off. When on: per target dataset its rule (OT.pack_kind) and pack root
    (dexengine.core.layout tex_packs layer; a missing folder is fatal except for uniform colour), and the
    `[OBJ_TEX_PACKS]` marker. `--pack-data-root` without `--packs` is fatal (an input nobody consumes)."""
    if not a.packs:
        if a.pack_data_root:
            raise InputMissingError("[OBJ_TEX_PACKS] --pack-data-root needs --packs — nothing consumes it")
        return None
    rules, roots = {}, {}
    for ds in dss:
        rule = OT.pack_kind(ds)
        if rule is None:
            continue
        root = OT.pack_root(ds, a.pack_data_root or None)
        if rule["kind"] != OT.SRC_UNIFORM and not os.path.isdir(root):
            raise InputMissingError(f"[OBJ_TEX_PACKS] {ds} pack root missing: {root} (writer: textures/pack_write.py)")
        rules[ds], roots[ds] = rule, root
    print(
        f"[OBJ_TEX_PACKS] on · rules {json.dumps(rules, ensure_ascii=False)} · roots "
        f"{json.dumps(roots, ensure_ascii=False)} · not targets {[d for d in dss if d not in rules]}",
        flush=True,
    )
    return {"rules": rules, "roots": roots}


def lookup(packs: dict | None, ds: str, raw_id: str, part: str) -> dict | None:
    """Pack lookup result (in the generator's _texture shape) for a rigid visual mesh of a dataset with a pack rule;
    otherwise None."""
    rule = (packs or {}).get("rules", {}).get(ds)
    if rule is None or part != "visual.obj":
        return None
    exc = OTF.UNIFORM_EXCEPTIONS.get(ds, {}).get(raw_id)
    if exc is not None:  # per-object uniform exception; a pack for it = stale entry: block
        if os.path.isdir(os.path.join(packs["roots"][ds], raw_id)):
            return {
                "status": f"{OT.PACK_MISMATCH} · solid exception overlap",
                "image": None,
                "uv_obj": None,
                "why": f"UNIFORM_EXCEPTIONS[{ds}][{raw_id}] but a pack exists",
                "pack": {"status": "-"},
            }
        rule = {"kind": OT.SRC_UNIFORM, "origin": f"object_texture_fill.UNIFORM_EXCEPTIONS[{ds}] — {exc}"}
    if rule["kind"] == OT.SRC_UNIFORM:  # uniform colour is per object (textures.colors.object_color)
        rule = dict(rule, color=OTF.uniform_rgb(ds, raw_id))
    k = OT.find_pack(packs["roots"][ds], raw_id, rule)
    return {"status": k["status"], "image": k["image"], "uv_obj": None, "why": k["why"], "pack": k}


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def pack_link(x: dict, um: tuple | None) -> None:
    """Identity guard of a pack link: load it only when the training USD visual mesh the pack was made for (sha of
    points float32 and faceVertexIndices int32) equals the current training USD and st is finite (3F,2). Otherwise
    the status becomes PACK_MISMATCH + detail and no material is loaded."""
    k = x["pack"]
    if k.get("color") is not None:
        x["color"] = k["color"]
        return
    if not k.get("st_path"):
        return
    if um is None:
        x.update(status="mismatch:no visual prim", image=None)
        return
    pts, fvi = um
    m = k["meta"]
    if _sha(pts.astype(np.float32)) != m["sha256_usd_points"] or _sha(fvi.astype(np.int32)) != m["sha256_usd_fvi"]:
        x.update(status=f"{OT.PACK_MISMATCH} training USD identity", image=None)
        return
    st = np.load(k["st_path"]).astype(np.float32)
    if st.shape != (fvi.size, 2) or not np.isfinite(st).all():
        x.update(status=f"{OT.PACK_MISMATCH} st {st.shape} ≠ ({fvi.size}, 2)", image=None)
        return
    x["st"] = st


def source_rows(links: dict) -> dict:
    """Per-link source record (table columns, overlay customLayerData): tex_source, tex_model, gen_fraction, license.
    Links without a texture get `-`; mesh MTL and raw re-intake get gen_fraction 0 (USD dictionaries cannot hold
    None)."""
    out = {}
    for k, x in links.items():
        m = (x.get("pack") or {}).get("meta") or {}
        src = OT.tex_source_of(x["status"])
        on = src != "-"
        gf = m.get("gen_fraction") if on else None
        gf = 0.0 if gf is None and src in (OT.SRC_MESH, OT.SRC_RAW) else gf
        out[k] = {
            "tex_source": src,
            "tex_model": str(m.get("model", "-")) if on else "-",
            "gen_fraction": "-" if gf is None else float(gf),
            "license": f"{m.get('dataset_license', '-')} | {m.get('tool_license', '-')}" if on else "-",
        }
        x["source_row"] = out[k]
    return out


def uniform_material(stage, link: str, color) -> None:
    """Uniform-colour pack: bind a constant-colour UsdPreviewSurface (no image, no UV) to the visual mesh."""
    from pxr import Sdf, UsdShade

    mp = f"/Object/RenderLooks/{link}"
    mat = UsdShade.Material.Define(stage, mp)
    sh = UsdShade.Shader.Define(stage, f"{mp}/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(float(c) for c in color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.OverridePrim(f"/Object/{link}/visuals/mesh")).Bind(mat)


TABLE_EXTRA = ("tex_source", "tex_model", "gen_fraction", "license")


#: Overlay record key (customLayerData), written by render_overlay.write_overlay (name owner: artifact_paths).
REC_STATUS = artifact_paths.OVERLAY_TEX_KEY


def _link_covered(layer, odir: str, link: str, status: str, policy, color) -> str:
    """Whether one visual link of an overlay layer is really covered: "" = covered, else the reason. Checks structure,
    not only the recorded text: the status's source kind (tex_source_of) is a texture kind and, for policy datasets,
    that policy; the visual mesh prim has a material binding; the bound material prim is in the layer; for texture
    policies the DiffuseTex file exists next to the overlay; for the uniform policy diffuseColor equals the policy
    colour."""
    src = OT.tex_source_of(status)
    if src == "-":
        return f"{link}={status}"
    if policy is not None and src != policy:
        return f"{link} source {src} ≠ policy {policy}"
    mesh = layer.GetPrimAtPath(f"/Object/{link}/visuals/mesh")
    rel = mesh.relationships.get("material:binding") if mesh else None
    tgt = list(rel.targetPathList.GetAddedOrExplicitItems()) if rel else []
    if not tgt:
        return f"{link} no material binding"
    mat = layer.GetPrimAtPath(tgt[0])
    if not mat:
        return f"{link} bound material {tgt[0]} is not in the layer"
    if status == OT.OK_MTL_COLOR:  # material colour: constant-colour material (no image), colour in Surface
        sh = layer.GetPrimAtPath(tgt[0].AppendChild("Surface"))
        a = sh.attributes.get("inputs:diffuseColor") if sh else None
        got = a.default if a is not None else None
        return "" if got is not None and len(got) == 3 else f"{link} no material colour"
    if src == OT.SRC_UNIFORM:
        sh = layer.GetPrimAtPath(tgt[0].AppendChild("Surface"))
        a = sh.attributes.get("inputs:diffuseColor") if sh else None
        got = tuple(round(float(c), 6) for c in a.default) if a is not None and a.default is not None else None
        want = tuple(round(float(c), 6) for c in color) if color is not None else None
        return "" if got is not None and got == want else f"{link} uniform {got} ≠ policy colour {want}"
    tx = layer.GetPrimAtPath(tgt[0].AppendChild("DiffuseTex"))
    a = tx.attributes.get("inputs:file") if tx else None
    f = a.default.path if a is not None and a.default is not None else ""
    if not f or not os.path.isfile(os.path.join(odir, f)):
        return f"{link} texture file missing {f!r}"
    return ""


def cover(ds: str, overlay_root: str | None = None, used: dict | None = None, info: dict | None = None) -> dict:
    """Cover check: used objects minus covered overlays. Used set = textures.rules.used_objects(ds) (single owner;
    only tests pass `used`). For each used object, open training USD (artifact_paths usd_file) -> overlay
    (artifact_paths.overlay_file, or under overlay_root when given; never falls back to the canonical place); the object
    counts as covered when every link of the overlay record passes _link_covered. For the uniform policy the
    per-object colour material is the cover (no pack files). Datasets outside the pack policy (mesh MTL, raw
    re-intake; rules 1-5, RAW_SOURCES) count any texture status as covered, whatever the source (policy None). The same
    check serves all datasets.
    Returns {ds, policy, used, covered, missing: [(asset id, reason)], info}."""
    from pxr import Sdf

    policy = OT.PACK_POLICY.get(ds)
    info = {} if info is None else info
    if used is None:
        used = OT.used_objects(ds, info=info)
    t = TP.get(ds)
    miss: list[tuple[str, str]] = []
    unis: list[str] = []
    for aid, art in sorted(used):
        u = t.usd_file(aid, art)
        if not os.path.isfile(u):
            miss.append((aid, "no training USD"))
            continue
        o = TP.overlay_file(u, overlay_root)
        if not os.path.isfile(o):
            miss.append((aid, "no overlay"))
            continue
        layer = Sdf.Layer.OpenAsAnonymous(o)
        rec = dict((layer.customLayerData or {}).get(REC_STATUS) or {})
        if not rec:
            miss.append((aid, "no overlay record"))
            continue
        od = os.path.dirname(o)
        uni = any(OT.tex_source_of(v) == OT.SRC_UNIFORM for v in rec.values())
        pol = OTF.object_policy(ds, aid)
        col = OTF.uniform_rgb(ds, aid) if uni else None  # per-object uniform colour, only when a link is uniform
        if uni:
            unis.append(aid)
        why = [w for k, v in sorted(rec.items()) if (w := _link_covered(layer, od, k, v, pol, col))]
        if why:
            miss.append((aid, " · ".join(why)))
    for a, b, c in OC.color_collisions(ds, unis):  # different physical objects with one colour: not covered
        miss.append((b, f"colour clash {a} {c}"))
    return {
        "ds": ds,
        "policy": policy,
        "used": len(used),
        "covered": len(used) - len(miss),
        "missing": miss,
        "info": info,
    }


def cover_line(r: dict) -> str:
    """One line for a cover result (used by the check and the bake marker)."""
    import collections

    def _key(w: str) -> str:  # head of the first link reason without numbers (status prefix or reason)
        st = w.split(" · ")[0].split("=", 1)[-1]
        return st if ":" not in st else st.split(":")[0] + ":" + st.split(":")[1].split(" ")[0]

    why = collections.Counter(_key(w) for _a, w in r["missing"])
    i = r.get("info") or {}
    return (
        f"ds={r['ds']} policy={r['policy']} used={r['used']} covered={r['covered']} missing={len(r['missing'])}"
        f" reasons={dict(why)} e.g.={[a for a, _w in r['missing'][:12]]} ledgers={i.get('ledgers')}"
        f" adopted={i.get('clips')} hand={i.get('hand')}"
    )
