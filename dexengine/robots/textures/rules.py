"""Object mesh texture rules: the only rules the render overlay (dexengine/robots/objects/render_overlay.py) reads.

A texture is found by following the material the mesh itself names, never by guessing image file names (meshes from
different intake tools use `material_0.png`, `PBR_.012.png`, `Material.001.png`, ...).

Rules for one mesh (OBJ). Status words are recorded in overlays and tables:
  1. the OBJ has `vt` (UV)                                       else `none:no UV`
  2. the MTL named by `mtllib` exists next to the OBJ              else `none:no MTL` (`none:broken MTL` when the
                                                                   reference exists but the file does not)
  3. the OBJ uses exactly one material via `usemtl`                else `none:several materials` (several would need
                                                                   GeomSubsets, unsupported)
  4. that material has `map_Kd`                                    else `none:no map_Kd`
  5. the `map_Kd` image exists at its path relative to the MTL      else `none:no image`
  -> all pass: `applied` (image path). A `textured.png` next to the mesh that no material names is still `none`.

Raw re-intake (`raw_source`): datasets that lost their textures at intake (H2O, ARCTIC, OakInk2 supplementary objects)
apply the same rules to the original OBJ at the data_layout `raw` layer. The raw UV is used only when the training mesh
has the same face-corner positions (face and corner order; measured by the generator; an OakInk2 merged asset's raw
source is a part list -> atlas). When the topology differs (ARCTIC) the generator transfers the raw UV
(dexengine.robots.objects.uv_transfer; if it fails the alignment gate, uniform colour OK_ALIGN_UNIFORM). A raw source
missing on this machine is RAW_NEEDED; a raw OBJ that also fails rules 1-5 is NONE with the reason `raw also …`.
RAW_SOURCES encodes each dataset's distribution layout; defects in a distribution's own MTL are corrected only in the
named table RAW_MTL_FIXES.

Final stage (material colour, uniform colour when the raw has no texture, per-object exceptions):
dexengine/robots/textures/fill.py.

Texture packs (PACK_* below): datasets whose raw meshes have no UV or material (HO-Cap rebake, TACO video projection,
GRAB uniform colour) use prebuilt packs (uniform colour needs no pack: per-object colours, textures.colors), loaded only
with the generator's `--packs` (off by default). PACK_POLICY sets the allowed source kind per dataset.

This module runs without pxr or trimesh (rules only); mesh comparison and USD authoring are done by the generator.
"""

from __future__ import annotations

import os
import zipfile

#: Status prefixes, written into tables and overlay records.
OK = "applied"
OK_RAW = "applied(raw)"
NONE = "none"
RAW_NEEDED = "raw needed"
RAW_MISMATCH = "raw mismatch"
OK_MTL_COLOR = "applied(material color)"  # Kd of a material without an image (textures.fill)


def parse_obj(obj_path: str) -> dict:
    """OBJ header info: vt count, mtllib list, usemtl names (order kept, duplicates removed)."""
    nvt = 0
    libs: list[str] = []
    use: list[str] = []
    with open(obj_path, "rb") as f:
        for ln in f:
            if ln.startswith(b"vt "):
                nvt += 1
            elif ln.startswith(b"mtllib"):
                libs.append(ln.split(None, 1)[1].strip().decode("utf-8", "replace"))
            elif ln.startswith(b"usemtl"):
                name = ln.split(None, 1)[1].strip().decode("utf-8", "replace")
                if name not in use:
                    use.append(name)
    return {"n_vt": nvt, "mtllib": libs, "usemtl": use}


def parse_mtl(mtl_path: str) -> dict[str, dict[str, str]]:
    """MTL -> {material: {key (lower case): value string}}. A map_Kd value is the last token (path) after options."""
    out: dict[str, dict[str, str]] = {}
    cur = None
    with open(mtl_path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            t = ln.strip().split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "newmtl":
                cur = " ".join(t[1:])
                out[cur] = {}
            elif cur is not None and len(t) > 1:
                out[cur][t[0].lower()] = t[-1] if t[0].lower().startswith("map_") else " ".join(t[1:])
    return out


def find_texture(obj_path: str) -> dict:
    """Rules 1-5 -> {status, image (absolute path | None), n_vt, material, why}."""
    info = parse_obj(obj_path)
    d = os.path.dirname(os.path.abspath(obj_path))
    res = {"status": NONE, "image": None, "n_vt": info["n_vt"], "material": None, "why": ""}
    if info["n_vt"] == 0:
        res.update(status=f"{NONE}:no UV")
        return res
    if not info["mtllib"]:
        res.update(status=f"{NONE}:no MTL")
        return res
    mats: dict[str, dict[str, str]] = {}
    broken = []
    for lib in info["mtllib"]:
        p = os.path.normpath(os.path.join(d, lib))
        if os.path.isfile(p):
            for k, v in parse_mtl(p).items():
                mats[k] = dict(v, _mtl_dir=os.path.dirname(p))
        else:
            broken.append(lib)
    if not mats:
        res.update(status=f"{NONE}:broken MTL", why=f"referenced file {broken} missing")
        return res
    used = info["usemtl"] or list(mats)
    if len(used) != 1:
        res.update(status=f"{NONE}:several materials", why=f"usemtl {used[:4]}")
        return res
    m = mats.get(used[0])
    res["material"] = used[0]
    if m is None:
        res.update(status=f"{NONE}:broken MTL", why=f"material {used[0]!r} not in the MTL")
        return res
    if "map_kd" not in m:
        res.update(status=f"{NONE}:no map_Kd", kd=m.get("kd"))
        return res
    img = os.path.normpath(os.path.join(m["_mtl_dir"], m["map_kd"]))
    if not os.path.isfile(img):
        res.update(status=f"{NONE}:no image", why=m["map_kd"], mtl_dir=m["_mtl_dir"])
        return res
    res.update(status=OK, image=img)
    return res


# raw re-intake: each dataset's original distribution layout (raw place = data_layout `raw` layer)
def _h2o_raw(raw_root: str, obj_id: str, part: str, scratch: str, asset_dir=None) -> str | None:
    """H2O official `object.zip`: `object/<object>/<name>.obj` + MTL + images. Returns the extracted OBJ (or None)."""
    z = os.path.join(raw_root, "dl", "object.zip")
    if part != "visual.obj" or not os.path.isfile(z):
        return None
    with zipfile.ZipFile(z) as zf:
        mem = [n for n in zf.namelist() if n.startswith(f"object/{obj_id}/") and not n.endswith("/")]
        objs = [n for n in mem if n.lower().endswith(".obj")]
        if len(objs) != 1:
            return None
        out = os.path.join(scratch, "h2o", obj_id)
        os.makedirs(out, exist_ok=True)
        for n in mem:
            dst = os.path.join(out, os.path.basename(n))
            if not os.path.isfile(dst):
                with zf.open(n) as src, open(dst, "wb") as f:
                    f.write(src.read())
        return os.path.join(out, os.path.basename(objs[0]))


def _arctic_raw(raw_root: str, obj_id: str, part: str, scratch: str, asset_dir=None) -> str | None:
    """ARCTIC official `meta/object_vtemplates/<object>/mesh_tex.obj` (UV + material.jpg). Parts (top/bottom) use the
    same textured mesh as their source (the raw part OBJs have no UV)."""
    p = os.path.join(raw_root, "data", "arctic_data", "data", "meta", "object_vtemplates", obj_id, "mesh_tex.obj")
    return p if os.path.isfile(p) else None


def _oakink2_raw(
    raw_root: str, obj_id: str, part: str, scratch: str, asset_dir: str | None = None
) -> str | list | None:
    """OakInk2 official HF `align_ds/<rid_of(id)>/model.obj` (+ MTL, images): the original of a supplementary object's
    visual mesh (whose copy lacks the MTL). A merged asset (FIXED.json) returns a part list."""
    if part != "visual.obj":
        return None
    if asset_dir is None:
        raise ValueError(f"oakink2/{obj_id}: no asset folder (FIXED.json decides the path)")
    from dexengine.data.oakink2 import fixed_merge as FM
    from dexengine.data.oakink2 import parts as OP

    align = os.path.join(raw_root, "hf", "obj", "object_repair", "align_ds")
    parts = FM.raw_parts(asset_dir, align)
    if parts is not None:
        return parts
    p = os.path.join(align, OP.rid_of(obj_id), "model.obj")
    # NO_TEXTURE object -> OBJ derived from the raw scan colour (vertex colour, material Kd), tried before name colours
    from dexengine.robots.textures import raw_color as ORC

    return ORC.for_repair(p, raw_root, OP.rid_of(obj_id), scratch) if os.path.isfile(p) else None


#: Datasets that lost textures at intake -> where their original OBJ lives. Other datasets have no raw re-intake.
RAW_SOURCES = {"h2o": _h2o_raw, "arctic": _arctic_raw, "oakink2": _oakink2_raw}
#: Corrections of MTL defects in original distributions: {dataset: {object id: (MTL map_Kd, actual image in the
#: distribution folder, reason)}}. The only place for such corrections (objects whose distribution folder was checked).
#: The reason text is recorded data. A stale entry makes raw_source fail (then delete the line): the raw MTL no longer
#: names that file, the named image now exists, or the actual image is missing.
RAW_MTL_FIXES: dict[str, dict[str, tuple[str, str, str]]] = {
    "h2o": {
        "cocoa": (
            "cocoa.png",
            "ovomaltine.png",
            "map_Kd of the official object.zip object/cocoa/ovomaltine.mtl is cocoa.png, "
            "which is not in the distribution; the only image in that folder is ovomaltine.png",
        )
    },
}


def _mtl_fix(t: dict, fix: tuple[str, str, str], where: str) -> dict:
    """Apply one RAW_MTL_FIXES entry to the raw OBJ's rule result: only when rule 5 stopped on that missing image,
    otherwise raise."""
    bad, real, why = fix
    img = os.path.join(t.get("mtl_dir") or "", real)
    if t["status"] != f"{NONE}:no image" or t["why"] != bad or not os.path.isfile(img):
        raise ValueError(f"RAW_MTL_FIXES {where} ({bad} → {real}) is stale — rule result {t['status']} {t['why']!r}")
    return dict(t, status=OK, image=img, why=f"raw MTL corrected {bad} → {real} (RAW_MTL_FIXES): {why}")


def raw_source(
    ds: str, raw_root: str, obj_id: str, part: str, scratch: str, asset_dir: str | None = None
) -> dict | None:
    """Raw re-intake candidate. None = this dataset has no raw re-intake. Otherwise {status, raw_obj, image, …}:
    RAW_NEEDED when the raw file is missing; when the raw OBJ passes rules 1-5 (+ RAW_MTL_FIXES) the status is the
    OK_RAW candidate (the generator does the alignment check)."""
    fn = RAW_SOURCES.get(ds)
    if fn is None:
        return None
    p = fn(raw_root, obj_id, part, scratch, asset_dir)
    fix = RAW_MTL_FIXES.get(ds, {}).get(obj_id)
    if isinstance(p, list):  # merged asset: rules 1-5 per part
        if fix is not None:
            raise ValueError(f"RAW_MTL_FIXES {ds}/{obj_id}: asset is a part list")
        miss = [rid for q, rid, _ in p if q is None]
        if miss:
            return {"status": f"{RAW_NEEDED}:parts {','.join(miss)}", "raw_obj": None, "image": None, "why": ""}
        ts = [find_texture(q) for q, _, _ in p]
        bad = [t["status"] for t in ts if t["status"] != OK]
        if bad:
            return {"status": f"{NONE}:raw also {bad[0].split(':', 1)[-1]}", "raw_obj": None, "image": None, "why": ""}
        return {
            "status": OK_RAW,
            "raw_obj": [(q, pose) for q, _, pose in p],
            "image": [t["image"] for t in ts],
            "why": f"{len(p)}-part atlas",
        }
    if p is None:
        if fix is not None:
            raise ValueError(
                f"RAW_MTL_FIXES {ds}/{obj_id} exists but the raw source is missing ({raw_root}) — the "
                "correction cannot be checked"
            )
        return {"status": RAW_NEEDED, "raw_obj": None, "image": None, "why": f"{ds} raw missing ({raw_root})"}
    t = find_texture(p)
    if fix is not None:
        t = _mtl_fix(t, fix, f"{ds}/{obj_id}")
    if t["status"] != OK:  # the raw has no texture either (not a mismatch)
        return {
            "status": f"{NONE}:raw also {t['status'].split(':', 1)[-1]}",
            "raw_obj": p,
            "image": None,
            "why": f"raw {p}",
        }
    return {"status": OK_RAW, "raw_obj": p, "image": t["image"], "why": t["why"]}


# texture packs: render textures of datasets whose raw meshes have no UV or material (HO-Cap, TACO, GRAB)
# Pack = `<data_layout tex_packs layer>/<object id>/{tex.png, st.npy, meta.json}`: face-vertex UVs (st, float32 (3F,2),
# faceVarying) laid out in the training USD visual mesh's face order, plus the image. meta.tex_source states how it was
# made; a pack whose kind differs from the dataset's allowed kind (PACK_POLICY) is not loaded. Only the generator's
# `--packs` reads packs (off by default; when off, overlays and tables are byte-identical).
SRC_MESH = "mesh_mtl"  # the mesh's own MTL map_Kd (rules 1-5)
SRC_RAW = "original_reintake"  # original distribution texture (H2O, ARCTIC = raw_source; HO-Cap = rebake pack)
SRC_VIDEO = "video_projection"  # original multi-view video projected onto our UV (TACO: 12 views, pose, masks)
SRC_GEN = "generated"  # generative model (name / photo conditioned); may not match the real colour
SRC_UNIFORM = "uniform"  # uniform colour (value per object: textures.colors.object_color)
TEX_SOURCES = (SRC_MESH, SRC_RAW, SRC_VIDEO, SRC_GEN, SRC_UNIFORM)
#: Pack status words (tables, overlay records).
OK_PACK = {
    SRC_RAW: "applied(raw·rebake)",
    SRC_VIDEO: "applied(raw video projection)",
    SRC_GEN: "applied(generated)",
    SRC_UNIFORM: "applied(solid)",
}
PACK_NONE = f"{NONE}:no pack"
PACK_MISMATCH = "mismatch:pack"
PACK_LAYER = "tex_packs"
PACK_FILES = ("tex.png", "st.npy", "meta.json")
#: Required meta.json fields (writer: dexengine/robots/textures/pack_write.py). sha256_usd_* = bytes of the points
#: (float32) and faceVertexIndices (int32) of the training USD visual mesh the pack was made for; the reader compares
#: them with the current training USD (identity guard).
PACK_META_KEYS = (
    "tex_source",
    "model",
    "source",
    "gen_fraction",
    "dataset_license",
    "tool_license",
    "sha256_tex",
    "sha256_st",
    "sha256_usd_points",
    "sha256_usd_fvi",
)
#: Dataset -> pack source kind. Datasets not listed are not pack datasets; a new dataset that needs packs is added here
#: explicitly (no default). GRAB is uniform colour (no video or photo source; a grey reference beat generated textures).
PACK_POLICY: dict[str, str] = {"hocap": SRC_RAW, "taco": SRC_VIDEO, "grab": SRC_UNIFORM}
#: Uniform colours are per object; rules and tables live in dexengine/robots/textures/colors.py.
OK_NO_TEX = "applied(solid · no raw texture)"
OK_ALIGN_UNIFORM = (
    "applied(solid · raw alignment failed)"  # raw texture exists but the UV transfer gate (raw_uv_transfer) failed
)


def pack_root(ds: str, data_root: str | None = None) -> str:
    """Pack root (shared by writer and reader): the data_layout tex_packs layer under the data root. With data_root
    (preview staging, caller passes `--pack-data-root`) the same template under that root."""
    from dexengine.core import paths as P

    rel = P.data_layout.render(PACK_LAYER, ds)
    return os.path.join(os.path.abspath(data_root), rel) if data_root else P.data_layout.data_abspath(rel)


def pack_kind(ds: str) -> dict | None:
    """Pack rule of a dataset: None = not a pack dataset, else {"kind"[, "origin"]}. For the uniform policy the colour
    is per object (textures.colors.object_color); the lookup (tex_packs.lookup) fills it from the object id."""
    k = PACK_POLICY.get(ds)
    if k is None:
        return None
    if k != SRC_UNIFORM:
        return {"kind": k}
    return {"kind": k, "origin": "object_colors.object_color"}


def _sha_file(p: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_pack(pack_root: str, obj_id: str, rule: dict) -> dict:
    """One pack -> {status, image, st_path, meta, color, why}. A uniform rule needs no pack, only a colour. Missing pack
    files -> PACK_NONE (this object has no pack yet). Missing meta fields, a different source kind or a sha
    mismatch -> PACK_MISMATCH + detail. The generator, which reads the USD, checks sha256_usd_*."""
    res = {"status": PACK_NONE, "image": None, "st_path": None, "meta": None, "color": None, "why": ""}
    if rule["kind"] == SRC_UNIFORM:
        how = f"uniform rgb={tuple(rule['color'])} ({rule.get('origin', '?')})"  # tex_model column: colour, origin
        res.update(
            status=OK_PACK[SRC_UNIFORM],
            color=rule["color"],
            meta={
                "tex_source": SRC_UNIFORM,
                "model": how,
                "gen_fraction": 0.0,
                "dataset_license": "-",
                "tool_license": "-",
                "source": how,
            },
        )
        return res
    d = os.path.join(pack_root, obj_id)
    if not all(os.path.isfile(os.path.join(d, f)) for f in PACK_FILES):
        res["why"] = d
        return res
    import json

    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    miss = [k for k in PACK_META_KEYS if k not in meta]
    if miss:
        res.update(status=f"{PACK_MISMATCH} meta field missing", why=",".join(miss))
        return res
    if meta["tex_source"] != rule["kind"]:
        res.update(status=f"{PACK_MISMATCH} source kind", why=f"{meta['tex_source']} ≠ {rule['kind']}")
        return res
    for f, k in (("tex.png", "sha256_tex"), ("st.npy", "sha256_st")):
        if _sha_file(os.path.join(d, f)) != meta[k]:
            res.update(status=f"{PACK_MISMATCH} sha", why=f)
            return res
    res.update(
        status=OK_PACK[rule["kind"]], image=os.path.join(d, "tex.png"), st_path=os.path.join(d, "st.npy"), meta=meta
    )
    return res


#: Ledger status that marks objects used in rendering (the texture target set): adopted rows of the folded ledger.
USED_STATUS = "ADOPTED"


def used_objects(
    ds: str, hand: str | None = None, arm: str | None = None, tree_ti=None, info: dict | None = None
) -> dict:
    """All objects used in rendering for dataset `ds` -> {(asset id, is articulated): {roles …}}. The single owner of
    the texture target set: the bake (`[OBJ_TEX_COVER]`), the pack writer (`[TEX_PACK_COVER]`) and the cover check
    (tex_packs.cover) all read it.

    Source = the folded ledgers of hand `hand` (datasets.ledger_files) whose dataset is ds. For each USED_STATUS clip,
    every entry of task_info.objects[] in that axis's built tree (paths.layer_path "trees", combo = axis variant): main
    object, supports and extras alike (not filtered by role). Asset id = artifact_paths.object_asset_id(obj_id,
    articulation). `hand` defaults to the canonical axis hand (datasets.AXIS_HAND): textures are shared across hands,
    so the target set is defined by the axis hand's ledgers. `arm` defaults to the only registered arm
    (paths.ARM_IDS) and raises when there are several. No ledger, an adopted clip without a tree task_info, or an
    empty objects list -> raise.
    `tree_ti(ds, hand, arm, axis, clip)` reads a tree task_info (tests swap it; the default uses the path above).
    `info` (dict), when given, receives the provenance: hand, arm, ledgers [(file name, sha256[:12])], clips (adopted
    clip count), so two machines can confirm they read the same ledgers."""
    import json

    from dexengine.core import paths as P
    from dexengine.data import rules as RR
    from dexengine.placement import final_ledger as FL

    hand = hand or RR.AXIS_HAND
    if arm is None:
        if len(P.ARM_IDS) != 1:
            raise ValueError(f"registered arms are {P.ARM_IDS} — pass arm= for the used-object set")
        arm = P.ARM_IDS[0]
    files = [
        f for f in RR.ledger_files("*", hand, root=P.main_root()) if RR.ledger_axis_parts(RR.ledger_axis(f))[0] == ds
    ]
    if not files:
        raise ValueError(f"hand {hand} has no folded ledger for dataset {ds} — cannot determine the used objects")

    def _tree_ti(ds_, hand_, arm_, axis, clip):
        _d, emb, _v = RR.ledger_axis_parts(axis)
        combo = P.combo_id(arm_, hand_, RR.ledger_axis_variant(axis))
        f = P.layer_path("trees", ds_, hand=hand_, emb=emb, combo=combo) / P.tree_name(clip) / "task_info.json"
        if not f.is_file():
            return None
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)

    read = tree_ti or _tree_ti
    from dexengine.core import artifact_paths as tp  # noqa: PLC0415

    out: dict[tuple[str, bool], set[str]] = {}
    nclip = 0
    for f in files:
        axis = RR.ledger_axis(f)
        for r in FL.read_final_rows(str(f)):
            if r["status"] != USED_STATUS:
                continue
            nclip += 1
            ti = read(ds, hand, arm, axis, r["clip"])
            if ti is None:
                raise ValueError(f"{axis} adopted clip {r['clip']} has no tree task_info — used objects unknown")
            objs = ti.get("objects") or []
            if not objs:
                raise ValueError(f"{axis} adopted clip {r['clip']}: tree task_info.objects is empty")
            for o in objs:
                art = o.get("articulation")
                key = (tp.object_asset_id(o["obj_id"], art), bool(art))
                out.setdefault(key, set()).add(str(o.get("role") or "-"))
    if info is not None:
        info.update(hand=hand, arm=arm, clips=nclip, ledgers=[(f.name, _sha_file(str(f))[:12]) for f in files])
    return out


def tex_source_of(status: str) -> str:
    """Status -> source kind (table column); `-` for statuses that carry no texture."""
    if status in (OK, OK_MTL_COLOR):
        return SRC_MESH
    if status == OK_RAW:
        return SRC_RAW
    if status in (OK_NO_TEX, OK_ALIGN_UNIFORM):
        return SRC_UNIFORM
    for k, v in OK_PACK.items():
        if status == v:
            return k
    return "-"
