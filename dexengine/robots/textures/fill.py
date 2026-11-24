"""Final stage of the object texture rules: the colour of objects without an image texture (material colour, uniform
colour when the raw has no texture or its alignment failed, per-object exceptions of pack datasets). The earlier stages
(mesh MTL map_Kd + UV, raw re-intake, texture packs) and the status words (OK_MTL_COLOR, OK_NO_TEX, OK_ALIGN_UNIFORM)
belong to dexengine/robots/textures/rules.py; per-object uniform colours belong to dexengine/robots/textures/colors.py
(object_color, COLOR_WORDS, KIND_COLORS). Readers: render_overlay._texture and tex_packs (lookup, cover).

Material colour (`mtl_color`): a material with a diffuse colour (Kd) but no image becomes a constant-colour material
(OK_MTL_COLOR, only without raw re-intake) only in datasets whose Kd is the original dataset colour
(MTL_COLOR_DATASETS). In other datasets a Kd-only MTL holds a converter fill value (ARCTIC parts and H2O: MeshLab
default 192/255; HRDexDB: trimesh default 102/255 or the texture-multiply identity 1.0), not a colour, so it keeps the
rule-4 status (no map_Kd).

Objects whose raw has no texture (`no_texture_status`): outside pack datasets, when the mesh (if the dataset has no raw
re-intake) or the raw OBJ has no UV, material or map_Kd, and no raw colour exists either (OakInk2 vertex colour or main
material Kd, loaded first by textures.raw_color through raw re-intake), the object gets its per-object uniform colour
object_color (OK_NO_TEX). A broken MTL link, a missing image on our side, several materials, RAW_NEEDED and
RAW_MISMATCH are not uniform colour (the raw may have a texture; they stay missing). MTL defects in original
distributions are corrected in rules.RAW_MTL_FIXES. An object whose raw texture exists but whose differing topology
fails the UV transfer gate (dexengine.robots.objects.uv_transfer) gets the uniform_rgb colour as OK_ALIGN_UNIFORM
(reason = residual, in the generator's table and markers). In pack datasets, per-object uniform colour comes only from
the named exception table UNIFORM_EXCEPTIONS (with a reason per entry).
"""

from __future__ import annotations

from dexengine.robots.textures import colors as OC
from dexengine.robots.textures.rules import NONE, PACK_POLICY, RAW_SOURCES, SRC_UNIFORM


def _rgb(text: str | None) -> tuple[float, float, float] | None:
    """MTL colour value (`r g b`) -> three floats in 0–1; None when not three values or out of range."""
    try:
        v = tuple(float(x) for x in (text or "").split())
    except ValueError:
        return None
    return v if len(v) == 3 and all(0.0 <= x <= 1.0 for x in v) else None


#: Datasets whose mesh MTL Kd (material without an image) is the original dataset colour; the only place for this
#: value. HOT3D: the intake OBJ/MTL carries the original asset's (GLB) material names (`Frames`, `PBR_.016` …) and base
#: colour (baseColorFactor) unchanged (trimesh to_simple writes the value as Kd). In other datasets a Kd-only MTL is a
#: converter fill value (see the module docstring).
MTL_COLOR_DATASETS = ("hot3d",)


def mtl_color(ds: str, tex: dict) -> tuple[float, float, float] | None:
    """Material colour from a find_texture result: only for MTL_COLOR_DATASETS, when rule 4 (no map_Kd) stopped it
    and Kd is a valid colour. The colour is the MTL value as is (no colour-space conversion; GLB value = Kd)."""
    if ds not in MTL_COLOR_DATASETS or tex.get("status") != f"{NONE}:no map_Kd":
        return None
    return _rgb(tex.get("kd"))


#: Rule reasons that mean "no texture" (rules 1, 2, 4), the same set for the mesh itself (no raw re-intake) and the
#: raw OBJ. The other reasons (broken MTL, several materials, missing image) may still have a texture; they are not
#: guessed from names and stay missing (objects with a confirmed distribution defect get a rules.RAW_MTL_FIXES entry).
NO_TEX = ("no UV", "no MTL", "no map_Kd")
#: Per-object uniform-colour exceptions of pack datasets: {dataset: {asset id: reason}}. Only objects for which the
#: policy source cannot be made. If such an object gets a pack, the bake stops with a PACK_MISMATCH status; then
#: delete the line. The reason text is recorded data (it reaches the overlay record through the pack origin).
UNIFORM_EXCEPTIONS: dict[str, dict[str, str]] = {
    "taco": {
        "130": "video projection impossible — the only sequence showing this object, 20231002_055, has no video "
        "(0/12 views · TACO coverage plan no_video)",
        "055": "video projection gate failed — 15 tries (stride 10 · 5) over the 8 sequences with video, own-label "
        "IoU 0.34–0.39 < 0.4 · target IoU in the same views 0.93–0.97 · the raw segmentation's tool label (3) "
        "is off the real ruler position (frames of 20231104_108)",
        "062": "video projection gate failed — in the only sequence with video, 20230923_008 at stride 5, own-label "
        "IoU 0.275 < 0.4 · target IoU 0.771 · the raw segmentation paints the spatula blade with the pan label (4)",
    },
}


def no_texture_status(ds: str, status: str) -> bool:
    """Whether rule result `status` means "the raw has no texture". Never for pack datasets (exception table only). For
    raw re-intake datasets only the raw OBJ's NONE reasons, otherwise the mesh's own."""
    if ds in PACK_POLICY:
        return False
    if ds in RAW_SOURCES:
        return status in {f"{NONE}:raw also {r}" for r in NO_TEX}
    return status in {f"{NONE}:{r}" for r in NO_TEX}


def object_policy(ds: str, oid: str) -> str | None:
    """Source policy of one object: uniform when in the exception table, else the dataset policy (PACK_POLICY/None)."""
    return SRC_UNIFORM if oid in UNIFORM_EXCEPTIONS.get(ds, {}) else PACK_POLICY.get(ds)


def uniform_rgb(ds: str, oid: str) -> tuple[float, float, float] | None:
    """Uniform colour allowed for one object: for exception-table objects, datasets outside the pack policy (raw has no
    texture, raw alignment failed) and uniform-policy datasets, the object's stated colour textures.colors.solid_color
    (configs/objects/solid_colors.tsv; ValueError when the object has no row); otherwise (video projection, rebake)
    None."""
    if oid in UNIFORM_EXCEPTIONS.get(ds, {}) or ds not in PACK_POLICY or PACK_POLICY[ds] == SRC_UNIFORM:
        return OC.solid_color(ds, oid)
    return None
