"""Random-look allow rule (looks.augvis): allowed maps a scene to the options each axis may take, the tints and the
exclusion reasons, with the material families (family, FAMILY_RULES), the desk brightness prediction (albedo,
predict_desk_rgb, _desk_luma) and the tint table (tints). The value tables stay in looks.augvis.catalog."""

from __future__ import annotations

from dexengine.core.errors import InputMissingError
from dexengine.record.looks.augvis.catalog import (
    BIG_M,
    CONTRAST_MIN_DE,
    CONTRAST_REL,
    LIGHT_K,
    LIGHT_SCALE,
    MARK,
    MAX_CHROMA,
    MIN_LUM,
    PANEL_PATTERNS,
    SATS,
    TINT_HUES,
    TINT_NATURAL,
    TINT_SATS,
    distractors,
    materials,
)
from dexengine.record.looks.augvis.color import chroma, delta_e, lin, luma, srgb, tint_rgb

# material families: families are rotated first so looks of the same clip get different families. Rule = the first
# pattern found in the words (lower case) of the material key, id and cat (order matters: painted before wood). A
# material in no family dies (new materials change the rule).
FAMILY_RULES = {
    "desk": (
        ("solid", ("painted", "plastic", "laminate", "paint")),
        ("wood", ("wood", "veneer", "oak", "ash", "walnut", "cherry", "bamboo", "birch", "chipboard")),
        ("stone", ("marble", "terrazzo", "granite", "concrete", "stone", "tile")),
        ("soft", ("fabric", "leather", "cork", "felt", "carpet")),
    ),
    "floor": (
        ("wood", ("woodfloor", "plank", "parquet", "laminate", "wood")),
        ("stone", ("tile", "terrazzo", "marble", "concrete", "cartago")),
        ("soft", ("carpet", "tatami", "rubber")),
    ),
    "wall": (
        ("masonry", ("brick", "tile")),
        ("wood_paper", ("plank", "wallpaper")),
        ("plaster", ("plaster", "paint", "beige")),
    ),
}


def family(axis: str, key: str, entry: dict) -> str:
    """Material family (FAMILY_RULES): the first pattern in the key, id and cat words. None dies."""
    text = " ".join(str(x).lower() for x in (key, entry.get("id", ""), entry.get("cat", "")))
    for fam, pats in FAMILY_RULES[axis]:
        if any(p in text for p in pats):
            return fam
    raise SystemExit(f"{MARK} unknown material family: {axis} {key}: add a rule to FAMILY_RULES")


def tints(entry: dict) -> list[tuple[int, float]]:
    """Tint palette of one material: paint class = neutral + 12 hues x TINT_SATS, natural = TINT_NATURAL, D = neutral
    only."""
    if entry["kind"] == "look_d":
        return [(0, 0.0)]
    if entry.get("tint_class") == "paint":
        return [(0, 0.0)] + [(h, sv) for h in TINT_HUES for sv in TINT_SATS]
    return list(TINT_NATURAL)


def albedo(entry: dict, manifest: dict, look_d_albedo, tint=(0, 0.0)) -> list[float]:
    """Mean albedo of one material (sRGB 0-1): a texture is the received diffuse mean (manifest, sRGB) x fixed tint x
    tint (linear multiply); a flat colour is rgb (OmniPBR colour constants are **linear** values) x tint; D is the D
    texture mean."""
    if entry["kind"] == "look_d":
        return list(look_d_albedo)
    t2 = tint_rgb(tint)
    if entry["kind"] == "solid":
        return [srgb(entry["rgb"][i] * t2[i]) for i in range(3)]
    m = manifest["textures"][entry["id"]]["mean_srgb"]
    t = entry.get("tint", (1.0, 1.0, 1.0))
    return [srgb(lin(m[i]) * t[i] * t2[i]) for i in range(3)]


def predict_desk_rgb(obs_d_desk_rgb, alb_d, alb_new) -> list[float]:
    """The top colour measured in the D render (sRGB), made linear, x (new albedo / D albedo, linear), back to sRGB
    (under the same light and exposure linear pixel values ∝ albedo; tonemapper compression ignored; the ΔE measured
    after rendering checks it)."""
    return [min(1.0, srgb(lin(obs_d_desk_rgb[i]) * lin(alb_new[i]) / max(lin(alb_d[i]), 1e-4))) for i in range(3)]


# ── scene rules · sampler ──
def words_of(names) -> set[str]:
    out = set()
    for n in names:
        for w in str(n).lower().replace("-", "_").replace(".", "_").split("_"):
            if len(w) >= 3 and not w.isdigit():
                out.add(w)
    return out


def _desk_options(tn, tabs, bright, obj, dd, alb_d, manifest, lim):
    """Desk options: dark or low-contrast desks (to the object) and tint pairs are excluded."""
    tn["desk"], desk_x, n_pair_x, n_dark = {}, {}, 0, 0
    for k, e in tabs["desk"].items():
        ok = []
        for t in tints(e):
            if not bright("desk", k, e, t):
                n_dark += 1
                continue
            if obj is None or dd is None or k == "D_birch":  # D always stays (the original render itself)
                ok.append(t)
                continue
            de = delta_e(obj, predict_desk_rgb(dd, alb_d, albedo(e, manifest, alb_d, t)))
            if de >= lim:
                ok.append(t)
            else:
                n_pair_x += 1
        if ok:
            tn["desk"][k] = ok
        else:
            desk_x[k] = (
                f"every tint has object contrast < {lim:.1f} · luma < {MIN_LUM['desk']} · chroma > {MAX_CHROMA['desk']}"
            )
    return desk_x, n_pair_x, n_dark


def _distractor_options(scene, dis, manifest, out, tn):
    """Distractors: names that match the clip or object words are excluded; big ones listed."""
    bad = words_of([scene["clip"], *scene.get("object_names", [])])
    dis_ok, dis_x = [], {}
    for k, e in dis.items():
        m = manifest["models"].get(e["id"])
        if m is None:
            dis_x[k] = "not in the receipt list"
        elif m.get("excluded"):
            dis_x[k] = m["excluded"]
        elif bad & set(e["words"]):
            dis_x[k] = f"name overlap {sorted(bad & set(e['words']))}"
        else:
            dis_ok.append(k)
    out["desk"], out["distractors"] = sorted(tn["desk"]), sorted(dis_ok)
    big = []
    for k in dis_ok:
        dims = manifest["models"][dis[k]["id"]].get("dims_mm")
        if not dims:
            raise InputMissingError(
                f"{MARK} distractor {k} has no size (dims_mm): receipt (augvis.assets) must measure it"
            )
        if max(float(x) for x in dims) / 1000.0 >= BIG_M:
            big.append(k)
    out["distractors_big"] = sorted(big)
    return dis_x


def allowed(scene: dict, manifest: dict) -> dict:
    """Scene -> allowed options per axis, tints (tints[axis][material]) and exclusions (reasons). scene = dict(aerial,
    clip, object_names, object_rgb (measured in the D render | None), desk_rgb_d (measured top in the D render | None),
    look_d_albedo{desk, floor, wall}). Exclusion reasons are data (render_clip.json looks)."""
    tabs = materials(manifest)
    excl = {}
    look = scene["look_d_albedo"]

    def bright(ax, k, e, t):
        """Brightness floor and chroma cap (MIN_LUM · MAX_CHROMA); look D materials always stay."""
        if k.startswith("D_"):
            return True
        rgb = albedo(e, manifest, look[ax], t)
        return luma(rgb) >= MIN_LUM[ax] and chroma(rgb) <= MAX_CHROMA[ax]

    tn = {
        ax: {k: [t for t in tints(e) if bright(ax, k, e, t)] for k, e in tabs[ax].items()} for ax in ("floor", "wall")
    }
    excl["dark"] = {ax: sorted(k for k, v in tn[ax].items() if not v) for ax in ("floor", "wall")}
    tn = {ax: {k: v for k, v in tn[ax].items() if v} for ax in tn}
    out = {
        "floor": sorted(tn["floor"]),
        "wall": sorted(tn["wall"]),
        "light_scale": list(LIGHT_SCALE),
        "light_k": list(LIGHT_K),
        "panels": sorted(PANEL_PATTERNS),
        "dome": ["none"] + sorted(manifest.get("hdris") or {}),
        "sat": list(SATS),
    }
    dis = distractors(manifest)
    if scene["aerial"]:
        out["desk"], out["distractors"], out["distractors_big"] = ["D_birch"], [], []
        tn["desk"] = {"D_birch": [(0, 0.0)]}
        excl["desk"] = {k: "aerial scene — no desk" for k in tabs["desk"] if k != "D_birch"}
        excl["distractors"] = dict.fromkeys(dis, "aerial scene — no desk")
        return {
            "allowed": out,
            "tints": tn,
            "excluded": excl,
            "families": _families(out, tabs),
            "desk_luma": _desk_luma(out, tn, tabs, manifest, look),
        }
    obj, dd = scene.get("object_rgb"), scene.get("desk_rgb_d")
    alb_d = scene["look_d_albedo"]["desk"]
    lim = min(CONTRAST_MIN_DE, CONTRAST_REL * delta_e(obj, dd)) if (obj is not None and dd is not None) else 0.0
    desk_x, n_pair_x, n_dark = _desk_options(tn, tabs, bright, obj, dd, alb_d, manifest, lim)
    dis_x = _distractor_options(scene, dis, manifest, out, tn)
    excl["desk"], excl["distractors"] = desk_x, dis_x
    excl["desk_tint_pairs"] = n_pair_x
    excl["desk_dark_pairs"] = n_dark
    return {
        "allowed": out,
        "tints": tn,
        "excluded": excl,
        "contrast_limit": round(lim, 2),
        "families": _families(out, tabs),
        "desk_luma": _desk_luma(out, tn, tabs, manifest, look),
    }


def _desk_luma(out: dict, tn: dict, tabs: dict, manifest: dict, look: dict) -> dict:
    """Albedo brightness per allowed (top, tint), read by the intensity pairing (sample_set). Key =
    '<material>|<hue>,<saturation>'."""
    return {
        f"{k}|{t[0]},{t[1]}": round(luma(albedo(tabs["desk"][k], manifest, look["desk"], t)), 4)
        for k in out["desk"]
        for t in tn["desk"][k]
    }


def _families(out: dict, tabs: dict) -> dict:
    """Family per allowed material (FAMILY_RULES); unknown materials die here."""
    return {ax: {k: family(ax, k, tabs[ax][k]) for k in out[ax]} for ax in ("desk", "floor", "wall")}
