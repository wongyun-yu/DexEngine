"""Visual augment of the record render (experiments only, not the default path): single owner of the variation axes,
options, scene rules and sampler values (standard library + look_d only, so the sampler can be tested on a sim CPU).
Asset sources, licences and choice rules are in augvis.sources; their result (chosen materials, models, HDRIs) is
recorded by the asset manifest, and materials and distractors here read it.

Values change on top of look D (look_d: closed room, 9 ceiling LED panels, fixed exposure, concrete floor, white
walls, wooden top):
  desk · floor · wall   material (Poly Haven table + the manifest's ambientCG) x tint (a palette per material class);
                         brightness floor MIN_LUM · chroma cap MAX_CHROMA
  light_scale · light_k  panel intensity scale · colour temperature 2,700-7,500 K (warm, neutral, cool groups
                         LIGHT_K_GROUPS)
  panels                 lit panel pattern (light direction; corrected so the top illuminance equals look D) · per-panel
                         intensity jitter ±PANEL_JITTER
  dome                   indoor HDRI dome light (the canonical dome prim; the room walls and ceiling cast no shadow, so
                         the dome light comes in) or none
  sat                    camera tonemapper saturation (look D 0.7)
  distractor             distractors on the desk (visual only): Poly Haven models + the manifest's Google Scanned
                         Objects
  The camera white balance follows the colour temperature by WB_ADAPT.

Scene rules:
  aerial scene       no desk or distractor axis.
  object vs top      the measured object colour of the D render and the candidate top colour predicted as the D render
  contrast           top colour x (candidate albedo x tint / D albedo): when their ΔE76 is below min(CONTRAST_MIN_DE,
                     CONTRAST_REL x D contrast) that (top, tint) pair is dropped (allowed filters per pair).
  distractor names   distractors sharing a word with the manipulated objects or the clip name are dropped.
The sampler (sample_set, card rotation) is owned by augvis.sample, which reads the values and rules here.
"""

from __future__ import annotations

import math

from dexengine.record.looks import look_d as LOOK  # owner of the look D values (imports the standard library only)
from dexengine.record.looks.augvis import sources as SR  # asset choice rules (size floor GSO_MIN_DIM; stdlib only)

MARK = "[AUG_VIS]"
# object vs top colour difference (ΔE76) threshold: a predicted value below min(this, CONTRAST_REL x look D contrast)
# drops that (top, tint)
CONTRAST_MIN_DE = 12.0
# scenes whose look D contrast is already low (grey objects, TACO) drop only tops that hide more than D
CONTRAST_REL = 0.8
# share of the panel colour temperature the camera WB follows (0 = fixed · 1 = full correction). 0.6 removes 60 % of
# the temperature difference and every random look looks neutral; with 0.3, 2,700 K stays warm and 7,500 K cool.
WB_ADAPT = 0.3
D_KELVIN = float(LOOK.PANEL_K)  # look D panel colour temperature (value owner look_d)
SAT_KEY = "/rtx/post/tonemap/irayReinhard/saturation"
D_SAT = float(LOOK.TONEMAP[SAT_KEY])  # look D tonemapper saturation (value owner look_d)

TINT_HUES = tuple(range(0, 360, 30))  # 12 hues
TINT_SATS = (0.12, 0.25, 0.4)  # saturation (HSV, a natural range: no primary-colour paint)
# (hue, saturation): neutral, warm, cool, red, green + stains (honey, red-brown, amber, blue-grey). The old five all
# had saturation <= 0.08, so wood, stone and tile materials (tops 24/38, floors 37/45) were effectively achromatic;
# stains also survive only within MIN_LUM and MAX_CHROMA.
TINT_NATURAL = (
    (0, 0.0),
    (30, 0.08),
    (210, 0.08),
    (0, 0.06),
    (100, 0.05),
    (25, 0.15),
    (12, 0.25),
    (38, 0.22),
    (215, 0.14),
)
# materials: key -> dict(kind, id (Poly Haven), res, tile_m (one tile, measured), rough (without texture), tint, rgb)
#   kind  look_d = the look D MDL as it is · tex = Poly Haven texture (OmniPBR, world projection) · solid = flat
#         colour (OmniPBR)
DESK = {
    "D_birch": {"kind": "look_d"},
    "oak": {"kind": "tex", "id": "oak_veneer_01", "res": "2k"},
    "ash": {"kind": "tex", "id": "ash_veneer", "res": "2k"},
    "walnut": {"kind": "tex", "id": "american_walnut_veneer", "res": "2k"},
    "black_walnut": {"kind": "tex", "id": "black_walnut_veneer_01", "res": "2k"},
    "cherry": {"kind": "tex", "id": "cherry_veneer", "res": "2k"},
    "grey_oak": {"kind": "tex", "id": "grey_oak_veneer_01", "res": "2k"},
    "bamboo": {"kind": "tex", "id": "bamboo_veneer", "res": "2k"},
    "kitchen_wood": {"kind": "tex", "id": "kitchen_wood", "res": "2k"},
    "laminate_white": {"kind": "solid", "tint_class": "paint", "rgb": (0.80, 0.80, 0.78), "rough": 0.45},
    "laminate_grey": {"kind": "solid", "tint_class": "paint", "rgb": (0.42, 0.43, 0.44), "rough": 0.5},
    "laminate_dark": {"kind": "solid", "tint_class": "paint", "rgb": (0.12, 0.12, 0.13), "rough": 0.45},
}
FLOOR = {
    "D_concrete": {"kind": "look_d"},
    "concrete_2": {"kind": "tex", "id": "concrete_floor_02", "res": "1k"},
    "smooth_concrete": {"kind": "tex", "id": "smooth_concrete_floor", "res": "1k"},
    "laminate": {"kind": "tex", "id": "laminate_floor_02", "res": "1k"},
    "parquet": {"kind": "tex", "id": "herringbone_parquet", "res": "1k"},
    "planks": {"kind": "tex", "id": "plank_flooring", "res": "1k"},
    "tiles": {"kind": "tex", "id": "floor_tiles_06", "res": "1k"},
    "terrazzo": {"kind": "tex", "id": "terrazzo_tiles", "res": "1k"},
    "cartago_tile": {"kind": "tex", "id": "grey_cartago_01", "res": "1k"},
    "carpet": {"kind": "tex", "id": "dirty_carpet", "res": "1k", "tint_class": "paint"},
}
WALL = {
    "D_plaster": {"kind": "look_d"},
    "white_plaster": {"kind": "tex", "id": "white_plaster_02", "res": "1k", "tint_class": "paint"},
    "beige": {"kind": "tex", "id": "beige_wall_001", "res": "1k", "tint_class": "paint"},
    "grey_plaster": {"kind": "tex", "id": "grey_plaster", "res": "1k", "tint_class": "paint"},
    "painted_brick": {"kind": "tex", "id": "painted_brick", "res": "1k", "tint_class": "paint"},
}
# light: the look D head camera mean brightness 0.31-0.42 (pilot measurement, 'dark') is moved toward the target
# 0.40-0.55 by weighting the bright side
# flat object colour looks (default off; only when dense_render DENSE_OBJ_COLORS turns them on): every task object
#   gets one OmniPBR of this albedo (instead of its original cover texture; robot, desk and room keep that slot's
#   look). This table alone owns the values; name = the look name tail.
OBJ_COLORS = {"red": (0.9, 0.05, 0.05), "green": (0.05, 0.8, 0.05), "blue": (0.05, 0.1, 0.9)}
# print looks (default off; only when dense_render DENSE_PRINT_LOOKS turns them on): the real kit prints every object
#   in green 3D print -> one kind per slot of the 16-slot deck (r0_s01 … r3_s16, columns 0-15) = green 12, blue 2,
#   original texture 2 (per row green 3 + blue|texture 1, spread evenly over rows). Room, light and distractors =
#   that slot as it is. noise = a whole-object value drawn per episode from a seed (not pixel noise; the same filament
#   gives the objects the same colour). This table and print_look own the values; look name tails are p… (no clash
#   with v1's r0_s02_green and r0_s03_blue).
PRINT_LOOK_KINDS = (
    "green",
    "green",
    "green",
    "blue",
    "green",
    "green",
    "green",
    "tex",
    "green",
    "green",
    "green",
    "blue",
    "green",
    "green",
    "green",
    "tex",
)
PRINT_LOOK_SUFFIX = {"green": "pgreen", "blue": "pblue", "tex": "ptex"}
PRINT_NOISE = {"hue_deg": 10.0, "sat": (0.8, 1.2), "val": (0.8, 1.2), "rough": (0.3, 0.7), "tint": (0.85, 1.15)}
# objects not painted in print looks (the real kit buys a similar brush and uses a real spam can): (clip, object id),
#   all 16 looks with the original texture (no paint, tint or noise). Other objects of the same clip, the P&P plate and
#   the background look stay. This table alone owns it; render, the full gate and the set build read it.
PRINT_KEEP = (
    ("spam_can__1", "spam_can"),
    ("brush_brush_box__20230927_029", "071"),
    ("brush_brush_plate__20230929_033", "072"),
)
# variants dropped from the v2 print set (the arm hides the camera at the end; all five with camera <-> right_arm_3
#   origin < 0.10 m; the v1 set stays). This table alone owns it; mkrun (MKRUN_DROP), the full gate, the set manifest
#   and the set check read it. Present in a set = dies.
PRINT_SET_DROP = tuple(f"spam_can__1_pv{k:02d}_pl_rt" for k in (5, 6, 8, 53, 55))


# clips not painted in flat colour looks either (OBJ_COLORS · v1's r0_s02_green · r0_s03_blue): spam keeps its
#   original texture in every look; the brushes (071 · 072) are painted in flat colour looks. PRINT_KEEP alone owns
#   object ids; only clip names here. Used by dense_render DENSE_COLOR_KEEP=1 (default off = v2 byte-identical) and
#   the set rule check (experiments.real_kit.vis_rules).
COLOR_KEEP_CLIPS = ("spam_can__1",)


def color_keep(variant: str) -> list[str]:  # noqa: V103 — used by experiments/policies/dense_render.py
    """Object ids not painted in flat colour looks: only clips of COLOR_KEEP_CLIPS · ids from PRINT_KEEP. A wrong name
    form dies as in print_keep."""
    import re

    m = re.fullmatch(r"(.+)_pv\d+_pl_rt", variant)
    if not m:
        raise SystemExit(f"[COLOR_KEEP] variant name form {variant!r}: only <clip>_pv<n>_pl_rt")
    bad = [c for c in COLOR_KEEP_CLIPS if c not in {k for k, _ in PRINT_KEEP}]
    if bad:
        raise SystemExit(f"[COLOR_KEEP] clips {bad} not in PRINT_KEEP (PRINT_KEEP owns the object ids)")
    return print_keep(variant) if m[1] in COLOR_KEEP_CLIPS else []


def print_keep(variant: str) -> list[str]:
    """Object ids not painted in a variant (<clip>_pv<n>_pl_rt); [] when none. Another name form dies (never read as
    'none' silently)."""
    import re

    m = re.fullmatch(r"(.+)_pv\d+_pl_rt", variant)
    if not m:
        raise SystemExit(f"[PRINT_KEEP] variant name form {variant!r}: only <clip>_pv<n>_pl_rt")
    return sorted(o for c, o in PRINT_KEEP if c == m[1])


def print_look(kind: str, key: str) -> dict:  # noqa: V103 — used by experiments/policies/dense_render.py
    """Print look values of one episode (key = '<variant>|<look>' -> seed; the same value on reruns).
    Flat colour = hue, saturation and value noise + roughness on the HSV of OBJ_COLORS[kind]; original texture =
    per-channel tint. Unknown kinds die."""
    import colorsys
    import hashlib
    import random

    h8 = hashlib.sha256(f"print_look|{key}".encode()).digest()[:8]
    rng, N = random.Random(int.from_bytes(h8, "big")), PRINT_NOISE
    if kind == "tex":
        tint = [rng.uniform(*N["tint"]) for _ in range(3)]
        return {"obj_tint": tint, "obj_noise": {"kind": kind, "seed": h8.hex(), "tint": tint}}
    if kind not in PRINT_LOOK_SUFFIX:
        raise SystemExit(f"[PRINT_LOOKS] unknown kind {kind!r}: only {sorted(PRINT_LOOK_SUFFIX)}")
    h, s, v = colorsys.rgb_to_hsv(*OBJ_COLORS[kind])
    dh, ks, kv = rng.uniform(-N["hue_deg"], N["hue_deg"]), rng.uniform(*N["sat"]), rng.uniform(*N["val"])
    rough = rng.uniform(*N["rough"])
    rgb = list(colorsys.hsv_to_rgb((h + dh / 360.0) % 1.0, min(s * ks, 1.0), min(v * kv, 1.0)))
    return {
        "obj_color": rgb,
        "obj_rough": rough,
        "obj_noise": {
            "kind": kind,
            "seed": h8.hex(),
            "base": list(OBJ_COLORS[kind]),
            "hue_deg": dh,
            "sat": ks,
            "val": kv,
            "rough": rough,
        },
    }


LIGHT_SCALE = (1.0, 1.4, 2.0, 2.8, 4.0)  # pilot: 0 % saturation even at x3 (Iray tonemapper); rotated across looks
# floor of the material mean albedo brightness (sRGB luminance, after tint): very dark tops and walls darken the whole
# image (first v2 run: leather and dark plastic top variants gave a head mean of 0.21-0.29). Look D materials always
# stay.
MIN_LUM = {"desk": 0.25, "floor": 0.12, "wall": 0.35}
# cap of the material mean albedo chroma (CIELAB C*, after tint): ambientCG painted walls (Paint001 · 003 ·
# PaintedPlaster001 · 004 · 005 …) have a base C* of 42-90 (strong green, blue, red), so the whole wall becomes a
# primary colour (v2 sheet S2 outside s03 · s09 · s11). Wall 32 keeps up to wooden planks (Planks021 31.5) and drops
# strong paint. Top and floor 50 keep every base (max 43) and only stop tints pushing them to primary colours.
MAX_CHROMA = {"desk": 50.0, "floor": 50.0, "wall": 32.0}
LIGHT_K = (2700.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5700.0, 6500.0, 7500.0)
LIGHT_K_GROUPS = {
    "warm": (2700.0, 3000.0, 3500.0),
    "neutral": (4000.0, 4500.0, 5000.0),
    "cool": (5700.0, 6500.0, 7500.0),
}  # rotate groups, choose within a group (partitions LIGHT_K)
# panel patterns: name -> lit panel numbers (look_d.PANEL_XY order = x(-2.1, -0.6, 0.9) x y(-1.5, 0, 1.5))
PANEL_PATTERNS = {
    "all9": tuple(range(9)),
    "xlo6": (0, 1, 2, 3, 4, 5),
    "xhi6": (3, 4, 5, 6, 7, 8),
    "ylo6": (0, 1, 3, 4, 6, 7),
    "yhi6": (1, 2, 4, 5, 7, 8),
    "checker5": (0, 2, 4, 6, 8),
    "cross5": (1, 3, 4, 5, 7),
    "row3": (3, 4, 5),
}
PANEL_JITTER = 0.15  # per-panel intensity x U(1 ± this) (continuous, not counted)
# HDRI dome intensity = DOME_BASE x scale / HDRI mean luminance. Measured (4 scenes, 48 samples, head mean): no dome
# 0.30-0.60 · scale 0.3 0.48-0.62 · 0.6 0.52-0.76 · 1.0 0.61-0.80 (beyond a bright window): the cap is the measured 0.3.
DOME_SCALE = (0.1, 0.2, 0.3)
DOME_BASE = 32.0
DOME_YAW = (0, 90, 180, 270)  # dome direction (°, vertical axis)
SATS = (0.7, 0.85, 1.0)  # tonemapper saturation
# distractors: key -> Poly Haven model id · words (name overlap rule); sizes are measured in Kit from the received USD
DISTRACTORS = {
    "stapler": {"id": "vintage_stapler", "words": ("stapler",)},
    "stationery": {"id": "stationery_supplies", "words": ("pen", "pencil", "stationery")},
    "alarm_clock": {"id": "alarm_clock_01", "words": ("clock",)},
    "apple": {"id": "food_apple_01", "words": ("apple",)},
    "lemon": {"id": "lemon", "words": ("lemon",)},
    "lime": {"id": "food_lime_01", "words": ("lime",)},
    "clay_pot": {"id": "planter_pot_clay", "words": ("pot", "planter")},
    "succulent": {"id": "potted_plant_04", "words": ("plant", "pot")},
    "wood_bowl": {"id": "wooden_bowl_01", "words": ("bowl",)},
    "jug": {"id": "jug_01", "words": ("jug", "pitcher", "pour")},
    "spray_bottle": {"id": "multi_cleaner_bottle", "words": ("bottle", "spray")},
    "tape_measure": {"id": "measuring_tape_01", "words": ("tape",)},
    "magnifier": {"id": "magnifying_glass_01", "words": ("glass", "magnifier")},
    "camera": {"id": "Camera_01", "words": ("camera",)},
    "watch": {"id": "digital_wrist_watch", "words": ("watch",)},
    "multimeter": {"id": "retro_multimeter", "words": ("meter",)},
    "cutting_board": {"id": "wooden_cutting_board", "words": ("board", "cutting")},
    "croissant": {"id": "croissant", "words": ("croissant", "bread")},
}
# distractors per row (trajectory), inclusive range; 4 dropped for lack of room (pilot placed 82 %)
N_DISTRACT = (0, 3)
# 'big distractor' = longest side >= this (m, near the GSO median 0.176); the first of each row is one
BIG_M = 0.18
DISTRACT_MAX_H = 0.30  # height cap (m); taller models are dropped on receipt (tall objects block the view)
DISTRACT_MAX_XY = 0.48  # footprint diagonal cap (m)
AXES = ("desk", "floor", "wall", "light_scale", "light_k", "panels", "dome", "sat", "distractors")


# share of a clip's random looks with the dome on (rounded; with an odd count the seed decides the last)
DOME_FRAC = 0.5


def size_why(dims_m) -> str | None:
    """Distractor size rule (both receipt and placement use it; the same input = the unrotated extent dims_mm measured
    on receipt). The rotated bounding box is not used: turning the yaw grows the diagonal up to √2, and bowls and pans
    that passed receipt would drop at placement. The reasons are data (manifest excluded · render meta look.apply
    dropped)."""
    h, xy = float(dims_m[2]), math.hypot(float(dims_m[0]), float(dims_m[1]))
    if h > DISTRACT_MAX_H or xy > DISTRACT_MAX_XY:
        return (
            f"size {[round(float(x), 3) for x in dims_m]} m > cap (height {DISTRACT_MAX_H} · "
            f"floor diagonal {DISTRACT_MAX_XY})"
        )
    if max(float(x) for x in dims_m) < SR.GSO_MIN_DIM:
        return f"size {[round(float(x), 3) for x in dims_m]} m < floor ({SR.GSO_MIN_DIM})"
    return None


def materials(manifest: dict | None = None) -> dict:
    """Material table per axis: the fixed Poly Haven table + the ambientCG choices the manifest recorded
    (manifest['acg'][axis])."""
    out = {"desk": dict(DESK), "floor": dict(FLOOR), "wall": dict(WALL)}
    for ax, ents in ((manifest or {}).get("acg") or {}).items():
        out[ax].update(ents)
    return out


def distractors(manifest: dict | None = None) -> dict:
    """Distractor table: the fixed Poly Haven table + the GSO choices the manifest recorded (manifest['gso_pick'])."""
    out = {k: dict(v, src="polyhaven") for k, v in DISTRACTORS.items()}
    for name in (manifest or {}).get("gso_pick") or []:
        out[f"gso:{name}"] = {"id": f"gso:{name}", "words": tuple(name_words(name)), "src": "gso"}
    return out


def name_words(name: str) -> list[str]:
    """Product name -> words of the name overlap rule (3+ characters; numbers, common colours and modifiers removed;
    trailing s dropped)."""
    stop = {
        "with",
        "pack",
        "inch",
        "count",
        "black",
        "white",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "grey",
        "silver",
        "gold",
        "small",
        "large",
        "medium",
        "mini",
        "set",
        "assorted",
        "color",
        "colors",
        "kids",
        "toddler",
        "months",
        "years",
        "edition",
        "series",
        "classic",
        "original",
        "style",
        "and",
        "the",
        "for",
        "all",
        "kit",
        "new",
        "red",
        "tan",
        "one",
        "two",
        "box",
        "pcs",
        "oz",
    }
    out = [
        w[:-1] if w.endswith("s") and len(w) > 4 else w
        for w in str(name).lower().replace("-", "_").replace(".", "_").split("_")
        if len(w) >= 3 and not w.isdigit() and w not in stop
    ]
    return sorted(set(out))


def ph_ids() -> list[tuple[str, str, str]]:
    """(kind texture|model, Poly Haven id, resolution) to receive from the fixed tables (deduplicated). HDRIs are
    chosen by pick_hdris."""
    out = set()
    for tab in (DESK, FLOOR, WALL):
        for v in tab.values():
            if v["kind"] == "tex":
                out.add(("texture", v["id"], v["res"]))
    for v in DISTRACTORS.values():
        out.add(("model", v["id"], "1k"))
    return sorted(out)


def wb_kelvin(light_k: float) -> float:
    return D_KELVIN + WB_ADAPT * (float(light_k) - D_KELVIN)
