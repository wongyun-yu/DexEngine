"""Per-object uniform colour rules: the render overlay colour of objects with no raw texture, material colour or video
projection. This module is the only place for these values.

Rules:
  1. Raw textures, material Kd and video projection are handled by the earlier stages (textures.rules, textures.fill,
     packs); this module applies only when none exists.
  2. Colour stated by the source = a colour word in the object name (COLOR_WORDS; OakInk2 obj_desc 'blue mug',
     HRDexDB 'white_candle_holder').
  3. Kind colour = a kind word (KIND_COLORS regexes, first match from the top).
  No match raises (no silent grey). Objects sharing a base colour are separated by a lightness ±8 % and hue ±2 % shift
  derived from the physical object key (physical_key; left, right and raw variants are one physical object and get one
  colour). Different physical objects in one dataset with the same colour are caught by color_collisions and blocked
  by the cover check (tex_packs.cover).
  Values are sRGB hex (hand-picked); _lin alone converts them to linear 0–1 (UsdPreviewSurface diffuseColor).
The rule no longer colours an object at bake time: an object is solid only when configs/objects/solid_colors.tsv states
its colour (SOLID_FILE · solid_color); object_color writes and checks those statements
(a solid-colour tool outside this package) and keeps the collision check (color_collisions).
Reader of the statements: textures.fill.uniform_rgb (bake, pack lookup, cover check). Runs without pxr or trimesh.
Usage (shell): python3 dexengine/robots/textures/colors.py pack-kind <ds>  -> pack | nopack
"""

from __future__ import annotations

import argparse
import os
import sys

from dexengine.robots.textures import rules as OT

#: The solid-colour statements (dataset · object · linear r g b · why) — written and checked only by
#: a solid-colour tool outside this package; an object without a row has no solid colour (solid_color raises).
SOLID_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs", "objects", "solid_colors.tsv")
)
_SOLID: dict | None = None


def solid_table() -> dict[tuple[str, str], tuple[tuple[float, float, float], str]]:
    """{(dataset, object): ((r, g, b) linear, why)} of SOLID_FILE (read once). A bad row raises."""
    global _SOLID
    if _SOLID is None:
        out = {}
        with open(SOLID_FILE, encoding="utf-8") as f:
            head = f.readline().rstrip("\n").split("\t")
            if head != ["dataset", "object", "r", "g", "b", "why"]:
                raise ValueError(f"{SOLID_FILE}: header {head}")
            for n, ln in enumerate(f, 2):
                c = ln.rstrip("\n").split("\t")
                if len(c) != 6 or (c[0], c[1]) in out:
                    raise ValueError(f"{SOLID_FILE}:{n}: bad or repeated row {c[:2]}")
                out[(c[0], c[1])] = ((float(c[2]), float(c[3]), float(c[4])), c[5])
        _SOLID = out
    return _SOLID


def solid_color(ds: str, oid: str) -> tuple[float, float, float]:
    """The stated solid colour of one object (linear RGB). No row = ValueError: an object is never coloured by its
    name silently — state it as a row of SOLID_FILE (value = object_color)."""
    row = solid_table().get((ds, oid))
    if row is None:
        raise ValueError(
            f"{ds}/{oid}: no solid-colour statement in {SOLID_FILE} — the object has no texture; state its colour "
            f"(a row {ds} · {oid} of that file; value = object_color)"
        )
    return row[0]


COLOR_WORDS: dict[str, str] = {
    "red": "#C0392B",
    "blue": "#2E64B4",
    "yellow": "#F2C230",
    "orange": "#E67E22",
    "green": "#3F9B4A",
    "white": "#ECEAE4",
    "black": "#232323",
    "pink": "#F2A2B8",
    "purple": "#7D4E9E",
    "brown": "#7B4A2A",
    "wood": "#B07A4A",
    "wooden": "#B07A4A",
    "silver": "#C3C7CC",
    "ceramic": "#E9E4DA",
}
KIND_COLORS: tuple[tuple[str, str, str], ...] = (  # (kind-word regex, sRGB, rationale); first match from the top
    ("apple", "#B5262A", "apple = red"),
    ("banana", "#EBC944", "banana = yellow"),
    ("airplane|train", "#B83A3A", "toy vehicle = red"),
    ("alarm ?clock|clock", "#D9D6CF", "clock = white face"),
    (
        "binoculars|camera|phone|headphones|game ?control|eyeglasses|flashlight|mouse",
        "#2A2A2C",
        "electronics, optics = black",
    ),
    ("bowl|cup|teapot|mug|jug", "#E6E1D6", "ceramic = white"),
    ("door ?knob", "#C3C7CC", "door knob = silver metal"),
    ("cube", "#3A6FB0", "primitive cube = blue"),
    ("cylinder", "#3C9A5B", "cylinder = green"),
    ("pyramid", "#E0892B", "pyramid = orange"),
    ("sphere", "#8A55A8", "sphere = purple"),
    ("torus", "#2A9D9A", "torus = teal"),
    ("duck", "#F0C93A", "duck = yellow"),
    ("elephant", "#8F9AA6", "elephant = blue grey"),
    ("flute|knife|scoop|saucepan|sharpener|tongs|watch|spatula", "#BFC4CA", "metal = silver"),
    ("frying ?pan", "#2B2B2B", "frying pan = black"),
    ("hammer|stamp|rack|hanger|holder", "#9A6A3E", "wood = brown"),
    ("^hand$", "#DDA88A", "hand model = skin tone"),
    ("light ?bulb", "#F3EED8", "light bulb = milk white"),
    ("piggy", "#F2A7BB", "piggy bank = pink"),
    ("scissors", "#C23B33", "scissor handles = red"),
    ("bunny", "#C9A27C", "bunny = tan"),
    ("stapler", "#B23434", "stapler = red"),
    ("toothbrush", "#3E7DC8", "toothbrush = blue"),
    ("toothpaste|lotion|spray|shower|candle", "#EDEDEA", "container = white"),
    ("water ?bottle", "#4A8FC6", "water bottle = blue"),
    ("wine ?glass|beaker|flask|test tube|glass rod", "#DCE6EE", "glass = pale blue"),
    ("donut", "#C58B52", "donut = tan"),
    ("frog", "#5DA34A", "frog = green"),
    ("plant ?pot", "#B5653E", "plant pot = terracotta"),
    ("screwdriver", "#E8C02F", "screwdriver handle = yellow"),
    ("book", "#3C5A8A", "book = navy"),
    ("laptop|microwave", "#3A3B3E", "appliance = dark grey"),
    ("drawer", "#A87A50", "drawer = wood"),
)
#: Names of objects whose asset id does not say the name -> {dataset: {id: (name, source)}}. Datasets whose
#: distribution has a name index are not listed here; physical_key/object_name read it (OakInk2 obj_desc).
OBJECT_NAMES: dict[str, dict[str, tuple[str, str]]] = {
    "taco": {
        "130": ("spatula", "TACO Object_Poses '(stir, spatula, plate)/20231002_055/tool_130.npy'"),
        "055": (
            "blue ruler",
            "TACO Object_Poses '(measure, ruler, cup)/20231104_108/tool_055.npy' · "
            "colour = source video (12 views of the same sequence: a blue translucent set square)",
        ),
        "062": ("spatula", "TACO Object_Poses '(stir, spatula, pan)/20230923_008/tool_062.npy'"),
    },
}
_OAK_KEY = r"(?i)(?<![a-z0-9])([cs]\d{5}|o\d{2}_\d{4}_\d{5})"
_DESC: dict = {}


def physical_key(ds: str, oid: str) -> str:
    """Physical object key: for OakInk2 the first original id inside the asset id (left_, right_, raw_, articulated
    merges; C10001, O02@0011@00002); otherwise the asset id itself."""
    import re

    if ds == "oakink2":
        m = re.search(_OAK_KEY, oid)
        if not m:
            raise ValueError(f"cannot read the original object id from oakink2 asset id {oid!r}")
        k = m[1].upper()
        return k.replace("_", "@") if k.startswith("O") else k
    return oid


def object_name(ds: str, oid: str) -> str:
    """Object name (lower case, spaces): OBJECT_NAMES; OakInk2 = original obj_desc.json (object_raw, object_repair);
    otherwise the asset id."""
    if oid in OBJECT_NAMES.get(ds, {}):
        return OBJECT_NAMES[ds][oid][0]
    if ds == "oakink2":
        if "oakink2" not in _DESC:
            import json

            from dexengine.core import paths as P

            root = P.data_layout.data_abspath(P.data_layout.render("raw", "oakink2"))
            d = {}
            for sub in ("object_raw", "object_repair"):
                f = os.path.join(root, "hf", "obj", sub, "obj_desc.json")
                if not os.path.isfile(f):
                    raise ValueError(f"OakInk2 name index {f} missing — cannot decide the object colour")
                with open(f, encoding="utf-8") as fh:
                    d.update(json.load(fh))
            _DESC["oakink2"] = d
        k = physical_key(ds, oid)
        n = (_DESC["oakink2"].get(k) or {}).get("obj_name")
        if not n:
            raise ValueError(f"OakInk2 {oid}({k}) is not in obj_desc — cannot decide the object colour")
        return n.lower()
    return oid.replace("_", " ").replace("-", " ").lower()


def _lin(x: float) -> float:
    """sRGB 0–1 -> linear 0–1 (IEC 61966-2-1), rounded to 4 decimals."""
    return round(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4, 4)


def _shade(h: str, key: str) -> tuple[float, float, float]:
    """Shift base colour h by the key-derived lightness ±8 % and hue ±2 % (reflected at the ends so shades do not
    pile up there)."""
    import colorsys
    import hashlib

    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    z = hashlib.sha256(key.encode()).digest()
    dl = (int.from_bytes(z[:4], "big") / 0xFFFFFFFF - 0.5) * 0.16
    dh = (int.from_bytes(z[4:8], "big") / 0xFFFFFFFF - 0.5) * 0.04
    ll = ll + dl if 0.04 <= ll + dl <= 0.96 else ll - dl
    return colorsys.hls_to_rgb((hh + dh) % 1.0, ll, ss)


def object_color(ds: str, oid: str) -> tuple[tuple[float, float, float], str]:
    """Object uniform colour -> (linear RGB, rationale). Rules: module docstring. ValueError when none applies."""
    import re

    name = object_name(ds, oid)
    words = re.split(r"[^a-z]+", name)
    cw = next((w for w in words if w in COLOR_WORDS), None)
    if cw:
        base, why = COLOR_WORDS[cw], f"source colour — name '{name}' has '{cw}'"
    else:
        k = next(((c, w) for p, c, w in KIND_COLORS if re.search(p, name)), None)
        if k is None:
            raise ValueError(
                f"{ds}/{oid}: name '{name}' has no colour word or kind word — add it to KIND_COLORS or OBJECT_NAMES"
            )
        base, why = k[0], f"kind colour — {k[1]} (name '{name}')"
    return tuple(_lin(min(1.0, max(0.0, x))) for x in _shade(base, f"{ds}/{physical_key(ds, oid)}")), why


def color_collisions(ds: str, oids) -> list[tuple[str, str, tuple]]:
    """Pairs of different physical objects in a dataset that got the same colour -> [(id a, id b, colour)]. The cover
    check requires none."""
    seen: dict[tuple, tuple[str, str]] = {}
    out = []
    for o in sorted(set(oids)):
        c, _w = object_color(ds, o)
        pk = physical_key(ds, o)
        if c in seen and seen[c][1] != pk:
            out.append((seen[c][0], o, c))
        seen.setdefault(c, (o, pk))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pack-kind", help="overlay kind of a dataset -> pack | nopack").add_argument("ds")
    a = ap.parse_args(argv)
    k = OT.PACK_POLICY.get(a.ds)
    print("pack" if k and k != OT.SRC_UNIFORM else "nopack")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
