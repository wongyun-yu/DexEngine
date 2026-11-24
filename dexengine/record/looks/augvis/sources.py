"""Visual augmentation of the record render: the single owner of asset sources, licences and selection rules
(fetching = looks.augvis.assets only).

Sources: Poly Haven (CC0: textures, models, indoor HDRIs) · ambientCG (CC0: materials) ·
Google Scanned Objects (CC BY 4.0: attribution required).
Selection is a rule (category, tag / name-word exclusion, cap, ordering); its result is recorded in the asset manifest
(the sampler reads the manifest: looks.augvis.catalog materials and distractors).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SOURCES = {
    "polyhaven": {
        "name": "Poly Haven",
        "url": "https://polyhaven.com",
        "license": "CC0 1.0 Universal",
        "license_url": "https://polyhaven.com/license",
        "attribution": None,
    },
    "ambientcg": {
        "name": "ambientCG",
        "url": "https://ambientcg.com",
        "license": "CC0 1.0 Universal",
        "license_url": "https://docs.ambientcg.com/license/",
        "attribution": None,
    },
    "gso": {
        "name": "Google Scanned Objects (Gazebo Fuel · GoogleResearch)",
        "url": "https://fuel.gazebosim.org/1.0/GoogleResearch",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Google Scanned Objects by Google Research (Downs et al., ICRA 2022), CC BY 4.0, "
        "https://app.gazebosim.org/GoogleResearch",
    },
}
API_FILES = "https://api.polyhaven.com/files/{id}"
ACG_API = "https://ambientcg.com/api/v2/full_json?type=Material&limit=3000&include=tagData,displayData"
ACG_ZIP = "https://ambientcg.com/get?file={id}_1K-JPG.zip"
GSO_API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models?per_page=100&page={page}"
GSO_ZIP = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}/tip/{name}.zip"
PH_HDRI_API = "https://api.polyhaven.com/assets?t=hdris&c=indoor"
# ambientCG selection rule: axis -> {shown category: cap} · tag exclusion · by downloads descending (ties by id),
# up to the cap
ACG_RULES = {
    "desk": {
        "Wood": 10,
        "Painted Wood": 6,
        "Plastic": 6,
        "Marble": 4,
        "Terrazzo": 3,
        "Leather": 3,
        "Fabric": 4,
        "Cork": 2,
        "Chipboard": 2,
    },
    "floor": {
        "Wood Floor": 10,
        "Tiles": 10,
        "Carpet": 6,
        "Terrazzo": 3,
        "Marble": 3,
        "Concrete": 4,
        "Tatami": 2,
        "Rubber": 2,
    },
    "wall": {
        "Painted Plaster": 8,
        "Plaster": 5,
        "Paint": 5,
        "Wallpaper": 5,
        "Painted Bricks": 4,
        "Tiles": 5,
        "Bricks": 3,
        "Planks": 3,
    },
}
ACG_TAG_EXCLUDE = frozenset(
    {
        "dirty",
        "damaged",
        "old",
        "rust",
        "rusty",
        "moss",
        "mossy",
        "weathered",
        "grunge",
        "grungy",
        "cracked",
        "broken",
        "wet",
        "outdoor",
        "exterior",
        "ruins",
        "ruined",
        "stained",
        "burnt",
        "burned",
        "decayed",
        "decay",
        "mold",
        "dusty",
        "peeling",
        "graffiti",
        "worn",
    }
)
# Tint class (category -> class). paint = paint, fabric, plastic (any colour) · natural = wood, stone, tile
# (small jitter)
TINT_CLASS = {
    "Painted Plaster": "paint",
    "Plaster": "paint",
    "Paint": "paint",
    "Painted Wood": "paint",
    "Plastic": "paint",
    "Fabric": "paint",
    "Carpet": "paint",
    "Leather": "paint",
    "Painted Bricks": "paint",
    "Rubber": "paint",
}
# GSO selection rule: category exclusion · name-word exclusion · GSO_MAX spread evenly by name (refiltered by size
# after download)
GSO_CAT_EXCLUDE = frozenset({"Shoe", "Bag", "Hat", "Car Seat"})
GSO_WORD_EXCLUDE = frozenset(
    {
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "boot",
        "boots",
        "sandal",
        "sandals",
        "slipper",
        "slippers",
        "heel",
        "heels",
        "loafer",
        "moccasin",
        "clog",
        "oxford",
        "runner",
        "trainer",
        "backpack",
        "purse",
        "handbag",
        "hat",
        "cap",
        "seat",
    }
)
GSO_MAX = 120
GSO_MIN_DIM = 0.04  # dropped when the longest side is below this (a dot on screen)
# HDRI selection rule: Poly Haven indoor maps whose name has a word of HDRI_WORDS and none of HDRI_EXCLUDE,
# HDRI_MAX spread evenly by name
HDRI_WORDS = frozenset(
    {
        "office",
        "room",
        "studio",
        "lounge",
        "kitchen",
        "cafe",
        "workshop",
        "lab",
        "library",
        "apartment",
        "house",
        "interior",
        "hall",
        "museum",
        "shop",
        "classroom",
        "bedroom",
        "living",
        "hotel",
        "restaurant",
        "gym",
        "loft",
        "attic",
        "corridor",
        "lobby",
        "store",
        "bathroom",
    }
)
HDRI_EXCLUDE = frozenset(
    {
        "abandoned",
        "creepy",
        "burnt",
        "cave",
        "tunnel",
        "underpass",
        "vault",
        "boiler",
        "garage",
        "factory",
        "construction",
        "church",
        "chapel",
        "cinema",
        "circus",
        "castle",
        "cellar",
        "ruin",
        "warehouse",
        "night",
        "dark",
        "christmas",
        "boxing",
        "billiard",
        "aircraft",
        "auto",
        "autoshop",
        "bunker",
        "prison",
        "hospital",
        "photostudio",
        "cyclorama",
    }
)
HDRI_MAX = 24
HDRI_RES = "2k"
#: The canonical asset generation, recognised by content: the sha256 of the root's SHA256SUMS (one `<sha256>  <path>`
#: row per file) · its rows · the bytes of the listed files. Rendering from an AUGVIS_ASSETS root of another generation
#: dies (e.g. an augvis_v1 root). Its list (manifest.json + SHA256SUMS) is kept in git at configs/record/<name>/
#: and is the input of the fetch (augvis.assets.canonical_list checks it against this line). A new generation
#: (augvis.assets --new-generation) changes this line and that folder in one commit.
#: manifest_sha256 = the list's manifest.json: the canonical root's manifest (manifest_from) with its stored old values
#: translated to English; the asset files
#: and SHA256SUMS are the canonical root's bytes.
ASSET_GEN = {
    "name": "augvis_v2",
    "sums_sha256": "6fb4f721144f57eb7ab33a1ec663744aba208174a9343228390b26c827538fee",
    "files": 840,
    "bytes": 864779498,
    "manifest_sha256": "644e6d019f06d02902d62d79b5e9e393230b22daf6e294bf96113c54bdbad0b1",
    "manifest_from": "7247c5e75a6d96ab7f14a10fecd155283d9a1b39b17f9216d1d33dfa19d0a623",
}


def asset_gen_line(root) -> str:
    """The marker line when the asset root holds the canonical generation: its SHA256SUMS has the canonical sha256 and
    row count, every listed file is there and their sizes add up to the canonical bytes. A missing SHA256SUMS, another
    list or a missing file dies. When the files were fetched (manifest `created`) is not compared: a copy or another
    fetch of the same files passes. File contents are checked by `augvis.assets --verify`."""
    r = Path(root)
    if not (r / "SHA256SUMS").is_file():
        raise SystemExit(
            f"[AUG_VIS] assets {root}: SHA256SUMS missing — the generation is recognised by that list "
            f"(canonical {ASSET_GEN['name']})"
        )
    raw = (r / "SHA256SUMS").read_bytes()
    rows = [ln.split("  ", 1)[1] for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    missing = [f for f in rows if not (r / f).is_file()]
    got = {
        "sums_sha256": hashlib.sha256(raw).hexdigest(),
        "files": len(rows),
        "bytes": sum((r / f).stat().st_size for f in rows if f not in set(missing)),
    }
    want = {k: ASSET_GEN[k] for k in got}
    if got == want and not missing:
        return f"[AUG_VIS] assets canonical {ASSET_GEN['name']} · files {got['files']} · {got['bytes']:,} B · {root}"
    raise SystemExit(
        f"[AUG_VIS] assets {root} are not the canonical {ASSET_GEN['name']}: {got} != {want}"
        + (f" · listed files missing {len(missing)} {missing[:3]}" if missing else "")
        + " (use the canonical generation)"
    )


def _evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)] if n > 1 else items[:1]


def pick_acg(assets: list[dict]) -> dict:
    """ambientCG API list -> {axis: [(id, category)]}: ACG_RULES · ACG_TAG_EXCLUDE · by downloads descending."""
    out = {}
    for ax, cats in ACG_RULES.items():
        got = []
        for cat, cap in cats.items():
            c = [
                a
                for a in assets
                if a.get("displayCategory") == cat
                and a.get("dataType") == "Material"
                and not (set(map(str.lower, a.get("tags") or [])) & ACG_TAG_EXCLUDE)
            ]
            c.sort(key=lambda a: (-int(a.get("downloadCount") or 0), a["assetId"]))
            got += [(a["assetId"], cat) for a in c[:cap]]
        out[ax] = got
    return out


def pick_gso(models: list[dict]) -> list[str]:
    """Fuel GoogleResearch list -> model names (after category and word exclusion, GSO_MAX spread evenly by name)."""
    ok = []
    for m in models:
        cats = set(m.get("categories") or [])
        words = set(str(m["name"]).lower().split("_"))
        if cats & GSO_CAT_EXCLUDE or words & GSO_WORD_EXCLUDE:
            continue
        ok.append(m["name"])
    return _evenly(sorted(set(ok)), GSO_MAX)


def pick_hdris(names) -> list[str]:
    """Poly Haven indoor HDRI names -> those with an HDRI_WORDS word and no HDRI_EXCLUDE word, HDRI_MAX spread
    evenly."""
    ok = []
    for n in sorted(names):
        w = set(n.lower().split("_"))
        if w & HDRI_WORDS and not (w & HDRI_EXCLUDE):
            ok.append(n)
    return _evenly(ok, HDRI_MAX)
