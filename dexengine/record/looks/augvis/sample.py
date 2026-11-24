"""Sampler of the record render visual augment: draws one clip set (rows = trajectories · columns = looks)
reproducibly from one key (standard library only). The variation axes, options and scene rules (values) are owned by
augvis.catalog; this module reads them and owns only the choice rules.

Sampler: sample_set(allow, key, n_looks, n_rows), one clip set (rows = trajectories · columns = looks) from one key.
  columns (look)  room materials · light · HDRI · saturation; per axis 'card dealing' (_deal: in turn from a shuffled
                  deck · reshuffled and continued when the deck runs out) keeps the looks of one clip apart. Materials
                  rotate by family first (FAMILY_RULES), so top, floor and wall get a different family per look (wood
                  · stone · fabric · flat …); look D materials (D_*) are not used in random looks (only when that
                  family has no other material). The dome is on for DOME_FRAC of the looks.
  rows (trajectory)  distractor kinds · count · place seed; looks of one row share the distractors (one depth per row,
                  the record protocol). Counts rotate over the N_DISTRACT range, kinds rotate without repeats within
                  the clip, and rows with one or more take the first from the big distractors (longest side >= BIG_M)
                  (so a row with only small props does not look unchanged).
"""

from __future__ import annotations

import hashlib
import json
import math
import random

from dexengine.record import episodes as EPI  # owner of the look slot names (look_name)
from dexengine.record.looks import look_d as LOOK  # owner of the look D values
from dexengine.record.looks.augvis import catalog as AC  # owner of the axis, option and rule values


def seed_of(key: str, i: int) -> int:
    return int(hashlib.sha256(f"{key}|{i}".encode()).hexdigest()[:12], 16)


def _deal(rng: random.Random, items, n: int) -> list:
    """Card dealing: n cards in turn from a shuffled deck, reshuffled and continued when it runs out (no repeats when n
    <= deck size)."""
    items, out, deck = list(items), [], []
    if not items:
        raise SystemExit(f"{AC.MARK} tried to deal {n} cards from an empty deck")
    while len(out) < n:
        if not deck:
            deck = items[:]
            rng.shuffle(deck)
        out.append(deck.pop())
    return out


def _deal_materials(rng: random.Random, allow: dict, ax: str, n: int) -> list[tuple[str, list]]:
    """n (material, tint) pairs of one axis: families rotate, materials rotate within a family. Look D materials are
    not used when the family has other materials. Tints come from that material's allowed list."""
    a, tn = allow["allowed"][ax], allow["tints"][ax]
    fams: dict[str, list[str]] = {}
    for k in a:
        fams.setdefault(allow["families"][ax][k], []).append(k)
    fams = {f: ([k for k in ks if not k.startswith("D_")] or ks) for f, ks in fams.items()}
    decks = {f: [] for f in fams}
    out = []
    for f in _deal(rng, sorted(fams), n):
        if not decks[f]:
            decks[f] = sorted(fams[f])
            rng.shuffle(decks[f])
        k = decks[f].pop()
        out.append((k, list(rng.choice(tn[k]))))
    return out


def sample_set(allow: dict, key: str, n_looks: int, n_rows: int = 1) -> list[list[dict]]:
    """One clip set -> [row][column] variation specs (row = trajectory · distractors · place / column = random look ·
    room · light), reproducible from one key. With one row the names are s01 … (the pilot form), with several
    r<row>_s<column>."""
    rng = random.Random(seed_of(key, 0))
    a = allow["allowed"]
    mats = {ax: _deal_materials(rng, allow, ax, n_looks) for ax in ("desk", "floor", "wall")}
    scales = _deal(rng, a["light_scale"], n_looks)
    # intensity pairing: strong panels for dark tops (the meaning of the old sampler's bright-side weight LIGHT_W:
    # prevents dark variations with a head mean of 0.21-0.29; blue plastic x 1.0 gave a head mean of 0.278). The drawn
    # intensity values (no repeats) stay; only their order changes.
    dl = [allow["desk_luma"][f"{k}|{t[0]},{t[1]}"] for k, t in mats["desk"]]
    by_dark = sorted(range(n_looks), key=lambda i: (dl[i], i))
    strong = sorted(scales, reverse=True)
    scales = [0.0] * n_looks
    for r, i in enumerate(by_dark):
        scales[i] = strong[r]
    kgrp = _deal(rng, sorted(AC.LIGHT_K_GROUPS), n_looks)
    panels = _deal(rng, a["panels"], n_looks)
    sats = _deal(rng, a["sat"], n_looks)
    domes_all = a["dome"][1:]
    n_dome = int(n_looks * AC.DOME_FRAC) + (rng.random() < (n_looks * AC.DOME_FRAC) % 1) if domes_all else 0
    on = set(rng.sample(range(n_looks), n_dome))
    # a dome (indoor HDRI, daylight with window light) and warm panels in one look make the WB follow the panels and the
    # dome light tints the top blue (walnut albedo (0.45, 0.41, 0.38) -> (0.52, 0.54, 0.67) at 2,700 K + dome). Dome
    # looks swap only their group so they get a non-warm one (group counts and no repeats stay).
    for i in sorted(on):
        if kgrp[i] == "warm":
            j = next((j for j in range(n_looks) if j not in on and kgrp[j] != "warm"), None)
            if j is not None:
                kgrp[i], kgrp[j] = kgrp[j], kgrp[i]
    hdr = _deal(rng, domes_all, n_dome) if n_dome else []
    dsc = _deal(rng, AC.DOME_SCALE, n_dome) if n_dome else []
    looks, h = [], 0
    for i in range(n_looks):
        lk = {}
        for ax in ("desk", "floor", "wall"):
            lk[ax], lk[f"{ax}_tint"] = mats[ax][i]
        dome = "none"
        if i in on:
            dome, lk["dome_scale"], lk["dome_yaw"] = hdr[h], dsc[h], rng.choice(AC.DOME_YAW)
            h += 1
        else:
            lk["dome_scale"], lk["dome_yaw"] = None, None
        lk.update(
            light_scale=scales[i],
            light_k=rng.choice(AC.LIGHT_K_GROUPS[kgrp[i]]),
            panels=panels[i],
            jitter=[round(rng.uniform(1 - AC.PANEL_JITTER, 1 + AC.PANEL_JITTER), 3) for _ in LOOK.PANEL_XY],
            dome=dome,
            sat=sats[i],
        )
        looks.append(lk)
    rows = _deal_rows(random.Random(seed_of(key, 2000)), a, n_rows)
    out = []
    for r, rw in enumerate(rows):
        out.append(
            [
                dict(name=EPI.look_name(r, i, n_rows), **lk, **rw, place_seed=seed_of(key, 1000 + r))
                for i, lk in enumerate(looks)
            ]
        )
    return out


def _deal_rows(rng: random.Random, a: dict, n_rows: int) -> list[dict]:
    """Distractors per row (trajectory): counts rotate (N_DISTRACT), kinds rotate without repeats within the clip
    (reshuffled when the deck runs out). The first of a row with one or more comes from the big distractor deck."""
    pool, big = list(a["distractors"]), [k for k in a.get("distractors_big", []) if k in a["distractors"]]
    if not pool:
        return [{"distractors": []} for _ in range(n_rows)]
    lo, hi = AC.N_DISTRACT[0], min(AC.N_DISTRACT[1], len(pool))
    counts = _deal(rng, range(lo, hi + 1), n_rows)
    deck, bdeck, used = [], [], set()
    rows = []
    for n in counts:
        pick = []
        if n and big:
            if not bdeck:
                bdeck = [k for k in big if k not in used] or big[:]
                rng.shuffle(bdeck)
            pick.append(bdeck.pop())
        while len(pick) < n:
            if not deck:
                deck = [k for k in pool if k not in used] or pool[:]
                rng.shuffle(deck)
            k = deck.pop()
            if k not in pick:
                pick.append(k)
        used |= set(pick)
        bdeck = [k for k in bdeck if k not in used]
        deck = [k for k in deck if k not in used]
        rows.append({"distractors": pick})
    return rows


def count(allow: dict, max_slots: int | None = None) -> dict:
    """Discrete variations per scene: materials count (material, tint) pairs, distractors only kind combinations
    (count 0..cap, order ignored); places and panel jitter are continuous and not counted."""
    a, tn = allow["allowed"], allow["tints"]
    nd = len(a["distractors"])
    hi = min(AC.N_DISTRACT[1], nd, max_slots if max_slots is not None else AC.N_DISTRACT[1])
    combos = sum(math.comb(nd, k) for k in range(AC.N_DISTRACT[0], hi + 1)) if nd else 1
    mat = {ax: sum(len(tn[ax][k]) for k in a[ax]) for ax in ("desk", "floor", "wall")}
    n_dome = 1 + (len(a["dome"]) - 1) * len(AC.DOME_SCALE) * len(AC.DOME_YAW)
    light = len(a["light_scale"]) * len(a["light_k"]) * len(a["panels"]) * n_dome * len(a["sat"])
    base = mat["desk"] * mat["floor"] * mat["wall"] * light
    return {
        "desk": len(a["desk"]),
        "floor": len(a["floor"]),
        "wall": len(a["wall"]),
        "desk_pairs": mat["desk"],
        "floor_pairs": mat["floor"],
        "wall_pairs": mat["wall"],
        "light_scale": len(a["light_scale"]),
        "light_k": len(a["light_k"]),
        "panels": len(a["panels"]),
        "dome": n_dome,
        "sat": len(a["sat"]),
        "light_total": light,
        "distractor_kinds": nd,
        "distractor_max": hi,
        "distractor_sets": combos,
        "without_distractors": base,
        "total": base * combos,
    }


def spec_line(spec: dict, placed: list | None = None) -> str:
    d = spec["distractors"] if placed is None else [p["key"] for p in placed]
    return (
        f"{AC.MARK} {spec['name']} desk={spec['desk']}{spec.get('desk_tint')} floor={spec['floor']}"
        f"{spec.get('floor_tint')} wall={spec['wall']}{spec.get('wall_tint')} "
        f"light=x{spec['light_scale']},{int(spec['light_k'])}K,{spec.get('panels')} "
        f"wb={int(AC.wb_kelvin(spec['light_k']))}K dome={spec.get('dome')}x{spec.get('dome_scale')} "
        f"sat={spec.get('sat')} distractors={json.dumps(d)}"
    )
