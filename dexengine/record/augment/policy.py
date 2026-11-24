#!/usr/bin/env python3
"""Placement augment policy values: **single owner** of the new placements kept per clip, the region cell choice and
the choice rule (the module header uses the standard library only and is read from shell too; only the coordinate
round trip imports the scene/capture transform owner augment.scene lazily).

PER_CLIP: the number of new placements (the original placement pv00 excluded) kept from one clip as training and
record data.
Candidate source (judged-cell contract): from the **region cells** the judge dumped (judge
  mode D, one canonical dz layer) SAMPLE['cells'] cells are drawn uniformly at random per clip (pick_cells; first row =
  the canonical cell as the control), and judge mode L judges all of them with the same function as the canonical
  one. Only passing cells become variants (abs_to_variant), and of those that also pass the pre gate the first
  SAMPLE['phys_max'] in sample order go to physics (phys_set; the canonical cell pv00 is the control outside the cap).
Sampling seed (clip_seed): the sha256 of the clip identity (dataset + clip base name; hand name excluded) mixed with
  SAMPLE['seed']. Cell rank (cell_rank) = sha256(seed · cell key (ky, kx, kyy)): even when the regions (volumes)
  differ per hand, overlapping cells get the same rank for all four hands (random.sample gives other cells for other
  populations) · independent of the Python version.
Canonical cell (canon_index): the one region cell with dz = canonical dz and yaw and displacement within
  CANON_MATCH_FRAC of the grid spacing of the canonical placement (canon: the baked tree placement) (a rule that covers
  plan-row rounding and never confuses a neighbour cell); none or two die.
Choice rule (pick_per_clip, name PICK_RULE: placement augments as different as possible): among 'salvaged' variants
  (sim success >= 1, pre collision gate passed, post gate passed) the **greedy farthest point** PER_CLIP with the
  original placement (identity = origin) as a fixed point, in normalised coordinates (dx/x, dy/y, yaw/yaw) (scale =
  the judge region's half widths x, y and yaw half width, scale_of(the region dump's grid); the values are owned by
  the judge dump and not written again here): repeatedly the candidate with the largest minimum distance to the points
  chosen so far (origin included) · ties by ascending k. With too few, as many as exist (nothing invented) · pv00
  (k = 0) is always left out · rows without dx, dy, yaw die. The record of the chosen set = pick_meta.
Post collision gate (a variant must not be worse than the same clip's original placement): per term of the executed
  rollouts max penetration <= max(judge threshold, original placement value + POST_REL_MM). Terms = POST_TERMS (desk,
  support, torso, both arms crossing; keys of collision.term_values) · judge thresholds are collision constants (the
  caller passes them as floors; not written here) · original placement value = the term's max over the kept successful
  rollouts of the same clip's pv00 (post_ref; without pv00 successes the threshold alone). Hand-hand is outside the
  gate (recorded only).
Judged bundle read format (augment side; places in augment.judged): chunk jsonl end line {"done": true · DONE_KEYS} ·
  judge identity fields JUDGE_ID_KEYS (judge input fingerprint) · judge commits of the region dumps (the bundle part of
  the augment cache key).
Read: python3 dexengine/record/augment/policy.py --per-clip
"""

from __future__ import annotations

import argparse
import hashlib
import math

PER_CLIP = 3  # 4 trajectories per clip = original placement 1 + new placements 3 (x 5 looks = 20 episodes)
POST_REL_MM = 1.0
POST_TERMS = ("desk", "support", "torso", "cross")
#: region cell sampling: judge cells outside the canonical cell · max sent to physics (front of the sample order) ·
#: the base seed (basis of clip_seed).
SAMPLE = {"cells": 32, "phys_max": 16, "seed": 42}
EPC = 8
#: cap of successful rollouts the augment rollout keeps per variant (the record recorder's keep cap; the augment run
#: run_spec passes it to the session as AUG_K_KEEP; session marker '[COLLECT_SESSION] hook: … K_TOP -> K_KEEP
#: (AUG_K_KEEP)'). The post gate (post_ok) looks at these rollouts, and the original placement reference (post_ref) is
#: the max over these rollouts of pv00. 3 = the old K_TOP value kept so the augment conditions did not change when
#: record stage 1 started keeping only the chosen one (rollout.recorder.K_TOP 3 -> 1) (changes go here only).
K_KEEP = 3
#: name of the choice rule (meta and docs write this name); the rule is pick_per_clip alone (not an env variable).
PICK_RULE = "farthest_from_identity"
#: number of candidate variants kept through their trajectory files on the record-only (per-clip render) path (from the
#: front of the PICK_RULE order; the first PER_CLIP equal the old choice). The render session chooses trajectory 1 +
#: PER_CLIP, grasp-visible from the original camera first (rule owner record.episodes.choose). The old traj path
#: (render true) keeps PER_CLIP.
CAND_MAX = 8
#: name of the cell choice rule (rule of the cells json); the rule is pick_cells alone.
CELL_RULE = "region_sha256_rank"
#: cell choice for originally off-screen clips (clips whose orig was skipped by record.episodes.ORIG_FOV_RULE): the same
#: pick_cells with another key (fov_key: higher predicted visibility -> closer to the image centre -> cell_rank). Cell
#: count = SAMPLE_FOV_CELLS (outside canonical; measured 0.3-0.7 s per judge-L cell; pass rate of cells moved toward
#: the image 0.06-0.10 gives an expected 4 rows). Prediction owner render.fov.predict (no render); acceptance is the
#: render session's measured vis_strict.
CELL_RULE_FOV = "region_fov_vis_rank"
#: dense placement rule (the E5 L1 dense set: no box, nearest first, 100 per clip): every cell inside the box around
#: the canonical cell, judged nearest first (list built by the E5 dense-cell builder). Not a
#: canonical record rule: accepted only when the bundle has a dense/dense_<clip>.json mark and env AUG_DENSE_CELLS=1
#: (default off = same behaviour; off with a mark present dies).
CELL_RULE_DENSE = "region_near_canon_rank"
DENSE_ENV = "AUG_DENSE_CELLS"
SAMPLE_FOV_CELLS = 96


#: home of CANON_MATCH_FRAC (canonical cell match tolerance) = the judge side canon_fit; the judge never imports augment
#: code.
def __getattr__(name: str):
    if name == "CANON_MATCH_FRAC":
        from dexengine.placement.judge import canon_fit  # noqa: PLC0415

        return canon_fit.CANON_MATCH_FRAC
    raise AttributeError(name)


#: judge identity fields: the judge input fingerprint carried by the region dump, the cells json and every cell row
#: (checked by augment.judged.load; the bundle part of the augment cache key = augment.judged.bundle_commits).
JUDGE_ID_KEYS = ("judge_fp",)
#: augment clip hold reasons: a clip defect holds that clip only (chain.after_store, line [AUG_CANON_HOLD], kind in
#: counts.json held). Readers split by these constants, not by text prefixes. Values are data (older values ->
#: a deployment's migration table; a deployment's dashboard reads the hold lines).
# canonical cell (pv00) technical failure of the generation record (augment.stage.canon_tech_fail)
HOLD_CANON_TECH = "canonical cell technical failure"
# judged cell bundle check failed (augment.judged.load)
HOLD_INPUT = "judge input refused"
# the judged bundle was made by other judge code (no producer in this tree; existing hold records carry it)
HOLD_STALE = "placement augment judge version differs"
HOLD_KINDS = (HOLD_CANON_TECH, HOLD_INPUT, HOLD_STALE)
#: chunk end line (the same text as the end line of judge mode L stdout): clips, cells, passes.
#: fields of the chunk end line (contract v2.2): {"done": true, "clips", "cells", "pass"}. Chunk identity = the line's
#: judge_commit (augment.judged.read_chunks: one chunk = one commit).
#: judge_fp differs per clip (dexengine.placement.judge.input_record = one clip's line fingerprint), so it is not the
#: chunk identity; clips are checked by augment.judged.load.
DONE_KEYS = ("clips", "cells", "pass")
#: fields of the region grid the scale reads.
GRID_KEYS = ("half_x_m", "half_y_m", "yaw_lo", "yaw_hi", "step")
#: acceptance ladder (exclude the box around the baked placement from the search; the box is the default, but a
#: rotation of 30° or more is also fine … when candidates run short the criterion relaxes … 4 placement augments per
#: clip must be secured): per level (box half side m, rotation threshold deg). A cell outside the canonical one is
#: accepted at that level when its placement move (dx, dy: d_scene of abs_to_variant = the object move relative to the
#: original placement) is outside the box (|dx| > half side ∨ |dy| > half side) or |Δyaw| >= threshold · last level
#: (None) = only the canonical cell is excluded.
#: A cell's level (admit_level) = the strictest level that accepts it; physics (phys_set) and choice (pick_per_clip)
#: fill by ascending level (the strictest level that fills the needed count; the render choice follows the candidate
#: order as it is, record.episodes.choose).
ADMIT_LADDER = ((0.075, 30.0), (0.05, 20.0), (0.025, 10.0), None)
#: target to secure: 4 trajectories per clip (original 1 + PER_CLIP placement augments; 4 placement augments when the
#: original is off-screen) = candidate-side target PER_CLIP + 1.
SECURE = PER_CLIP + 1


def clip_seed(dataset: str, clip: str, base: int | None = None) -> int:
    """Sampling seed per clip = the first 8 bytes of sha256(base seed · dataset · clip base name) (Python hash()
    differs per process, so it is not used). The hand name is not included (four hands get the same candidates from
    the same clip). clip = the name without the tree suffix (paths.TREE_SFX)."""
    base = int(SAMPLE["seed"] if base is None else base)
    if not dataset or not clip:
        raise ValueError(f"[AUG_POLICY] clip_seed: dataset {dataset!r} · clip {clip!r}: empty value")
    h = hashlib.sha256(f"{base}\0{dataset}\0{clip}".encode()).digest()
    return int.from_bytes(h[:8], "big")


# ── region cells ────────────────────────────────────────────────────────────────
def scale_of(grid: dict) -> dict:
    """Region dump grid -> choice scale {x: half width x m, y: half width y m, yaw: yaw half width deg}; missing fields
    or values <= 0 die."""
    miss = [k for k in GRID_KEYS if not isinstance((grid or {}).get(k), (int, float))]
    if miss:
        raise ValueError(f"[AUG_POLICY] region grid lacks {miss}: no scale")
    s = {
        "x": float(grid["half_x_m"]),
        "y": float(grid["half_y_m"]),
        "yaw": (float(grid["yaw_hi"]) - float(grid["yaw_lo"])) / 2.0,
    }
    if min(s.values()) <= 0.0:
        raise ValueError(f"[AUG_POLICY] scale {s}: <= 0")
    return s


def cell_rank(seed: int, cell) -> str:
    """Cell rank key = sha256(seed · ky · kx · kyy); dz is left out (the same plane cell even when the canonical dz
    layer differs per hand)."""
    ky, kx, kyy = (int(cell[0]), int(cell[1]), int(cell[2]))
    return hashlib.sha256(f"{int(seed)}\0{ky}\0{kx}\0{kyy}".encode()).hexdigest()


def fov_key(seed: int, score: dict):
    """Cell key of CELL_RULE_FOV: score = cell (ky, kx, kyy, dz) -> (predicted visibility 0..1, distance px from the
    image centre). Cells without a score die."""

    def key(c):
        if c not in score:
            raise ValueError(f"[AUG_POLICY] off-screen cell choice: cell {c} has no predicted score")
        v, dist = score[c]
        return (-round(float(v), 3), round(float(dist), 1), cell_rank(seed, c))

    return key


def pick_cells(cells: list, canon, seed: int, n: int | None = None, key=None) -> list[tuple]:
    """cells = region cells [(ky, kx, kyy, dz)] · canon = canonical cell -> [canonical cell] + n cells outside it (key
    key: cell_rank by default, fov_key for off-screen clips; from the front, no duplicates). Overlapping cells, a
    canonical cell missing from the region or cells outside the canonical dz layer die. With too few, as many as
    exist."""
    n = int(SAMPLE["cells"] if n is None else n)
    if n < 0:
        raise ValueError(f"[AUG_POLICY] n={n} < 0")
    keys = [(int(c[0]), int(c[1]), int(c[2]), float(c[3])) for c in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("[AUG_POLICY] region cells overlap")
    c0 = (int(canon[0]), int(canon[1]), int(canon[2]), float(canon[3]))
    if c0 not in set(keys):
        raise ValueError(f"[AUG_POLICY] canonical cell {c0} is not in the region")
    off = sorted({c[3] for c in keys} - {c0[3]})
    if off:
        raise ValueError(f"[AUG_POLICY] region cells outside the canonical dz layer {c0[3]} (dz {off[:3]})")
    rest = sorted((c for c in keys if c != c0), key=lambda c: ((cell_rank(seed, c) if key is None else key(c)), c))
    return [c0] + rest[:n]


def _min_gap(vals: list[float]) -> float | None:
    u = sorted({round(float(v), 9) for v in vals})
    gaps = [b - a for a, b in zip(u, u[1:], strict=False) if b - a > 0]
    return min(gaps) if gaps else None


def canon_index(theta_deg: list, d_abs: list, dz: list, canon: dict, grid: dict) -> int:
    """Index of the canonical cell among the region cells (θ deg, d (x, y) m, dz in the same order); module doc
    'Canonical cell'. None or two die."""
    miss = [k for k in ("dx", "dy", "yaw_deg", "dz") if not isinstance(canon.get(k), (int, float))]
    if miss:
        raise ValueError(f"[AUG_POLICY] canon lacks {miss}")
    scale_of(grid)
    frac = __getattr__("CANON_MATCH_FRAC")
    tol_m = frac * float(grid["step"])
    gap = _min_gap(theta_deg)
    tol_y = frac * (gap if gap else float(grid["yaw_hi"]) - float(grid["yaw_lo"]))
    hit = [
        i
        for i, (t, d, z) in enumerate(zip(theta_deg, d_abs, dz, strict=True))
        if float(z) == float(canon["dz"])
        and abs(_wrap(float(t) - float(canon["yaw_deg"]))) <= tol_y
        and abs(float(d[0]) - float(canon["dx"])) <= tol_m
        and abs(float(d[1]) - float(canon["dy"])) <= tol_m
    ]
    if len(hit) != 1:
        raise ValueError(
            f"[AUG_POLICY] {len(hit)} canonical cells (must be 1): canon {canon} · "
            f"tolerance {tol_m:.2e} m · {tol_y:.3g}°"
        )
    return hit[0]


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _rot(deg: float, x: float, y: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return c * x - s * y, s * x + c * y


def _canon_xy(canon: dict) -> tuple[float, float, float, float, float]:
    p = canon.get("pivot")
    if not isinstance(p, (list, tuple)) or len(p) < 2:
        raise ValueError(f"[AUG_POLICY] canon pivot {p!r}")
    return float(canon["yaw_deg"]), float(p[0]), float(p[1]), float(canon["dx"]), float(canon["dy"])


def abs_to_variant(theta_abs_deg: float, d_abs, canon: dict, q_xy) -> tuple[float, tuple[float, float]]:
    """Judge cell (absolute; capture = tree coordinates) -> variant (ψ deg, d_scene m); contract 'coordinates':
    ψ = θa - θc · v = d_a + p - q - R(ψ)(p + dc - q) · d_scene = scene vector(v) (augment.scene owns the scene/capture
    transform). q = the frame-0 object xy mean of the baked tree (placement Src.object_xy_frame0)."""
    from dexengine.record.augment import scene as S

    th_c, px, py, dcx, dcy = _canon_xy(canon)
    psi = _wrap(float(theta_abs_deg) - th_c)
    qx, qy = float(q_xy[0]), float(q_xy[1])
    rx, ry = _rot(psi, px + dcx - qx, py + dcy - qy)
    v = (float(d_abs[0]) + px - qx - rx, float(d_abs[1]) + py - qy - ry)
    ds = S.tree_to_scene_pts([v[0], v[1], 0.0])
    return psi, (float(ds[0]), float(ds[1]))


def parse_done(d) -> tuple[int, int, int]:
    """Chunk end line (dict) -> (clips, cells, passes); done not true or missing fields die."""
    if not isinstance(d, dict) or d.get("done") is not True:
        raise ValueError(f"[AUG_POLICY] not a done line: {str(d)[:120]!r}")
    miss = [k for k in DONE_KEYS if k not in d]
    if miss:
        raise ValueError(f"[AUG_POLICY] done line lacks {miss}")
    return int(d["clips"]), int(d["cells"]), int(d["pass"])


def admit_level(dx: float, dy: float, yaw: float) -> int:
    """Acceptance level of a variant (ADMIT_LADDER index; 0 = strictest); module comment. Non-number values die."""
    v = [float(dx), float(dy), float(yaw)]
    if not all(math.isfinite(x) for x in v):
        raise ValueError(f"[AUG_POLICY] acceptance level undecidable dx={dx} dy={dy} yaw={yaw}")
    for i, step in enumerate(ADMIT_LADDER):
        if step is None or abs(v[0]) > step[0] or abs(v[1]) > step[0] or abs(_wrap(v[2])) >= step[1]:
            return i
    raise ValueError("[AUG_POLICY] the last ADMIT_LADDER level is not None")


def ladder_name(i: int) -> str:
    s = ADMIT_LADDER[i]
    return f"L{i}" + (f"({s[0] * 100:g}cm/{s[1]:g}°)" if s else "(only the canonical cell excluded)")


def phys_set(variants: list[dict], n: int | None = None) -> list[str]:
    """Names of the first n variants outside the canonical cell (k != 0) that passed the pre gate (status ok), by
    (acceptance level, sample order k): the set sent to physics (ladder: cells of strict levels get physics first)."""
    n = int(SAMPLE["phys_max"] if n is None else n)
    ks = [v["k"] for v in variants]
    if len(set(ks)) != len(ks):
        raise ValueError("[AUG_POLICY] variant k overlap: no order")
    ok = [v for v in variants if v["k"] != 0 and v["status"] == "ok"]
    return [v["name"] for v in sorted(ok, key=lambda v: (admit_level(v["dx"], v["dy"], v["yaw"]), v["k"]))][:n]


# ── choice ────────────────────────────────────────────────────────────────────
def norm_point(v: dict, scale: dict) -> tuple[float, float, float]:
    """Variant row -> normalised coordinates (dx/x, dy/y, yaw/yaw) (scale of the scale_of form); missing fields die."""
    miss = [k for k in ("dx", "dy", "yaw") if not isinstance(v.get(k), (int, float))]
    if miss:
        raise ValueError(f"[AUG_POLICY] variant {v.get('name')} lacks {miss}: no choice coordinates")
    bad = [k for k in ("x", "y", "yaw") if not isinstance((scale or {}).get(k), (int, float)) or scale[k] <= 0]
    if bad:
        raise ValueError(f"[AUG_POLICY] scale {scale!r} lacks {bad} (scale_of)")
    return (float(v["dx"]) / scale["x"], float(v["dy"]) / scale["y"], float(v["yaw"]) / scale["yaw"])


def pick_per_clip(variants: list[dict], salvaged, scale: dict, n: int | None = None) -> list[str]:
    """variants = variant rows of the generation record (k, name, dx, dy, yaw) · salvaged = set of salvaged names (or
    name -> bool) · scale = the scale (scale_of) -> chosen names (chosen order, <= n); rule PICK_RULE (module doc) ·
    ladder (ADMIT_LADDER: fill the farthest points with candidates accepted up to level L, and when short continue with
    the next level's candidates = the strictest level that fills the needed count)."""
    n = PER_CLIP if n is None else int(n)
    if n < 0:
        raise ValueError(f"[AUG_POLICY] n={n} < 0")
    ks = [v["k"] for v in variants]
    if len(set(ks)) != len(ks):
        raise ValueError("[AUG_POLICY] variant k overlap: no order")
    pts = {v["name"]: norm_point(v, scale) for v in variants}
    ok = (lambda nm: bool(salvaged.get(nm, False))) if isinstance(salvaged, dict) else (lambda nm: nm in salvaged)
    lv = {v["name"]: admit_level(v["dx"], v["dy"], v["yaw"]) for v in variants if v["k"] != 0}
    fixed = [(0.0, 0.0, 0.0)]  # original placement (identity): a fixed point (never chosen)
    out: list[str] = []
    for top in range(len(ADMIT_LADDER)):
        cand = sorted(
            (v for v in variants if v["k"] != 0 and ok(v["name"]) and lv[v["name"]] <= top and v["name"] not in out),
            key=lambda v: v["k"],
        )
        while cand and len(out) < n:
            best = max(cand, key=lambda v: (min(math.dist(pts[v["name"]], p) for p in fixed), -v["k"]))
            out.append(best["name"])
            fixed.append(pts[best["name"]])
            cand.remove(best)
    return out


def pick_meta(variants: list[dict], picked: list[str], scale: dict) -> dict:
    """Record of the chosen set: rule name, scale, (k, dx, dy, yaw, distance to origin) of the chosen variants, minimum
    pair distance (among the chosen; None below 2), minimum distance to origin (normalised) and the ladder (admit: level
    per chosen variant · the loosest level used to fill the first SECURE slots = the level this clip used · secured
    count)."""
    by = {v["name"]: v for v in variants}
    pts = [norm_point(by[nm], scale) for nm in picked]
    pairs = [math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :]]
    o = (0.0, 0.0, 0.0)
    lv = [admit_level(by[nm]["dx"], by[nm]["dy"], by[nm]["yaw"]) for nm in picked]
    return {
        "rule": PICK_RULE,
        "norm": dict(scale),
        "picked": [
            {
                "name": nm,
                "k": by[nm]["k"],
                "dx": by[nm]["dx"],
                "dy": by[nm]["dy"],
                "yaw": by[nm]["yaw"],
                "d_identity": round(math.dist(p, o), 6),
                "admit": lvl,
            }
            for nm, p, lvl in zip(picked, pts, lv, strict=True)
        ],
        "min_pair": round(min(pairs), 6) if pairs else None,
        "min_identity": round(min(math.dist(p, o) for p in pts), 6) if pts else None,
        "admit": {
            "ladder": [list(s) if s else None for s in ADMIT_LADDER],
            "secure": SECURE,
            "level": max(lv[:SECURE]) if lv else None,
            "secured": min(SECURE, len(picked)),
        },
    }


def admit_counts(variants: list[dict]) -> dict:
    """Ladder count of one clip's generation record (cells outside the canonical one): candidates m = judge-passed
    cells (why != judge) · accepted per level (cumulative) · cells excluded at L0 n = m - accepted at L0."""
    cand = [v for v in variants if v["k"] != 0 and v.get("why") != "judge"]
    lv = [admit_level(v["dx"], v["dy"], v["yaw"]) for v in cand]
    cum = [sum(x <= i for x in lv) for i in range(len(ADMIT_LADDER))]
    return {"cand": len(cand), "cum": cum, "excluded_l0": len(cand) - cum[0]}


def post_ref(pv00_values: list[dict]) -> dict:
    """Term values of the original placement rollouts (collision.term_values form) -> max per term (None without
    values)."""
    out = {}
    for t in POST_TERMS:
        v = [r[t] for r in pv00_values if r.get(t) is not None]
        out[t] = max(v) if v else None
    return out


def post_limit_mm(floor_mm: float, ref_mm: float | None) -> float:
    return float(floor_mm) if ref_mm is None else max(float(floor_mm), float(ref_mm) + POST_REL_MM)


def post_ok(values: dict, ref: dict | None, floors: dict) -> tuple[bool, str]:
    """One rollout: per term value <= post_limit_mm(threshold, original placement value). Unknown terms or terms
    without a threshold die."""
    unknown = set(values) - set(POST_TERMS)
    if unknown:
        raise ValueError(f"[AUG_POLICY] unknown terms {sorted(unknown)}")
    for t in POST_TERMS:
        v = values.get(t)
        if v is None:
            continue
        if t not in floors:
            raise ValueError(f"[AUG_POLICY] term {t} has no judge threshold")
        lim = post_limit_mm(floors[t], (ref or {}).get(t))
        if v > lim:
            return False, f"{t} {v:.1f}mm > {lim:.1f}mm"
    return True, ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", action="store_true")
    a = ap.parse_args()
    if a.per_clip:
        print(PER_CLIP)


FOVX_RULE = "fovx_first"


def is_fovx(v: dict, k0: int | None) -> bool:
    return k0 is not None and int(v["k"]) > k0


def fovx_phys_set(variants: list[dict], k0: int | None, n: int | None = None) -> list[str]:
    if k0 is None:
        return phys_set(variants, n)
    n = int(SAMPLE["phys_max"] if n is None else n)
    first = phys_set([v for v in variants if is_fovx(v, k0)], n)
    return first + phys_set([v for v in variants if not is_fovx(v, k0)], n - len(first))


def fovx_pick_per_clip(variants: list[dict], salvaged, scale: dict, n: int, k0: int | None) -> list[str]:
    if k0 is None:
        return pick_per_clip(variants, salvaged, scale, n)
    first = pick_per_clip([v for v in variants if is_fovx(v, k0)], salvaged, scale, n)
    return first + pick_per_clip([v for v in variants if not is_fovx(v, k0)], salvaged, scale, n - len(first))
