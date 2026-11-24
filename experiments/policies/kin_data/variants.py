#!/usr/bin/env python3
"""E5 kinematic baseline — the **only owner** of the placement variant distribution.

Rule: per dataset, the empirical distribution of the variant offsets in our final data (meta/episodes.jsonl of the final
  augmented root) — the episode meta selection (dx, dy, yaw) = variant relative to the source placement (baked tree)
  (augment policy.abs_to_variant: d = scene vector m, yaw = ψ °, pivot = mean frame-0 object xy of that tree — the
  definition place_src.make_variant uses). One variant = one point (several episodes of one variant count once).
Draws: per clip seed = sha256(SEED · dataset · clip), sampling with replacement from the distribution, until PER_CLIP
  (augment policy) draws pass the reach rule (reach.py), at most MAX_DRAWS. Short → as many as passed (the shortfall
  is counted in the manifest).
Frozen values: the JSON derive writes (rule text, source file sha256, point list per dataset) — the render chain reads
  only this JSON.
  python3 experiments/policies/kin_data/variants.py derive --variants <variants.tsv> --sources <sources.json> --out
  <json> (variants.tsv and sources.json are extracted read-only from the final root's episodes.jsonl, with source sha)
Exact placements (our scene = kinematic scene pairs): the same JSON form (rule, per_clip, sources, datasets) plus
  mode = "exact", exact_rule, datasets[ds].clips[clip] = [[k, dx, dy, yaw, our scene], …]. With this JSON, draws does
  not sample but yields the clip's own variants (k, offsets) in order; need = their count (not PER_CLIP).
  python3 experiments/policies/kin_data/variants.py exact --scenes <tsv: dataset base_clip clip placement k dx dy yaw
      scene> --sources <json> --out <json>
  A JSON without mode behaves as before.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

MARK = "[KIN_VARIANTS]"
SEED = 42
MAX_DRAWS = 20
RULE = (
    "per-dataset empirical distribution of our final data's variant offsets (episode meta selection dx, dy [scene m], "
    "yaw [deg] relative to the orig placement; one point per variant); per clip draw with replacement, rng seed = "
    f"sha256(SEED={SEED}, dataset, clip); keep draws passing kin_reach until aug_policy.PER_CLIP pass or "
    f"MAX_DRAWS={MAX_DRAWS} draws"
)

MODE_EXACT = "exact"
RULE_EXACT = (
    "exact placements (E5 matched pair): per clip, the variants of our final-data scenes "
    "listed in the scene table, each with our variant index k and offsets dx, dy [scene m], yaw [deg] "
    "from the episode meta selection, in table order; no draw, no reach filter in this module "
    "(need = number listed)"
)


def per_clip() -> int:
    from dexengine.record.augment import policy as AP  # noqa: PLC0415 — owner of the variant count per clip

    return int(AP.PER_CLIP)


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("rule") != RULE:
        raise SystemExit(f"{MARK} {path}: rule text differs from this module")
    if "mode" in d and (d["mode"] != MODE_EXACT or d.get("exact_rule") != RULE_EXACT):
        raise SystemExit(f"{MARK} {path}: mode {d.get('mode')!r} · exact-placement rule text differs")
    return d


def is_exact(dist: dict) -> bool:
    return dist.get("mode") == MODE_EXACT


def need(dist: dict, ds: str, clip: str) -> int:
    """Variants to keep for one clip — distribution mode = augment policy.PER_CLIP · exact mode = the clip's listed
    variant count.
    """
    if is_exact(dist):
        return len(dist["datasets"].get(ds, {}).get("clips", {}).get(clip, []))
    return per_clip()


def draws(dist: dict, ds: str, clip: str):
    """(j, dx, dy, yaw, source variant name) in order — j = 1 … MAX_DRAWS. Exact mode = (our k, our offsets, our
    scene).
    """
    if is_exact(dist):
        for k, dx, dy, yaw, scene in dist["datasets"].get(ds, {}).get("clips", {}).get(clip, []):
            yield int(k), float(dx), float(dy), float(yaw), scene
        return
    pts = dist["datasets"][ds]["points"]
    h = hashlib.sha256(f"{SEED}\0{ds}\0{clip}".encode()).hexdigest()
    rng = np.random.default_rng(int(h[:16], 16))
    for j in range(1, MAX_DRAWS + 1):
        dx, dy, yaw, src = pts[int(rng.integers(len(pts)))]
        yield j, float(dx), float(dy), float(yaw), src


def derive(a) -> int:
    src = json.loads(Path(a.sources).read_text())
    with open(a.variants) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = {
        "rule": RULE,
        "per_clip": per_clip(),
        "sources": src["sources"],
        "variants_tsv_sha256": hashlib.sha256(Path(a.variants).read_bytes()).hexdigest()[:16],
        "datasets": {},
    }
    for ds in sorted({r["ds"] for r in rows}):
        rr = sorted((r for r in rows if r["ds"] == ds), key=lambda r: r["variant"])
        pts = [[float(r["dx"]), float(r["dy"]), float(r["yaw"]), r["variant"]] for r in rr]
        a_ = np.array([p[:3] for p in pts])
        out["datasets"][ds] = {
            "n": len(pts),
            "points": pts,
            "pct": {
                k: [round(float(x), 4) for x in np.percentile(a_[:, i], [5, 25, 50, 75, 95])]
                for i, k in enumerate(("dx", "dy", "yaw"))
            },
        }
        print(
            f"{MARK} {ds}: variants {len(pts)} · dx median {out['datasets'][ds]['pct']['dx'][2]} · dy median "
            f"{out['datasets'][ds]['pct']['dy'][2]} · yaw median {out['datasets'][ds]['pct']['yaw'][2]}",
            flush=True,
        )
    tmp = a.out + f".tmp{os.getpid()}"
    Path(tmp).write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.out)
    print(f"{MARK} DONE → {a.out}", flush=True)
    return 0


def exact(a) -> int:
    """Our scene table → exact-placement JSON (same form, mode exact). Only variant scenes (source-placement scenes
    keep the source tree).
    """
    with open(a.scenes) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    src = json.loads(Path(a.sources).read_text())
    out = {
        "rule": RULE,
        "per_clip": per_clip(),
        "sources": src["sources"],
        "variants_tsv_sha256": hashlib.sha256(Path(a.scenes).read_bytes()).hexdigest()[:16],
        "datasets": {},
        "mode": MODE_EXACT,
        "exact_rule": RULE_EXACT,
    }
    seen = {}
    for r in rows:
        if r["placement"] not in ("orig", "variant"):
            raise SystemExit(f"{MARK} {r['scene']}: placement {r['placement']!r}")
        if r["placement"] != "variant":
            continue
        k = int(r["k"])
        if r["clip"] != f"{r['base_clip']}_pv{k:02d}_pl_rt":
            raise SystemExit(f"{MARK} {r['scene']}: clip {r['clip']} ≠ {r['base_clip']}_pv{k:02d}_pl_rt")
        key = (r["dataset"], r["base_clip"], k)
        v = [k, float(r["dx"]), float(r["dy"]), float(r["yaw"]), r["scene"]]
        if key in seen:  # another task row of the same variant — offsets must be equal
            if seen[key][1:4] != v[1:4]:
                raise SystemExit(f"{MARK} {key}: one variant with two offsets {seen[key]} ≠ {v}")
            continue
        seen[key] = v
        dd = out["datasets"].setdefault(r["dataset"], {"clips": {}})
        dd["clips"].setdefault(r["base_clip"], []).append(v)
    for ds, dd in sorted(out["datasets"].items()):
        pts = [[v[1], v[2], v[3], v[4]] for c in sorted(dd["clips"]) for v in dd["clips"][c]]
        a_ = np.array([p[:3] for p in pts])
        pct = {
            k: [round(float(x), 4) for x in np.percentile(a_[:, i], [5, 25, 50, 75, 95])]
            for i, k in enumerate(("dx", "dy", "yaw"))
        }
        dd.update(n=len(pts), points=pts, pct=pct)
        print(f"{MARK} exact {ds}: clips {len(dd['clips'])} · variants {len(pts)}", flush=True)
    tmp = a.out + f".tmp{os.getpid()}"
    Path(tmp).write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.out)
    print(f"{MARK} DONE exact → {a.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--variants", required=True)
    d.add_argument("--sources", required=True)
    d.add_argument("--out", required=True)
    x = sub.add_parser("exact", help="exact placements of our scene table (not the default branch · derive unchanged)")
    x.add_argument("--scenes", required=True)
    x.add_argument("--sources", required=True)
    x.add_argument("--out", required=True)
    a = ap.parse_args()
    return exact(a) if a.cmd == "exact" else derive(a)


if __name__ == "__main__":
    sys.exit(main())
