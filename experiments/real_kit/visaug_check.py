#!/usr/bin/env python3
"""GR00T visual-augmentation set contract — the DP FT set's episodes with the ours normalisation (the task's init set:
Pour, Sweep v2, P&P v1).

  python -m experiments.real_kit.visaug_check <set dir> --manifest <FT vis_manifest.tsv> --ref-meta <dir with the ours
  set meta files>

GR00T overwrites normalisation with the dataset's meta/stats.json, so a subset's own stats would disagree with the ours
checkpoint. Dies (last line [E5_VISAUG] FAIL …) unless:
  stats   meta/stats.json is byte-equal to the ours file · its min/max match field by field · it meets stats_rule
  meta    meta/modality.json byte-equal · info.json has the same keys · same features (dtype · shape · names) · fps
  and
          path templates equal · two cameras
  count   manifest rows = episodes.jsonl lines = info.total_episodes = unique e5_source keys = parquet files = mp4 / 2
  ·
          frames: manifest = episodes = info · key sets of manifest and episodes equal
  range   every episode's state/action min/max (episodes_stats.jsonl) inside the ours min/max — the episodes are ours
          train variants, so anything outside means they are not, or the stats are from another set
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys

from experiments.policies.data import stats_rule as SR  # the one stats rule

TAG = "[E5_VISAUG]"
EPS = 1e-5  # float32 stats next to float32 stats written through JSON
CAMERAS = 2


def die(m: str) -> None:
    print(f"{TAG} FAIL {m}", flush=True)
    sys.exit(1)


def md5(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load(p: str):
    with open(p) as f:
        return json.load(f)


def flat(x):
    if isinstance(x, list):
        for y in x:
            yield from flat(y)
    else:
        yield x


def _parse_args():
    """Parse the command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref-meta", required=True)
    a = ap.parse_args()
    return a


def _check_stats(m, r):
    """stats.json equals the ours set (bytes, keys, min/max) and passes the stats rule."""
    if md5(f"{m}/stats.json") != md5(f"{r}/stats.json"):
        die("stats.json bytes ≠ ours")
    info, rinfo = load(f"{m}/info.json"), load(f"{r}/info.json")
    st, rst = load(f"{m}/stats.json"), load(f"{r}/stats.json")
    fails = [(k, s) for k in rst for s in ("min", "max") if st.get(k, {}).get(s) != rst[k].get(s)]
    if set(st) != set(rst) or fails:
        die(f"stats min/max ≠ ours {fails[:3]}")
    pr = SR.problems(info["features"], st)
    if pr:
        die(f"stats rule {pr[:3]}")
    return info, rinfo, rst


def _check_meta(m, r, info, rinfo):
    """modality.json and info.json equal the ours set (keys, fields, features, two cameras)."""
    if md5(f"{m}/modality.json") != md5(f"{r}/modality.json"):
        die("modality.json bytes ≠ ours")
    if set(info) != set(rinfo):
        die(f"info keys differ {sorted(set(info) ^ set(rinfo))}")
    for k in ("fps", "data_path", "video_path", "codebase_version", "chunks_size", "robot_type"):
        if info.get(k) != rinfo.get(k):
            die(f"info {k} {info.get(k)!r} ≠ ours {rinfo.get(k)!r}")
    if set(info["features"]) != set(rinfo["features"]):
        die(f"features differ {sorted(set(info['features']) ^ set(rinfo['features']))}")
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    if len(cams) != CAMERAS:
        die(f"cameras {cams} ≠ {CAMERAS}")
    for k, v in info["features"].items():
        w = rinfo["features"][k]
        for j in ("dtype", "shape", "names"):
            if v.get(j) != w.get(j):
                die(f"feature {k}.{j} {v.get(j)!r} ≠ ours {w.get(j)!r}")
    return cams


def _check_counts(a, m, info):
    """Manifest rows = episodes = info = unique (clip, look) = parquet = mp4 / cameras; frame totals agree."""
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    with open(f"{m}/episodes.jsonl") as f:
        eps = [json.loads(ln) for ln in f if ln.strip()]

    def key(d):
        return (d["dataset"], d["root"], d["clip"], d["round"], d["look"])

    ca, cb = collections.Counter(key(e["e5_source"]) for e in eps), collections.Counter(key(x) for x in rows)
    if set(ca) != set(cb) or max(ca.values()) != 1 or max(cb.values()) != 1:
        die(
            f"manifest ≠ episodes: {len(set(ca) ^ set(cb))} keys differ · repeats {max(ca.values())}/{max(cb.values())}"
        )
    uniq = len({(e["e5_source"]["clip"], e["e5_source"]["look"]) for e in eps})
    pq = sum(len(fs) for _, _, fs in os.walk(os.path.join(a.set, "data")))
    mp4 = sum(len(fs) for _, _, fs in os.walk(os.path.join(a.set, "videos")))
    if not len(rows) == len(eps) == info["total_episodes"] == uniq == pq == mp4 / CAMERAS:
        die(
            f"counts manifest {len(rows)} · episodes {len(eps)} · info {info['total_episodes']} · unique (clip, look) "
            f"{uniq} · parquet {pq} · mp4 {mp4}"
        )
    fa, fb = sum(int(e["length"]) for e in eps), sum(int(x["video_frames"]) for x in rows)
    if not fa == fb == info["total_frames"]:
        die(f"frames episodes {fa} · manifest {fb} · info {info['total_frames']}")
    return eps, fa


def _check_ranges(m, info, rst):
    """Every episode stays inside the ours min/max (float features, timestamp excluded)."""
    out = collections.Counter()
    with open(f"{m}/episodes_stats.jsonl") as f:
        for ln in f:
            if not ln.strip():
                continue
            es = json.loads(ln)["stats"]
            for k in SR.float_features(info["features"]):
                if k == "timestamp" or k not in es:
                    continue
                lo = list(flat(rst[k]["min"]))
                hi = list(flat(rst[k]["max"]))
                if any(x < y - EPS for x, y in zip(flat(es[k]["min"]), lo, strict=False)) or any(
                    x > y + EPS for x, y in zip(flat(es[k]["max"]), hi, strict=False)
                ):
                    out[k] += 1
    if out:
        die(f"episodes outside the ours min/max {dict(out)}")


def main() -> int:
    a = _parse_args()
    m, r = os.path.join(a.set, "meta"), a.ref_meta
    # stats
    info, rinfo, rst = _check_stats(m, r)
    # meta
    cams = _check_meta(m, r, info, rinfo)
    # count
    eps, fa = _check_counts(a, m, info)
    # range
    _check_ranges(m, info, rst)
    print(
        f"{TAG} OK episodes {len(eps)} = manifest = info = unique = parquet = mp4/2 · frames {fa} · stats.json = ours "
        f"{md5(f'{m}/stats.json')[:12]} · modality = ours · {len(info['features'])} features · cameras {cams}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
