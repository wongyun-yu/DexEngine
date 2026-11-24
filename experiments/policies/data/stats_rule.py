"""The stats completeness rule a LeRobot v2.1 dataset of ours must meet — one owner, no dependencies (the shipper and
the data job import it without numpy).

GR00T's finetune factory (Isaac-GR00T 23ace64f gr00t/data/dataset/factory.py:59–63) calls
gr00t.data.stats.generate_stats, which keeps meta/stats.json only if EVERY float feature of meta/info.json has all of
STAT_KEYS (gr00t/data/stats.py:98–112 check_stats_validity); otherwise it silently rewrites stats.json from the
parquets (a stats.json without `timestamp` would have been rewritten).

Who reads what (the union is the rule — every maker writes it, every READY writer checks it):
  GR00T  every float feature: mean std min max q01 q99 (check_stats_validity above)
  DP     LeRobot 0.3.3 diffusion: VISUAL mean/std (MEAN_STD) · STATE/ACTION min/max (MIN_MAX) — float rule covers
  state/action Cosmos action q01/q99 (the state row uses the action's)
→ float features: STAT_KEYS, each a list of feature-shape length · video features: VIDEO_STAT_KEYS, each 3 × 1 × 1.
problems(features, stats) lists every violation (empty = the dataset may be READY);
`python -m experiments.policies.data.stats_rule <root>…` checks <root>/meta/{info,stats}.json and exits 1 on any
problem.
"""

from __future__ import annotations

import json
import sys

STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
VIDEO_STAT_KEYS = ("mean", "std", "min", "max")


def float_features(features: dict) -> list[str]:
    """The features GR00T checks: dtype contains 'float' (gr00t/data/stats.py generate_stats)."""
    return [k for k, ft in features.items() if "float" in str(ft.get("dtype"))]


def missing(features: dict, stats: dict) -> list[str]:
    """Float features whose stats entry lacks any of STAT_KEYS (empty list = GR00T keeps stats.json as is)."""
    return [
        k
        for k in float_features(features)
        if not isinstance(stats.get(k), dict) or not all(s in stats[k] for s in STAT_KEYS)
    ]


def video_features(features: dict) -> list[str]:
    return [k for k, ft in features.items() if ft.get("dtype") == "video"]


def _flat(x) -> list:
    out = []
    for e in x if isinstance(x, list) else [x]:
        out.extend(_flat(e) if isinstance(e, list) else [e])
    return out


def problems(features: dict, stats: dict) -> list[str]:
    """Every violation of the rule (feature.key: what) — empty list = complete for GR00T · DP · Cosmos."""
    out = []
    for k in float_features(features):
        n = 1
        for d in features[k].get("shape") or [1]:
            n *= int(d)
        st = stats.get(k)
        if not isinstance(st, dict):
            out.append(f"{k}: no stats")
            continue
        for s in STAT_KEYS:
            if s not in st:
                out.append(f"{k}.{s}: missing")
            elif len(_flat(st[s])) != n:
                out.append(f"{k}.{s}: {len(_flat(st[s]))} values ≠ shape {n}")
    for k in video_features(features):
        st = stats.get(k)
        if not isinstance(st, dict):
            out.append(f"{k}: no stats")
            continue
        for s in VIDEO_STAT_KEYS:
            if s not in st:
                out.append(f"{k}.{s}: missing")
            elif len(_flat(st[s])) != 3:
                out.append(f"{k}.{s}: {len(_flat(st[s]))} values ≠ 3 channels")
    return out


def check_root(root: str) -> list[str]:
    with open(f"{root}/meta/info.json") as f:
        feats = json.load(f)["features"]
    with open(f"{root}/meta/stats.json") as f:
        stats = json.load(f)
    return problems(feats, stats)


if __name__ == "__main__":
    bad = 0
    for r in sys.argv[1:]:
        try:
            pr = check_root(r)
        except (OSError, ValueError, KeyError) as e:
            pr = [f"unreadable: {e!r}"]
        bad += bool(pr)
        print(
            f"[E5_STATS_RULE] {'OK' if not pr else 'FAIL'} {r}" + (f" · {len(pr)}: " + "; ".join(pr[:6]) if pr else "")
        )
    sys.exit(1 if bad or len(sys.argv) < 2 else 0)
