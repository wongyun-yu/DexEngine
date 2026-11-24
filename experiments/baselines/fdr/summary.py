"""FlashDexRetarget scoring row summary: the 64 per-clip score records of one tag -> <record dir>/summary.json.

A clip is scored when its record carries steps_scored, the rollout's motion name, N worlds and no missing step; a clip
the authors' rules dropped before training (fdr_dropped) has no rollout and scores 0; anything else is a failure.
count_ok is true only when 64 = scored + dropped, nothing failed and the scored clips are exactly the recorder's
(meta.json). A clip passes SP / MP when >= 1 of its N rollouts passes. Prints the [SCORE] line (summary without
per_clip) and a
[FDR_RECORD] COUNT_OK|COUNT_FAIL line; exit 0 only on COUNT_OK.
usage: summary.py <record dir> <clips.tsv: clip \\t motion name> <N> <tag> [--eval-gpu <card name>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def summarize(out_dir: str, clips: list, N: int, tag: str, eval_gpu: str | None) -> dict:
    meta = json.loads(Path(os.path.join(out_dir, "meta.json")).read_text())
    per, scored, dropped, failed = {}, [], {}, []
    for c, taco in clips:
        f = os.path.join(out_dir, c + ".json")
        try:
            j = json.loads(Path(f).read_text())
        except (OSError, ValueError):  # missing or unreadable record: the clip counts as failed
            failed.append(c)
            continue
        if j.get("fdr_dropped"):
            if taco in meta["clips"]:  # a clip with a rollout is never a drop
                failed.append(c)
                continue
            dropped[c] = j["fdr_dropped"]
        elif (
            j.get("steps_scored")
            and j.get("fdr", {}).get("motion") == taco
            and j.get("worlds") == N
            and j.get("steps_missing") == 0
        ):
            scored.append(c)
        else:
            failed.append(c)
            continue
        w = int(j["worlds"])
        per[c] = {
            "sp_k": round(j["spider_sr"] * w),
            "mp_k": round(j["maniptrans_sr"] * w),
            "add_auc_best": j["add_auc_best"],
            "worlds": w,
        }
    ok = len(clips) == 64 and len(scored) + len(dropped) == 64 and not failed and len(scored) == len(meta["clips"])
    sp = sum(p["sp_k"] >= 1 for p in per.values())
    mp = sum(p["mp_k"] >= 1 for p in per.values())
    return {
        "tag": tag,
        "env_step": meta["env_step"],
        "replicas": N,
        "clips": len(clips),
        "scored": len(scored),
        "dropped": dropped,
        "failed": failed,
        "count_ok": ok,
        "sp_sr_64": round(100 * sp / 64, 1),
        "mp_sr_64": round(100 * mp / 64, 1),
        "add_auc_best_mean_64": round(sum(p["add_auc_best"] or 0 for p in per.values()) / 64, 4) if ok else None,
        "sp_pass": sp,
        "mp_pass": mp,
        "eval_gpu": eval_gpu or None,
        "variant": meta.get("variant"),
        "variant_sha256": meta.get("variant_sha256"),
        "rule": "a clip passes when ≥ 1 of its N rollouts passes (collect_results.py) · dropped = 0",
        "per_clip": per,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir")
    ap.add_argument("clips")
    ap.add_argument("replicas", type=int)
    ap.add_argument("tag")
    ap.add_argument("--eval-gpu", default=None, help="the card named in the summary")
    a = ap.parse_args(argv)
    with open(a.clips) as fh:
        clips = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    s = summarize(a.record_dir, clips, a.replicas, a.tag, a.eval_gpu)
    with open(os.path.join(a.record_dir, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print("[SCORE] " + json.dumps({k: v for k, v in s.items() if k != "per_clip"}))
    print(
        f"[FDR_RECORD] {'COUNT_OK' if s['count_ok'] else 'COUNT_FAIL'} clips={s['clips']} scored={s['scored']} "
        f"dropped={len(s['dropped'])} failed={len(s['failed'])} SP {s['sp_pass']}/64 MP {s['mp_pass']}/64"
    )
    return 0 if s["count_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
