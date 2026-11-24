"""Which clips of the Tab. 2 Ours arm rows were judged again with the object-tracking termination off.

The baselines roll out to the clip end; our judge ends an episode at the object-tracking termination (5 cm · 90°) and
the scorer counts the unreached steps as ∞. So the to-end rows rerun, per arm Ours row (K64 V1 · K32 · K16 · K4 ·
K1 MI1600 · K1 MI200 — never K8), the clips whose stored record has episodes_short > 0 (≥ 1 of the 64 scored rollouts
ended before the span — any cause, a superset of object-tracking ends). The other clips keep their stored rollouts
(all 64 already ran to the clip end).
Records: <ours-rl>/<job>/<run>/{meta.json, <clip>.json} (arm rows; meta rc 0 and iter = MI, the last final judge) and
<v1-prefill>/<clip>.json (the K64 V1 row). Every row must hold the 64 frozen clips (``experiments.scoring.clips``).
usage: python -m experiments.scoring.k_sweep.rerun_clips [--ours-rl DIR] [--v1-prefill DIR]   (defaults: workspace
results/)
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from experiments import workspace
from experiments.scoring import clips

V1_JOB = "v1prefill"  # the V1 row's job key when no clip → V1 bundle map is given


def rerun_sets(ours_rl: Path, v1_prefill: Path, v1_of: dict | None = None) -> dict:
    """{row: {original judge job: [clips to rerun]}}. ``v1_of`` = {clip: V1 run name O64_<nn>_…} keys the V1 row by its
    bundle's judge job (e5-na-e123-v1p-o<nn>); without it the V1 clips sit under ``V1_JOB``."""
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = collections.defaultdict(set)
    for m in sorted(Path(ours_rl).glob("*/*/meta.json")):
        meta = json.loads(m.read_text())
        jn = meta.get("job", "")
        mm = re.match(r"e5-na-e123-ojq-k(\d+)-(\d+)-m(\d+)$", jn)
        if not mm or meta.get("rc") != 0:
            continue
        K, MI = int(mm[1]), int(mm[3])
        if int(meta["iter"]) != MI or K == 8:
            continue
        for f in sorted(m.parent.glob("*.json")):
            if f.name == "meta.json":
                continue
            j = json.loads(f.read_text())
            if "error" in j:
                raise SystemExit(f"[RERUN_OFF] FAIL {jn}: stored record {f.name} has an error — not a 64/64 row")
            key = f"K{K}_MI{MI}"
            seen[key].add(j["clip"])
            if j["episodes_short"] > 0:
                rows[key][jn].append(j["clip"])
    for f in sorted(Path(v1_prefill).glob("*.json")):
        j = json.loads(f.read_text())
        if "error" in j:
            raise SystemExit(f"[RERUN_OFF] FAIL V1 record {f.name} has an error")
        seen["K64_V1"].add(j["clip"])
        if j["episodes_short"] > 0:
            nn = re.match(r"O64_(\d+)_", v1_of[j["clip"]])[1] if v1_of is not None else None
            job = f"e5-na-e123-v1p-o{nn}" if nn is not None else V1_JOB
            rows["K64_V1"][job].append(j["clip"])
    want = set(clips.e123_clips())
    for k, s in seen.items():
        if s != want:
            raise SystemExit(f"[RERUN_OFF] FAIL {k}: stored clips {len(s)} ≠ the 64 of E123_CLIPS_v1")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ours-rl", type=Path, default=None, help="default <workspace>/results/ours_rl")
    ap.add_argument("--v1-prefill", type=Path, default=None, help="default <workspace>/results/ours_v1prefill")
    a = ap.parse_args(argv)
    rows = rerun_sets(a.ours_rl or workspace.results("ours_rl"), a.v1_prefill or workspace.results("ours_v1prefill"))
    for row in sorted(rows):
        n = sum(len(v) for v in rows[row].values())
        print(f"[RERUN_OFF] {row}: {n} clips to rerun over {len(rows[row])} judge jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
