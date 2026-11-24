#!/usr/bin/env python3
"""Extract from a data episode (LeRobot v2.1 parquet) the arrays an evaluation session reads; writes only `--out-dir`.

Arrays: action.joint_target.robot (row k = the target that produced state k), observation.joint_pos.robot,
video_frame_index, timestamp, every observation.object_pose.* column and the joint names (info.json feature names —
the two joint columns must name the same joints, else it dies).
Input: one job.json (`--job`, the episode it replays; file name = episode_key) or episode lines (`--episodes`, name =
base). pyarrow must be importable (put its folder on PYTHONPATH).
Marker `[E5_ARRAYS] <clip> rows n · …`, last line `[E5_ARRAYS] DONE <n>`."""

import argparse
import json
import os
import sys
from pathlib import Path

from experiments.textio import read_lines


def episode_key(ep: dict) -> str:
    """Name of the replayed data episode (variants `_pvNN` of one base and episodes of other roots do not collide)."""
    return f"{ep['base']}__{ep['dataset']}_{ep['root']}_{int(ep['episode_index'])}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", default="", help="episode lines jsonl (name = base)")
    ap.add_argument(
        "--job", default="", help="one job.json — the episode it replays (job.episode) · name = episode_key"
    )
    ap.add_argument("--clips", default="", help="base clip names (comma) — all when empty")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import numpy as np
    import pyarrow.parquet as pq

    want = {c for c in a.clips.split(",") if c}
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    if bool(a.episodes) == bool(a.job):
        raise SystemExit("[E5_ARRAYS] give exactly one of --episodes and --job")
    rows = (
        [(json.loads(ln), None) for ln in read_lines(a.episodes)]
        if a.episodes
        else [(json.loads(Path(a.job).read_text())["episode"], "job")]
    )
    for r, kind in rows:
        if r.get("root") == "missing" or (want and r["base"] not in want):
            continue
        if kind == "job" and not (r.get("files") or {}).get("parquet"):
            raise SystemExit(f"[E5_ARRAYS] the job's episode has no files.parquet ({a.job})")
        info = json.loads(Path(os.path.join(r["root_dir"], "meta", "info.json")).read_text())
        feats = info["features"]
        sn, an = feats["observation.joint_pos.robot"]["names"], feats["action.joint_target.robot"]["names"]
        if sn != an:
            raise SystemExit(f"[E5_ARRAYS] {r['base']}: state and target joint names differ")
        t = pq.read_table(r["files"]["parquet"]).to_pandas()
        obj = {
            c: np.stack(t[c].to_numpy()).astype(np.float32)
            for c in t.columns
            if c.startswith("observation.object_pose.")
        }
        out = os.path.join(a.out_dir, f"{episode_key(r) if kind == 'job' else r['base']}.npz")
        np.savez_compressed(
            out + ".tmp.npz",
            action_joint_target=np.stack(t["action.joint_target.robot"].to_numpy()).astype(np.float32),
            joint_pos=np.stack(t["observation.joint_pos.robot"].to_numpy()).astype(np.float32),
            video_frame_index=t["video_frame_index"].to_numpy().astype(np.int64),
            timestamp=t["timestamp"].to_numpy().astype(np.float64),
            names=json.dumps(list(sn)),
            object_keys=json.dumps(sorted(obj)),
            **{f"obj::{k}": v for k, v in obj.items()},
        )
        os.replace(out + ".tmp.npz", out)
        n += 1
        print(
            f"[E5_ARRAYS] {r['base']} rows {len(t)} · joints {len(sn)} · object columns {sorted(obj)} → {out}",
            flush=True,
        )
    print(f"[E5_ARRAYS] DONE {n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
