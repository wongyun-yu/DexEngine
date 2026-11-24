"""E5 policy dataset converter (convert.py), finalize phase: the meta/* files (info · episodes · tasks ·
episodes_stats · stats with q01/q99 · modality) and the in = out checks. convert.finalize calls these in order
and writes meta/e5_conversion.json itself."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.policies.data import stats_rule as SR
from experiments.policies.data.convert_episode import AXES, GROUPS, NAMES, TAG, die


def frame_stats(x: np.ndarray) -> dict:
    """Per-dimension stats over every frame (q01/q99 need the whole column) — the one formula of stats.json."""
    x = x.astype(np.float64)
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
        "count": [int(len(x))],
    }


def image_stats_agg(recs: list[dict], c: str, total: int) -> dict:
    """Dataset image stats of feature c from per-episode stats (recs: {"length", "stats": {c: …}}) — frame-weighted
    mean · pooled std · min of mins · max of maxes; the one formula."""
    m = np.array([np.ravel(r["stats"][c]["mean"]) for r in recs])
    s = np.array([np.ravel(r["stats"][c]["std"]) for r in recs])
    w = np.array([r["length"] for r in recs], dtype=np.float64)[:, None]
    mu = (m * w).sum(0) / w.sum()
    var = (((s**2) + (m - mu) ** 2) * w).sum(0) / w.sum()
    return {
        "min": [[[float(v)]] for v in np.min([np.ravel(r["stats"][c]["min"]) for r in recs], 0)],
        "max": [[[float(v)]] for v in np.max([np.ravel(r["stats"][c]["max"]) for r in recs], 0)],
        "mean": [[[float(v)]] for v in mu],
        "std": [[[float(v)]] for v in np.sqrt(var)],
        "count": [int(total)],
    }


def complete_float_stats(features: dict, stats: dict, parquets: list[str]) -> list[str]:
    """Fill, from the data, every float feature of info.json that lacks one of SR.STAT_KEYS (rule =
    experiments.policies.data.stats_rule); returns the filled names and dies if any is still incomplete. Other derived
    datasets (e.g. +force) import this."""
    need = SR.missing(features, stats)
    if need:
        cols: dict = {k: [] for k in need}
        for p in parquets:
            df = pd.read_parquet(p, columns=need)
            for k in need:
                v = df[k].to_numpy()
                cols[k].append(
                    np.stack([np.atleast_1d(np.asarray(e, np.float32)) for e in v])
                    if v.dtype == object
                    else v.astype(np.float32).reshape(len(v), -1)
                )
        for k in need:
            stats[k] = frame_stats(np.concatenate(cols[k]))
    bad = SR.missing(features, stats)
    if bad:
        die(f"stats.json still lacks {SR.STAT_KEYS} for float features {bad} — GR00T would rewrite it")
    return need


def load_recs(out: str, jobs: list[dict]) -> tuple[list[dict], set]:
    """Every episode's _work record (each must exist); dies unless the unique key sets in and out agree."""
    N = len(jobs)
    recs = []
    for j in jobs:
        f = os.path.join(out, "_work", f"ep_{j['episode_index']:06d}.json")
        if not os.path.isfile(f):
            die(f"episode {j['episode_index']} not converted ({f} missing)")
        recs.append(json.loads(Path(f).read_text()))
    keys_in = {tuple(j["key"]) for j in jobs}
    keys_out = {tuple(r["key"]) for r in recs}
    if len(keys_in) != N or keys_in != keys_out:
        die(f"unique key sets differ: in {len(keys_in)} out {len(keys_out)} (rows {N})")
    return recs, keys_in


def column_stats(out: str, jobs: list[dict], recs: list[dict], a, total: int) -> dict:
    """stats.json values: state · action · environment state over every frame, images from the episodes."""
    # stats over every frame (q01/q99 need the whole column)
    acc = {"observation.state": [], "action": [], "observation.environment_state": []}
    for j in jobs:
        i = j["episode_index"]
        p = os.path.join(out, f"data/chunk-{i // a.chunks_size:03d}/episode_{i:06d}.parquet")
        df = pd.read_parquet(p, columns=list(acc))
        for c in acc:
            acc[c].append(np.stack(df[c].to_numpy()))
    stats = {}
    for c, chunks in acc.items():
        x = np.concatenate(chunks).astype(np.float64)
        if len(x) != total:
            die(f"{c}: rows {len(x)} ≠ total frames {total}")
        stats[c] = frame_stats(x)
    for c in (f"observation.images.camera_{cam}" for cam in a.cameras):
        stats[c] = image_stats_agg(recs, c, total)
    return stats


def write_info(meta: str, src_info: dict, a, N: int, total: int, tasks: list[str]) -> dict:
    """meta/info.json (features: action · state · cameras · indices · axis); returns the features."""
    vfps = float(src_info["collect"]["video_fps"])
    vfps = int(vfps) if vfps.is_integer() else vfps
    cam_f = {}
    for cam in a.cameras:
        cam_f[f"observation.images.camera_{cam}"] = {
            "dtype": "video",
            "shape": [3, a.height, a.width],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": vfps,
                "video.height": a.height,
                "video.width": a.width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
                "source_key": f"observation.images.{cam}",
            },
        }
    features = {
        "action": {"dtype": "float32", "shape": [56], "names": list(NAMES)},
        "observation.state": {"dtype": "float32", "shape": [56], "names": list(NAMES)},
        **cam_f,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "axis_index": {"dtype": "int64", "shape": [1], "names": list(AXES)},
        "observation.environment_state": {"dtype": "float32", "shape": [len(AXES)], "names": list(AXES)},
    }
    n_chunks = (N + a.chunks_size - 1) // a.chunks_size
    info = {
        "codebase_version": "v2.1",
        "robot_type": src_info["robot_type"],
        "total_episodes": N,
        "total_frames": total,
        "total_tasks": len(tasks),
        "total_videos": N * len(a.cameras),
        "total_chunks": n_chunks,
        "chunks_size": a.chunks_size,
        "fps": vfps,
        "splits": {"train": f"0:{N}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(os.path.join(meta, "info.json"), "w") as fh:
        json.dump(info, fh, indent=4)
    return features


def write_episodes(meta: str, jobs: list[dict], recs: list[dict], tasks: list[str]) -> None:
    """meta/tasks.jsonl, meta/episodes.jsonl (with the e5_* source record) and meta/episodes_stats.jsonl."""
    with open(os.path.join(meta, "tasks.jsonl"), "w") as fh:
        for k, t in enumerate(tasks):
            fh.write(json.dumps({"task_index": k, "task": t}) + "\n")
    with (
        open(os.path.join(meta, "episodes.jsonl"), "w") as fh,
        open(os.path.join(meta, "episodes_stats.jsonl"), "w") as fs,
    ):
        for j, r in zip(jobs, recs, strict=True):
            row = j["row"]
            fh.write(
                json.dumps(
                    {
                        "episode_index": r["episode_index"],
                        "tasks": [tasks[j["task_index"]]],
                        "length": r["length"],
                        "e5_axis": row["axis"],
                        "e5_hand_use": row["hand_use"],
                        "e5_side": row["side"],
                        "e5_lang_source": row["lang_source"],
                        "e5_source": {
                            "dataset": row["dataset"],
                            "root": row["root"],
                            "clip": row["clip"],
                            "round": row["round"],
                            "look": row["look"],
                            "variant": row["variant"],
                            "base_clip": row["base_clip"],
                            "src_episode_index": r["src_episode_index"],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fs.write(json.dumps({"episode_index": r["episode_index"], "stats": r["stats"]}) + "\n")


def write_stats(out: str, jobs: list[dict], a, meta: str, features: dict, stats: dict) -> list[str]:
    """meta/stats.json, completed from the data and checked by the stats rule; returns the filled features."""
    parquets = [
        os.path.join(
            out, f"data/chunk-{j['episode_index'] // a.chunks_size:03d}/episode_{j['episode_index']:06d}.parquet"
        )
        for j in jobs
    ]
    stats_filled = complete_float_stats(features, stats, parquets)
    print(
        f"{TAG} stats complete: {len(SR.float_features(features))} float features have {'/'.join(SR.STAT_KEYS)} · "
        f"filled from the data: {stats_filled}",
        flush=True,
    )
    if pr := SR.problems(features, stats):  # the whole rule (video stats too) before the file is written
        die(f"stats rule: {len(pr)} problems ({'; '.join(pr[:4])})")
    with open(os.path.join(meta, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    return stats_filled


def write_modality(meta: str, a) -> None:
    """meta/modality.json: the GROUPS spans of state and action, the cameras and the task annotation."""
    spans, cur = {}, 0
    for g, names in GROUPS.items():
        spans[g] = {"start": cur, "end": cur + len(names)}
        cur += len(names)
    modality = {
        "state": spans,
        "action": spans,
        "video": {f"camera_{c}": {"original_key": f"observation.images.camera_{c}"} for c in a.cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(os.path.join(meta, "modality.json"), "w") as fh:
        json.dump(modality, fh, indent=2)


def count_files(out: str, a, N: int) -> tuple[int, int]:
    """File-count check (in = out): one parquet per episode, one mp4 per episode and camera."""
    n_pq = sum(1 for d, _, fs in os.walk(os.path.join(out, "data")) for f in fs if f.endswith(".parquet"))
    n_mp4 = sum(
        1 for d, _, fs in os.walk(os.path.join(out, "videos")) for f in fs if f.endswith(".mp4") and ".partial" not in f
    )
    if n_pq != N or n_mp4 != N * len(a.cameras):
        die(f"file counts: parquet {n_pq} (want {N}) · mp4 {n_mp4} (want {N * len(a.cameras)})")
    return n_pq, n_mp4
