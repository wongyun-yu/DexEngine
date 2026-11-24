"""Synthetic pieces for the merge tests — the same write order as store.store (lerobot.write_parquet → video_features →
episode_stats), merge / verify under the final lock. Ported from the old merge fixture (merge · verify · dies ·
Src · make_piece). No GPU, no Isaac; needs pyarrow, ffmpeg and zstd (the caller skips without them)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from dexengine.record import piece as CP
from dexengine.record.render import visibility as HV
from dexengine.record.render import writer as CD
from dexengine.record.store import form as FM
from dexengine.record.store import lerobot as CL
from dexengine.record.store import merge as CMG
from dexengine.record.store import merge_check as MC
from dexengine.record.store import slots as KP
from dexengine.record.store import store as ST

L, N_VID = 50, 30  # 50 Hz x 50 steps → 30 fps video of 30 frames ((L-1)·0.02·30 + 1 = 30.4 · gap < 1)
DATA_PATH, VIDEO_PATH = ST.DATA_PATH, ST.VIDEO_PATH
DEPTH_PATH = "depth/chunk-{episode_chunk:03d}/" + ST.DEPTH_KEY + "/episode_{episode_index:06d}" + CD.EXT
FORCE_PATH = "forces/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"
JOINTS = [f"j{k}" for k in range(3)]


def merge(fin, piece, pid):
    """The driver's merge_step form — merge_piece under piece.final_lock."""
    with CP.final_lock(Path(fin)):
        return CMG.merge_piece(fin, piece, pid)


def verify(fin, frames=True):
    with CP.final_lock(Path(fin)):
        return MC.verify_final(fin, frames=frames)


def dies(fn, *a, want: str = "", **kw) -> str | None:
    """The message when it dies (<other reason> when want is not in it) · None when it does not die."""
    try:
        fn(*a, **kw)
    except SystemExit as ex:
        msg = str(ex)
        return msg if want in msg else f"<other reason> {msg}"
    return None


class Src:
    """Source files (one video · two depth files with different content)."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.mp4 = root / "testsrc.mp4"
        subprocess.run(
            [
                CL.ffmpeg_bin(),
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30",
                "-frames:v",
                str(N_VID),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.mp4),
            ],
            check=True,
        )
        self.depth = []
        for k in range(2):
            p = root / f"depth{k}{CD.EXT}"
            CD.write(p, np.full((N_VID, 4, 4), k, dtype=np.uint16), level=1)
            self.depth.append(p)


def _write_fixture_episodes(clips, tasks, eps, index0, cs, feats, ds, src, round_):
    """Parquet, video and depth files plus the episode lines of every fixture clip."""
    for clip, n, task in clips:
        ti = tasks.setdefault(task, len(tasks))
        for k in range(n):
            i = len(eps)
            name = clip if k % 2 == 0 else clip.replace("_pl_rt", f"_pv{k:02d}_pl_rt")
            cols = {
                "timestamp": np.arange(L) * 0.02,
                "frame_index": np.arange(L),
                "observation.joint_pos.robot": np.full((L, 3), i, dtype=np.float32),
            }
            extra = {
                "episode_index": np.full(L, i),
                "index": index0 + np.arange(L),
                "task_index": np.full(L, ti),
                "video_frame_index": np.clip(np.rint(np.arange(L) * 0.02 * 30).astype(np.int64), 0, N_VID - 1),
            }
            fm = {"episode_chunk": i // cs, "episode_index": i}
            feats.append(
                CL.write_parquet(ds / DATA_PATH.format(**fm), cols, extra, {"observation.joint_pos.robot": JOINTS})
            )
            for key in ST.VIDEO_KEYS.values():
                dst = ds / VIDEO_PATH.format(video_key=key, **fm)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src.mp4, dst)
            CD.place(src.depth[i % 2], ds / DEPTH_PATH.format(**fm))
            for t in (FORCE_PATH, HV.PATH):
                (ds / t.format(**fm)).parent.mkdir(parents=True, exist_ok=True)
                np.savez(ds / t.format(**fm), t=np.arange(L) + i)
            eps.append(
                dict(
                    episode_index=i,
                    tasks=[task],
                    length=L,
                    clip=name,
                    video_frames=N_VID,
                    round=round_,
                    look={"name": "D" if k == 0 else f"r{k}"},
                    selection={} if name == clip else {"source_clip": clip},
                    loophole=None,
                    **{CL.ABSENT: []},
                )
            )
            index0 += L
    return index0


def make_piece(
    root: Path,
    src: Src,
    clips: list[tuple[str, int, str]],
    cs: int = 1000,
    robot: str = "rby1_test",
    round_: str = "orig",
) -> Path:
    """clips = [(clip, episode count, task text)] → a piece (store.store form) · odd positions = placement variants ·
    round_ = the episodes' round field (merge overlap key = (round, clip) · card = (episode clip, look name))."""
    ds = root / "dataset"
    tasks, eps, feats, index0 = {}, [], [], 0
    index0 = _write_fixture_episodes(clips, tasks, eps, index0, cs, feats, ds, src, round_)
    pq = [ds / DATA_PATH.format(episode_chunk=i // cs, episode_index=i) for i in range(len(eps))]
    features = {f["name"]: {k: f[k] for k in ("dtype", "shape", "names")} for f in feats[0]}
    vids = {
        k: [ds / VIDEO_PATH.format(video_key=k, episode_chunk=i // cs, episode_index=i) for i in range(len(eps))]
        for k in ST.VIDEO_KEYS.values()
    }
    features |= CL.video_features(ST._ffprobe, vids, 30)
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    info = dict(
        codebase_version="v2.1",
        robot_type=robot,
        **CL.totals(len(eps), index0, len(tasks), cs, 3),
        chunks_size=cs,
        fps=50,
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
        features=features,
        collect=dict(
            video_fps=30,
            video_keys=list(ST.VIDEO_KEYS.values()),
            depth_path=DEPTH_PATH,
            depth_codec=CD.CODEC,
            force_path=FORCE_PATH,
            head_view_path=HV.PATH,
            loophole_counts={"odd_piece": 0, "assist_idle": 0},
            cameras={"fx": 1.0},
            **{
                FM.FORM_KEY: {
                    "form_version": FM.FORM_VERSION,
                    "K": 0,
                    "objects": FM.objects(0),
                    "k_source": {KP.STATED_KEY: True},
                }
            },
        ),
    )
    (ds / "meta/info.json").write_text(json.dumps(info, indent=1))
    (ds / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (ds / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": v, "task": k}) + "\n" for k, v in tasks.items())
    )
    (ds / "meta/collect.json").write_text(
        json.dumps(
            {"robot_type": robot, "cells": [{"campaign": "c", "bundle": root.name, "clips": [c for c, _, _ in clips]}]}
        )
    )
    CL.write_stats(
        ds,
        [
            {
                "episode_index": i,
                "stats": CL.episode_stats(
                    pq[i], features, [], {k: v[i] for k, v in vids.items()}, ST._ffprobe, CL.ffmpeg_bin()
                ),
            }
            for i in range(len(eps))
        ],
    )
    return ds
