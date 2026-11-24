"""One episode of the E5 policy dataset conversion (convert.py): the contract constants, the per-episode parquet + mp4
writer and the reuse of an earlier conversion's episode.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAG = "[E5_CONVERT]"


AXES = ("Lift", "Put", "Pour", "Sweep")


_FING = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


_KIND = (
    {"thumb": "cmc_flex", None: "mcp_flex"},
    {"thumb": "cmc_abd", None: "mcp_abd"},
    {"thumb": "mcp", None: "pip"},
    {"thumb": "ip", None: "dip"},
)


def _hands() -> tuple[str, ...]:
    """`hands_joints` order: joint kind × (left, right) × (thumb, index, middle, ring, pinky)."""
    out = []
    for kind in _KIND:
        for side in ("left", "right"):
            out += [f"{side}_{f}_{kind['thumb'] if f == 'thumb' else kind[None]}" for f in _FING]
    return tuple(out)


#: The E5 joint contract — the RB-Y1 + Wuji 2 joint groups minus the torso joints (D1: 56-D, no torso).
GROUPS: dict[str, tuple[str, ...]] = {
    "left_arm_joints": tuple(f"left_arm_{i}" for i in range(7)),
    "right_arm_joints": tuple(f"right_arm_{i}" for i in range(7)),
    "head_joints": ("head_0", "head_1"),
    "hands_joints": _hands(),
}


NAMES = tuple(n for g in GROUPS.values() for n in g)


def die(msg: str) -> None:
    sys.exit(f"{TAG} FAIL {msg}")


# ---------------------------------------------------------------- one episode


def reuse_one(job: dict) -> dict:
    """The same source episode converted by an earlier run with the same contract (--reuse): copy its videos, rewrite
    the parquet's numbering columns (episode_index · index · task_index) and keep its stats. The old record's key and
    length must equal this job's — anything else dies (the caller then converts it from the source)."""
    import shutil

    old, oi, i, out = job["reuse"][0], job["reuse"][1], job["episode_index"], job["out"]
    rec = json.loads(Path(os.path.join(old, "_work", f"ep_{oi:06d}.json")).read_text())
    if rec["key"] != job["key"] or int(rec["length"]) != int(job["e"]["video_frames"]):
        raise RuntimeError(
            f"reuse {old}#{oi}: key/length {rec['key']}/{rec['length']} ≠ {job['key']}/{job['e']['video_frames']}"
        )
    ocs, cs = job["chunks_size"], job["chunks_size"]
    opq = os.path.join(old, f"data/chunk-{oi // ocs:03d}/episode_{oi:06d}.parquet")
    df = pd.read_parquet(opq)
    n = len(df)
    if n != rec["length"]:
        raise RuntimeError(f"reuse {opq}: rows {n} ≠ record {rec['length']}")
    g0 = job["global_index"]
    df["episode_index"] = np.full(n, i, dtype=np.int64)
    df["index"] = np.arange(g0, g0 + n, dtype=np.int64)
    df["task_index"] = np.full(n, job["task_index"], dtype=np.int64)
    if int(df["axis_index"].iloc[0]) != job["axis_index"]:
        raise RuntimeError(f"reuse {opq}: axis_index differs")
    rel_pq = f"data/chunk-{i // cs:03d}/episode_{i:06d}.parquet"
    os.makedirs(os.path.join(out, os.path.dirname(rel_pq)), exist_ok=True)
    tmp = os.path.join(out, rel_pq + ".partial")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, os.path.join(out, rel_pq))
    for cam in job["cameras"]:
        dkey = f"observation.images.camera_{cam}"
        ov = os.path.join(old, f"videos/chunk-{oi // ocs:03d}/{dkey}/episode_{oi:06d}.mp4")
        nv = os.path.join(out, f"videos/chunk-{i // cs:03d}/{dkey}/episode_{i:06d}.mp4")
        os.makedirs(os.path.dirname(nv), exist_ok=True)
        shutil.copyfile(ov, nv + ".partial.mp4")
        os.replace(nv + ".partial.mp4", nv)
    rec = dict(rec, episode_index=i, reused_from=[old, oi])
    tmpj = os.path.join(out, "_work", f"ep_{i:06d}.json.partial")
    with open(tmpj, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmpj, os.path.join(out, "_work", f"ep_{i:06d}.json"))
    return rec


def work_one(job: dict) -> dict:
    return reuse_one(job) if job.get("reuse") else convert_one(job)


def _frame_rows(nvf, vfi, ts, vfps):
    """Source row of every video frame j: among the rows of frame j, the one whose timestamp is nearest j / fps."""
    rows = np.empty(nvf, dtype=np.int64)
    for j in range(nvf):
        cand = np.flatnonzero(vfi == j)
        rows[j] = cand[np.argmin(np.abs(ts[cand] - j / vfps))]
    return rows


def _encode_video(ffmpeg, vin, W, H, tmpv):
    """Transcode one camera video to WxH H.264 (crf 18, GOP 2) into tmpv."""
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            vin,
            "-vf",
            f"scale={W}:{H}:flags=area",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-g",
            "2",
            "-threads",
            "2",
            "-an",
            tmpv,
        ],
        check=True,
    )


def _decoded_frames(ffmpeg, tmpv, W, H, nvf, vin):
    """Decode the written video, check the frame count; returns ~32 frames scaled to [0, 1] and the count."""
    raw = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", tmpv, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    fr = np.frombuffer(raw, dtype=np.uint8)
    if fr.size % (W * H * 3):
        raise RuntimeError(f"{tmpv}: decoded size not a whole number of {W}x{H} frames")
    n = fr.size // (W * H * 3)
    if n != nvf:
        raise RuntimeError(f"{vin}: transcoded frames {n} ≠ video_frames {nvf}")
    fr = fr.reshape(n, H * W, 3)[:: max(1, n // 32)].astype(np.float64) / 255.0
    return fr, n


def _state_action(job, df, rows, perm, pq):
    """State and action of every frame in output columns (action = target action_shift rows later, optional clip)."""
    K = len(df)
    S = np.stack(df[job["state_key"]].to_numpy())[rows][:, perm].astype(np.float32)
    # action of frame j = the target recorded `shift` rows after its state row (row k's target is the one applied in
    # step k — it produced state k; measured + traj_record s_k = after step k) · past the last row: the last target
    arow = np.minimum(rows + job["action_shift"], K - 1)
    A = np.stack(df[job["action_key"]].to_numpy())[arow][:, perm].astype(np.float32)
    n_clamped = int((rows + job["action_shift"] > K - 1).sum())
    if not (np.isfinite(S).all() and np.isfinite(A).all()):
        raise RuntimeError(f"{pq}: non-finite state/action")
    clip_rec = {}
    if job.get("clip"):
        # D1b — clip each joint's target to its URDF limit (owner json · by NAMES order = the output columns)
        lo, hi = (np.asarray(x, dtype=np.float32) for x in job["clip"])
        A0 = A
        A = np.clip(A0, lo, hi).astype(np.float32)
        ex = np.maximum(np.maximum(lo - A0, A0 - hi), 0.0)
        clip_rec = {
            "action_clip_frames": int((A0 != A).any(axis=1).sum()),
            "action_clip_max_rad": float(ex.max()),
            "runaway_frames": int(((ex > 0) & (np.abs(A0 - S) > 1.0)).any(axis=1).sum()),
        }
    return S, A, n_clamped, clip_rec


def _write_parquet(job, out, i, nvf, vfps, S, A):
    """Write the episode parquet (state, action, time, indices, task/axis, axis one-hot) via a partial file."""
    chunk = i // job["chunks_size"]
    rel_pq = f"data/chunk-{chunk:03d}/episode_{i:06d}.parquet"
    os.makedirs(os.path.join(out, os.path.dirname(rel_pq)), exist_ok=True)
    g0 = job["global_index"]
    tab = pd.DataFrame(
        {
            "observation.state": list(S),
            "action": list(A),
            "timestamp": (np.arange(nvf) / vfps).astype(np.float32),
            "frame_index": np.arange(nvf, dtype=np.int64),
            "episode_index": np.full(nvf, i, dtype=np.int64),
            "index": np.arange(g0, g0 + nvf, dtype=np.int64),
            "task_index": np.full(nvf, job["task_index"], dtype=np.int64),
            "axis_index": np.full(nvf, job["axis_index"], dtype=np.int64),
            # the task axis as a float one-hot — DP's conditioning input (LeRobot reads it as environment_state)
            # axis_index -1 (--allow-axis-none · a row with no axis) = an all-zero row (np.eye(...)[[-1]] would be the
            # last axis)
            "observation.environment_state": list(
                np.repeat(np.eye(len(AXES), dtype=np.float32)[[job["axis_index"]]], nvf, 0)
                if job["axis_index"] >= 0
                else np.zeros((nvf, len(AXES)), dtype=np.float32)
            ),
        }
    )
    tmp = os.path.join(out, rel_pq + ".partial")
    tab.to_parquet(tmp, index=False)
    os.replace(tmp, os.path.join(out, rel_pq))
    return chunk, tab


def _transcode_cameras(out, i, src, si, info, chunk_src, chunk, cams, ffmpeg, W, H, nvf):
    """Transcode every camera video and collect its image stats from the decoded output."""
    img_stats = {}
    for cam in cams:
        skey = f"observation.images.{cam}"
        dkey = f"observation.images.camera_{cam}"
        vin = os.path.join(src, info["video_path"].format(episode_chunk=chunk_src, video_key=skey, episode_index=si))
        rel_v = f"videos/chunk-{chunk:03d}/{dkey}/episode_{i:06d}.mp4"
        vout = os.path.join(out, rel_v)
        os.makedirs(os.path.dirname(vout), exist_ok=True)
        tmpv = vout + ".partial.mp4"
        _encode_video(ffmpeg, vin, W, H, tmpv)
        # decode what was written: frame count must equal the episode's video frames; image stats from it
        fr, n = _decoded_frames(ffmpeg, tmpv, W, H, nvf, vin)
        img_stats[dkey] = {
            "min": [[[float(x)]] for x in fr.min((0, 1))],
            "max": [[[float(x)]] for x in fr.max((0, 1))],
            "mean": [[[float(x)]] for x in fr.mean((0, 1))],
            "std": [[[float(x)]] for x in fr.std((0, 1))],
            "count": [int(n)],
        }
        os.replace(tmpv, vout)
    return img_stats


def _write_record(job, out, i, nvf, src, si, n_clamped, clip_rec, S, A, tab, img_stats):
    """The per-episode record (lengths, source, clamps, stats) written to _work/ep_<i>.json; returns it."""

    def st(x):
        return {
            "min": x.min(0).tolist(),
            "max": x.max(0).tolist(),
            "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(),
            "count": [int(len(x))],
        }

    rec = dict(
        episode_index=i,
        length=nvf,
        src=src,
        src_episode_index=si,
        key=job["key"],
        action_clamped=n_clamped,
        **clip_rec,
        force_names=None,
        stats={
            "observation.state": st(S),
            "action": st(A),
            "observation.environment_state": st(np.stack(tab["observation.environment_state"].to_numpy())),
            **img_stats,
        },
    )
    tmpj = os.path.join(out, "_work", f"ep_{i:06d}.json.partial")
    with open(tmpj, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmpj, os.path.join(out, "_work", f"ep_{i:06d}.json"))
    return rec


def convert_one(job: dict) -> dict:
    """One source episode → its parquet + videos under out. Returns the per-episode record (stats included)."""
    src, e, i, out = job["src"], job["e"], job["episode_index"], job["out"]
    W, H, ffmpeg, cams = job["width"], job["height"], job["ffmpeg"], job["cameras"]
    info, perm = job["info"], np.asarray(job["perm"])
    si = int(e["episode_index"])
    chunk_src = si // int(info["chunks_size"])
    pq = os.path.join(src, info["data_path"].format(episode_chunk=chunk_src, episode_index=si))
    cols = ["timestamp", "video_frame_index", job["state_key"], job["action_key"]]
    df = pd.read_parquet(pq, columns=cols)
    if len(df) != int(e["length"]):
        raise RuntimeError(f"{pq}: rows {len(df)} ≠ episodes.jsonl length {e['length']}")
    vfps = float(info["collect"]["video_fps"])
    nvf = int(e["video_frames"])
    vfi = df["video_frame_index"].to_numpy().astype(np.int64)
    ts = df["timestamp"].to_numpy().astype(np.float64)
    if not (np.array_equal(np.unique(vfi), np.arange(nvf))):
        raise RuntimeError(f"{pq}: video_frame_index does not cover 0..{nvf - 1} exactly")
    rows = _frame_rows(nvf, vfi, ts, vfps)
    S, A, n_clamped, clip_rec = _state_action(job, df, rows, perm, pq)
    chunk, tab = _write_parquet(job, out, i, nvf, vfps, S, A)
    img_stats = _transcode_cameras(out, i, src, si, info, chunk_src, chunk, cams, ffmpeg, W, H, nvf)
    return _write_record(job, out, i, nvf, src, si, n_clamped, clip_rec, S, A, tab, img_stats)
