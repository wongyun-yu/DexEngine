"""Decode every video frame of a LeRobot v2.1 dataset once into uint8 arrays, so DP training reads frames instead of
decoding mp4 on the CPU (a 1-GPU DP run waited 0.20 s/step on decode vs 0.15 s/step of GPU compute).

Same decoder as LeRobot 0.3.3 `decode_video_frames_torchvision(backend="pyav")` (torchvision VideoReader · pyav), so
the cached uint8 are the frames LeRobot would decode. The lookup (`query`) keeps LeRobot's rule: for each query
timestamp, the loaded frame with the closest pts, which must be within tolerance_s. `check` proves it: for random
dataset items, the image tensors from the cache are torch.equal to LeRobot's own __getitem__ output.

  build : python -m experiments.policies.dp.frame_cache build --root <dataset> --out <cache dir> [--procs 24]
  check : python -m experiments.policies.dp.frame_cache check --root <dataset> --cache <cache dir> [--n 300]
Cache dir: <key>.u8.npy (N, C, H, W) uint8 · <key>.pts.npy (N,) float64 (pts inside its episode) · ep_from.npy
(episodes+1,) · DONE.json (dataset info sha · frames · keys). Marks: [E5_FCACHE] DONE|FAIL · [E5_FCACHE_CHECK]
OK|FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MARK = "[E5_FCACHE]"


def _meta(root: Path):
    info = json.loads((root / "meta/info.json").read_text())
    eps = [json.loads(x) for x in (root / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
    eps.sort(key=lambda e: e["episode_index"])
    if [e["episode_index"] for e in eps] != list(range(len(eps))):
        raise SystemExit(f"{MARK} FAIL episode_index not 0..{len(eps) - 1}")
    keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    lengths = np.array([e["length"] for e in eps], dtype=np.int64)
    ep_from = np.concatenate([[0], np.cumsum(lengths)])
    if int(ep_from[-1]) != int(info["total_frames"]):
        raise SystemExit(f"{MARK} FAIL sum(length) {ep_from[-1]} != total_frames {info['total_frames']}")
    return info, eps, keys, ep_from


def _video_path(root: Path, info: dict, ep: int, key: str) -> Path:
    return root / info["video_path"].format(episode_chunk=ep // info["chunks_size"], video_key=key, episode_index=ep)


def _decode_all(path: Path):
    """All frames of one mp4 with LeRobot's pyav reader (torchvision VideoReader) — uint8 (n, C, H, W) and pts."""
    import torchvision

    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(path), "video")
    reader.seek(0.0, keyframes_only=True)
    frames, pts = [], []
    for fr in reader:
        frames.append(fr["data"].numpy())
        pts.append(fr["pts"])
    reader.container.close()
    return np.stack(frames), np.array(pts, dtype=np.float64)


_G = {}


def _init(out: str, keys: list[str]):
    _G["arr"] = {k: np.load(f"{out}/{k}.u8.npy", mmap_mode="r+") for k in keys}
    _G["pts"] = {k: np.load(f"{out}/{k}.pts.npy", mmap_mode="r+") for k in keys}


def _work(job):
    root, info, ep, a, n, keys = job
    for k in keys:
        fr, pts = _decode_all(_video_path(Path(root), info, ep, k))
        if len(fr) != n:
            return f"episode {ep} {k}: decoded {len(fr)} frames != length {n}"
        _G["arr"][k][a : a + n] = fr
        _G["pts"][k][a : a + n] = pts
    return None


def build(root: Path, out: Path, procs: int) -> None:
    t0 = time.time()
    info, eps, keys, ep_from = _meta(root)
    n = int(info["total_frames"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists():
        raise SystemExit(f"{MARK} FAIL {out}/DONE.json exists — a cache is never rebuilt in place")
    for k in keys:
        c, h, w = info["features"][k]["shape"]
        np.lib.format.open_memmap(out / f"{k}.u8.npy", mode="w+", dtype=np.uint8, shape=(n, c, h, w)).flush()
        np.lib.format.open_memmap(out / f"{k}.pts.npy", mode="w+", dtype=np.float64, shape=(n,)).flush()
    np.save(out / "ep_from.npy", ep_from)
    jobs = [(str(root), info, i, int(ep_from[i]), int(ep_from[i + 1] - ep_from[i]), keys) for i in range(len(eps))]
    with Pool(procs, initializer=_init, initargs=(str(out), keys)) as p:
        errs = [e for e in p.imap_unordered(_work, jobs, chunksize=4) if e]
    if errs:
        raise SystemExit(f"{MARK} FAIL {len(errs)} episodes: {errs[:3]}")
    sha = hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest()
    (out / "DONE.json").write_text(
        json.dumps(
            {
                "root": str(root.resolve()),
                "info_sha256": sha,
                "total_frames": n,
                "episodes": len(eps),
                "keys": keys,
                "decoder": "torchvision VideoReader pyav (= LeRobot 0.3.3 pyav)",
            }
        )
    )
    gb = sum(os.path.getsize(out / f"{k}.u8.npy") for k in keys) / 1e9
    print(
        f"{MARK} DONE {root.name} episodes {len(eps)} frames {n} keys {len(keys)} · {gb:.1f} GB · "
        f"{time.time() - t0:.0f} s → {out}",
        flush=True,
    )


class FrameCache:
    """Read side, opened lazily in each DataLoader worker (memmap)."""

    def __init__(self, cache: Path, root: Path):
        self.cache, self.root = Path(cache), Path(root)
        done = json.loads((self.cache / "DONE.json").read_text())
        sha = hashlib.sha256((self.root / "meta/info.json").read_bytes()).hexdigest()
        if done["info_sha256"] != sha:
            raise SystemExit(
                f"{MARK} FAIL cache {self.cache} was built from another info.json "
                f"({done['info_sha256'][:12]} != {sha[:12]})"
            )
        self.keys, self.n = done["keys"], int(done["total_frames"])
        self.ep_from = np.load(self.cache / "ep_from.npy")
        self._arr = None

    def _open(self):
        if self._arr is None:
            self._arr = {k: np.load(self.cache / f"{k}.u8.npy", mmap_mode="r") for k in self.keys}
            self._pts = {k: np.load(self.cache / f"{k}.pts.npy", mmap_mode="r") for k in self.keys}

    def query(self, key: str, ep_idx: int, timestamps: list[float], tolerance_s: float):
        import torch

        self._open()
        a, b = int(self.ep_from[ep_idx]), int(self.ep_from[ep_idx + 1])
        pts = np.asarray(self._pts[key][a:b])
        q = np.asarray(timestamps, dtype=np.float64)
        d = np.abs(q[:, None] - pts[None, :])
        j = d.argmin(1)
        if not (d[np.arange(len(q)), j] < tolerance_s).all():
            raise AssertionError(f"{MARK} query ts {q} outside tolerance {tolerance_s} of episode {ep_idx} {key}")
        return torch.from_numpy(np.ascontiguousarray(self._arr[key][a + j])).type(torch.float32) / 255


def install(cache: Path, root: Path, dataset_cls=None) -> FrameCache:
    """Replace LeRobotDataset._query_videos with the cache lookup (same return as LeRobot: frames.squeeze(0))."""
    from lerobot.datasets import lerobot_dataset as LD

    cls = dataset_cls or LD.LeRobotDataset
    fc = FrameCache(cache, root)

    def _query_videos(self, query_timestamps, ep_idx):
        return {k: fc.query(k, ep_idx, ts, self.tolerance_s).squeeze(0) for k, ts in query_timestamps.items()}

    cls._query_videos = _query_videos
    return fc


def install_columns(dataset_cls=None) -> None:
    """Serve the parquet columns (state · action · timestamp · indices) from numpy arrays built once per dataset
    instead of hf_dataset[idx] + hf_dataset.select(...) per item (5 Arrow selects per item · the CPU cost left after
    the frame cache).
    Same tensors as LeRobot 0.3.3 (dtype · shape · values) — `check --columns` proves it item by item."""
    import numpy as np
    import torch
    from lerobot.datasets import lerobot_dataset as LD

    cls = dataset_cls or LD.LeRobotDataset
    orig_init = cls.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        cols = self.hf_dataset.with_format("numpy")[:]
        self._e5_cols = {key: np.ascontiguousarray(v) for key, v in cols.items() if key not in self.meta.video_keys}
        bad = [key for key, v in self._e5_cols.items() if v.dtype.kind not in "fiub"]
        if bad:
            raise SystemExit(f"{MARK} FAIL columns not numeric: {bad}")

    def _row(self, idx):
        out = {}
        for key, v in self._e5_cols.items():
            x = v[idx]
            out[key] = (
                torch.tensor(x.item())
                if np.ndim(x) == 0
                else torch.from_numpy(np.array(x, dtype=np.float32 if v.dtype.kind == "f" else np.int64))
            )
        return out

    def _query_hf_dataset(self, query_indices):
        return {
            key: torch.from_numpy(
                np.array(self._e5_cols[key][q], dtype=np.float32 if self._e5_cols[key].dtype.kind == "f" else np.int64)
            )
            for key, q in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _get_query_timestamps(self, current_ts, query_indices=None):
        ts = self._e5_cols["timestamp"]
        return {
            key: (
                ts[query_indices[key]].astype(np.float32).tolist()
                if query_indices is not None and key in query_indices
                else [current_ts]
            )
            for key in self.meta.video_keys
        }

    def __getitem__(self, idx):
        item = _row(self, idx)
        ep_idx = item["episode_index"].item()
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        if len(self.meta.video_keys) > 0:
            query_timestamps = self._get_query_timestamps(item["timestamp"].item(), query_indices)
            item = {**self._query_videos(query_timestamps, ep_idx), **item}
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                item[cam] = self.image_transforms(item[cam])
        item["task"] = self.meta.tasks[item["task_index"].item()]
        return item

    cls.__init__, cls._query_hf_dataset, cls._get_query_timestamps, cls.__getitem__ = (
        __init__,
        _query_hf_dataset,
        _get_query_timestamps,
        __getitem__,
    )


def check(root: Path, cache: Path, n: int, columns: bool = False) -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("e5/check", root=root)
    fps = meta.fps
    obs = [-1 / fps, 0.0]  # DP n_obs_steps 2
    dts = dict.fromkeys(meta.video_keys, obs)
    dts["observation.state"] = obs
    dts["action"] = [i / fps for i in range(-1, 15)]  # DP horizon 16
    ref = LeRobotDataset("e5/check", root=root, delta_timestamps=dts, video_backend="pyav")
    rng = np.random.default_rng(42)
    idx = sorted(set(rng.integers(0, ref.num_frames, size=n).tolist()) | {0, ref.num_frames - 1})
    t0 = time.time()
    want = {i: ref[i] for i in idx}
    t_ref = (time.time() - t0) / len(idx)
    fc = FrameCache(cache, root)
    if fc.n != ref.num_frames:
        raise SystemExit(f"[E5_FCACHE_CHECK] FAIL cache frames {fc.n} != dataset {ref.num_frames}")
    install(cache, root)
    if columns:
        install_columns()
        ref = LeRobotDataset("e5/check", root=root, delta_timestamps=dts, video_backend="pyav")
    t0 = time.time()
    got = {i: ref[i] for i in idx}
    t_new = (time.time() - t0) / len(idx)
    keys = set(want[idx[0]]) if columns else set(meta.video_keys)
    bad = []
    for i in idx:
        a, b = want[i], got[i]
        if columns and set(a) != set(b):
            bad.append((i, "keys", sorted(set(a) ^ set(b))))
            continue
        for k in keys:
            x, y = a[k], b[k]
            same = (torch.equal(x, y) and x.dtype == y.dtype) if isinstance(x, torch.Tensor) else x == y
            if not same:
                bad.append((i, k))
    if bad:
        raise SystemExit(f"[E5_FCACHE_CHECK] FAIL {len(bad)} not equal (columns={columns}): {bad[:4]}")
    print(
        f"[E5_FCACHE_CHECK] OK {root.name} items {len(idx)} × keys {len(keys)} torch.equal + dtype to "
        "LeRobot (frame cache"
        f"{' + columns' if columns else ''}) · per item {t_ref * 1000:.1f} → {t_new * 1000:.1f} ms · frames {fc.n}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--columns", action="store_true", help="check also install_columns (all keys of every item)")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.root, a.out, a.procs)
    else:
        check(a.root, a.cache, a.n, a.columns)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — a loud FAIL line for the watcher
        print(f"{MARK} FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
        sys.exit(1)
