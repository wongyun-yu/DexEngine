"""LeRobot v2.1 metadata for the recorded datasets (CPU: pyarrow, numpy, ffmpeg).

store.store writes one episode at a time; store.merge concatenates pieces into the final dataset.

Feature rules:
  Features = one schema per hand x dataset (owner: store.form). A feature that an episode lacks is written as NaN and
    listed in that episode's ABSENT column (absent_features). unify() takes the union of the episodes' features and
    checks their types; the writer uses it to check that every episode matches the schema.
    A wholly absent feature (store.form.stats_skip) has no stats in that episode: LeRobot's aggregate_stats only
    averages episodes that have the feature and its count has shape (1), so it cannot weight per dimension. The
    schema therefore never has features that are NaN in only some dimensions.
  names: only what the episode builder passed from the source (joint features = trajectory entity joint_names, joint
    targets = recording channel names); null when the source has none (LeRobot convention).
  Video features: dtype video, shape [h, w, 3], names [height, width, channels] and video.* info. Resolution, codec
    and fps are measured on the first episode's video; differing resolutions across episodes are an error.
  meta/episodes_stats.jsonl uses the LeRobot v2.1 compute_episode_stats layout: numeric features get min/max/mean/std
    over frames and count (length-1 arrays for scalars); video features get per-channel [0, 1] values of shape
    (3, 1, 1) over the sampled frames (sample_indices, auto_downsample as in LeRobot), count = number of samples.
    Depth and force are outside LeRobot and have no stats.
Difference from LeRobot: video is 30 fps while low-dimensional data is 50 Hz. LeRobotDataset needs
tolerance_s >= 1/60 s to read the videos (nearest frame, same rule as video_frame_index); the default 1e-4 s fails on
timestamps off the 30 fps grid.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError, input_missing
from dexengine.record.store.parquet import read_table

MARK = "[COLLECT_LEROBOT]"
STATS = "meta/episodes_stats.jsonl"
ABSENT = "absent_features"
VIDEO_NAMES = ["height", "width", "channels"]
TOOLS_MARK = "[COLLECT_TOOLS]"


def die(msg: str) -> None:
    raise SystemExit(f"{MARK} ✗ {msg}")


def _tool(name: str) -> str | None:
    return shutil.which(name) or shutil.which(name, path=str(Path.home() / ".local" / "bin"))


def ffprobe_bin() -> str:
    """ffprobe from PATH, else ~/.local/bin. Missing is an error (videos are not stored without a frame-count check)."""
    b = _tool("ffprobe")
    if not b:
        raise InputMissingError(
            f"{TOOLS_MARK} ffprobe not found on PATH or in ~/.local/bin (cannot check video frame counts)"
        )
    return b


def ffmpeg_bin() -> str:
    """ffmpeg, found like ffprobe_bin. Missing is an error."""
    b = _tool("ffmpeg")
    if not b:
        raise InputMissingError(f"{TOOLS_MARK} ffmpeg not found on PATH or in ~/.local/bin")
    return b


def write_parquet(path: Path, cols: dict, extra: dict, names: dict | None = None) -> list[dict]:
    """Columns -> parquet; returns the feature specs [{name, dtype, shape, names}] (names from the source or None)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrays, keys, feats = [], [], []
    for k, v in {**cols, **extra}.items():
        v = np.asarray(v)
        arrays.append(
            pa.array(v)
            if v.ndim == 1
            else pa.array(list(v.astype(np.float32)), type=pa.list_(pa.float32(), v.shape[1]))
        )
        keys.append(k)
        nm = (names or {}).get(k)
        if nm is not None and len(nm) != (v.shape[1] if v.ndim > 1 else 1):
            die(f"{k}: {len(nm)} names for width {v.shape[1:]} (the names do not belong to this column)")
        feats.append(
            {
                "name": k,
                "dtype": str(v.dtype) if v.ndim == 1 else "float32",
                "shape": list(v.shape[1:]) or [1],
                "names": None if nm is None else list(nm),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, names=keys), path)
    return feats


def union_order(lists: list[list[str]]) -> list[str]:
    """Union of name lists; a new name goes right after its predecessor in its own list (keeps each list's order)."""
    out: list[str] = []
    for names in lists:
        for k, n in enumerate(names):
            if n not in out:  # the predecessor is already in out (we walk each list in order)
                out.insert(out.index(names[k - 1]) + 1 if k else 0, n)
    return out


def unify(feats_by_ep: list[list[dict]]) -> tuple[dict, list[list[str]]]:
    """Per-episode feature specs -> (features {name: {dtype, shape, names}}, absent names per episode).

    A feature whose spec differs between episodes is an error, and so is an absent integer feature (no NaN fill)."""
    seen: dict = {}
    for i, fs in enumerate(feats_by_ep):
        for f in fs:
            spec = {k: f[k] for k in ("dtype", "shape", "names")}
            if seen.setdefault(f["name"], spec) != spec:
                die(f"feature {f['name']} differs between episodes (ep{i}: {spec} != {seen[f['name']]})")
    order = union_order([[f["name"] for f in fs] for fs in feats_by_ep])
    absent = [[n for n in order if n not in {f["name"] for f in fs}] for fs in feats_by_ep]
    bad = sorted({n for a in absent for n in a if seen[n]["dtype"] not in ("float32", "float64")})
    if bad:
        die(input_missing(f"integer features {bad} are missing in some episodes and cannot be NaN-filled"))
    return {n: seen[n] for n in order}, absent


# Video measurement. One cache owns every per-video value: _PRE[content sha256] = {"probe": probe, (h, w): stats}.
# The key is the file content (the store copies the rendered videos and the preview probes the copies, so paths
# and inodes differ); the same content gives the same values, so stored output is unchanged. When a clip's render
# finishes, a small thread pool measures its episode videos (render.clip calls prewarm) so the store step overlaps
# the next render; video_features waits for pending jobs (drain), and a failed pre-measurement is simply measured
# again by the store (and fails there).
_PRE: dict[str, dict] = {}
_PRE_JOBS: list = []
_PRE_POOL: list = []
#: pre-measurement threads (one ffmpeg each): capacity.NEEDS key — a quarter of the usable CPUs (capacity.cpu_limit:
#: affinity and cgroup quota, never the host core count; those CPUs also render), at most its cap; the pool starts a
#: thread per queued video only, so it never runs more threads than videos
PREWARM_STEP = "dexengine.record.store.lerobot:prewarm"


def _ckey(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _ffprobe_stream(p: Path, count: bool) -> dict:
    """ffprobe of stream v:0 -> {codec, width, height, frames, fps}. count=True decodes every frame (frames =
    nb_read_frames, the decoder's output count); count=False reads the container header (frames = nb_frames, the
    stated count; 0 when the container states none)."""
    n = "nb_read_frames" if count else "nb_frames"
    r = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-select_streams", "v:0"]
        + (["-count_frames"] if count else [])
        + ["-show_entries", f"stream=codec_name,width,height,{n},r_frame_rate", "-of", "json", str(p)],
        capture_output=True,
        text=True,
    )
    s = json.loads(r.stdout or "{}").get("streams", [{}])[0]
    frames = s.get(n, 0) or 0
    return {
        "codec": s.get("codec_name"),
        "width": s.get("width"),
        "height": s.get("height"),
        "frames": int(frames) if count or str(frames).isdigit() else 0,
        "fps": s.get("r_frame_rate"),
    }


def _measure(p: Path) -> dict:
    """One video's cache entry from a single decode: the probe and the sampled-frame stats at the video's size.
    The frames are sampled for the container's stated count n; the decode's own count is the probe's frame count
    when ffmpeg reports every decoded frame written once (no duplicate, no drop, so it equals the decoder's output
    count, which ffprobe -count_frames measures) and it equals n (so the sample is exactly the one taken for that
    count). Otherwise ffprobe -count_frames measures the count and video_stats decodes again for the stats."""
    head = _ffprobe_stream(p, count=False)
    h, w, n = head["height"], head["width"], head["frames"]
    if h and w and n:
        imgs, got, exact = _decode(ffmpeg_bin(), p, h, w, sample_indices(n))
        if exact and got == n:
            return {"probe": head, (h, w): _sampled_stats(np.stack(imgs))}
    return {"probe": _ffprobe_stream(p, count=True)}


def probe_video(p: Path) -> dict:
    """Probe of one video (codec, width, height, decoded frame count, fps), measured once per content. A content
    not yet measured costs one ffprobe -count_frames (callers that want only the count)."""
    c = _PRE.setdefault(_ckey(Path(p)), {})
    if "probe" not in c:
        c["probe"] = _ffprobe_stream(Path(p), count=True)
    return dict(c["probe"])


def measure_video(p: Path) -> dict:
    """probe_video for callers that also want the stats: a content not yet measured costs one decode (_measure),
    which also keeps the stats at the video's own size."""
    c = _PRE.setdefault(_ckey(Path(p)), {})
    if "probe" not in c:
        c.update(_measure(Path(p)))
    return dict(c["probe"])


def video_stats(ffmpeg: str, p: Path, h: int, w: int, probe) -> dict:
    """Per-channel stats (/255, count) of one video's sampled frames, computed once per content and (h, w)."""
    measure_video(p)
    c = _PRE.setdefault(_ckey(Path(p)), {})
    if (h, w) not in c:
        c[(h, w)] = _sampled_stats(_frames(ffmpeg, p, h, w, sample_indices(probe(p)["frames"])))
    return c[(h, w)]


def prewarm(eps: list[dict]) -> int:
    """Queue background measurement of the episodes' videos (render_dir) on the pre-measurement pool."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from dexengine.record.store import store as ST

    if not _PRE_POOL:
        from dexengine.core import capacity

        usable = capacity.cpu_limit()[0]
        p = capacity.plan(PREWARM_STEP, capacity.NEEDS[PREWARM_STEP].cap, share=max(1, usable // 4))
        print(p.line(), file=sys.stderr, flush=True)
        print(
            f"{MARK} pre-measurement pool {p.workers} threads (usable CPUs {usable} · at most {p.need.cap})", flush=True
        )
        _PRE_POOL.append(ThreadPoolExecutor(p.workers, thread_name_prefix="prewarm"))
    ff = ffmpeg_bin()

    def one(v: Path):
        r = measure_video(v)
        video_stats(ff, v, r["height"], r["width"], probe_video)

    vids = [Path(e["render_dir"]) / f"{k}.mp4" for e in eps for k in ST.VIDEO_KEYS]
    _PRE_JOBS.extend(_PRE_POOL[0].submit(one, v) for v in vids)
    return len(vids)


def drain() -> None:
    """Wait for pending pre-measurements; failures leave one log line (the store measures again)."""
    bad = [f.exception() for f in _PRE_JOBS if f.exception() is not None]
    _PRE_JOBS.clear()
    if bad:
        print(
            f"{MARK} video pre-measurement failed {len(bad)} (measured again at store) · first {bad[0]!r}"[:300],
            flush=True,
        )


def video_features(probe, paths: dict[str, list[Path]], fps: float) -> dict:
    """Video feature specs. probe = store._ffprobe, paths = {key: mp4 per episode}; differing resolutions die."""
    drain()
    out = {}
    for k, ps in paths.items():
        rs = [probe(p) for p in ps]
        if len({(r["width"], r["height"]) for r in rs}) != 1:
            die(f"{k}: resolution differs between episodes {sorted({(r['width'], r['height']) for r in rs})[:3]}")
        r = rs[0]
        out[k] = {
            "dtype": "video",
            "shape": [r["height"], r["width"], 3],
            "names": list(VIDEO_NAMES),
            "info": {
                "video.fps": fps,
                "video.height": r["height"],
                "video.width": r["width"],
                "video.channels": 3,
                "video.codec": r["codec"],
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return out


def sample_indices(n: int) -> list[int]:
    """LeRobot compute_stats.sample_indices: max(min(n, 100), min(floor(n^0.75), 10000)) evenly spaced frames."""
    k = max(min(n, 100), min(int(n**0.75), 10_000))
    return np.round(np.linspace(0, n - 1, k)).astype(int).tolist()


def _decode(ffmpeg: str, p: Path, h: int, w: int, want: list[int]) -> tuple[list, int, bool]:
    """mp4 -> (the wanted frames as uint8 [C, H, W] (LeRobot auto_downsample: every floor(side/150) px if side >=
    300), number of frames written, whether ffmpeg's progress report ends with 0 duplicated and 0 dropped frames)."""
    import tempfile

    f = int(w / 150) if w > h else int(h / 150)
    with tempfile.NamedTemporaryFile("r", suffix=".progress") as prog:
        proc = subprocess.Popen(
            [ffmpeg, "-v", "error", "-progress", prog.name, "-i", str(p), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        out, keep, i = [], set(want), 0
        while (buf := proc.stdout.read(h * w * 3)) and len(buf) == h * w * 3:
            if i in keep:
                img = np.frombuffer(buf, np.uint8).reshape(h, w, 3).transpose(2, 0, 1)
                out.append(img if max(h, w) < 300 else img[:, ::f, ::f])
            i += 1
        proc.wait()
        rep = dict(x.split("=", 1) for x in prog.read().splitlines() if "=" in x)  # the last report wins
    exact = proc.returncode == 0 and rep.get("progress") == "end"
    return out, i, exact and rep.get("dup_frames") == "0" and rep.get("drop_frames") == "0"


def _frames(ffmpeg: str, p: Path, h: int, w: int, want: list[int]):
    """mp4 -> the wanted frames stacked as uint8 [N, C, H', W']; a missing wanted frame dies."""
    out, i, _ = _decode(ffmpeg, p, h, w, want)
    if len(out) != len(want):
        die(f"{p}: {len(out)} sampled frames != {len(want)} (video has {i} frames)")
    return np.stack(out)


def _sampled_stats(imgs: np.ndarray) -> dict:
    """Per-channel stats of sampled frames [N, C, H, W] in [0, 1], with the sample count."""
    return {s: v[0] / 255.0 for s, v in _stat(imgs, (0, 2, 3), True).items()} | {"count": [len(imgs)]}


def _stat(a: np.ndarray, axis, keepdims: bool) -> dict:
    return {
        "min": np.min(a, axis=axis, keepdims=keepdims),
        "max": np.max(a, axis=axis, keepdims=keepdims),
        "mean": np.mean(a, axis=axis, keepdims=keepdims),
        "std": np.std(a, axis=axis, keepdims=keepdims),
    }


def numeric_stats(parquet: Path, features: dict, names: list[str]) -> dict:
    """Numeric feature stats over frames (min, max, mean, std, count); shared by episode_stats and retext."""
    import pyarrow.types as PAT

    t, out = read_table(parquet, columns=names), {}
    for n in t.column_names:
        # Stack list-typed columns. Do not decide by the schema shape: write_parquet writes width-1 [L, 1] columns
        # as lists but records shape [1] (same as a scalar), and an object array would fail in np.std.
        col = t.column(n)
        lst = PAT.is_fixed_size_list(col.type) or PAT.is_list(col.type) or PAT.is_large_list(col.type)
        a = np.stack(col.to_numpy(zero_copy_only=False)) if lst else col.to_numpy()
        if a.dtype == object:
            die(input_missing(f"{parquet}: column {n} is not numeric (type {col.type}), no stats"))
        out[n] = dict(_stat(a, 0, a.ndim == 1), count=[len(a)])
    return {n: {s: np.asarray(v).tolist() for s, v in d.items()} for n, d in out.items()}


def episode_stats(parquet: Path, features: dict, skip: list[str], videos: dict, probe, ffmpeg: str) -> dict:
    """Stats of one episode in the LeRobot get_feature_stats layout (numeric = numeric_stats, video = sampled
    frames /255). skip = features left out (store.form.stats_skip)."""
    out = numeric_stats(parquet, features, [n for n, f in features.items() if f["dtype"] != "video" and n not in skip])

    def one(k):  # one thread per video (ffmpeg runs as a process); pre-measured values are reused
        h, w = features[k]["shape"][:2]
        return k, video_stats(ffmpeg, videos[k], h, w, probe)

    from concurrent.futures import ThreadPoolExecutor

    from dexengine.core import capacity

    step = "dexengine.record.store.lerobot:stats"  # capacity.NEEDS key: one thread per video (ffmpeg process)
    with capacity.sized(step, len(videos)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        vs = dict(ex.map(one, list(videos)))
    return out | {n: {s: np.asarray(v).tolist() for s, v in d.items()} for n, d in vs.items()}


def write_stats(out: Path, lines: list[dict]) -> None:
    """Write meta/episodes_stats.jsonl ({episode_index, stats} per line, LeRobot v2.1) via a temp file + rename."""
    tmp = out / (STATS + ".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    tmp.replace(out / STATS)


def read_stats(out: Path) -> list[dict]:
    """episodes_stats lines (episode_index must equal the line order); [] when the file is absent."""
    p = out / STATS
    lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
    if [x["episode_index"] for x in lines] != list(range(len(lines))):
        die(f"{p}: episode_index != line order")
    return lines


def totals(n: int, frames: int, n_tasks: int, chunks_size: int, n_video_keys: int) -> dict:
    """info total_* and splits fields (LeRobot v2.1), shared by the per-piece store and the final merge."""
    return {
        "total_episodes": n,
        "total_frames": frames,
        "total_tasks": n_tasks,
        "total_chunks": -(-n // chunks_size),
        "total_videos": n * n_video_keys,
        "splits": {"train": f"0:{n}"},
    }


def parquet_columns(info: dict) -> list[str]:
    """Non-video features of info.features, in order."""
    return [n for n, f in info["features"].items() if f.get("dtype") != "video"]
