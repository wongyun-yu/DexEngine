"""Single owner of the record render output write layer: one background thread writes an episode's mp4s (egoL, egoR,
exo; H.264) and depth (format below). Both render entries (render.episode.render_selected,
render.clip_session.render_clip) use the same path.

  EpisodeWriter(out, n, videos, fps, depth_file): one thread + a bounded queue (QUEUE_FRAMES frames; RGB is never
    held for the whole episode). add(cap) = put one frame into the queue (main thread; waits when full). Thread =
    append_data to the imageio writers (args MP4_ARGS) and fill the depth uint16 conversion
    (render.cameras.depth_to_png16) into an [n, H, W] array allocated at the first frame.
  Frame guard: the constructor and add() call looks.look_guard.require (no frame without the canonical look).
  seal(): end mark -> the thread closes the mp4s and compresses the depth (write, round trip checked) in the
    background.
  wait(): join the thread and return its result (depth info, seconds). A failed thread, or one not done within
    STALL_S, gives SystemExit (never passes silently). A frame count != n fails too.
Overlap (depth 1; owner render.episode.render_episode): after its own frame loop an episode finishes the previous one
  (wait -> render_meta, DONE line) and seals its own: at most one episode's write is pending. The last one is finished
  before the session end (DONE line, PK.finish_render). Marker [COLLECT_RENDER] background write … (per episode,
  line()).
Bytes: same frames, same args (MP4_ARGS), same order -> the same mp4s and depth file as a synchronous inline write. The
  x264 thread count is fixed by MP4_ARGS (-threads 4). The default (= cores the process sees) gives different bytes
  per machine, and many workers each start three encoders with as many threads as cores, flooding kernel time (15
  workers: kernel 55 %, ffmpeg stime >> utime).
Frame array ownership: rig.rgb/depth are new host copies of the GPU sensor buffers (.detach().cpu()), so the next
  render does not overwrite a queued frame; a session device other than cuda dies (check_device).
"""

from __future__ import annotations

import contextlib
import io
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from dexengine.record.looks import look_guard as LG  # stdlib only (frame guard)

# numpy and imageio are imported inside functions: the render session (Kit) imports this file at module level, and
# numpy before AppLauncher makes the [PRE_APP] guard kill the session.

MARK = "[COLLECT_RENDER]"
#: head of the render progress line `<FRAME_HEAD><i>/<n> (step <s>)` (every 60 steps): the session progress watch
#: (session.launch.PROGRESS) and the brightness gate count (looks.rev.gate_problems, frame 0 = one drawn episode)
#: read it
FRAME_HEAD = f"{MARK} frame "
#: single owner of the mp4 args (changing them changes the bytes). veryfast vs medium: x264 CPU ≈ -50 %; crf 17 equals
#: medium crf 20 in quality (SSIM difference <= 0.002, measured by re-encoding the three cameras of one episode)
MP4_ARGS = {
    "codec": "libx264",
    "quality": None,
    "bitrate": None,
    "pixelformat": "yuv420p",
    "macro_block_size": 8,
    "ffmpeg_params": ["-crf", "17", "-preset", "veryfast", "-threads", "4"],
}
QUEUE_FRAMES = 4  # cap of queued frames (1280x720 x2 + 1920x1920 RGB ≈ 16.6 MB per frame -> <= 66 MB)
# cap of a queue put or finish wait; beyond it the write counts as stalled (measured ≈ 8 s/episode)
STALL_S = 600.0
_END = object()


def check_device(dev) -> None:
    """Premise that the next render does not overwrite a queued frame: true only when the sensor buffer is on the GPU
    and .cpu() makes a new copy."""
    if not str(dev).startswith("cuda"):
        raise SystemExit(
            f"{MARK} the background write needs the session device cuda (sensor buffer .cpu() = a new "
            f"copy): device {dev}"
        )


class EpisodeWriter:
    """mp4 and depth writing of one episode (contract in the module doc)."""

    def __init__(self, out: Path, n: int, videos: tuple, fps: int, depth_file: str):
        LG.require(f"render.writer.EpisodeWriter({Path(out).name})")  # no mp4 without the canonical look
        self.out, self.n, self.videos, self.fps, self.depth_file = Path(out), int(n), tuple(videos), fps, depth_file
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self.err: BaseException | None = None
        self.depth_info: dict | None = None
        self.frames = 0  # frames written by the thread
        self.work = {
            "mp4": 0.0,
            "depth": 0.0,
            "close": 0.0,
            "zstd": 0.0,
        }  # thread-side seconds (main reads after wait only)
        self.wait_frames, self.wait_end, self.qmax, self.added = 0.0, 0.0, 0, 0
        self.nice, self._depth, self._ended = None, None, False
        self.th = threading.Thread(target=self._run, name=f"collect_writer:{self.out.name}", daemon=True)
        self.th.start()

    # ── main thread ──────────────────────────────────────────────────────────
    def add(self, cap: dict) -> None:
        LG.require("render.writer.EpisodeWriter.add")
        self._raise_if_failed()
        item = {v: cap[v] for v in self.videos}
        item["depth_m"] = cap["depth_m"]
        self._put(item)
        self.added += 1

    def seal(self) -> None:
        self._raise_if_failed()
        self._put(_END)

    def wait(self) -> dict:
        c = time.perf_counter()
        self.th.join(STALL_S)
        self.wait_end = time.perf_counter() - c
        if self.th.is_alive():
            raise SystemExit(f"{MARK} write stalled: not done within {STALL_S:.0f} s · {self.out}")
        self._raise_if_failed()
        busy = sum(self.work.values())
        return {
            "depth": self.depth_info,
            "frames": self.frames,
            "work": {k: round(v, 2) for k, v in self.work.items()},
            "busy_s": round(busy, 2),
            "wait_frames_s": round(self.wait_frames, 2),
            "wait_end_s": round(self.wait_end, 2),
            "overlap_s": round(busy - self.wait_frames - self.wait_end, 2),
            "qmax": self.qmax,
            "queue_frames": QUEUE_FRAMES,
            "nice": self.nice,
        }

    def _put(self, item) -> None:
        c = time.perf_counter()
        try:
            self.q.put(item, timeout=STALL_S)
        except queue.Full:
            raise SystemExit(f"{MARK} write stalled: queue did not drain for {STALL_S:.0f} s · {self.out}") from None
        self.wait_frames += time.perf_counter() - c
        self.qmax = max(self.qmax, self.q.qsize())

    def _raise_if_failed(self) -> None:
        if self.err is not None:
            raise SystemExit(
                f"{MARK} write failed (background): {type(self.err).__name__}: {self.err} · {self.out}"
            ) from self.err

    # ── background thread ─────────────────────────────────────────────────────
    def _run(self) -> None:
        wr: dict = {}
        try:
            self.nice = os.getpriority(os.PRIO_PROCESS, 0)
            self._stream(wr)
            c = time.perf_counter()
            self.depth_info = write(self.out / self.depth_file, self._depth)
            self.work["zstd"] = time.perf_counter() - c
            self._depth = None
        except BaseException as e:  # noqa: BLE001 — the main thread re-raises it as SystemExit in wait/add
            self.err = e
            for w in wr.values():  # clean up the failed writes; the original exception is reported
                with contextlib.suppress(Exception):
                    w.close()
            while not self._ended:  # drain to the end mark so the main thread does not block on the queue
                self._ended = self.q.get() is _END

    def _stream(self, wr: dict) -> None:
        """Take frames, stream them into the mp4s and fill the depth -> close the mp4s after the end mark. A frame count
        != n dies."""
        import imageio
        import numpy as np

        from dexengine.record.render import cameras as CC

        clk = time.perf_counter
        self.out.mkdir(parents=True, exist_ok=True)
        wr.update({v: imageio.get_writer(str(self.out / f"{v}.mp4"), fps=self.fps, **MP4_ARGS) for v in self.videos})
        while True:
            item = self.q.get()
            if item is _END:
                self._ended = True
                break
            c = clk()
            for v in self.videos:
                wr[v].append_data(item[v])
            c2 = clk()
            self.work["mp4"] += c2 - c
            d = CC.depth_to_png16(item["depth_m"])
            if self._depth is None:
                self._depth = np.empty((self.n, *d.shape), np.uint16)
            self._depth[self.frames] = d  # beyond n: IndexError · another shape: ValueError -> failure
            self.frames += 1
            self.work["depth"] += clk() - c2
        c = clk()
        for v in self.videos:
            wr.pop(v).close()
        self.work["close"] = clk() - c
        if self.frames != self.n:
            raise RuntimeError(f"frames {self.frames} != plan {self.n}")


def line(r: dict, out: Path, mode: str) -> str:
    """One marker line: proof the write ran in the background · wait = seconds the main thread was blocked · overlap =
    background work - wait."""
    import resource

    w = r["work"]
    return (
        f"{MARK} background write ({mode}) · background work {r['busy_s']:.1f} s (mp4 {w['mp4']:.1f} · depth "
        f"conversion {w['depth']:.1f} · close {w['close']:.1f} · compression {w['zstd']:.1f}) · wait "
        f"{r['wait_frames_s'] + r['wait_end_s']:.1f} s (during frames {r['wait_frames_s']:.1f} · end "
        f"{r['wait_end_s']:.1f}) · overlap {r['overlap_s']:.1f} s · "
        f"queue max {r['qmax']}/{r['queue_frames']} · nice {r['nice']} · max RSS "
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB · {out}"
    )


DEPTH_FILE = "depth_mm.npy.zst"
EXT = ".npy.zst"
LEGACY_EXT = ".npz"
ZSTD_LEVEL = 12  # the level: the CODEC note and the write command read this one value
ZSTD_THREADS_MAX = 8  # shares the machine with the record driver: never all cores (-T0)
ZSTD_NICE = 10  # compression runs in the background (overlaps the next episode): the render loop gets cores first
CODEC = {
    "container": "npy",
    "compression": f"zstd -{ZSTD_LEVEL}",
    "dtype": "uint16",
    "unit": "mm",
    "invalid": 0,
    "lossless": True,
    "owner": "scripts/collect/collect_depth.py",
}  # format tag in info.json (kept)


def zstd_threads(cores: int | None = None) -> int:
    """zstd -T = half the CPUs this process may use (capacity.cpu_limit: cpuset and cpu.max; the whole machine on a
    workstation), clamped to 1..ZSTD_THREADS_MAX. The output bytes are the same for -T1…-T8 (measured: zstd 1.5.5, depth
    212 MB -> same sha); on a small allocation the compression burst (8 threads) does not take the render loop's
    cores."""
    if cores is None:
        from dexengine.core import capacity  # noqa: PLC0415

        cores = capacity.cpu_limit()[0]
    n = int(cores)
    return max(1, min(ZSTD_THREADS_MAX, n // 2))


def _nice_bin() -> str:
    z = shutil.which("nice")
    if z is None:
        raise RuntimeError("[COLLECT_DEPTH] no nice executable: the compression priority cannot be lowered")
    return z


def _zstd_bin() -> str:
    z = shutil.which("zstd")
    if z is None:
        raise RuntimeError("[COLLECT_DEPTH] no zstd executable: depth cannot be written")
    return z


def _npy_bytes(arr) -> bytes:
    import numpy as np

    b = io.BytesIO()
    np.save(b, arr, allow_pickle=False)
    return b.getvalue()


def _decompress(p: Path) -> bytes:
    try:
        import zstandard
    except ImportError:
        return subprocess.run([_zstd_bin(), "-q", "-d", "-c", str(p)], capture_output=True, check=True).stdout
    data = Path(p).read_bytes()
    if zstandard.frame_content_size(data) > 0:  # the write records the raw size with --stream-size: one-shot decode
        return zstandard.ZstdDecompressor().decompress(data)
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as r:
        return r.read()


def write(path: Path, depth_mm, level: int = ZSTD_LEVEL) -> dict:
    """uint16 [N, H, W] -> path (*.npy.zst). Returns {bytes, raw_bytes, frames} after the round trip check."""
    import numpy as np

    path = Path(path)
    if not str(path).endswith(EXT):
        raise ValueError(f"[COLLECT_DEPTH] a depth file must end with {EXT}: {path}")
    a = np.ascontiguousarray(depth_mm)
    if a.dtype != np.uint16 or a.ndim != 3:
        raise ValueError(f"[COLLECT_DEPTH] depth must be uint16 [N, H, W]: {a.dtype} {a.shape}")
    raw = _npy_bytes(a)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    subprocess.run(
        [
            _nice_bin(),
            "-n",
            str(ZSTD_NICE),
            _zstd_bin(),
            "-q",
            "-f",
            f"-{level}",
            f"-T{zstd_threads()}",
            f"--stream-size={len(raw)}",
            "-o",
            str(tmp),
        ],
        input=raw,
        check=True,
    )
    if _decompress(tmp) != raw:
        tmp.unlink()
        raise RuntimeError(f"[COLLECT_DEPTH] round trip mismatch: {path}")
    os.replace(tmp, path)
    return {"bytes": path.stat().st_size, "raw_bytes": len(raw), "frames": int(a.shape[0])}


def read(path: Path):
    """Depth file -> uint16 [N, H, W] (new *.npy.zst, old *.npz)."""
    import numpy as np

    path = Path(path)
    if str(path).endswith(EXT):
        return np.load(io.BytesIO(_decompress(path)), allow_pickle=False)
    if path.suffix == LEGACY_EXT:
        return np.load(path)["depth_mm"]
    raise ValueError(f"[COLLECT_DEPTH] unknown depth format: {path}")


def frames(path: Path) -> int:
    """Frame count of a depth file (reads only the npy header)."""
    import zipfile

    path = Path(path)
    if str(path).endswith(EXT):
        try:
            import zstandard
        except ImportError:
            return int(read(path).shape[0])
        with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
            return _header_rows(io.BytesIO(r.read(4096)))
    if path.suffix == LEGACY_EXT:
        with zipfile.ZipFile(path) as z, z.open("depth_mm.npy") as f:
            return _header_rows(f)
    raise ValueError(f"[COLLECT_DEPTH] unknown depth format: {path}")


def _header_rows(f) -> int:
    import numpy as np

    ver = np.lib.format.read_magic(f)
    rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
    return int(rd(f)[0][0])


def place(src: Path, dst: Path) -> str | None:
    """Put depth file src at the dataset place dst: a hard link to a file in the dst folder with byte-identical content
    (first by name) when there is one, else a copy. Returns the linked file name (None if none)."""
    import filecmp

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    same = next(
        (
            q
            for q in sorted(dst.parent.glob(f"*{EXT}"))
            if q.name != dst.name and q.stat().st_size == size and filecmp.cmp(src, q, shallow=False)
        ),
        None,
    )
    if same is not None and dst.exists() and os.path.samefile(same, dst):
        return same.name  # already a link of that file (bundle run again)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    if same is not None:
        os.link(same, tmp)
    else:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    if tmp.exists():  # POSIX rename of two names of one file does nothing: remove the left tmp
        tmp.unlink()
    return same.name if same is not None else None
