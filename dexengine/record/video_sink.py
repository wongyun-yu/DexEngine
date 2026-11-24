"""Video stream encoder: single owner of the encode settings (CRF, preset, executable).

Frames are not held in RAM: one ffmpeg per stream is opened on a stdin pipe (rawvideo rgb24), and one writer thread
takes frames from a small queue (QUEUE) and writes them to the pipe (pipe writes release the GIL). RAM = O(streams x
QUEUE x one frame). The quality defaults (CRF 18, slow) equal the old render's (rl.render.clip_video reads them from
this module); the evidence and measurements are in that file's header.
The executable is the system ffmpeg (/usr/bin/ffmpeg, a build with nvenc); the imageio-ffmpeg bundle has no nvenc.
Marker: `[VIDEO_SINK] exe=… codec=libx264 preset=… crf=… fps=… threads=…` (once, when the first stream opens).
"""

from __future__ import annotations

import contextlib
import os
import queue
import subprocess
import threading

import numpy as np

FFMPEG_EXE = "/usr/bin/ffmpeg"
FFPROBE_EXE = "/usr/bin/ffprobe"
CODEC = "libx264"
QUEUE = 3
_announced = False


def crf() -> str:
    """CRF: an explicit empty value (DEXENGINE_VIDEO_CRF="") selects the old render's qscale path
    (rl.render.clip_video header)."""
    return os.environ.get("DEXENGINE_VIDEO_CRF", "18").strip()


def preset() -> str:
    return os.environ.get("DEXENGINE_VIDEO_PRESET", "slow").strip()


class VideoSink:  # noqa: V102 — used by dexengine/rl/render/replay.py
    """push(frame[H,W,3] uint8) · finish() (end of input, no wait) · close() -> frames written (failures die here)."""

    def __init__(self, path: str, width: int, height: int, fps: int, threads: int = 1):
        global _announced
        c = crf()
        if not c:
            raise RuntimeError("[VIDEO_SINK] CRF is empty: the stream encoder uses CRF only (no qscale path)")
        self.path, self.w, self.h = path, int(width), int(height)
        self.n = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".part.mp4"
        self.tmp = tmp
        cmd = [
            FFMPEG_EXE,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.w}x{self.h}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-c:v",
            CODEC,
            "-preset",
            preset(),
            "-crf",
            c,
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(int(threads)),
            "-movflags",
            "+faststart",
            tmp,
        ]
        if not _announced:
            print(
                f"[VIDEO_SINK] exe={FFMPEG_EXE} codec={CODEC} preset={preset()} crf={c} fps={int(fps)} "
                f"threads={int(threads)}/stream · queue {QUEUE}",
                flush=True,
            )
            _announced = True
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE)
        self.err: BaseException | None = None
        self.finished = False
        self.th = threading.Thread(target=self._writer, daemon=True)
        self.th.start()

    def _writer(self) -> None:
        try:
            while True:
                fr = self.q.get()
                if fr is None:
                    break
                self.proc.stdin.write(memoryview(fr).cast("B"))
        except BaseException as e:  # noqa: BLE001 — any writer failure (broken pipe) is surfaced in close
            self.err = e
        with contextlib.suppress(BrokenPipeError, OSError):
            # EOF: ffmpeg starts flushing its lookahead tail (concurrently with other streams)
            self.proc.stdin.close()

    def push(self, frame: np.ndarray) -> None:  # noqa: V105 — used by dexengine/rl/render/replay.py
        if frame.shape != (self.h, self.w, 3) or frame.dtype != np.uint8:
            raise ValueError(f"[VIDEO_SINK] frame shape {frame.shape}/{frame.dtype} != ({self.h},{self.w},3)/uint8")
        if self.err is not None or self.finished:
            raise RuntimeError(f"[VIDEO_SINK] {self.path}: write failed {self.err or 'already finished'}")
        self.q.put(np.ascontiguousarray(frame))
        self.n += 1

    def finish(self) -> None:
        """End of input, without waiting. Waiting for each stream's x264 lookahead tail (dozens of frames for slow) in
        turn would serialise the encodes: finished streams only finish, and the wait (close) comes later, all at
        once."""
        if not self.finished:
            self.finished = True
            self.q.put(None)

    def close(self) -> int:
        self.finish()
        self.th.join()
        rc = self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace").strip()
        if rc != 0 or self.err is not None:
            raise RuntimeError(f"[VIDEO_SINK] {self.path}: ffmpeg rc={rc} {err[-400:]} {self.err or ''}")
        os.replace(self.tmp, self.path)
        return self.n


def probe_frames(path: str) -> int:  # noqa: V103 — used by dexengine/rl/render/replay.py
    """Frame count from ffprobe (decode count: a measurement, not the container's statement)."""
    r = subprocess.run(
        [
            FFPROBE_EXE,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return -1
