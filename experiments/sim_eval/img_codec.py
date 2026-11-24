"""E5 closed loop — the policy's images through the training data's codec chain.

The training images are not raw renders. They went through two lossy encodes:
  1. record render → mp4 at 1280×720 (the record writer's MP4_ARGS — libx264 · yuv420p · its ffmpeg_params);
  2. converter (experiments/policies/data/convert.py) → scale=<W>:<H>:flags=area · libx264 · yuv420p · preset
  medium ·
     crf · -g (values from the training dataset's meta/e5_conversion.json "video"), decoded by the trainer.
The closed loop otherwise feeds the raw render box-averaged to 320×180 — mean 1.7–2.2/255 · p99 8–10/255 off the
training frame. With the session option policy_img_codec=dataset the policy (observe · query) gets the same chain
applied to the full-resolution capture of that frame instead; the recorded frames (episode.npz · videos · judge) stay
raw.

Each frame is encoded alone (one-frame streams = intra frames). Training frames are intra for every second frame
(-g 2); the others are predicted, so this matches intra frames exactly and the rest approximately.

  policy_img_codec unset → off · "dataset" → on · anything else dies.
Marker: [E5_IMG_CODEC] at start (both stages' arguments and the dataset meta they came from).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MARK = "[E5_IMG_CODEC]"
VALUES = ("dataset",)


def from_env(policy_cfg: dict | None):
    from experiments.sim_eval.session_cfg import value

    v = value("policy_img_codec", "")
    if not v:
        return None
    if v not in VALUES:
        raise SystemExit(f"{MARK} session option policy_img_codec={v!r} — {VALUES} or unset")
    meta = (policy_cfg or {}).get("dataset_meta")
    if not meta:
        raise SystemExit(f"{MARK} policy_img_codec=dataset needs policy.dataset_meta (the training dataset root)")
    return ImgCodec(meta)


class ImgCodec:
    def __init__(self, root: str):
        from dexengine.record.render import writer as CW  # the collect render's mp4 argument owner (stage 1)

        conv = Path(root) / "meta" / "e5_conversion.json"
        if not conv.is_file():
            raise SystemExit(f"{MARK} {conv} missing — the converter's video settings are read from there")
        v = json.loads(conv.read_text())["video"]
        v = json.loads(v.replace("'", '"')) if isinstance(v, str) else v
        if v.get("codec") != "h264" or v.get("filter") != "area":
            raise SystemExit(f"{MARK} {conv} video {v} — only h264 + area is implemented")
        self.W, self.H = int(v["to"][0]), int(v["to"][1])
        a = CW.MP4_ARGS
        if a.get("codec") != "libx264" or a.get("pixelformat") != "yuv420p" or a.get("quality") or a.get("bitrate"):
            raise SystemExit(
                f"{MARK} collect_writer.MP4_ARGS {a} — only libx264 · yuv420p · ffmpeg_params is implemented"
            )
        self.s1 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", *[str(x) for x in a.get("ffmpeg_params", [])]]
        self.s2 = [
            "-vf",
            f"scale={self.W}:{self.H}:flags=area",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            str(int(v["crf"])),
            "-g",
            str(int(v["gop"])),
        ]
        self.n = 0
        print(
            f"{MARK} on · stage 1 (collect mp4) {' '.join(self.s1)} · stage 2 (converter {conv}) {' '.join(self.s2)} "
            f"· one-frame streams",
            flush=True,
        )

    def __call__(self, img):
        import numpy as np

        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
        base = ["ffmpeg", "-nostdin", "-loglevel", "error"]
        s1 = subprocess.run(
            base
            + [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                "30",
                "-i",
                "-",
                *self.s1,
                "-f",
                "h264",
                "-",
            ],
            input=rgb.tobytes(),
            capture_output=True,
            check=True,
        ).stdout
        s2 = subprocess.run(
            base + ["-f", "h264", "-i", "-", *self.s2, "-f", "h264", "-"], input=s1, capture_output=True, check=True
        ).stdout
        raw = subprocess.run(
            base + ["-f", "h264", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            input=s2,
            capture_output=True,
            check=True,
        ).stdout
        if len(raw) != self.W * self.H * 3:
            raise RuntimeError(f"{MARK} decoded {len(raw)} bytes ≠ one {self.W}×{self.H} frame")
        self.n += 1
        return np.frombuffer(raw, np.uint8).reshape(self.H, self.W, 3).copy()
