"""GR00T N1.7 fine-tuning recipe (Isaac-GR00T n1.7-release `gr00t/experiment/launch_finetune.py` under torchrun).

  python -m experiments.policies.gr00t.train --gr00t-repo <Isaac-GR00T checkout with .venv> --base <N1.7 base or
      checkpoint dir> --dataset <LeRobot dir> --out <run dir> --gpus 0,1,2,3,4,5,6,7 (--steps N | --exposure 150)
      --save-steps N [--global-batch 256] [--lr 1.5e-4] [--ft.<option>=<value> …] [other options below]

Recipe A (the paper's GR00T runs): global batch 256 split over the GPUs, lr 1.5e-4, warm-up 0.05, weight decay 1e-5,
gradient accumulation 1, frozen vision tower, state dropout 0, episode sampling rate 0.1, shard size 1024, 32 shards per
GPU per epoch, 8 data workers per rank, colour jitter (brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08), every
frame a training start (`--ft.tail_starts=1`) and min/max normalisation (`--ft.gr00t_minmax=1`); seed = Isaac-GR00T's
default 42. Steps from an exposure: round(exposure × dataset frames / global batch / 1000) × 1000. `--base` given a
fine-tuned checkpoint starts a new run from its weights (optimizer and schedule fresh — the mid-training init).
With ≥ 4 GPUs the NCCL defaults P2P_LEVEL=SYS, MIN_NCHANNELS=32, BUFFSIZE=16 MiB apply unless the caller set them.
The modality config is modality_wuji2.py; checkpoints go to <out>/ckpt, the log to <out>/stage_train.log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MARK = "[GR00T_TRAIN]"
RECIPE_FT = ["--ft.tail_starts=1", "--ft.gr00t_minmax=1"]
JITTER = ["--color-jitter-params", "brightness", "0.3", "contrast", "0.4", "saturation", "0.5", "hue", "0.08"]
NCCL_DEFAULTS = {"NCCL_P2P_LEVEL": "SYS", "NCCL_MIN_NCHANNELS": "32", "NCCL_BUFFSIZE": "16777216"}


def steps_from_exposure(exposure: float, frames: int, global_batch: int) -> int:
    """Optimizer steps that show every frame `exposure` times (rounded to thousands)."""
    return round(exposure * frames / global_batch / 1000) * 1000


def shards_per_epoch(a, n_gpu: int) -> int:
    """Shards dealt per epoch: every (rank × worker) slot must get one, or later ranks have an empty schedule."""
    spe = a.shards_per_epoch or a.shards_per_gpu * n_gpu
    if spe < n_gpu * a.workers:
        raise SystemExit(
            f"{MARK} shards per epoch {spe} < {n_gpu} gpus × {a.workers} workers — later ranks get no shard"
        )
    return spe


def finetune_args(a, n_gpu: int, steps: int) -> list[str]:
    """launch_finetune arguments of one run."""
    if a.global_batch % n_gpu:
        raise SystemExit(f"{MARK} global batch {a.global_batch} does not split over {n_gpu} GPUs")
    out = [
        "--base-model-path",
        a.base,
        "--dataset-path",
        a.dataset,
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(ROOT / "experiments/policies/gr00t/modality_wuji2.py"),
        "--output-dir",
        str(Path(a.out) / "ckpt"),
        "--num-gpus",
        str(n_gpu),
        "--global-batch-size",
        str(a.global_batch),
        "--gradient-accumulation-steps",
        str(a.grad_accum),
        "--learning-rate",
        a.lr,
        "--warmup-ratio",
        a.warmup,
        "--max-steps",
        str(steps),
        "--save-steps",
        str(a.save_steps),
        "--save-total-limit",
        str(a.save_limit),
        "--state-dropout-prob",
        a.state_dropout,
        "--shard-size",
        str(a.shard_size),
        "--episode-sampling-rate",
        a.episode_sampling_rate,
        "--num-shards-per-epoch",
        str(shards_per_epoch(a, n_gpu)),
        "--weight-decay",
        a.weight_decay,
        "--dataloader-num-workers",
        str(a.workers),
    ]
    if a.tune_visual:
        out.append("--tune-visual")
    return out + JITTER


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gr00t-repo", required=True, help="Isaac-GR00T n1.7-release checkout with its .venv")
    ap.add_argument("--base", required=True, help="N1.7 base snapshot dir, or a checkpoint dir to start from")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--steps", type=int)
    g.add_argument("--exposure", type=float)
    ap.add_argument("--save-steps", type=int, required=True)
    ap.add_argument("--save-limit", type=int, default=3)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--lr", default="1.5e-4")
    ap.add_argument("--warmup", default="0.05")
    ap.add_argument("--weight-decay", default="1e-5")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--tune-visual", action="store_true")
    ap.add_argument("--state-dropout", default="0")
    ap.add_argument("--episode-sampling-rate", default="0.1")
    ap.add_argument("--shard-size", type=int, default=1024)
    ap.add_argument("--shards-per-gpu", type=int, default=32)
    ap.add_argument("--shards-per-epoch", type=int, default=0, help="absolute count in place of --shards-per-gpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-recipe-ft", action="store_true", help="leave out recipe A's tail_starts and gr00t_minmax")
    ap.add_argument("--no-nccl-defaults", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=0)
    a, rest = ap.parse_known_args(argv)
    bad = [x for x in rest if not x.startswith("--ft.")]
    if bad:
        ap.error(f"unknown arguments {bad}")
    a.ft = ([] if a.no_recipe_ft else RECIPE_FT) + rest
    return a


def main(argv=None) -> int:
    a = parse(argv)
    gpus = [x for x in a.gpus.split(",") if x]
    steps = a.steps
    if steps is None:
        frames = json.loads((Path(a.dataset) / "meta/info.json").read_text())["total_frames"]
        steps = steps_from_exposure(a.exposure, frames, a.global_batch)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(a.gr00t_repo)
    entry = (
        str(ROOT / "experiments/policies/gr00t/launch_finetune.py") if a.ft else "gr00t/experiment/launch_finetune.py"
    )
    cmd = [
        str(repo / ".venv/bin/torchrun"),
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={len(gpus)}",
        entry,
        *finetune_args(a, len(gpus), steps),
        *a.ft,
    ]
    if a.timeout_s:
        cmd = ["timeout", str(a.timeout_s)] + cmd
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
    if len(gpus) >= 4 and not a.no_nccl_defaults:
        env.update({k: v for k, v in NCCL_DEFAULTS.items() if k not in os.environ})
    print(f"{MARK} steps {steps} · {' '.join(cmd)[:600]}", flush=True)
    with open(out / "stage_train.log", "a") as lf:
        rc = subprocess.run(cmd, cwd=repo, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"{MARK} rc={rc} · {out / 'stage_train.log'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
