"""Diffusion Policy training recipe (LeRobot 0.3.3 `policies/diffusion`, the last release that reads LeRobot v2.1).

  python -m experiments.policies.dp.train --dataset <LeRobot dir> --out <run dir> --batch 64 --steps 166650
      --save-freq 16665 [--gpus 0 | 0,1,2,3] [--lr X] [--workers 10] [--trainer ema|plain]
      [--inet-resnet18 <resnet18-f37072fd.pth>] [--episodes "[0,1]"] [--wandb-project P --wandb-entity E
      --wandb-mode online|offline --wandb-run NAME] [--resume] [--timeout-s N] [--dp.<option>=<value> …]

Recipe: two cameras, random crop (160, 288), n_obs_steps 2, horizon 16, n_action_steps 8, seed 42, pyav video decode,
log every 20 steps, the rest LeRobot defaults. Checkpoints go to <out>/train (LeRobot's output_dir); the stage log is
<out>/stage_train.log. One GPU runs `train_ema` (LeRobot's train loop + weight EMA, pretrained_model = the EMA weights)
or plain `lerobot-train` (`--trainer plain`); several GPUs run `train_ddp` under torchrun (always with the EMA; --batch
is per GPU). `--inet-resnet18` starts the visual encoder from torchvision ResNet18 IMAGENET1K_V1 (the given file,
checked by its sha256 prefix, copied into TORCH_HOME; never downloaded) together with BatchNorm
(`use_group_norm=false`, which LeRobot requires for pretrained weights). `--dp.*` options go to the wrappers
(train_opts.py).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MARK = "[DP_TRAIN]"
RESNET18 = ("resnet18-f37072fd.pth", "f37072fd")
RECIPE = [
    "--dataset.video_backend=pyav",
    "--policy.type=diffusion",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--policy.crop_shape=[160,288]",
    "--policy.n_obs_steps=2",
    "--policy.horizon=16",
    "--policy.n_action_steps=8",
    "--log_freq=20",
    "--save_checkpoint=true",
    "--seed=42",
]


def wandb_args(a) -> list[str]:
    """LeRobot W&B arguments (off unless a project is given)."""
    if not a.wandb_project:
        return ["--wandb.enable=false"]
    return [
        "--wandb.enable=true",
        f"--wandb.project={a.wandb_project}",
        f"--wandb.entity={a.wandb_entity}",
        "--wandb.disable_artifact=true",
        f"--wandb.mode={a.wandb_mode}",
    ]


def lerobot_args(a, dataset: Path) -> list[str]:
    """The lerobot-train arguments of one training run (recipe + run values)."""
    out = [
        f"--dataset.repo_id=e5/{dataset.name}",
        f"--dataset.root={dataset}",
        *RECIPE[:4],
        *RECIPE[4:8],
        f"--batch_size={a.batch}",
        f"--steps={a.steps}",
        f"--num_workers={a.workers}",
        RECIPE[8],
        f"--save_freq={a.save_freq}",
        *RECIPE[9:],
        *wandb_args(a),
        f"--output_dir={Path(a.out) / 'train'}",
        f"--job_name={a.wandb_run or 'train'}",
    ]
    if a.inet_resnet18:
        out += ["--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1", "--policy.use_group_norm=false"]
    if a.lr:
        out.append(f"--policy.optimizer_lr={a.lr}")
    if a.episodes:
        out.append(f"--dataset.episodes={a.episodes}")
    return out


def trainer(a, gpus: list[str]) -> list[str]:
    """Command prefix of the trainer for the GPU count."""
    if len(gpus) > 1:
        if a.trainer != "ema":
            raise SystemExit(f"{MARK} several GPUs run train_ddp, which always keeps the EMA (--trainer ema)")
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(gpus)}",
            "-m",
            "experiments.policies.dp.train_ddp",
        ]
    if a.trainer == "plain":
        if a.wandb_project:
            raise SystemExit(
                f"{MARK} W&B needs the EMA trainer (plain lerobot-train lacks the wandb ≥ 0.31 Run.get_url shim)"
            )
        return [str(Path(sys.executable).with_name("lerobot-train"))]
    return [sys.executable, "-m", "experiments.policies.dp.train_ema"]


def place_resnet18(src: str) -> None:
    """Copy the ResNet18 IMAGENET1K_V1 weights into TORCH_HOME (sha256 prefix checked; never downloaded)."""
    th = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")) / "hub/checkpoints"
    th.mkdir(parents=True, exist_ok=True)
    dst = th / RESNET18[0]
    if not dst.is_file():
        shutil.copyfile(src, dst)
    if hashlib.sha256(dst.read_bytes()).hexdigest()[:8] != RESNET18[1]:
        raise SystemExit(f"{MARK} FAIL {dst}: sha256 is not {RESNET18[1]}…")
    print(f"[E5_DP_INET] on · ResNet18 IMAGENET1K_V1 from {dst} · use_group_norm=false", flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="LeRobot v2.1 dataset dir (its name is the repo id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, required=True, help="per GPU")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--save-freq", type=int, required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--lr", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--trainer", choices=("ema", "plain"), default="ema")
    ap.add_argument("--inet-resnet18", default="", help="torchvision resnet18-f37072fd.pth to start the encoder from")
    ap.add_argument("--episodes", default="", help="LeRobot episode list, e.g. [0,1] (default all)")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--wandb-entity", default="")
    ap.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    ap.add_argument("--wandb-run", default="")
    ap.add_argument("--resume", action="store_true", help="continue <out>/train/checkpoints/last (one GPU)")
    ap.add_argument("--timeout-s", type=int, default=0)
    a, rest = ap.parse_known_args(argv)
    bad = [x for x in rest if not x.startswith("--dp.")]
    if bad:
        ap.error(f"unknown arguments {bad}")
    a.dp = rest
    return a


def main(argv=None) -> int:
    a = parse(argv)
    gpus = [g for g in a.gpus.split(",") if g]
    ds, out = Path(a.dataset).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.inet_resnet18:
        place_resnet18(a.inet_resnet18)
    wb = [f"--dp.wandb={'on' if a.wandb_project else 'off'}"] if a.trainer == "ema" else []
    last = out / "train/checkpoints/last/pretrained_model/train_config.json"
    if a.resume:
        if len(gpus) != 1 or not last.is_file():
            raise SystemExit(f"{MARK} --resume needs one GPU and {last}")
        cmd = trainer(a, gpus) + [
            "--resume=true",
            f"--config_path={last}",
            f"--dataset.root={ds}",
            f"--num_workers={a.workers}",
            *wb,
            *a.dp,
        ]
    else:
        cmd = trainer(a, gpus) + lerobot_args(a, ds) + wb + a.dp
    if a.timeout_s:
        cmd = ["timeout", str(a.timeout_s)] + cmd
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), PYTHONPATH=str(ROOT))
    print(f"{MARK} {' '.join(cmd)[:600]}", flush=True)
    with open(out / "stage_train.log", "a") as lf:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"{MARK} rc={rc} · {out / 'stage_train.log'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
