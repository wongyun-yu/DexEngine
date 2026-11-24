"""Options of the DP training wrappers (train_ema, train_ddp) that are not LeRobot's.

They are given as `--dp.<name>=<value>` among the lerobot-train arguments and removed from sys.argv before LeRobot
parses it (`take`). Values are strings; the defaults are plain LeRobot behaviour. An unknown `--dp.` name dies.

  wandb               on|off — the launcher's W&B switch, checked against --wandb.enable (train_ema.install_wandb)
  inet_proof          1 = check that the visual backbone's conv1 is torchvision ResNet18 IMAGENET1K_V1 at init
  init_from           <pretrained_model dir> — fine-tuning start: the whole state_dict (weights + normalisation
                      buffers) loaded strict after make_policy (train_ddp)
  state_dropout       p — per training sample, the whole normalised observation.state set to 0 with probability p
  state_noise         σ — N(0, σ²) added to the normalised observation.state of the other samples
  frame_cache         <dir> — decoded uint8 frames (frame_cache.py) in place of the per-item mp4 decode
  columns             1 = parquet columns served from numpy (frame_cache.install_columns)
  compile             1 = nn.Module.compile of diffusion.rgb_encoder and diffusion.unet
  amp                 fp32|bf16 — forward + loss under bf16 autocast (weights, grads, optimizer stay fp32)
  ddp_comm            fp32|bf16|fp16 — gradient all-reduce compression hook
  ddp_bucket_mb       DDP bucket size (MB)
  ddp_bucket_view     1 = gradient_as_bucket_view
  fused_adamw         1 = torch AdamW with fused=True (same update)
  persistent_workers  1 = DataLoader workers kept across epochs
  step_sync           1 = barrier at each step start and a stream sync before grad clipping
"""

from __future__ import annotations

PREFIX = "--dp."
DEFAULTS = {
    "wandb": "",
    "inet_proof": "0",
    "init_from": "",
    "state_dropout": "",
    "state_noise": "",
    "frame_cache": "",
    "columns": "0",
    "compile": "0",
    "amp": "fp32",
    "ddp_comm": "fp32",
    "ddp_bucket_mb": "25",
    "ddp_bucket_view": "0",
    "fused_adamw": "0",
    "persistent_workers": "0",
    "step_sync": "0",
}
_VALUES = dict(DEFAULTS)


def take(argv: list[str]) -> list[str]:
    """Remove the `--dp.<name>=<value>` options from argv (stored for `get`) and return the rest."""
    rest = []
    for x in argv:
        if not x.startswith(PREFIX):
            rest.append(x)
            continue
        k, sep, v = x[len(PREFIX) :].partition("=")
        k = k.replace("-", "_")
        if not sep or k not in DEFAULTS:
            raise SystemExit(f"[DP_OPTS] FAIL {x!r} — --dp.<name>=<value>, names {sorted(DEFAULTS)}")
        _VALUES[k] = v
    return rest


def get(name: str) -> str:
    """Value of one option (its default when not given)."""
    return _VALUES[name]


def given() -> dict:
    """Options that differ from their default (for the start marker)."""
    return {k: v for k, v in _VALUES.items() if v != DEFAULTS[k]}
