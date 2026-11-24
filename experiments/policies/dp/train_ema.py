"""LeRobot 0.3.3 `lerobot-train` + weight EMA (called with the lerobot-train arguments in place of `lerobot-train`,
see train.py). The DP paper keeps an EMA of the weights and evaluates it; LeRobot 0.3.3 diffusion has none
(neither its diffusion policy nor its train script has EMA code).

EMA = Chi's diffusion_policy `diffusion_policy/model/diffusion/ema_model.py` EMAModel, same formula and defaults:
  update_after_step 0 · inv_gamma 1.0 · power 0.75 · min_value 0.0 · max_value 0.9999
  decay(k) = 0 if s <= 0 else clamp(1 - (1 + s / inv_gamma) ** -power, min_value, max_value) with s = max(0, k -
  update_after_step - 1), k = number of EMA updates done before this one. One update after every optimizer step (after
  lr_scheduler.step). Parameters with requires_grad and outside BatchNorm: ema = decay * ema + (1 - decay) * param.
  Other parameters (frozen
  · BatchNorm) and all buffers: copied.

LeRobot's train() is run as is (no copy of the loop, site-packages untouched); three module globals of
lerobot.scripts.train are
wrapped:
  update_policy   → the original, then ema.step(policy)
  save_checkpoint → pretrained_model/ = the EMA weights (what the judge loads); training_state/ also gets
                    e5_ema_raw_model.pt (the raw weights) · e5_ema_shadow.pt (the EMA weights) · e5_ema.json (updates
                    · decay · constants)
  load_training_state (resume) → the raw weights back into the policy (pretrained_model holds the EMA) and the EMA
  shadow restored;
                    a resume without those files dies.
Marks: '[E5_DP_EMA] on …' at start · '[E5_DP_EMA] saved ema step=N …' at each save · '[E5_DP_EMA] resume …'.
wandb (install_wandb — also used by train_ddp): --dp.wandb=on|off from the launcher (unset = no claim, LeRobot's
  arguments decide) — on with --wandb.enable≠true (or off with true) dies before training (the switch must reach the
  trainer). Rank 0: Run.get_url shim (wandb ≥ 0.31 has only Run.url; LeRobot 0.3.3 calls get_url) and, right after
  wandb.init, '[WANDB] on project=… entity=… run=… id=… mode=online|offline url=…'.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from torch.nn.modules.batchnorm import _BatchNorm

UPDATE_AFTER_STEP, INV_GAMMA, POWER, MIN_VALUE, MAX_VALUE = 0, 1.0, 0.75, 0.0, 0.9999
MARK = "[E5_DP_EMA]"
WB_MARK = "[WANDB]"


def install_wandb(T, rank: int = 0) -> None:
    """dp.wandb guard + rank-0 wandb marker (module docstring). Ranks > 0 never build a WandBLogger."""
    from experiments.policies.dp import train_opts as TO

    sw = TO.get("wandb")
    if sw not in ("", "on", "off"):
        raise SystemExit(f"{WB_MARK} FAIL dp.wandb '{sw}' — on|off")
    argv = sys.argv[1:]
    en = [x.split("=", 1)[1].strip().lower() for x in argv if x.startswith("--wandb.enable=")]
    if en:
        on = en[-1] in ("true", "1", "yes")
    else:  # resume (--config_path=<train_config.json>, no --wandb.* flags): the saved config decides
        cp = next((x.split("=", 1)[1] for x in argv if x.startswith("--config_path=")), "")
        on = bool(cp) and bool(json.loads(Path(cp).read_text()).get("wandb", {}).get("enable", False))
    if sw == "on" and not on:
        raise SystemExit(
            f"{WB_MARK} FAIL dp.wandb=on but the trainer got wandb.enable={en[-1] if en else 'unset'} — "
            "the switch did not reach it"
        )
    if sw == "off" and on:
        raise SystemExit(f"{WB_MARK} FAIL dp.wandb=off but the trainer got --wandb.enable=true")
    if rank > 0:
        T.WandBLogger = lambda cfg: None
        return
    if not on:
        print(f"{WB_MARK} off (dp.wandb={sw or 'unset'} · --wandb.enable={en[-1] if en else 'unset'})", flush=True)
        return
    import wandb
    from wandb.sdk.wandb_run import Run

    if not hasattr(Run, "get_url"):
        Run.get_url = lambda self: self.url
        print(f"{MARK} wandb {wandb.__version__}: Run.get_url → Run.url (LeRobot 0.3.3 compat)", flush=True)
    orig = T.WandBLogger

    def wandb_logger(cfg):
        lg = orig(cfg)  # wandb.init inside — a failure raises here (no silent drop)
        r = wandb.run
        off = bool(getattr(r, "offline", False)) or str(getattr(r.settings, "mode", "")) == "offline"
        print(
            f"{WB_MARK} on project={r.project} entity={r.entity} run={r.name} id={r.id} "
            f"mode={'offline' if off else 'online'} "
            f"url={'-' if off else r.url} · wandb {wandb.__version__}",
            flush=True,
        )
        return lg

    T.WandBLogger = wandb_logger


def ema_decay(k: int) -> float:
    """Chi's EMAModel.get_decay(optimization_step=k)."""
    s = max(0, k - UPDATE_AFTER_STEP - 1)
    if s <= 0:
        return 0.0
    return max(MIN_VALUE, min(1 - (1 + s / INV_GAMMA) ** -POWER, MAX_VALUE))


class EMAModel:
    def __init__(self, model: torch.nn.Module):
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.optimization_step = 0
        self.decay = 0.0

    @torch.no_grad()
    def step(self, new_model: torch.nn.Module) -> None:
        self.decay = ema_decay(self.optimization_step)
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules(), strict=True):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False), strict=True
            ):
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            for buf, ema_buf in zip(module.buffers(recurse=False), ema_module.buffers(recurse=False), strict=True):
                ema_buf.copy_(buf)
        self.optimization_step += 1


STATE = {"policy": None, "ema": None}


def inet_proof(policy) -> None:
    """dp.inet_proof=1: the backbone's conv1 at init must be torchvision's ResNet18 IMAGENET1K_V1 conv1 — printed, dies
    if not.
    """
    import hashlib

    from experiments.policies.dp import train_opts as TO

    if TO.get("inet_proof") != "1":
        return
    from torchvision.models import ResNet18_Weights

    ref = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)["conv1.weight"]
    convs = [
        (n, m)
        for n, m in policy.named_modules()
        if "rgb_encoder" in n and isinstance(m, torch.nn.Conv2d) and tuple(m.weight.shape) == (64, 3, 7, 7)
    ]
    if not convs:
        raise SystemExit("[E5_DP_INET] FAIL no 64×3×7×7 conv in rgb_encoder")

    def h(t):
        return hashlib.sha256(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]

    hr = h(ref)
    bad = [n for n, m in convs if h(m.weight) != hr]
    gn = sum(isinstance(m, torch.nn.GroupNorm) for n, m in policy.named_modules() if "rgb_encoder" in n)
    bn = sum(isinstance(m, _BatchNorm) for n, m in policy.named_modules() if "rgb_encoder" in n)
    print(
        f"[E5_DP_INET] conv1 sha {h(convs[0][1].weight)} = torchvision IMAGENET1K_V1 conv1 sha {hr} · "
        f"{len(convs) - len(bad)}/{len(convs)} equal · "
        f"rgb_encoder BatchNorm {bn} · GroupNorm {gn}",
        flush=True,
    )
    if bad or gn:
        raise SystemExit(f"[E5_DP_INET] FAIL conv1 differs {bad} · GroupNorm {gn}")


def _cpu_state(m: torch.nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()}


def _write_ema_state(raw, ts, ema, step):
    """Write the raw weights, the EMA shadow and the EMA meta into the training-state dir ts; print the save marker."""
    torch.save(raw, ts / "e5_ema_raw_model.pt")
    torch.save(_cpu_state(ema.averaged_model), ts / "e5_ema_shadow.pt")
    (ts / "e5_ema.json").write_text(
        json.dumps(
            {
                "step": step,
                "optimization_step": ema.optimization_step,
                "last_decay": ema.decay,
                "update_after_step": UPDATE_AFTER_STEP,
                "inv_gamma": INV_GAMMA,
                "power": POWER,
                "min_value": MIN_VALUE,
                "max_value": MAX_VALUE,
                "pretrained_model": "ema",
            }
        )
    )
    print(
        f"{MARK} saved ema step={step} updates={ema.optimization_step} decay={ema.decay:.6f} → "
        "pretrained_model = EMA · "
        f"training_state = raw + EMA",
        flush=True,
    )


def install(T) -> None:
    """Wrap the three globals of lerobot.scripts.train (module T)."""
    from lerobot.utils import train_utils as TU

    TSD = TU.TRAINING_STATE_DIR
    orig_update, orig_save, orig_load, orig_make = (
        T.update_policy,
        T.save_checkpoint,
        T.load_training_state,
        T.make_policy,
    )

    def make_policy(*a, **kw):
        p = orig_make(*a, **kw)
        STATE["policy"] = p
        inet_proof(p)
        return p

    def update_policy(train_metrics, policy, *a, **kw):
        if STATE["ema"] is None:
            STATE["ema"] = EMAModel(policy)
        out = orig_update(train_metrics, policy, *a, **kw)
        STATE["ema"].step(policy)
        return out

    def save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, scheduler=None):
        ema = STATE["ema"]
        if ema is None:
            raise SystemExit(f"{MARK} FAIL save at step {step} before any EMA update")
        raw = _cpu_state(policy)
        policy.load_state_dict(ema.averaged_model.state_dict())  # strict: same module → same keys
        try:
            orig_save(checkpoint_dir, step, cfg, policy, optimizer, scheduler)
        finally:
            policy.load_state_dict(raw)
        ts = Path(checkpoint_dir) / TSD
        _write_ema_state(raw, ts, ema, step)

    def load_training_state(checkpoint_dir, optimizer, scheduler):
        step, optimizer, scheduler = orig_load(checkpoint_dir, optimizer, scheduler)
        ts = Path(checkpoint_dir) / TSD
        need = [ts / "e5_ema_raw_model.pt", ts / "e5_ema_shadow.pt", ts / "e5_ema.json"]
        if not all(f.exists() for f in need):
            raise SystemExit(f"{MARK} FAIL resume without EMA state in {ts}")
        pol = STATE["policy"]
        dev = next(pol.parameters()).device
        pol.load_state_dict({k: v.to(dev) for k, v in torch.load(need[0], map_location="cpu").items()})
        ema = EMAModel(pol)
        ema.averaged_model.load_state_dict({k: v.to(dev) for k, v in torch.load(need[1], map_location="cpu").items()})
        meta = json.loads(need[2].read_text())
        ema.optimization_step, ema.decay = int(meta["optimization_step"]), float(meta["last_decay"])
        STATE["ema"] = ema
        print(f"{MARK} resume step={step} updates={ema.optimization_step} raw → policy · shadow restored", flush=True)
        return step, optimizer, scheduler

    T.make_policy, T.update_policy, T.save_checkpoint, T.load_training_state = (
        make_policy,
        update_policy,
        save_checkpoint,
        load_training_state,
    )


def main() -> None:
    from experiments.policies.dp import train_opts as TO

    sys.argv = [sys.argv[0]] + TO.take(sys.argv[1:])
    import lerobot.scripts.train as T

    install(T)
    install_wandb(T, 0)
    print(
        f"{MARK} on power={POWER} max={MAX_VALUE} inv_gamma={INV_GAMMA} min={MIN_VALUE} "
        f"update_after_step={UPDATE_AFTER_STEP} "
        f"(Chi diffusion_policy EMAModel) · argv {' '.join(sys.argv[1:])[:200]}",
        flush=True,
    )
    T.main()


if __name__ == "__main__":
    main()
