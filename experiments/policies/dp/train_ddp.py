"""LeRobot 0.3.3 DP training on several GPUs of one machine (torchrun · PyTorch DDP · NCCL) + the same weight EMA as
train_ema. LeRobot 0.3.3 has no multi-GPU path (accelerate arrives in 0.4, which needs dataset v3.0), so this wraps
module globals of lerobot.scripts.train like the EMA wrapper does (no copy of the loop · site-packages untouched):

  EpisodeAwareSampler → each rank takes its own slice of one shared permutation (seed 42 + epoch), equal lengths.
                        Without this every rank trains on the same batches (LeRobot's sampler uses the global RNG).
  update_policy       → the policy is wrapped once in DDP; LeRobot's update runs on the wrapper (gradients averaged
                        across ranks before clip + step). EMA steps on the raw policy (train_ema).
  make_policy         → after the policy exists, ranks > 0 reseed (42 + 1000·rank) so crop · noise · timesteps
  differ
                        across ranks (weights are still rank 0's: DDP broadcasts them).
  save_checkpoint · update_last_checkpoint · WandBLogger → rank 0 only; a barrier after each save.
  ranks > 0           → --output_dir gets '.rank<r>' (never written: no save, no wandb) so cfg.validate never races
  rank 0.

Options (`--dp.<name>=<value>`, train_opts.py; defaults = plain behaviour):
  frame_cache=<dir>     frames come from frame_cache (uint8 decoded once, torch.equal to LeRobot's pyav decode — that
                        script's `check`), not from mp4 decode per item.
  init_from=<pretrained_model dir>  fine-tuning start: the new policy's whole state_dict — weights and the
  normalization
                        buffers (= the source run's stats) — is loaded strict from <dir>/model.safetensors right after
                        make_policy. The optimizer and the LR schedule are new (LeRobot builds them after make_policy);
                        the EMA shadow starts from the loaded weights at the first update. Mark [E5_DP_INIT] on … (keys
                        · three tensors · normalization buffers equal to the file); a missing/unexpected/shape key
                        dies.
  state_dropout=<p> · state_noise=<σ>  a forward hook on policy.normalize_inputs, active only while the policy is in
                        train mode, changes the NORMALIZED observation.state ([-1, 1] min-max) of each training batch:
                        each sample with probability p gets its whole state (all obs steps · all 56 dims) set to 0, the
                        rest get N(0, σ²) added. Eval / select_action (policy.eval()) and the saved weights are
                        untouched — a checkpoint loads and runs exactly like one trained without it. Marks
                        [E5_DP_STATE_AUG] on … and, after 200 training batches, [E5_DP_STATE_AUG] measured dropped
                        fraction … (must sit near p).
  compile · amp · ddp_comm · ddp_bucket_mb · ddp_bucket_view · fused_adamw · persistent_workers · step_sync ·
  columns:
                        speed and stability switches (train_opts.py).

--batch_size is per GPU; global batch = batch_size × WORLD_SIZE. Logged loss is rank 0's batch.
Marks: [E5_DP_DDP] on … · [E5_DP_DDP] sampler rank r first-batch sha … (must differ across ranks) · [E5_DP_DDP]
saved …. Run: torchrun --standalone --nproc_per_node=4 -m experiments.policies.dp.train_ddp <lerobot-train arguments>
[--dp.…] (or experiments.policies.dp.train, which builds the recipe arguments).
"""

from __future__ import annotations

import datetime
import hashlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from experiments.policies.dp import train_ema as EMA
from experiments.policies.dp import train_opts as TO

MARK = "[E5_DP_DDP]"
RANK, WORLD, LOCAL = (
    int(os.environ.get("RANK", 0)),
    int(os.environ.get("WORLD_SIZE", 1)),
    int(os.environ.get("LOCAL_RANK", 0)),
)
SEED = 42


class DistEpisodeSampler:
    """LeRobot EpisodeAwareSampler indices, one permutation per epoch shared by all ranks, rank r takes perm[r::W]."""

    def __init__(
        self,
        episode_data_index,
        episode_indices_to_use=None,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=False,
    ):
        from lerobot.datasets.sampler import EpisodeAwareSampler

        self.indices = EpisodeAwareSampler(
            episode_data_index, episode_indices_to_use, drop_n_first_frames, drop_n_last_frames, shuffle
        ).indices
        if not shuffle:
            raise SystemExit(f"{MARK} FAIL DistEpisodeSampler expects shuffle=True (LeRobot train)")
        self.per_rank = len(self.indices) // WORLD
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(SEED + self.epoch)
        perm = torch.randperm(len(self.indices), generator=g)[RANK::WORLD][: self.per_rank].tolist()
        if self.epoch == 0:
            sha = hashlib.sha256(str(perm[:64]).encode()).hexdigest()[:12]
            print(
                f"{MARK} sampler rank {RANK} first-batch sha {sha} · {self.per_rank} of {len(self.indices)} indices",
                flush=True,
            )
        self.epoch += 1
        return iter(self.indices[i] for i in perm)

    def __len__(self):
        return self.per_rank


def install_ddp(T) -> None:
    from lerobot.utils.random_utils import set_seed

    orig_update, orig_make = T.update_policy, T.make_policy
    wrapped = {}

    def make_policy(*a, **kw):
        p = orig_make(*a, **kw)
        if RANK > 0:
            set_seed(SEED + 1000 * RANK)
        return p

    def update_policy(train_metrics, policy, *a, **kw):
        if "ddp" not in wrapped:
            if TO.get("compile") == "1":  # nn.Module.compile keeps the state_dict keys
                # the EMA shadow already exists (train_ema's update_policy deep-copies before this one) — a copy of the
                # uncompiled modules
                policy.diffusion.rgb_encoder.compile()
                policy.diffusion.unet.compile()
                print(
                    f"{MARK} compile rank {RANK}: diffusion.rgb_encoder · diffusion.unet "
                    "(nn.Module.compile · keys unchanged)",
                    flush=True,
                )
            # bf16 = forward + loss under torch.autocast(bf16) · weights · grads · optimizer stay fp32 · default fp32
            amp = TO.get("amp")
            if amp == "bf16":
                fwd = policy.forward

                def forward_bf16(*fa, **fk):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return fwd(*fa, **fk)

                policy.forward = forward_bf16
            elif amp != "fp32":
                raise SystemExit(f"{MARK} FAIL dp.amp '{amp}' — fp32|bf16")
            print(f"{MARK} forward {amp} rank {RANK}", flush=True)
            bucket = int(TO.get("ddp_bucket_mb"))  # DDP default 25 MB
            ddp = torch.nn.parallel.DistributedDataParallel(
                policy,
                device_ids=[LOCAL],
                output_device=LOCAL,
                bucket_cap_mb=bucket,
                gradient_as_bucket_view=TO.get("ddp_bucket_view") == "1",
            )
            comm = TO.get("ddp_comm")  # bf16 = gradients all-reduced in bf16 (half the PCIe bytes) · default fp32
            if comm == "bf16":
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

                ddp.register_comm_hook(None, default_hooks.bf16_compress_hook)
            elif comm == "fp16":  # the fp16 variant of the same hook (narrower exponent, same bytes as bf16)
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

                ddp.register_comm_hook(None, default_hooks.fp16_compress_hook)
            elif comm != "fp32":
                raise SystemExit(f"{MARK} FAIL dp.ddp_comm '{comm}' — fp32|bf16|fp16")
            wrapped["ddp"] = ddp
            print(
                f"{MARK} DDP wrap rank {RANK} · params {sum(p.numel() for p in policy.parameters())} · "
                f"grad all-reduce {comm} · bucket {bucket} MB",
                flush=True,
            )
        wrapped["n"] = wrapped.get("n", 0) + 1
        if _step_sync():
            dist.barrier()
        return orig_update(train_metrics, wrapped["ddp"], *a, **kw)

    T.make_policy, T.update_policy, T.EpisodeAwareSampler = make_policy, update_policy, DistEpisodeSampler


def install_rank0_io(T) -> None:
    """After EMA.install: its save_checkpoint (EMA weights → pretrained_model) runs on rank 0 only."""
    ema_save, orig_last = T.save_checkpoint, T.update_last_checkpoint

    def save_checkpoint(checkpoint_dir, step, *a, **kw):
        if RANK == 0:
            ema_save(checkpoint_dir, step, *a, **kw)
            print(f"{MARK} saved step={step} world={WORLD} → {checkpoint_dir}", flush=True)
        dist.barrier()

    def update_last_checkpoint(checkpoint_dir):
        return orig_last(checkpoint_dir) if RANK == 0 else None

    T.save_checkpoint, T.update_last_checkpoint = save_checkpoint, update_last_checkpoint
    # ranks > 0: no WandBLogger · rank 0: dp.wandb guard + get_url shim (wandb ≥ 0.31 has only run.url) + [WANDB] on
    # …
    EMA.install_wandb(T, RANK)


def _step_sync() -> bool:
    return {"0": False, "1": True}[TO.get("step_sync")]  # an unknown value dies (KeyError)


def install_init(T) -> None:
    """dp.init_from (module docstring) — the outermost make_policy wrapper (after EMA.install, install_rank0_io)."""
    INIT = TO.get("init_from")
    if not INIT:
        return
    from safetensors.torch import load_file

    orig = T.make_policy

    def make_policy(*a, **kw):
        p = orig(*a, **kw)
        f = Path(INIT) / "model.safetensors"
        sd = load_file(str(f))
        own = p.state_dict()
        miss, unexp = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
        shape = [k for k in sd if k in own and tuple(sd[k].shape) != tuple(own[k].shape)]
        if miss or unexp or shape:
            raise SystemExit(f"[E5_DP_INIT] FAIL {f}: missing {miss[:3]} · unexpected {unexp[:3]} · shape {shape[:3]}")
        normk = sorted(k for k in sd if "normalize" in k)
        before = {k: own[k].detach().float().cpu().clone() for k in normk}
        p.load_state_dict(sd, strict=True)
        now = p.state_dict()
        ks = sorted(sd)
        probe = [ks[0], ks[len(ks) // 2], ks[-1]]

        def same(k):
            return torch.equal(now[k].detach().cpu(), sd[k].to(now[k].dtype))

        if not normk or not all(same(k) for k in probe + normk):
            raise SystemExit(f"[E5_DP_INIT] FAIL tensors differ from {f} after load (normalization keys {len(normk)})")
        d = max(float((before[k] - sd[k].float()).abs().max()) for k in normk)
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        print(
            f"[E5_DP_INIT] on {f} sha {h} · keys {len(sd)} strict (missing 0 · unexpected 0) · "
            f"tensors {probe} = file · normalization buffers {len(normk)} = file (sim stats · "
            f"max |this dataset − sim| {d:.4g}) · optimizer · LR schedule new · rank {RANK}",
            flush=True,
        )
        return p

    T.make_policy = make_policy


def _state_aug_cfg() -> tuple[float, float] | None:
    ps, ss = TO.get("state_dropout"), TO.get("state_noise")
    if not ps and not ss:
        return None
    try:
        p, s = float(ps or 0), float(ss or 0)
    except ValueError:
        raise SystemExit(f"[E5_DP_STATE_AUG] FAIL dp.state_dropout '{ps}' · dp.state_noise '{ss}' — numbers") from None
    if not (0.0 <= p < 1.0 and 0.0 <= s <= 0.5) or (p == 0 and s == 0):
        raise SystemExit(f"[E5_DP_STATE_AUG] FAIL p {p} (0 ≤ p < 1) · σ {s} (0 ≤ σ ≤ 0.5) · not both 0")
    return p, s


def install_state_aug(T) -> None:
    """dp.state_dropout · dp.state_noise (module docstring) — outermost make_policy wrapper (after install_init)."""
    cfg = _state_aug_cfg()
    if cfg is None:
        return
    pd, sg = cfg
    orig = T.make_policy

    def make_policy(*a, **kw):
        pol = orig(*a, **kw)
        key = "observation.state"
        stat = {"n": 0, "drop": 0, "said": False}

        def hook(mod, inp, out):
            if not mod.training or key not in out:
                return out
            x = out[key]
            out = dict(out)
            keep = (torch.rand(x.shape[0], device=x.device) >= pd).to(x.dtype).view(-1, *([1] * (x.dim() - 1)))
            y = x + sg * torch.randn_like(x) if sg > 0 else x
            out[key] = y * keep
            stat["n"] += x.shape[0]
            stat["drop"] += int((keep == 0).sum())
            if not stat["said"] and stat["n"] >= 200 * x.shape[0]:
                stat["said"] = True
                print(
                    f"[E5_DP_STATE_AUG] measured dropped fraction {stat['drop'] / stat['n']:.3f} (p {pd}) over "
                    f"{stat['n']} samples · state shape {tuple(x.shape)} · rank {RANK}",
                    flush=True,
                )
            return out

        pol.normalize_inputs.register_forward_hook(hook)
        print(
            f"[E5_DP_STATE_AUG] on p {pd} · σ {sg} · key {key} (normalized [-1, 1]) · hook on "
            "normalize_inputs · train mode only · "
            f"rank {RANK}",
            flush=True,
        )
        return pol

    T.make_policy = make_policy


def _rank_argv(argv: list[str]) -> list[str]:
    if RANK == 0:
        return argv
    return [f"{x}.rank{RANK}" if x.startswith("--output_dir=") else x for x in argv]


def _check_launch():
    """Refuse a launch that cannot run: one GPU, no --output_dir, copy-engine SHM with lazy CUDA module loading."""
    if WORLD < 2:
        raise SystemExit(f"{MARK} FAIL WORLD_SIZE {WORLD} — run with torchrun --nproc_per_node ≥ 2 (1 GPU = train_ema)")
    if not any(x.startswith("--output_dir=") for x in sys.argv[1:]):
        raise SystemExit(f"{MARK} FAIL --output_dir=<dir> is required (ranks > 0 get <dir>.rank<r>)")
    # NCCL SHM by copy engine (NCCL_SHM_USE_CUDA_MEMCPY=1 · L40S cards without P2P: bf16 0.54 GB all-reduce 293 → 104
    # ms) makes NCCL progress depend on the proxy thread's cudaMemcpyAsync; with CUDA lazy module loading the first
    # launch of a new kernel takes cudart's lock and waits on the GPU while the collective waits on the proxy →
    # first-step deadlock (py-spy: main thread in cudaLaunchKernel → cuLibraryGetModule · pin-memory thread on the
    # same mutex in cudaHostAlloc). EAGER loads every module at init.
    if os.environ.get("NCCL_SHM_USE_CUDA_MEMCPY", "0") == "1" and os.environ.get("CUDA_MODULE_LOADING") != "EAGER":
        raise SystemExit(
            f"{MARK} FAIL NCCL_SHM_USE_CUDA_MEMCPY=1 needs CUDA_MODULE_LOADING=EAGER (lazy loading "
            "deadlocks the copy-engine "
            f"SHM proxy on the first step) — got {os.environ.get('CUDA_MODULE_LOADING')!r}"
        )


def _install_step_sync():
    """step_sync=1: drain the stream before gradient clipping."""
    # step_sync=1: no queue of hundreds of clip/Adam kernels after the gradient exchange. The NCCL future completes on
    # the CPU at once and the default stream waits for the exchange on the GPU; a full launch queue then spins the main
    # thread in cuLaunchKernel holding the CUDA lock, the copy-engine proxy cannot take it and the exchange never ends.
    # So each step starts with a barrier and the stream is drained before clipping.
    if _step_sync():
        _clip = torch.nn.utils.clip_grad_norm_

        def clip_after_sync(*a, **k):
            torch.cuda.current_stream().synchronize()
            return _clip(*a, **k)

        torch.nn.utils.clip_grad_norm_ = clip_after_sync  # lerobot train calls torch.nn.utils.clip_grad_norm_(...)
        print(f"{MARK} step sync on rank {RANK}", flush=True)


def _install_persistent_workers():
    """persistent_workers=1: DataLoader workers survive epoch boundaries; 0 leaves torch as is."""
    # persistent_workers=1: data workers are not restarted every epoch. With copy-engine SHM (_check_launch) one rank
    # starting its workers while the others ran ahead deadlocked at an epoch boundary.
    pw = TO.get("persistent_workers")
    if pw == "1":
        _dl = torch.utils.data.DataLoader

        class PersistentDataLoader(_dl):
            def __init__(self, *a, **k):
                k["persistent_workers"] = True  # with num_workers 0 torch raises ValueError
                super().__init__(*a, **k)

        torch.utils.data.DataLoader = PersistentDataLoader  # lerobot train calls torch.utils.data.DataLoader(...)
        print(f"{MARK} persistent workers on rank {RANK}", flush=True)
    elif pw != "0":
        raise SystemExit(f"{MARK} FAIL dp.persistent_workers '{pw}' — 0|1")


def _install_data_caches():
    """frame_cache=<dir> and columns=1: frames and parquet columns from numpy caches; returns the frame-cache dir."""
    fc = TO.get("frame_cache")
    if fc:
        from experiments.policies.dp import frame_cache as FC

        root = next(x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--dataset.root="))
        c = FC.install(Path(fc), Path(root))
        print(f"{MARK} frame cache on {fc} · frames {c.n} · keys {len(c.keys)}", flush=True)
    if (
        TO.get("columns") == "1"
    ):  # parquet columns from numpy (no Arrow select per item · frame_cache check --columns proves equality)
        from experiments.policies.dp import frame_cache as FC

        FC.install_columns()
        print(f"{MARK} columns from numpy (install_columns) rank {RANK}", flush=True)
    return fc


def _install_fused_adamw():
    """fused_adamw=1: the same AdamW update in one fused kernel."""
    if TO.get("fused_adamw") == "1":  # the same AdamW update in one fused kernel
        _adamw = torch.optim.AdamW

        class FusedAdamW(_adamw):
            def __init__(self, *a, **k):
                k.setdefault("fused", True)
                super().__init__(*a, **k)

        torch.optim.AdamW = FusedAdamW  # LeRobot AdamWConfig.build calls torch.optim.AdamW(params, **kwargs)
        print(f"{MARK} AdamW fused=True rank {RANK}", flush=True)


def _quiet_rank_logging(T):
    """Ranks > 0 log warnings only."""
    if RANK > 0:
        orig_init = T.init_logging

        def init_logging(*a, **kw):
            orig_init(*a, **kw)
            import logging

            logging.getLogger().setLevel(logging.WARNING)

        T.init_logging = init_logging


def _print_launch(fc):
    """Print the launch marker lines (world, batch, NCCL, EMA)."""
    bs = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--batch_size=")), "?")
    print(
        f"{MARK} on world={WORLD} rank={RANK} local={LOCAL} per-GPU batch {bs} → global "
        f"{int(bs) * WORLD if bs.isdigit() else '?'} · "
        f"nccl {torch.cuda.nccl.version()} · frame cache {'on' if fc else 'off'} · "
        f"NCCL_SHM_USE_CUDA_MEMCPY={os.environ.get('NCCL_SHM_USE_CUDA_MEMCPY', '-')} "
        f"CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING', '-')}",
        flush=True,
    )
    print(
        f"{EMA.MARK} on power={EMA.POWER} max={EMA.MAX_VALUE} inv_gamma={EMA.INV_GAMMA} min={EMA.MIN_VALUE} "
        f"update_after_step={EMA.UPDATE_AFTER_STEP} (Chi diffusion_policy EMAModel) · rank {RANK}",
        flush=True,
    )


def main() -> None:
    sys.argv = [sys.argv[0]] + TO.take(sys.argv[1:])
    _check_launch()
    torch.cuda.set_device(LOCAL)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    import lerobot.scripts.train as T

    _install_step_sync()
    _install_persistent_workers()
    fc = _install_data_caches()
    _install_fused_adamw()
    install_ddp(T)
    EMA.install(T)
    install_rank0_io(T)
    install_init(T)
    install_state_aug(T)
    _quiet_rank_logging(T)
    sys.argv = [sys.argv[0]] + _rank_argv(sys.argv[1:])
    _print_launch(fc)
    try:
        T.main()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
