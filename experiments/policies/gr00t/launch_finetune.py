"""Isaac-GR00T launch_finetune.py with the training knobs its CLI does not expose (run from the Isaac-GR00T cwd).

Knobs are `--ft.<name>=<value>` among the launch_finetune arguments (FT_OPTS below); unset = Isaac-GR00T's own
behaviour. Each applied knob prints a mark read from the built consumer where possible; a run without the mark did not
apply it.
  deepspeed_stage · grad_ckpt   config.training.deepspeed_stage / gradient_checkpointing, set on the config
                                launch_finetune builds, right before its run(config); mark [E5_TRAIN_CFG].
  data_time=1                   (measurement) times transformers.Trainer.get_batch_samples (the loop's only data fetch,
                                transformers 4.57) against the wall time of each optimizer step; every 10 steps on rank
                                0:
                                [E5_DATA_T] steps=… data_s=… wall_s=… data_frac=… (≈ 0 → the GPU side bounds the
                                step).
  tail_starts=1                 every frame is a training start (0 … L−1, LeRobot's standard form), not only 0 … L−H
                                (Isaac-GR00T N1.7 get_effective_episode_length = L − H + 1 whatever allow_padding
                                says). Chunk indices past the end clamp to the last row (config.data.allow_padding=True
                                → the last recorded target repeated = 'hold the final command'); the loss is not
                                masked there (LeRobot diffusion default). Without it the last H−1 frames of every
                                episode are never an input. Mark on rank 0: [E5_TAIL_STARTS] on starts=<n> padded=<n>.
  gr00t_minmax=1                config.model.use_percentiles = False — the processor normalises state/action by min/max
                                instead of q01/q99. With clip_outliers (on) the percentile form clamps every value
                                outside q01–q99, so labels beyond them are not learnable.
                                Gr00tN1d7Processor.from_pretrained (the
                                start_from_checkpoint path) overrides only a fixed key list without use_percentiles, so
                                the knob forces use_percentiles=False in Gr00tN1d7Processor.__init__ (both paths build
                                through it); the mark [E5_MINMAX] is read from the built processor and its
                                state_action_processor — anything but False dies.
  extra_save_steps · save_at_epoch · adam_beta2 · log_nan_filter · adam_beta2_force · adam_probe · stop_at_step ·
  lr_scale · wandb · wandb_project · wandb_run   see the comments at their code.
"""

import os
import runpy
import sys
import time

import gr00t.experiment.experiment as X

#: Launcher options: `--ft.<name>=<value>` among the launch_finetune arguments, removed from sys.argv before
#: Isaac-GR00T parses it. Values are strings; unset = Isaac-GR00T's own behaviour. An unknown name dies.
FT_OPTS = dict.fromkeys(
    [
        "adam_beta2",
        "adam_beta2_force",
        "adam_probe",
        "data_time",
        "deepspeed_stage",
        "extra_save_steps",
        "gr00t_minmax",
        "grad_ckpt",
        "log_nan_filter",
        "lr_scale",
        "save_at_epoch",
        "stop_at_step",
        "tail_starts",
        "wandb",
        "wandb_project",
        "wandb_run",
    ],
    "",
)
_argv = []
for _x in sys.argv[1:]:
    if _x.startswith("--ft."):
        _k, _sep, _v = _x[5:].partition("=")
        if not _sep or _k not in FT_OPTS:
            raise SystemExit(f"[E5_FT] FAIL {_x!r} — --ft.<name>=<value>, names {sorted(FT_OPTS)}")
        FT_OPTS[_k] = _v
    else:
        _argv.append(_x)
sys.argv = [sys.argv[0]] + _argv


def _ft(name: str, default=""):
    """Launcher option `name` (`default` when not given)."""
    return FT_OPTS[name] or default


def _ft_req(name: str) -> str:
    """A launcher option that must be given."""
    if not FT_OPTS[name]:
        raise SystemExit(f"[E5_FT] FAIL --ft.{name} is required here")
    return FT_OPTS[name]


if _ft("data_time", None) == "1":
    import transformers.trainer as _T

    _gbs = _T.Trainer.get_batch_samples
    _st = {"n": 0, "data": 0.0, "t0": None, "last": None, "wall": 0.0}
    _rank0 = os.environ.get("RANK", "0") == "0"

    def _get_batch_samples(self, *args, **kwargs):
        a = time.perf_counter()
        if _st["last"] is not None:
            _st["wall"] += a - _st["last"]
        out = _gbs(self, *args, **kwargs)
        b = time.perf_counter()
        _st["n"] += 1
        if _st["n"] > 2:  # the first two fetches include worker start-up
            _st["data"] += b - a
        else:
            _st["wall"] = 0.0
        _st["last"] = a
        k = _st["n"] - 2
        if _rank0 and k > 0 and k % 10 == 0:
            print(
                f"[E5_DATA_T] steps={k} data_s={_st['data']:.1f} wall_s={_st['wall']:.1f} "
                f"data_frac={_st['data'] / max(_st['wall'], 1e-9):.2f}",
                flush=True,
            )
        return out

    _T.Trainer.get_batch_samples = _get_batch_samples

TAIL = _ft("tail_starts", "")
if TAIL not in ("", "0", "1"):
    raise ValueError(f"ft.tail_starts={TAIL!r} — 0 or 1")
if TAIL == "1":
    from gr00t.data.dataset import sharded_single_step_dataset as _SSD

    _C = _SSD.ShardedSingleStepDataset
    _shard = _C.shard_dataset

    def _effective_length(self, episode_index: int) -> int:
        if not self.allow_padding:  # starts past L − H need the clamp, else extract_step_data reads past the episode
            raise RuntimeError("ft.tail_starts=1 with allow_padding=False — the wrapper sets config.data.allow_padding")
        return self.episode_loader.get_episode_length(episode_index)

    def _shard_dataset(self):
        _shard(self)
        lens = [int(self.episode_loader.get_episode_length(i)) for i in range(len(self.episode_loader.episode_lengths))]
        padded = sum(min(n, self.action_horizon - 1) for n in lens)
        if int(sum(self.shard_lengths)) != sum(lens):
            raise RuntimeError(f"ft.tail_starts: {int(sum(self.shard_lengths))} starts ≠ {sum(lens)} frames")
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[E5_TAIL_STARTS] on starts={int(sum(self.shard_lengths))} padded={padded} episodes={len(lens)} "
                f"horizon={self.action_horizon} dataset={self.episode_loader.dataset_path}",
                flush=True,
            )

    _C.get_effective_episode_length = _effective_length
    _C.shard_dataset = _shard_dataset

_run = X.run

# ft.extra_save_steps=<n>[,<n>…] (keep chosen steps while SAVE_STEPS is a short resume interval):
# DefaultFlowCallback also asks for a save at these global steps. Mark on rank 0 at each: [E5_EXTRA_SAVE] step=<n>.
# Unset = transformers' own save schedule, unchanged.
EXTRA_SAVE = [int(x) for x in _ft("extra_save_steps", "").split(",") if x]
# ft.save_at_epoch=<unix s> (a job time limit): one extra full save (optimizer · scheduler · rng · step) at the
# first step boundary after that wall time, so a continuation job resumes from ≤ 100 steps before the cut. The decision
# is made every 100 steps from rank 0's clock and broadcast (all ranks must enter the save collectively). Mark:
# [E5_SAVE_AT] step=<n>.
SAVE_AT = _ft("save_at_epoch", "")
if SAVE_AT and not SAVE_AT.isdigit():
    raise ValueError(f"ft.save_at_epoch={SAVE_AT!r} — unix seconds")
if EXTRA_SAVE or SAVE_AT:
    from transformers import trainer_callback as _TC

    _ose = _TC.DefaultFlowCallback.on_step_end
    _save_at = {"done": False}

    def _on_step_end(self, args, state, control, **kwargs):
        control = _ose(self, args, state, control, **kwargs)
        r0 = os.environ.get("RANK", "0") == "0"
        if state.global_step in EXTRA_SAVE:
            control.should_save = True
            if r0:
                print(f"[E5_EXTRA_SAVE] step={state.global_step}", flush=True)
        if SAVE_AT and not _save_at["done"] and state.global_step % 100 == 0:
            import torch
            import torch.distributed as dist

            f = torch.tensor([1.0 if time.time() >= int(SAVE_AT) else 0.0], device="cuda")
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(f, 0)
            if f.item() > 0:
                _save_at["done"] = True
                control.should_save = True
                if r0:
                    print(f"[E5_SAVE_AT] step={state.global_step} wall={int(time.time())} at>={SAVE_AT}", flush=True)
        return control

    _TC.DefaultFlowCallback.on_step_end = _on_step_end

# ft.adam_beta2=<0<b<1> · ft.log_nan_filter=0|1: Isaac-GR00T leaves TrainingArguments' AdamW β2 at 0.999 (with
# global batch 256 at peak lr the loss jumped back to its step-0 value in 3/3 runs) and HF's logging_nan_inf_filter on
# (a nan step is logged as the window mean · an all-nan window as 0.0, which hides a dead run). Both are set on the
# trainer's args right after Trainer.__init__; the optimizer is built later from args (the deepspeed config has no
# optimizer block). Applied mark on rank 0 at on_train_begin, read from the built optimizer (walks DeepSpeed's
# .optimizer wrappers):
# [E5_OPT] betas=(b1, b2) eps=… weight_decay=… nan_filter=… — a set value the built optimizer/args do not carry
# dies. Unset = Isaac-GR00T's defaults, unchanged. Branch test knobs: two runs resumed from one checkpoint that differ
# only in β2. A resume restores the checkpoint's param_groups (β2 0.999), so
#   ft.adam_beta2_force=1 — after the load, every param group of the innermost optimizer gets betas (b1,
# ft.adam_beta2), then the
#                           check above runs. Mark: [E5_OPT] betas forced after load: <old> → <new>. Needs
# ft.adam_beta2.
#   ft.adam_probe=<n>     — every n steps rank 0 prints [E5_ADAM] step= lr= rms_u= max_u= frac_gt1= n= from its own
# ZeRO
#                           partition (u = AdamW's per-element step / lr, bias-corrected, exactly as torch AdamW forms
# it).
#   ft.stop_at_step=<n>   — training ends at global step n (the lr schedule still spans max_steps). Mark: [E5_STOP_AT]
# step=<n>.
#   ft.lr_scale=<k>       — stress test: after the load, the scheduler's base_lrs and every param group's
#                           current lr are multiplied by k, so the run continues on k × the same schedule.
#                           Mark: [E5_LR] scale=k base_lrs <old> → <new> · group lr <old> → <new> at step <n>.
B2 = _ft("adam_beta2", "")
NANF = _ft("log_nan_filter", "")
B2F = _ft("adam_beta2_force", "")
PROBE = _ft("adam_probe", "")
STOP_AT = _ft("stop_at_step", "")
LRS = _ft("lr_scale", "")
if NANF not in ("", "0", "1"):
    raise ValueError(f"ft.log_nan_filter={NANF!r} — 0 or 1")
if B2 and not 0.0 < float(B2) < 1.0:
    raise ValueError(f"ft.adam_beta2={B2!r} — 0 < b < 1")
if B2F not in ("", "0", "1") or (B2F == "1" and not B2):
    raise ValueError(f"ft.adam_beta2_force={B2F!r} — 0 or 1, and 1 needs ft.adam_beta2")
try:
    _lrs_ok = not LRS or 0.0 < float(LRS) <= 10.0
except ValueError:
    _lrs_ok = False
if not _lrs_ok:
    raise ValueError(f"ft.lr_scale={LRS!r} — a number in (0, 10]")
for _k, _v in (("ft.adam_probe", PROBE), ("ft.stop_at_step", STOP_AT)):
    if _v and not (_v.isdigit() and int(_v) > 0):
        raise ValueError(f"{_k}={_v!r} — a positive integer")


def _inner_opt(opt):
    """Innermost optimizer whose param_groups carry betas (accelerate wrapper → DeepSpeed ZeRO → torch AdamW)."""
    found = None
    while opt is not None:
        if getattr(opt, "param_groups", None) and "betas" in opt.param_groups[0]:
            found = opt
        opt = getattr(opt, "optimizer", None)
    return found


def _inner_sched(sch):
    """Innermost lr scheduler that holds base_lrs (accelerate AcceleratedScheduler → torch LambdaLR)."""
    found = None
    while sch is not None:
        if getattr(sch, "base_lrs", None) is not None:
            found = sch
        sch = getattr(sch, "scheduler", None)
    return found


def _force_betas(opt, state):
    """ft.adam_beta2_force=1: set beta2 of every param group after the optimizer state is loaded."""
    if B2F == "1":
        old = [tuple(float(x) for x in pg["betas"]) for pg in opt.param_groups]
        for pg in opt.param_groups:
            pg["betas"] = (float(pg["betas"][0]), float(B2))
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[E5_OPT] betas forced after load: {old} → {[tuple(pg['betas']) for pg in opt.param_groups]} "
                f"at step {state.global_step}",
                flush=True,
            )


def _scale_lr(kwargs, opt, state):
    """ft.lr_scale=k: scale the scheduler base_lrs, the group lrs and the last lr of the training optimizer by k."""
    if LRS:
        sch = _inner_sched(kwargs.get("lr_scheduler"))
        if sch is None:
            raise RuntimeError("[E5_LR] no lr scheduler with base_lrs reachable from on_train_begin")
        k = float(LRS)
        so = getattr(sch, "optimizer", None)
        if (
            so is None
            or len(so.param_groups) != len(opt.param_groups)
            or any(x is not y for x, y in zip(so.param_groups, opt.param_groups, strict=True))
        ):
            raise RuntimeError("[E5_LR] the scheduler's optimizer groups are not the training optimizer's groups")
        ob, og = list(sch.base_lrs), [pg["lr"] for pg in opt.param_groups]
        sch.base_lrs = [b * k for b in sch.base_lrs]
        for pg in opt.param_groups:
            pg["lr"] = pg["lr"] * k
        if getattr(sch, "_last_lr", None) is not None:
            sch._last_lr = [x * k for x in sch._last_lr]
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[E5_LR] scale={k} base_lrs {ob} → {sch.base_lrs} · group lr {og} → "
                f"{[pg['lr'] for pg in opt.param_groups]} at step {state.global_step}",
                flush=True,
            )


if B2 or NANF or B2F or PROBE or STOP_AT or LRS:
    import transformers as _TF
    from transformers import trainer_callback as _TCB

    _tr_init = _TF.Trainer.__init__

    def _trainer_init(self, *a, **kw):
        _tr_init(self, *a, **kw)
        if B2:
            self.args.adam_beta2 = float(B2)
        if NANF:
            self.args.logging_nan_inf_filter = NANF == "1"

    _TF.Trainer.__init__ = _trainer_init
    _otb = _TCB.DefaultFlowCallback.on_train_begin

    def _on_train_begin(self, args, state, control, **kwargs):
        opt = _inner_opt(kwargs.get("optimizer"))
        if opt is None:
            raise RuntimeError("[E5_OPT] no optimizer with betas reachable from on_train_begin")
        g = opt.param_groups[0]
        _force_betas(opt, state)
        bad = [tuple(pg["betas"]) for pg in opt.param_groups if B2 and abs(float(pg["betas"][1]) - float(B2)) > 1e-12]
        if bad:
            raise RuntimeError(f"[E5_OPT] param groups with betas {bad} ≠ ft.adam_beta2={B2}")
        betas = tuple(float(x) for x in g["betas"])
        if B2 and abs(betas[1] - float(B2)) > 1e-12:
            raise RuntimeError(f"[E5_OPT] built optimizer betas {betas} ≠ ft.adam_beta2={B2}")
        if NANF and args.logging_nan_inf_filter != (NANF == "1"):
            raise RuntimeError(
                f"[E5_OPT] logging_nan_inf_filter={args.logging_nan_inf_filter} ≠ ft.log_nan_filter={NANF}"
            )
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[E5_OPT] betas={betas} eps={g.get('eps')} weight_decay={g.get('weight_decay')} "
                f"groups={len(opt.param_groups)} nan_filter={args.logging_nan_inf_filter} "
                f"optimizer={type(opt).__module__}.{type(opt).__name__}",
                flush=True,
            )
        _scale_lr(kwargs, opt, state)
        if STOP_AT and os.environ.get("RANK", "0") == "0":
            print(f"[E5_STOP_AT] step={STOP_AT} (from {state.global_step} · max_steps {args.max_steps})", flush=True)
        return _otb(self, args, state, control, **kwargs) if _otb else control

    _TCB.DefaultFlowCallback.on_train_begin = _on_train_begin

    if PROBE or STOP_AT:
        _ose2 = _TCB.DefaultFlowCallback.on_step_end

        def _on_step_end2(self, args, state, control, **kwargs):
            control = _ose2(self, args, state, control, **kwargs)
            if PROBE and state.global_step % int(PROBE) == 0 and os.environ.get("RANK", "0") == "0":
                import torch

                opt = _inner_opt(kwargs.get("optimizer"))
                n = big = 0
                ss = mx = 0.0
                lr = None
                with torch.no_grad():
                    for pg in opt.param_groups:
                        b1, b2 = pg["betas"]
                        lr = pg["lr"]
                        for p in pg["params"]:
                            st = opt.state.get(p, {})
                            if "exp_avg" not in st:
                                continue
                            t = float(st["step"])
                            bc1, bc2 = 1 - b1**t, 1 - b2**t
                            m, v = st["exp_avg"].view(-1), st["exp_avg_sq"].view(-1)
                            ch = 1 << 24  # 16 M elements at a time (64 MB fp32 temporaries)
                            for i in range(0, m.numel(), ch):
                                mi, vi = m[i : i + ch].float(), v[i : i + ch].float()
                                u = (mi / bc1) / (vi.sqrt() / bc2**0.5 + pg["eps"])
                                ss += float((u * u).sum())
                                mx = max(mx, float(u.abs().max()))
                                big += int((u.abs() > 1).sum())
                                n += u.numel()
                rms = (ss / max(n, 1)) ** 0.5
                print(
                    f"[E5_ADAM] step={state.global_step} lr={lr:.3e} rms_u={rms:.4f} max_u={mx:.3f} "
                    f"frac_gt1={big / max(n, 1):.3e} n={n} betas=({b1}, {b2})",
                    flush=True,
                )
            if STOP_AT and state.global_step >= int(STOP_AT):
                control.should_training_stop = True
                if os.environ.get("RANK", "0") == "0":
                    print(f"[E5_STOP_AT] stopping at step={state.global_step}", flush=True)
            return control

        _TCB.DefaultFlowCallback.on_step_end = _on_step_end2

# ft.wandb=on: config.training.use_wandb=True · project ft.wandb_project · entity = env
# WANDB_ENTITY (wandb.init reads it) · run name ft.wandb_run (replaces GR00T's experiment_name). Key = WANDB_API_KEY in
# the env (never printed). Applied mark on rank 0, from the live run object: [WANDB] on project=… run=… url=… — no
# mark
# = void. Unset / off = Isaac-GR00T's default (no wandb), unchanged.
WB = _ft("wandb", "")
if WB not in ("", "on", "off"):
    raise ValueError(f"ft.wandb={WB!r} — on or off")
if WB == "on":
    import wandb as _wb

    _wb_init = _wb.init

    def _wandb_init(*a, **kw):
        kw["name"] = _ft_req("wandb_run")
        r = _wb_init(*a, **kw)
        print(f"[WANDB] on project={r.project} entity={r.entity} run={r.name} id={r.id} url={r.url}", flush=True)
        return r

    _wb.init = _wandb_init
    X.wandb.init = _wandb_init


def run(config):
    if WB == "on":
        config.training.use_wandb = True
        config.training.wandb_project = _ft_req("wandb_project")
    # These two marks print the config value, never the
    # built consumer's (the deepspeed config the trainer received · the built model's gradient_checkpointing), so a
    # value other than Isaac-GR00T's default (stage 2 · off) dies until its mark is read from the consumer.
    st = _ft("deepspeed_stage", "")
    if st:
        if st != "2":
            raise ValueError(f"ft.deepspeed_stage={st!r} — only the default 2 (the mark is not read from the trainer)")
        config.training.deepspeed_stage = int(st)
    gc = _ft("grad_ckpt", "")
    if gc:
        if gc != "0":
            raise ValueError(f"ft.grad_ckpt={gc!r} — only the default 0 (the mark is not read from the built model)")
        config.training.gradient_checkpointing = False
    # extra_augmentation_config reaches the processor only on the direct path — from_pretrained's override list lacks
    # it (same hole as use_percentiles), so E5 never sets it
    if getattr(config.model, "extra_augmentation_config", None) is not None:
        raise ValueError("config.model.extra_augmentation_config is set — from_pretrained would drop it silently")
    if TAIL == "1":
        config.data.allow_padding = True
    mm = _ft("gr00t_minmax", "")
    if mm not in ("", "0", "1"):
        raise ValueError(f"ft.gr00t_minmax={mm!r} — 0 or 1")
    if mm == "1":
        config.model.use_percentiles = False
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor as _P

        _init = _P.__init__

        def _init_minmax(self, *a, **kw):
            kw["use_percentiles"] = False
            _init(self, *a, **kw)
            sap = getattr(self, "state_action_processor", None)
            got = (getattr(self, "use_percentiles", None), getattr(sap, "use_percentiles", None))
            if got != (False, False):
                raise RuntimeError(f"[E5_MINMAX] processor use_percentiles {got} after forcing False")
            print(f"[E5_MINMAX] processor use_percentiles={got[0]} state_action_processor={got[1]}", flush=True)

        _P.__init__ = _init_minmax
    t = config.training
    print(
        f"[E5_TRAIN_CFG] deepspeed_stage={t.deepspeed_stage} gradient_checkpointing={t.gradient_checkpointing} "
        f"tune_visual={config.model.tune_visual} global_batch={t.global_batch_size} "
        f"grad_accum={t.gradient_accumulation_steps} num_gpus={t.num_gpus} "
        f"data_time={int(_ft('data_time', None) == '1')} workers={t.dataloader_num_workers} "
        f"tail_starts={int(TAIL == '1')} allow_padding={config.data.allow_padding} "
        f"use_percentiles={config.model.use_percentiles}",
        flush=True,
    )
    return _run(config)


X.run = run
runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
