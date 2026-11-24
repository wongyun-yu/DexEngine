"""Model side of the DP policy server: loading (with an optional sampler override), the training-stats check, the
constant-state normalisation fix and the oracle label source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GROUPS = (("left_arm_joints", 0, 7), ("right_arm_joints", 7, 14), ("head_joints", 14, 16), ("hands_joints", 16, 56))
CAMS = ("camera_ego_left", "camera_ego_right")
LANG = "human.task_description"
SCHEMA = "e5_policy_modality/1"  # the session client's schema (experiments.sim_eval.policy_modality.SCHEMA)
SAMPLERS = ("DDIM", "DDPM")


def stats_check(policy, root: str, ckpt: str) -> str:
    """Training stats = inference stats: every normalisation buffer kept in the weights (normalize_inputs,
    unnormalize_outputs) must equal what training normalised with, bit for bit as float32 (sha256 over the concatenated
    float32 bytes). That is `--root`'s meta/stats.json, except the camera mean/std when the run's train_config.json says
    dataset.use_imagenet_stats (LeRobot then overwrites them with IMAGENET_STATS). Unequal or missing → SystemExit: the
    root is not the training dataset (its task → axis table could differ too)."""
    st = json.loads(Path(f"{root}/meta/stats.json").read_text())
    with open(f"{ckpt}/train_config.json") as fh:
        tc = json.load(fh)
    imagenet = bool(tc["dataset"]["use_imagenet_stats"])
    if imagenet:
        from lerobot.datasets.factory import IMAGENET_STATS
    cams = set(policy.config.image_features)
    keys = set(policy.config.input_features) | set(policy.config.output_features)
    bufs, ref = [], []
    for mod in (policy.normalize_inputs, policy.unnormalize_outputs):
        for name, t in sorted(list(mod.named_parameters()) + list(mod.named_buffers())):
            head, stat = name.split(".", 1)
            key = next((k for k in keys if "buffer_" + k.replace(".", "_") == head), None)
            if key is None:
                raise SystemExit(f"normalisation buffer {name} matches no feature of the checkpoint")
            bufs.append(t.detach().float().cpu().numpy().ravel())
            if imagenet and key in cams and stat in ("mean", "std"):
                ref.append(np.asarray(IMAGENET_STATS[stat], dtype=np.float32).ravel())
                continue
            if key not in st or stat not in st[key]:
                raise SystemExit(f"{root}/meta/stats.json has no {key}.{stat} (checkpoint buffer {name})")
            ref.append(np.asarray(st[key][stat], dtype=np.float32).ravel())
    hb = hashlib.sha256(b"".join(x.astype(np.float32).tobytes() for x in bufs)).hexdigest()
    hr = hashlib.sha256(b"".join(x.tobytes() for x in ref)).hexdigest()
    ok = hb == hr and len(bufs) > 0
    print(
        f"[E5_STATS_CHECK] dp {'OK' if ok else 'FAIL'} buffers={len(bufs)} train(weights)={hb[:12]} "
        f"root_stats={hr[:12]} root={root} cameras={'imagenet (use_imagenet_stats)' if imagenet else 'stats.json'}",
        flush=True,
    )
    if not ok:
        d = max(
            (float(np.max(np.abs(x - y))) for x, y in zip(bufs, ref, strict=True) if x.shape == y.shape),
            default=float("nan"),
        )
        raise SystemExit(f"training stats (checkpoint buffers) ≠ {root}/meta/stats.json (max |diff| {d})")
    return hb


def const_state_dims(policy, ckpt: str, eps: float):
    """Constant-state fix: state columns whose float32 (max − min) in the checkpoint's own normalize_inputs buffers is
    ≤ eps → (indices, their min, marker line).

    LeRobot MIN_MAX normalisation is (x − min)/(max − min + 1e-8): a column that was constant in training (kinematic
    data: a head joint) turns a float-noise difference at eval into values of 1e2…1e5. Training always saw −1 there,
    which is what feeding the min gives. MIN_MAX state normalisation only. `eps` > 0 also catches columns whose width is
    float noise rather than exactly 0 (e.g. 8.34e-07)."""
    mode = str(policy.config.normalization_mapping.get("STATE", "")).split(".")[-1]
    if mode != "MIN_MAX":
        raise SystemExit(f"--const-state-fix needs MIN_MAX state normalisation (checkpoint: {mode})")
    if not (0.0 <= eps < 1e-4):
        raise SystemExit(f"--const-state-eps {eps} — 0 ≤ width < 1e-4 only")
    buf = dict(policy.normalize_inputs.named_buffers())
    buf.update(dict(policy.normalize_inputs.named_parameters()))
    mn = buf["buffer_observation_state.min"].detach().float().cpu().numpy().ravel()
    mx = buf["buffer_observation_state.max"].detach().float().cpu().numpy().ravel()
    dims = np.where((mx - mn) <= eps)[0]
    names = [f"{g}[{i - a}]" for i in dims for g, a, b in GROUPS if a <= i < b]
    parts = ckpt.rstrip("/").split("/")
    tag = (
        f"{parts[-5]}@{parts[-2]}"
        if len(parts) >= 5 and parts[-1] == "pretrained_model" and parts[-3] == "checkpoints"
        else ckpt
    )
    wd = ",".join(f"{float(mx[i] - mn[i]):.3g}" for i in dims)
    return (
        dims,
        mn[dims].astype(np.float32),
        (f"[DP_CONST_STATE] dims={','.join(names) or '-'} ckpt={tag} eps={eps:g} widths={wd or '-'}"),
    )


def load_policy(ckpt: str, oracle_map: str, dev, sampler: tuple[str, int] | None):
    """The DiffusionPolicy and its config; `sampler` = (scheduler, steps) overrides the checkpoint's sampler at load
    (same weights: only noise_scheduler_type and num_inference_steps change)."""
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    if sampler is None:
        pol = DiffusionPolicy.from_pretrained(ckpt).to(dev).eval()
        return pol, pol.config
    if oracle_map:
        raise SystemExit("[E5_DP_SAMPLER] the oracle returns labels — no sampling to override")
    from lerobot.configs.policies import PreTrainedConfig

    name, steps = sampler
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    was = (cfg.noise_scheduler_type, cfg.num_inference_steps)
    cfg.noise_scheduler_type, cfg.num_inference_steps = name, steps
    pol = DiffusionPolicy.from_pretrained(ckpt, config=cfg).to(dev).eval()
    d = pol.diffusion
    got = type(d.noise_scheduler).__name__
    if got != f"{name}Scheduler" or int(d.num_inference_steps) != steps:
        raise SystemExit(f"[E5_DP_SAMPLER] asked {name} steps={steps} — built {got} steps={d.num_inference_steps}")
    print(
        f"[E5_DP_SAMPLER] {name} steps={steps} · scheduler {got} · train_timesteps "
        f"{d.noise_scheduler.config.num_train_timesteps} · clip_sample {cfg.clip_sample} {cfg.clip_sample_range} · "
        f"checkpoint config was {was[0]} steps={was[1]}",
        flush=True,
    )
    return pol, pol.config


def sampler_override(sampler: str, steps: int) -> tuple[str, int] | None:
    """(scheduler, steps) — both given or neither (None)."""
    if not sampler and not steps:
        return None
    if sampler not in SAMPLERS:
        raise SystemExit(f"[E5_DP_SAMPLER] --sampler {sampler!r} — DDIM | DDPM (with --infer-steps)")
    if steps < 1:
        raise SystemExit(f"[E5_DP_SAMPLER] --infer-steps {steps} — an integer ≥ 1 (with --sampler)")
    return sampler, steps


def oracle_label_source(root: str, oracle_map: str, oracle_root: str, exact: bool) -> str:
    """Folder the oracle reads its labels from (`--oracle-root`, a labels-only conversion, else `--root`).

    Model, normaliser and task texts stay `--root`'s; the label root's modality groups must be GROUPS. With `exact` the
    oracle returns the training label itself (no normalise · clamp · unnormalise): the policy's normaliser cannot
    carry
    a
    label column it never saw varying. Both need --oracle-map."""
    if (exact or oracle_root) and not oracle_map:
        raise SystemExit("[E5_DP_ORACLE_ROOT] --oracle-root · --oracle-exact need --oracle-map")
    if oracle_root:
        with open(f"{oracle_root}/meta/modality.json") as fh:
            lmj = json.load(fh)
        for kind in ("state", "action"):
            got = tuple((g, int(v["start"]), int(v["end"])) for g, v in lmj[kind].items())
            if got != GROUPS:
                raise SystemExit(f"[E5_DP_ORACLE_ROOT] {oracle_root}/meta/modality.json {kind} {got} ≠ {GROUPS}")
        conv = f"{oracle_root}/meta/e5_conversion.json"
        with open(conv, "rb") as fh:
            csha = hashlib.sha256(fh.read()).hexdigest()
        with open(conv) as fh:
            cj = json.load(fh)
        print(
            f"[E5_DP_ORACLE_ROOT] labels from {oracle_root} · conversion_sha256={csha} · "
            f"action_clip={'on' if cj.get('action_clip') else 'off'} · git_commit={cj.get('git_commit')}",
            flush=True,
        )
    return oracle_root or root
