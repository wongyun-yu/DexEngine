"""Resume weights, the skrl sequential trainer with iteration-preserving resume, and the curriculum-clock offset."""

import os

import torch
from skrl.trainers.torch import SequentialTrainer

from dexengine.core.errors import InputMissingError
from dexengine.rl.agents.models import assert_checkpoint_obs_dim


def build_trainer(*, agent, env, rollouts, timesteps, wrapped, args_cli):
    """Load resume weights, build the sequential trainer (with iteration-preserving resume) and wrap it."""
    # DIAGNOSTIC: optionally resume from a checkpoint BEFORE training.
    # Use this to verify a trained policy inside the training code path
    # (bypasses run_eval and isolates eval-path bugs).
    if args_cli.resume_from:
        resume_path = os.path.abspath(args_cli.resume_from)
        print(f"[run_train] RESUMING weights from: {resume_path}")
        assert_checkpoint_obs_dim(resume_path, int(wrapped.observation_space.shape[0]), "resume_from")
        agent.load(resume_path)
        print(f"[run_train] Resume OK. Running {args_cli.max_iterations} iterations...")

    # Trainer config
    trainer_cfg = {
        "timesteps": timesteps,
        "environment_info": "log",
        "close_environment_at_exit": False,
    }

    # Create trainer (replaces the SKRL Runner)
    class _TrainRunner:
        """Thin wrapper matching Runner interface for monkey-patching compatibility."""

        def __init__(self, trainer, agent):
            self.trainer = trainer
            self.agent = agent

        def run(self):
            self.trainer.train()

    trainer = SequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent)
    return _install_iteration_resume(agent, env, rollouts, timesteps, wrapped, args_cli, _TrainRunner, trainer)


def _install_iteration_resume(agent, env, rollouts, timesteps, wrapped, args_cli, _TrainRunner, trainer):
    """Iteration-preserving resume: the skrl trainer loops over range(initial_timestep, timesteps) and the Isaac Lab
    curricula (threshold_tightening, ...) measure progress with env.common_step_counter — set both.
    """
    _ri = int(getattr(args_cli, "resume_iteration", 0) or 0)
    if _ri > 0:
        if not args_cli.resume_from:
            raise SystemExit("--resume_iteration needs --resume_from.")
        _rs = _ri * int(rollouts)
        if _rs >= timesteps:
            raise InputMissingError(
                f"--resume_iteration {_ri} ≥ max_iterations({timesteps // int(rollouts)}) — nothing left to train."
            )
        trainer.initial_timestep = _rs
        _uenv = getattr(wrapped, "unwrapped", None) or env
        _uenv.common_step_counter = _rs
        print(
            f"[RESUME] continuing from iteration {_ri}: trainer.initial_timestep={_rs} "
            f"env.common_step_counter={_rs} (remaining {timesteps - _rs} steps = "
            f"{(timesteps - _rs) // int(rollouts)} iters; curriculum progress kept)",
            flush=True,
        )

    # Resume from a checkpoint if provided
    if args_cli.resume:
        resume_path = os.path.abspath(args_cli.resume)
        # Bimanual DIM-GUARD: a bimanual run is from-scratch (24-dim
        # action, larger obs); the right-only ckpt (12-dim) is NOT a valid
        # warm-start. the learner's load() does a state_dict load that would
        # shape-mismatch or (worse) silently skip mismatched keys, leaving half
        # the policy random. Hard-error (NOT strict=False) on any encoder/mu/
        # value dim mismatch. Gated by --bimanual (off by default).
        if args_cli.bimanual:
            _live_obs = int(wrapped.observation_space.shape[0])
            _live_act = int(wrapped.action_space.shape[0])
            _ck = torch.load(resume_path, map_location="cpu", weights_only=False)
            _pol = _ck.get("policy", {}) if isinstance(_ck, dict) else {}

            def _shape_of(d, *keys):
                for k in keys:
                    if k in d:
                        return tuple(d[k].shape)
                return None

            _enc_in = _shape_of(_pol, "encoder.0.weight", "net_container.0.weight")
            _mu_out = _shape_of(_pol, "mu_head.weight", "policy_layer.weight")
            _ck_obs = _enc_in[1] if _enc_in else None
            _ck_act = _mu_out[0] if _mu_out else None
            print(f"  [RESUME DIM-GUARD] live obs={_live_obs} act={_live_act} | ckpt obs={_ck_obs} act={_ck_act}")
            if (_ck_obs is not None and _ck_obs != _live_obs) or (_ck_act is not None and _ck_act != _live_act):
                raise RuntimeError(
                    f"--resume dim mismatch (bimanual): checkpoint policy obs/act "
                    f"({_ck_obs}/{_ck_act}) != live spaces ({_live_obs}/{_live_act}). "
                    f"A right-only checkpoint CANNOT warm-start the bimanual "
                    f"policy — the bimanual agent must train from scratch (do NOT "
                    f"strict=False around this)."
                )
        print(f"  [RESUME] Loading checkpoint: {resume_path}")
        assert_checkpoint_obs_dim(resume_path, int(wrapped.observation_space.shape[0]), "resume")
        agent.load(resume_path)
        print("  [RESUME] agent weights loaded successfully")

    runner = _TrainRunner(trainer, agent)
    return runner


def _resolve_curriculum_step_offset(args_cli, rollouts: int) -> int:
    """Where a continued training resumes the schedule clock (``common_step_counter`` units).

    ``--curriculum_step_offset`` wins when given. Otherwise the offset is derived from the checkpoint:

      * ``agent_<N>.pt``  -> N (the total steps when that file was saved)
      * ``best_agent.pt`` -> no number; the run folder's ``TRAIN_DONE`` sentinel records ``MI=<iterations>``, so the
        offset is ``MI x rollouts``. The sentinel is written only by a run that finished; without it the run did not
        reach its final schedule, so this returns 0 with a warning.

    0 when the launch is not a resume.
    """
    if args_cli.curriculum_step_offset is not None:
        return max(int(args_cli.curriculum_step_offset), 0)
    ck = args_cli.resume or args_cli.resume_from
    if not ck:
        return 0
    import re as _re_cso

    p = os.path.abspath(ck)
    m = _re_cso.search(r"agent_(\d+)\.pt$", os.path.basename(p))
    if m:
        n = int(m.group(1))
        print(f"[CURRICULUM CONTINUE] step {n} taken from the checkpoint name (= {n // rollouts} iter).")
        return n
    sentinel = os.path.join(os.path.dirname(os.path.dirname(p)), "TRAIN_DONE")
    try:
        with open(sentinel) as fh:
            txt = fh.read()
    except OSError:
        print(
            f"[CURRICULUM CONTINUE] WARN: {os.path.basename(p)} carries no step number and there is no "
            f"TRAIN_DONE ({sentinel}) — continuing with offset 0. The curriculum tightens again from "
            f"the start, so this is not a continued training."
        )
        return 0
    mi = _re_cso.search(r"MI=(\d+)", txt)
    if not mi:
        print("[CURRICULUM CONTINUE] WARN: TRAIN_DONE has no MI= — offset 0.")
        return 0
    n = int(mi.group(1)) * rollouts
    print(f"[CURRICULUM CONTINUE] TRAIN_DONE MI={mi.group(1)} x rollouts {rollouts} = step {n} used as the offset.")
    return n
