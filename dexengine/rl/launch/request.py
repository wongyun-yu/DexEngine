"""What one RL launch is asked to do: the typed request and its reader for recorded launch environments.

A launch is one run name plus a side (right|left = one hand, both = bimanual), the robot (arm, hand), the data view
(dataset, embodiment, SEQ_DIR) and one mode:

  train   PPO training (cold start), then for bimanual an optional post-training video (`video`)
  smoke   64-env 1-iteration boot check
  judge   fresh-process judge of one checkpoint (the paper's clip acceptance; `dexengine.rl.judge`)
  render  bimanual physics render of finished checkpoints (`render_ckpts`, read from `render_rundir`)
  replay  kinematic reference replay video, one Isaac session per clip

Campaign cells, judge sidecars and recorded launch lines store a launch as environment tokens
(`env SIDE=both GPU=0 SEQ_DIR=… DEXENGINE_ARM=… FROZEN_REPLAY_CKPT=… <entry> RUN`); `from_env` reads that form, so a
finished cell can be judged or rendered again with the exact inputs it was trained with.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path

from dexengine.core.repo import REPO_ROOT
from dexengine.rl.launch.recipe import RecipeError, path_of

MODES = ("train", "smoke", "judge", "render", "replay")
#: Out-name suffix per mode (logs, manifests and the run snapshot are named `<run><suffix>`); readers of the judge
#: and replay logs (judge sidecar, data recording) use these names.
OUT_SUFFIX = {"train": "", "smoke": "_smoke", "judge": "_frozenreplay", "render": "_render", "replay": "_replay"}
#: Environment keys that select a mode in a recorded launch line (one at most).
_MODE_KEYS = ("SMOKE", "FROZEN_REPLAY_CKPT", "RENDER_ONLY")
#: Inputs only the judge mode consumes (trajectory record/render hooks of the judge loop).
_JUDGE_ONLY_ENV = ("DEXENGINE_TRAJ_RECORD", "DEXENGINE_TRAJ_RENDER", "FROZEN_RENDER_OUT")


@dataclasses.dataclass
class Request:
    """One launch. `overrides` are recipe knobs (configs/rl/recipe.yaml keys); `env` holds the caller's DEXENGINE_*
    environment, which the trainer process inherits unchanged (data root, clip list, record hooks, ablation
    switches)."""

    run: str
    side: str
    arm: str
    hand: str
    dataset: str
    embodiment: str
    seq_dir: str
    gpu: str = "0"
    mode: str = "train"
    extra: str | None = None  # trainer flags; None = "--arm <arm> --hand <hand>"
    overrides: dict = dataclasses.field(default_factory=dict)
    env: dict = dataclasses.field(default_factory=dict)
    ckpt: str = ""  # judge: checkpoint path ("none" = policy-free PARK probe)
    judge_ne: str = ""  # judge: env count (default = the recipe NE)
    judge_epc: str = ""  # judge: evaluations per clip
    judge_iter: str = ""  # judge: PPO iteration of the checkpoint (label of the judge record)
    render_ckpts: tuple = ()  # render: checkpoint file names under <render_rundir>/checkpoints
    render_rundir: str = ""  # render: finished run folder (default logs/skrl/<train dir>/<run>)
    render_out_dir: str = ""  # render/replay: video folder override
    traj_render_out: str = ""  # judge with DEXENGINE_TRAJ_RENDER: video folder
    frozen_render_out: str = ""  # one-hand judge: render the judged rollouts into this folder
    video: bool = True  # bimanual train: render the final checkpoint afterwards
    markers: str = "0"  # render/replay: draw reference markers
    videos_only_clips: str = ""  # bimanual render: only clips whose name contains one of these tokens
    wandb_group: str = ""
    dry_run: bool = False  # write the trainer argv + env dump instead of launching
    need_mib: int | None = None  # free GPU memory to wait for before a launch (None = mode default)
    switches_from: str = ""  # judge/render: the training run's switches record
    root: Path = REPO_ROOT  # code tree the trainer runs from (its logs/ and overfit/ live here)

    def __post_init__(self):
        self.root = Path(self.root)
        if self.mode not in MODES:
            raise RecipeError(f"mode must be one of {MODES} (got {self.mode!r})")
        self.path = path_of(self.side)
        for k in ("run", "arm", "hand", "dataset", "embodiment"):
            if not getattr(self, k):
                raise RecipeError(f"{k} is required (no default)")
        if not self.seq_dir:
            raise RecipeError("seq_dir (SEQ_DIR) is required (no default)")
        self._check_mode_inputs()

    def _check_mode_inputs(self):
        """Inputs that only one mode (judge / render / one-hand judge) reads must not reach another mode."""
        if self.mode == "judge" and not self.ckpt:
            raise RecipeError("judge mode needs a checkpoint (FROZEN_REPLAY_CKPT)")
        if self.mode == "render" and self.path != "bi":
            raise RecipeError("render mode is bimanual only (one-hand judged rollouts render with frozen_render_out)")
        if self.mode == "render" and not self.render_ckpts:
            raise RecipeError("render mode needs checkpoint names (RENDER_ONLY=<agent_N.pt …>)")
        if self.switches_from and self.mode not in ("judge", "render"):
            raise RecipeError("switches_from (--switches-from) is a judge or render input: the training run's switches")
        if self.render_rundir and self.mode != "render":
            raise RecipeError("render_rundir (RENDER_RUNDIR) is only read by the render mode")
        if self.frozen_render_out and (self.path != "uni" or self.mode != "judge"):
            raise RecipeError("frozen_render_out is a one-hand judge input (bimanual renders use the render mode)")
        if self.mode != "judge":
            given = [k for k in _JUDGE_ONLY_ENV if self.env.get(k)]
            if self.traj_render_out:
                given.append("TRAJ_RENDER_OUT")
            if given:
                raise RecipeError(f"{given} are judge-mode inputs (FROZEN_REPLAY_CKPT); this launch is {self.mode}")
        if self.env.get("DEXENGINE_TRAJ_RENDER") and not self.traj_render_out:
            raise RecipeError("DEXENGINE_TRAJ_RENDER needs TRAJ_RENDER_OUT (the video folder)")
        if self.env.get("DEXENGINE_TRAJ_RENDER") and self.frozen_render_out:
            raise RecipeError("DEXENGINE_TRAJ_RENDER and FROZEN_RENDER_OUT are two renders of one judge; pick one")

    @property
    def run_tree(self) -> Path:
        """The run tree of this launch's trainer — where its logs, run folders, roster record and default videos go:
        paths.run_root_of(its data root, root). The trainer inherits `env`, not this process's DEXENGINE_* variables
        (process.child_environ), so its data root is env DEXENGINE_MAIN_ROOT, else the layout file's data_root, else
        the code tree — dexengine.core.layout.data_root's rule applied to that environment."""
        from dexengine.core import paths as P
        from dexengine.core.layout import roots as LR

        dr = self.env.get(LR.DATA_ROOT_ENV) or LR.layout_config().data_root or str(self.root)
        return P.run_root_of(Path(dr).resolve(), self.root)

    @property
    def out(self) -> str:
        """Output name of this launch: the run name, plus the mode suffix for every mode but training."""
        return self.run + OUT_SUFFIX[self.mode]


def from_env(run: str, environ: Mapping[str, str] | None = None, **fields) -> Request:
    """Request from a recorded launch environment (`env K=V … <entry> RUN`); `fields` override what it says.

    Recipe knobs (any key of configs/rl/recipe.yaml, e.g. NE, LAGW, KL_THRESH) become overrides; DEXENGINE_* keys
    pass to the trainer. A switch (`dexengine.rl.tasks.switches`) is given by its field name (`--set arm_ff=0`) or,
    in a recorded launch environment, by its old environment name (`DEXENGINE_ARM_FF=0`); both forms with different
    values fail."""
    environ = dict(os.environ if environ is None else environ)
    from dexengine.rl.launch.recipe import columns
    from dexengine.rl.tasks import switches

    for field in [k for k in environ if k in switches.ENV_NAME]:
        name, val = switches.ENV_NAME[field], environ.pop(field)
        if environ.get(name, val) != val:
            raise RecipeError(f"switch {field}={val!r} and {name}={environ[name]!r} disagree (give one)")
        environ[name] = val
    knobs = set().union(*columns().values())
    modes = [k for k in _MODE_KEYS if environ.get(k) and not (k == "SMOKE" and environ[k] == "0")]
    if len(modes) > 1:
        raise RecipeError(f"one mode per launch, got {modes} (SMOKE · FROZEN_REPLAY_CKPT · RENDER_ONLY)")
    return _request_from_environ(run, environ, fields, knobs)


def _request_from_environ(run, environ, fields, knobs):
    """Reject retired shell-launcher inputs, then build the Request from the environment (mode, sides, GPU)."""
    # Inputs of the old shell launcher that no caller sets any more: dying beats being silently ignored.
    for k in (
        "DO_SMOKE",
        "PROBE",
        "EVAL_CKPT",
        "TRAIN_EXTRA",
        "FORCE_TRAIN",
        "RENDER",
        "SEQ0",
        "NOEVAL",
        "RES",
        "CRF",
        "VPRESET",
        "QUALITY",
    ):
        if environ.get(k) and environ.get(k) != "0":
            raise RecipeError(f"{k} is not a launch input (removed); see scripts/stage3_rl.py --help")
    mode = "train"
    if environ.get("SMOKE") == "1":
        mode = "smoke"
    elif environ.get("FROZEN_REPLAY_CKPT"):
        mode = "judge"
    elif environ.get("RENDER_ONLY") == "replay":
        mode = "replay"
    elif environ.get("RENDER_ONLY"):
        mode = "render"
    kw = {
        "run": run,
        "side": environ.get("SIDE", ""),
        "gpu": environ.get("GPU", ""),
        "arm": environ.get("ARM") or environ.get("DEXENGINE_ARM", ""),
        "hand": environ.get("HAND") or environ.get("DEXENGINE_HAND", ""),
        "dataset": environ.get("DEXENGINE_DATASET", ""),
        "embodiment": environ.get("DEXENGINE_EMBODIMENT", ""),
        "seq_dir": environ.get("SEQ_DIR", ""),
        "mode": mode,
        "extra": environ.get("EXTRA"),
        "overrides": {k: v for k, v in environ.items() if k in knobs and not k.startswith("DEXENGINE_")},
        "env": {k: v for k, v in environ.items() if k.startswith("DEXENGINE_")},
        "ckpt": environ.get("FROZEN_REPLAY_CKPT", ""),
        "judge_ne": environ.get("FROZEN_REPLAY_NE", ""),
        "judge_epc": environ.get("FROZEN_REPLAY_EVALS_PER_CLIP", ""),
        "judge_iter": environ.get("FROZEN_REPLAY_ITER", ""),
        "render_ckpts": tuple(environ.get("RENDER_ONLY", "").split()) if mode == "render" else (),
        "render_rundir": environ.get("RENDER_RUNDIR", ""),
        "render_out_dir": environ.get("RENDER_OUT_DIR", ""),
        "traj_render_out": environ.get("TRAJ_RENDER_OUT", ""),
        "frozen_render_out": environ.get("FROZEN_RENDER_OUT", ""),
        "video": environ.get("DO_VIDEO", "1") == "1",
        "markers": environ.get("MARKERS", "0"),
        "videos_only_clips": environ.get("VIDEOS_ONLY_CLIPS", ""),
        "wandb_group": environ.get("WANDB_GROUP", ""),
        "dry_run": environ.get("DRYRUN", "0") == "1",
    }
    need = environ.get("BI_RENDER_NEED_MIB" if mode in ("render", "replay") else "BI_NEED_MIB")
    if need:
        kw["need_mib"] = int(need)
    kw.update(fields)
    if not kw["gpu"]:
        raise RecipeError("GPU is required (GPU=<index>)")
    return Request(**kw)
