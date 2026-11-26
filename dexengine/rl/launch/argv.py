"""Trainer argv of one Isaac Lab process (`dexengine/rl/cli.py`): the residual-PPO recipe flags in launch order.

The order is part of the contract: argparse keeps the last value of a repeated flag, and recorded launches
(dry-run dumps replayed by data recording, judge-vs-training manifest checks) compare argv token by token. The
segments below therefore keep their historical order, including flags that appear twice (the later copy wins):
`--arm_kp/--arm_kd`, the wrist caps, `--policy_arch` (perhand, then token) and `--enable_contact_termination`.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from dexengine.rl.launch import hand_recipe
from dexengine.rl.launch.recipe import RecipeError

ENTRY = "dexengine/rl/cli.py"
#: Weights & Biases project of every training run (the name existing runs are filed under).
WANDB_PROJECT = "dexengine-rl"
#: Flags owned by recipe keys; a copy in the caller's extra flags would silently win over the stated value.
TABLE_OWNED = {"--obj_obs_fixed": "OBJ_OBS_FIXED", "--enable_force_dir_reward": "FORCE_DIR"}


@dataclasses.dataclass
class Call:
    """One trainer process. `task` = train (training, judge, frozen render) or eval (the kinematic replay of the record
    render). `entry` = what the process is (dexengine.rl.app.ENTRIES — train · judge on the plain kit, render · replay
    on the rendering kit, dumps for a record session); the launcher sets it for every process, the trainer reads it
    from DEXENGINE_ENTRY."""

    exp: str  # --exp_name; also names the trainer log
    extra_flags: str  # branch flags, appended after the recipe segments (argparse last wins)
    entry: str  # dexengine.rl.app.ENTRIES (no default: every mode names its processes)
    task: str = "train"
    ne: str = "4096"
    mi: str = "2000"
    smoke: bool = False
    wandb: bool = True
    wandb_group: str = ""
    bimanual: bool = False
    seq_dir: str = ""
    render_out: str = ""  # replay dump: --videos_out_dir (the record session's output folder)
    render_tag: str = ""  # replay: log name tag (_<clip>)
    env: dict = dataclasses.field(default_factory=dict)  # extra trainer environment of this process
    gpu: str = "0"  # CUDA_VISIBLE_DEVICES of the process
    process_env: dict = dataclasses.field(default_factory=dict)  # non-DEXENGINE environment (e.g. WANDB_MODE)
    #: switch names (environment form) this process sets for its own context (the record scene env of a drawing dump);
    #: over a training run's switches record they are the only values the record does not own
    context_switches: tuple = ()

    @property
    def log_name(self) -> str:
        """Trainer log file name (in the run's log folder) — readers grep these names."""
        if self.task == "eval":
            return f"{self.exp}{self.render_tag}_replay_render.log"
        return f"{self.exp}_smoke.log" if self.smoke else f"{self.exp}_train.log"


def split_flags(flags: str) -> list[str]:
    """Split a flag string on blanks without quote handling (the launch-record convention)."""
    return hand_recipe.launch_tokens(flags)


def _check_table_owned(tokens: list[str]) -> None:
    for t in tokens:
        name = t.split("=", 1)[0]
        if not name.startswith("--") or len(name) <= 2:
            continue
        for owned, key in TABLE_OWNED.items():
            if owned.startswith(name):
                raise RecipeError(
                    f"extra flag {t!r} sets {owned}, which the recipe key {key} owns — override {key} instead"
                )


def _policy_size(tokens: list[str], size: str) -> list[str]:
    """Replace every --policy_size value in `tokens` by `size` (append it when absent)."""
    out, hit, i = [], False, 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--policy_size":
            out += ["--policy_size", size]
            i, hit = i + 2, True
            continue
        if t.startswith("--policy_size="):
            out.append(f"--policy_size={size}")
            hit = True
        else:
            out.append(t)
        i += 1
    return out if hit else out + ["--policy_size", size]


def _flag(v: str, name: str) -> str:
    if v not in ("0", "1"):
        raise RecipeError(f"{name} must be 0 or 1 (got {v!r})")
    return v


def trainer_args(call: Call, v: dict, dataset: str) -> list[str]:
    """argv of one trainer process (without the interpreter) from the resolved recipe values `v`."""
    flags = split_flags(call.extra_flags)
    if v.get("POLICY_SIZE"):
        flags = _policy_size(flags, v["POLICY_SIZE"])
    _check_table_owned(flags)
    obj_side = "left" if "--side left" in f" {call.extra_flags} " else "right"
    force_dir = _flag(v["FORCE_DIR"], "FORCE_DIR")
    contact_match = _flag(v["CONTACT_MATCH"], "CONTACT_MATCH")
    if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", v["FT_HINGE_CM"]):
        raise RecipeError(f"FT_HINGE_CM must be a number in cm (got {v['FT_HINGE_CM']!r})")
    gains = ["--arm_kp", v["ARM_KP"], "--arm_kd", v["ARM_KD"]] if v["ARM_JOINT_PD"] == "1" else []
    phys = ["--reference_base", "1", *gains, f"--object_mass_{obj_side}", "0.2", "--hand_friction", "2.0"]
    obs = ["--obj_obs_fixed", _flag(v["OBJ_OBS_FIXED"], "OBJ_OBS_FIXED")]
    return _train_task_argv(call, v, dataset, contact_match, flags, force_dir, obs, phys)


def _train_task_argv(call, v, dataset, contact_match, flags, force_dir, obs, phys):
    """argv of a train-task process from the resolved recipe values."""
    wrist = [
        "--wrist_pos_residual_max_m",
        "0.07",
        "--wrist_rot_residual_max_deg",
        "120",
        "--wrist_residual_l2_weight",
        "0.9",
    ]
    resid = [
        "--residual_scale",
        "0.3",
        "--finger_residual_scale",
        "1.5",
        "--wrist_residual_scale",
        "0",
        "--residual_l2_weight",
        v["RESL2"],
    ]
    contact = [
        "--enable_contact_termination",
        "1",
        "--contact_term_grace_steps",
        "75",
        "--contact_term_force_threshold",
        "0.05",
    ]
    gate = ["--obj_rot_thresh_const", v["ROTC"]] if v["ROTC"] != "0" else []
    gate += ["--obj_pos_thresh_const", f"{float(v['POSC']) / 100:.4f}"] if v["POSC"] != "0" else []
    seq = ["--sequence_dir", call.seq_dir]
    if call.task == "eval":
        return _eval_task_argv(call, dataset, contact, flags, force_dir, gate, obs, phys, resid, seq, wrist)
    closure, follow, keypoint, mi, ne, rewards, rsi, wandb = _trainer_args_tail(call, v, contact_match, force_dir)
    return [
        ENTRY,
        "--task",
        "train",
        "--dataset",
        dataset,
        *seq,
        "--num_envs",
        ne,
        "--max_iterations",
        mi,
        "--seed",
        v["SEED"],
        "--eval_interval",
        v["EVAL_INTERVAL"],
        "--headless",
        *phys,
        *obs,
        *wrist,
        *resid,
        "--entropy_loss_scale",
        v["ENT_SCALE"],
        "--kl_threshold",
        v["KL_THRESH"],
        *contact,
        *gate,
        *closure,
        *rsi,
        *follow,
        *keypoint,
        *(["--bimanual", "1", "--policy_arch", "perhand"] if call.bimanual else []),
        *flags,
        *rewards,
        "--exp_name",
        call.exp,
        *wandb,
    ]


def _eval_task_argv(call, dataset, contact, flags, force_dir, gate, obs, phys, resid, seq, wrist):
    """argv of an eval-task process: the kinematic replay scene a record session draws (--record_session)."""
    return [
        ENTRY,
        "--task",
        "eval",
        "--dataset",
        dataset,
        *seq,
        "--num_envs",
        "1",
        "--headless",
        "--seed",
        "42",
        "--slowdown_factor",
        "1.0",
        "--freeze_curriculum",
        *phys,
        *obs,
        *wrist,
        *resid,
        *contact,
        *gate,
        "--fingertip_hinge_cm",
        "2.0",
        "--enable_contact_match_reward",
        "1",
        "--contact_reward_mode",
        "soft",
        "--contact_source",
        "mano",
        "--contact_softness_k",
        "15",
        "--enable_grasp_target",
        "1",
        "--enable_approach_reward",
        "1",
        "--enable_force_dir_reward",
        force_dir,
        "--enable_pregrasp_reward",
        "1",
        "--pregrasp_dist",
        "0.12",
        "--keep_level1",
        "1",
        "--disable_fingertip_termination",
        "1",
        "--record_session",
        "--enable_cameras",
        "--kinematic_replay",
        *(["--bimanual", "1", "--policy_arch", "perhand"] if call.bimanual else []),
        *flags,
        "--videos_out_dir",
        call.render_out,
    ]


def _trainer_args_tail(call, v, contact_match, force_dir):
    """The trailing trainer flags (wandb and the rest, in launch-contract order)."""
    closure = [
        "--enable_closure_force",
        "1",
        "--closure_force_weight",
        v["CF_W"],
        "--closure_force_cap",
        "5.0",
        "--closure_force_nonthumb_scale",
        "1.5",
        "--enable_closure_potential",
        "1",
        "--closure_potential_weight",
        "2.0",
        "--closure_potential_clamp_cm",
        "1.0",
        "--closure_perfinger",
        "1",
        "--closure_margins",
        v["MARGINS"],
        "--closure_force_immediate",
        "1",
    ]
    rsi = [
        "--rsi_enable",
        "1",
        "--rsi_fraction_start",
        v["RSI_FRACTION"],
        "--rsi_target_fraction",
        v["RSI_FRACTION"],
        "--rsi_ramp_episodes",
        "1",
        "--rsi_lo_frac",
        "0.0",
        "--rsi_hi_frac",
        "0.35",
        "--rsi_hi_frac_end",
        "1.0",
        "--rsi_ramp_frac",
        v["RSI_BAND_RAMP"],
    ]
    follow = ["--obj_follow_schedule_target", v["OBJ_FOLLOW_RAMP"]] if v["OBJ_FOLLOW_RAMP"] != "1" else []
    keypoint = ["--obj_keypoint_reward", "1", "--obj_keypoint_mode", "farthest"]
    keypoint += ["--obj_keypoint_weight", v["KP_W"]] if v["KP_W"] != "1.25" else []
    rewards = [
        "--enable_physics_dr",
        v["PHYSDR"],
        "--fingertip_hinge_cm",
        v["FT_HINGE_CM"],
        "--enable_contact_match_reward",
        contact_match,
        "--contact_reward_mode",
        "soft",
        "--contact_source",
        "mano",
        "--contact_softness_k",
        "15",
        "--enable_grasp_target",
        "1",
        "--enable_approach_reward",
        "0",
        "--enable_force_dir_reward",
        force_dir,
        "--enable_pregrasp_reward",
        "1",
        "--pregrasp_dist",
        "0.12",
        "--keep_level1",
        "1",
        "--disable_fingertip_termination",
        "1",
    ]
    ne, mi = ("64", "1") if call.smoke else (call.ne, call.mi)
    wandb = []
    if call.wandb and not call.smoke:
        wandb = ["--wandb", "--wandb_project", WANDB_PROJECT]
        wandb += ["--wandb_group", call.wandb_group] if call.wandb_group else []
    return closure, follow, keypoint, mi, ne, rewards, rsi, wandb


def log_path(log_dir: Path, call: Call) -> Path:
    """Where the trainer process writes its stdout/stderr (and a dry run its argv dump)."""
    return log_dir / call.log_name
