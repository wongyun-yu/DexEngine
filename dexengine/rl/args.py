"""Argument parser of the trainer entry `dexengine/rl/cli.py`.

Pure argparse — no Isaac imports. The entry calls build_parser() and then appends the AppLauncher args before
parsing. Flag names, types, defaults and actions are the launcher's argv contract (`dexengine.rl.launch`,
recorded in tests/rl/launch_cases/*.case); the flags live in `dexengine.rl.cli_args` by group.
"""

import argparse

from dexengine.rl.cli_args import pilot as _pilot
from dexengine.rl.cli_args import pilot_contact as _pilot_contact
from dexengine.rl.cli_args import rewards as _rewards
from dexengine.rl.cli_args import rewards_closure_voc as _rewards_closure_voc
from dexengine.rl.cli_args import run as _run
from dexengine.rl.cli_args import run_outputs as _run_outputs
from dexengine.rl.cli_args import tracking as _tracking
from dexengine.rl.cli_args import tracking_ppo as _tracking_ppo


def build_parser() -> argparse.ArgumentParser:
    """The trainer argv parser (without the AppLauncher arguments)."""
    parser = argparse.ArgumentParser(description="DexEngine residual-PPO trainer entry point.")
    parser.set_defaults(arm_gains_given=())  # filled by _ArmGainGiven; empty = no arm gain flag was given
    for add in (
        _run.add_run_args,
        _run_outputs.add_run_output_args,
        _tracking.add_tracking_args,
        _tracking_ppo.add_tracking_ppo_args,
        _pilot.add_pilot_args,
        _pilot_contact.add_pilot_contact_args,
        _rewards.add_reward_args,
        _rewards_closure_voc.add_closure_voc_args,
    ):
        add(parser)
    return parser
