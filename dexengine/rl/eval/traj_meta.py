"""Run metadata of a trajectory recording: the run axes written to meta.env and the extra-slot session totals."""

from __future__ import annotations

import os


def run_meta(args_cli, iteration=None, evals_per_clip=None) -> dict:
    """The run axes written to meta.env of a recording: run, checkpoint, arm, hand, dataset, embodiment, asset
    variant, clip list and sequence directory. Used by the frozen-judge recording of training and by the kinematic
    data recording; the data store reads hand, arm and dataset from it."""
    return {
        "run": getattr(args_cli, "experiment_name", None),
        "ckpt": os.path.abspath(getattr(args_cli, "resume_from", None) or ""),
        "iteration": iteration,
        "evals_per_clip": evals_per_clip,
        "arm": getattr(args_cli, "arm", None),
        "hand": getattr(args_cli, "hand", None),
        "dataset": os.environ.get("DEXENGINE_DATASET", ""),
        "embodiment": os.environ.get("DEXENGINE_EMBODIMENT", ""),
        "usd_variant": os.environ.get("DEXENGINE_COMBO_USD_VARIANT", ""),
        "seq_list": os.environ.get("DEXENGINE_SEQ_LIST", ""),
        "sequence_dir": getattr(args_cli, "sequence_dir", None),
    }


def _extra_slots_meta(raw_env) -> dict:
    """Session totals of the extra object slots (``extra_slots.summary_of``) under the key "extra_slots"; no key when
    the env has no slots. The collection metadata reads the per-clip robot ↔ extra-object contact ratio from it."""
    import sys as _sys

    mod = _sys.modules.get("dexengine.rl.tasks.scene.extra_slots")  # only when the env loaded it
    summ = None if mod is None else mod.summary_of(raw_env)
    return {} if summ is None else {"extra_slots": summ}
