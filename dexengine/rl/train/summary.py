"""Training run, final summary, held-out manifest, W&B and env shutdown."""

import os

from dexengine.rl.train.post_hook import _Exp1EarlyStop


def run_and_report(*, _dist_rank, _fast_diag_flush, diag, exp_dir, frozen_eval_snapshot, runner, wrapped, args_cli):
    """Run training, then print the summary, write the held-out manifest and close W&B and the env."""
    # Run training (Exp-1: catch early-stop so we still write the final summary)
    try:
        runner.run()
    except _Exp1EarlyStop:
        if getattr(frozen_eval_snapshot, "_mem_restart", None) is not None:
            print(
                f"  [MEM_RESTART] iter {frozen_eval_snapshot._mem_restart}: planned restart — the launcher's "
                f"automatic resume continues from this iteration's checkpoint.",
                flush=True,
            )
        else:
            print(
                "  [EXP1] training stopped early on return collapse — "
                "best_return_agent.pt holds the best pre-collapse weights."
            )

    # Final summary (dexengine.rl.launch.training parses "  Training Complete" and "  Checkpoints:")
    _fast_diag_flush()  # collect the step diagnostics left after the last eval
    print(f"\n{'=' * 70}")
    print("  Training Complete")
    print(f"  Total episodes: {diag['total_episodes']}")
    if diag["total_episodes"] > 0:
        print(
            f"  Early term: {diag['early_terminations']} "
            f"({diag['early_terminations'] / diag['total_episodes'] * 100:.1f}%)"
        )
    print(f"  Checkpoints: {exp_dir}/checkpoints")
    print(f"{'=' * 70}")

    # Save held-out eval manifest CSV if --heldout_data_dir was specified.
    # The CSV records the best_agent.pt path + held-out dir so a separate
    # eval pass can compute eval/heldout/completion_rate post-training.
    if args_cli.heldout_data_dir and _dist_rank == 0:
        import csv as _csv

        best_ckpt_path = os.path.join(exp_dir, "checkpoints", "best_agent.pt")
        manifest_path = os.path.join(exp_dir, "heldout_eval_manifest.csv")
        heldout_abs = os.path.abspath(args_cli.heldout_data_dir)
        with open(manifest_path, "w", newline="") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["best_agent_pt", "heldout_data_dir", "train_data_dir"])
            _w.writerow([best_ckpt_path, heldout_abs, os.path.abspath(args_cli.data_dir)])
        print(f"  Held-out manifest: {manifest_path}")
        print("  Run held-out eval with:")
        print(f"    --task eval --checkpoint {best_ckpt_path}")
        print(f"    --data_dir {heldout_abs}  # → logs as eval/heldout/completion_rate")

    if args_cli.wandb:
        import wandb

        wandb.finish()

    wrapped.close()
