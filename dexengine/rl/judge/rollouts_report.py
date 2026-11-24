"""Console reports of the clip-balanced judge loop: one block per batch and the final aggregate.

The aggregate block (`AGGREGATE EVALUATION REPORT`, `Completion rate:`, the `Per-clip breakdown` table, the
`[PER_CLIP]` json line) is parsed by judge/metrics.py and scripts/rl/judge_sidecar.sh; its text is fixed.
"""

from __future__ import annotations

import json

from dexengine.rl.judge.rollouts_types import BatchResult

#: head of the gate-A exclusion tally; scripts/rl/judge_sidecar.sh (park_excl_a_note) reads the clip count after it
EXCL_A_TALLY = "[EVAL_PARK_EXCL_A] tally: clips left out of the gate checks"


def _print_batch_report(
    batch: BatchResult,
    clip_results: dict,
    num_clips: int,
    clip_names: list[str] | None,
    warmup_prefix: str = "",
):
    """Print a summary for one completed batch.

    Args:
        warmup_prefix: If non-empty, prepend to the header line to indicate
            this batch is a warmup and was discarded from aggregates.
    """
    print(f"\n{'=' * 70}")
    print(f"  {warmup_prefix}Batch {batch.batch_idx + 1}  ({batch.total} episodes, {batch.batch_size} envs)")
    print(f"{'=' * 70}")
    print(f"  Completion rate:     {batch.completion_rate:.1%}  ({batch.completed}/{batch.total})")
    print(f"  Early termination:   {batch.early_rate:.1%}  ({batch.early_terminated}/{batch.total})")
    print(f"  Mean episode length: {batch.mean_ep_len:.1f} steps")

    # Clip distribution in this batch
    counts = {}
    for cid, (c, e) in batch.clip_counts.items():
        counts[cid] = c + e
    if counts:
        vals = sorted(counts.values())
        min_c, max_c = vals[0], vals[-1]
        if min_c == max_c:
            print(f"  Clip distribution:   {min_c} episodes/clip x {len(counts)} clips")
        else:
            n_min = sum(1 for v in vals if v == min_c)
            n_max = sum(1 for v in vals if v == max_c)
            print(f"  Clip distribution:   {min_c} eps/clip x {n_min} clips, {max_c} eps/clip x {n_max} clips")

    # Per-clip completion rates for this batch (compact)
    if batch.clip_counts:
        lines = []
        for cid in sorted(batch.clip_counts.keys()):
            c, e = batch.clip_counts[cid]
            total = c + e
            rate = c / total * 100 if total > 0 else 0
            name = clip_names[cid] if clip_names and cid < len(clip_names) else f"clip_{cid}"
            # Shorten name for display
            short = name.split("__")[1] if "__" in name else name
            lines.append(f"{short}:{rate:4.0f}%")
        # Print 4 per line
        print(f"  Per-clip rates:")  # noqa: F541
        for i in range(0, len(lines), 4):
            print(f"    {' | '.join(lines[i : i + 4])}")
    print()


def _print_aggregate_report(aggregate: dict, num_clips: int):
    """Print final aggregate report across all batches."""
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE EVALUATION REPORT")  # noqa: F541
    print(f"{'=' * 70}")
    print(f"  Total episodes:      {aggregate['num_episodes']}")
    print(f"  Clips evaluated:     {num_clips}")
    print(f"  Completion rate:     {aggregate['completion_rate']:.1%}")
    print(f"  Early termination:   {aggregate['early_termination_rate']:.1%}")
    print(f"  Mean episode length: {aggregate['mean_episode_length']:.1f} steps")

    # Per-clip breakdown
    per_clip = aggregate.get("per_clip", [])
    if per_clip:
        print(f"\n  Per-clip breakdown:")  # noqa: F541
        print(f"  {'Clip':<50s} {'Complete':>8s} {'Early':>8s} {'Total':>6s} {'Rate':>6s}")
        print(f"  {'-' * 50} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 6}")
        for pc in sorted(per_clip, key=lambda x: x["clip_name"]):
            name = pc["clip_name"]
            if len(name) > 50:
                name = "..." + name[-47:]
            print(
                f"  {name:<50s} {pc['completed']:>8d} {pc['early_terminated']:>8d} "
                f"{pc['total']:>6d} {pc['completion_rate']:>5.1%}"
            )
        # Machine-readable line: the table above truncates names over 50 characters. Its one parser is
        # judge/metrics.py parse_per_clip (pass10 rule).
        # excl_a = clips left out of the PARK gates by the gate-A exclusion, {clip: [batches left out, real batches]}
        # (only those clips; display only, they are still scored)
        nb = int(aggregate.get("num_batches", 0))
        xa = {pc["clip_name"]: [int(pc.get("excl_a_batches", 0)), nb] for pc in per_clip if pc.get("excl_a_batches", 0)}
        print(
            f"{EXCL_A_TALLY} {len(xa)} · still scored"
            + "".join(f" · {k} {c}/{t} batches" for k, (c, t) in sorted(xa.items())),
            flush=True,
        )
        print(
            "[PER_CLIP] "
            + json.dumps(
                {
                    "n": len(per_clip),
                    "clips": [[pc["clip_name"], int(pc["completed"]), int(pc["total"])] for pc in per_clip],
                    "excl_a": xa,
                },
                ensure_ascii=False,
            )
        )
    print(f"{'=' * 70}\n")
