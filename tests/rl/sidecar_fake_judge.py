#!/usr/bin/env python3
"""Stand-in for `scripts/stage3_rl.py <judge name>` in the judge sidecar tests: writes the judge manifest and a judge
log shaped like the trainer's (markers the sidecar checks), from the judge inputs the sidecar sets and FAKE_* knobs.

Completion rate of a judge at iteration i = 0.5 + i / 10000; per-clip completions put clips 7 and 9 at 0 and clip 8
exactly at the 10 % threshold, so a 64-clip judge passes 62 clips."""

import json
import os
import sys
from pathlib import Path

E = os.environ


def main() -> int:
    alias = sys.argv[1]
    # the judge's switches come from the training run's record, never from the environment
    if sys.argv[2:3] != ["--switches-from"] or not Path(sys.argv[3]).is_file():
        print(f"[FAKE_JUDGE] no --switches-from <existing record>: {sys.argv[2:]}", file=sys.stderr)
        return 2
    print(f"[FAKE_JUDGE] switches_from={sys.argv[3]}")
    out = f"{alias}_frozenreplay"
    k, log, rate, ref, tot = _fake_judge_inputs(alias, out)
    comp = {7: 0, 8: tot // 10, 9: 0}
    left = int(float(rate) * k * tot + 0.5) - comp[8]
    rest = [c for c in range(k) if c not in comp]
    for i, c in enumerate(rest):
        comp[c] = left // len(rest) + (1 if i < left % len(rest) else 0)
    log += [
        f"  {f'fake__c{c:02d}':<50} {comp[c]:8d} {tot - comp[c]:8d} {tot:6d} {100 * comp[c] / tot:5.1f}%"
        for c in range(k)
    ]
    log.append("[PER_CLIP] " + json.dumps({"n": k, "clips": [[f"fake__c{c:02d}", comp[c], tot] for c in range(k)]}))
    log += [
        "=" * 70,
        f"[FROZEN_REPLAY] DONE headline_completion=0.7 aggregate_completion_rate={rate} "
        "mean_episode_length=9.00 side_rates=None",
        "[FROZEN_REPLAY] EXIT (x)",
    ]
    rec = E.get("DEXENGINE_TRAJ_RECORD")
    if rec and not E.get("FAKE_TRAJ_SKIP"):
        Path(rec).write_text("trj\n")
        log.append(f"[TRAJ_REC] DONE fake · file {rec} sha256=0 · 0.0 MB")
    (ref / f"{out}_train.log").write_text("\n".join(log) + "\n")
    return 0


def _fake_judge_inputs(alias, out):
    """Reference folder, eval counts, log and completion rate of the fake judge."""
    ref = Path("overfit/logs/refbase")
    ne, k = int(E["FROZEN_REPLAY_NE"]), int(E.get("FAKE_K", "64"))
    mn, ce = ne // k, -(-ne // k)
    epc = max(int(E["FROZEN_REPLAY_EVALS_PER_CLIP"]), ce)
    nreal = int(E.get("FAKE_NREAL") or -(-epc // mn))
    it = int(E["FROZEN_REPLAY_ITER"])
    side = E.get("SIDE", "both")
    warm = int(E.get("FAKE_WARMUP", "1"))
    code = E.get("FAKE_CODE", "c0de00000001")
    man = [
        f"date=x git={E.get('FAKE_GIT', 'abc1234')} dirty=0 code={code} codedirty={E.get('FAKE_CODEDIRTY', '0')} "
        f"MODE=frozenreplay OUT={out}",
        f"RUN={alias} {'' if side == 'both' else f'SIDE={side} '}SEQ=dir:s TOOL=0 LOBJ=0 NE={ne} MI=1600 MB=16 "
        "TIGHTEN=0.5 NOCOL=0 ROBOT_Z=0",
    ]
    man += [] if side == "both" else ["TRAIN_EXTRA="]
    man += [
        "COLL_PRESET=canon_r2r3off",
        "RESET_SCHEME=or PSA_FLAGS=x",
        f"EXTRA={E.get('FAKE_EXTRA', '--arm rby1 --hand h')} LGATE_RAMP=0.10,180",
        f"DEXENGINE_FINGER_EFFORT_SCALE={E.get('FAKE_SCALE', '0.603')}",
    ]
    (ref / f"{out}_cmd.txt").write_text("\n".join(man) + "\n")
    rate = f"{0.5 + it / 10000:.4f}"
    sides = ["right", "left"] if side == "both" else [side]
    log = [
        f"[EVAL_LAUNCH] ctx=fresh · DEXENGINE_EVAL_PARK={E.get('FAKE_PARK', '1(default)')} · "
        "DEXENGINE_EVAL_EXIT_RESET=0",
        f"[FROZEN_REPLAY] resume_from=x iteration=0 eval_evals_per_clip(requested)={epc} num_envs={ne}",
        "[HAND_RECIPE] train: checkpoint ledger matches · trained switch=1(default) <-> now switch=1(default) · x",
    ]
    if side == "both":
        log.append("[bimanual] --bimanual 1: remapping gym id a -> b")
    log += [
        f"[FINGER_PENALTY] path=frozen_replay span=derived_norm side={s} dims=(6,26) n=20 · x"
        for s in sides
        if s != E.get("FAKE_FINGER_DROP")
    ]
    log.append("[HAND_RECIPE] create_env: hand=h path=train check PASS · x")
    log.append(f"[JUDGE_TERM] contact_term={E.get('FAKE_JTERM', '0')}(judge standard — x)")
    log.append(
        "[ROSTER] hand=h axis=a source=ledger:x universe=64 trees=64 mode=list:x → "
        f"train {k} (trees left out of the folder 0) "
        f"sha256={E.get('FAKE_ROSTER_SHA', '0')}"
    )
    log.append(
        f"    [FROZEN EVAL iter 0] 1 evals/clip x {k} clips physics=canon_r2r3off eps_per_clip=1 "
        f"contact_term={E.get('FAKE_CT', '0')} cross_pairs=ON artic_theta=0 warmup_batches={warm}"
    )
    log.append(
        f"    [C41][static] fixed-env clip collection: real batches {nreal} (+warmup {warm}), "
        f"{mn}~{mn} eps/batch per clip, target ≥{epc}"
    )
    bad = " — INVALID(p99 after exclusion > 10 mm)" if E.get("FAKE_BAD_PARK") else ""
    for b in range(1 - warm, nreal + 1):
        log.append("  Completion rate:     80.0%  (1/1)")
        log.append(f"[EVAL_PARK_CUT] batch {b}: parked-hand touch (at the cut) R 1/9 · L 0/9")
        log.append(
            f"[EVAL_PARK] batch {b}: parked 9/9 (max 9) · lifted-object free fall 0 · "
            "free fall over (this batch start) "
            f"R +0 / L +0 (gate > 0.09){bad}"
        )
    tot = int(E.get("FAKE_TOT") or nreal * mn)
    log += [
        "  AGGREGATE EVALUATION REPORT",
        f"  Clips evaluated:     {k}",
        "  Completion rate:     70.0%",
        "",
        "  Per-clip breakdown:",
    ]
    return k, log, rate, ref, tot


if __name__ == "__main__":
    sys.exit(main())
