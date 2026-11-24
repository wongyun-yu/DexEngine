#!/usr/bin/env python3
"""Stage 1 — MANO (human hand) → floating robot hand retargeting of TACO (MuJoCo IK).

In   : data/taco/raw_unzipped/ (TACO source) + a clip (base) list + the lift input (read-only, the TACO source tree
       ``layout.taco_source_root`` — the only TACO object poses).
Out  : <DS>/processed/taco/<hand spec>/bimanual/<clip>/0/trajectory_kinematic.npz — the input to stage 2.
       <DS> = <data root>/data/taco_retarget (canonical); raw/taco, raw/taco_maniptrans and processed/taco/{mano,assets,
       <hand>} all live under <DS>.
Done : count match — clips in the list == trajectory_kinematic.npz count (``dexengine.retarget.run`` writes DONE).

Usage: python scripts/stage1_retarget.py --bases <clips.txt> --arm rby1 --hand H --dataset taco --emb bimanual
         [--robot-phases-only] [--ik-force] [--out-root DIR | --variant SFX] [--log-dir DIR] [--data-root DIR]
  --bases             base clip names, one per line (may contain spaces). Required (rc 2 without it)
  --robot-phases-only skip the shared steps (lift … detect-contact) and run only the per-hand steps (for resuming)
  --ik-force          force the ik step to re-run on every clip
  --out-root DIR      isolated output: <DS>=DIR, a relative DIR under the run tree (writes nothing into the
                      canonical data/taco_retarget);
                      log /tmp/f32_batch_<parent>_<name> unless --log-dir
  --variant SFX       isolated output, short form: <DS>=<run tree>/overfit/stage1<SFX>/taco_retarget, log
                      /tmp/f32_batch<SFX> unless --log-dir
Only --dataset taco --emb bimanual are accepted (TACO source → bimanual staging). No GPU needed (CPU MuJoCo), but
several minutes per clip. rc 0 = DONE · 1 = not DONE · 2 = bad input. The closing npz count uses the list names as
they are (the DONE verdict uses the canonical names).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this tree's package (an entry script, not a module)

from dexengine.core import arms, config, hands, mounts  # noqa: E402
from dexengine.core import paths as P  # noqa: E402
from dexengine.retarget import run as R  # noqa: E402


def registry_check(arm: str, hand: str) -> bool:
    """The arm, the hand and their mount are registered; prints the registry line (with the provisional-mount mark)."""
    bad = [x for x, reg in ((arm, arms.ARM_REGISTRY), (hand, hands.HAND_REGISTRY)) if reg.get(x) is None]
    m = mounts.MOUNT_REGISTRY.get((arm, hand))
    if bad or m is None:
        print(f"[pipeline] not registered: {bad or (arm, hand)}", file=sys.stderr)
        return False
    print(
        f"[pipeline] registry OK: arm={arm} hand={hand} mount={m.ee_to_wrist_offset}"
        + (
            " ★ MOUNT_PROVISIONAL — this run's outputs use a provisional mount"
            if getattr(m, "provisional", False)
            else ""
        ),
        flush=True,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--robot-phases-only", action="store_true")
    ap.add_argument("--ik-force", action="store_true")
    iso = ap.add_mutually_exclusive_group()
    iso.add_argument("--out-root", type=Path, default=None)
    iso.add_argument("--variant", default=None)
    ap.add_argument("--log-dir", type=Path, default=None)
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)

    if not registry_check(a.arm, a.hand):
        return 2
    if a.dataset != "taco" or a.emb != "bimanual":
        print(
            f"[stage1] stage 1 accepts only --dataset taco --emb bimanual (TACO source → bimanual staging); got "
            f"dataset={a.dataset} emb={a.emb}. Other datasets: python -m dexengine.retarget.reretarget",
            file=sys.stderr,
        )
        return 2
    data_root = P.main_root()
    py = P.retarget_py()
    for what, path in (
        ("retargeting python (RETARGET_PY)", Path(py)),
        ("base list", a.bases),
        ("TACO source", data_root / "data/taco/raw_unzipped"),
    ):
        if not path.exists():
            print(f"[pipeline] {what} missing: {path}", file=sys.stderr)
            return 2
    robot = P.hand_spec_dir(a.hand)
    if a.out_root is not None:
        ds = P.run_out(a.out_root)  # relative = under the run tree (paths.run_root_of), never the cwd
        log = a.log_dir or Path(f"/tmp/f32_batch_{ds.parent.name}_{ds.name}")
        ds.mkdir(parents=True, exist_ok=True)
    elif a.variant is not None:
        ds = P.run_out(Path("overfit") / f"stage1{a.variant}" / "taco_retarget")
        log = a.log_dir or Path(f"/tmp/f32_batch{a.variant}")
        ds.mkdir(parents=True, exist_ok=True)
    else:
        ds, log = data_root / "data" / "taco_retarget", a.log_dir or R.DEFAULT_LOG_DIR
    clips = R.list_clips(a.bases)
    print(f"[stage1] bases={a.bases} ({len(clips)} clips) robots={robot} py={py} ds={ds} log={log}", flush=True)
    status, rc = R.taco_batch(
        a.bases,
        [robot],
        data_root=data_root,
        stage_dir=ds,
        log_dir=log,
        robot_phases_only=a.robot_phases_only,
        ik_force=a.ik_force,
        python=py,
    )
    n = sum((ds / "processed/taco" / robot / "bimanual" / c / "0/trajectory_kinematic.npz").is_file() for c in clips)
    print(
        f"[stage1] STATUS={status} rc={rc} · npz {n}/{len(clips)} · outputs {ds}/processed/taco/{robot}/bimanual/",
        flush=True,
    )
    return 0 if status == "DONE" else 1


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("scripts.stage1_retarget")
    sys.exit(main())
