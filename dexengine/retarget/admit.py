"""Stage 1 for one dataset and hand, retargeting through admission: retarget → stage manifest → intake → stage stamps →
remove the stage links. Each step is a process whose marker must say it did what was asked; the first step
that fails or lacks its marker stops the chain with that step's rc.

  retarget   (only with --retarget) `python -m dexengine.retarget.reretarget <ds> [WORKERS] --no-manifest [--scope]`
             rc 0 and the marker `[reretarget] scope ds=<ds>` (rc 8 when the marker names another scope)
  manifest   `reretarget <ds> --manifest-only` — the stage manifest of that dataset alone, right before intake
             (marker `… (manifest only)`; rc 5)
  intake     `python -m dexengine.data.intake <stage>/processed --hands <spec> --datasets <ds> --ingest
             --handoff-id reretarget_<hand> [--supersede-label L]` — the one way into the canonical layout; with a
             label an earlier generation of this hand is moved aside, proven by `[INTAKE_SUPERSEDE]` (rc 9 without it)
  stamps     `<stage>/processed/<ds>/.stage1_rig_sha256` and `.stage1_prior_sha256` (stage1.plan): the rig and prior
             the stage was solved and admitted with — written only after intake (rc 5 on a write failure). Values:
             --rig-stamp / --prior-stamp (a caller that planned them), else the identities stage1.plan.emit_rig
             reports now (paths.hand_rig_sha · plan.prior_identity of the hand's spec).
  unstage    `reretarget <ds> --unstage` (marker `… (unstage links)`; rc 5)
TACO has no stage: retarget (optional) and stamps only. Logs: <log dir>/<ds>_1<step>.log (the names the stage-1
campaign driver reads). Marker: `[ADMIT] ds=<ds> hand=<hand> steps <k>/<n> rc <rc>`.

Usage:
  python -m dexengine.retarget.admit <ds> --hand <hand> [--retarget [--workers N] [--scope FILE]]
      [--supersede-label L] [--rig-stamp SHA --prior-stamp SHA] [--log-dir DIR] [--data-root DIR]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dexengine.core import config
from dexengine.core.repo import REPO_ROOT

TAG = "[ADMIT]"
STEPS = ("retarget", "manifest", "intake", "stamps", "unstage")


def _run(name: str, argv: list[str], log: Path, hand: str) -> tuple[int, str]:
    extra = os.environ.get("PYTHONPATH")
    env = {**os.environ, "DEXENGINE_HAND": hand, "PYTHONPATH": str(REPO_ROOT) + (os.pathsep + extra if extra else "")}
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(
            [sys.executable, *argv], stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT
        ).returncode
    print(f"{TAG} step {name} rc={rc} log={log}", flush=True)
    return rc, log.read_text(encoding="utf-8", errors="replace")


def _marked(text: str, line_start: str) -> bool:
    return any(ln.startswith(line_start) for ln in text.splitlines())


def plan(ds: str, retarget: bool) -> list[str]:
    """The steps for one dataset (TACO: no stage — retarget and stamps only)."""
    steps = [s for s in STEPS if retarget or s != "retarget"]
    return [s for s in steps if s in ("retarget", "stamps")] if ds == "taco" else steps


def admit(
    ds: str,
    hand: str,
    log_dir: Path,
    *,
    retarget: bool = False,
    workers: str = "",
    scope: str = "",
    label: str = "",
    rig: str = "",
    prior: str = "",
) -> int:
    from dexengine.core import paths as P  # noqa: PLC0415 — after the data root is set
    from dexengine.core.locations.hands import hand_spec_dir  # noqa: PLC0415
    from dexengine.data.stage1 import plan as SP  # noqa: PLC0415
    from dexengine.retarget import reretarget_stage as RT  # noqa: PLC0415 — owner of the reretarget markers

    steps, done, rc = plan(ds, retarget), 0, 0
    log_dir.mkdir(parents=True, exist_ok=True)
    rr = ["-m", "dexengine.retarget.reretarget", ds]
    stage, spec = P.retarget_stage_root(hand) / "processed", hand_spec_dir(hand)
    for st in steps:
        log = log_dir / f"{ds}_1{st}.log"
        if st == "retarget":
            rc, out = _run(
                st,
                [*rr, *([workers] if workers else []), "--no-manifest", *(["--scope", scope] if scope else [])],
                log,
                hand,
            )
            if rc == 0 and not _marked(out, f"{RT.SCOPE}{ds} "):
                rc = 8
        elif st == "manifest":
            rc, out = _run(st, [*rr, "--manifest-only"], log, hand)
            rc = rc or (0 if _marked(out, f"{RT.SCOPE}{ds} {RT.MANIFEST_ONLY}") else 5)
        elif st == "intake":
            rc, out = _run(
                st,
                [
                    "-m",
                    "dexengine.data.intake",
                    str(stage),
                    "--hands",
                    spec,
                    "--datasets",
                    ds,
                    "--ingest",
                    "--handoff-id",
                    f"reretarget_{hand}",
                    *(["--supersede-label", label] if label else []),
                ],
                log,
                hand,
            )
            if rc == 0 and label and not _marked(out, "[INTAKE_SUPERSEDE] "):
                rc = 9
        elif st == "stamps":
            if not (rig and prior):  # the identities emit_rig reports (registry · prior)
                rig, prior = P.hand_rig_sha(spec), SP.prior_identity(spec)
            try:
                d = stage / ds
                d.mkdir(parents=True, exist_ok=True)
                (d / SP.RIG_STAMP_NAME).write_text(f"{rig}\n")
                (d / SP.PRIOR_STAMP_NAME).write_text(f"{prior}\n")
                log.write_text(f"{d / SP.RIG_STAMP_NAME} {rig}\n{d / SP.PRIOR_STAMP_NAME} {prior}\n", encoding="utf-8")
                print(f"{TAG} step stamps rig={rig[:12]} prior={prior[:12]} → {d}", flush=True)
            except OSError as e:
                print(f"{TAG} ✗ stamps: {e}", file=sys.stderr, flush=True)
                rc = 5
        elif st == "unstage":
            rc, out = _run(st, [*rr, "--unstage"], log, hand)
            rc = rc or (0 if _marked(out, f"{RT.SCOPE}{ds} {RT.UNSTAGE} ") else 5)
        if rc:
            print(f"{TAG} ✗ step {st} rc={rc} — see {log}", file=sys.stderr, flush=True)
            break
        done += 1
    print(f"{TAG} ds={ds} hand={hand} steps {done}/{len(steps)} rc {rc}", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dexengine.retarget.admit",
        description=__doc__.split("\n")[0],
        epilog=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset")
    ap.add_argument("--hand", required=True, help="hand id (DEXENGINE_HAND of every step)")
    ap.add_argument("--retarget", action="store_true", help="run the retargeting step first")
    ap.add_argument("--workers", default="", help="reretarget WORKERS (with --retarget)")
    ap.add_argument("--scope", default="", help="reretarget --scope FILE (with --retarget)")
    ap.add_argument("--supersede-label", default="", help="intake --supersede-label (move an earlier generation aside)")
    ap.add_argument("--rig-stamp", default="", help="rig sha to stamp (default: stage1.plan.emit_rig now)")
    ap.add_argument("--prior-stamp", default="", help="prior identity to stamp (default: stage1.plan.emit_rig now)")
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/stage1_admit"),
        help="step logs (relative = under the run tree, paths.run_out — not the cwd)",
    )
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    if (a.workers or a.scope) and not a.retarget:
        ap.error("--workers and --scope belong to --retarget")
    if bool(a.rig_stamp) != bool(a.prior_stamp):
        ap.error("--rig-stamp and --prior-stamp go together")
    from dexengine.core.paths import run_out  # noqa: PLC0415

    return admit(
        a.dataset,
        a.hand,
        run_out(a.log_dir),
        retarget=a.retarget,
        workers=a.workers,
        scope=a.scope,
        label=a.supersede_label,
        rig=a.rig_stamp,
        prior=a.prior_stamp,
    )


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.admit")
    sys.exit(main())
