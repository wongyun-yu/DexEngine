"""Convert the GR00T mid-training corpus (one call for the sample and the full corpus; only the arguments differ).

  python -m experiments.policies.mid.convert_corpus --manifest <manifest.tsv> --roots orig=<dir>,aug=<dir>
      --out <dataset dir> --ffmpeg <bin> --limits <limits.json> --git-commit <sha> [--jobs 8] [--timeout-s N]

Recipe (convert.py): --combo rby1_wuji2, --action-shift 1, --action-clip-limits <URDF limits>, --allow-axis-none (rows
without a task axis get axis_index -1 and an all-zero environment_state; GR00T does not read the axis). numpy's
transparent-huge-page request is turned off (NUMPY_MADVISE_HUGEPAGE=0 — an allocation hint only; the bytes are the
same) because direct compaction stalled finalize on large arrays. Last line `[MIDCONV] DONE …` or `[MIDCONV] FAIL …`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ["--combo", "rby1_wuji2", "--action-shift", "1", "--allow-axis-none"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--manifest", "--roots", "--out", "--ffmpeg", "--limits", "--git-commit"):
        ap.add_argument(k, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=86400)
    a = ap.parse_args(argv)
    t0 = time.time()
    log = Path(f"{a.out}.log")
    cmd = [
        "timeout",
        str(a.timeout_s),
        sys.executable,
        "-m",
        "experiments.policies.data.convert",
        "--manifest",
        a.manifest,
        "--roots",
        a.roots,
        *RECIPE[:2],
        "--ffmpeg",
        a.ffmpeg,
        "--git-commit",
        a.git_commit,
        *RECIPE[2:4],
        "--action-clip-limits",
        a.limits,
        *RECIPE[4:],
        "--jobs",
        str(a.jobs),
        "--out",
        a.out,
    ]
    with open(log, "w") as lf:
        rc = subprocess.run(
            cmd, cwd=ROOT, env=dict(os.environ, NUMPY_MADVISE_HUGEPAGE="0"), stdout=lf, stderr=subprocess.STDOUT
        ).returncode
    txt = log.read_text(errors="replace")
    with open(a.manifest) as fh:
        n = sum(1 for _ in fh) - 1
    axis = next((ln for ln in txt.splitlines() if ln.startswith("[AXIS_NONE_OK] n=")), "")
    if not axis:
        print("[MIDCONV] FAIL no [AXIS_NONE_OK] line — void")
        return 1
    done = next((ln for ln in txt.splitlines() if f"[E5_CONVERT] DONE episodes={n} " in ln), "")
    if not done:
        print(f"[MIDCONV] FAIL rc {rc} (see {log})")
        return 1
    print(f"[MIDCONV] DONE {int(time.time() - t0)} s · {axis[:80]} · {done[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
