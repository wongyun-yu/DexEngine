"""Run or emit a job list.

  python -m dexengine.cluster run   --jobs jobs.jsonl --gpus 0,1 --log-dir logs/
  python -m dexengine.cluster slurm --jobs jobs.jsonl --name rl --log-dir logs/ --out rl.sbatch [--set partition=gpu]
  python -m dexengine.cluster k8s   --jobs jobs.jsonl --out rl.yaml [--set image=registry/dexengine:latest]

`slurm` and `k8s` read their settings from configs/cluster/{slurm,k8s}.yaml (overridable with --config / --set).
`run` exits 0 when every job exited 0, else 1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dexengine.cluster import k8s, runner, slurm
from dexengine.cluster.jobs import load_jobs
from dexengine.core import config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dexengine.cluster", description="Run or emit a job list.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run on local GPUs")
    r.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""), help="GPU ids, comma-separated")
    r.add_argument("--log-dir", required=True)
    s = config.add_cli(sub.add_parser("slurm", help="write a Slurm array script"))
    s.add_argument("--name", required=True)
    s.add_argument("--log-dir", required=True)
    k = config.add_cli(sub.add_parser("k8s", help="write Kubernetes Job manifests"))
    for p in (r, s, k):
        p.add_argument("--jobs", required=True, help="job list (JSON Lines)")
    for p in (s, k):
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    jobs = load_jobs(a.jobs)
    if a.cmd == "run":
        rcs = runner.run_local(jobs, [g for g in a.gpus.split(",") if g.strip()], a.log_dir)
        bad = {n: rc for n, rc in rcs.items() if rc != 0}
        print(f"[cluster] {len(rcs) - len(bad)}/{len(jobs)} jobs ok" + (f" · failed {bad}" if bad else ""))
        return 0 if not bad and len(rcs) == len(jobs) else 1
    if a.cmd == "slurm":
        text = slurm.render(
            jobs,
            str(Path(a.jobs).resolve()),
            a.name,
            a.log_dir,
            config.from_args(slurm.SlurmConfig, "cluster/slurm", a),
        )
    else:
        text = k8s.render(jobs, config.from_args(k8s.K8sConfig, "cluster/k8s", a))
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"[cluster] {a.cmd}: {len(jobs)} jobs → {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
