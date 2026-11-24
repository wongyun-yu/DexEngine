"""Training time of one RL run, measured while it runs: s/iter from the tqdm samples of its own train.log.

Runs the training command given after ``--`` and samples the newest ``<ref>/<run>*train.log`` every STEP_S s (the tqdm
stamp ``n/total [h:mm:ss<`` → iterations = n ÷ (total ÷ MI) · elapsed s). It reports (a) the mean s/iter (first
iteration included), (b) the steady s/iter (the first WARM fraction of the iterations left out), (c) minutes per 100
iterations. Used for the floating-hand Ours row's speed ratio (the L40S short run, ``ratio_entries.ours_na_entry``).
Outputs: <out>/progress_<run>.json (latest sample, rewritten while the run goes) · <out>/time_<run>.json (result) ·
the line ``[E5NA_TIME] run=<run> arm=<arm> rc=<rc> iters=<n>/<MI> s_per_iter=<mean> s_per_iter_steady=<steady>
min_per_100=<min> wall_s=<wall>`` (job scripts grep it; steady=None with fewer than two samples).
usage (in the training job, from the tree root):
       python3 experiments/scoring/cost/na_time.py --run <run> --mi <MI> --out <dir> [--arm <arm>]
       -- <training command…>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

TAG = "[E5NA_TIME]"
STEP_S = 30
WARM = 0.2  # the first 20 % of the iterations are left out of the steady value (compile · first reset)
RATE_SAMPLES = 3  # s_per_iter_recent spans this many samples
_TQDM = re.compile(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<")
_RUN_URL = re.compile(r"View run\b.*? at:?\s*(https?://\S+)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class Progress:
    """The tqdm progress of one run's newest train.log (the same sample rule as the training jobs)."""

    def __init__(self, run: str, mi: int, ref: Path) -> None:
        self.run, self.mi, self.ref = run, mi, ref
        self.t0, self.hist, self.url = time.time(), [], None

    def sample(self) -> dict | None:
        logs = sorted(glob.glob(str(self.ref / f"{glob.escape(self.run)}*train.log")), key=os.path.getmtime)
        if not logs:
            return None
        with open(logs[-1], "rb") as fh:
            if self.url is None:
                hit = _RUN_URL.search(_ANSI.sub("", fh.read(1 << 18).decode("utf-8", "replace")))
                self.url = hit.group(1) if hit else None
            fh.seek(max(0, os.path.getsize(logs[-1]) - 65536))
            hits = _TQDM.findall(fh.read().decode("utf-8", "replace"))
        if not hits:
            return None
        n, tot, h, m, s = hits[-1]
        el = int(h or 0) * 3600 + int(m) * 60 + int(s)
        it = int(n) / (int(tot) / self.mi)
        self.hist = (self.hist + [(it, el)])[-(RATE_SAMPLES + 1) :]
        (i0, e0), recent = self.hist[0], None
        if it > i0:
            recent = round((el - e0) / (it - i0), 3)
        return {
            "run": self.run,
            "log": logs[-1],
            "timesteps": int(n),
            "total": int(tot),
            "iters": round(it, 1),
            "mi": self.mi,
            "tqdm_s": el,
            "s_per_iter": round(el / it, 3) if it else None,
            "s_per_iter_recent": recent,
            "recent_s": el - e0 if recent is not None else None,
            "wandb_url": self.url,
            "pod_s": round(time.time() - self.t0),
            "at": time.strftime("%FT%T"),
            "at_epoch": round(time.time(), 1),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--mi", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="", help="arm label written into the result")
    ap.add_argument("--ref", type=Path, default=Path("overfit/logs/refbase"))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no training command (after --)")
    a.out.mkdir(parents=True, exist_ok=True)
    pr = Progress(a.run, a.mi, a.ref)
    samples: list[tuple[float, float]] = []
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(STEP_S):
            try:
                r = pr.sample()
            except Exception as e:  # noqa: BLE001 — the log is being moved: try again at the next sample
                print(TAG, "sample failed", type(e).__name__, str(e)[:200], flush=True)
                continue
            if r and (not samples or r["iters"] > samples[-1][0]):
                samples.append((r["iters"], r["tqdm_s"]))
                atomic_json(a.out / f"progress_{a.run}.json", r)

    t0 = time.time()
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    rc = subprocess.run(cmd).returncode
    stop.set()
    th.join(timeout=5)
    try:
        last = pr.sample()
    except Exception:  # noqa: BLE001
        last = None
    if last and (not samples or last["iters"] > samples[-1][0]):
        samples.append((last["iters"], last["tqdm_s"]))
    wall = round(time.time() - t0)
    it, el = samples[-1] if samples else (0.0, 0.0)
    mean = round(el / it, 3) if it else None
    steady = None
    warm = [s for s in samples if s[0] >= WARM * a.mi]
    if len(warm) >= 2 and warm[-1][0] > warm[0][0]:
        steady = round((warm[-1][1] - warm[0][1]) / (warm[-1][0] - warm[0][0]), 3)
    per100 = round((steady or mean or 0) * 100 / 60, 2) if (steady or mean) else None
    rec = {
        "run": a.run,
        "arm": a.arm,
        "rc": rc,
        "mi": a.mi,
        "iters": it,
        "tqdm_s": el,
        "s_per_iter": mean,
        "s_per_iter_steady": steady,
        "min_per_100": per100,
        "wall_s": wall,
        "samples": samples,
        "log": (last or {}).get("log"),
        "host": os.uname().nodename,
        "at": time.strftime("%FT%T"),
    }
    atomic_json(a.out / f"time_{a.run}.json", rec)
    print(
        f"{TAG} run={a.run} arm={a.arm} rc={rc} iters={it}/{a.mi} s_per_iter={mean} s_per_iter_steady={steady} "
        f"min_per_100={per100} wall_s={wall}",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
