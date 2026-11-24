"""The plan step of run.py (rules.PLAN_JOBS plan.py pieces merged into plan/plan.json) and the join-floor-off re-plan
that the roll step calls; also the [AUG_V4_RUN] marker printer both modules use."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dexengine.record.augment import stage as AC
from experiments.real_kit.v4 import rules as V4


def say(*x):
    print("[AUG_V4_RUN]", *x, flush=True)


def _launch_pieces(parts, out, d, x, info, floor_off, env):
    """Start one plan process per piece (its own folder and log, time cap by its variant count)."""
    procs = []
    for i, names in enumerate(parts):
        po, lg = out / f"part{i}", d / f"plan.part{i}.log"
        if po.exists():
            shutil.rmtree(po)
        tmo = V4.PLAN_BASE_S + V4.PLAN_S_PER_VARIANT * len(names)
        f = open(lg, "w")  # noqa: SIM115 — open until the piece process ends (closed after wait below)
        procs.append(
            (
                po,
                lg,
                f,
                subprocess.Popen(
                    [
                        "timeout",
                        "-k",
                        "60",
                        str(tmo),
                        sys.executable,
                        "experiments/real_kit/v4/plan.py",
                        "--shadow",
                        str(x.shadow),
                        "--dataset",
                        info["ds"],
                        "--emb",
                        info["emb"],
                        "--names",
                        ",".join(names),
                        "--out",
                        str(po),
                        *(["--join-floor-off"] if floor_off else []),
                    ],
                    cwd=AC.ROOT,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                ),
            )
        )
    return procs


def _wait_pieces(procs, d, floor_off):
    """Wait for every piece, join the piece logs into plan.log, and die when a piece failed."""
    bad = []
    for _po, lg, f, p in procs:
        rc = p.wait()
        f.close()
        if rc or "[AUG_V4_PLAN] DONE" not in lg.read_text(errors="replace"):
            bad.append(f"{lg.name} rc {rc}")
    with open(d / "plan.log", "a" if floor_off else "w") as g:  # a re-plan appends after the earlier record
        for _po, lg, _f, _p in procs:
            g.write(lg.read_text(errors="replace"))
    if bad:
        raise SystemExit(f"[AUG_V4_RUN] plan pieces failed {bad} — {d}/plan.part*.log")


def step_plan(a, x, okn: list, floor_off: bool = False) -> int:
    """plan — split the variants into pieces (rules.PLAN_JOBS) run in parallel, merged into one plan/plan.json. Piece
    time cap = PLAN_BASE_S + variants × PLAN_S_PER_VARIANT. Merged name set = given name set; variant npz files move to
    plan/ with their paths rewritten (same output as one process); piece logs are joined into plan.log."""
    d, info = x.d, x.info
    out = d / "plan"
    out.mkdir(parents=True, exist_ok=True)
    J = max(1, min(V4.PLAN_JOBS, len(okn)))
    parts = [okn[i::J] for i in range(J)]
    env = dict(x.env, OMP_NUM_THREADS=str(V4.PLAN_THREADS), MKL_NUM_THREADS=str(V4.PLAN_THREADS))
    t0 = time.time()
    procs = _launch_pieces(parts, out, d, x, info, floor_off, env)
    _wait_pieces(procs, d, floor_off)
    pj = out / "plan.json"
    plan = json.loads(pj.read_text()) if pj.is_file() else {}
    got = []
    for po, _lg, _f, _p in procs:
        for n, rec in json.loads((po / "plan.json").read_text()).items():
            if rec.get("npz"):
                src = Path(rec["npz"])
                if src.parent != po:
                    raise SystemExit(f"[AUG_V4_RUN] {n} npz {src} is outside the piece folder {po}")
                os.replace(src, out / src.name)
                rec["npz"] = str(out / src.name)
            plan[n] = rec
            got.append(n)
    if sorted(got) != sorted(okn):
        raise SystemExit(f"[AUG_V4_RUN] plan pieces total {len(got)} ≠ variants given {len(okn)}")
    pj.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    for po, lg, _f, _p in procs:
        shutil.rmtree(po)
        lg.unlink()
    say(f"plan DONE {a.clip} · variants {len(okn)} · pieces {J} · floor off {floor_off} · {time.time() - t0:.0f} s")
    return 0


def replan(a, x, okn: list, stat: dict) -> list:
    """Re-plan (rules.REPLAN_FLOOR_OFF): variants with 0 successes at the attempt cap → plan again with the join floor
    off
    (join at the reference speed), once per variant; no schedule → the earlier schedule is restored, the variant
    failed."""
    if not V4.REPLAN_FLOOR_OFF:
        return []
    names = [n for n in okn if not stat[n]["post_ok"] and stat[n]["successes"] == 0 and stat[n]["replan"] is None]
    if not names:
        return []
    pj = x.d / "plan" / "plan.json"
    say(
        f"{V4.REPLAN_TAG} {a.clip} {V4.RETRY_MAX_ATTEMPTS} attempts, 0 successes → join floor off (join at "
        "the reference speed), "
        f"re-plan · {names}"
    )
    old = json.loads(pj.read_text())
    step_plan(a, x, names, floor_off=True)
    new, got = json.loads(pj.read_text()), []
    for n in names:
        if new[n].get("fail"):
            stat[n]["replan"] = {"floor_off": True, "fail": new[n]["fail"]}
            new[n] = old[n]
            continue
        stat[n]["replan"] = {
            "floor_off": True,
            "K": new[n]["K"],
            "join_v": {s: sd.get("join_v") for s, sd in new[n]["sides"].items()},
        }
        stat[n]["cand_attempts"] = 0
        got.append(n)
    pj.write_text(json.dumps(new, indent=1, ensure_ascii=False))
    say(f"{V4.REPLAN_TAG} {a.clip} new schedules {len(got)}/{len(names)} · {[(n, stat[n]['replan']) for n in names]}")
    return got
