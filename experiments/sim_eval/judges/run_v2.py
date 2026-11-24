"""Apply the v2 task judge to every scene of one batch → `<run>/<mode>/judge_v2.json` per scene.

The v2 rules (Lift `all_zm_k5`, Sweep against the data episode with both-hand tool hold) are pinned by md5: the rules
file is data whose sha256 is stored in every judge record. Each record gets `v2_marks` = rules sha256, judge and v1
judge md5, the sha256 of the instruction text map the session actually used (from the policy JSON) and whether the
session ran with job text (`[E5_POLICY_JOB_TEXT] on` in its log).

Log lines (one per scene, then a total), kept in the judge log of the batch:
`[OVFJ] <scene> <axis> judgeable <b> success <b> invalid <x> why <w> v2 text|notext`, `[OVFJ] <scene> no_episode`,
`[OVFJ] <scene> judge_fail <last log line>`, `[OVFJ] DONE n=<judged>/<listed>`.
CLI: `python -m experiments.sim_eval.judges.run_v2 --batch B --mode M --list <job list> --runs-root R --policy-json P
      --data-root DR --home-map HM --scratch <dir>`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES_V2 = HERE / "rules_v2.yaml"
#: md5 of the v2 rules file (the rules are data; a different file is a different judge). The paper evaluation's
#: deployed copy had md5 dcce6f92af41f89d62a12f79f12cf33a; this copy differs only in comments and status text (yaml
#: data equal otherwise). No reader compares a stored rules hash with this file: records only carry it.
RULES_V2_MD5 = "41f7f4ca611ad11e3e3bb8f1edc689a6"
#: the frozen v1 judge, hashed as one (frozen_md5 = md5 of their bytes in this order, first 8 hex).
V1_FILES = ("judge_v1.py", "judge_v1_pair.py")


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def text_map_of(policy_json: Path, batch: str) -> Path | None:
    """Instruction text map named by the batch's policy JSON (None when the policy ran without one)."""
    p = json.loads(policy_json.read_text())["policies"][batch]
    t = p.get("instructions") or ""
    return Path(t) if t else None


def _run_judge(scratch, batch, sc, run, mode, data_root, home_map, root, timeout_s):
    """Run the judge on one scene in a child process, its log next to the run."""
    tmp = scratch / f"{batch}_{sc}.json"
    with open(run / mode / "judge_v2.log", "w") as lf:
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.sim_eval.judges.judge_v2",
                "--run",
                str(run),
                "--mode",
                mode,
                "--rules",
                str(RULES_V2),
                "--data-root",
                data_root,
                "--home-map",
                home_map,
                "--out",
                str(tmp),
            ],
            cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1"),
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    return tmp, p


def _store_record(tmp, rules_sha, m2, mf, ts, jt, run, mode, log, sc, out, n):
    """Add the v2 marks, write judge_v2.json atomically and log the verdict."""
    d = json.loads(tmp.read_text())
    d["v2_marks"] = {
        "rules_sha256": rules_sha,
        "judge_md5": m2,
        "frozen_md5": mf,
        "text_map_sha256": ts,
        "session_job_text": jt,
    }
    dst = run / mode / "judge_v2.json"
    part = dst.with_name(dst.name + ".tmp")
    part.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))
    os.replace(part, dst)
    tmp.unlink()
    why = str(d.get("judgeable_why") or d.get("why") or "-").replace(" ", "_")[:80]
    log(
        f"[OVFJ] {sc} {d.get('axis')} judgeable {d.get('judgeable')} success {d.get('success')} "
        f"invalid {d.get('invalid')} "
        f"why {why} v2 {'text' if jt else 'notext'}"
    )
    out[sc] = d
    n += 1
    return n


def judge_batch(
    batch: str,
    mode: str,
    jobs: list[str],
    runs_root: Path,
    policy_json: Path,
    data_root: str,
    home_map: str,
    scratch: Path,
    log=print,
    timeout_s: int = 1800,
) -> dict:
    """Judge every listed scene of a batch → {scene: record | "no_episode" | "judge_fail: <why>"}."""
    if _md5(RULES_V2) != RULES_V2_MD5:
        raise SystemExit(f"[OVFJ2] rules {RULES_V2} md5 {_md5(RULES_V2)} ≠ pinned {RULES_V2_MD5}")
    rules_sha = hashlib.sha256(RULES_V2.read_bytes()).hexdigest()
    m2, mf = (
        _md5(HERE / "judge_v2.py"),
        hashlib.md5(b"".join((HERE / f).read_bytes() for f in V1_FILES)).hexdigest()[:8],
    )
    tmap = text_map_of(policy_json, batch)
    if tmap is not None and not tmap.is_file():
        raise SystemExit(f"[OVFJ2] text map {tmap} named by the policy JSON not found")
    ts = hashlib.sha256(tmap.read_bytes()).hexdigest() if tmap else "none"
    log(
        f"[OVFJ2] v2 judge md5 {m2} · rules sha256 {rules_sha} · v1 md5 {mf} · text map {tmap or '-'} sha256 {ts} · "
        f"batch {batch} · mode {mode}"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    out, n = {}, 0
    for job in jobs:
        sc = os.path.basename(os.path.dirname(job))
        run = runs_root / f"{batch}_{sc}"
        if not (run / mode / "episode.npz").is_file():
            log(f"[OVFJ] {sc} no_episode")
            out[sc] = "no_episode"
            continue
        try:
            jt = any(
                ln.startswith("[E5_POLICY_JOB_TEXT] on")
                for ln in (run / "session.log").read_text(errors="replace").splitlines()
            )
        except OSError:
            jt = False
        tmp, p = _run_judge(scratch, batch, sc, run, mode, data_root, home_map, root, timeout_s)
        if p.returncode != 0:
            last = [x for x in (run / mode / "judge_v2.log").read_text(errors="replace").splitlines() if x.strip()]
            why = (last[-1] if last else "")[:200]
            log(f"[OVFJ] {sc} judge_fail {why}")
            out[sc] = f"judge_fail {why}"
            tmp.unlink(missing_ok=True)
            continue
        n = _store_record(tmp, rules_sha, m2, mf, ts, jt, run, mode, log, sc, out, n)
    log(f"[OVFJ] DONE n={n}/{len(jobs)}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--batch", "--mode", "--list", "--runs-root", "--policy-json", "--data-root", "--home-map", "--scratch"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    with open(a.list) as fh:
        jobs = [x.strip() for x in fh if x.strip()]
    judge_batch(a.batch, a.mode, jobs, Path(a.runs_root), Path(a.policy_json), a.data_root, a.home_map, Path(a.scratch))


if __name__ == "__main__":
    main(sys.argv[1:])
