"""Count one evaluation level (L0/L1/L2) of one checkpoint: one verdict per listed scene → `<LEVEL>.json`.

For each scene the last batch that ran it wins (a retry batch, tag ending `b_r1`, beats the first one). Verdicts:
- success / fail: the task judge (P&P: the P&P judge line of `pnp.jsonl`; Pour/Sweep: `judge_v2.json` success),
- invalid: the v2 judge flagged the rollout (start jump, blow-up, …); counted as not successful, denominator unchanged,
- unjudgeable: the judge could not decide (or no P&P line),
- judge failure: the session ended with rc 0 but left no judge record (reason from the judge log),
- session failure: no rc-0 session in the batch ledgers.

The verdict values and the P&P judge name are stored data read by the report tables (report/verdicts.JV).
CLI (the arguments of the original level counter):
`python -m experiments.sim_eval.levels <task> <list> <runs.txt> <pnp.jsonl> <out.json> <run> <step> <level> <model sha>
 <session failures after retry> <rc0.txt> <jfail.tsv> <session list>`
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

MARK = "[DPF_COUNT]"
# Stored verdict values (keys of `counts`, `verdict` of each scene) — data, not prose.
SUCCESS = "success"
FAIL = "fail"
INVALID = "invalid"
UNJUDGEABLE = "unjudgeable"
JUDGE_FAIL = "judge failed"
SESSION_FAIL = "session failed"
# Stored reason strings of the two "unfinished" verdicts and the missing P&P line.
WHY_NO_JUDGE_RECORD = "no judge record · no reason in judge.log"
WHY_NO_RC0 = "no rc 0 in the ledger"
WHY_NO_PNP = "no P&P verdict"
#: P&P judge name used when an old `pnp.jsonl` line has no `judge` field.
PNP_JUDGE_DEFAULT = "e5_judge_pnp_20261030"


def _scene_from_job(x: str) -> str:
    return os.path.basename(os.path.dirname(x.strip()))


def _read_judge_failures(jfail_tsv, norm):
    """Judge-failure reasons per scene (scene names normalised)."""
    jfail = {}
    with open(jfail_tsv) as fh:
        for x in fh:
            if "\t" in x:
                s, why = x.rstrip("\n").split("\t", 1)
                jfail[norm(s)] = why
    return jfail


def _last_runs(runs_txt):
    """Last run folder per scene (a `b_r1` rerun wins)."""
    last = {}
    with open(runs_txt) as fh:
        for d in [x.strip() for x in fh if x.strip()]:
            sc = d.split("_r1_", 1)[1]
            tag = os.path.basename(d).split("_r1_")[0] + "_r1"
            if sc not in last or tag.endswith("b_r1"):
                last[sc] = (tag, d)
    return last


def _read_pnp(task, pnp_jsonl, norm):
    """P&P verdict per scene and the judge names (lift only; one judge per batch)."""
    pnp, judges = {}, set()
    if task == "lift" and os.path.exists(pnp_jsonl):
        with open(pnp_jsonl) as fh:
            for ln in fh:
                r = json.loads(ln)
                if "error" not in r:
                    pnp[norm(r["scene"])] = r
                    judges.add(r.get("judge", PNP_JUDGE_DEFAULT))
    if len(judges) > 1:
        raise SystemExit(f"{MARK} pnp.jsonl names more than one judge {sorted(judges)} — one batch, one judge")
    return judges, pnp


def _scene_verdict(sc, last, rc0, jfail, task, pnp):
    """Verdict and reason of one listed scene."""
    if sc not in last:
        if sc in rc0:
            v, why = JUDGE_FAIL, jfail.get(sc, WHY_NO_JUDGE_RECORD)[:200]
        else:
            v, why = SESSION_FAIL, WHY_NO_RC0
    else:
        tag, d = last[sc]
        j = json.loads(Path(f"{d}/policy-{tag}/judge_v2.json").read_text())
        if not j.get("judgeable"):
            v, why = UNJUDGEABLE, str(j.get("judgeable_why"))[:120]
        elif j.get("invalid"):
            v, why = INVALID, str(j.get("invalid"))[:120]
        elif task == "lift":
            p = pnp.get(sc)
            if p is None:
                v, why = UNJUDGEABLE, WHY_NO_PNP
            else:
                v, why = (SUCCESS, "") if p["success"] else (FAIL, p.get("why") or "")
        else:
            v, why = (SUCCESS, "") if j.get("success") else (FAIL, "")
    return v, why


def count_level(
    task: str,
    list_path: str,
    runs_txt: str,
    pnp_jsonl: str,
    run: str,
    step: str,
    level: str,
    model_sha: str,
    session_fail: int,
    rc0_txt: str,
    jfail_tsv: str,
    session_list: str,
) -> dict:
    """Verdict per listed scene and totals (the `<LEVEL>.json` record)."""
    with open(rc0_txt) as fh:
        rc0 = {x.strip() for x in fh if x.strip()}
    with open(list_path) as fh:
        want = [_scene_from_job(x) for x in fh if x.strip()]
    wanted = set(want)

    def norm(s: str) -> str:
        # A scene in the list keeps its name (original placement `<clip>_pl_rt`); otherwise it is a variant (`.D`).
        if s in wanted:
            return s
        if s.endswith(".D") and s[:-2] in wanted:
            return s[:-2]
        return s if s.endswith(".D") else s + ".D"

    jfail = _read_judge_failures(jfail_tsv, norm)
    last = _last_runs(runs_txt)
    judges, pnp = _read_pnp(task, pnp_jsonl, norm)
    rows, cnt, per = [], collections.Counter(), collections.defaultdict(collections.Counter)
    for sc in want:
        v, why = _scene_verdict(sc, last, rc0, jfail, task, pnp)
        clip = sc.rsplit("_pv", 1)[0]
        rows.append({"scene": sc, "verdict": v, "why": why})
        cnt[v] += 1
        per[clip][v] += 1
    judge = f"P&P {(sorted(judges) or [PNP_JUDGE_DEFAULT])[0]}" if task == "lift" else f"v2 {task}"
    return {
        "run": run,
        "step": step,
        "level": level,
        "task": task,
        "session_list": session_list,
        "judge": judge,
        "model_sha": model_sha,
        "n": len(want),
        "counts": dict(cnt),
        "success": cnt[SUCCESS],
        "session_fail_after_retry": int(session_fail),
        "per_clip": {c: dict(v) for c, v in sorted(per.items())},
        "scenes": rows,
    }


def main(argv: list[str]) -> None:
    if len(argv) != 13:
        raise SystemExit(__doc__)
    task, lst, runs, pj, out, run, step, lv, hk, bad, rc0, jf, ls = argv
    res = count_level(task, lst, runs, pj, run, step, lv, hk, int(bad), rc0, jf, ls)
    with open(out, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"success {res['success']}/{res['n']} · {res['counts']}")


if __name__ == "__main__":
    main(sys.argv[1:])
