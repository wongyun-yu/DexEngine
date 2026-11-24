"""Steps of the judged-cell bundle runner (record.augment.cells, whose main orders them): the judge run (judge_script,
judge_env, run_judge and its time caps), the stage state of modes D and L (d_state, l_state), the cell choice
(run_pick) and the final count. Owns the runner marker MARK, the tree root ROOT and the die/say helpers the runner
shares."""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from dexengine.core import log_markers as LM
from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.core.repo import REPO_ROOT
from dexengine.placement.judge import aug_cells as JAC
from dexengine.placement.judge import input_record as JFP
from dexengine.record.augment import cells_cond as ACC
from dexengine.record.augment import cells_pick as ACP

MARK = "[AUG_CELLS_RUN]"


ROOT = REPO_ROOT  # tree root


#: time cap of one judge run = boot + per clip; measured (hocap 2 clips, RUN_DONE.json) boot ≈15 s · per clip D ≈2-3 s
#: · L ≈18-32 s, about 20 x
TIMEOUT_BOOT_S = 600


TIMEOUT_PER_CLIP_S = 600


def die(msg: str):
    raise SystemExit(f"{MARK} ✗ {msg}")


def say(*x) -> None:
    print(MARK, *x, flush=True)


# ── judge call
def judge_script(
    cond: dict, mode: str, bundle: Path, clips_file: Path, work: Path, chunk: str, timeout_s: int, mem: str, rsf=None
) -> str:
    """The same form (same order) as the stage2_ground.sh judge line + the judge mode arguments."""
    q = shlex.quote
    head = [
        "--arm",
        cond["arm"],
        "--hand",
        cond["hand"],
        "--clips",
        str(clips_file),
        "--dataset",
        cond["dataset"],
        "--emb",
        cond["emb"],
        "--src-layer",
        cond["src_layer"],
    ]
    tail = (
        [
            f"--dz-cells={cond['dz']}",
            "--pool-root",
            cond["pool"],
            "--active-sides",
            cond["active"],
            "--end-marks",
            str(work / "judged.tsv"),
            "--save-ik",
            str(work / "ikc"),
            "--out",
            str(work / "p0"),
            "--aug-cells-mode",
            mode,
            "--aug-cells-dir",
            str(bundle),
        ]
        + (["--aug-cells-chunk", chunk] if chunk else [])
        + (["--place-reselect", str(rsf)] if rsf else [])
    )
    return "\n".join(
        [
            "set -o pipefail",
            f"cd {q(str(ROOT))}",
            "source configs/placement/canon_env.sh || exit 2",
            ". scripts/_isaac_py.sh || exit 2",
            "PYI=$(isaac_py) || exit 2",
            f"exec bash scripts/capped.sh -m {q(mem)} -n augcells-{mode}-{os.getpid()}-{int(time.time())} -- "
            "nice -n 19 ionice -c3 "
            f'timeout -k 60 {int(timeout_s)} "$PYI" dexengine/placement/judge/core.py {" ".join(map(q, head))} '
            f"$CANON_JUDGE_FLAGS {' '.join(map(q, tail))}",
        ]
    )


def judge_env(cond: dict, gpu: str) -> dict:
    return dict(
        os.environ,
        DEXENGINE_MAIN_ROOT=str(P.main_root()),
        DEXENGINE_ARM=cond["arm"],
        DEXENGINE_HAND=cond["hand"],
        DEXENGINE_DATASET=cond["dataset"],
        DEXENGINE_EMBODIMENT=cond["emb"],
        DEXENGINE_SCENE=cond["scene"],
        DEXENGINE_F0_STAGE=cond["f0"],
        CUDA_VISIBLE_DEVICES=str(gpu),
    )


def run_judge(cond: dict, mode: str, bundle: Path, clips: list[str], gpu: str, chunk: str, mem: str) -> dict:
    """One judge run: rc 0 ∧ clips of the DONE line = list ∧ end marker lines = list (same set) -> measured seconds."""
    work = bundle / "_judge" / f"{mode}_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    work.mkdir(parents=True)
    cf = work / "clips.txt"
    cf.write_text("".join(f"{c}\n" for c in clips))
    timeout_s = TIMEOUT_BOOT_S + TIMEOUT_PER_CLIP_S * len(clips)
    script = judge_script(cond, mode, bundle, cf, work, chunk, timeout_s, mem, ACC.write_reselect(cond, clips, work))
    (work / "judge_cmd.sh").write_text(script + "\n")
    log = work / "judge.log"
    say(f"judge {mode} start · clips {len(clips)} · GPU {gpu} · {work}")
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(["bash", "-c", script], env=judge_env(cond, gpu), stdout=f, stderr=subprocess.STDOUT)
    sec = time.time() - t0
    text = log.read_text(errors="replace")
    # DONE line of placement.judge.aug_cells
    done = JAC.done_clips(text, mode)
    marks = (
        [ln.split("\t") for ln in (work / "judged.tsv").read_text().splitlines()]
        if (work / "judged.tsv").is_file()
        else []
    )
    if r.returncode or done != [str(len(clips))] or sorted(m[0] for m in marks) != sorted(clips):
        tb = [ln for ln in text.splitlines() if "✗" in ln or "Error" in ln or "Traceback" in ln][-3:]
        ends = f"end markers {len(marks)}/{len(clips)}"
        die(f"judge {mode} rc {r.returncode} · DONE {done} · {ends} · {LM.judge_log(log)} · {tb}")
    per = {m[0]: float(m[1]) for m in marks}
    boot = sec - sum(per.values())
    say(f"judge {mode} DONE clips {len(clips)} · total {sec:.0f} s · per clip {per} · boot etc. {boot:.0f} s")
    return {"work": str(work), "sec": round(sec, 1), "per_clip": per, "boot_sec": round(boot, 1), "rc": r.returncode}


# ── stage state (skipped only with the same fingerprint)
def d_state(bundle: Path, clips: list[str], rows: dict, dataset: str) -> tuple[list, list]:
    """Region dumps -> (done clips, clips to do); mismatched dumps or dumps outside the list die (reuse refused)."""
    from dexengine.record.augment import judged_region as JR

    RR, head = JFP._rr(), JAC.head_commit()
    done, todo, bad = [], [], []
    for c in clips:
        p = JAC.region_path(bundle, c)
        if not p.is_file():
            todo.append(c)
            continue
        R = JR.read_region(p)
        verdict, comps = JFP.diff_ruleset(
            rows[c][JFP.COLUMN], R["judge_fp"], "ADOPTED", (dataset, rows[c]["_emb"], RR.hand_dir(), c)
        )
        why = ACP.reuse_why(p.name, R, verdict, comps, head)
        if why == ACP.AGAIN:
            todo.append(c)
        elif why:
            bad.append(why)
        else:
            done.append(c)
    over = (
        sorted(f.name for f in (bundle / "region").glob("region_*.npz") if JR.clip_of_region(f) not in set(clips))
        if (bundle / "region").is_dir()
        else []
    )
    if bad or over:
        die(
            f"reuse refused: region dumps with another fingerprint {bad[:3]} · dumps outside the bundle {over[:3]}: "
            f"clear {bundle}/region and run again"
        )
    return done, todo


def l_state(bundle: Path, clips: list[str]) -> str:
    """Judge chunks -> "absent" | "done"; a mismatch with the list, the fingerprint or the judge code commit (rows of
    another commit than head: cells_pick.other_commit, as the region dumps) dies (reuse refused)."""
    from dexengine.record.augment import judged as AJ

    files = sorted((bundle / "judged").glob("augcells_*__*.jsonl")) if (bundle / "judged").is_dir() else []
    ikc = sorted((bundle / "ikc").glob("*.npz")) if (bundle / "ikc").is_dir() else []
    if not files:
        if ikc:
            die(f"reuse refused: 0 judge chunks but {len(ikc)} ikc: clear {bundle}/ikc and run again")
        return "absent"
    try:
        rows = AJ.read_chunks(bundle / "judged")
    except SystemExit as e:
        die(f"reuse refused: judge chunks are not whole ({e}): clear {bundle}/judged · ikc and run again")
    why, head = [], JAC.head_commit()
    for c in clips:
        if not JAC.cells_path(bundle, c).is_file():
            why.append(f"{c}: no cell list")
            continue
        cj = json.loads(JAC.cells_path(bundle, c).read_text())
        want = {(i, JAC.cell_key(x)) for i, x in enumerate(cj["cells"])}
        rs = [r for r in rows if r["clip"] == c]
        got = {(int(r["k"]), JAC.cell_key(r["cell"])) for r in rs}
        if got != want or len(rs) != len(got):
            why.append(f"{c}: rows {len(rs)} != list {len(want)}")
        elif any(r["judge_fp"] != cj["judge_fp"] for r in rs):
            why.append(f"{c}: row judge_fp != list judge_fp")
        elif other := sorted({r["judge_commit"][:9] for r in rs if ACP.other_commit(r, head)}):
            why.append(f"{c}: rows of another judge commit {other} != {head[:9]}")
    over = sorted({r["clip"] for r in rows} - set(clips))
    if why or over:
        die(
            f"reuse refused: judge chunks differ from the current list and fingerprint {why[:3]} · outside the bundle "
            f"{over[:3]}: clear {bundle}/judged · ikc and run again"
        )
    return "done"


def run_pick(hand: str, campaign: str, bundle_id: str, dataset: str, clips: list[str], bundle: Path) -> dict:
    """Cell choice (augment.cells_pick, rules policy): rerun every time; when it differs from the existing list the
    choice writes nothing and dies (cells_pick.write_cells owns the change gate)."""
    t0 = time.time()
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "dexengine/record/augment/cells_pick.py"),
            "--hand",
            hand,
            "--campaign",
            campaign,
            "--bundle",
            bundle_id,
            "--dataset",
            dataset,
            "--clips",
            ",".join(clips),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = r.stdout + r.stderr
    if r.returncode or f"{ACP.DONE_HEAD}{len(clips)} · " not in out:
        die(f"cell choice rc {r.returncode}: {out.strip().splitlines()[-3:]}")
    n = {c: len(json.loads(JAC.cells_path(bundle, c).read_text())["cells"]) for c in clips}
    done = [x for x in out.splitlines() if "[AUG_CELLS_PICK] DONE" in x][-1]
    say(f"choice DONE clips {len(clips)} · cells {n} · {time.time() - t0:.1f} s · {done.split(' · ', 1)[-1]}")
    return n


# ── end check
def final_count(bundle: Path, hand: str, dataset: str, clips: list[str], conds: dict) -> dict:
    """Bundle clips = region = list = judged clips · chunk done lines · ikc per passing row · augment.judged.load per
    clip."""
    from dexengine.record.augment import judged as AJ
    from dexengine.record.augment import judged_region as JR
    from dexengine.record.augment.place_src import Src

    regions = sorted(JR.clip_of_region(f) for f in (bundle / "region").glob("region_*.npz"))
    cells = sorted(f.name[len("cells_") : -len(".json")] for f in (bundle / "cells").glob("cells_*.json"))
    rows = AJ.read_chunks(bundle / "judged")
    judged = sorted({r["clip"] for r in rows})
    want = sorted(clips)
    if not (regions == cells == judged == want):
        die(f"count mismatch: bundle {len(want)} · region {len(regions)} · list {len(cells)} · judged {len(judged)}")
    ikc = {f.name for f in (bundle / "ikc").glob("*.npz")}
    need = set()
    for r in rows:
        if r["ok"]:
            hit = ikc & {P.ikc_name(r["clip"], f"k{int(r['k']):02d}", s) for s in ("right", "left")}
            if not hit:
                die(input_missing(f"{r['clip']} k={r['k']}: passing row without ikc"))
            need |= hit
    if ikc - need:
        die(f"ikc without a passing row {sorted(ikc - need)[:3]}")
    per = {}
    for c in clips:
        cd = conds[c]
        src = Src(P.main_root(), hand, dataset, cd["emb"], c, arm=cd["arm"], aerial=cd["scene"] == "aerial")
        J = AJ.load(bundle, c, dataset, src)
        per[c] = {
            "region": J["n_region"],
            "cells": len(J["rows"]),
            "passed": sum(bool(r["ok"]) for r in J["rows"]),
            "canon_pass": bool(J["rows"][0]["ok"]),
            "stages": {
                s: sum(r["stage"] == s for r in J["rows"])
                for s in JAC.STAGES
                if any(r["stage"] == s for r in J["rows"])
            },
        }
    chunks = sorted(f.name for f in (bundle / "judged").glob("augcells_*__*.jsonl"))
    return {
        "clips": len(want),
        "region": len(regions),
        "cells_files": len(cells),
        "judged": len(judged),
        "cells": sum(p["cells"] for p in per.values()),
        "passed": sum(p["passed"] for p in per.values()),
        "ikc": len(ikc),
        "chunks": chunks,
        "per_clip": per,
    }
