#!/usr/bin/env python3
"""Tools over an RL campaign's bundle cells (the loop is dexengine/cluster/scheduler.py; none of these run in it):

  relaunch  a FAILED bundle's earlier attempt leaves its cell and the work areas for <cell>/attempts/<launch id>/
            (cell files → cell/, the run's work-area pieces → work/{ckpt,log}/; moved, never deleted), with a ledger
            (RELAUNCH.json: what · when · why · who, and attempts/relaunch.tsv). The cell then reads queued and the
            loop launches it cold again. Refused unless the state is failed, no process of the run lives (pid by
            command line) and the host gate holds no claim of it. Dry by default (--apply). Order: the pieces move,
            then the ledger is written, then launch.json moves last, so the cell never reads queued without its
            ledger. An interrupted apply (attempts/<id>/plan.tsv whose cell/launch.json is not there yet) is resumed by
            the next call, only when the cell's launch record is that attempt's.
  render    removed (the trainer's checkpoint video): dies with the pointer of dexengine.rl.render_removed — robot
            videos come only from the record render (scripts/stage4_record.py). render_targets (the best and last
            checkpoints of a finalized cell) stays: the record bundle and cell_session pick their checkpoint with it.
  collect   the campaign's result tables (acceptance.collect).
Usage:
  python3 dexengine/rl/cell_tools.py relaunch --hand <h> --campaign <c> --bundle <id> --why "<text>" [--apply]
  python3 dexengine/rl/cell_tools.py collect  --hand <h> --campaign <c>
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.rl import cell_state as CH
from dexengine.rl import cells as CL
from dexengine.rl import render_removed as RR
from dexengine.rl import root_guard as HR
from dexengine.rl.acceptance import bundle_of, collect, manifest_of
from dexengine.rl.judge import record as CX


# ── relaunch
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _live_pieces(sdir: Path, run: str) -> list[str]:
    """What of a cell's run still lives: its train · judge · render processes and host gate claims."""
    out = [
        k
        for k in ("train_pid", "judge_pid", "render_pid")
        if (p := CH.pid_of(CH.cell_file(sdir, k))) is not None and CH.alive(p, run)
    ]
    out += [f"gate claim {c['kind']}{c['gpu']}" for c in CH.gate_claims() if c["run"] == run]
    return out


def relaunch_plan(sdir: Path, run: str, att: Path, root=None) -> list[tuple[str, Path, Path]]:
    """[(how, src, dst)]: the run's work-area pieces first (import), then the cell's files (move), launch.json last
    (until the very end the cell still reads failed, so the loop never touches a half-cleared cell)."""
    ref, _ckr = CH.work_areas(root)
    plan = [
        ("import", p, att / "work" / ("log" if p.parent == ref else "ckpt") / p.name)
        for v in CH.work_items(run, root).values()
        for p in v
    ]
    names = [n for n in CH.cell_leftovers(sdir) if n != CH.CELL_FILES["launch"]]
    plan += [("move", sdir / n, att / "cell" / n) for n in names]
    plan.append(("move", CH.cell_file(sdir, "launch"), att / "cell" / CH.CELL_FILES["launch"]))
    return plan


def open_attempt(sdir: Path) -> Path | None:
    """The attempt an interrupted apply left open (plan.tsv written, cell/launch.json not moved yet: the last step),
    resumable only while the cell's launch record is that attempt's (the same launch id, pending.json); two open
    attempts, or one whose record is not in the cell, raise (the ledger and the cell disagree: a person looks)."""
    adir = sdir / CH.CELL_FILES["attempts"]
    open_ = [
        d.parent for d in sorted(adir.glob("*/plan.tsv")) if not (d.parent / "cell" / CH.CELL_FILES["launch"]).is_file()
    ]
    if not open_:
        return None
    rec = json.loads((open_[0] / "pending.json").read_text(encoding="utf-8"))
    if (
        len(open_) > 1
        or not CH.cell_file(sdir, "launch").is_file()
        or CH.launch_id_of(sdir) != rec.get("cell_launch_id")
    ):
        raise SystemExit(
            f"[cells] interrupted relaunch {[d.name for d in open_]} and the cell's launch record "
            f"({CH.launch_id_of(sdir)}) disagree — not resumed (a person looks)"
        )
    return open_[0]


def relaunch(hand: str, campaign: str, b: dict, why: str, apply: bool, root=None) -> dict:
    """Clear a failed bundle's attempt (module doc) → the ledger dict. Resumes an interrupted clear first."""
    sdir = CL.bundle_dir(hand, campaign, b, root)
    adir = sdir / CH.CELL_FILES["attempts"]
    att = open_attempt(sdir)
    if att is not None:
        rec = json.loads((att / "pending.json").read_text(encoding="utf-8"))
        if why != rec["why"]:
            rec["resume_why"] = why
        plan = [
            (h, Path(s), Path(d))
            for h, s, d in (
                ln.split("\t") for ln in (att / "plan.tsv").read_text(encoding="utf-8").splitlines()[1:] if ln
            )
        ]
    else:
        if not CH.cell_file(sdir, "launch").is_file():
            raise InputMissingError(
                f"[cells] {b['id']}: no launch record in the cell — no attempt to clear (state queued)"
            )
        run = json.loads(CH.cell_file(sdir, "launch").read_text(encoding="utf-8"))["run"]
        st = CH.bundle_state(sdir, run)
        if st != "failed":
            raise SystemExit(f"[cells] {b['id']}: state {st} — only failed bundles are cleared")
        lid = CH.launch_id_of(sdir) or time.strftime("nolid_%Y%m%dT%H%M%S")
        att = adir / lid
        if att.exists():
            raise SystemExit(f"[cells] {att} exists — the same attempt is not cleared twice")
        trc = CH.cell_file(sdir, "train_rc")
        v = CH.judge_verdict(run, sdir)
        rec = {
            "hand": hand,
            "campaign": campaign,
            "bundle": b["id"],
            "run": run,
            "launch_id": lid,
            "state": st,
            "why": why,
            "cell_launch_id": CH.launch_id_of(sdir),
            "train_rc": trc.read_text().strip() if trc.is_file() else None,
            "judge": list(v) if v else None,
            "sidecar_starts": CH.sidecar_starts(sdir),
            "by": f"{getpass.getuser()}@{socket.gethostname()} pid {os.getpid()}",
        }
        plan = relaunch_plan(sdir, run, att, root)
    live = _live_pieces(sdir, rec["run"])
    if live:
        raise SystemExit(f"[cells] {b['id']}: pieces are still alive {live} — clean up after they stop")
    rec.update(n_items=len(plan), applied=apply)
    if not apply:
        return dict(rec, plan=[(h, str(s), str(d)) for h, s, d in plan])
    return _apply_relaunch(hand, att, rec, plan, root)


def _apply_relaunch(hand: str, att: Path, rec: dict, plan: list, root=None) -> dict:
    """Apply a relaunch plan (pieces, then the ledger, then launch.json) → the ledger dict."""
    HR.assert_writable(att / "plan.tsv", hand, root=root)
    att.mkdir(parents=True, exist_ok=True)
    if not (att / "plan.tsv").is_file():
        (att / "pending.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        (att / "plan.tsv").write_text(
            "how\tsrc\tdst\n" + "".join(f"{h}\t{s}\t{d}\n" for h, s, d in plan), encoding="utf-8"
        )
    log = att / "moved.log"
    done = set(log.read_text(encoding="utf-8").splitlines()) if log.is_file() else set()

    def move(items: list) -> None:
        with open(log, "a", encoding="utf-8") as lg:
            for how, s, d in items:
                if str(s) in done or (not s.exists() and d.exists()):
                    continue
                (HR.guarded_import if how == "import" else HR.guarded_move)(s, d, hand, root=root)
                lg.write(f"{s}\n")
                lg.flush()

    move(plan[:-1])  # everything but launch.json (the cell still reads failed)
    led = att / "RELAUNCH.json"
    if led.is_file():  # written before an interruption — the ledger stands
        rec = json.loads(led.read_text(encoding="utf-8"))
    else:
        rec.update(when=time.strftime("%Y-%m-%dT%H:%M:%S"), moved=[(h, str(s), str(d)) for h, s, d in plan])
        tmp = att / ".RELAUNCH.json.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, led)
    tsv = att.parent / "relaunch.tsv"
    rows = tsv.read_text(encoding="utf-8").splitlines() if tsv.is_file() else []
    if not any(r.split("\t")[1:2] == [rec["launch_id"]] for r in rows):  # one row per attempt, also on a resume
        with open(tsv, "a", encoding="utf-8") as f:
            f.write(
                f"{rec['when']}\t{rec['launch_id']}\t{rec['state']}\t{rec['train_rc']}\t{rec['judge']}\t"
                f"{rec['n_items']}\t{rec['by']}\t{' '.join(rec['why'].split())}\n"
            )
    move(plan[-1:])  # launch.json last — from here the cell reads queued
    return rec


# ── checkpoint targets
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def render_targets(run: str, sdir: Path) -> list[tuple[str, Path, str]]:
    """[(checkpoint, its run folder in the cell, label)]: the judge record's best and last (kept by finalize); one row
    when they coincide. Folder = the first holding it (<RUN> before <RUN>_R*, finalize's order)."""
    x = CX.judge_extras(run, sdir)
    if x.best is None:
        raise ValueError(
            f"{run}: the judge record has no best/last — no checkpoint to pick (a sidecar from before best/last)"
        )
    best, last = x.best["ckpt"], x.last["ckpt"]
    ck = CL.cell_ckpt_dir(sdir)
    folders = sorted(
        (d for d in ck.iterdir() if d.is_dir() and (d.name == run or d.name.startswith(run + "_R"))),
        key=lambda d: (d.name != run, d.name),
    )
    kept = {f.name: d for d in reversed(folders) for f in (d / "checkpoints").glob("agent_*.pt")}
    if not {best, last} <= set(kept):
        raise ValueError(f"{run}: the cell does not hold both best {best} · last {last} ({sorted(kept)[:4]})")
    out = [(best, kept[best], "best" + ("_last" if best == last else ""))]
    return out + ([] if best == last else [(last, kept[last], "last")])


def main(argv=None) -> int:
    """The relaunch · collect command line (module doc); `render` dies (removed)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("relaunch", "render", "collect"))
    ap.add_argument("--hand", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--why", default="", help="relaunch: why the attempt is cleared (kept in the ledger; required)")
    ap.add_argument("--apply", action="store_true", help="relaunch: really move (without it: the plan only)")
    a, rest = ap.parse_known_args(argv)
    if a.what == "render":
        raise SystemExit(f"{RR.MARK} cell_tools render removed (the trainer's checkpoint video) — {RR.POINTER}")
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    P.check_rl_campaign(a.campaign)
    if a.what == "collect":
        f, fd, n = collect(a.hand, a.campaign)
        print(f"[cells] collected {n} bundles → {f} · {fd}")
        return 0
    if not a.bundle:
        raise SystemExit(f"[cells] {a.what} needs --bundle")
    b = bundle_of(manifest_of(a.hand, a.campaign), a.bundle)
    if a.what == "relaunch":
        if not a.why.strip():
            raise SystemExit("[cells] relaunch needs --why (the ledger's reason)")
        rec = relaunch(a.hand, a.campaign, b, a.why, a.apply)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("plan", "moved")}, ensure_ascii=False))
        for row in rec.get("plan") or rec.get("moved") or []:
            print("  " + "\t".join(row))
        return 0
    raise SystemExit(f"[cells] unknown action {a.what!r}")


if __name__ == "__main__":
    sys.exit(main())
