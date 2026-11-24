#!/usr/bin/env python3
"""Rejudge a FAILED bundle whose only judge defect was the PARK gate, with its existing checkpoints and no retraining
(the sibling of cell_tools relaunch: that one clears a whole attempt for a cold relaunch; this one clears only the
judge pieces and starts the cell's recorded sidecar again under the same launch id).

Refused (every check names what it read):
  · the bundle is not failed, or its train.rc is not 0 (a training that died is not rejudged)
  · the judge rc is not FAIL, or the INVALID rows' reasons (judge table column 10, ';'-joined by the sidecar) are not
    {PARK_REASON} only: any other reason means the judge itself must be looked at, not re-run. --allow-theta adds
    THETA_REASON (the θ-term mismatch of an idle side), and then the named commits must carry THETA_FIX (by patch
    id): a θ-invalid table re-run on the code that made it is the same table
  · a checkpoint the old table judged is not in the run's checkpoint folders (the work area: nothing finalized yet)
  · this tree's HEAD is not "the training's commit (training manifest git=) + exactly the named commits": the commits
    between them are compared with the named ones by patch id (a cherry-pick keeps its patch id); merges refused
  · a live process of the run (train · judge pids) or a host-gate claim of it
  · the recorded judge command names its cell under another tree than the cell's here (a pulled cell judged from a
    side worktree) and no --relocate
Moves (never deletes): into <cell>/rejudged/<launch id>_<k>/ — cell/: the sidecar's table, rc, pid, extras, shadow
rc, judge.{out,pid,starts} and every <RUN>_j[qfs]<iter>* file (per-judge clips · launch out); work/log · work/ckpt:
every <RUN>_j[qfs]<iter>* of the work areas (judge logs, manifests, rosters, snapshots, alias folders). Kept:
<RUN>_judge.id (the sidecar resumes under the same launch id), launch.json, launch_cmd.txt, train.*, the training's
checkpoints and logs. Ledger REJUDGE.json (what · when · why · who · old table rows · commits) and
rejudged/rejudge.tsv; an interrupted apply resumes from its plan. Then the cell's recorded judge= command runs again
with RLJUDGE_ALLOW_TREE_CHANGE=1 in front (the sidecar marks the row as explicitly allowed), one line in judge.starts.
Named rewrites of that command (default off: the recorded command as it is; each prints its marker and goes to the
ledger, and is planned before anything moves):
  --judge-gpus "<i …>"  RLJUDGE_GPUS (the recording machine's GPUs) → these — [REJUDGE_GPUS]
  --relocate            every absolute path under the recorded tree (the prefix the recorded RLJUDGE_OUT_DIR puts
                        before this cell's tree-relative path) → the same path under this tree, each one the SAME file
                        (os.path.samefile), and SEQ_DIR=$PWD/ → the data root of this tree (paths.data_root_of: env
                        DEXENGINE_MAIN_ROOT, else this tree; a side worktree holds no data) — [REJUDGE_RELOCATE]
Usage:
  python3 dexengine/rl/judge/rejudge.py --hand <h> --campaign <c> --bundle <id> --commits <sha> [<sha> …]
      --why "<text>" [--allow-theta] [--judge-gpus "<i …>"] [--relocate] [--apply]
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.rl import cell_state as CH
from dexengine.rl import cells as CL
from dexengine.rl import root_guard as HR
from dexengine.rl.acceptance import bundle_of, manifest_of
from dexengine.rl.cell_tools import _live_pieces

#: The sidecar's row reason for a PARK-invalid judge (scripts/rl/judge_sidecar.sh verify_judge).
PARK_REASON = "[EVAL_PARK] invalid batch"
#: The sidecar's row reason head for a θ-term mismatch (judge_sidecar.sh verify_judge) and the commit that fixed its
#: cause (an idle side registered the θ term), compared by patch id so a cherry-pick counts.
THETA_REASON = "θ termination term:"
THETA_FIX = "82860f38c59752f21facc589c798b070d68c9f14"
ALLOW_TREE = "RLJUDGE_ALLOW_TREE_CHANGE=1"
GPUS_RX = re.compile(r"RLJUDGE_GPUS=('[^']*'|\"[^\"]*\"|\S+)")
OUT_RX = re.compile(r"\bRLJUDGE_OUT_DIR=(\S+)")
_CELL_FIXED = ("judge_out", "judge_pid", "judge_starts")
#: The launch env word that roots a run's tree dir at the launching tree (seq_root rewrites it).
SEQ_PWD = "SEQ_DIR=$PWD/"


def seq_root(text: str, root: str) -> str:
    """`text` (a launch env word or a whole command) with its first SEQ_DIR=$PWD/ rooted at `root`."""
    return text.replace(SEQ_PWD, f"SEQ_DIR={root.rstrip('/')}/", 1)


def _git(tree: Path, *a: str, inp: str | None = None) -> str:
    r = subprocess.run(["git", *a], cwd=str(tree), capture_output=True, text=True, input=inp, timeout=120)
    if r.returncode != 0:
        raise SystemExit(f"[rejudge] git {' '.join(a)} rc {r.returncode}: {r.stderr.strip()[-200:]}")
    return r.stdout


def patch_id(tree: Path, sha: str) -> str:
    """The stable patch id of commit `sha` in `tree` (an empty commit or a merge raises SystemExit)."""
    out = _git(tree, "patch-id", "--stable", inp=_git(tree, "show", sha)).split()
    if not out:
        raise InputMissingError(f"[rejudge] {sha} has no patch id (empty commit · merge) — cannot prove it")
    return out[0]


def tree_proof(tree: Path, base: str, commits: list[str]) -> str:
    """'' when HEAD = base + exactly the named commits (patch ids, as a multiset); else why not."""
    full = _git(tree, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", full, "HEAD"], cwd=str(tree)).returncode != 0:
        return f"HEAD is not a descendant of the training commit {base}"
    if _git(tree, "rev-list", "--merges", f"{full}..HEAD").strip():
        return "there is a merge commit between the training commit and HEAD"
    between = _git(tree, "rev-list", "--reverse", f"{full}..HEAD").split()
    got, want = sorted(patch_id(tree, c) for c in between), sorted(patch_id(tree, c) for c in commits)
    if got != want:
        return (
            f"HEAD = training commit + {len(between)} commits — their patch ids differ from the "
            f"{len(commits)} named commits"
        )
    return ""


def judge_piece(run: str) -> re.Pattern:
    """The names of a run's judge pieces: <RUN>_j[qfs]<iter ≥4 digits>[_… | .…]."""
    return re.compile(re.escape(run) + f"_j[{CH.JUDGE_KINDS}]" + r"\d{4,}(?:[_.].*)?")


def table_rows(run: str, sdir: Path) -> list[list[str]]:
    """The sidecar's judge table rows (header dropped)."""
    f = CH.judge_files(run, sdir)[1]
    if not f.is_file():
        raise InputMissingError(f"[rejudge] no judge table: {f}")
    return [ln.split("\t") for ln in f.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]


def invalid_reasons(rows: list[list[str]]) -> set[str]:
    """Reasons of the INVALID rows (column 9 = status · column 10 = reasons joined by ';')."""
    # a PARK reason carries per-gate counts after it, a θ reason its measured values: match by the head
    heads = (PARK_REASON, THETA_REASON)
    return {
        next((h for h in heads if r.strip().startswith(h)), r.strip())
        for row in rows
        if len(row) > 9 and row[8] == "INVALID"
        for r in row[9].split(";")
        if r.strip()
    }


def missing_ckpts(run: str, rows: list[list[str]], root=None) -> list[str]:
    """Checkpoints the judge table judged (q · f rows) that are not in the run's checkpoint folders."""
    it = CH.work_items(run, root)
    have = {f.name for d in it["ckpt"] + it["resume"] for f in (d / "checkpoints").glob("agent_*.pt")}
    want = {row[3] for row in rows if len(row) > 3 and row[1] in ("q", "f") and row[3].startswith("agent_")}
    return sorted(want - have)


def train_git(run: str, root=None) -> str:
    """The training's commit (git= on the first line of <RUN>_cmd.txt in the log work area)."""
    ref, _ = CH.work_areas(root)
    f = ref / f"{run}_cmd.txt"
    m = re.search(r"\bgit=([0-9a-f]{7,40})\b", f.read_text(encoding="utf-8").splitlines()[0]) if f.is_file() else None
    if m is None:
        raise SystemExit(f"[rejudge] could not read git= of the training manifest: {f}")
    return m.group(1)


def judge_command(
    cmd: str, sdir: Path, base: Path, gpus: str | None, relocate: bool, same=None, seq_same=None
) -> tuple[str, dict]:
    """The recorded judge command (launch id filled) with the named rewrites of the module doc → (command, ledger
    facts). Refuses a cell recorded under another tree without --relocate, a GPU list of another shape, a rewritten
    path that is not the same file, or a GPU slot that is not exactly one. `same(recorded, here)` = the identity test
    (default: both exist and os.path.samefile) · `seq_same` = SEQ_DIR's (default: the recorded one absent or the same
    file)."""
    facts: dict = {}
    if gpus is not None:
        if not re.fullmatch(r"\d+( \d+)*", gpus):
            raise SystemExit(f"[rejudge] --judge-gpus {gpus!r} — space-separated GPU indices only")
        hit = GPUS_RX.findall(cmd)
        if len(hit) != 1:
            raise SystemExit(
                f"[rejudge] the recorded command has {len(hit)} RLJUDGE_GPUS — rewritten only when there is exactly one"
            )
        cmd = GPUS_RX.sub(f"RLJUDGE_GPUS='{gpus}'", cmd)
        facts["judge_gpus"] = {"recorded": hit[0], "used": gpus}
        print(f"[REJUDGE_GPUS] {hit[0]} → '{gpus}'", flush=True)
    m = OUT_RX.findall(cmd)
    if not m and not relocate:
        return cmd, facts  # no cell folder named (the sidecar's old shape) — no recorded tree to compare
    rel = sdir.relative_to(base).as_posix()
    if len(m) != 1 or not m[0].endswith("/" + rel):
        raise SystemExit(f"[rejudge] the recorded command's RLJUDGE_OUT_DIR {m} does not end with this cell ({rel})")
    rec_tree = m[0][: -len(rel) - 1]
    if rec_tree == str(base) and not relocate:
        return cmd, facts
    if not relocate:
        raise SystemExit(
            f"[rejudge] recorded tree {rec_tree} ≠ this tree {base} — make the cell and work areas "
            "visible from this tree, then --relocate (it checks they are the same files)"
        )
    paths = sorted(set(re.findall(re.escape(rec_tree) + r"/[^\s'\";]+", cmd)))
    same = same or (lambda p, q: os.path.exists(p) and os.path.exists(q) and os.path.samefile(p, q))
    bad = [p for p in paths if not same(p, base / p[len(rec_tree) + 1 :])]
    data = P.data_root_of(base)  # the data root of this tree (base), not of the calling code tree
    seq = re.findall(re.escape(SEQ_PWD) + r"([^\s'\";]+)", cmd)
    seq_same = seq_same or (lambda p, q: not os.path.exists(p) or os.path.samefile(p, q))
    bad += [
        f"SEQ_DIR $PWD/{s} (data root {data})"
        for s in seq
        if not (data / s).is_dir() or not seq_same(f"{rec_tree}/{s}", data / s)
    ]
    if bad:
        raise InputMissingError(
            f"[rejudge] --relocate: files differ or are missing {bad[:3]} — tree {base} (data root "
            f"{data}) does not see the cell, lists, work place and SEQ_DIR of the recorded tree "
            f"{rec_tree}"
        )
    cmd = seq_root(cmd.replace(rec_tree + "/", str(base) + "/"), str(data))
    facts["relocate"] = {
        "recorded_tree": rec_tree,
        "tree": str(base),
        "paths": len(paths),
        "seq_root": str(data),
        "seq_dirs": len(seq),
    }
    print(
        f"[REJUDGE_RELOCATE] {rec_tree} → {base} · same files {len(paths)} · SEQ_DIR root {data}({len(seq)})",
        flush=True,
    )
    return cmd, facts


def rejudge_plan(run: str, sdir: Path, dst: Path, root=None) -> list[tuple[str, Path, Path]]:
    """[(how, src, dst)]: the cell's judge records and pieces (move), then the work areas' judge pieces (import)."""
    pat = judge_piece(run)
    rc, tsv, _id = CH.judge_files(run, sdir)
    cell = [rc, tsv, sdir / f"{run}_judge.pid", CH.judge_extras_file(run, sdir), CH.judge_shadow_file(run, sdir)]
    cell += [CH.cell_file(sdir, k) for k in _CELL_FIXED]
    cell += sorted(p for p in sdir.iterdir() if pat.fullmatch(p.name))
    ref, ckr = CH.work_areas(root)
    work = [("log", p) for p in sorted(ref.glob(f"{run}_j*")) if pat.fullmatch(p.name)]
    work += [("ckpt", p) for p in sorted(ckr.glob(f"{run}_j*")) if pat.fullmatch(p.name)]
    return [("move", p, dst / "cell" / p.name) for p in cell if p.exists()] + [
        ("import", p, dst / "work" / k / p.name) for k, p in work
    ]


def check_rejudgeable(
    run: str, sdir: Path, b: dict, commits: list[str], root, tree: Path, live, allow_theta: bool = False
) -> dict:
    """The refusals of the module doc → the ledger's facts (old table, train rc, launch id, …). live() = the live
    campaign loops that could restart the old sidecar on this cell (the caller's check; [] = none)."""
    st = CH.bundle_state(sdir, run)
    trc = CH.cell_file(sdir, "train_rc")
    train_rc = trc.read_text().strip() if trc.is_file() else None
    if st != "failed" or train_rc != "0":
        raise SystemExit(
            f"[rejudge] {b['id']}: state {st} · train.rc {train_rc} — only failed bundles with train.rc 0 "
            "are judged again"
        )
    v = CH.judge_verdict(run, sdir)
    rows = table_rows(run, sdir)
    why = invalid_reasons(rows)
    ok = {PARK_REASON, THETA_REASON} if allow_theta else {PARK_REASON}
    if v is None or v[0] != "FAIL" or not why or not why <= ok:
        raise SystemExit(
            f"[rejudge] {b['id']}: judge {v} · invalid reasons {sorted(why)} — only a judge that "
            f"FAILed on {' · '.join(sorted(ok))} invalid rows alone is judged again"
        )
    miss = missing_ckpts(run, rows, root)
    if miss:
        raise InputMissingError(
            f"[rejudge] {b['id']}: checkpoints the old judge measured are not in the work area {miss[:4]}"
        )
    lv = live()
    if lv:
        raise SystemExit(f"[rejudge] live campaigns {lv[:3]} — the loop could restart this cell's old sidecar")
    base = train_git(run, root)
    bad = tree_proof(tree, base, commits)
    if bad:
        raise SystemExit(f"[rejudge] tree proof failed: {bad}")
    if THETA_REASON in why and patch_id(tree, THETA_FIX) not in {patch_id(tree, c) for c in commits}:
        raise SystemExit(
            f"[rejudge] {b['id']}: judging a θ-invalid table again needs the θ fix {THETA_FIX[:9]} among "
            "the named commits"
        )
    return {
        "run": run,
        "train_rc": train_rc,
        "judge": list(v),
        "old_rows": ["\t".join(r) for r in rows],
        "train_git": base,
        "head": _git(tree, "rev-parse", "HEAD").strip(),
        "commits": list(commits),
        "cell_launch_id": CH.launch_id_of(sdir),
    }


def _open(sdir: Path) -> Path | None:
    """The attempt an interrupted apply left open (plan.tsv without REJUDGE.json); two of them raise."""
    rdir = sdir / CH.CELL_FILES["rejudged"]
    open_ = [d.parent for d in sorted(rdir.glob("*/plan.tsv")) if not (d.parent / "REJUDGE.json").is_file()]
    if len(open_) > 1:
        raise SystemExit(
            f"[rejudge] two interrupted rejudges {[d.name for d in open_]} — not resumed (for a person to look at)"
        )
    return open_[0] if open_ else None


def rejudge(
    hand: str,
    campaign: str,
    b: dict,
    commits: list[str],
    why: str,
    apply: bool,
    root=None,
    tree: Path | None = None,
    live=None,
    *,
    allow_theta: bool = False,
    judge_gpus: str | None = None,
    relocate: bool = False,
) -> dict:
    """Plan (and with apply, run) the rejudge of one bundle (module doc) → the ledger dict. Resumes an interrupted
    apply first."""
    sdir = CL.bundle_dir(hand, campaign, b, root)
    run = json.loads(CH.cell_file(sdir, "launch").read_text(encoding="utf-8"))["run"]
    tree = Path(tree or P.WT)
    att = _open(sdir)
    if att is None:
        rec = check_rejudgeable(run, sdir, b, commits, root, tree, live or (list), allow_theta)
        lid = rec["cell_launch_id"] or "nolid"
        k = len(list((sdir / CH.CELL_FILES["rejudged"]).glob(f"{lid}_*")))
        att = sdir / CH.CELL_FILES["rejudged"] / f"{lid}_{k}"
        rec.update(
            hand=hand,
            campaign=campaign,
            bundle=b["id"],
            why=why,
            attempt=att.name,
            by=f"{getpass.getuser()}@{socket.gethostname()} pid {os.getpid()}",
        )
        plan = rejudge_plan(run, sdir, att, root)
    else:
        rec = json.loads((att / "pending.json").read_text(encoding="utf-8"))
        plan = [
            (h, Path(s), Path(d))
            for h, s, d in (
                ln.split("\t") for ln in (att / "plan.tsv").read_text(encoding="utf-8").splitlines()[1:] if ln
            )
        ]
    live_p = _live_pieces(sdir, run)
    if live_p:
        raise SystemExit(f"[rejudge] {b['id']}: pieces are still alive {live_p}")
    cmds = dict(
        ln.split("=", 1)
        for ln in CH.cell_file(sdir, "launch_cmd").read_text(encoding="utf-8").splitlines()
        if "=" in ln
    )
    lid = CH.launch_id_of(sdir)
    if not cmds.get("judge") or cmds["judge"] == "None" or not lid:
        raise InputMissingError(f"[rejudge] {sdir}: no recorded judge command or launch id — nothing is started")
    cmd, facts = judge_command(
        cmds["judge"].replace(CH.LAUNCH_ID_SLOT, lid), sdir, Path(root or P.WT), judge_gpus, relocate
    )
    rec.update(facts, n_items=len(plan), applied=apply, command=f"{ALLOW_TREE} {cmd}")
    if not apply:
        return dict(rec, plan=[(h, str(s), str(d)) for h, s, d in plan])
    return _apply(hand, sdir, att, rec, plan, tree, root)


def _apply(hand: str, sdir: Path, att: Path, rec: dict, plan: list, tree: Path, root=None) -> dict:
    """Move the plan's pieces, start the recorded judge command again, write the ledger → the ledger dict."""
    HR.assert_writable(att / "plan.tsv", hand, root=root)
    att.mkdir(parents=True, exist_ok=True)
    if not (att / "plan.tsv").is_file():
        (att / "pending.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        (att / "plan.tsv").write_text(
            "how\tsrc\tdst\n" + "".join(f"{h}\t{s}\t{d}\n" for h, s, d in plan), encoding="utf-8"
        )
    log = att / "moved.log"
    done = set(log.read_text(encoding="utf-8").splitlines()) if log.is_file() else set()
    with open(log, "a", encoding="utf-8") as lg:
        for how, s, d in plan:
            if str(s) in done or (not s.exists() and d.exists()):
                continue
            (HR.guarded_import if how == "import" else HR.guarded_move)(s, d, hand, root=root)
            lg.write(f"{s}\n")
            lg.flush()
    cmd = rec["command"]  # planned before anything moved (the named rewrites refuse in rejudge)
    pid = CH.spawn(sdir, "judge", cmd, tree)
    with open(CH.cell_file(sdir, "judge_starts"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} pid={pid} rejudge={att.name}\n")
    rec.update(when=time.strftime("%Y-%m-%dT%H:%M:%S"), pid=pid, moved=[(h, str(s), str(d)) for h, s, d in plan])
    tmp = att / ".REJUDGE.json.tmp"
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, att / "REJUDGE.json")
    with open(att.parent / "rejudge.tsv", "a", encoding="utf-8") as f:
        f.write(
            f"{rec['when']}\t{att.name}\t{rec['judge']}\t{rec['train_git']}\t{rec['head'][:12]}\t"
            f"{' '.join(c[:12] for c in rec['commits'])}\t{rec['n_items']}\t{rec['by']}\t"
            f"{' '.join(rec['why'].split())}\n"
        )
    return rec


def main(argv=None) -> int:
    """The rejudge command line (module doc)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hand")
    ap.add_argument("--campaign")
    ap.add_argument("--bundle")
    ap.add_argument("--commits", nargs="+", help="the commits on top of the training commit (by name) — the HEAD proof")
    ap.add_argument("--why", help="why it is judged again (kept in the ledger)")
    ap.add_argument(
        "--allow-theta",
        action="store_true",
        help="also accept θ-term invalid rows — the θ fix commit must be among the named commits",
    )
    ap.add_argument(
        "--judge-gpus",
        default=None,
        help="replace the recorded command's RLJUDGE_GPUS by this list (space-separated; default as is)",
    )
    ap.add_argument(
        "--relocate",
        action="store_true",
        help="absolute paths of the recorded tree → the same files under this tree · SEQ_DIR root = the "
        "data root (side worktrees)",
    )
    ap.add_argument("--apply", action="store_true", help="really move and start (without it: the plan only)")
    a = ap.parse_args(argv)
    miss = [k for k in ("hand", "campaign", "bundle", "commits", "why") if not getattr(a, k)]
    if miss:
        raise SystemExit(f"[rejudge] missing inputs {miss}")
    P.check_rl_campaign(a.campaign)
    if not a.why.strip():
        raise SystemExit("[rejudge] --why is empty")
    b = bundle_of(manifest_of(a.hand, a.campaign), a.bundle)
    rec = rejudge(
        a.hand,
        a.campaign,
        b,
        a.commits,
        a.why,
        a.apply,
        allow_theta=a.allow_theta,
        judge_gpus=a.judge_gpus,
        relocate=a.relocate,
    )
    print(json.dumps({k: v for k, v in rec.items() if k not in ("plan", "moved", "old_rows")}, ensure_ascii=False))
    for row in rec.get("plan") or rec.get("moved") or []:
        print("  " + "\t".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
