#!/usr/bin/env python3
"""Campaign entry of trajectory recording for a finalized cell (the replay render it once fed is gone: the RL trainer
draws no frames — robot videos come only from the record render, scripts/stage4_record.py).

  record   for a finalized cell (FINALIZED), re-run the target checkpoint (RENDER_LABELS: the judge record's best)
           with the same settings as its final judge and record its trajectories only: the recorded judge command's
           env · the judging shape (judge.metrics.judge_shape) · the same frozen-replay entry (PARK on · warm-up 1) +
           DEXENGINE_TRAJ_RECORD. The judge table, judge record and judge value are not touched; this pass's
           completion rate is only an independent oracle against that final judge of the table (ORACLE_BAND_PP).
           This is the entry for finished campaigns; new campaigns record at the final judge itself
           (campaign_model.TRAJ_RECORD_HANDS → RLJUDGE_TRAJ=1).
           Output = the cell's <RUN>_jf<iter>_traj.npz (judge.metrics.TRAJ) · session pieces (logs · command) =
           <cell>/traj_record/<checkpoint>/ · ledger record.json.
  replay   removed: dies with the pointer of dexengine.rl.render_removed.
Tree = --tree (training code + recorder: the eval tree). The recorded judge command is moved onto that tree with
rejudge.judge_command (--relocate · SEQ_DIR root: same-file proof) · the host gate (cell_state.launch; the session's
launch files live in the tree's work area). Marker: [CAMPAIGN_REPLAY] ….
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.rl import cell_state as CH
from dexengine.rl import cells as CL
from dexengine.rl import render_removed as RR
from dexengine.rl import root_guard as HR
from dexengine.rl.acceptance import bundle_of, manifest_of
from dexengine.rl.cell_tools import render_targets
from dexengine.rl.judge import metrics as JM
from dexengine.rl.judge import record as CX
from dexengine.rl.judge import rejudge as RJ

#: Record targets (label heads of render_targets over the judge record): best only.
RENDER_LABELS = ("best",)
#: Allowed difference (pp) between the record pass's completion rate and the same final judge of the judge table (sim
#: eval is not bit-stable: the judging standard's ±1.5 pp band).
ORACLE_BAND_PP = 1.5
SESSION_TIMEOUT_S = 5400  #: a record session's timeout (s)
WAIT_S = 10.0  #: poll interval while a session runs (s)
_AGG = re.compile(r"^\[FROZEN_REPLAY\] DONE .*aggregate_completion_rate=([0-9.]+)", re.M)


def _cell(hand: str, campaign: str, bid: str, tree: Path):
    """(manifest, bundle, cell, run) of a finalized bundle cell under `tree`."""
    man = manifest_of(hand, campaign, tree)
    b = bundle_of(man, bid)
    sdir = CL.bundle_dir(hand, campaign, b, tree)
    if not CH.cell_file(sdir, "finalized").is_file():
        raise SystemExit(f"[CAMPAIGN_REPLAY] {bid}: the cell is not finalized — finalized cells only")
    run = json.loads(CH.cell_file(sdir, "launch").read_text(encoding="utf-8"))["run"]
    return man, b, sdir, run


def _target(run: str, sdir: Path) -> tuple[str, Path, str, int, float]:
    """(checkpoint, its run folder in the cell, label, final judge iteration, judge table completion %): the first
    target of RENDER_LABELS."""
    x = CX.judge_extras(run, sdir)
    for ck, folder, label in render_targets(run, sdir):
        if label.split("_")[0] in RENDER_LABELS:
            pick = x.best if label.startswith("best") else x.last
            if pick.get("kind") != "f":
                raise SystemExit(
                    f"[CAMPAIGN_REPLAY] {run}: {label} is not a final judge ({pick}) — outside the record naming rule"
                )
            return ck, folder, label, int(pick["iter"]), float(pick["agg_pct"])
    raise InputMissingError(f"[CAMPAIGN_REPLAY] {run}: render targets {RENDER_LABELS} are not in the judge record")


def session_env(sdir: Path, tree: Path) -> tuple[list[str], dict]:
    """The recorded judge command (launch id filled) moved onto this tree (rejudge.judge_command --relocate), as the
    training env tokens without the sidecar's RLJUDGE_* → (tokens, relocation facts)."""
    cmds = dict(
        ln.split("=", 1)
        for ln in CH.cell_file(sdir, "launch_cmd").read_text(encoding="utf-8").splitlines()
        if "=" in ln
    )
    lid = CH.launch_id_of(sdir)
    if not cmds.get("judge") or cmds["judge"] == "None" or not lid:
        raise InputMissingError(f"[CAMPAIGN_REPLAY] {sdir}: no recorded judge command or launch id")
    cmd, facts = RJ.judge_command(cmds["judge"].replace(CH.LAUNCH_ID_SLOT, lid), sdir, tree, None, True)
    tok = shlex.split(cmd)
    i = tok.index("bash") if "bash" in tok else -1
    if tok[:1] != ["env"] or i < 0 or not tok[i + 1].endswith("judge_sidecar.sh"):
        raise SystemExit(
            f"[CAMPAIGN_REPLAY] the judge command is not of the form env … bash …judge_sidecar.sh: {cmd[:160]}"
        )
    return [t for t in tok[1:i] if not t.startswith("RLJUDGE_")], facts


def _train_ne(sdir: Path, run: str) -> int:
    """The training's NE (from <RUN>_cmd.txt in the cell's logs)."""
    f = CL.cell_log_dir(sdir) / f"{run}_cmd.txt"
    m = re.search(r"(?:^| )NE=(\d+)", f.read_text(errors="replace")) if f.is_file() else None
    if not m:
        raise InputMissingError(f"[CAMPAIGN_REPLAY] {f}: no training NE — the judging shape cannot be set")
    return int(m.group(1))


def stage_coll_preset(folder: Path, tree: Path, alias: str, data_root=None) -> bool:
    """Place the run folder's (the checkpoint's parent) collision-preset record coll_preset.txt into the alias's run
    folder of the work area: the same preparation as the judge sidecar's launch_judge (replay and data collection
    call this). The launcher reads it there (dexengine/rl/launch/records.py); a run folder without one places nothing
    and leaves the decision to the launcher (no default is made here). Returns whether it was placed.
    The work area is the launch's run tree — paths.run_root_of(its data root, code tree `tree`), the rule the launcher
    applies (Request.run_tree); `data_root` = the DEXENGINE_MAIN_ROOT the launch gets (default: this process's data
    root for `tree`, paths.data_root_of)."""
    cp = folder / "coll_preset.txt"
    if not cp.is_file():
        return False
    data = cp.read_bytes()
    if not data.strip():  # an empty record is no record (its writers write atomically)
        raise InputMissingError(f"[CAMPAIGN_REPLAY] {cp} is empty — the collision preset is unknown")
    rt = P.run_root_of(Path(data_root) if data_root is not None else P.data_root_of(tree), tree)
    ckr = CH.work_areas(rt)[1] / alias
    ckr.mkdir(parents=True, exist_ok=True)
    (ckr / "coll_preset.txt").write_bytes(data)
    return True


def _prepare(sdir, run, folder, alias, tree) -> None:
    """The work area is clean of the run (a running record / replay session is caught here too: its pieces are in the
    work area) and the collision preset is placed (stage_coll_preset)."""
    CH.assert_clean_work_area(run, tree)
    stage_coll_preset(folder, tree, alias)


def launch_dir(tree: Path, alias: str) -> Path:
    """Where a session's launch files live (replay.pid · replay.out · <kind>.rc): <alias>_launch/ in the session tree's
    log work area. Not the cell: an eval tree's hand folder links to the main checkout, so the cell lies outside the
    launch tree (cell_state.launch refuses that), and record / replay have no finalize that would move across trees.
    When the session ends they move with the work pieces into the folder beside the cell."""
    return CH.work_areas(tree)[0] / f"{alias}_launch"


def _run(kind: str, hand, run, alias, cmd, gpus, tree, maxc, side: Path) -> tuple[str, str, list[str]]:
    """Host gate → session → wait for its end → (rc, GPU, work pieces moved into the folder beside the cell)."""
    ld = launch_dir(tree, alias)
    ld.mkdir(parents=True, exist_ok=False)
    rcf = ld / f"{kind}.rc"
    HR.assert_writable(side / rcf.name, hand, root=tree)
    pid, g = CH.launch(
        "replay",
        gpus,
        run,
        f"{cmd}; echo $? > {rcf}",
        ld,
        CH.TRAIN_RESERVE_GB,
        CH.TRAIN_RAMP_S,
        cwd=tree,
        gate_kind="train",
        maxc=maxc,
    )
    if pid is None:  # gate closed (a normal case): undo only what this call created (checked absent before)
        undo = [ld, CH.work_areas(tree)[1] / alias, side]
        for p in undo:
            if p.exists():
                shutil.rmtree(p)
        raise SystemExit(f"[CAMPAIGN_REPLAY] the host gate is closed — {g} · undone {[p.name for p in undo]}")
    print(f"[CAMPAIGN_REPLAY] {kind} started pid {pid} GPU {g} · {run}", flush=True)
    while CH.alive(pid, run):
        time.sleep(WAIT_S)
    rc = rcf.read_text().strip() if rcf.is_file() else "missing"
    CH.cell_file(ld, "replay_pid").unlink(missing_ok=True)
    side.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in [p for v in CH.work_items(run, tree).values() for p in v]:
        HR.guarded_import(p, side / p.name, hand, root=tree)
        moved.append(p.name)
    return rc, g, moved


def record(
    hand: str, campaign: str, bid: str, tree: Path, gpus: list[str], cmd_only: bool = False, maxc: int = 0
) -> dict:
    """The record pass of one finalized bundle (module doc) → its ledger dict (record.json)."""
    _man, b, sdir, run = _cell(hand, campaign, bid, tree)
    ck, folder, label, it, agg = _target(run, sdir)
    out = sdir / JM.TRAJ.format(run=run, kind="f", it=it)
    if out.exists():
        raise SystemExit(f"[CAMPAIGN_REPLAY] {out.name} exists — a record is not overwritten")
    env, facts = session_env(sdir, tree)
    ne, epc = JM.judge_shape(_train_ne(sdir, run), int(b["n"]))
    alias, att = f"{run}_trec{it:04d}", CL.traj_record_dir(sdir) / ck[:-3]
    led = dict(
        run=run,
        bundle=bid,
        hand=hand,
        ckpt=ck,
        label=label,
        iter=it,
        judge_agg_pct=agg,
        judge_shape=[ne, epc],
        traj=str(out),
        tree=str(tree),
        **facts,
    )
    body = (
        f"env {shlex.join(env)} GPU={CH.GPU_SLOT} FROZEN_REPLAY_CKPT={folder / 'checkpoints' / ck} "
        f"FROZEN_REPLAY_NE={ne} FROZEN_REPLAY_EVALS_PER_CLIP={epc} FROZEN_REPLAY_ITER={it} "
        f"DEXENGINE_TRAJ_RECORD={out} timeout -k 120 {SESSION_TIMEOUT_S} python3 scripts/stage3_rl.py {alias}"
    )
    if cmd_only:
        return dict(led, cmd=body)
    if att.exists():
        raise SystemExit(f"[CAMPAIGN_REPLAY] {att} exists — an interrupted record is for a person to look at")
    _prepare(sdir, run, folder, alias, tree)
    cmd = body
    rc, g, moved = _run("record", hand, run, alias, cmd, gpus, tree, maxc, att)
    log = att / f"{alias}_frozenreplay_train.log"
    text = log.read_text(errors="replace") if log.is_file() else ""
    done = next((ln for ln in text.splitlines() if ln.startswith("[TRAJ_REC] DONE ") and f" file {out} " in ln), "")
    m = _AGG.search(text)
    rate = 100 * float(m.group(1)) if m else float("nan")
    why = [
        w
        for w, bad in (
            ("rc", rc != "0"),
            ("[TRAJ_REC] DONE missing", not done),
            ("file missing", not out.is_file()),
            (f"completion {rate:.2f} ≠ judge table {agg} ± {ORACLE_BAND_PP}", not abs(rate - agg) <= ORACLE_BAND_PP),
        )
        if bad
    ]
    led.update(cmd=cmd, gpu=g, rc=rc, done=done, completion_pct=round(rate, 2), work_pieces=moved, ok=not why, why=why)
    (att / "record.json").write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"[CAMPAIGN_REPLAY] record {'DONE' if not why else 'FAIL ' + ' · '.join(why)} {bid} · {ck}({label}) · "
        f"completion {rate:.2f} vs judge table {agg} · {out.name}",
        flush=True,
    )
    return led


def main(argv=None) -> int:
    """The record command line (module doc); `replay` dies (removed)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("record", "replay"))
    ap.add_argument("--hand", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument(
        "--tree", required=True, help="the tree the session runs in (training code + recorder: the eval tree)"
    )
    ap.add_argument("--gpus", required=True, help="<host>:<index,…> — the host gate gives one of them")
    ap.add_argument("--cmd-only", action="store_true")
    a = ap.parse_args(argv)
    if a.what == "replay":
        raise SystemExit(f"{RR.MARK} cell_session replay removed (tiled trajectory replay render) — {RR.POINTER}")
    P.check_rl_campaign(a.campaign)
    host, sep, gs = a.gpus.partition(":")
    if not sep or not gs or (not a.cmd_only and CH.this_host() != host):
        raise SystemExit(
            f"[CAMPAIGN_REPLAY] --gpus {a.gpus!r} — only <host>:<index,…> of this machine ({CH.this_host()})"
        )
    tree = Path(a.tree).resolve()
    led = record(a.hand, a.campaign, a.bundle, tree, gs.split(","), a.cmd_only)
    print(json.dumps({k: v for k, v in led.items() if k not in ("work_pieces",)}, ensure_ascii=False, indent=1))
    return 0 if a.cmd_only or led.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
