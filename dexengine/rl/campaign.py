#!/usr/bin/env python3
"""RL campaign runner: manifest bundles → host-gated training slots → the judge sidecar → finalized cells.

The dry run (default) plans; --launch runs the loop (dexengine/cluster/local.py around the ticks of
dexengine/cluster/scheduler.py) on the GPU host after its checks (rc 3): no campaign-wide BLOCKER
(dexengine.rl.run_queue) · this machine is the GPU host · every loaded hand's manifest is staged in its campaign cell
(--stage-hand) · no --allow-no-judge.

  inputs    one ledger-mode campaign manifest per hand (<manifests>/<hand>/manifest.json)
  order     --order (required, no default hand): registered hand ids or @artic (queue.parse_order)
  state     keyed by (hand, bundle id), derived from the bundle cell's files (cell_state.bundle_state · read_states)
  names     run_name = V2C_<campaign>__<hand>_<bundle>_s42, campaign-scoped (paths.check_rl_campaign); a launch proves
            its cell and work area empty first (cell_state.launch_record: cold start)
  gate      one host gate for trainings, judges and data lanes (scripts/rl/host_gate.sh)
  GPUs      --judge-gpus (default <this host>:last) and --gpus (default every other GPU of that host), disjoint
            (cell_state.gpu_split); the loop reserves the judge GPUs judge-only (host gate rsv). Overflow
            (--judge-overflow-wait-h, default off): a training's finals run on its own GPU when the expected judge wait
            exceeds the threshold (overflow_due over live_wait_h)
  judge     scripts/rl/judge_sidecar.sh with the training's env + RLJUDGE_OUT_DIR = the cell + RLJUDGE_LAUNCH_ID +
            sidecar_env (quick judges off · deadline · concurrency) · SIDE=both|right|left
  train     scripts/stage3_rl.py with DO_VIDEO=0 under a timeout (TRAIN_TIMEOUT_X × the class estimate); rc → train.rc
  finalize  cells.finalize (move into the cell; --prune-ckpt on keeps the judged checkpoints ∪ the last)
  after     dexengine/rl/cell_tools.py: relaunch (a failed bundle's attempt → attempts/ with a ledger) · collect
            (per-hand result tables); robot videos come only from the record render (scripts/stage4_record.py)
The planning model (dexengine.rl.campaign_model: train_hours · schedule · split_table) estimates how long a queue takes
on a host split into training and judge GPUs, with or without overflow, and the checkpoint disk it peaks at.

Usage (dry run):
  DEXENGINE_ARM=rby1 python3 dexengine/rl/campaign.py --manifests <dir> --campaign <id> [--dry-root <scratch tree>]
      [--judge-gpus <host>:last] [--gpus <host>:0,…] [--judge-overflow-wait-h H] --order <hand>,…,@artic,<hand>
      [--defer-hand <hand>] [--allow-class <class>] [--plan-out <tsv>] [--stage-hand <hand>,…]
Launch (on the GPU host, from a committed tree; one loop per campaign):
  DEXENGINE_ARM=rby1 setsid nohup python3 dexengine/rl/campaign.py --launch --manifests <dir> --campaign <id>
      --order <hand>,… [--judge-overflow-wait-h H] [--prune-ckpt on] [--tick-s 60] [--change-params "<why>"]
      [--allow-train-path-change "<why>"] > <log> …
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.placement.bundle import run_inputs as RI
from dexengine.rl import cell_state as CH
from dexengine.rl import cells as CL
from dexengine.rl.campaign_model import (
    JUDGE_FINAL_H,
    TRAIN_H,
    TRAIN_TIMEOUT_X,
    mid_judges,
    schedule,
    sidecar_env,
    split_table,
    train_hours,
)
from dexengine.rl.cell_state import JUDGE_FINALS, bkey, read_states
from dexengine.rl.run_queue import (
    BLOCKERS,
    allow_tokens,
    campaign_block_reason,
    campaign_view,
    judge_supported,
    order_bundles,
    parse_order,
)

#: The trainer entry and the judge sidecar the campaign commands run (paths relative to the code tree).
TRAIN_ENTRY = "scripts/stage3_rl.py"
SIDECAR = "scripts/rl/judge_sidecar.sh"
#: The judge GPUs by default: the last GPU of the host this runs on (cell_state.this_host; resolved by measuring the
#: host's GPU count: <host>:last = <host>:7 on 8 GPUs).
JUDGE_GPUS_DEFAULT = "last"
#: Run name prefixes: a campaign run and a boot smoke never share a prefix (finalize moves <RUN>_* files); the
#: campaign id follows (run_name).
RUN_PREFIX, SMOKE_PREFIX = "V2C_", "V2S_"


# ── commands · launch ───────────────────────────────────────────────────────────────────────────────────────────────
def run_name(campaign: str, hand: str, bid: str, smoke: bool = False) -> str:
    """Run name of a bundle (or its boot smoke) in a campaign: V2C_<campaign>__<hand>_<bundle>_s42, [A-Za-z0-9_] only,
    seed suffix (one seed, 42). The campaign id is checked by its rule (paths.check_rl_campaign: no "__" or ".", so
    one campaign's name is never another's head)."""
    P.check_rl_campaign(campaign)
    sep = P.RL_CAMPAIGN_SEP
    name = (
        (SMOKE_PREFIX if smoke else RUN_PREFIX)
        + campaign
        + sep
        + re.sub(r"[^A-Za-z0-9]+", "_", f"{hand}{sep}{bid}")
        + "_s42"
    )
    return name


def canon_value(b: dict, man: dict, names: tuple[str, ...]) -> list[str]:
    """Canonical launch-table values for this bundle's path (run_inputs.canon_env); every one must be set."""
    got = RI.canon_env(b, man, names)
    if not all(got.values()):
        raise SystemExit(
            f"[campaign] the launch table leaves {[n for n in names if not got[n]]} empty ({b['side']} path)"
        )
    return [got[n] for n in names]


env_of = RI.launch_env  # the launch env of a bundle (run_inputs evaluates the launch table under it)


def git_state() -> dict:
    """The code tree's git state a launch record carries: commit · branch · tree · dirty (tracked files only)."""

    def g(*a):
        return subprocess.run(["git", "-C", str(P.WT), *a], capture_output=True, text=True, timeout=120).stdout.strip()

    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "tree": str(P.WT),
        "dirty": bool(g("status", "--porcelain", "--untracked-files=no")),
    }


def loop_dir(campaign: str, root=None) -> Path:
    """The launch loop's own files (pidfile · status.json · states.tsv · events.tsv · params):
    run_rl_loops/<campaign>."""
    P.check_rl_campaign(campaign)
    return P.run_path("run_rl_loops", root=root) / campaign


def commands(
    b: dict,
    man: dict,
    mdir: Path,
    sdir: Path,
    judge_gpus: list[str],
    campaign: str,
    train_h: float,
    judge_conc: int,
    sidecar_deadline_h: float | None = None,
) -> dict:
    """Train and judge command lines of one bundle (mdir = the manifest copy in the campaign cell, sdir = the bundle
    cell, train_h = the bundle's training estimate, judge_conc = concurrent judges; the GPU and launch-id slots are
    filled by cell_state.launch).
    Train: renders off (DO_VIDEO=0; the cell's readers render later) under a timeout (TRAIN_TIMEOUT_X × train_h),
    rc → train.rc.
    Judge: the sidecar writes into the cell (RLJUDGE_OUT_DIR) under this launch's id (RLJUDGE_LAUNCH_ID) with the
    sidecar_env values."""
    run = run_name(campaign, b["hand"], b["id"])
    env = " ".join(env_of(b, man) + [f"DEXENGINE_SEQ_LIST={mdir / b['list']}"])
    to = int(TRAIN_TIMEOUT_X * train_h * 3600)
    train = (
        f"env {env} DO_VIDEO=0 GPU={CH.GPU_SLOT} timeout -k 120 {to} python3 {TRAIN_ENTRY} {run}; "
        f"echo $? > {CH.cell_file(sdir, 'train_rc')}"
    )
    judge = (
        f"env {env} RLJUDGE_GPUS='{' '.join(judge_gpus)}' RLJUDGE_OUT_DIR={sdir} "
        f"RLJUDGE_LAUNCH_ID={CH.LAUNCH_ID_SLOT} "
        f"{' '.join(sidecar_env(judge_conc, sidecar_deadline_h, hand=b['hand']))} "
        f"bash {SIDECAR} {run}"
        if judge_supported(b)
        else None
    )
    return {"run": run, "train": train, "judge": judge}


def cmd_of(mans: dict, mdirs: dict, judge_gpus: list[str], campaign: str, th, judge_conc: int):
    """(bundle, cell) → commands() of that bundle under this run's settings (th = the bundle's training estimate):
    what the loop launches."""
    return lambda b, sdir: commands(b, mans[b["hand"]], mdirs[b["hand"]], sdir, judge_gpus, campaign, th(b), judge_conc)


def load(mdir: Path, toks: list[str], defer: set[str]) -> tuple[dict, list[str]]:
    """Manifests of the order's hands (ledger mode only); deferred hands are not loaded → ({hand: manifest}, notes)."""
    mans, notes = {}, []
    for h in [t for t in toks if t != "@artic"]:
        f = mdir / h / "manifest.json"
        if h in defer:
            notes.append(f"{h}: deferred by --defer-hand" + ("" if f.is_file() else " (no manifest)"))
            continue
        man = json.loads(f.read_text(encoding="utf-8"))
        if man["mode"] != "ledger":
            raise SystemExit(f"[campaign] {h}: manifest mode {man['mode']} — only ledger mode is accepted (rebuild it)")
        mans[h] = man
    return mans, notes


def write_plan(
    path: Path,
    sched: list[dict],
    mans: dict,
    mdirs: dict,
    jg: list[str],
    root: Path,
    campaign: str,
    th,
    judge_conc: int,
) -> None:
    """The --plan-out TSV: one row per scheduled bundle with its slot, estimated times, cell, run and commands."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "bundle\thand\tclass\tclips\tsmall\tlaunch evidence\tslot\ttrain_start_h\ttrain_end_h\t"
            "judge_end_h\toverflow\tcell\trun\ttrain\tjudge\n"
        )
        for r in sched:
            sdir = CL.bundle_dir(r["hand"], campaign, r, root)
            c = commands(r, mans[r["hand"]], mdirs[r["hand"]], sdir, jg, campaign, th(r), judge_conc)
            f.write(
                f"{r['id']}\t{r['hand']}\t{r['klass']}\t{r['n']}\t{int(r['small'])}\t"
                f"ready\t{r['slot']}\t{r['start_h']}\t{r['train_end_h']}\t"
                f"{r['judge_end_h']}\t{int(r['overflow'])}\t{sdir.relative_to(root)}\t{c['run']}\t{c['train']}\t"
                f"{c['judge']}\n"
            )


def args(argv):
    """The runner's command line (module doc)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--manifests", type=Path, required=True, help="root of the per-hand manifests (<hand>/manifest.json)"
    )
    ap.add_argument(
        "--campaign", required=True, help="campaign id: the cell name in the RL campaign layer (no default)"
    )
    ap.add_argument(
        "--dry-root",
        type=Path,
        default=None,
        help="output tree for a dry run only (scratch: the plan and the --stage-hand cells). A launch "
        "writes only into this code tree (cells and work areas in one tree); dies with --launch",
    )
    ap.add_argument(
        "--judge-gpus",
        default=JUDGE_GPUS_DEFAULT,
        help="judge GPUs <host>:<index,…|last> (without a host: this machine's; default: its last GPU)",
    )
    ap.add_argument("--gpus", default="", help="training GPUs <host>:<index,…> (default = the judge host's others)")
    ap.add_argument(
        "--judge-overflow-wait-h",
        type=float,
        default=None,
        help="overflow threshold in hours: past it a run's finals run on its own training GPU (default off)",
    )
    ap.add_argument(
        "--prune-ckpt",
        choices=("off", "on"),
        default="off",
        help="finalize deletes every checkpoint outside the judged ones ∪ the last (default off)",
    )
    ap.add_argument(
        "--order",
        required=True,
        help="campaign order (registered hand ids, comma-separated · @artic = the articulated bundles of "
        "the hands before it); the hands are an input (no default)",
    )
    ap.add_argument("--defer-hand", default="", help="hands without a manifest, deferred explicitly (comma-separated)")
    ap.add_argument("--allow-class", default="", help="unblock classes that BLOCKERS blocks (comma-separated)")
    ap.add_argument(
        "--allow-no-judge",
        action="store_true",
        help="also queue bundles without a judge path (one-hand) — they never reach done (explicit choice)",
    )
    ap.add_argument("--plan-out", type=Path, default=None, help="plan TSV")
    ap.add_argument(
        "--stage-hand",
        default="",
        help="write the frozen manifest copy and the by_dataset index into the campaign cell for the named "
        "hands only (comma-separated); unprepared hands are not bound to the campaign (default none)",
    )
    ap.add_argument("--launch", action="store_true", help="launch loop (dexengine/cluster/local.py), after the checks")
    ap.add_argument("--tick-s", type=float, default=60.0, help="--launch: tick interval (s)")
    ap.add_argument("--once", action="store_true", help="--launch: one tick only (writes the status files and stops)")
    ap.add_argument(
        "--change-params",
        default="",
        help="--launch: why the loop starts with other settings than the frozen params.json",
    )
    ap.add_argument(
        "--allow-train-path-change",
        default="",
        help="--launch resume: why a code change of the training path is accepted (default refused)",
    )
    return ap.parse_args(argv)


def launch_loop(
    a,
    host: str,
    tg: list[str],
    jg: list[str],
    mans: dict,
    mdirs: dict,
    cdirs: dict,
    bundles: list,
    allow: set[str],
    th,
    per_run_gb: float,
) -> int:
    """--launch (module doc): the checks, then the loop (dexengine.cluster.local.run_loop). rc 3 = refused."""
    from dexengine.cluster import local as CLP
    from dexengine.cluster import scheduler as CT

    why = [w for k, w in BLOCKERS if k is None]
    if host != CH.this_host():
        why.append(f"this machine ({CH.this_host()}) is not the GPU host {host} — launch on the GPU host")
    for h in mans:
        if not (mdirs[h] / "manifest.json").is_file():
            why.append(f"{h}: no manifest copy in the campaign cell — run --stage-hand {h} first")
        else:
            CL.stage_manifest(a.manifests / h / "manifest.json", h, a.campaign)  # identical copy, else dies
    if a.allow_no_judge:
        why.append("--allow-no-judge — bundles without a judge path never reach done (dry runs only)")
    if why:
        print("[campaign] --launch refused:\n  " + "\n  ".join(why), file=sys.stderr)
        return 3
    cell_of = lambda b: CL.bundle_dir(b["hand"], a.campaign, b)  # noqa: E731 — one expression, two readers
    ctx = CT.Ctx(
        campaign=a.campaign,
        host=host,
        train_gpus=tg,
        judge_gpus=jg,
        overflow_h=a.judge_overflow_wait_h,
        prune=a.prune_ckpt == "on",
        root=P.WT,
        bundles=bundles,
        mans=mans,
        cdirs=cdirs,
        blocked={bkey(b): w for b in bundles if (w := campaign_block_reason(b, allow, a.campaign))},
        cell_of=cell_of,
        commands=cmd_of(mans, mdirs, jg, a.campaign, th, len(tg) + len(jg)),
        canary=lambda b, fp, ex: "",
        git=git_state,
        code=lambda: CH.code_tree(P.WT),
        train=lambda: "",
        per_run_gb=per_run_gb,
        inputs=lambda b: "",
    )
    print(
        f"[campaign] launch loop — campaign {a.campaign} · queue {len(bundles) - len(ctx.blocked)} · state "
        f"{loop_dir(a.campaign)}/status.json",
        flush=True,
    )
    return CLP.run_loop(ctx, a.tick_s, a.once, a.change_params, a.allow_train_path_change)


def main(argv=None) -> int:
    """Dry-run plan, or --launch (module doc)."""
    a = args(argv)
    if a.launch and a.dry_root is not None:
        raise SystemExit(
            "[campaign] --dry-root is for dry runs only — a launch writes only into this code tree's "
            "cells and work areas"
        )
    root = a.dry_root if a.dry_root is not None else P.WT
    P.check_rl_campaign(a.campaign)
    host, tg, jg = CH.gpu_split(a.judge_gpus if ":" in a.judge_gpus else f"{CH.this_host()}:{a.judge_gpus}", a.gpus)
    allow, bundles, cdirs, jc, mans, mdirs, per_run_gb, sched, th = _plan_campaign(a, host, jg, root, tg)
    if a.plan_out:
        write_plan(a.plan_out, sched, mans, mdirs, jg, root, a.campaign, th, jc)
    if a.launch:
        return launch_loop(a, host, tg, jg, mans, mdirs, cdirs, bundles, allow, th, per_run_gb)
    return 0


def _plan_campaign(a, host, jg, root, tg):
    """Resolve classes, bundles, manifests and the schedule of the campaign plan."""
    jc = len(tg) + len(jg)
    allow = {x for x in a.allow_class.split(",") if x}
    if unknown := allow - allow_tokens():
        raise SystemExit(f"[campaign] unknown classes in --allow-class {sorted(unknown)}")
    defer = {x for x in a.defer_hand.split(",") if x}
    have = {d.name for d in a.manifests.iterdir() if (d / "manifest.json").is_file()} if a.manifests.is_dir() else set()
    toks = parse_order(a.order, have - defer, defer)
    mans, notes = load(a.manifests, toks, defer)
    cdirs = {h: CL.campaign_dir(h, a.campaign, root) for h in mans}
    mdirs = {h: CL.manifest_dir(c) for h, c in cdirs.items()}
    CL.check_run_names(
        [run_name(a.campaign, h, b["id"], sm) for h, m in mans.items() for b in m["bundles"] for sm in (0, 1)]
    )
    stage = [x for x in a.stage_hand.split(",") if x]
    bad = [h for h in stage if h not in mans]
    if bad or len(set(stage)) != len(stage):
        raise SystemExit(
            f"[campaign] --stage-hand {stage}: only hands with a loaded manifest {sorted(mans)}, once each"
        )
    for h in stage:
        CL.stage_manifest(a.manifests / h / "manifest.json", h, a.campaign, root)
        f, n = CL.write_index(mans[h], a.campaign, root)
        print(f"[campaign] cell {h}: manifest copy {mdirs[h]} · index {f} ({n} clips)")
    byid = {(h, b["id"]): b for h, m in mans.items() for b in m["bundles"]}
    states = read_states(cdirs, lambda h, bid: CL.bundle_dir(h, a.campaign, byid[(h, bid)], root))
    bundles = order_bundles(mans, toks)
    if not bundles:
        raise InputMissingError("[campaign] no bundle in the queue — check --order and --defer-hand")
    keys = [bkey(b) for b in bundles]
    done, queue, blocked = campaign_view(mans, bundles, states, allow, a.campaign, a.allow_no_judge)
    mi, ckint = (int(x) for x in canon_value(bundles[0], mans[bundles[0]["hand"]], ("MI", "DEXENGINE_CKPT_INTERVAL")))
    per_run_gb, kept_gb = math.ceil(mi / ckint) * CH.CKPT_GB, JUDGE_FINALS * CH.CKPT_GB
    th = lambda b: train_hours(b["klass"], mi)  # noqa: E731 — the queue's one MI (canon_value above)
    md = lambda b: mid_judges(mi)  # noqa: E731 — the queue's one MI (canon_value above)
    sched, end = schedule(queue, len(tg), len(jg), th, mids=md, overflow_h=a.judge_overflow_wait_h)
    word = "plan before launch" if a.launch else "dry run — no launch"
    print(
        f"[campaign] {word} · campaign {a.campaign} · manifests {len(mans)} hands · bundles {len(keys)} · "
        f"states {len(states)} · done {len(done)}/{len(keys)}"
    )
    ov = "off" if a.judge_overflow_wait_h is None else f"wait > {a.judge_overflow_wait_h:g} h"
    print(f"[GPUS] host {host}: training {','.join(tg)} · judge {','.join(jg)} · overflow {ov} · prune {a.prune_ckpt}")
    for n in notes:
        print(f"[campaign] ⚠ {n}")
    for why, c in blocked.items():
        print(f"[campaign] blocked {c} bundles — {why}")
    print(
        f"[campaign] queue {len(queue)} bundles · MI {mi} (launch table) · end {end:.1f} h ({end / 24:.1f} d) "
        f"[estimate — training {TRAIN_H['_']} h · judge (finals {JUDGE_FINALS} + mid {mid_judges(mi)})×"
        f"{JUDGE_FINAL_H} h]"
    )
    print(
        "[campaign] split × overflow (same queue · same model):\n"
        + "\n".join(split_table(queue, len(tg) + len(jg), th, per_run_gb, kept_gb, md))
    )
    print(
        f"[campaign] disk: checkpoints per run {math.ceil(mi / ckint)} × {CH.CKPT_GB} = {per_run_gb:.2f} GB · after "
        f"prune {kept_gb:.2f} GB · whole queue unpruned {len(queue) * per_run_gb / 1e3:.2f} TB · "
        f"pruned {len(queue) * kept_gb / 1e3:.2f} TB"
    )
    print(
        f"[campaign] campaign-wide blocks {sum(k is None for k, _w in BLOCKERS)} rows"
        + "".join(f"\n  ⛔ {w}" for k, w in BLOCKERS if k is None)
        + "".join(f"\n[ALLOW_CLASS] {k}: {[w.split(':')[0] for c, w in BLOCKERS if c == k]}" for k in allow)
    )
    print(
        "[campaign] sidecar values (plan constants · same as the model): "
        f"{' · '.join(h + ' ' + ' '.join(sidecar_env(jc, hand=h)) for h in mans)} · "
        f"training timeout {TRAIN_TIMEOUT_X:g} × class estimate"
    )
    return allow, bundles, cdirs, jc, mans, mdirs, per_run_gb, sched, th


if __name__ == "__main__":
    sys.exit(main())
