"""Where an RL campaign keeps its results (the layer cells; paths come from dexengine.core.layout.RUN_LAYERS through
paths.run_path / paths.rl_bundle_dir) and the steps that fill them:

  campaign cell   run_rl_campaign(hand, campaign): manifest/ (manifest_dir: the hand manifest + lists, copied once,
                  sha-checked) · canary/ (canary_dir) · smoke/ (smoke_dir) · the by-dataset index (run_rl_index,
                  write_index)
  bundle cell     run_rl_bundle / run_rl_mixed (the cell rule is paths.rl_bundle_dir over cell_files.bundle_datasets):
                  cell_files.CELL_FILES (launch.json · train.{pid,rc,out} · …) · the sidecar's
                  <RUN>_judge.{tsv,rc,id,pid} and <alias>_judge_launch.out (RLJUDGE_OUT_DIR = this cell) · after
                  finalize: logs/ (cell_log_dir: the run's work-area log files) · skrl/<dir>/ (cell_ckpt_dir: its
                  checkpoint folders, pruned) · finalize.tsv (plan) + finalize.log (done steps) + FINALIZED.json
                  (which also carries the shadow verdict of older sidecars: OK … · MISSING … · off)
Training and judging write their logs and checkpoints to the trainer's work areas (cell_state.work_areas:
dexengine.core.layout.RUN_WORK_AREAS) while they run; `finalize` moves them here once the bundle is done (cell_state.
bundle_state), through root_guard.guarded_import, with a ledger. Renders, rescoring and collection read a finalized
run through cell_ckpt_dir / cell_log_dir.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

from dexengine.core import paths as P
from dexengine.rl import cell_files as CF
from dexengine.rl import cell_state as CH
from dexengine.rl import root_guard as HR
from dexengine.rl.cell_files import bundle_files, manifest_dir  # re-exported (manifest copy, bundle files)
from dexengine.rl.judge import record as CX

_ITER = CX.CKPT_NAME
FIN_TAG = "[FINALIZE_KEEP]"


def campaign_dir(hand: str, campaign: str, root=None) -> Path:
    """A hand's campaign cell."""
    return P.run_path("run_rl_campaign", hand, root=root, campaign=campaign)


def bundle_dir(hand: str, campaign: str, b: dict, root=None) -> Path:
    """The bundle's cell: the cell rule (paths.rl_bundle_dir) over its datasets (cell_files.bundle_datasets)."""
    return P.rl_bundle_dir(hand, campaign, CF.bundle_datasets(b), b["id"], root=root)


def canary_dir(cdir: Path) -> Path:
    """Canary result files inside a campaign cell."""
    return cdir / "canary"


def smoke_dir(cdir: Path) -> Path:
    """Boot smoke + launch probe results inside a campaign cell (one per run key)."""
    return cdir / "smoke"


def cell_ckpt_dir(sdir: Path) -> Path:
    """A finalized run's checkpoint folders inside its bundle cell (renders · rescoring · collection read here)."""
    return sdir / "skrl"


def cell_log_dir(sdir: Path) -> Path:
    """A finalized run's work-area logs inside its bundle cell."""
    return sdir / "logs"


def traj_record_dir(sdir: Path) -> Path:
    """The record-only pass's session pieces inside a finalized cell (cell_session record:
    <cell>/traj_record/<checkpoint>/: logs · command · record.json). The trajectory file itself sits in the cell as a
    judge piece (judge.metrics.TRAJ)."""
    return sdir / "traj_record"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_manifest(mfile: Path, hand: str, campaign: str, root=None) -> Path:
    """Copy the hand manifest and its lists and clip tables into the campaign cell (manifest/) once. A copy already
    there must be byte-identical (manifest and every list and table): a campaign runs on one frozen manifest; a
    different one raises."""
    dst = manifest_dir(campaign_dir(hand, campaign, root))
    HR.assert_writable(dst, hand, root=root)
    man = json.loads(mfile.read_text(encoding="utf-8"))
    if man["hand"] != hand:
        raise ValueError(f"{mfile}: manifest hand {man['hand']} ≠ {hand}")
    files = [Path("manifest.json")] + [Path(f) for f in bundle_files(man)]
    if dst.is_dir():
        bad = [str(f) for f in files if not (dst / f).is_file() or _sha(dst / f) != _sha(mfile.parent / f)]
        if bad:
            raise ValueError(
                f"{dst}: campaign {campaign}'s manifest copy differs {bad[:3]} — one campaign = one frozen manifest"
            )
        return dst
    tmp = dst.with_name(dst.name + ".tmp")
    for f in files:
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mfile.parent / f, tmp / f)
    HR.guarded_move(tmp, dst, hand, root=root)  # the whole copy appears at once (no half-staged manifest)
    return dst


def write_index(man: dict, campaign: str, root=None) -> tuple[Path, int]:
    """The by-dataset index (run_rl_index): one row per (dataset, bundle): primary clips of that dataset in the
    bundle, the bundle's run clips (primaries + dups), its cell (relative to the root), its dups. Σ over the rows =
    Σ bundle primaries = the manifest's target (a dup is counted once, in its primary bundle)."""
    hand = man["hand"]
    out = P.run_path("run_rl_index", hand, root=root, campaign=campaign)
    HR.assert_writable(out, hand, root=root)
    base = Path(root if root is not None else P.WT)
    rows = []
    for b in man["bundles"]:
        cell = bundle_dir(hand, campaign, b, root).relative_to(base)
        rows += [(ds, b["id"], k, b["n"], str(cell), len(b["dups"])) for ds, k in sorted(CF.bundle_datasets(b).items())]
    total, target = sum(r[2] for r in rows), sum(a["target"] for a in man["axes"])
    prim = sum(b["n_primary"] for b in man["bundles"])
    if total != prim or total != target:
        raise ValueError(f"index total {total} ≠ bundle primaries {prim} · target {target}")
    tmp = out.with_name(out.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("dataset\tbundle\tclips_of_dataset\tbundle_clips\tcell\tbundle_dups\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in sorted(rows))
    os.replace(tmp, out)
    return out, total


def write_launch_cmd(sdir: Path, cmds: dict) -> Path:
    """launch_cmd.txt: the exact train and judge command lines of this bundle (key=value lines)."""
    f = CH.cell_file(sdir, "launch_cmd")
    f.write_text("".join(f"{k}={cmds[k]}\n" for k in ("run", "train", "judge")), encoding="utf-8")
    return f


def check_run_names(names: list[str]) -> None:
    """A run's work-area files are `<RUN>_*` (finalize moves them by that shape), so no run name may begin with
    another's name + "_"; checked over every name of the campaign (runs and smokes) before anything launches. Across
    campaigns the campaign id is the name's first field and may not hold the separator (paths.RL_CAMPAIGN_SEP), so a
    name of one campaign is never the head of another's; a launch also proves its work area empty
    (cell_state.assert_clean_work_area)."""
    s = sorted(set(names))
    if len(s) != len(names):
        raise ValueError("run names repeat")
    bad = [(a, b) for a, b in zip(s, s[1:], strict=False) if b.startswith(a + "_")]
    if bad:
        raise ValueError(
            f"run name {bad[0][0]} is the head of {bad[0][1]} — their work-area files cannot be told apart"
        )


def _ckpt_saved_rows(run: str, ref: Path) -> list[tuple[str, int]]:
    """(name, iteration) of every checkpoint the run's train logs say was saved: the [CKPT_SAVED] parser
    (dexengine.rl.launch.records.ckpt_saved) over <RUN>_train*.log and the auto-resume <RUN>_R<iter>_train.log, in that
    order; an unreadable log is skipped. No log raises."""
    from dexengine.rl.launch import records as LR

    logs = sorted(ref.glob(f"{run}_train*.log")) + sorted(ref.glob(f"{run}_R[0-9]*_train.log"))
    if not logs:
        raise ValueError(f"{run}: no training log ({ref}) — [CKPT_SAVED] cannot be read")
    rows = []
    for log in logs:
        try:
            rows += LR.ckpt_saved(log.read_text(errors="replace"))
        except OSError:  # noqa: PERF203 — an unreadable log is skipped; the other logs still count
            continue
    return rows


def ckpt_saved(run: str, ref: Path) -> set[str]:
    """Checkpoint names the run's train logs say were saved (_ckpt_saved_rows)."""
    return {name for name, _ in _ckpt_saved_rows(run, ref)}


def ckpt_saved_last_iter(run: str, ref: Path) -> int:
    """The highest training iteration the run's logs say was saved: the trainer's own count ([CKPT_SAVED] … iter=N),
    the unit of MI. A checkpoint file's number is env steps (iteration × rollouts), not iterations. 0 = no train log
    yet or nothing saved yet (a run that has just started)."""
    if not (any(ref.glob(f"{run}_train*.log")) or any(ref.glob(f"{run}_R[0-9]*_train.log"))):
        return 0
    return max((it for _, it in _ckpt_saved_rows(run, ref)), default=0)


def finalize_plan(run: str, sdir: Path, prune: bool, root=None) -> list[tuple[str, Path, Path | None, int]]:
    """[(action, src, dst, bytes)] in apply order: every move first (the run's checkpoint folder, auto-resume pieces,
    judge aliases → cell_ckpt_dir; its work-area logs → cell_log_dir), then (prune) the deletes, addressed at their
    moved place inside the cell. Keep = the judged checkpoints only: the finals the sidecar's JUDGE row names (what
    judge.metrics.final_iters stated at its launch) ∪ the special, pending, best and last checkpoints of the judge's
    record (judge.record.kept_ckpts), one copy each, plus the training run's best_agent.pt when it trained with
    in-training eval on (judge.record.intrain_eval; off = skrl's training-reward pick, not kept). Every other *.pt of
    the moved checkpoint folders goes (saves between finals, auto-resume duplicates, the judge processes'
    best_agent.pt in their alias folders). Refuses unless the bundle is done (bundle_state), its finals are OK, the
    agent_*.pt names on disk == the [CKPT_SAVED] set, and nothing of the run in the checkpoint area has an unknown
    shape (cell_state.work_items, e.g. a render's <RUN>_vidB); also when nothing is pruned (the move is checked the
    same way)."""
    finals, keep = CX.judged_keep(run, sdir)
    ref, _ckr = CH.work_areas(root)
    it = CH.work_items(run, root)
    if it["unknown"]:
        raise ValueError(
            f"{run}: unknown shapes in the work area {[d.name for d in it['unknown']]} — neither moved nor kept"
        )
    if not it["ckpt"]:
        raise ValueError(f"{run}: no checkpoint folder ({_ckr / run})")
    dirs = it["ckpt"] + it["resume"]
    files = [f for d in dirs for f in sorted((d / "checkpoints").glob("agent_*.pt"))]
    names, saved = {f.name for f in files}, ckpt_saved(run, ref)
    if names != saved:
        raise ValueError(
            f"{run}: checkpoints in the folders {len(names)} ≠ [CKPT_SAVED] {len(saved)} "
            f"(folders only {sorted(names - saved)[:3]} · logs only {sorted(saved - names)[:3]})"
        )
    if not keep <= names:
        raise ValueError(f"{run}: checkpoints to keep {sorted(keep - names)} are not in the folders")
    ck, lg = cell_ckpt_dir(sdir), cell_log_dir(sdir)
    plan = [("move", d, ck / d.name, 0) for d in dirs + it["alias"]] + [("move", f, lg / f.name, 0) for f in it["logs"]]
    if not prune:
        return plan
    best = CX.intrain_eval(run, ref)
    first: set[str] = set()
    for d in dirs + it["alias"]:  # folder order (= the sidecar's ckpt_of order): the first copy is the one kept
        for f in sorted((d / "checkpoints").glob("*.pt")):
            if d in dirs and f.name in keep and f.name not in first:
                first.add(f.name)
                continue
            if best and f.name == CX.BEST_AGENT and d in it["ckpt"]:
                continue
            plan.append(("delete", ck / d.name / "checkpoints" / f.name, None, f.stat().st_size))
    gone = sum(a == "delete" for a, *_x in plan)
    bw = "kept (in-training eval on)" if best else "deleted (in-training eval off)"
    print(
        f"{FIN_TAG} {run} final judges {finals} · judge record {sorted(keep - set(finals)) or '-'} · "
        f"best_agent.pt {bw} · agents kept {len(first)} · deleted {gone}",
        flush=True,
    )
    return plan


def _read_plan(f: Path) -> list[tuple[str, Path, Path | None, int]]:
    """The rows of a finalize.tsv plan."""
    rows = [ln.split("\t") for ln in f.read_text(encoding="utf-8").splitlines()[1:] if ln]
    return [(a, Path(s), None if d == "-" else Path(d), int(n)) for a, s, d, n in rows]


#: The first words of the shadow verdict that older sidecars wrote: PENDING while the sidecar ran with the shadow on ·
#: OK <n> <clips|-> park_memo_flags=<m> · MISSING <reason>.
SHADOW_WORDS = ("PENDING", "OK", "MISSING")


def shadow_verdict(run: str, sdir: Path) -> tuple[str, str] | None:
    """(word, rest) of the cell's shadow verdict (cell_state.judge_shadow_file); None = no file (every current sidecar:
    the shadow judge is gone). A word outside SHADOW_WORDS raises."""
    f = CH.judge_shadow_file(run, sdir)
    if not f.is_file():
        return None
    w, _, rest = f.read_text(encoding="utf-8").strip().partition(" ")
    if w not in SHADOW_WORDS:
        raise ValueError(f"{f}: unknown shadow verdict {w!r} — {SHADOW_WORDS}")
    return w, rest


def finalized_shadow(fin: dict | None) -> dict | None:
    """The shadow verdict a FINALIZED record carries (`shadow`, written by finalize from shadow_verdict): OK <n>
    <clips|-> park_memo_flags=<m> → {flagged, clips}; MISSING · off · no record → None."""
    w = ((fin or {}).get("shadow") or "off").split()
    if w[0] not in SHADOW_WORDS + ("off",):
        raise ValueError(f"FINALIZED shadow {w[0]!r} — {SHADOW_WORDS} or off")
    return {"flagged": int(w[1]), "clips": [] if w[2] == "-" else w[2].split(",")} if w[0] == "OK" else None


def sequences_of(sdir: Path) -> dict:
    """Which sequences a bundle cell's checkpoints are for: the launch record's frozen clip record (cell_files.
    sequence_record, written by cell_state.launch_record): {dataset, clips, dups {clip: primary bundle},
    clips_sha256, table}. The per-clip rows must rebuild the table the sha256 names, row clips == clips, and a
    clips.tsv already in the cell must be that table. A record without it, or one that does not add up, raises."""
    rec = json.loads(CH.cell_file(sdir, "launch").read_text(encoding="utf-8"))
    if not all(k in rec for k in ("clips", "dups", "clip_rows", "clips_sha256")):
        raise ValueError(
            f"{sdir}: the launch record has no clip list — which sequences the checkpoints are for is unknown"
        )
    text = CF.table_text(rec["clip_rows"])
    tab = CH.cell_file(sdir, "clips")
    stale = tab.is_file() and tab.read_text(encoding="utf-8") != text
    if (
        hashlib.sha256(text.encode()).hexdigest() != rec["clips_sha256"]
        or [r[0] for r in rec["clip_rows"]] != rec["clips"]
        or stale
    ):
        raise ValueError(f"{sdir}: the clip record does not add up (table sha256 · row clips · {tab.name})")
    return {
        "dataset": rec["dataset"],
        "clips": rec["clips"],
        "dups": rec["dups"],
        "clips_sha256": rec["clips_sha256"],
        "table": text,
    }


def finalize(run: str, hand: str, sdir: Path, *, prune: bool, apply: bool = False, root=None) -> dict:
    """After done: move the run's work-area files into its bundle cell, then (prune) delete every checkpoint but the
    kept set there. The plan goes to finalize.tsv first; with apply, every step is done through root_guard
    (guarded_import for the moves work area → cell, guarded_rmtree for the deletes inside the cell) and appended to
    finalize.log, then FINALIZED.json. Under a cell lock. An interrupted apply resumes from finalize.tsv +
    finalize.log (a step already done is skipped, one that is half-way raises); a finalized cell returns its record.
    FINALIZED.json carries the sequences the kept checkpoints are for (`sequences_of`) and the sha256 of the judge's
    record it kept them by (`extras_sha256` = judge.record.FIN_SHA_KEY · '-' = no record; the record is read before
    anything moves and its values are not copied here). The plan header names the same sha and a resume under other
    extras raises; a checkpoint the record keeps that is not kept after the apply raises before FINALIZED. A
    destination that exists raises (no overwrite). root = the tree of the cell and the work areas."""
    tsv, log, fin = (CH.cell_file(sdir, k) for k in ("finalize_tsv", "finalize_log", "finalized"))
    HR.assert_writable(tsv, hand, root=root)
    with open(CH.cell_file(sdir, "finalize_lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if fin.is_file():
            return json.loads(fin.read_text(encoding="utf-8"))
        plan, res, resume, x = _finalize_inputs(run, sdir, prune, apply, root, log, tsv)
        if not apply:
            return res
        done = set(log.read_text(encoding="utf-8").splitlines()) if resume else set()
        with open(log, "a", encoding="utf-8") as lg:
            for a, s, d, _n in plan:
                key = f"{a}\t{s}"
                if key in done:
                    continue
                if a == "move":
                    if resume and not s.exists() and d.exists():
                        pass  # moved before the interruption, not yet logged
                    else:
                        HR.guarded_import(s, d, hand, root=root)
                elif s.exists():
                    HR.guarded_rmtree(s, hand, root=root)
                lg.write(key + "\n")
                lg.flush()
        res["kept"] = sorted(str(f.relative_to(sdir)) for f in cell_ckpt_dir(sdir).glob("*/checkpoints/agent_*.pt"))
        lost = CX.kept_ckpts(x) - {Path(k).name for k in res["kept"]}
        if lost:
            raise ValueError(
                f"{run}: checkpoints of the judge record (special · pending · best · last) {sorted(lost)} "
                "were not kept — FINALIZED is not written"
            )
        seq = sequences_of(sdir)
        CH.cell_file(sdir, "clips").write_text(seq.pop("table"), encoding="utf-8")
        res.update(seq, clips_table=CH.CELL_FILES["clips"])
        fin.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return res


def _finalize_inputs(run, sdir, prune, apply, root, log, tsv):
    """Sequences, judge extras, plan and resume state of the run being finalized."""
    seq = sequences_of(sdir)  # before anything moves: the sequences these checkpoints are for
    x = CX.judge_extras(run, sdir, seq["clips"])
    xsha = x.sha256
    resume = log.is_file()
    if resume:
        st = CH.bundle_state(sdir, run)
        if st != "done":
            raise ValueError(f"{run}: state {st} — an interrupted finalize is not resumed")
        plan = _read_plan(tsv)
        said = tsv.read_text(encoding="utf-8").split("\n", 1)[0].split("\t")[4:] or ["extras=-"]
        if said != [f"extras={xsha}"]:
            raise ValueError(
                f"{run}: the judge record (special checkpoints) changed after the interrupted plan "
                f"({said[0]} → {xsha[:12]}) — the keep set would differ, not resumed"
            )
        if prune != any(a == "delete" for a, *_x in plan) and any(a == "delete" for a, *_x in plan):
            raise ValueError(
                f"{run}: the interrupted finalize planned prune {'on' if not prune else 'off'} — "
                "resume with the same value"
            )
    else:
        plan = finalize_plan(run, sdir, prune, root)
        clash = [d for a, _s, d, _n in plan if a == "move" and (d.exists() or d.is_symlink())]
        if clash:
            raise ValueError(f"{run}: move targets already exist {clash[:2]} — not overwritten")
        with open(tsv, "w", encoding="utf-8") as out:
            out.write(f"action\tsrc\tdst\tbytes\textras={xsha}\n")  # what the keep set read (resume compares)
            out.writelines(f"{a}\t{s}\t{d or '-'}\t{n}\n" for a, s, d, n in plan)
    sv = shadow_verdict(run, sdir)  # done + PENDING = the sidecar never closed its shadow
    if sv is not None and sv[0] == "PENDING":
        raise ValueError(f"{run}: done with the shadow verdict PENDING — the sidecar never closed its shadow rc")
    res = dict(
        run=run,
        prune=prune,
        shadow=" ".join(sv).strip() if sv else "off",
        delete=sum(a == "delete" for a, *_x in plan),
        move=sum(a == "move" for a, *_x in plan),
        freed_gb=round(sum(n for a, _s, _d, n in plan if a == "delete") / 1e9, 2),
        applied=apply,
        resumed=resume,
        **{CX.FIN_SHA_KEY: xsha},
    )
    return plan, res, resume, x
