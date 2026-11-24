"""Where a launch writes its records, and the record formats other programs read.

Both places are work areas of the launch's run tree (Request.run_tree = paths.run_root_of(the trainer's data root,
the code tree): the data root for a release or copied data root, the code tree for the canonical checkout) — never
the cwd:
* log folder `<run tree>/overfit/logs/refbase` (a dry run writes its twin `overfit/logs/refbase.dryrun`): trainer
  logs `<exp>_train.log`, the launch manifest `<out>_cmd.txt` and the summary tables `bi_summary.tsv` /
  `uni_lag_summary.tsv` (the judge sidecar watches them for training failures).
* run folder `<run tree>/<RUN_CKPT_ROOT>/<run>` (layout.RUN_CKPT_ROOT): checkpoints, the collision-preset record
  `coll_preset.txt` and the switches record.

The manifest is a key=value text; the judge sidecar compares a judge's manifest with its training's (tokens
`code= codedirty= MI= NE= COLL_PRESET= RESET_SCHEME= SIDE=` and the `EXTRA=` / `TRAIN_EXTRA=` lines) and data
recording reads the trainer environment lines (`DEXENGINE_…=`). Paths written into records (ROSTER_RECORD=) stay
relative to the run tree; the code-tree fields (REPO_ROOT=, code=, git=) name the code tree.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
import socket
import subprocess
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.layout.runs import RUN_CKPT_ROOT, RUN_WORK_AREAS

#: Run folders of every training (checkpoints, run records) — the on-disk name existing runs use.
RUN_ROOT = RUN_CKPT_ROOT
#: Launch log folder, relative to the run tree (the name in records; the place is log_dir).
LOG_DIR = RUN_WORK_AREAS["work_log"]
#: Code paths whose git tree id identifies "the code a run ran with" (manifest `code=`).
CODE_PATHS = ("dexengine", "scripts", "configs")
_SAVED = re.compile(r"\[CKPT_SAVED\] (agent_[0-9]+\.pt) iter=([0-9]+)")


def log_dir(dry_run: bool, tree=None) -> Path:
    """Launch log folder (absolute: the work area work_log of run tree `tree` — Request.run_tree; None = this
    process's paths.run_root()), or its dry-run twin (a dry run never overwrites a real launch's records)."""
    return P.mode_record(P.work_area("work_log", tree), "dryrun" if dry_run else None)


def arg_path(req, p) -> str:
    """A run-tree path as the trainer's argv and the launch records carry it: relative when the run tree is the code
    tree the trainer runs from (req.run_tree == req.root — the canonical checkout; the form existing records hold, read
    the same against the trainer's cwd and against its run tree, paths.run_out), else absolute."""
    p, rt = Path(p), req.run_tree
    if Path(os.path.realpath(rt)) == Path(os.path.realpath(req.root)) and p.is_relative_to(rt):
        return str(p.relative_to(rt))
    return str(p)


def run_dir(run: str, tree=None) -> Path:
    """Run folder of a training (absolute: the work area work_ckpt of run tree `tree`, as log_dir)."""
    return P.work_area("work_ckpt", tree) / run


def summary_tsv(path: str) -> str:
    """Summary table name of a recipe path (uni|bi)."""
    return "bi_summary.tsv" if path == "bi" else "uni_lag_summary.tsv"


SUMMARY_HEAD = {
    "bi": "run\tseq\tne\tmi\tjoint_rate\ttrain_sec\thost",
    "uni": "run\tseq\tside\tne\tmi\trate\ttrain_sec\thost",
}


def append_summary(folder: Path, path: str, fields: list, tail: str = "") -> None:
    """Append one row `fields + [host] (+ tail)` to the summary table in `folder` (log_dir; header first when the table
    is new)."""
    f = Path(folder) / summary_tsv(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    new = not f.exists()
    with open(f, "a") as fh:
        if new:
            fh.write(SUMMARY_HEAD[path] + "\n")
        fh.write("\t".join([*(str(x) for x in fields), socket.gethostname(), *([tail] if tail else [])]) + "\n")


def _git(root: Path, *args: str, stdin: str | None = None) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def code_tree(root: Path) -> str:
    """12-hex id of the committed code paths (the judge compares it between training and judge): the git blob id of
    the `git ls-tree HEAD` listing. Outside a git checkout the listing is empty and the id is the empty blob's."""
    data = _git(root, "ls-tree", "HEAD", "--", *CODE_PATHS).encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()[:12]


def code_dirty(root: Path) -> int:
    """Number of uncommitted changes under the code paths."""
    out = _git(root, "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS)
    return len([ln for ln in out.splitlines() if ln.strip()])


def manifest_head(root: Path, mode: str, out: str) -> str:
    """First manifest line: date, commit, dirty count, code tree id, mode and output name."""
    date = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    git = _git(root, "rev-parse", "--short", "HEAD").strip()
    dirty = len([ln for ln in _git(root, "status", "--porcelain").splitlines() if ln.strip()])
    return (
        f"date={date} git={git} dirty={dirty} code={code_tree(root)} codedirty={code_dirty(root)} MODE={mode} OUT={out}"
    )


def write_manifest(path: Path, body: str) -> None:
    """Publish a manifest without overwriting an earlier one: the previous file of the same name is first kept as
    `<name>.<YYYYMMDD_HHMMSS>[_n].txt` (its own date), then the new body is linked in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        m = re.match(r"date=(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)", path.read_text().split("\n", 1)[0])
        if not m:
            raise SystemExit(f"[launch] earlier manifest {path} has no date= first line — not a launch record")
        stamp = "".join(m.groups()[:3]) + "_" + "".join(m.groups()[3:])
        for n in ["", *range(1, 1000)]:
            keep = path.with_name(f"{path.stem}.{stamp}{f'_{n}' if n != '' else ''}.txt")
            try:
                fd = os.open(keep, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.close(fd)
            os.replace(path, keep)
            print(f"[launch] kept earlier manifest: {keep}")
            break
        else:
            raise SystemExit(f"[launch] no free name to keep the earlier manifest {path}")
    tmp = path.with_name(path.name + f".writing.{os.getpid()}")
    tmp.write_text(body)
    os.replace(tmp, path)


def ckpt_saved(text: str) -> list[tuple[str, int]]:
    """(checkpoint name, iteration) of every `[CKPT_SAVED]` marker in a trainer log, in order."""
    return [(m.group(1), int(m.group(2))) for m in _SAVED.finditer(text)]


def coll_preset_marker(tree: Path, run: str, render_rundir: str = "") -> Path:
    """The collision-preset record of a run (the folder that holds checkpoints/; `tree` = the run tree)."""
    return Path(render_rundir) / "coll_preset.txt" if render_rundir else run_dir(run, tree) / "coll_preset.txt"


def check_coll_preset(
    tree: Path, run: str, preset: str, what: str, *, fresh: bool = False, render_rundir: str = ""
) -> None:
    """Refuse to score or render a checkpoint under a different collision preset than it trained with (exit 7).

    `fresh` = the first training launch, before its record is written: a missing record passes. Any other launch
    without the record is refused too (exit 7): the preset the checkpoint trained under is unknown, never "none"."""
    f = coll_preset_marker(tree, run, render_rundir)
    if not f.exists():
        if fresh:
            return
        print(
            f"[launch] {what} refused: no collision-preset record {f} — the preset this checkpoint trained under is "
            "unknown (the record sits in the run folder by run name; restore that folder before judging or rendering)",
            flush=True,
        )
        raise SystemExit(7)
    rec = f.read_text().split("\n", 1)[0].strip()
    cur = preset or "none"
    if rec != cur:
        print(
            f"[launch] {what} refused: this run trained with COLL_PRESET={rec}, the launch has COLL_PRESET={cur} "
            f"— scoring one checkpoint under another collision physics is a train-test mismatch ({f})",
            flush=True,
        )
        raise SystemExit(7)


def switches_record(tree: Path, run: str) -> Path:
    """The switches record of a training run (next to its checkpoints; `tree` = the run tree): the training process's
    switches file. Judges of the run's checkpoints read it (`--switches-from`), so they run under the training's
    switches."""
    return run_dir(run, tree) / "switches.json"


def write_switches_record(tree: Path, run: str, launch) -> Path:
    """Write the switches record of a training run from its training process."""
    from dexengine.rl.tasks import switches  # stdlib only

    return switches.write(switches_record(tree, run), launch.switches, launch.given)


def write_coll_preset(tree: Path, run: str, preset: str, env: dict, train_flags: str, code: Path) -> None:
    """Write the collision-preset record next to the run's checkpoints (`tree` = the run tree; `code` = the code tree
    whose commit the record names)."""
    f = coll_preset_marker(tree, run)
    f.parent.mkdir(parents=True, exist_ok=True)
    date = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    f.write_text(
        f"{preset}\n"
        "# collision preset of this run — judge, replay and render must use the same value\n"
        f"# env: SPLIT_ARTIC={env['DEXENGINE_BI_SPLIT_ARTIC']} SELFCOLL={env['DEXENGINE_BI_SELFCOLL']} "
        f"DESK_NOCOLL={env['DEXENGINE_DESK_ROBOT_NOCOLL']} ARM_OBJ_NOCOLL={env['DEXENGINE_ARM_OBJ_NOCOLL']} "
        f"EVAL_CROSS_OFF={env['DEXENGINE_EVAL_CROSS_OFF']}\n"
        f"# train flags: {train_flags}\n"
        f"# date={date} git={_git(code, 'rev-parse', '--short', 'HEAD').strip()}\n"
    )


def main(argv=None) -> int:
    """Shell access for the judge sidecar: `summary-tsv SIDE` prints the summary table path (this process's run tree,
    in the --rel form paths.run_rel: code-tree relative when the run tree is the code tree, else absolute — the sidecar
    runs with the training environment, from the code tree);
    `ckpt-saved LOG...` prints `agent_<n>.pt <iter>` for every `[CKPT_SAVED]` marker of the logs, in order."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary-tsv").add_argument("side", choices=("right", "left", "both"))
    sub.add_parser("ckpt-saved").add_argument("logs", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "summary-tsv":
        folder = P.run_rel(log_dir(os.environ.get("DRYRUN") == "1"))
        print(f"{folder}/{summary_tsv('bi' if a.side == 'both' else 'uni')}")
        return 0
    for log in a.logs:
        try:
            text = Path(log).read_text(errors="replace")
        except OSError:
            continue
        for name, it in ckpt_saved(text):
            print(name, it)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
