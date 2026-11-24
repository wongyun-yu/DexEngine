"""Host-side pieces of the RL campaign runner (dexengine/rl/campaign.py): the one bundle state reader
(bundle_state), the trainer's work areas and their clean check, launch records, the host gate client (claims, the
judge-only reservation, the live claim list), launching, liveness (dexengine.cluster.proc), the code tree fingerprint
and the host label.

Files of a bundle cell (cells.bundle_dir): CELL_FILES (launch.json · train.pid · train.rc · …) and the judge
sidecar's <RUN>_judge.{rc,tsv,id} (judge_files; RLJUDGE_OUT_DIR = the cell). The state is derived from those files.
"""

from __future__ import annotations

import glob
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from dexengine.cluster import proc as PR
from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.rl import cell_files as CF
from dexengine.rl import root_guard as HR
from dexengine.rl.cell_files import CELL_FILES, cell_file  # re-exported (cell file names)
from dexengine.rl.judge import metrics as JM

STATES = ("queued", "training", "trained", "judging", "done", "failed")
GATE = P.WT / "scripts/rl/host_gate.sh"  # the host gate (trainings and judges)
#: Placeholders of a launch command, filled by `launch`: the GPU the gate hands out · the cell's launch id.
GPU_SLOT, LAUNCH_ID_SLOT = "{GPU}", "{LAUNCH_ID}"
#: A sidecar that dies without a judge rc is started again by the loop (same launch id, it resumes its table) at most
#: this many starts in all (judge.starts, one line per start); past it the bundle is failed (bundle_state).
SIDECAR_MAX_STARTS = 3
#: One training's host RAM in GB (PSS ≈ 46 GB; an RSS would count the 33 inductor workers' shared pages) and the
#: seconds it is assumed not to have allocated it yet.
TRAIN_RESERVE_GB, TRAIN_RAMP_S = 46, 600
#: agent_*.pt size in GB (160,940,109 B measured) for disk planning and the disk gate.
CKPT_GB = 0.161
#: The judge sidecar's final judges (count from judge.metrics.N_FINALS; the row label park_last3 is only an id).
JUDGE_FINALS = JM.N_FINALS
JUDGE_KINDS = "qfs"  #: the sidecar's judge names <RUN>_j<kind><iter>: q quick · f final · s shadow (old cells)
#: The host label of this machine when the layout file names no campaign hosts (the release default).
LOCAL_HOST = "local"
#: Longest wait on one host gate call in s (its flock wait + at most one 30 s nvidia-smi per candidate GPU): a hung
#: gate raises subprocess.TimeoutExpired instead of waiting forever.
GATE_TIMEOUT_S = 900


class UncommittedTreeError(ValueError):
    """The code tree a launch would run from has uncommitted changes (launch_record). Launches run from a committed
    state only; the scheduler stops launching on this error instead of trying the next bundle."""


def hosts() -> dict[str, str]:
    """Campaign host label -> hostname. A runner launches only on the host its GPU specs name. A site names its
    machines in its layout file (`hosts`, DEXENGINE_LAYOUT_CONFIG); without that table this machine is the only host,
    labelled LOCAL_HOST."""
    from dexengine.core import layout

    return dict(layout.layout_config().hosts) or {LOCAL_HOST: socket.gethostname()}


def this_host() -> str:
    """The label of the machine this process runs on — dies on a machine the host table does not name."""
    h, table = socket.gethostname(), hosts()
    for k, v in table.items():
        if v == h:
            return k
    raise InputMissingError(
        f"[campaign] this machine ({h}) is not in the campaign hosts {table} (layout file "
        "`hosts`) — its GPUs belong to no known host"
    )


def host_gpu_count(label: str) -> int:
    """GPUs of a host, measured (nvidia-smi -L): locally on that host, else with the layout file's
    `host_gpu_count_cmd` of that host (run from the code tree). No answer dies; a GPU count is never assumed."""
    from dexengine.core import layout

    table = hosts()
    if label not in table:
        raise SystemExit(f"[campaign] unknown host {label!r} — {sorted(table)}")
    remote = layout.layout_config().host_gpu_count_cmd.get(label, "")
    if label == this_host():
        cmd = ["bash", "-c", "nvidia-smi -L | grep -c '^GPU '"]
    elif remote:
        cmd = ["bash", "-c", remote]
    else:
        raise InputMissingError(
            f"[campaign] no way to measure the GPUs of host {label} from this machine (layout "
            "file `host_gpu_count_cmd` names none)"
        )
    r = subprocess.run(cmd, cwd=str(P.WT), capture_output=True, text=True, timeout=60)
    n = r.stdout.strip().splitlines()[-1:] if r.returncode == 0 else []
    if not n or not n[0].isdigit() or int(n[0]) < 1:
        raise SystemExit(
            f"[campaign] could not measure the GPU count of host {label} (rc {r.returncode}): {r.stderr[-200:]}"
        )
    return int(n[0])


def gpu_split(judge_spec: str, train_spec: str = "", n_gpus: int | None = None) -> tuple[str, list[str], list[str]]:
    """(host, training GPUs, judge GPUs) from host-qualified specs `<host>:<i,j,…>` (`last` = the host's last GPU).
    judge_spec names the judge GPUs; train_spec "" = every other GPU of that host. The host's GPU count is measured
    (host_gpu_count) unless given. Both sides non-empty, disjoint, on one host, in range; else dies."""

    def parse(spec: str) -> tuple[str, list[str]]:
        host, sep, gs = spec.partition(":")
        if not sep or host not in hosts() or not gs:
            raise SystemExit(f"[campaign] GPU spec {spec!r} — form <host>:<index,…|last>, hosts {sorted(hosts())}")
        return host, [x for x in gs.split(",") if x]

    host, jg = parse(judge_spec)
    n = n_gpus if n_gpus is not None else host_gpu_count(host)
    jg = [str(n - 1) if x == "last" else x for x in jg]
    if train_spec:
        th, tg = parse(train_spec)
        if th != host:
            raise SystemExit(f"[campaign] training GPU host {th} ≠ judge GPU host {host} — one host only")
    else:
        tg = [str(i) for i in range(n) if str(i) not in jg]
    bad = [x for x in tg + jg if not x.isdigit() or int(x) >= n]
    if bad or not tg or not jg or set(tg) & set(jg) or len(set(tg)) != len(tg) or len(set(jg)) != len(jg):
        raise SystemExit(
            f"[campaign] {host} ({n} GPUs): training {tg} · judge {jg} — empty, overlapping or out of range {bad}"
        )
    return host, tg, jg


def bkey(b: dict) -> tuple[str, str]:
    """A bundle's campaign key (hand, id): ids repeat across hands (e.g. taco.both.01 in every hand)."""
    return b["hand"], b["id"]


alive, stop = PR.alive, PR.stop  #: the liveness rule and its stop (dexengine.cluster.proc)


def pid_of(f: Path) -> int | None:
    """The pid in a pid file's first word (None: no file or no number)."""
    try:
        return int(f.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def judge_files(run: str, sdir: Path) -> tuple[Path, Path, Path]:
    """(<RUN>_judge.rc, <RUN>_judge.tsv, <RUN>_judge.id): what the judge sidecar (scripts/rl/judge_sidecar.sh) writes
    into its RLJUDGE_OUT_DIR, the bundle cell."""
    return sdir / f"{run}_judge.rc", sdir / f"{run}_judge.tsv", sdir / f"{run}_judge.id"


def judge_shadow_file(run: str, sdir: Path) -> Path:
    """<RUN>_judge_shadow.rc: the NE64 shadow verdict that older sidecars wrote (no longer written; absent = off)."""
    return sdir / f"{run}_judge_shadow.rc"


def judge_extras_file(run: str, sdir: Path) -> Path:
    """<RUN>_judge_extras.json: the judge step's per-clip results beyond park_last3 (10 % completion per clip ·
    special checkpoints), written into the cell before its DONE rc so finalize sees it before it prunes. Writer
    dexengine/rl/judge/metrics.py, reader dexengine.rl.judge.record.judge_extras (absent = not measured)."""
    return sdir / f"{run}_judge_extras.json"


def work_areas(root=None) -> tuple[Path, Path]:
    """(log area, checkpoint area) of the trainer under run tree `root` (default paths.run_root() — the same base as
    root_guard, whose guarded_import takes pieces only from these areas): paths.work_area work_log · work_ckpt."""
    return P.work_area("work_log", root), P.work_area("work_ckpt", root)


def work_items(run: str, root=None) -> dict[str, list[Path]]:
    """Everything of run `run` in the work areas, by shape. Checkpoint area: the run folder `<RUN>` (ckpt) · auto-resume
    pieces `<RUN>_R<iter>` (resume) · judge aliases `<RUN>_j<JUDGE_KINDS><iter ≥4 digits>[_frozenreplay]` (alias; the
    second is the folder the trainer makes for that judge) · any other `<RUN>_*` (unknown, e.g. a render's
    `<RUN>_vidB`; campaign trainings render nothing, so this is someone else's or a leftover). Log area: every
    `<RUN>_*` (logs; the launcher writes shapes that cannot be listed, and the run name is campaign-scoped,
    campaign.run_name)."""
    ref, ckr = work_areas(root)
    res = re.compile(re.escape(run) + r"_R\d+")
    alias = re.compile(re.escape(run) + f"_j[{JUDGE_KINDS}]" + r"\d{4,}(?:_frozenreplay)?")
    out: dict[str, list[Path]] = {
        "ckpt": [ckr / run] if (ckr / run).exists() else [],
        "resume": [],
        "alias": [],
        "unknown": [],
        "logs": sorted(ref.glob(glob.escape(run) + "_*")),
    }
    for d in sorted(ckr.glob(glob.escape(run) + "_*")):
        out["resume" if res.fullmatch(d.name) else "alias" if alias.fullmatch(d.name) else "unknown"].append(d)
    return out


def assert_clean_work_area(run: str, root=None) -> None:
    """A launch starts from nothing of its run in the work areas (cold start): any `<RUN>`, `<RUN>_*` in the
    checkpoint area or `<RUN>_*` in the log area raises, because an earlier attempt's pieces would be read as this
    attempt's (resume, [CKPT_SAVED], finalize)."""
    got = [p for v in work_items(run, root).values() for p in v]
    if got:
        more = max(len(got) - 3, 0)
        raise ValueError(
            f"{run}: the work areas already hold pieces of this run {[str(p) for p in got[:3]]}(+{more}) "
            "— only a cold start is launched (clearing an old attempt is the relaunch tool's job)"
        )


def cell_leftovers(sdir: Path) -> list[str]:
    """Names in a bundle cell other than its attempts/ (the relaunch tool's); empty = an empty cell."""
    return sorted(x.name for x in sdir.iterdir() if x.name != CELL_FILES["attempts"]) if sdir.is_dir() else []


def launch_id_of(sdir: Path) -> str | None:
    """The launch id this cell's launch record carries (None: no record, or a record without one)."""
    f = cell_file(sdir, "launch")
    return json.loads(f.read_text(encoding="utf-8")).get("launch_id") if f.is_file() else None


def judge_verdict(run: str, sdir: Path) -> tuple[str, str | None] | None:
    """(word, launch id) of the sidecar's <RUN>_judge.rc: word DONE · FAIL · INCOMPLETE (anything else raises), launch
    id from its `launch=<id>` line (None when the sidecar ran without RLJUDGE_LAUNCH_ID). None = no rc."""
    rc = judge_files(run, sdir)[0]
    if not rc.is_file():
        return None
    lines = rc.read_text(encoding="utf-8").splitlines()
    w = (lines[0].split() if lines else []) or ["?"]
    if w[0] not in ("DONE", "FAIL", "INCOMPLETE"):
        raise ValueError(f"{rc}: unknown judge rc {w[0]!r}")
    lid = next((ln[len("launch=") :] for ln in lines[1:] if ln.startswith("launch=")), None)
    return w[0], lid


def bundle_state(sdir: Path, run: str) -> str:
    """The bundle's state, derived from its cell's files (the runner, the judge backlog and finalize ask this):
    a judge rc whose launch id ≠ the launch record's raises (another attempt's table) ·
    live training process (train.pid, its post-training steps included) → training ·
    train.rc ≠ 0 → failed · judge rc DONE → done only with train.rc 0 (a DONE while the training died after its
    last checkpoint is not done), FAIL / INCOMPLETE → failed · live sidecar → judging · train.rc 0 → trained
    (the loop starts the sidecar again) unless its starts (judge.starts) reached SIDECAR_MAX_STARTS → failed ·
    launched but nothing alive and no rc → failed (dead) · else queued."""
    v = judge_verdict(run, sdir)
    lid = launch_id_of(sdir)
    if v is not None and v[1] != lid:
        raise ValueError(
            f"{sdir}: the judge rc's launch id {v[1]!r} ≠ the launch record's {lid!r} — another attempt's judge table"
        )
    trc = cell_file(sdir, "train_rc")
    rc = trc.read_text().strip() if trc.is_file() else None
    tp, jp = pid_of(cell_file(sdir, "train_pid")), pid_of(cell_file(sdir, "judge_pid"))
    if tp is not None and alive(tp, run):
        return "training"
    if rc not in (None, "0"):
        return "failed"
    if v is not None:
        return "done" if v[0] == "DONE" and rc == "0" else "failed"
    if jp is not None and alive(jp, run):
        return "judging"
    if rc == "0":
        return "failed" if sidecar_starts(sdir) >= SIDECAR_MAX_STARTS else "trained"
    return "failed" if cell_file(sdir, "launch").is_file() else "queued"


def failure_kind(sdir: Path, run: str) -> str | None:
    """A failed bundle's cause: 'train' (train.rc ≠ 0) · 'judge_invalid' (rc FAIL … INVALID — re-judge) · 'judge'."""
    if bundle_state(sdir, run) != "failed":
        return None
    trc, rc = cell_file(sdir, "train_rc"), judge_files(run, sdir)[0]
    if trc.is_file() and trc.read_text().strip() != "0":
        return "train"
    first = (rc.read_text(encoding="utf-8").splitlines() or [""])[0] if rc.is_file() else ""
    return "judge_invalid" if first.startswith("FAIL") and "INVALID" in first else "judge"


def sidecar_starts(sdir: Path) -> int:
    """How many times the cell's sidecar was started (judge.starts — the loop appends a line per start)."""
    f = cell_file(sdir, "judge_starts")
    return len([ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]) if f.is_file() else 0


def read_states(cdirs: dict[str, Path], cell_of) -> dict[tuple[str, str], str]:
    """{(hand, id): state} of every bundle cell with a launch record under each hand's campaign cell ({hand: dir});
    a cell is <campaign>/<dataset or mixed>/<bundle>, two levels down. cell_of(hand, id) = the cell the rule puts that
    bundle in (cells.bundle_dir over the campaign's manifest; a bundle outside it raises). Raises on a record that
    names another hand or bundle than where it lies, a cell not where the rule puts it, or a second record of the same
    (hand, bundle): no silent merge (the count match of queue.complete is a set match)."""
    out: dict[tuple[str, str], str] = {}
    for hand, cdir in cdirs.items():
        for f in sorted(cdir.glob("*/*/" + CELL_FILES["launch"])):
            rec = json.loads(f.read_text(encoding="utf-8"))
            key = (rec.get("hand"), rec.get("bundle"))
            if key != (hand, f.parent.name):
                raise ValueError(f"{f}: launch record {key} ≠ place ({hand}, {f.parent.name})")
            if key in out:
                raise ValueError(f"{f}: two launch records of the same bundle {key} — not merged")
            want = Path(os.path.abspath(cell_of(hand, key[1])))
            if Path(os.path.abspath(f.parent)) != want:
                raise ValueError(
                    f"{f.parent}: the cell is not where the rule (cells.bundle_dir) puts it — rule = {want}"
                )
            out[key] = bundle_state(f.parent, rec["run"])
    return out


def launch_record(sdir: Path, b: dict, man: dict, run: str, git: dict, root=None) -> dict:
    """<sdir>/launch.json: what a launch ran on. The code tree · commit · branch · dirty (`git`, the caller's git
    state of the tree it launches from), the manifest's data fingerprint, the bundle axis' tree fingerprint, a fresh
    launch id (handed to the sidecar as RLJUDGE_LAUNCH_ID; its rc must carry it back, bundle_state) and which
    sequences this run's checkpoints are for (cell_files.sequence_record: clips · dups (clip → primary bundle) ·
    per-clip rows · table sha256; finalize writes them out as clips.tsv beside the kept checkpoints). Refused: a dirty
    tree (launches run from a committed state) · a cell that is not empty (an earlier attempt's record or judge
    table; re-launching is the relaunch tool's, which clears them with a ledger) · anything of the run in the work
    areas (assert_clean_work_area). The cell must be the bundle's hand's (root_guard, the first write into a campaign
    cell). root = the tree the run trains in (its work areas)."""
    if git.get("dirty"):
        raise UncommittedTreeError(
            f"{run}: code tree {git.get('tree')} has uncommitted changes — launches run from a committed state only"
        )
    HR.assert_writable(cell_file(sdir, "launch"), b["hand"], root=root)
    left = cell_leftovers(sdir)
    if left:
        raise ValueError(
            f"{sdir}: the cell is not empty {left[:4]} — no launch on top of an old attempt's record or "
            "judge table (the relaunch tool clears them into attempts/ with a ledger)"
        )
    assert_clean_work_area(run, root)
    axis = next(a for a in man["axes"] if a["axis"] == b["axis"])
    rec = dict(
        run=run,
        hand=b["hand"],
        bundle=b["id"],
        run_key=b["run_key"],
        n=b["n"],
        git=git,
        launch_id=time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}_{os.urandom(3).hex()}",  # unique per launch
        manifest_fingerprint=man["fingerprint"],
        tree_sha256=axis["tree_sha256"],
        rule=man["rule"]["version"],
        **CF.sequence_record(b),
    )
    sdir.mkdir(parents=True, exist_ok=True)
    cell_file(sdir, "launch").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def gate(*args: str) -> subprocess.CompletedProcess:
    """The host gate CLI (scripts/rl/host_gate.sh); env RLJUDGE_LOCK_DIR · RLJUDGE_MIN_AVAIL_GB pass through. A call
    past GATE_TIMEOUT_S raises subprocess.TimeoutExpired."""
    return subprocess.run(["bash", str(GATE), *args], capture_output=True, text=True, timeout=GATE_TIMEOUT_S)


#: The form of a code tree fingerprint "<sha1> <code files>" (0 files = the command failed).
CODE_FP_RE = re.compile(r"[0-9a-f]{40} [1-9]\d*")


#: The code tree rule (code_tree_of, scripts/code_tree.sh — its one owner).
CODE_TREE_RULE = P.WT / "scripts" / "code_tree.sh"


def code_tree(tree: Path | None = None, rev: str = "HEAD") -> str:
    """The code tree fingerprint "<sha1> <code files>" of a commit of `tree` (default its HEAD): the rule code_tree_of
    of CODE_TREE_RULE, this code tree's (sha1 over `git ls-tree -r` without docs/ and *.md, so docs-only commits keep
    it). No answer or 0 code files raises."""
    r = subprocess.run(
        ["bash", "-c", 'source "$0" && code_tree_of "$1"', str(CODE_TREE_RULE), rev],
        cwd=str(tree or P.WT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = r.stdout.strip()
    if r.returncode != 0 or not CODE_FP_RE.fullmatch(out):
        raise RuntimeError(f"no code tree fingerprint (rc {r.returncode}): '{out}' {r.stderr[-200:]}")
    return out


def reserve_judges(gpus: list[str], pid: int, campaign: str) -> tuple[bool, str]:
    """The judge-only reservation of the campaign's judge GPUs (host gate kind rsv, bound to the loop's pid): training
    and data claims, holds and the data lanes' room skip them. Returns (ok, why); idempotent for the same pid."""
    r = gate("reserve", " ".join(gpus), str(pid), campaign)
    return r.returncode == 0, r.stderr.strip()


def unreserve_judges(gpus: list[str], pid: int) -> None:
    """Release the judge-only reservation that reserve_judges made for this pid."""
    gate("unreserve", " ".join(gpus), str(pid))


#: The argv a launched command runs under (spawn).
CHILD_ARGV = ("bash", "-c")


def spawn(sdir: Path, kind: str, cmd: str, tree: Path) -> int:
    """Fork `cmd` in its own session (stdin /dev/null · output → <kind>.out · cwd tree) → <kind>.pid. The sidecar is
    started with this alone (it claims its judges itself); `launch` claims the gate first. A fork that fails (OSError)
    leaves no file of its own: the output file this call created is removed again (an earlier start's judge.out
    stays), else the cell is not empty and every later launch is refused. The child is reaped here
    (proc.reap_when_done): a zombie would keep its gate claim alive."""
    outf = cell_file(sdir, f"{kind}_out")
    fresh = not outf.exists()
    try:
        with open(outf, "ab") as out:
            p = subprocess.Popen(
                [*CHILD_ARGV, cmd],
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                cwd=str(tree),
            )
    except OSError:
        if fresh:
            outf.unlink(missing_ok=True)
        raise
    cell_file(sdir, f"{kind}_pid").write_text(f"{p.pid}\n", encoding="utf-8")
    PR.reap_when_done(p)
    return p.pid


def launch(
    kind: str,
    gpus: list[str],
    run: str,
    cmd: str,
    sdir: Path,
    reserve_gb: int,
    ramp_s: int,
    cwd: Path | None = None,
    gate_kind: str | None = None,
    maxc: int = 0,
) -> tuple[int | None, str]:
    """Gate → claim (recorded under this process's pid inside the gate lock, so no second launch passes the RAM gate
    on the same headroom) → fork → claim re-bound to the child → <kind>.pid. The child's stdout/err go to
    <sdir>/<kind>.out and stdin is /dev/null: the caller never waits on it. GPU_SLOT / LAUNCH_ID_SLOT in `cmd` are
    filled (the gate's GPU · the cell's launch record id; a slot without a record id raises). The cell must lie inside
    the tree the child runs in (cwd, its work areas): a cell and a work area in two trees would make finalize move
    across trees. gate_kind = the gate's claim kind when not `kind` (a render claims as a training). Returns (pid, gpu)
    or (None, the gate's reason); a fork failure releases the claim and raises OSError. maxc = the gate's cap on live
    claims of that kind (0 = none)."""
    tree = Path(cwd or P.WT).resolve()
    if not Path(os.path.abspath(sdir)).resolve().is_relative_to(tree):
        raise ValueError(
            f"{run}: cell {sdir} is outside the launch tree {tree} — a cell and its work areas share one tree"
        )
    if LAUNCH_ID_SLOT in cmd:
        lid = launch_id_of(sdir)
        if not lid:
            raise ValueError(f"{run}: the command has a launch id slot but the launch record of cell {sdir} has no id")
        cmd = cmd.replace(LAUNCH_ID_SLOT, lid)
    gk = gate_kind or kind
    r = gate("claim", gk, " ".join(gpus), run, kind, str(reserve_gb), str(ramp_s), str(maxc), str(os.getpid()))
    if r.returncode != 0:
        return None, r.stderr.strip() or f"gate rc {r.returncode}"
    g = r.stdout.strip()
    sdir.mkdir(parents=True, exist_ok=True)
    try:
        pid = spawn(sdir, kind, cmd.replace(GPU_SLOT, g), tree)
    except OSError as e:
        gate("release", gk, g)
        raise OSError(f"{run}: fork failed (GPU {g} claim released) — {e}") from e
    gate("bind", gk, g, str(pid), run, kind, str(reserve_gb), str(ramp_s))
    return pid, g


def gate_claims() -> list[dict]:
    """Live claims and holds of the host gate (`host_gate.sh list`, the one reader of the claim file format)."""
    r = gate("list")
    if r.returncode != 0:
        raise RuntimeError(f"host gate list rc {r.returncode}: {r.stderr[-200:]}")
    keys = ("kind", "gpu", "pid", "t0", "run", "alias", "reserve_gb", "ramp_s")
    return [dict(zip(keys, ln.split(), strict=False)) for ln in r.stdout.splitlines() if ln.strip()]
