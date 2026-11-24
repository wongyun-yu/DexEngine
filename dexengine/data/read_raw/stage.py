"""Raw re-read of one dataset through the staging folder (driver ``stage`` of the reader registry).

``read``: every recording (the row's ``tasks`` lister) -> reader (the row's ``module``) -> decompose and
detect_contact -> staging folder ``paths.reread_stage_root(ds)``. Recordings already read are skipped (re-writing a read
recording changes its npz bytes and the ingest would refuse it). It refuses to start while the canonical mano layer
still holds clips of an older generation.

``intake <hand>``: retargets that hand in the stage (generate-xml and ik per embodiment, skip-existing), writes
``MANIFEST.sha256``, ingests mano + that hand (``dexengine.data.intake --ingest``) and runs the row's INTAKE
derivations in order.

Gates before the end marker: one reader marker per recording read, no reader/decompose/contact failure, retarget
output count = stage mano count per embodiment, canonical mano count after ingest = stage count, the derivations' log
texts and the base-layer INTAKE files they must leave (``read_raw.derive``). End marker:
``[REREAD] ds=<ds> <step> DONE``. Exit codes: 0 done, 2 refused or failed, 3 ``--dry`` that would be refused. Child
output is kept under ``<stage>/_logs``: ``read_<recording>.txt`` per reader run and
``<decomp|contact>_<emb>_<recording>.txt`` per decompose or contact row; each line of ``<decomp|contact>_fail.txt``
names its row's log.

Subset runs (scratch generation roots only): ``--root <generation root> [--recordings <list file>]``. The root must
carry a generation record of the same dataset (``layout.data_generation``); the canonical data root is refused, and
every listed recording must be one of the root's recordings.
"""

from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dexengine.core import capacity
from dexengine.core import paths as P
from dexengine.data import readers as R
from dexengine.data.read_raw import derive as D
from dexengine.retarget.reretarget_stage import write_manifest

REPO = R.REPO
STEP = "dexengine.data.read_raw.stage"  # capacity.NEEDS key prefix: :read (reader per recording) · :decomp · :contact
CONTACT_REF_DT = "0.02"


class Refused(SystemExit):
    """A gate refused the run (exit code 2)."""


def die(tag: str, msg: str) -> None:
    print(f"{tag} ✗ {msg}", file=sys.stderr, flush=True)
    raise Refused(2)


def _nice(argv: list[str]) -> list[str]:
    return ["nice", "-n", "10", *argv]


def _lines(text: str) -> list[str]:
    return [x for x in text.splitlines() if x.strip()]


def _nlines(f: Path) -> int:
    return len(_lines(f.read_text())) if f.is_file() else 0


class Run:
    """One invocation: dataset, data root, stage and interpreters."""

    def __init__(self, ds: str, dry: bool, root_arg: str, recordings: str):
        self.ds, self.dry, self.recf = ds, dry, recordings
        self.tag = f"[REREAD] ds={ds}"
        self.spec = R.stage_spec(ds)
        if root_arg:
            self._use_generation_root(root_arg)
        self.gen = P.data_layout.data_generation(str(P.main_root()))
        if recordings:
            if not root_arg:
                die(self.tag, "--recordings needs --root <generation root> (no subset on the canonical data root)")
            if self.gen is None:
                die(
                    self.tag,
                    f"--root {root_arg} has no generation record (canonical layout) -- subsets run only on "
                    "a generation root",
                )
            if self.gen.get("dataset") != ds:
                die(self.tag, f"the generation root's dataset is not {ds}: {json.dumps(self.gen, sort_keys=True)}")
            if not Path(recordings).is_file() or Path(recordings).stat().st_size == 0:
                die(self.tag, f"recording list {recordings} is missing or empty")
        self.root = P.main_root()
        self.raw = P.layer_path("raw", ds)
        self.pys = P.retarget_py()
        self.st = P.reread_stage_root(ds)

    def _use_generation_root(self, root_arg: str) -> None:
        saved = os.environ.pop("DEXENGINE_MAIN_ROOT", None)
        try:
            canon = P.main_root()
        finally:
            if saved is not None:
                os.environ["DEXENGINE_MAIN_ROOT"] = saved
        if not Path(root_arg).is_dir():
            die(self.tag, f"--root {root_arg} does not exist")
        root = Path(root_arg).resolve()
        if saved and Path(saved).resolve() != root:
            die(self.tag, f"--root {root} != DEXENGINE_MAIN_ROOT {saved} -- two data roots")
        if root == Path(canon).resolve():
            die(self.tag, f"--root is the canonical data root ({root}) -- only generation roots are accepted")
        os.environ["DEXENGINE_MAIN_ROOT"] = str(root)

    def header(self, step: str) -> None:
        def git(*a):
            return subprocess.run(["git", *a], cwd=REPO, stdout=subprocess.PIPE, text=True).stdout

        dirty = len(_lines(git("status", "--porcelain", "--", ".", ":!docs", ":!*.md")))
        gen = "canon" if self.gen is None else json.dumps(self.gen, ensure_ascii=False, sort_keys=True)
        print(
            f"{self.tag} {step} root={self.root} gen={gen} code={REPO} commit="
            f"{git('rev-parse', '--short', 'HEAD').strip()} "
            f"dirty={dirty} stage={self.st} dry={int(self.dry)}",
            flush=True,
        )

    def mano_count(self) -> int:
        """Clip folders in the canonical mano layer (all embodiments)."""
        return D.mano_clips(self.ds)

    def recordings(self) -> list[str]:
        """Every recording of the raw layer (the row's lister); its failure is fatal."""
        r = R.list_tasks(self.ds, str(self.raw), self.pys)
        if r.returncode:
            die(self.tag, "recording list (the reader row's tasks lister failed)")
        return _lines(r.stdout)

    def subset_of(self, gtl: list[str]) -> list[str]:
        """The recordings of the list file (deduplicated, sorted); a name outside `gtl` is fatal."""
        want = sorted({x.strip() for x in Path(self.recf).read_text().splitlines() if x.strip()})
        miss = [x for x in want if x not in set(gtl)]
        if miss:
            die(self.tag, f"listed recordings are not recordings of root {self.root}: {' '.join(miss)[:200]}")
        return want

    def subset_marker(self, sub: list[str], gtl: list[str]) -> None:
        print(f"[REREAD_SUBSET] ds={self.ds} n={len(sub)}/{len(gtl)} root={self.root} list={self.recf}", flush=True)


def _pool(jobs, fn, part: str) -> list:
    """fn over jobs (each starts one subprocess) on a pool sized by capacity for step STEP:part -> results in order."""
    with capacity.sized(f"{STEP}:{part}", len(jobs)) as cap, _cf.ThreadPoolExecutor(cap.workers) as ex:
        return list(ex.map(fn, jobs))


def read(run: Run) -> int:
    """Step ``read`` (see the module docstring)."""
    tag, ds, st = run.tag, run.ds, run.st
    n_mano = run.mano_count()
    if n_mano > 0:
        msg = f"the canonical mano layer holds {n_mano} clips of an older generation -- archive it first"
        if not run.dry:
            die(tag, msg)
        print(f"{tag} ✗(dry) {msg}", flush=True)
    if not run.raw.is_dir():
        die(tag, f"raw recordings folder is missing: {run.raw}")
    gtl = run.recordings()
    if run.recf:
        sub = run.subset_of(gtl)
        run.subset_marker(sub, gtl)
        gtl = sub
    recs = [x for x in gtl if x]
    if not recs:
        die(tag, f"0 recordings -- check the raw folder {run.raw}")
    lg = st / "_logs"
    ok_f, pend_f = lg / "read_ok.txt", lg / "pending.tsv"
    done = set(_lines(ok_f.read_text())) if ok_f.is_file() else set()
    new = [t for t in recs if t not in done]
    print(
        f"{tag} read recordings {len(recs)} · already read {len(recs) - len(new)} · to read {len(new)} · "
        f"pending clips {_nlines(pend_f)} · canonical mano {n_mano}",
        flush=True,
    )
    if run.dry:
        print(
            f"{tag} (dry) stage {st} · reader: {run.pys} -m {run.spec.module} --dataset-dir {st} --task <recording> "
            f"(pool: the {capacity.TAG} line) · marker {run.spec.marker}",
            flush=True,
        )
        print(
            f"{tag} (dry) then per new clip decompose · detect_contact --ref-dt {CONTACT_REF_DT} · nothing written",
            flush=True,
        )
        return 3 if n_mano > 0 else 0
    for d in (st / "raw/taco/body_models", st / f"processed/{ds}/assets", lg):
        d.mkdir(parents=True, exist_ok=True)
    ok_f.touch()
    pend_f.touch()
    _relink(st / "raw/taco/body_models/mano", P.mano_model_dir())
    for sr, rr in run.spec.links:
        if not (run.raw / rr).exists():
            die(tag, f"raw {run.raw / rr} is missing (reader row links)")
        (st / sr).parent.mkdir(parents=True, exist_ok=True)
        _relink(st / sr, run.raw / rr)
    link_object_assets(st, ds)
    if new:
        _read_new(run, new, lg, ok_f, pend_f)
    pend = sorted(set(_lines(pend_f.read_text())))
    pend_f.write_text("".join(x + "\n" for x in pend))
    if pend:
        _decompose_contact(run, pend, lg, pend_f)
    nok = len(set(_lines(ok_f.read_text())) & set(recs))
    if nok != len(recs):
        die(tag, f"recordings read {nok} != recordings {len(recs)}")
    n_st = sum(1 for _ in _stage_clips(st / f"processed/{ds}/mano"))
    print(f"{tag} read DONE recordings {nok}/{len(recs)} · stage clips {n_st} · pending 0", flush=True)
    return 0


def link_object_assets(st: Path, ds: str) -> Path:
    """Link processed/<ds>/assets/objects to the dataset's object-asset layer, creating the layer first (the readers
    mkdir under this link, so a dangling link on a fresh data root would fail every reader)."""
    (target := P.layer_path("assets_objects", ds)).mkdir(parents=True, exist_ok=True)
    _relink(link := st / f"processed/{ds}/assets/objects", target)
    return link


def _relink(link: Path, target) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def _stage_clips(mano: Path, pattern: str = "*"):
    """(embodiment, clip) of the stage's mano clip folders matching `pattern`."""
    if not mano.is_dir():
        return
    for e in sorted(x for x in mano.iterdir() if x.is_dir()):
        for c in sorted(e.glob(pattern)):
            if c.is_dir():
                yield e.name, c.name


def _read_new(run: Run, new: list[str], lg: Path, ok_f: Path, pend_f: Path) -> None:
    (lg / "read_new.txt").write_text("".join(t + "\n" for t in new))

    def one(t: str) -> str | None:
        with open(lg / f"read_{t}.txt", "w") as fh:
            r = subprocess.run(
                _nice([run.pys, "-m", run.spec.module, "--dataset-dir", str(run.st), "--task", t]),
                cwd=REPO,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        return t if r.returncode else None

    failed = [t for t in _pool(new, one, "read") if t]
    (lg / "read_fail.txt").write_text("".join(t + "\n" for t in failed))
    nmark = 0
    mano = run.st / f"processed/{run.ds}/mano"
    for t in new:
        if t in failed:
            continue
        log = (lg / f"read_{t}.txt").read_text(errors="replace")
        if not re.search(run.spec.marker.replace("{task}", t), log, re.M):
            continue
        nmark += 1
        with open(pend_f, "a") as fh:
            for e, c in _stage_clips(mano, run.spec.clip.replace("{task}", t)):
                fh.write(f"{e}\t{c}\n")
        with open(ok_f, "a") as fh:
            fh.write(t + "\n")
    print(f"{run.tag} read failed {len(failed)} · reader markers {nmark}/{len(new)}", flush=True)
    if failed or nmark != len(new):
        die(
            run.tag,
            f"read (failed {len(failed)} · markers {nmark}/{len(new)}) -- a new call reads only the failed recordings",
        )


def _row_step(run: Run, lg: Path, name: str, mod: str, extra: list[str], row: str) -> str | None:
    """One (embodiment, recording) row of a decompose or contact step. The child's stdout and stderr go to
    ``<lg>/<name>_<emb>_<recording>.txt``; a non-zero exit returns the FAIL line, which ends with that log path."""
    emb, task = row.split("\t")
    argv = [run.pys, "-m", mod, "--dataset-dir", str(run.st), "--dataset-name", run.ds]
    argv += ["--embodiment-type", emb, "--task", task, *extra]
    log = lg / f"{name}_{emb}_{task}.txt"
    with open(log, "w") as fh:
        r = subprocess.run(_nice(argv), cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    return f"FAIL {emb} {task} log={log}" if r.returncode else None


def _decompose_contact(run: Run, pend: list[str], lg: Path, pend_f: Path) -> None:
    steps = (
        ("decomp", "dexengine.retarget.decompose", ["--data-id", "0"]),
        ("contact", "dexengine.retarget.contact", ["--data-id", "0", "--ref-dt", CONTACT_REF_DT, "--no-show-viewer"]),
    )
    counts = {}
    for name, mod, extra in steps:

        def one(row: str, name=name, mod=mod, extra=extra) -> str | None:
            return _row_step(run, lg, name, mod, extra, row)

        fails = [x for x in _pool(pend, one, name) if x]
        (lg / f"{name}_fail.txt").write_text("".join(x + "\n" for x in fails))
        counts[name] = len(fails)
    print(
        f"{run.tag} read pending clips {len(pend)} · decompose failed {counts['decomp']} · contact failed "
        f"{counts['contact']}",
        flush=True,
    )
    if counts["decomp"] or counts["contact"]:
        die(
            run.tag,
            f"decompose {counts['decomp']} · contact {counts['contact']} failed -- the pending list "
            f"{pend_f} stays (a new call runs it again)",
        )
    pend_f.write_text("")


def intake(run: Run, hand: str) -> int:
    """Step ``intake <hand>`` (see the module docstring)."""
    tag, ds, st = run.tag, run.ds, run.st
    try:
        spec_dir = P.hand_spec_dir(hand)
    except (KeyError, ValueError) as e:
        die(tag, f"hand spec ({hand}): {e}")
    proc = st / f"processed/{ds}"
    if not (proc / "mano").is_dir():
        die(tag, f"the stage has no mano layer ({proc / 'mano'}) -- run read first")
    if run.recf:
        gtl = run.recordings()
        sub = run.subset_of(gtl)
        srec = sorted({c.split("__")[0] for _e, c in _stage_clips(proc / "mano")})
        if srec != sub:
            die(tag, f"stage recordings {len(srec)} != list {len(sub)} -- run read with the same list first")
        run.subset_marker(sub, gtl)
    n_mano = run.mano_count()
    hid = f"{ds}_self_{time.strftime('%Y%m%d')}"
    n_stage = sum(1 for _ in _stage_clips(proc / "mano"))
    print(
        f"{tag} intake hand {hand} (spec {spec_dir}) · stage mano {n_stage} · canonical mano {n_mano} · "
        f"handoff id {hid}",
        flush=True,
    )
    if run.dry:
        return _intake_dry(run, spec_dir, hid, n_mano)
    if n_mano:
        die(tag, f"the canonical mano layer holds {n_mano} clips -- archive the generation first")
    lg = st / "_logs"
    for d in (proc / "assets/robots", lg):
        d.mkdir(parents=True, exist_ok=True)
    if not (proc / "assets/robots" / spec_dir).exists():
        shutil.copytree(P.hand_rig_dir(spec_dir), proc / "assets/robots" / spec_dir)
    for e in sorted(x.name for x in (proc / "mano").iterdir()):
        _retarget_emb(run, spec_dir, e, lg)
        _check_retarget_count(tag, proc, spec_dir, e)
    write_manifest(st, f"./{ds}")
    _ingest_stage(tag, st, spec_dir, ds, hid, lg)
    nm = run.mano_count()
    print(f"{tag} intake canonical mano {nm} (must equal stage {n_stage})", flush=True)
    if nm != n_stage:
        die(tag, f"ingest count {nm} != {n_stage}")
    nd = D.run(run.spec.derive, ds, st / "intake_out", lg, run.pys, lambda m: die(tag, m))
    print(f"{tag} intake DONE hand {hand} · canonical mano {nm} · handoff {hid} · derivations {nd}", flush=True)
    return 0


def _intake_dry(run: Run, spec_dir: str, hid: str, n_mano: int) -> int:
    """The dry intake: print the retarget and ingest commands it would run; 3 when the canonical mano layer is
    occupied (the ingest would collide)."""
    tag, ds, st = run.tag, run.ds, run.st
    print(
        f"{tag} (dry) python -m dexengine.retarget.batch --phase generate-xml · ik (per embodiment · "
        "skip-existing) -> MANIFEST.sha256",
        flush=True,
    )
    names = " ".join(d[0] for d in run.spec.derive)
    print(
        f"{tag} (dry) python -m dexengine.data.intake {st / 'processed'} --hands {spec_dir} --datasets {ds} "
        f"--ingest --handoff-id {hid} -> derivations {names} · nothing written",
        flush=True,
    )
    if n_mano > 0:
        print(
            f"{tag} ✗(dry) the canonical mano layer holds {n_mano} clips -- the ingest would collide "
            "(archive the generation first)",
            flush=True,
        )
        return 3
    return 0


def _check_retarget_count(tag: str, proc: Path, spec_dir: str, e: str) -> None:
    """Every staged mano clip of embodiment `e` must have a retarget output; dies otherwise."""
    nin = sum(1 for _ in (proc / "mano" / e).iterdir())
    nout = (
        sum(1 for _ in (proc / spec_dir / e).glob("*/0/trajectory_kinematic.npz"))
        if (proc / spec_dir / e).is_dir()
        else 0
    )
    print(f"{tag} intake {e} retarget input {nin} -> output {nout}", flush=True)
    if nin != nout:
        die(tag, f"retarget count {e} {nin} != {nout}")


def _ingest_stage(tag: str, st: Path, spec_dir: str, ds: str, hid: str, lg: Path) -> None:
    """Run the intake CLI on the staged handoff (--ingest); its log goes to <stage>/_logs/ingest.log."""
    with open(lg / "ingest.log", "w") as fh:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "dexengine.data.intake",
                str(st / "processed"),
                "--hands",
                spec_dir,
                "--datasets",
                ds,
                "--ingest",
                "--handoff-id",
                hid,
            ],
            cwd=REPO,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    if r.returncode:
        last = (_lines((lg / "ingest.log").read_text(errors="replace")) or [""])[-1][:120]
        die(tag, f"ingest rc={r.returncode} ({last})")


def _retarget_emb(run: Run, spec_dir: str, emb: str, lg: Path) -> None:
    for ph in ("generate-xml", "ik"):
        log = lg / f"intake_{emb}_{ph}.log"
        with open(log, "w") as fh:
            r = subprocess.run(
                _nice(
                    [
                        run.pys,
                        "-m",
                        "dexengine.retarget.batch",
                        "--dataset-dir",
                        str(run.st),
                        "--dataset-name",
                        run.ds,
                        "--robot-type",
                        spec_dir,
                        "--embodiment-type",
                        emb,
                        "--phase",
                        ph,
                        "--skip-existing",
                    ]
                ),
                cwd=REPO,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        if r.returncode:
            die(run.tag, f"retarget {emb} {ph}")
        hits = re.findall(rf"\[SKIP_EXISTING\] phase={re.escape(ph)} .*", log.read_text(errors="replace"))
        if not hits:
            die(run.tag, f"retarget {emb} {ph}: no [SKIP_EXISTING] marker -- the reuse decision cannot be trusted")
        print(f"{run.tag} intake {emb} {hits[-1]}", flush=True)
