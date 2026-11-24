"""Apply the long-clip split (rule D36) to one hand and run the downstream steps on the pieces that need them.

Order: define the hand-independent split registry for this hand (``segment.split --define-step``: the reference hand
redefines when the rule's table changed, other hands only compare), cut (``--apply``), the aerial tail cut (rule D41,
``--tail-step``), then on every piece whose later outputs are missing or stale (``segment.todo``, the same
per-tree judgement as the stage-1 cells) the steps the stage table reruns on pieces (``table.split_steps()["run"]``,
in row order; the steps the cut carries and f0 are not run here). A step whose script takes a piece list runs scoped to
the pieces (``Apply._piece_runner``); every other step runs the argv of the stage table (``plan.emit_run``) per dataset.

    ARM=<arm> CAMPAIGN_DIR=<log folder> python -m dexengine.data.stage1.split_apply <hand> [--datasets a,b]

Logs and piece lists go to the campaign folder (made when missing). Any failure stops with ``[v2split] FAIL ...``
and exit code 2.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.data.stage1 import plan
from dexengine.data.stage1 import table as T

REPO = Path(__file__).resolve().parents[3]
EMBS = ("bimanual", "right", "left")
#: The summary line of a table-argv step per dataset, by the row's argv script: (label, marker regex). Other scripts
#: print the log tail; wrist_poses.py is checked by its [WRIST_CACHE] count (Apply._wrist).
SUMMARY = {
    "filtering.py": ("quality", r"^\[ref\] [a-z0-9]+: trees"),
    "gap.py": ("support gap", r"^\[C640\] "),
}
WORKERS = "8"


class Failed(SystemExit):
    """A step failed (exit code 2)."""


def fail(msg: str) -> None:
    print(f"[v2split] FAIL {msg}", flush=True)
    raise Failed(2)


def _tail(f: Path, n: int = 1, width: int = 160) -> str:
    lines = [x for x in f.read_text(errors="replace").splitlines() if x.strip()] if f.is_file() else []
    return " ".join(lines[-n:])[:width]


def _grep(f: Path, pat: str) -> list[str]:
    return [x for x in f.read_text(errors="replace").splitlines() if re.search(pat, x)] if f.is_file() else []


def _table_argv(step: str) -> tuple:
    """The argv the stage table's row for `step` names (its script first)."""
    return next(g[3].argv for g in T.STAGES if T.step_id(g) == step)


class Apply:
    """One run for one hand over a dataset scope."""

    def __init__(self, hand: str, scope: str, logdir: Path):
        self.h, self.s = hand, logdir
        logdir.mkdir(parents=True, exist_ok=True)  # the campaign folder: logs and piece lists (CAMPAIGN_DIR)
        self.spec_dir = P.hand_spec_dir(P.hand_id(hand))
        self.pys = P.retarget_py()
        self.dss = list(RR.datasets_scope(scope))
        if not self.dss:
            fail(f"unknown datasets in the scope ({scope})")
        if scope:
            print(f"[v2split] scope {','.join(self.dss)}", flush=True)
        self.env = dict(os.environ, DEXENGINE_HAND=hand)
        part = T.split_steps()  # dies on a table step the split would leave out (before any cut)
        self.steps = part["run"]
        print(
            f"[v2split] steps {' '.join(self.steps)} (stage table · carried by the cut {' '.join(part['carried'])} · "
            f"f0 by the feeder {' '.join(part['f0'])})",
            flush=True,
        )

    def run(self, argv: list[str], log: Path, what: str, err_pat: str = "") -> None:
        with open(log, "w") as fh:
            r = subprocess.run(argv, cwd=REPO, env=self.env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode:
            detail = " ".join(_grep(log, err_pat)[:2])[:160] if err_pat else _tail(log)
            fail(f"{what} {detail}")

    def snap(self, out: Path) -> list[str]:
        """Rows ``<ds>\\t<emb>\\t<piece>`` of every piece (``_t<N>`` tree) in this hand's base layer, sorted."""
        rows = []
        for ds in self.dss:
            for emb in EMBS:
                d = P.layer_path("base", ds, hand=self.spec_dir, emb=emb)
                if d.is_dir():
                    rows += [f"{ds}\t{emb}\t{c.name}" for c in d.iterdir() if re.search(r"_t[0-9]+$", c.name)]
        rows.sort()
        out.write_text("".join(r + "\n" for r in rows))
        return rows

    def split_steps(self) -> None:
        day = time.strftime("%Y%m%d")
        split = [sys.executable, "-m", "dexengine.data.segment.split"]
        for emb in EMBS:
            log = self.s / f"define_{self.h}_{emb}.log"
            self.run(
                split + ["--define-step", "--label", f"d36_{day}", "--datasets", ",".join(self.dss), "--emb", emb],
                log,
                f"D36 {self.h}/{emb}",
                "✗",
            )
            marks = _grep(log, r"^\[D36_STEP\]")
            if len(marks) != len(self.dss):
                fail(
                    f"D36 marker count != dataset count {self.h}/{emb} -- no proof the registry comparison ran · {log}"
                )
            for m in marks:
                if not re.search(r"→ (verified )?same", m):
                    print(m[:200], flush=True)
        for emb in EMBS:
            log = self.s / f"apply_{self.h}_{emb}.log"
            self.run(
                split + ["--apply", "--datasets", ",".join(self.dss), "--emb", emb], log, f"apply {self.h}/{emb}", "✗"
            )
            for m in _grep(log, r"^\[split\].* applied "):
                if "parents 0 " not in m:
                    print(m[:160], flush=True)
        for emb in EMBS:
            log = self.s / f"tail_{self.h}_{emb}.log"
            self.run(
                split + ["--tail-step", "--label", f"d41_{day}", "--datasets", ",".join(self.dss), "--emb", emb],
                log,
                f"D41 {self.h}/{emb}",
                "✗",
            )
            for m in _grep(log, r"^\[D41_STEP\]"):
                if "skipped" not in m:
                    print(m[:160], flush=True)

    def todo(self, after: list[str], new: list[str]) -> list[list[str]]:
        """Pieces whose later outputs are missing or stale (``segment.todo``) as [ds, emb, piece, ...] rows."""
        out, log = self.s / f"units_todo_{self.h}.tsv", self.s / f"todo_{self.h}.log"
        self.run(
            [
                self.pys,
                "-m",
                "dexengine.data.segment.todo",
                "--hand",
                self.h,
                "--units",
                str(self.s / f"units_after_{self.h}.tsv"),
                "--stages",
                ",".join(self.steps),
                "--out",
                str(out),
            ],
            log,
            f"todo judgement {self.h}",
        )
        for m in _grep(log, r"^\[PIECE_TODO\] "):
            print(m[:200], flush=True)
        if not any(x.startswith(f"[PIECE_TODO] hand={self.h} total pieces {len(after)} ") for x in _grep(log, "")):
            fail(f"todo marker [PIECE_TODO] total != piece count -- no proof the judgement ran · {log}")
        rows = [x.split("\t") for x in out.read_text().splitlines() if x.strip()] if out.is_file() else []
        have = {tuple(r[:3]) for r in rows}
        miss = [r for r in new if tuple(r.split("\t")) not in have][:3]
        if miss:  # a piece just cut has no case yet, so it must be to-do
            fail(
                f"pieces just cut are not to-do ({' '.join(m.replace(chr(9), '/') for m in miss)}) -- the todo "
                "judgement misreads the outputs"
            )
        return rows

    def stage(self, step: str, ds: str, prefix: str) -> dict:
        """Run step `step` of the stage table on `ds` (argv from plan.emit_run); its plan must not be empty."""
        try:
            d = plan.emit_run(step, ds, f0_force=False)
        except SystemExit as e:
            fail(f"{step} plan {self.h} {ds} {str(e)[:160]}")
        (self.s / f"{prefix}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
        if d.get("not_applicable"):
            print(f"[v2split] {step} {self.h} {ds} not applicable -- {d['not_applicable']}", flush=True)
            return d
        if not d["runs"]:
            fail(f"{step} {self.h} {ds} the plan is empty -- {d.get('not_applicable') or 'no base layer'}")
        with open(self.s / f"{prefix}.log", "w") as fh:
            for r in d["runs"]:
                if subprocess.run(r["argv"], cwd=REPO, env=self.env, stdout=fh, stderr=subprocess.STDOUT).returncode:
                    fail(f"{step} {self.h} {ds} {_tail(self.s / f'{prefix}.log')}")
        return d

    def downstream(self, rows: list[list[str]]) -> None:
        """Re-run the stage table's piece steps (``self.steps``, row order) on the changed pieces (rows = ds, emb,
        piece): a step whose script takes a piece list runs scoped (``_piece_runner``), any other runs the table argv
        on each touched dataset."""
        by_ds: dict[str, list[str]] = {}
        for r in rows:
            by_ds.setdefault(r[0], []).append(r[2])
        dsu = sorted(by_ds)
        for step in self.steps:
            script = Path(_table_argv(step)[0]).name
            runner = self._piece_runner(script)
            if runner:
                runner(step, rows, by_ds)
            else:
                self._table_step(step, script, dsu)
        print(f"[v2split] {self.h} downstream steps {' '.join(self.steps)} done", flush=True)

    def _piece_runner(self, script: str):
        """The scoped runner of a table step whose script (the row's argv script) takes the changed pieces; None = the
        step runs the table argv on each touched dataset (``stage``). A scoped run is the table's own argv
        (``plan.step_argv``: interpreter, slots, run guard) with the piece list and the worker count appended."""
        return {
            "sidecar.py": self._sidecar,
            "taxonomy.py": self._cases,
            "object_roles.py": self._roles,
            "hand_clear_stage.py": self._hand_clear,
        }.get(script)

    def _sidecar(self, step: str, _rows: list[list[str]], by_ds: dict[str, list[str]]) -> None:
        """A contact sidecar step on the changed pieces, per dataset: all objects (the row's argv carries
        --all-objects; logs sidecar_<ds>) or the main object (logs canon_<hand>_<ds>)."""
        h, s = self.h, self.s
        every = "--all-objects" in _table_argv(step)
        for ds in [d for d in self.dss if by_ds.get(d)] if every else sorted(by_ds):
            name = f"sc_{ds}" if every else f"canon_{h}_{ds}"
            f = s / f"{name}.txt"
            f.write_text("".join(c + "\n" for c in by_ds[ds]))
            log = s / (f"sidecar_{ds}.log" if every else f"canon_{h}_{ds}.log")
            what = f"sidecar {ds}" if every else f"canon {h} {ds}"
            self.run(plan.step_argv(step, ds) + ["--clips-file", str(f), "--workers", WORKERS], log, what)
            label = f"sidecar {h} {ds}" if every else f"canon {h} {ds}"
            print(f"[v2split] {label} {_tail(log, width=110)}", flush=True)

    def _hand_clear(self, step: str, _rows: list[list[str]], by_ds: dict[str, list[str]]) -> None:
        """Hand clear (2f) on the changed pieces of each touched dataset: the step measures the listed trees only and
        merges their rows into its ledger (the other trees' lines and exclusion lines stay). A dataset with no
        changed piece is not run."""
        h, s = self.h, self.s
        for ds in sorted(by_ds):
            f = s / f"hc_{h}_{ds}.txt"
            f.write_text("".join(c + "\n" for c in by_ds[ds]))
            log = s / f"q{step}_{h}_{ds}.log"
            self.run(plan.step_argv(step, ds) + ["--clips-file", str(f)], log, f"hand clear {h} {ds}")
            hit = _grep(log, r"^\[HAND_CLEAR\] ")
            print(f"[v2split] hand clear {h} {ds} {(hit[-1] if hit else '')[:150]}", flush=True)

    def _table_step(self, step: str, script: str, dsu: list[str]) -> None:
        """A table step on each touched dataset (the table argv, ``stage``) and its summary line."""
        h, s = self.h, self.s
        for ds in dsu:
            prefix = f"q{step}_{h}_{ds}"
            d = self.stage(step, ds, prefix)
            if script == "wrist_poses.py":
                self._wrist(ds, s / f"{prefix}.log", d)
                continue
            label, pat = SUMMARY.get(script, (step, ""))
            line = (_grep(s / f"{prefix}.log", pat) or [""])[-1] if pat else _tail(s / f"{prefix}.log")
            print(f"[v2split] {label} {h} {ds} {line[:150]}", flush=True)

    def _cases(self, step: str, rows: list[list[str]], _by_ds: dict[str, list[str]]) -> None:
        """Case taxonomy (--write) of the changed pieces, per dataset and embodiment."""
        h, s = self.h, self.s
        for ds, emb in sorted({(r[0], r[1]) for r in rows}):
            cl = [r[2] for r in rows if r[0] == ds and r[1] == emb]
            log = s / f"case_{h}_{ds}_{emb}.log"
            argv = plan.step_argv(step, ds) + ["--emb", emb, "--clips", ",".join(cl), "--jobs", WORKERS]
            self.run(argv, log, f"case {h} {ds}/{emb}")
            print(f"[v2split] cases {h} {ds}/{emb} {len(cl)} pieces UNCLASS={len(_grep(log, 'UNCLASS'))}", flush=True)

    def _roles(self, step: str, _rows: list[list[str]], by_ds: dict[str, list[str]]) -> None:
        """Object roles (--write) over the touched datasets in one run (``{ds}`` = their comma list); the changed-role
        TSV is counted."""
        h, s = self.h, self.s
        roles = s / f"roles_changed_{h}.tsv"
        self.run(
            plan.step_argv(step, ",".join(sorted(by_ds))) + ["--out-changed", str(roles)],
            s / f"roles_{h}.log",
            f"roles {h}",
        )
        n_changed = max(len(roles.read_text().splitlines()) - 1, 0) if roles.is_file() else 0
        print(f"[v2split] roles {h} changed {n_changed}", flush=True)

    def _wrist(self, ds: str, log: Path, d: dict) -> None:
        """Wrist caches (6): one [WRIST_CACHE] marker per run or fail."""
        h = self.h
        marks = _grep(log, r"^\[WRIST_CACHE\] ")
        if len(marks) != len(d["runs"]):
            fail(f"wrist {h} {ds} markers [WRIST_CACHE] {len(marks)}/{len(d['runs'])} runs")
        for m in marks:
            print(re.sub(r" out=.*", "", m.replace("[WRIST_CACHE] ", f"[v2split] wrist {h} {ds} ")), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hand", help="hand id")
    ap.add_argument(
        "--datasets",
        default=os.environ.get("SPLIT_DSS", ""),
        help="comma list (default: the campaign scope, rules.datasets_scope)",
    )
    a = ap.parse_args(argv)
    if not os.environ.get("ARM"):
        raise SystemExit("ARM=<arm> is required (no default)")
    os.environ["DEXENGINE_ARM"] = os.environ["ARM"]
    os.environ["DEXENGINE_HAND"] = a.hand  # the in-process stage plans (table-argv steps) read the active hand
    if not os.environ.get("CAMPAIGN_DIR"):
        raise SystemExit("CAMPAIGN_DIR=<log folder> is required (no default)")
    run = Apply(a.hand, a.datasets, Path(os.environ["CAMPAIGN_DIR"]))
    h, s = run.h, run.s
    print(f"[v2split] {h} start {time.strftime('%H:%M:%S')}", flush=True)
    before = run.snap(s / f"units_before_{h}.tsv")
    run.split_steps()
    after = run.snap(s / f"units_after_{h}.tsv")
    new = sorted(set(after) - set(before))
    (s / f"units_new_{h}.tsv").write_text("".join(r + "\n" for r in new))
    print(f"[v2split] {h} new pieces {len(new)} (all pieces {len(after)})", flush=True)
    rows = run.todo(after, new)
    print(f"[v2split] {h} pieces to do {len(rows)} (by state)", flush=True)
    if not rows:
        print(f"[v2split] {h} no pieces to do -- ALL_DONE {time.strftime('%H:%M:%S')}", flush=True)
        return 0
    run.downstream(rows)
    print(f"[v2split] {h} ALL_DONE {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.stage1.split_apply")
    sys.exit(main())
