"""Apply the long-clip split (rule D36) to one hand and run the downstream steps on the pieces that need them.

Order: define the hand-independent split registry for this hand (``segment.split --define-step``: the reference hand
redefines when the rule's table changed, other hands only compare), cut (``--apply``), the aerial tail cut (rule D41,
``--tail-step``), then on every piece whose later outputs are missing or stale (``segment.todo``, the same
per-tree judgement as the stage-1 cells): all-object contact sidecar, cases, roles, quality (4b), main-object
sidecar, support gap (5b), wrist cache (6). Steps 4b, 5b and 6 run the argv of the stage table (``plan.emit_run``).

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

REPO = Path(__file__).resolve().parents[3]
EMBS = ("bimanual", "right", "left")
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
                "2d,3,4,4b,5,5b,6",
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
        from dexengine.data.stage1 import plan  # noqa: PLC0415

        try:
            d = plan.emit_run(step, ds, f0_force=False)
        except SystemExit as e:
            fail(f"{step} plan {self.h} {ds} {str(e)[:160]}")
        (self.s / f"{prefix}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
        if not d["runs"]:
            fail(f"{step} {self.h} {ds} the plan is empty -- {d.get('not_applicable') or 'no base layer'}")
        with open(self.s / f"{prefix}.log", "w") as fh:
            for r in d["runs"]:
                if subprocess.run(r["argv"], cwd=REPO, env=self.env, stdout=fh, stderr=subprocess.STDOUT).returncode:
                    fail(f"{step} {self.h} {ds} {_tail(self.s / f'{prefix}.log')}")
        return d

    def downstream(self, rows: list[list[str]]) -> None:
        """Re-run the stages downstream of the split on the changed pieces (rows = ds, emb, piece): contact
        sidecars, cases, roles, quality (4b), canonical sidecars, support gap (5b) and wrist caches (6)."""
        by_ds: dict[str, list[str]] = {}
        for r in rows:
            by_ds.setdefault(r[0], []).append(r[2])
        self._sidecars(by_ds)
        dsu = sorted(by_ds)
        self._cases(rows)
        self._roles(dsu)
        self._quality(dsu)
        self._canon(dsu, by_ds)
        self._support_gap(dsu)
        self._wrist(dsu)

    def _sidecars(self, by_ds: dict[str, list[str]]) -> None:
        """All-object contact sidecars of the changed pieces, per dataset."""
        h, s = self.h, self.s
        for ds in self.dss:
            if not by_ds.get(ds):
                continue
            f = s / f"sc_{ds}.txt"
            f.write_text("".join(c + "\n" for c in by_ds[ds]))
            log = s / f"sidecar_{ds}.log"
            self.run(
                [
                    self.pys,
                    "dexengine/data/contact/sidecar.py",
                    "--dataset",
                    ds,
                    "--hand",
                    h,
                    "--all-objects",
                    "--clips-file",
                    str(f),
                    "--workers",
                    WORKERS,
                ],
                log,
                f"sidecar {ds}",
            )
            print(f"[v2split] sidecar {h} {ds} {_tail(log, width=110)}", flush=True)

    def _cases(self, rows: list[list[str]]) -> None:
        """Case taxonomy (--write) of the changed pieces, per dataset and embodiment."""
        h, s = self.h, self.s
        for ds, emb in sorted({(r[0], r[1]) for r in rows}):
            cl = [r[2] for r in rows if r[0] == ds and r[1] == emb]
            log = s / f"case_{h}_{ds}_{emb}.log"
            self.run(
                [
                    self.pys,
                    "dexengine/data/cases/taxonomy.py",
                    "--dataset",
                    ds,
                    "--emb",
                    emb,
                    "--clips",
                    ",".join(cl),
                    "--write",
                    "--jobs",
                    WORKERS,
                ],
                log,
                f"case {h} {ds}/{emb}",
            )
            print(f"[v2split] cases {h} {ds}/{emb} {len(cl)} pieces UNCLASS={len(_grep(log, 'UNCLASS'))}", flush=True)

    def _roles(self, dsu: list[str]) -> None:
        """Object roles (--write) over the touched datasets; the changed-role TSV is counted."""
        h, s = self.h, self.s
        roles = s / f"roles_changed_{h}.tsv"
        self.run(
            [
                self.pys,
                "dexengine/data/object_roles.py",
                "--datasets",
                ",".join(dsu),
                "--write",
                "--out-changed",
                str(roles),
            ],
            s / f"roles_{h}.log",
            f"roles {h}",
        )
        n_changed = max(len(roles.read_text().splitlines()) - 1, 0) if roles.is_file() else 0
        print(f"[v2split] roles {h} changed {n_changed}", flush=True)

    def _quality(self, dsu: list[str]) -> None:
        """Stage 4b (quality) per touched dataset."""
        h, s = self.h, self.s
        for ds in dsu:
            self.stage("4b", ds, f"q4b_{h}_{ds}")
            hit = _grep(s / f"q4b_{h}_{ds}.log", r"^\[ref\] [a-z0-9]+: trees")
            print(f"[v2split] quality {h} {ds} {(hit[-1] if hit else '')[:150]}", flush=True)

    def _canon(self, dsu: list[str], by_ds: dict[str, list[str]]) -> None:
        """Canonical contact sidecars of the changed pieces, per dataset."""
        h, s = self.h, self.s
        for ds in dsu:
            f = s / f"canon_{h}_{ds}.txt"
            f.write_text("".join(c + "\n" for c in by_ds[ds]))
            log = s / f"canon_{h}_{ds}.log"
            self.run(
                [
                    self.pys,
                    "dexengine/data/contact/sidecar.py",
                    "--dataset",
                    ds,
                    "--hand",
                    h,
                    "--clips-file",
                    str(f),
                    "--workers",
                    WORKERS,
                ],
                log,
                f"canon {h} {ds}",
            )
            print(f"[v2split] canon {h} {ds} {_tail(log, width=110)}", flush=True)

    def _support_gap(self, dsu: list[str]) -> None:
        """Stage 5b (support gap) per touched dataset."""
        h, s = self.h, self.s
        for ds in dsu:
            self.stage("5b", ds, f"q5b_{h}_{ds}")
            hit = _grep(s / f"q5b_{h}_{ds}.log", r"^\[C640\] ")
            print(f"[v2split] support gap {h} {ds} {(hit[-1] if hit else '')[:150]}", flush=True)

    def _wrist(self, dsu: list[str]) -> None:
        """Stage 6 (wrist caches) per touched dataset; one [WRIST_CACHE] marker per run or fail."""
        h, s = self.h, self.s
        for ds in dsu:
            d = self.stage("6", ds, f"q6_{h}_{ds}")
            marks = _grep(s / f"q6_{h}_{ds}.log", r"^\[WRIST_CACHE\] ")
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
    os.environ["DEXENGINE_HAND"] = a.hand  # the in-process stage plans (4b, 5b, 6) read the active hand
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
