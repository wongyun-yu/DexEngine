"""INTAKE derivations of one dataset: the reader row's ``derive`` steps (``dexengine.data.readers.READERS``).

They are hand-independent audits of the canonical MANO layer, and the base layer refuses to run without two of them
(``INTAKE/support_audit.json``, ``INTAKE/segment_audit.json``). The ``stage`` driver runs them right after its
ingest. For the other drivers, run them once the canonical MANO layer and the object assets are in place (``mano``:
after the reader and ``dexengine.data.objects.build``; ``hoi``: after the intake of its stage; ``taco``: after the
stage-1 retarget):

    python -m dexengine.data.read_raw <dataset> derive [--dry]

The steps run in row order with the retargeting interpreter from the repository root. A step that exits non-zero or
whose log lacks the row's required text stops the run, and so does a run that leaves either base-layer file
unwritten, empty, or with every clip an error. Reports go to ``<reread stage>/intake_out`` and logs to
``<reread stage>/_logs`` (``paths.reread_stage_root``), the places the ``stage`` driver uses.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from dexengine.core import paths as P

REPO = Path(__file__).resolve().parents[3]
#: INTAKE files the base layer refuses to run without (dexengine/data/base_adapter.py); every derive run proves them
BASE_INPUTS = ("support_audit.json", "segment_audit.json")


def command(ds: str) -> str:
    """The command that runs the derivations of `ds` (named in the base layer's refusal)."""
    return f"python -m dexengine.data.read_raw {ds} derive"


def mano_clips(ds: str) -> int:
    """Clip folders in the canonical MANO layer of `ds` (all embodiments)."""
    n = 0
    for e in P.data_layout.EMBODIMENTS:
        d = P.layer_path("mano", ds, emb=e)
        if d.is_dir():
            n += sum(1 for x in d.iterdir() if x.is_dir())
    return n


def proof(ds: str, since: float) -> str:
    """'' when both base-layer INTAKE files of `ds` were written at or after `since`, hold clips, and the support
    audit has at least one clip that is not an error; otherwise what is wrong."""
    idir = P.layer_path("intake", ds)
    for name in BASE_INPUTS:
        f = idir / name
        if not f.is_file():
            return f"{f} was not written"
        if f.stat().st_mtime < since:
            return f"{f} was not rewritten by this run (older than its start)"
        try:
            doc = json.loads(f.read_text())
        except ValueError as e:
            return f"{f} is not JSON ({e})"
        clips = doc.get("clips") or {}
        if not clips:
            return f"{f} has no clips"
        if name == "support_audit.json" and all(c.get("cls") == "error" for c in clips.values()):
            return f"{f}: every clip ({len(clips)}) is an error -- an input of the audit is missing (object assets?)"
    return ""


def run(rows, ds: str, out: Path, logs: Path, pys: str, fail: Callable[[str], None]) -> int:
    """Run the derive rows of `ds` in order and prove the base-layer files; `fail(message)` stops the caller (it never
    returns). Returns the number of derivations run."""
    if not rows:
        fail(f"0 derivations -- the reader row of {ds} has no derive steps (dexengine.data.readers.READERS)")
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    start = time.time() - 1.0  # file times may be coarser than the clock
    for name, argv, must in rows:
        args = argv.replace("{out}", str(out)).replace("{ds}", ds).split()
        log = logs / f"derive_{name}.log"
        with open(log, "w") as fh:
            r = subprocess.run([pys, *args], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        text = log.read_text(errors="replace")
        if r.returncode:
            last = [x for x in text.splitlines() if x.strip()] or [""]
            fail(f"derivation {name} rc={r.returncode} ({last[-1][:120]})")
        if must and must not in text:
            fail(f"derivation {name}: its log does not contain '{must}'")
    if bad := proof(ds, start):
        fail(f"derivations ran but the base layer input is not there: {bad}")
    return len(rows)


def main(ds: str, dry: bool, tag: str, fail: Callable[[str], None]) -> int:
    """The ``derive`` step for a dataset of any driver: the MANO layer must hold clips, then the row's steps run."""
    from dexengine.data import readers as R  # noqa: PLC0415

    rows = R.spec(ds).derive
    n = mano_clips(ds)
    st = P.reread_stage_root(ds)
    print(f"{tag} derive · canonical mano clips {n} · steps {', '.join(r[0] for r in rows) or 'none'}", flush=True)
    if n == 0:
        fail(f"the canonical MANO layer has no clips ({P.layer_path('mano', ds, emb='bimanual')}) -- read {ds} first")
    if dry:
        print(f"{tag} (dry) would write {', '.join(BASE_INPUTS)} under {P.layer_path('intake', ds)}", flush=True)
        return 0
    nd = run(rows, ds, st / "intake_out", st / "_logs", P.retarget_py(), fail)
    print(f"{tag} derive DONE · derivations {nd} · {', '.join(BASE_INPUTS)} written", flush=True)
    return 0
