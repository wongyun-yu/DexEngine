#!/usr/bin/env python3
"""OakInk2 supplement stage: exports source object supplements **before** stage 1 step 2 (base layer); owns this order
(called by the stage-1 driver).

  required objects  dexengine/data/oakink2/required.py --export                 also retires old ownerless tags
  (retire_unowned) supports          dexengine/data/oakink2/raw_objects.py --supports --export
  sibling parts     dexengine/data/oakink2/raw_objects.py --parts --export       last, covering siblings of the two
  before

All three read the **root trees of the axis-hand base** (the root timeline, objects already in the tree), so the base
state decides the path:
  the axis-hand base has trees   → run the three in order. The following step 2 (base_adapter) rebuilds only trees whose
      supplement fingerprint changed.
  the axis-hand base is empty (new root) → when this hand is the axis hand, build base once **without supplements**
      (command = hand_pipeline --emit-run 2), then run the three.
      · files already in the supplement layer (INTAKE/raw_objects) fail: their owners cannot be measured on an empty
        base
        (a root that took in derived outputs; empty the supplement layer and rerun).
      · when this hand is not the axis hand it fails: supplements come from the axis-hand base (the axis hand's stage 1
        runs first).
Scope = the datasets of rule D1-e (dexengine.core.datasets; names not restated here). Out-of-scope datasets print the
marker only, rc 0.
Marker [OAK_SUPP]: path (exports | bootstrap) · rc per step · supplement file count before → after. Any rc≠0 exits with
that rc.
Usage: python3 dexengine/data/oakink2/supp_stage.py --dataset <ds> --hand <hand> [--plan] [--stamp <stamp file>]
       (--plan = print the commands only · no writes)
Stamp (--stamp, explicit; the stage-1 driver passes it only with DEXENGINE_STAGE1_SUPP_STAMP=1): when the supplements
finish with
  rc 0 the stamp = rule (RULE · its datasets) + axis hand + axis-hand base file fingerprint (path · size · mtime) +
  supplement layer file fingerprint + hashes of the supplement code files (STAMP_CODE). On the next run, a freshly
  measured stamp equal to the file skips the three (marker [SUPP_SKIP] ds=…). When it cannot be measured nothing is
  skipped ([SUPP_STAMP] ✗); the old stamp is removed before running. Inputs outside the fingerprint (modules outside
  STAMP_CODE, OakInk2 sources) are not seen, hence off by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import InputMissingError
from dexengine.data import rules as RR  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TAG = "[OAK_SUPP]"
#: word of the plan line for a dataset outside the rule's scope; the stage-1 driver drops that
#: line from the --plan output it shows
NOT_APPLICABLE = "not applicable"
RULE = "D1-e"  # supplement rule (part-group completeness); its datasets are this stage's scope
EXPORTS = (
    ("required", ("dexengine/data/oakink2/required.py", "--export")),
    ("support", ("dexengine/data/oakink2/raw_objects.py", "--supports", "--export")),
    ("siblings", ("dexengine/data/oakink2/raw_objects.py", "--parts", "--export")),
)

# Code fingerprint of the stamp = this file + the three scripts (read from EXPORTS) + raw_base.py, the shared layer the
# two raw_objects exports and required.py run on (split out of raw_objects.py).
STAMP_CODE = (
    "dexengine/data/oakink2/supp_stage.py",
    "dexengine/data/oakink2/raw_base.py",
    "dexengine/data/oakink2/required_measure.py",  # the measurement half of required.py (an EXPORTS entry)
    *sorted({argv[0] for _n, argv in EXPORTS}),
)


def _tree_fp(d: Path) -> dict:
    """(relative path · size · mtime_ns) fingerprint of every file below the folder; contents are not read (a touch
    reruns = the safe side)."""
    h, n = hashlib.sha256(), 0
    if d.is_dir():
        for x in sorted(d.rglob("*")):
            if x.is_file():
                st = x.stat()
                h.update(f"{x.relative_to(d)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
                n += 1
    return {"path": str(d), "n": n, "sha256": h.hexdigest()}


def stamp(ds: str) -> dict:
    """Stamp of the current state (module docstring). Raises when it cannot be measured; the caller then does not
    skip."""
    axis = RR.axis_hand()
    base = P.layer_path("base", ds, hand=P.hand_spec_dir(axis), emb="bimanual")
    return {
        "rule": RULE,
        "rule_datasets": list(RR.rule_datasets(RULE)),
        "ds": ds,
        "axis_hand": axis,
        "base": _tree_fp(base),
        "supp": _tree_fp(P.layer_path("intake", ds) / "raw_objects"),
        "code": {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in STAMP_CODE},
    }


def _stamp_same(path: Path, ds: str) -> bool:
    """True when the stamp file equals the current stamp. False when it cannot be measured, differs or is missing
    (reason marker)."""
    try:
        cur = stamp(ds)
    except Exception as e:  # noqa: BLE001 — any failure falls on the side of not skipping
        print(f"[SUPP_STAMP] ✗ ds={ds} stamp computation failed — not skipping: {type(e).__name__}: {e}", flush=True)
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError) as e:
        print(f"[SUPP_STAMP] ✗ ds={ds} cannot read the stamp file ({path}: {e}) — not skipping", flush=True)
        return False
    if old == cur:
        print(
            f"[SUPP_SKIP] ds={ds} reason=stamp unchanged — base files {cur['base']['n']} · supplement files "
            f"{cur['supp']['n']} · "
            f"code {len(cur['code'])} · rule {RULE} ({path})",
            flush=True,
        )
        return True
    why = "missing" if old is None else "differs (" + ",".join(k for k in cur if old.get(k) != cur[k]) + ")"
    print(f"[SUPP_STAMP] ds={ds} stamp {why} — running the supplements", flush=True)
    return False


def _stamp_write(path: Path, ds: str) -> None:
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(stamp(ds), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        print(f"[SUPP_STAMP] ds={ds} stamp written {path}", flush=True)
    except Exception as e:  # noqa: BLE001 — if writing fails the next run reruns (safe side); reported loudly
        print(
            f"[SUPP_STAMP] ✗ ds={ds} could not write the stamp ({type(e).__name__}: {e}) — the next run reruns",
            flush=True,
        )


def _roots(ds: str, axis: str) -> int:
    """Tree count of the axis-hand base (bimanual): the root population the three supplement steps read."""
    b = P.layer_path("base", ds, hand=P.hand_spec_dir(axis), emb="bimanual")
    return sum(1 for x in b.iterdir() if (x / "task_info.json").is_file()) if b.is_dir() else 0


def _supp_files(ds: str) -> list:
    d = P.layer_path("intake", ds) / "raw_objects"
    return sorted(x.name for x in d.glob("*.npz")) if d.is_dir() else []


def _base_argv(ds: str, hand: str) -> list:
    """Command of step 2 (base layer), owned by hand_pipeline STAGES (--emit-run 2)."""
    out = subprocess.run(
        [sys.executable, str(REPO / "dexengine/data/stages.py"), "--emit-run", "2", "--datasets", ds],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=_env(hand),
    )
    if out.returncode != 0:
        raise SystemExit(f"{TAG} ✗ hand_pipeline --emit-run 2 failed — {out.stderr.strip().splitlines()[-1:]}")
    runs = json.loads(out.stdout[out.stdout.index("{") :])["runs"]
    if len(runs) != 1:
        raise SystemExit(f"{TAG} ✗ --emit-run 2 gave {len(runs)} commands (expected 1)")
    return runs[0]["argv"]


def _env(hand: str) -> dict:
    import os

    return {**os.environ, "DEXENGINE_HAND": hand}


def plan(ds: str, hand: str) -> tuple[str, list]:
    """(path, [(name, argv)…]); out of scope = ("na", []). Failure conditions in the module docstring."""
    if ds not in RR.rule_datasets(RULE):
        return "na", []
    axis = RR.axis_hand()
    py = str(P.retarget_py())
    exports = [(n, [py, *argv]) for n, argv in EXPORTS]
    if _roots(ds, axis):
        return "exports", exports
    have = _supp_files(ds)
    if have:
        raise InputMissingError(
            f"{TAG} ✗ {ds}: the base of axis hand {axis} is empty but the supplement layer has "
            f"{len(have)} files (e.g. {have[0]}) — owners cannot be measured on an empty base. "
            "Empty the supplement layer (INTAKE/raw_objects) and rerun"
        )
    if hand != axis:
        raise SystemExit(
            f"{TAG} ✗ {ds}: the base of axis hand {axis} is empty — supplements come from the axis-hand base. "
            f"Run the axis hand's stage 1 first (this hand {hand})"
        )
    return "bootstrap", [("base (no supplements)", _base_argv(ds, hand)), *exports]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="OakInk2 supplement stage (required objects → supports → sibling parts), before stage 1 step 2"
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--plan", action="store_true", help="print the commands only (no writes)")
    ap.add_argument("--stamp", type=Path, help="stamp file (module docstring); skip when unchanged, write after rc 0")
    a = ap.parse_args()
    way, steps = plan(a.dataset, a.hand)
    if way == "na":
        print(
            f"{TAG} ds={a.dataset} {NOT_APPLICABLE} — outside the scope {list(RR.rule_datasets(RULE))} of rule {RULE}",
            flush=True,
        )
        return 0
    if a.plan:
        for n, argv in steps:
            print(f"    2-supplement ({way} · {n}) → {' '.join(argv)}", flush=True)
        return 0
    if a.stamp is not None:
        if _stamp_same(a.stamp, a.dataset):
            return 0
        a.stamp.unlink(missing_ok=True)
    n0 = len(_supp_files(a.dataset))
    rcs = []
    for n, argv in steps:
        print(f"{TAG} {n} → {' '.join(argv)}", flush=True)
        rc = subprocess.run(argv, cwd=REPO, env=_env(a.hand)).returncode
        rcs.append(f"{n} rc={rc}")
        if rc != 0:
            print(f"{TAG} ✗ ds={a.dataset} path={way} · {' · '.join(rcs)} — stopping", flush=True)
            return rc
    print(
        f"{TAG} ds={a.dataset} hand={a.hand} axis_hand={RR.axis_hand()} path={way} · {' · '.join(rcs)} · "
        f"supplement files {n0} → {len(_supp_files(a.dataset))} · roots {_roots(a.dataset, RR.axis_hand())}",
        flush=True,
    )
    if a.stamp is not None:
        _stamp_write(a.stamp, a.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
