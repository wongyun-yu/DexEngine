"""Generation hold of the judged cell bundle runner (augment.cells): the check before judging.

For every clip without a region dump, the judge input fingerprint the judge would write (input_record.fingerprint ·
run fields = the --canon-conditions entry of input_record · chunk active sides) is compared with the adopted row of
the final ledger. The reason function is the one the judge process uses (placement.judge.aug_cells.gen_reason, which
AugCells._check_gen dies on). A mismatched clip alone leaves judging, choice and count and is kept in RUN_DONE.json
held{clip: reason} (clips stay the whole bundle, so the queue comparison holds); the bundle goes on with the other
clips. The reader is augment.judged.held_of alone.
A clip with a region dump of this commit is not measured here: d_state compares the dump's fingerprint with the
adopted row. A dump of another commit is judged again (cells_pick.other_commit), so its clip is measured like one
without a dump.
A clip this check passes that the judge then finds mismatched still kills the judge (check and judge disagree = a
defect). Every clip held dies (a bundle-wide generation problem: look at the ledger and the bake first). Marker
[GEN_HOLD].
A clip whose hold is lifted (ledger repaired) has no region dump and becomes work; when that bundle's judged folder is
full the reuse gate dies, so repair the ledger before that bundle runs, or clear the bundle's judged · ikc and run L
again.
A run field that cannot be measured now (error:) dies instead of holding: an input missing on this machine is not a
ledger generation difference.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from dexengine.placement.judge import aug_cells as JAC
from dexengine.placement.judge import input_record as JFP
from dexengine.record.augment import cells_pick as ACP
from dexengine.record.augment import judged_region as JR
from dexengine.record.augment.cells_steps import ROOT

MARK = "[AUG_CELLS_RUN] [GEN_HOLD]"


def die(msg: str):
    raise SystemExit(f"{MARK} ✗ {msg}")


def canon_cond(cond: dict, env: dict) -> dict:
    """The run fields of the judge input fingerprint — the --canon-conditions entry of input_record, measured with the
    judge's interpreter (isaac_py) and environment (the run fields read the judge's dependencies)."""
    q = shlex.quote
    sh = (
        '. scripts/_isaac_py.sh && exec "$(isaac_py)" -m dexengine.placement.judge.input_record --canon-conditions '
        f"{q(cond['dataset'])} {q(cond['scene'])} {q(cond['arm'])}"
    )
    r = subprocess.run(["bash", "-c", sh], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=False)
    if r.returncode or not r.stdout.strip():
        die(
            f"judge run fields (--canon-conditions {cond['dataset']} {cond['scene']} {cond['arm']}) rc {r.returncode}: "
            f"{r.stderr[-300:]}"
        )
    return json.loads(r.stdout.strip().splitlines()[-1])


def gen_holds(dataset: str, clips: list[str], rows: dict, conds: dict, env_of) -> dict:
    """Clip -> reason (mismatched clips only); the run fields are measured once per (dataset, scene, arm);
    env_of(cond) = the judge environment."""
    hand_dir = JFP._rr().hand_dir()
    cc: dict = {}
    out = {}
    for c in clips:
        cd = conds[c]
        key = (cd["dataset"], cd["scene"], cd["arm"])
        if key not in cc:
            cc[key] = canon_cond(cd, env_of(cd))
        jfp = JFP.fingerprint(dataset, cd["emb"], cd["hand"], c, cd["scene"], cc[key], cd["active"])
        if bad := JFP.err_fields(jfp):
            die(f"{c}: the judge input cannot be measured on this machine {bad}: an input (object, tree) is missing")
        if why := JAC.gen_reason([rows[c]], jfp, (dataset, cd["emb"], hand_dir, c)):
            out[c] = why
    return out


def to_measure(bundle: Path, clips: list[str]) -> list[str]:
    """The clips the judge will judge in mode D: no region dump, or a dump of another commit (judged again)."""
    head = JAC.head_commit()
    return [
        c for c in clips if not (p := JAC.region_path(bundle, c)).is_file() or ACP.other_commit(JR.read_region(p), head)
    ]


def split(bundle: Path, dataset: str, clips: list[str], rows: dict, conds: dict, env_of) -> tuple[list, dict]:
    """-> (clips to judge, holds {clip: reason}); the clips judge D will judge are measured (to_measure); one
    [GEN_HOLD] line per hold; every clip held dies."""
    held = gen_holds(dataset, to_measure(bundle, clips), rows, conds, env_of)
    for why in held.values():
        print(f"{MARK} {why} — this clip alone is held (out of judge · choice · count; RUN_DONE held)", flush=True)
    live = [c for c in clips if c not in held]
    if not live:
        die(f"every clip of the bundle ({len(clips)}) is held: a bundle-wide generation problem (ledger · bake) first")
    return live, held
