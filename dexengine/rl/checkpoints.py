"""[CANON_USE]: the one place where a checkpoint meets a clip list. The trainer entry (dexengine/rl/cli.py) calls
check_args() in both data branches (--sequence_dir once its roster is resolved, else the --sequence/--data_dir trees)
for every argument that loads policy weights (WEIGHT_ARGS: a resumed training, the frozen-replay judge, a RENDER_ONLY
render, a rescore, the one-hand render and objtrace (--checkpoint), the dual eval), before the env is built.

  cell         a checkpoint under a campaign cell: <the hand's run_rl_campaigns layer>/<campaign>/<ds>/<bundle>/ holding
               launch.json, with the checkpoint below it (skrl/<RUN>/checkpoints/agent_N.pt, links resolved). A
               checkpoint outside a cell (a run's own work area, e.g. the in-run judge) is marked and passes without
               the checks below.
  checks       on this machine: every listed clip is one the cell trained (launch.json clips) · its tree
               task_info.json here (SEQ_DIR) against the slot's training fingerprint (the campaign's manifest copy →
               its frozen per-clip table under the queue root, slot_generation.trained_of) and the TURNOVER
               declarations (slot_generation.slot_state): an old-generation slot ('turned') raises, a tree that
               changed without a declaration raises, a same_gen successor passes. Queue root = env
               DEXENGINE_RL_QUEUE_ROOT (no default: without the declarations a machine cannot know the clip is still
               this checkpoint's).
  marker       [CANON_USE] host · tree commit · checkpoint · clips n/n current · same_gen k · turned 0.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

from dexengine.core import paths as P
from dexengine.rl import cell_files as CF  # cell file names: the trainer entry never imports the campaign runner
from dexengine.rl import slot_generation as SG

TAG = "[CANON_USE]"
#: words of the marker for a checkpoint outside every campaign cell; record.augment.run reads them from its session log
OUTSIDE_CELL = "checkpoint outside the campaign cells"
ENV = "DEXENGINE_RL_QUEUE_ROOT"
#: Every trainer argument that loads policy weights from a path.
WEIGHT_ARGS = ("resume_from", "resume", "checkpoint", "dual_ckpt_right", "dual_ckpt_left")


def weight_args(args) -> list[tuple[str, str]]:
    """[(argument, checkpoint path)] of the weights this run loads (WEIGHT_ARGS that are set)."""
    return [(n, str(v)) for n in WEIGHT_ARGS if (v := getattr(args, n, None))]


def check_args(args, seq_dir: str | Path, trees: list[str], hand: str, qroot: str | None = None) -> list[str]:
    """The marker line of each checkpoint the run loads (weight_args) against the clips it reads (check); ValueError
    naming the argument on the first that fails."""
    out = []
    for n, ck in weight_args(args):
        try:
            out.append(f"{check(ck, seq_dir, trees, hand, qroot)} · --{n}")
        except ValueError as e:  # noqa: PERF203 — the failing flag is named in the re-raised error
            raise ValueError(f"--{n} {e}") from None
    return out


def cell_of(ckpt: str | Path, hand: str) -> Path | None:
    """The campaign cell a checkpoint lies in (module doc); None = not in one."""
    cells = P.run_path("run_rl_campaigns", hand).name
    for d in Path(ckpt).resolve().parents:
        if cells in d.parts[:-1] and CF.cell_file(d, "launch").is_file():
            return d
    return None


def _commit() -> str:
    """The code tree's short HEAD commit ('?' when git does not answer)."""
    r = subprocess.run(
        ["git", "-C", str(P.WT), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=60
    )
    return r.stdout.strip() or "?"


def check(ckpt: str | Path, seq_dir: str | Path, trees: list[str], hand: str, qroot: str | None = None) -> str:
    """The marker line of a checkpoint about to read `trees` under `seq_dir` (module doc); ValueError on any failure
    (the caller dies before the env is built). qroot = the queue root (None = env DEXENGINE_RL_QUEUE_ROOT)."""
    cell = cell_of(ckpt, hand)
    where = f"host {socket.gethostname()} · tree {P.WT} @{_commit()}"
    if cell is None:
        return f"{TAG} {OUTSIDE_CELL} — not compared (training work area) · {where} · {ckpt}"
    q = qroot if qroot is not None else os.environ.get(ENV, "").strip()
    if not q:
        raise ValueError(
            f"checkpoint {ckpt} belongs to campaign cell {cell} but {ENV} (the queue root — no default) "
            "is not set — not running without knowing whether the clips are still this checkpoint's "
            "generation"
        )
    rec = json.loads(CF.cell_file(cell, "launch").read_text(encoding="utf-8"))
    if rec.get("hand") != hand:
        raise ValueError(f"cell {cell} hand {rec.get('hand')} ≠ this run's hand {hand}")
    mf = CF.manifest_dir(cell.parents[1]) / "manifest.json"
    if not mf.is_file():
        raise ValueError(f"cell {cell}: campaign manifest copy {mf} missing — cannot read the training fingerprints")
    man, sha = SG.load_manifest(mf)
    b = next((x for x in man["bundles"] if x["id"] == rec["bundle"]), None)
    if b is None or b["clips"] != rec["clips"]:
        raise ValueError(f"cell {cell}: the clips of bundle {rec['bundle']} differ from campaign manifest {sha[:12]}")
    tr = SG.trained_of(Path(q), man, sha)
    turns = SG.turns_of(SG.read_turnover(Path(q)), hand)
    snaps = SG.read_snaps(Path(q), hand)
    clips = [P.clip_name(t) for t in trees]
    out = sorted(set(clips) - set(b["clips"]))
    if out:
        raise ValueError(f"roster clips {len(out)} {out[:3]} are outside this checkpoint's bundle {b['id']}")
    bad, same, pend = [], 0, 0
    for c in clips:
        k = (b["axis"], c)
        cur = SG.tree_sha(Path(seq_dir), c)
        try:
            st = SG.slot_state(tr[k], cur, turns.get(k, []), snaps.get(k, ()), lineage=SG.lineage_of(man))
        except ValueError as e:
            bad.append(f"{c}: {e}")
            continue
        if st == "turned":
            bad.append(f"{c}: old-generation slot (TURNOVER) — not a canonical clip of this checkpoint")
        same += cur != tr[k] and st == "current"
        pend += st == "pending"  # a declared rebuild before its split result: the old checkpoint judges the new tree
    if bad:
        raise ValueError(
            f"{len(bad)}/{len(clips)} clips are not the checkpoint's generation in this machine's trees: {bad[:3]}"
        )
    return (
        f"{TAG} {where} · checkpoint {Path(ckpt).name}(cell {rec['bundle']} · manifest {sha[:12]}) · "
        f"SEQ_DIR {seq_dir} · clips {len(clips)}/{len(clips)} current generation · same_gen {same} · "
        f"pending {pend} "
        "· turned 0"
    )
