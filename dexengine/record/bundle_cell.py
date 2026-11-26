"""Cell of a record bundle (record.bundle's owner-ledger entry): cell_of resolves one finished RL cell to its
checkpoint, the per-clip judge records and the judge env tokens (dexengine.rl.cell_session rules); owner_cells groups
the requested clips by owner cell through sources.owner_cells; cells_arm is the arm the bundle records and
cells_datasets the datasets it records."""

from __future__ import annotations

from pathlib import Path

from dexengine.core.hashing import sha256_file
from dexengine.record import sources as OWN
from dexengine.rl import cell_session as CR
from dexengine.rl.judge import metrics as JM


def cell_of(
    a,
    campaign: str,
    bundle: str,
    ckpt: Path | None = None,
    it: int | None = None,
    kind: str = "f",
    label: str | None = None,
) -> dict:
    """One finished cell's checkpoint, the per-clip records of its judge and the judge env. No ckpt = the cell's best
    (dexengine.rl.cell_session._target)."""
    _man, _b, sdir, run = CR._cell(a.hand, campaign, bundle, a.main)
    agg = None
    if ckpt is None:
        ck, folder, label, it, agg = CR._target(run, sdir)
        ckpt = folder / "checkpoints" / ck
    rates = JM.load_clips(sdir / JM.CLIPS.format(run=run, kind=kind, it=it))
    # recorded judge command -> training env tokens: call the rule owner
    # dexengine.rl.judge.rejudge.judge_command(--relocate) through dexengine.rl.cell_session.session_env as it is (no
    # second rule)
    env, _moved = CR.session_env(sdir, a.main)
    seq = next(t.split("=", 1)[1] for t in env if t.startswith("SEQ_DIR="))
    return {
        "campaign": campaign,
        "bundle": bundle,
        "run": run,
        "sdir": sdir,
        "ckpt": ckpt,
        "ckpt_sha": sha256_file(ckpt),
        "folder": ckpt.parent.parent,
        "label": label,
        "iter": it,
        "judge_agg_pct": agg,
        "rates": rates,
        "env": env,
        "seq_dir": seq,
    }


def owner_cells(a, names: list[str]) -> list[tuple[dict, list[str]]]:
    """Owner-ledger entry (the name the record, redraw and augment call): the one rule is sources.owner_cells (copy
    check and marker included)."""
    return OWN.owner_cells(a, names, cell_of)


def cells_datasets(cells: list[dict]) -> list[str]:
    """The datasets of a bundle's cells — each cell's judge env states one (DEXENGINE_DATASET); a cell without exactly
    one dies (the recorded dataset is never assumed)."""
    out = set()
    for c in cells:
        got = [t.split("=", 1)[1] for t in c["env"] if t.startswith("DEXENGINE_DATASET=")]
        if len(got) != 1:
            raise SystemExit(
                f"[COLLECT] cell {c['campaign']}/{c['bundle']}: judge env DEXENGINE_DATASET {got} — one value"
            )
        out.add(got[0])
    return sorted(out)


def cells_arm(cells: list[dict]) -> str:
    """The arm of a bundle's cells — each cell's judge env states it (DEXENGINE_ARM); a cell without one, cells on
    different arms or no cell at all die (the recorded robot is never assumed)."""
    arms = set()
    for c in cells:
        got = [t.split("=", 1)[1] for t in c["env"] if t.startswith("DEXENGINE_ARM=")]
        if len(got) != 1:
            raise SystemExit(f"[COLLECT] cell {c['campaign']}/{c['bundle']}: judge env DEXENGINE_ARM {got} — one value")
        arms.add(got[0])
    if len(arms) != 1:
        raise SystemExit(f"[COLLECT] the bundle's cells name the arms {sorted(arms)} — one arm per bundle")
    return arms.pop()
