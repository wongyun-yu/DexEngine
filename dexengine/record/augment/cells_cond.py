"""Conditions the judged-cell bundle runner (record.augment.cells) adds around its stage state.

Re-selection margin (record form owner placement.bake.place_check): a clip in the place_reselect record of
its source chunk's run.env was judged canonically with the robot spot disc widened by the margin. The cell judge must
search under the same condition, so that clip's margin (mm) goes into the judge condition
place_reselect_mm and only that condition group's judge run gets --place-reselect. The margin is a search condition
outside the judge input record, so RUN_DONE conditions is its record. A clip outside the record has margin 0 (no
condition key). Marker [PLACE_RESELECT].

Manual holds: a clip adopted before the current placement rule whose canonical placement breaks the rule's
robot spot is taken out of cell judging and its canonical placement is judged again; so is a clip
whose scene the current rules cannot build yet (the reason names the missing rule). The one hold table is the repo
config HOLD_TABLE (header clip\treason) — every judging run reads it; only this bundle's clips on it are added to the
holds — outside judging, picking and the counts, recorded in RUN_DONE held with the stored prefix USER_HOLD. Markers
[USER_HOLD] table … (the table read, every judging run) · [USER_HOLD] <clip>.
The finished-bundle check (--verify-only · harvest) follows the USER_HOLD entries recorded in that bundle's
RUN_DONE held (recorded_holds — the record is the decision; a later table edit does not rewrite a finished bundle).

f0 shadow check (f0_again): a region dump about to be reused that did not come from the current f0 shadow
(placement.judge.f0_key.stale_why) is judged again, and that clip's old cell list is removed (drop_stale_cells).
Judging runs only — the finished-bundle check (--verify-only) is outside this gate. Markers [F0_KEY] · [REGION_AGAIN].
"""

from __future__ import annotations

from pathlib import Path

from dexengine.core.errors import input_missing
from dexengine.core.repo import REPO_ROOT

MARK = "[AUG_CELLS_RUN]"
#: stored prefix of a manual hold in RUN_DONE held (recorded_holds matches it)
USER_HOLD = "[USER_HOLD]"
HOLD_HEAD = ["clip", "reason"]
#: **The one manual hold table** (repo-relative; the one owner of its path) — read by every judging run (holds).
HOLD_TABLE = "configs/record/user_holds.tsv"


def hold_table() -> Path:
    """The hold table's path in this code checkout (REPO_ROOT / HOLD_TABLE)."""
    return REPO_ROOT / HOLD_TABLE


def _die(msg: str):
    raise SystemExit(f"{MARK} ✗ {msg}")


def reselect_of(chunk_dir: Path, rel: str, tag: str) -> str:
    """run.env place_reselect value of a source chunk → the record's absolute place ('' = none) · a value without the
    file dies (the margin is not made up)."""
    rel = (rel or "").strip()
    if rel and not (Path(chunk_dir) / rel).is_file():
        _die(
            input_missing(
                f"source chunk {tag} run.env place_reselect={rel} but {Path(chunk_dir) / rel} is missing — "
                "the margin is not made up"
            )
        )
    return str(Path(chunk_dir) / rel) if rel else ""


def with_reselect(cond: dict, ck: dict, row: dict) -> dict:
    """Judge condition + the clip's re-selection margin (mm) — a clip outside the record keeps the condition as is."""
    if ck.get("reselect"):
        from dexengine.placement.bake import place_check as PC  # noqa: PLC0415 — record form owner

        m = PC.read_reselect(ck["reselect"]).get(row.get("clip", ""))
        if m is not None:
            cond = dict(cond, place_reselect_mm=round(m * 1000.0, 3))
    return cond


def write_reselect(cond: dict, clips: list[str], work: Path) -> Path | None:
    """The condition group's re-selection record (place_check.RESELECT_IN) — with a margin in the condition, every
    clip of the group at that margin (group = same condition = same margin) · none → None. The judge reads clip and
    margin only (read_reselect; short_mm · verdict are information columns)."""
    mm = cond.get("place_reselect_mm")
    if mm is None:
        return None
    from dexengine.placement.bake import place_check as PC  # noqa: PLC0415

    f = Path(work) / PC.RESELECT_IN
    f.write_text(
        "\t".join(PC.RESELECT_COLS)
        + "\n"
        + "".join(f"{c}\t-\t{float(mm):.2f}\taug_cells_run(source chunk re-selection record)\n" for c in clips)
    )
    if set(PC.read_reselect(f)) != set(clips):
        _die(f"re-selection record {f} clips != judge list")
    print(f"{MARK} {PC.RESELECT_TAG} condition margin {mm} mm · clips {len(clips)} → {f.name}", flush=True)
    return f


def hold_rows(path: str | None = None) -> dict[str, str]:
    """{clip: reason} of the hold table (default the repo table, hold_table()). A missing table, a wrong header, an
    empty reason or a clip on two rows dies. Readers: user_holds (cell judging) · data.candidates (a first-pass drop of
    a held clip is held instead)."""
    path = str(path or hold_table())
    if not Path(path).is_file():
        _die(f"hold table {path} is missing (the repo table HOLD_TABLE — an empty table keeps its header)")
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if not rows or rows[0].split("\t")[:2] != HOLD_HEAD:
        _die(f"hold table {path} header is not clip\\treason")
    tab: dict = {}
    for n, ln in enumerate(rows[1:], 2):
        f = ln.split("\t")
        if len(f) < 2 or not f[0].strip() or not f[1].strip() or f[0] in tab:
            _die(f"hold table {path}:{n} bad row (columns · empty reason · same clip twice) — {ln!r}")
        tab[f[0]] = f[1]
    return tab


def user_holds(path: str, live: list[str], held: dict) -> tuple[list, dict]:
    """Adds this bundle's clips of the hold table to the holds → (live, held). A bad table (hold_rows) or a bundle held
    entirely dies."""
    tab = hold_rows(path)
    print(f"{MARK} {USER_HOLD} table {path} rows={len(tab)} · this bundle {sum(c in tab for c in live)}", flush=True)
    held = dict(held)
    for c in live:
        if c in tab:
            held[c] = f"{USER_HOLD} {tab[c]}"
            print(
                f"{MARK} {USER_HOLD} {c}: {tab[c]} — this clip only is held (outside judging · picking · counts · "
                "RUN_DONE held)",
                flush=True,
            )
    live = [c for c in live if c not in held]
    if not live:
        _die("every clip of the bundle is held — look at the bundle-level problem first")
    return live, held


def recorded_holds(bundle: Path, live: list[str], held: dict) -> tuple[list, dict]:
    """Finished-bundle check — adds the USER_HOLD entries of RUN_DONE held (reason starting with USER_HOLD) as they are
    (no record: unchanged)."""
    import json  # noqa: PLC0415

    f = Path(bundle) / "RUN_DONE.json"
    rec = (json.loads(f.read_text()).get("held") or {}) if f.is_file() else {}
    held = dict(held)
    for c in live:
        if str(rec.get(c, "")).startswith(USER_HOLD):
            held[c] = rec[c]
            print(f"{MARK} {USER_HOLD} {c}: user hold of the record (RUN_DONE) — held in the check too", flush=True)
    return [c for c in live if c not in held], held


def holds(live: list[str], held: dict, bundle: Path, verify_only: bool) -> tuple[list, dict]:
    """The one entry of manual holds — judging runs: the repo table (hold_table, missing = death) · finished-bundle
    check: the record."""
    if verify_only:
        return recorded_holds(bundle, live, held)
    return user_holds(str(hold_table()), live, held)


def f0_again(bundle: Path, done: list, todo: list, conds: dict, hand: str) -> tuple[list, list]:
    """Dumps to reuse (done) whose f0 shadow was built again → appended to the to-do list (todo).
    f0 place = the judge condition f0 (the judge's DEXENGINE_F0_STAGE)."""
    import numpy as np  # noqa: PLC0415

    from dexengine.core.paths import hand_spec_dir  # noqa: PLC0415
    from dexengine.placement.judge import aug_cells as JAC  # noqa: PLC0415
    from dexengine.placement.judge import f0_key as FK  # noqa: PLC0415

    keep, again = [], []
    for c in done:
        p = JAC.region_path(bundle, c)
        with np.load(p, allow_pickle=False) as z:
            key = str(z["f0_key"]) if "f0_key" in z.files else ""
        mk = FK.marker(conds[c]["f0"], hand_spec_dir(hand), conds[c]["emb"], c)
        why = FK.stale_why(key, p.stat().st_mtime, mk)
        (again if why else keep).append(c)
        if why:
            print(f"{MARK} [F0_KEY] {c}: {why} → judged again", flush=True)
    return keep, todo + again


def drop_stale_cells(bundle: Path, todo: list) -> int:
    """Old cell lists of the clips judged again — picked from the old dump, so useless (left in place, the cell choice
    would die on 'changed'). With judged chunks present the runner dies before this. Returns the number removed."""
    from dexengine.placement.judge import aug_cells as JAC  # noqa: PLC0415

    n = 0
    for c in todo:
        f = JAC.cells_path(bundle, c)
        if f.is_file():
            f.unlink()
            n += 1
            print(f"{MARK} [REGION_AGAIN] {c}: removed the old cell list {f}", flush=True)
    return n
