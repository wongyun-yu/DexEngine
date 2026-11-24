"""Record checkpoint choice: the owner-ledger entry (ledger = rl_clip_owner.tsv; writing and rule owner
dexengine.rl.acceptance).

Canonical ledger = acceptance.canon_ledger_path(hand): ledger_path under the owner table root paths.owner_root()
(the layout file's owner_root — a site names its ledger host's place there — else the data root; the ledger's writer is
dexengine.rl.owner_ledger). Launchers do not trust an --owner at an arbitrary path:
  host     at launch `sources.py canon --hand <hand>` measures the canonical ledger's sha256 (the root is not an
           argument; stdout `<sha>\t<path>`; missing file or another header -> dies).
  launcher the driver, record.bundle and the augment run take --owner (the received file; on a remote host the sent
           copy) and --owner-sha (the launching host's value) together; a received file sha256 != --owner-sha dies · no
           --owner-sha dies (no silent default). Marker `[CKPT_OWNER] src=<path> sha=<…> canon_sha=<…>` (the driver
           requires it in every bundle log; without it that bundle is invalid).
  extra-cell jobs (cells_ext) use verify_entry: when they get the canonical place itself they do not check the file
           sha (it is rewritten while jobs run), and at the end the owner rows of that bundle's clips (rows_sha) are
           checked against the start value (verify_rows; a change discards the results).
  the redraw (looks.rev) finds cells again with the ledger used at record time, not the canonical one (identity =
           the episode source_cell check): the recorded sha (meta/collect.json owner_sha) when present, else for old
           records that file's sha with a `recorded ledger` marker.
The copy the driver froze at its first launch (<operations place>/ledger.tsv) only records 'what was used' and the
queue condition (same tag = same clip list); it is not used for choosing. At launch (driver start, each bundle's
record.bundle) when the owner fields (OWNER_COLS) of those clips in the copy differ from the current ledger in any way
it dies (a clip whose owner changed by a tie break is not collected with the old checkpoint; relaunch with a new tag).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.hashing import sha256_file

MARK = "[CKPT_OWNER]"
#: same owner = all these fields equal (state, cell, run, checkpoint file, judge kind, iteration)
OWNER_COLS = ("state", "campaign", "bundle", "run", "ckpt_rel", "kind", "iter")
# ledger kind -> judge record name letter (dexengine.rl.judge.metrics.CLIPS; unknown dies)
OWNER_KIND = {"final": "f", "special": "q"}
OWNER_LINE = f"{MARK} owner ledger "  # head of the launch check line; log_problems requires it


def _sha(p: Path) -> str:
    return sha256_file(p)[:12]


def verify_src(src: Path, canon_sha: str | None) -> str:
    """Received ledger file vs the canonical sha measured at launch: missing, malformed or different dies. One marker
    line · returns the sha."""
    if not canon_sha:
        raise InputMissingError(
            f"{MARK} ✗ no --owner-sha (canonical owner ledger sha256 measured by the launching host): no "
            "default · on that host: sources.py canon --hand <hand>"
        )
    if len(canon_sha) != 64 or any(c not in "0123456789abcdef" for c in canon_sha):
        raise SystemExit(f"{MARK} ✗ --owner-sha {canon_sha!r}: must be 64 hex characters of a sha256")
    if not Path(src).is_file():
        raise InputMissingError(f"{MARK} ✗ ledger file missing {src}")
    got = sha256_file(src)
    if got != canon_sha:
        raise SystemExit(
            f"{MARK} ✗ src={src} sha={got[:12]} != canon_sha={canon_sha[:12]}: not the canonical ledger "
            "(an old sent copy or a copy) · send the canonical ledger from the launching host again and launch with "
            "its sha"
        )
    print(f"{MARK} src={src} sha={got} canon_sha={canon_sha}", flush=True)
    return got


def rows_sha(src: Path, names) -> str:
    """sha256 of **the owner rows of the job's clips** only, not of the whole ledger file. The rows of the names (axis,
    clip, OWNER_COLS) are sorted and hashed with the name list: added or changed unrelated rows leave it as is, and it
    changes when a row of those clips changes, appears or disappears. Long jobs take it at the start and measure it
    again at the end with the same names (verify_rows): the canonical ledger is rewritten while jobs run (OWNER_ADD),
    so measuring the file sha again at the end discarded judge results on unrelated changes."""
    import json

    from dexengine.rl import acceptance as CO

    want = sorted(set(names))
    keep = set(want)
    rows = sorted(
        "\t".join([r["axis"], r["clip"], *(str(r[c]) for c in OWNER_COLS)])
        for r in CO.read_ledger(src)
        if r["clip"] in keep
    )
    return hashlib.sha256(json.dumps([want, rows], ensure_ascii=False).encode()).hexdigest()


def verify_rows(src: Path, names, want: str) -> str:  # noqa: V103 — public API for plugin cell jobs
    """End check: rows_sha(names) measured on the current ledger != the start value dies (the owner of those clips
    changed = the results are discarded). One marker line."""
    names = sorted(set(names))
    got = rows_sha(src, names)
    if got != want:
        raise SystemExit(
            f"{MARK} ✗ owner rows changed: clips {len(names)} · start {want[:12]} != now {got[:12]} · "
            f"{src} "
            "(this job's results are discarded; run again with the new owners)"
        )
    print(f"{MARK} owner rows match {got[:12]} · clips {len(names)} · {src}", flush=True)
    return got


def verify_entry(src: Path, canon_sha: str | None, hand: str) -> str:  # noqa: V103 — public API for plugin cell jobs
    """Job entry ledger check: when given the canonical place itself (the ledger host's
    acceptance.canon_ledger_path(hand)), that file is the canonical one and its sha is not checked (a race: the
    host measures, then rewrites before the job reads; repeated runner failures; clip owners are checked by rows_sha
    and verify_rows at the job end). Other places (a sent copy, copies) use verify_src (file sha) as they are.
    canon_sha is required for both. One marker line · returns the current file sha."""
    from dexengine.rl import acceptance as CO

    if not canon_sha:
        return verify_src(src, canon_sha)
    if Path(src).resolve() != Path(CO.canon_ledger_path(hand)).resolve():
        return verify_src(src, canon_sha)
    got = sha256_file(src)
    print(
        f"{MARK} src={src} sha={got} canon_sha={canon_sha} · canonical place (no file sha check; owner rows "
        f"checked at the end)",
        flush=True,
    )
    return got


def canon(hand: str, path: Path) -> tuple[Path, str]:
    """The ledger file (CLI = dexengine.rl.acceptance.canon_ledger_path; tests = ledger_path of a temporary root) and
    its sha256; a missing file or another header dies."""
    import time

    from dexengine.rl import acceptance as CO

    p = Path(path).resolve()
    if not p.is_file():
        raise InputMissingError(f"{MARK} ✗ canonical ledger missing {p} (hand {hand})")
    CO.read_ledger(p)
    sha = sha256_file(p)
    import sys

    print(
        f"{MARK} canon {hand} {p} sha={sha} · written {time.time() - p.stat().st_mtime:.0f} s ago",
        file=sys.stderr,
        flush=True,
    )
    return p, sha


def diff(live: Path, pin: Path, names) -> list[str]:
    """Names whose owner fields in the copy != the current ledger -> list of reasons (name, copy -> now). Missing from
    the current ledger is a reason too."""
    from dexengine.rl import acceptance as CO

    want = set(names)
    now = {(r["axis"], r["clip"]): r for r in CO.read_ledger(live) if r["clip"] in want}
    out = []
    for r in CO.read_ledger(pin):
        if r["clip"] not in want:
            continue
        n = now.get((r["axis"], r["clip"]))
        if n is None:
            out.append(f"{r['clip']}: copy {r['bundle']}@{r['campaign']} -> missing from the current ledger")
        elif any(n[c] != r[c] for c in OWNER_COLS):
            out.append(
                f"{r['clip']}: copy {r['state']} {r['bundle']}@{r['campaign']} {r['kind']} {r['iter']} -> "
                f"now {n['state']} {n['bundle']}@{n['campaign']} {n['kind']} {n['iter']}"
            )
    return out


def check(live: Path, pin: Path | None, names) -> None:
    """Launch check: with a copy, diff must be empty (else dies) · one marker line."""
    names = list(names)
    if pin is not None:
        bad = diff(live, pin, names)
        if bad:
            raise SystemExit(
                f"{MARK} ✗ frozen copy {pin} != current owner ledger {live}: {len(bad)} clips "
                f"(relaunch with a new tag): {bad[:4]}"
            )
    print(
        f"{OWNER_LINE}{live} sha {_sha(live)} · clips {len(names)} · "
        + (f"copy {pin} sha {_sha(pin)} owners match {len(names)}/{len(names)}" if pin is not None else "no copy"),
        flush=True,
    )


def log_problems(text: str) -> list[str]:
    lines = text.splitlines()
    out = (
        []
        if any(ln.startswith(f"{MARK} src=") for ln in lines)
        else [f"marker '{MARK} src=' missing: that run is invalid"]
    )
    return out + ([] if any(ln.startswith(OWNER_LINE) for ln in lines) else [f"marker '{OWNER_LINE.strip()}' missing"])


def cell_groups(owner: Path, names: list[str], main: Path) -> tuple[list[dict], list[tuple], dict]:
    """Ledger -> (rows, (canonical axis, clip) per name, {(campaign, bundle, run, checkpoint rel): clips}): the one home
    of the cell grouping (owner_cells and the spec cut read it). A missing or doubled canonical axis dies."""
    from dexengine.rl import acceptance as CO

    rows = CO.read_ledger(owner)
    keys = []
    for n in names:
        ax = sorted({r["axis"] for r in rows if r["clip"] == n and r["state"] == "canon"})
        if len(ax) != 1:
            raise SystemExit(f"{MARK} ledger {owner}: canonical axis of {n} is {ax}: must be exactly one")
        keys.append((ax[0], n))
    return rows, keys, CO.use_rows(owner, keys, main)


def owner_cells(a, names: list[str], cell_of) -> list[tuple[dict, list[str]]]:
    """Owner-ledger entry: per name the one (axis, clip) of its canonical (canon) row (a missing or doubled canonical
    axis dies; rows in other states such as dropped are not owners and are left out of the axis count) -> copy check
    (check) -> the checkpoint groups of dexengine.rl.acceptance.use_rows -> [(cell (cell_of), that checkpoint's names)].
    A ledger row whose run differs from the cell record dies."""
    verify_src(a.owner, getattr(a, "owner_sha", None))
    rows, keys, groups = cell_groups(a.owner, names, a.main)
    check(a.owner, getattr(a, "owner_pin", None), names)
    by = {(r["axis"], r["clip"]): r for r in rows}
    out = []
    for (camp, bun, run, rel), clips in groups.items():
        r = by[next(k for k in keys if k[1] == clips[0])]
        if r["kind"] not in OWNER_KIND:
            raise SystemExit(f"{MARK} ledger kind {r['kind']!r}: outside {tuple(OWNER_KIND)}")
        it = int(r["iter"])
        cell = cell_of(
            a,
            camp,
            bun,
            ckpt=a.main / rel,
            it=it,
            kind=OWNER_KIND[r["kind"]],
            label=f"ledger {r['kind']} {it} ({r['rate_pct']} %)",
        )  # data: lands in meta/collect.json cells
        if cell["run"] != run:
            raise SystemExit(f"{MARK} ledger run {run} != cell record {cell['run']} ({bun})")
        out.append((cell, clips))
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="launching host: sha256 of the canonical owner ledger (the launchers' --owner-sha "
        "value); stdout `<sha>\t<path>`"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("canon")
    c.add_argument("--hand", required=True)
    a = ap.parse_args(argv)
    from dexengine.rl import acceptance as CO

    p, sha = canon(a.hand, CO.canon_ledger_path(a.hand))
    print(f"{sha}\t{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
