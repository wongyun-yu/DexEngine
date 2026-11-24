"""D37 retry — the one owner of the budget-exhausted UNPROVEN retry (rule D37 · values `datasets.EXHAUST_RETRY`).

An UNPROVEN clip whose exhaustive search stopped **because of the budget** at the canonical budget is judged **once**
more with budget x factor, continuing from the cell where the earlier search stopped. Stage 2 runs that round itself
after its status ledger (scripts/stage2_ground.sh · marker `[RETRY_BUDGET]`): the targets are judged in a sibling chunk
named by `retry_tag`, and the fold takes the newer chunk per clip. Three parts, read by every caller:

  record   retry.json (from · factor · round · clip_budget_s) — default = EXHAUST_RETRY with round 1. A retry chunk
           carries it (stage2 --retry-budget · run.env retry_budget=). budget = from x factor^round.
  targets  `classify` on the UNPROVEN rows of a chunk: n = cells the search covered when it stopped · m = candidate
           cells · checks = confirm attempts of the anchors (p0/noconfirm). Kinds `le2x` (m <= factor x n: the larger
           budget finishes the search) and `assembly` (the search reached confirm) are retried; a row of a retry chunk
           is not retried again (max_rounds 1) and stays UNPROVEN (a terminal state, `ledger_states`).
  resume   table clip · start · cands (stage2 --exhaust-resume → judge --exhaust-resume): the retry does not look at
           the first `start` cells again. The same input gives the same sorted list, so cell order · rule · budget equal
           a restart from scratch; the per-cell multi-start samples are drawn in turn from the clip's seed stream, so
           a resumed cell is not bit-equal to a from-scratch run. A candidate count that differs from the table dies.

Shell entry:
  python3 -m dexengine.placement.judge.retry plan --chunk OUT --tag TAG --out-dir D   # targets.txt · resume.tsv ·
                                                                                       # retry.json · tag.txt + marker
  python3 -m dexengine.placement.judge.retry --ensure|--check|--budget|--stage2 FILE | --path DIR   # the record
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

from dexengine.core import datasets as RR

# ── record ──────────────────────────────────────────────────────────────────────────────────────────────────────────
FILE = "retry.json"
TAG = "[RETRY_RECORD]"
FIELDS = ("from", "factor", "round", "clip_budget_s")
#: rounds with a launch path — there is one retry tag per chunk (retry_tag), so only round 1 is a chunk identity.
LAUNCH_ROUNDS = (1,)


class RecordError(SystemExit):
    """The record is missing or outside the rules — dies loudly."""


def path(camp: str | Path) -> Path:
    return Path(camp) / FILE


def default() -> dict:
    return {"from": RR.EXHAUST_RETRY["from"], "factor": RR.EXHAUST_RETRY["factor"], "round": 1, "clip_budget_s": None}


def _pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def validate(rec: dict, where: str) -> dict:
    if set(rec) != set(FIELDS):
        raise RecordError(f"{TAG} ✗ record fields {sorted(rec)} ≠ {list(FIELDS)} ({where})")
    bad = []
    if not _pos_int(rec["from"]):
        bad.append(f"from={rec['from']!r}(positive integer)")
    if not _pos_int(rec["factor"]) or rec["factor"] < 2:
        bad.append(f"factor={rec['factor']!r}(integer >= 2)")
    mr = RR.EXHAUST_RETRY["max_rounds"]
    if not _pos_int(rec["round"]) or rec["round"] > mr:
        bad.append(f"round={rec['round']!r}(1 … max_rounds {mr})")
    elif rec["round"] not in LAUNCH_ROUNDS:
        bad.append(f"round={rec['round']} — the only rounds with a launch path are {list(LAUNCH_ROUNDS)}")
    c = rec["clip_budget_s"]
    if c is not None and (isinstance(c, bool) or not isinstance(c, (int, float)) or c <= 0):
        bad.append(f"clip_budget_s={c!r}(null or positive)")
    if bad:
        raise RecordError(f"{TAG} ✗ the record is outside the rules({where}): {' · '.join(bad)}")
    return rec


def load(f: str | Path) -> dict:
    f = Path(f)
    if not f.is_file():
        raise RecordError(f"{TAG} ✗ no retry record: {f}")
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except ValueError as e:
        raise RecordError(f"{TAG} ✗ could not read the retry record: {f} ({e})") from None
    if not isinstance(rec, dict):
        raise RecordError(f"{TAG} ✗ the retry record is not an object: {f}")
    return validate(rec, str(f))


def ensure(f, factor=None, rnd=None, clip_budget_s=None, write: bool = True) -> dict:
    """No inputs = the existing record as is (the default when there is none) · inputs = the default plus the inputs,
    which must equal an existing record (the basis of retry runs already judged never changes). write=False decides
    the same without writing."""
    f = Path(f)
    given = {k: v for k, v in (("factor", factor), ("round", rnd), ("clip_budget_s", clip_budget_s)) if v is not None}
    if f.exists() and not given:
        return load(f)
    rec = validate({**default(), **given}, "input")
    if f.exists():
        old = load(f)
        if old != rec:
            raise RecordError(f"{TAG} ✗ the retry record {f} is {line(old)} — it differs from the input {line(rec)}")
        return old
    if write:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, f)
    return rec


def budget(rec: dict) -> int:
    return rec["from"] * rec["factor"] ** rec["round"]


def clip_arg(rec: dict) -> str:
    """Value of the judge argument --clip-budget-s — empty string when canonical (null)."""
    return "" if rec["clip_budget_s"] is None else f"{rec['clip_budget_s']:g}"


def judge_flags(rec: dict) -> dict[str, str]:
    """Judge arguments of a retry run {flag: value} — stage2 passes them to the judge."""
    return {"--exhaust-max": str(budget(rec)), **({"--clip-budget-s": clip_arg(rec)} if clip_arg(rec) else {})}


def line(rec: dict) -> str:
    clip = f"{clip_arg(rec)} s" if clip_arg(rec) else "canonical judge --clip-budget-s"
    return (
        f"from {rec['from']} × factor {rec['factor']}^{rec['round']} = {budget(rec)} · round {rec['round']}"
        f"(cap {RR.EXHAUST_RETRY['max_rounds']}) · clip wall clock {clip}"
    )


# ── resume table ────────────────────────────────────────────────────────────────────────────────────────────────────
RESUME_HEADER = ("clip", "start", "cands")
RESUME_TAG = "[EXH_RESUME]"


def write_resume(p, rows) -> int:
    """rows = [(clip, start, cands)] → table file. The same clip twice · negative · start > cands die. Returns the
    line count."""
    seen, lines = set(), ["\t".join(RESUME_HEADER)]
    for clip, start, cands in rows:
        start, cands = int(start), int(cands)
        if clip in seen:
            raise SystemExit(f"{RESUME_TAG} ✗ clip twice: {clip}")
        if not 0 <= start <= cands:
            raise SystemExit(f"{RESUME_TAG} ✗ {clip}: start {start} not in the 0..cands {cands} range")
        seen.add(clip)
        lines.append(f"{clip}\t{start}\t{cands}")
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(p)
    return len(seen)


def read_resume(p) -> dict[str, tuple[int, int]]:
    """Table file → {clip: (start, cands)}. A different header line or a wrong field count dies."""
    rows = Path(p).read_text().splitlines()
    if not rows or tuple(rows[0].split("\t")) != RESUME_HEADER:
        raise SystemExit(f"{RESUME_TAG} ✗ not the table format (header line {rows[:1]}): {p}")
    out = {}
    for ln in rows[1:]:
        if not ln:
            continue
        f = ln.split("\t")
        if len(f) != 3:
            raise SystemExit(f"{RESUME_TAG} ✗ field count {len(f)} ≠ 3: {ln[:80]} ({p})")
        if f[0] in out:
            raise SystemExit(f"{RESUME_TAG} ✗ clip twice: {f[0]} ({p})")
        out[f[0]] = (int(f[1]), int(f[2]))
    return out


# ── targets ─────────────────────────────────────────────────────────────────────────────────────────────────────────
#: retried kinds — the rest (gt2x · big_budget · unknown · retried) are outside the budget, have an unknown stop cell
#: or were already retried once (final).
RETRY_KINDS = ("le2x", "assembly")
#: the retry tag = the tag prefix (before the first `_`) + this letter — the one derivation rule (campaign prefixes too)
RETRY_SUFFIX = "r"
# The chunk judge.log is a record left on disk — the judge writes "checked n/m".
RE_BUDGET = re.compile(r"^\[JUDGE_BUDGET\] (\S+) \S+ checked (\d+)/(\d+)")
RE_ACT = re.compile(r"^\[ACTIVE_SIDES\] (\S+) .*exhaust_max=(\d+)")


def retry_prefix(prefix: str) -> str:
    return prefix + RETRY_SUFFIX


def retry_tag(tag: str) -> str:
    """Tag of the retry chunk of `tag`: G1<letters>_<axis>_<digits> → G1<letters>r_<axis>_<digits>."""
    pre, sep, rest = tag.partition("_")
    if not sep or not pre:
        raise SystemExit(f"[RETRY_BUDGET] ✗ tag {tag!r} has no prefix before '_' — it has no retry tag")
    return f"{retry_prefix(pre)}_{rest}"


def classify(n: int | None, m: int | None, checks: int, budget_: int | None, rec: dict) -> str:
    """kind: 'le2x' | 'assembly' | 'gt2x' | 'big_budget' | 'unknown'. n = cells the earlier search covered · m =
    candidate cells — either unknown = where it stopped is unknown. 'le2x' = within the factor (the name of the
    default factor 2)."""
    if budget_ and budget_ > budget(rec) * 2:  # already a large budget (a run outside the canon)
        return "big_budget"
    if n is None or m is None:
        return "unknown"
    if m <= rec["factor"] * n:
        return "le2x"
    return "assembly" if checks > 0 else "gt2x"


def chunk_info(d: str | Path) -> tuple[dict, dict, dict, str]:
    """({clip: (n, m)}, {clip: exhaust_max}, {clip: anchor confirm checks}, retry_budget of run.env) of one chunk."""
    d = Path(d)
    nm, bud, checks = {}, {}, {}
    f = d / "judge.log"
    if f.is_file():
        with open(f, errors="replace") as fh:
            for ln in fh:
                m = RE_BUDGET.match(ln)
                if m:
                    nm[m.group(1)] = (int(m.group(2)), int(m.group(3)))
                    continue
                m = RE_ACT.match(ln)
                if m:
                    bud[m.group(1)] = int(m.group(2))
    for g in glob.glob(str(d / "p0" / "noconfirm_*.tsv")):
        with open(g) as fh:
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 3 and p[1].startswith("anchor"):
                    checks[p[0]] = checks.get(p[0], 0) + sum(int(x) for x in re.findall(r"checks (\d+)", p[2]))
    retry = ""
    env = d / "run.env"
    if env.is_file():
        for ln in env.read_text(errors="replace").splitlines():
            if ln.startswith("retry_budget="):
                retry = ln.split("=", 1)[1].strip()
    return nm, bud, checks, retry


def row_kind(clip: str, info: tuple, rec: dict) -> str:
    """Retry kind of one UNPROVEN row (`info` = chunk_info of the chunk that judged it) — the one rule for folded
    ledger rows, regression chunks and the stage-2 round."""
    nm, bud, checks, retry = info
    if retry:  # a row of a run already retried — only once (max_rounds 1)
        return "retried"
    n, m = nm.get(clip, (None, None))
    return classify(n, m, checks.get(clip, 0), bud.get(clip), rec)


def chunk_target_rows(d: str | Path, rec: dict, kinds: dict | None = None) -> list[tuple[str, int, int]]:
    """[(clip, stop cell n, candidates m)] — UNPROVEN rows of the chunk's status ledger whose kind is retried (ledger
    order). `kinds` (optional) receives the count per kind of all UNPROVEN rows. Dies without a status ledger."""
    from dexengine.placement import ledger as SL  # noqa: PLC0415 — the one reader of chunk status ledgers

    info = chunk_info(d)
    out = []
    for r in SL.read_status_ledger(Path(d) / "status_ledger.tsv"):
        if r["status"] != "UNPROVEN":
            continue
        k = row_kind(r["clip"], info, rec)
        if kinds is not None:
            kinds[k] = kinds.get(k, 0) + 1
        if k in RETRY_KINDS:
            out.append((r["clip"], *info[0][r["clip"]]))
    return out


def plan(chunk: str | Path, tag: str, out_dir: str | Path) -> int:
    """The stage-2 D37 round of one chunk: writes targets.txt · resume.tsv · retry.json · tag.txt in out_dir and prints
    the marker `[RETRY_BUDGET] r<round> exhaust_max=<budget> n=<targets>`. Returns the target count."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = ensure(out / FILE)
    kinds: dict = {}
    rows = chunk_target_rows(chunk, rec, kinds)
    rtag = retry_tag(tag)
    write_resume(out / "resume.tsv", rows)
    (out / "targets.txt").write_text("".join(f"{c}\n" for c, _, _ in rows))
    (out / "tag.txt").write_text(rtag + "\n")
    print(
        f"{RR.RETRY_RUN_MARKER} r{rec['round']} exhaust_max={budget(rec)} n={len(rows)} · UNPROVEN kinds "
        f"{dict(sorted(kinds.items()))} · retry tag {rtag} · {line(rec)}",
        flush=True,
    )
    return len(rows)


def record_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D37 retry record")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--path", metavar="CAMP", help="record place of that campaign")
    g.add_argument("--ensure", metavar="FILE", help="writes the record (when missing) or checks it against the inputs")
    g.add_argument("--check", metavar="FILE", help="the same decision as --ensure — writes nothing")
    g.add_argument("--budget", metavar="FILE", help="retry budget (the value of run.env retry_budget=)")
    g.add_argument("--stage2", metavar="FILE", help="budget · judge arguments · [RETRY_BUDGET] marker, tab separated")
    ap.add_argument("--factor", type=int, default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--clip-budget-s", type=float, default=None)
    a = ap.parse_args(argv)
    f = a.ensure or a.check
    if not f and (a.factor, a.round, a.clip_budget_s) != (None, None, None):
        raise RecordError(f"{TAG} ✗ --factor · --round · --clip-budget-s go only with --ensure · --check")
    if a.path:
        print(path(a.path))
    elif f:
        rec = ensure(f, a.factor, a.round, a.clip_budget_s, write=bool(a.ensure))
        print(f"{TAG} {f}{'' if a.ensure else ' (check only)'} · {line(rec)}")
    elif a.budget:
        print(budget(load(a.budget)))
    else:
        rec = load(a.stage2)
        args = " ".join(f"{k} {v}" for k, v in judge_flags(rec).items())
        mark = f"{RR.RETRY_RUN_MARKER} --exhaust-max {budget(rec)} (D37 record {a.stage2} — {line(rec)})"
        print(f"{budget(rec)}\t{args}\t{mark}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["plan"]:
        return record_main(argv)
    ap = argparse.ArgumentParser(description="stage-2 D37 round of one chunk")
    ap.add_argument("--chunk", required=True, help="chunk folder (status_ledger.tsv · judge.log · p0 · run.env)")
    ap.add_argument("--tag", required=True, help="tag of that chunk (the retry tag is derived from it)")
    ap.add_argument("--out-dir", required=True, help="where targets.txt · resume.tsv · retry.json · tag.txt go")
    a = ap.parse_args(argv[1:])
    plan(a.chunk, a.tag, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
