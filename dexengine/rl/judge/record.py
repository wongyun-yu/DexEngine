"""The judge record of an RL campaign cell: its one reader and shape check (the writer is
dexengine/rl/judge/metrics.py `extras`; the rules live there):

  <RUN>_judge_extras.json (cell_state.judge_extras_file, in the cell, written before the DONE rc) →
  judge_extras(run, sdir) = Extras(ten_pct, special, special_pending, sha256, best, last, …) — every reader reads the
  record here, finalized cell or not. FINALIZED.json carries no copy of these values: finalize writes only
  `extras_sha256` (the record it kept checkpoints by · '-' = none) and finalized_claim_check holds the record to that
  statement. Older FINALIZED.json files still carry copied keys; nothing reads their values, their presence alone
  says a record existed.

  ten_pct          {k, n, clips {clip: [complete, total, pass 0|1]}[, evals]} — k = Σ pass · n = clips
  special          [{tag, ckpt, clips[, counts {clip: [complete, total]}]}] — special-on only
  special_pending  [{iter, ckpt | None, status}] — special-on mid judges not OK (their checkpoints are kept)
  best · last      {iter, kind q|f, ckpt, agg_pct, pass10 [k, n]} (+ best: among = the OK judged iterations it was
                   picked from) — last = the last final (one of among); both or neither.
  judge            {ne, epc} — the judging setting these were measured with (judge NE · evaluations asked per clip):
                   a table read next to one of another setting is marked as such.
  finals           [{iter, agg_pct, pass10 [k, n]}] — each final judge (judge/metrics.py final_iters: MI × 7/8 and
                   MI), ascending; the last is `last`'s iteration. Absent = an older record (three finals).
  arm_pen          {final iter: {pair kind: {max_mm ≥ 0, frac ∈ [0, 1]}}} — the penetration of each final judge in
                   the pairs the simulator does not collide (judge/arm_penetration.py owns the kinds); an auxiliary
                   value, never the judge's.
A missing key = not measured (never 0); an unknown key dies.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple

from dexengine.core import paths as P
from dexengine.rl import cell_state as CH

CKPT_NAME = re.compile(r"agent_(\d+)\.pt")  # a checkpoint's file name (the trainer's agent_<steps>.pt)
TEN_KEYS, SPECIAL_KEYS, PEND_KEYS = {"k", "n", "clips"}, {"tag", "ckpt", "clips"}, {"iter", "ckpt", "status"}
PICK_KEYS = {"iter", "kind", "ckpt", "agg_pct", "pass10"}  # best (+ among) · last
#: the record's measured keys (module doc) — finalize no longer copies them; an old-shape FINALIZED.json does
EXTRAS_KEYS = ("ten_pct", "special", "special_pending", "best", "last", "judge", "arm_pen", "finals")
#: Keys the judge record carries for display only (judge/metrics.py judge_record: park_excl_a = PARK gate A's excluded
#: rollouts per final judge; the scores still count them). Accepted by judge_extras, carried nowhere; any other key
#: still dies.
DISPLAY_KEYS = ("park_excl_a",)
FINAL_KEYS = {"iter", "agg_pct", "pass10"}


class Extras(NamedTuple):
    ten_pct: dict | None
    special: list | None
    special_pending: list | None
    sha256: str  # of the record file ('-' = no file)
    best: dict | None
    last: dict | None
    judge: dict | None = None  # {ne, epc} — None = a record from before the judge wrote its setting
    arm_pen: dict | None = None  # arm penetration (auxiliary) — None = not measured
    finals: list | None = None  # each final judge {iter, agg_pct, pass10} — None = a record from before


def _counts_ok(x: dict) -> bool:
    """A special entry's optional counts: exactly its clips, each [complete, total] with 0 ≤ complete ≤ total > 0."""
    c = x.get("counts", {k: [0, 1] for k in x["clips"]})
    return (
        isinstance(c, dict)
        and set(c) == set(x["clips"])
        and all(
            isinstance(v, list) and len(v) == 2 and all(type(i) is int for i in v) and 0 <= v[0] <= v[1] > 0
            for v in c.values()
        )
    )


def extras_check(t, s, p, where) -> tuple[dict | None, list | None]:
    """(ten_pct, special) after the shape check (module doc); p = special_pending. None = not measured."""
    if t is not None:
        c = t.get("clips") if isinstance(t, dict) else None
        ok = (
            isinstance(c, dict)
            and c
            and TEN_KEYS <= set(t) <= TEN_KEYS | {"evals"}
            and all(
                isinstance(v, list)
                and len(v) == 3
                and all(isinstance(x, int) for x in v)
                and 0 <= v[0] <= v[1]
                and v[1] > 0
                and v[2] in (0, 1)
                for v in c.values()
            )
        )
        if not ok or t["k"] != sum(v[2] for v in c.values()) or t["n"] != len(c):
            raise ValueError(
                f"{where} ten_pct has the wrong shape ([complete, total, pass] · k = Σ pass · n = clips): {str(t)[:90]}"
            )
    if s is not None and not (
        isinstance(s, list)
        and all(
            isinstance(x, dict)
            and SPECIAL_KEYS <= set(x) <= SPECIAL_KEYS | {"counts"}
            and isinstance(x["tag"], str)
            and x["tag"]
            and isinstance(x["ckpt"], str)
            and CKPT_NAME.fullmatch(x["ckpt"])
            and isinstance(x["clips"], list)
            and x["clips"]
            and len(set(x["clips"])) == len(x["clips"])
            and _counts_ok(x)
            for x in s
        )
    ):
        raise ValueError(f"{where} special has the wrong shape: {str(s)[:120]}")
    if p is not None and not (
        isinstance(p, list)
        and all(isinstance(x, dict) and set(x) == PEND_KEYS for x in p)
        and all(type(x["iter"]) is int and (x["ckpt"] is None or CKPT_NAME.fullmatch(str(x["ckpt"]))) for x in p)
    ):
        raise ValueError(f"{where} special_pending has the wrong shape: {str(p)[:120]}")
    return t, s


def _pick_ok(x, among: bool) -> bool:
    """One best/last entry (module doc): the keys · an iteration ≥ 0 · kind q|f · a checkpoint name · agg_pct in
    [0, 100] · pass10 [k, n] with 0 ≤ k ≤ n > 0 · best's among = distinct ascending iterations holding its own."""
    if not (isinstance(x, dict) and set(x) == PICK_KEYS | ({"among"} if among else set())):
        return False
    p, a = x["pass10"], x.get("among", [x["iter"]])
    return (
        type(x["iter"]) is int
        and x["iter"] >= 0
        and x["kind"] in ("q", "f")
        and isinstance(x["ckpt"], str)
        and bool(CKPT_NAME.fullmatch(x["ckpt"]))
        and type(x["agg_pct"]) in (int, float)
        and 0 <= x["agg_pct"] <= 100
        and isinstance(p, list)
        and len(p) == 2
        and all(type(i) is int for i in p)
        and 0 <= p[0] <= p[1] > 0
        and isinstance(a, list)
        and all(type(i) is int for i in a)
        and a == sorted(set(a))
        and x["iter"] in a
    )


def best_last_check(b, la, where) -> tuple[dict | None, dict | None]:
    """(best, last) after the shape check: both or neither (None = not measured) · last is a final (kind f) at the
    highest judged iteration best was picked from · best's value ≥ last's (last is one of the candidates)."""
    if b is None and la is None:
        return None, None
    if not (_pick_ok(b, True) and _pick_ok(la, False)):
        raise ValueError(f"{where} best/last have the wrong shape: best {str(b)[:100]} · last {str(la)[:100]}")
    if la["kind"] != "f" or la["iter"] != b["among"][-1] or b["agg_pct"] < la["agg_pct"]:
        raise ValueError(
            f"{where} best/last disagree (last = a final judge · its iteration = the last candidate · "
            f"best ≥ last): best {b['iter']}:{b['agg_pct']} candidates {b['among']} · last "
            f"{la['kind']}:{la['iter']}:{la['agg_pct']}"
        )
    return b, la


def judge_check(j, where) -> dict | None:
    """The judging setting {ne, epc} (positive ints) or None (not recorded)."""
    if j is not None and not (
        isinstance(j, dict) and set(j) == {"ne", "epc"} and all(type(v) is int and v > 0 for v in j.values())
    ):
        raise ValueError(f"{where} judge has the wrong shape ({{ne, epc}} positive ints): {str(j)[:80]}")
    return j


def arm_pen_check(a, where) -> dict | None:
    """{final iter: {kind: {max_mm, frac}}} — every final the same kinds (named by their owner, not here) · None = not
    measured."""

    def one(v):
        return (
            isinstance(v, dict)
            and set(v) == {"max_mm", "frac"}
            and all(type(x) in (int, float) for x in v.values())
            and v["max_mm"] >= 0
            and 0 <= v["frac"] <= 1
        )

    ok = isinstance(a, dict) and all(isinstance(v, dict) for v in a.values())
    kinds = {frozenset(v) for v in a.values()} if ok else None
    if a is not None and not (
        a
        and kinds
        and len(kinds) == 1
        and all(k.isdigit() for k in a)
        and all(one(x) for v in a.values() for x in v.values())
    ):
        raise ValueError(f"{where} arm_pen has the wrong shape ({{iter: {{kind: {{max_mm, frac}}}}}}): {str(a)[:100]}")
    return a


def finals_check(fs, la, where) -> list | None:
    """finals (module doc): a non-empty ascending list of {iter, agg_pct ∈ [0, 100], pass10 [k, n] 0 ≤ k ≤ n > 0} whose
    last iteration is `last`'s (la) with the same AGG — None = not recorded."""
    if fs is None:
        return None
    ok = (
        isinstance(fs, list)
        and fs
        and all(
            isinstance(x, dict)
            and set(x) == FINAL_KEYS
            and type(x["iter"]) is int
            and type(x["agg_pct"]) in (int, float)
            and 0 <= x["agg_pct"] <= 100
            and isinstance(x["pass10"], list)
            and len(x["pass10"]) == 2
            and all(type(i) is int for i in x["pass10"])
            and 0 <= x["pass10"][0] <= x["pass10"][1] > 0
            for x in fs
        )
    )
    its = [x["iter"] for x in fs] if ok else []
    if not (
        ok
        and its == sorted(set(its))
        and (la is None or (its[-1] == la["iter"] and fs[-1]["agg_pct"] == la["agg_pct"]))
    ):
        raise ValueError(
            f"{where} finals has the wrong shape ([{{iter, agg_pct, pass10}}] ascending · last = last): {str(fs)[:120]}"
        )
    return fs


#: The FINALIZED.json key naming the judge record finalize kept checkpoints by (its sha256 · '-' = no record).
FIN_SHA_KEY = "extras_sha256"


def finalized_claim_check(fin: dict | None, sha: str, where: str) -> None:
    """A finalized cell's judge record (sha = Extras.sha256 of judge_extras · '-' = no file) against what its
    FINALIZED.json states (measured against stated; never a fallback to a copy): a FINALIZED.json stating
    `extras_sha256` must name this very file ('-' = none then, none now); an older one (copied keys, no sha) that
    carries any of EXTRAS_KEYS says a record existed, so the file must be there. Their values are not read.
    fin None (not finalized) passes."""
    if fin is None:
        return
    said = fin.get(FIN_SHA_KEY)
    if said is not None and said != sha:
        raise ValueError(
            f"{where}: the judge record differs from the one at finalize (FINALIZED {FIN_SHA_KEY} "
            f"{str(said)[:12]} ≠ now {sha[:12]}) — not the record the kept checkpoints were chosen by; "
            "a copy does not replace it"
        )
    if said is None and sha == "-" and any(k in fin for k in EXTRAS_KEYS):
        raise ValueError(
            f"{where}: an older FINALIZED.json carries judge record keys but the judge record file is "
            "missing — a copy does not replace it"
        )


def judge_extras(run: str, sdir: Path, clips: list[str] | None = None) -> Extras:
    """The cell's judge record (cell_state.judge_extras_file; the one reader; no file = not measured; an unknown key
    dies). With the run's `clips`: ten_pct judged exactly those clips and every special clip is one of them, compared
    as clip names (paths.clip_name)."""
    f = CH.judge_extras_file(run, sdir)
    if not f.is_file():
        return Extras(None, None, None, "-", None, None, None, None, None)
    d = json.loads(raw := f.read_bytes())
    known = set(EXTRAS_KEYS) | set(DISPLAY_KEYS)
    if not isinstance(d, dict) or not set(d) <= known:
        raise ValueError(f"{f}: unknown judge record keys {sorted(set(d) - known) if isinstance(d, dict) else d}")
    t, s = extras_check(*map(d.get, EXTRAS_KEYS[:3]), f.name)
    b, la = best_last_check(d.get("best"), d.get("last"), f.name)
    if clips is not None:  # one namespace: the judge may name trees — paths.clip_name
        want = sorted(map(P.clip_name, clips))
        if t is not None and sorted(map(P.clip_name, t["clips"])) != want:
            raise ValueError(f"{f.name}: ten_pct clips {len(t['clips'])} ≠ this run's clips {len(want)}")
        stray = sorted({P.clip_name(c) for x in s or [] for c in x["clips"]} - set(want))
        if stray:
            raise ValueError(f"{f.name}: special clips outside this run {stray[:3]}")
    return Extras(
        t,
        s,
        d.get("special_pending"),
        hashlib.sha256(raw).hexdigest(),
        b,
        la,
        judge_check(d.get("judge"), f.name),
        arm_pen_check(d.get("arm_pen"), f.name),
        finals_check(d.get("finals"), la, f.name),
    )


def pick_label(p: dict | None) -> str:
    """A best/last entry in a table cell: '<iter>:<AGG>' ('-' = not measured)."""
    return "-" if p is None else f"{p['iter']}:{p['agg_pct']:.2f}"


def judge_label(j: dict | None) -> str:
    """The judging setting in a table cell: 'NE<ne>x<epc>' ('-' = not recorded)."""
    return "-" if j is None else f"NE{j['ne']}x{j['epc']}"


def judged_finals(run: str, sdir: Path) -> set[str]:
    """The judged finals (f rows OK) of a done cell; bundle_state decides done (judge DONE with this launch's id ·
    train.rc 0 · training process gone). Which iterations are the finals is what the sidecar's own JUDGE row states
    (its iter column: the finals it averaged, judge/metrics.py final_iters at its launch); every one of them must
    have an OK f row. A table judged with three finals keeps its three."""
    st = CH.bundle_state(sdir, run)
    if st != "done":
        raise ValueError(f"{run}: state {st} — checkpoints are neither deleted nor moved before the run is done")
    ok, fins = {}, None
    with open(CH.judge_files(run, sdir)[1], encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 8 and c[1] == "f" and c[8] == "OK":
                ok[c[2]] = c[3]
            elif len(c) > 2 and c[1] == "JUDGE":
                fins = [i for i in c[2].split(",") if i]
    if not fins or set(fins) - set(ok):
        raise ValueError(
            f"{run}: final judges {fins} of the JUDGE row without an OK f row "
            f"{sorted(set(fins or []) - set(ok))} (OK f rows {sorted(ok)})"
        )
    return {ok[i] for i in fins}


def judged_keep(run: str, sdir: Path) -> tuple[list[str], set[str]]:
    """(the finals in iteration order, the checkpoint names a done run keeps) — judged_finals ∪ kept_ckpts of its
    judge record: what finalize keeps in the cell and what a cluster's copy of the cell keeps after the volume prune
    (the one keep rule)."""
    fin = judged_finals(run, sdir)
    return sorted(fin, key=lambda n: int(CKPT_NAME.fullmatch(n).group(1))), fin | kept_ckpts(judge_extras(run, sdir))


BEST_AGENT = "best_agent.pt"  # the trainer's best pick (skrl · in-training eval) — kept only when that eval ran
INTRAIN_TAG = "[INTRAIN_EVAL]"  # judge/launch.py check_launch's line; its first word: off · (⚠) on
INTRAIN_WORDS = {"off": False, "on": True}


def intrain_eval(run: str, ref: Path) -> bool:
    """Whether the run trained with in-training eval on, by the training processes' own word: every `[INTRAIN_EVAL]`
    line (judge/launch.py check_launch writes one per training process: off · ⚠ on) of <RUN>_train*.log and the
    auto-resume <RUN>_R<iter>_train.log. No line, another word, or two words disagreeing dies (finalize must know
    whether the run's best_agent.pt means anything)."""
    logs = sorted(ref.glob(f"{run}_train*.log")) + sorted(ref.glob(f"{run}_R[0-9]*_train.log"))
    words = set()
    for f in logs:
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if INTRAIN_TAG in ln:
                w = [t for t in ln.split(INTRAIN_TAG, 1)[1].split() if t != "⚠"][:1]
                words.add(w[0] if w else "")
    if not words or not words <= set(INTRAIN_WORDS) or len(words) != 1:
        raise ValueError(
            f"{run}: the training logs' {INTRAIN_TAG} words {sorted(words)} — exactly one of "
            f"{list(INTRAIN_WORDS)} decides whether best_agent.pt is kept"
        )
    return INTRAIN_WORDS[words.pop()]


def kept_ckpts(x: Extras) -> set[str]:
    """The checkpoint names the judge record asks finalize to keep: special · pending (with a name) · best · last."""
    return {e["ckpt"] for e in (x.special or []) + (x.special_pending or []) + [x.best, x.last] if e and e["ckpt"]}
