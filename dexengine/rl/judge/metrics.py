#!/usr/bin/env python3
"""Judge shape, the 10 % pass rule, final judges and best/last of the fresh-process PARK judge (standard library only;
the judge sidecar scripts/rl/judge_sidecar.sh calls the CLI below).

Judge shape: judge NE = ``JUDGE_NE`` (1024; None = the training NE) and evaluations requested per clip =
  ``MIN_EVALS_PER_CLIP``. The trainer raises the per-clip evaluations to max(request, ceil(NE / K)); the static
  assignment
  then runs ceil(epc / floor(NE / K)) real batches (3-4 for K 43-64, 1 for K <= 16). A judge row is valid when every
  clip has >= MIN_EVALS_PER_CLIP evaluations (read from the ``[PER_CLIP]`` line) and the per-clip sums equal the
  AGGREGATE (a parse check: both come from the same accumulator). The host gate's judge memory reservation
  (``RLJUDGE_RSS_GB``, scripts/rl/host_gate.sh) and the campaign planner's judge hours move with this shape.
10 % pass: clip c passes when complete_c >= max(1, floor(evals_c / 10 + 1/2)) (integer half-up: 64 -> 6, 68 -> 7,
  345 -> 35). A judge's value is passing clips / clips (judge table column pass10). Each judge leaves its per-clip
  [complete, evals] in ``<judge name>_clips.json`` next to the judge outputs, plus the PARK E-clip oracle line
  (``PARK_E_TAG``) as park_e and the arm-penetration line (``ARM_PEN_TAG``) as arm_pen when the log has them.
  The run value over the final judges: a clip passes when any final judge passes it. The stored [complete, evals] is
  the passing final judge with the highest completion rate (ties: later iteration), else the highest rate overall,
  so the stored values reproduce the pass.
Final judges: ``FINAL_FRACS`` of MI (7/8 and 1: MI 1600 -> 1400, 1600); ``final_iters``. The judge value is the mean of
  their AGGREGATE values. The judge record's finals = per final judge {iter, agg_pct, pass10 [k, n]}.
Quick judges during training: ``mid_iters`` (multiples of the interval below the first final judge) — a trend read;
  they are not candidates for best/last.
best / last: best = the final judge (f, OK) with the highest AGGREGATE (ties: later iteration), last = the last final
  judge; each {iter, kind, ckpt, agg_pct, pass10 [k, n]}, best also carries among = the compared iterations.
Judge record ``<RUN>_judge_extras.json``: {"ten_pct": {k, n, evals, clips {clip: [complete, evals, pass 0|1]}},
  "best": {...}, "last": {...}, "finals": [...], "judge": {ne, epc}[, "arm_pen": ..., "park_excl_a": ...]}, written
  atomically by the sidecar before its DONE rc line.

CLI (scripts/rl/judge_sidecar.sh):
  shape --train-ne NE --n-clips K      -> "judgeNE requestedEPC"
  finals --mi MI                       -> final judge iterations (MI must be a positive multiple of 8, else rc 2)
  mids --mi MI --quick-every Q         -> quick judge iterations (empty line when none)
  traj-name --run RUN --kind f --it N  -> trajectory record file name of a final judge (``TRAJ``)
  rule                                 -> one-line rule summary for the sidecar's start-up log
  row --log LOG --n-clips K [--standard] [--save-dir DIR]
                                       -> "k/K" and DIR/<judge name>_clips.json; --standard checks the judge shape
                                          (every clip >= the minimum, per-clip sums = AGGREGATE); rc 2 + reason
  extras --tsv TSV --dir DIR --run RUN --finals "i j" --judge-ne NE --judge-epc E --out FILE
                                       -> FILE = the judge record; prints "pass10=k/n best=<it>:<agg> last=<it>:<agg>"
  picks --extras FILE                  -> the values of the judge table's best and last rows
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

JUDGE_NE: int | None = 1024  # judge environments (None = the training NE)
MIN_EVALS_PER_CLIP = 64  # evaluations requested per clip = the minimum for a valid judge row
#: Final judges = these fractions of MI (MI 1600 -> 1400, 1600).
FINAL_FRACS: tuple[tuple[int, int], ...] = ((7, 8), (1, 1))
N_FINALS = len(FINAL_FRACS)
#: Per-clip marker: one JSON line printed by the judge's aggregate report (unlike the table it does not cut names).
TAG = "[PER_CLIP] "
_JUDGE_DIR = Path(__file__).resolve().parent


def _owner_str(file: str, name: str) -> str:
    """A string constant of a judge module in this package, read from its source (those modules import Isaac Lab)."""
    for n in ast.parse((_JUDGE_DIR / file).read_text(encoding="utf-8")).body:
        if (
            isinstance(n, ast.Assign)
            and [getattr(x, "id", "") for x in n.targets] == [name]
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        ):
            return n.value.value
    raise RuntimeError(f"{file}.{name}: no string constant")


PARK_E_TAG = _owner_str("park_e_oracle.py", "TAG")  # PARK E-clip oracle line (report)
ARM_PEN_TAG = _owner_str("arm_penetration.py", "TAG")  # arm-penetration line (auxiliary value)
_AGG = re.compile(r"^\[FROZEN_REPLAY\] DONE .*aggregate_completion_rate=([0-9.]+)", re.M)
LOG_SFX = "_frozenreplay_train.log"  # judge log name = <judge name><LOG_SFX> (written by the launcher)
CLIPS = "{run}_j{kind}{it:04d}_clips.json"  # per-clip record of one judge (judge name = the sidecar's alias)
#: Trajectory record of one final judge (dexengine/rl/eval/traj_record), next to the judge outputs. It has the judge
#: piece name form <RUN>_j<kind><iter>..., so moving, collecting and finishing a cell handle it with the judge pieces.
TRAJ = "{run}_j{kind}{it:04d}_traj.npz"


class JudgeMetricsError(RuntimeError):
    """A judge log or judge table breaks a rule; the sidecar records that row as INVALID.

    The messages are written verbatim into judge tables (`<RUN>_judge.tsv` reasons, `FAIL` rc lines) that
    dexengine.rl.judge.rejudge and the cell readers parse, so their text is kept as recorded."""


def judge_shape(train_ne: int, k: int) -> tuple[int, int]:
    """(judge NE, evaluations requested per clip); dies when K is outside (0, judge NE]."""
    ne = train_ne if JUDGE_NE is None else JUDGE_NE
    if not 0 < k <= ne:
        raise JudgeMetricsError(f"clip count {k} is outside (0, judge NE {ne}]")
    return ne, MIN_EVALS_PER_CLIP


def final_iters(mi: int) -> list[int]:
    """Final judge iterations (FINAL_FRACS of MI); dies unless MI is a positive multiple of 8."""
    if mi <= 0 or mi % 8:
        raise JudgeMetricsError(
            f"MI {mi} is not a positive multiple of 8 — the final judges (7/8 · 1 of MI) are not whole iterations"
        )
    return [mi * a // b for a, b in FINAL_FRACS]


def final_value(mi: int, aggs: dict[int, float]) -> tuple[float, list[int]]:
    """(judge value = mean AGGREGATE of the final judges, 2 decimals; final iterations) from aggs = {iter: AGGREGATE}
    of OK f rows. Dies when a final judge is missing; other iterations in aggs are ignored."""
    fin = final_iters(mi)
    miss = [i for i in fin if i not in aggs]
    if miss:
        raise JudgeMetricsError(f"final judges {miss} have no AGGREGATE (iterations present {sorted(aggs)})")
    return round(sum(aggs[i] for i in fin) / len(fin), 2), fin


def mid_iters(mi: int, quick_every: int) -> list[int]:
    """Quick judge iterations: multiples of `quick_every` below the first final judge (sidecar and campaign plan)."""
    if quick_every < 0:
        raise JudgeMetricsError(f"interval {quick_every} (not negative)")
    first = final_iters(mi)[0]
    return list(range(quick_every, first, quick_every)) if quick_every else []


def threshold(n: int) -> int:
    """10 % threshold: completions needed out of n evaluations, max(1, floor(n / 10 + 1/2)) in integers."""
    return max(1, (n + 5) // 10)


def passes(complete: int, total: int) -> bool:
    """True when `complete` of `total` evaluations meet the 10 % threshold."""
    return complete >= threshold(total)


def rule_text() -> str:
    """One-line summary of the judge rule (the sidecar prints it at start-up)."""
    ne = "training NE" if JUDGE_NE is None else str(JUDGE_NE)
    return (
        f"judge NE {ne} · per-clip evals >= {MIN_EVALS_PER_CLIP} · 10 % pass = complete >= floor(evals/10 + 1/2) · "
        f"final judges MI x {', '.join(f'{a}/{b}' for a, b in FINAL_FRACS)} · run value = OR over final judges"
    )


def parse_excl_a(text: str, where: str = "log") -> dict[str, tuple[int, int]]:
    """PARK gate A exclusions of the last per-clip marker: {clip: (real batches left out of the gate check, real
    batches)} ({} for logs without it). Display only: the clip's completions are counted as usual."""
    lines = [ln for ln in text.replace("\r", "\n").split("\n") if ln.startswith(TAG)]
    try:
        xa = json.loads(lines[-1][len(TAG) :]).get("excl_a", {}) if lines else {}
        out = {str(k): (int(v[0]), int(v[1])) for k, v in xa.items()}
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
        raise JudgeMetricsError(f"{where}: cannot read excl_a of the per-clip marker ({e})") from e
    if any(not 0 <= c <= t or t <= 0 for c, t in out.values()):
        raise JudgeMetricsError(f"{where}: excl_a value makes no sense {str(out)[:80]}")
    return out


def parse_per_clip(text: str, where: str = "log") -> dict[str, tuple[int, int]]:
    """The last per-clip marker of a judge log -> {clip name as printed: (complete, evals)}."""
    lines = [ln for ln in text.replace("\r", "\n").split("\n") if ln.startswith(TAG)]
    if not lines:
        raise JudgeMetricsError(f"{where}: per-clip marker {TAG.strip()} missing")
    try:
        d = json.loads(lines[-1][len(TAG) :])
        n, rows = d["n"], d["clips"]
    except (ValueError, KeyError, TypeError) as e:
        raise JudgeMetricsError(f"{where}: cannot read the per-clip marker ({e})") from e
    out: dict[str, tuple[int, int]] = {}
    for r in rows:
        if not (
            isinstance(r, list)
            and len(r) == 3
            and isinstance(r[0], str)
            and r[0]
            and all(isinstance(x, int) and not isinstance(x, bool) for x in r[1:])
        ):
            raise JudgeMetricsError(f"{where}: per-clip row has another shape {str(r)[:80]}")
        name, c, t = r
        if name in out:
            raise JudgeMetricsError(f"{where}: duplicate clip name {name!r}")
        if not 0 <= c <= t or t <= 0:
            raise JudgeMetricsError(f"{where}: clip {name} completions {c} / evaluations {t} make no sense")
        out[name] = (c, t)
    if len(out) != n:
        raise JudgeMetricsError(f"{where}: per-clip rows {len(out)} ≠ the marker's n {n}")
    return out


def check_row(text: str, k: int, standard: bool, where: str = "log") -> tuple[int, int, dict[str, tuple[int, int]]]:
    """(passing clips, clips, per clip). The clip count must equal K (the roster).
    standard: every clip has >= MIN_EVALS_PER_CLIP evaluations and the per-clip sums equal the AGGREGATE."""
    pc = parse_per_clip(text, where)
    bad = [] if len(pc) == k else [f"per-clip rows {len(pc)} ≠ roster {k}"]
    if standard:
        low = sorted((t, c) for c, (_x, t) in pc.items() if t < MIN_EVALS_PER_CLIP)
        if low:
            bad.append(f"evaluations per clip {low[0][0]} < {MIN_EVALS_PER_CLIP} ({len(low)} clips · e.g. {low[0][1]})")
        a = _AGG.findall(text)
        sc, st = sum(c for c, _t in pc.values()), sum(t for _c, t in pc.values())
        if not a:
            bad.append("AGGREGATE marker missing")
        elif abs(sc / st - float(a[-1])) > max(1.0 / st, 1e-4):
            bad.append(f"per-clip sum {sc}/{st} ≠ AGGREGATE {a[-1]}")
    if bad:
        raise JudgeMetricsError(f"{where}: " + " · ".join(bad))
    return sum(passes(c, t) for c, t in pc.values()), len(pc), pc


def parse_park_e(text: str, tag: str = PARK_E_TAG) -> dict | None:
    """The JSON of the last `tag` line (PARK E oracle, arm penetration) of a judge log, or None; dies when
    unreadable."""
    lines = [ln for ln in text.replace("\r", "\n").split("\n") if ln.startswith(tag)]
    try:
        return json.loads(lines[-1][len(tag) :]) if lines else None
    except ValueError as e:
        raise JudgeMetricsError(f"cannot read the {tag.strip()} line ({e})") from e


def save_clips(
    log: Path,
    out_dir: Path,
    pc: dict[str, tuple[int, int]],
    park_e: dict | None = None,
    arm_pen: dict | None = None,
    excl_a: dict | None = None,
) -> Path:
    """Write the per-clip [complete, evals] of judge log <judge name>{LOG_SFX} (+ park_e, arm_pen, park_excl_a —
    display)
    to out_dir/<judge name>_clips.json."""
    if not log.name.endswith(LOG_SFX):
        raise JudgeMetricsError(f"{log.name}: not a judge log name (suffix {LOG_SFX})")
    f = out_dir / f"{log.name[: -len(LOG_SFX)]}_clips.json"
    k = sum(passes(*v) for v in pc.values())
    write_json(
        f,
        {
            "n": len(pc),
            "pass": k,
            "clips": {c: list(v) for c, v in pc.items()},
            **{key: v for key, v in (("park_e", park_e), ("arm_pen", arm_pen)) if v is not None},
            **({"park_excl_a": {c: list(v) for c, v in excl_a.items()}} if excl_a else {}),
        },
    )
    return f


def load_clips(f: Path) -> dict[str, tuple[int, int]]:
    """{clip: (complete, evals)} of a <judge name>_clips.json file."""
    d = json.loads(f.read_text(encoding="utf-8"))
    return {c: (v[0], v[1]) for c, v in d["clips"].items()}


def _rows(tsv: Path) -> dict[tuple[str, int], dict]:
    """Judge table -> {(kind, iter): last row (ABORT rows skipped)}, read by its header."""
    lines = tsv.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    out: dict[tuple[str, int], dict] = {}
    for ln in lines[1:]:
        r = dict(zip(head, ln.split("\t"), strict=False))
        if r.get("kind") in ("q", "f") and r.get("status") != "ABORT" and r.get("iter", "").isdigit():
            out[(r["kind"], int(r["iter"]))] = r
    return out


def _pick(cands: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """[(iter, complete, evals)] -> the passing one with the highest rate (ties: later iter), else the highest rate."""
    ok = [x for x in cands if passes(x[1], x[2])] or cands
    return max(ok, key=lambda x: (x[1] * 10**9 // x[2], x[0]))


def _judged_one(r: dict, cdir: Path, run: str, kind: str, it: int) -> dict:
    """One OK judge table row -> {iter, kind, ckpt, agg_pct, pass10 [k, n]} (pass10 from that judge's clips file)."""
    try:
        agg = float(r["agg_pct"])
    except (KeyError, ValueError) as e:
        raise JudgeMetricsError(f"cannot read the AGGREGATE of judge {kind}:{it} ({e})") from e
    pc = load_clips(cdir / CLIPS.format(run=run, kind=kind, it=it))
    return {
        "iter": it,
        "kind": kind,
        "ckpt": r.get("ckpt", ""),
        "agg_pct": agg,
        "pass10": [sum(passes(*v) for v in pc.values()), len(pc)],
    }


def best_last(rows: dict[tuple[str, int], dict], cdir: Path, run: str, finals: list[int]) -> dict:
    """{"best", "last"}: best = the final judge (f, OK) with the highest AGGREGATE (ties: later iteration), last = the
    last final judge f:max(finals), which must be OK. Quick judges (q rows) are not candidates."""
    last_it = max(finals)
    if ("f", last_it) not in rows or rows[("f", last_it)].get("status") != "OK":
        raise JudgeMetricsError(f"the last final judge f:{last_it} is not OK")
    ok = sorted((it, "f") for it in finals if rows.get(("f", it), {}).get("status") == "OK")
    got = {(k, it): _judged_one(rows[(k, it)], cdir, run, k, it) for it, k in ok}
    b = max(got.values(), key=lambda x: (x["agg_pct"], x["iter"]))
    return {"best": dict(b, among=sorted({it for it, _k in ok})), "last": got[("f", last_it)]}


def picks_rows(extras: dict) -> list[str]:
    """Values of the judge table's best and last rows (the sidecar writes the rows): kind, iter, ckpt, AGG, note,
    k/n."""
    b, la = extras["best"], extras["last"]
    return [
        f"best\t{b['iter']}\t{b['ckpt']}\t{b['agg_pct']:.2f}\tbest = highest AGGREGATE of the "
        f"{len(b['among'])} measured judges ({','.join(map(str, b['among']))}) · {b['kind']} row\t"
        f"{b['pass10'][0]}/{b['pass10'][1]}",
        f"last\t{la['iter']}\t{la['ckpt']}\t{la['agg_pct']:.2f}\tlast = the last final judge\t"
        f"{la['pass10'][0]}/{la['pass10'][1]}",
    ]


def build_extras(
    tsv: Path, cdir: Path, run: str, finals: list[int], judge: tuple[int, int] | None = None
) -> tuple[dict, str]:
    """(judge record, one summary line). Every final judge must be OK. `judge` = the (judge NE, requested per-clip
    evaluations) the judges ran with (record key "judge")."""
    rows = _rows(tsv)
    per: dict[int, dict[str, tuple[int, int]]] = {}
    pen: dict[int, dict | None] = {}
    xa_f: dict[str, dict] = {}  # final judge -> PARK gate A exclusions {clip: [excluded batches, batches]} (display)
    for it in finals:
        if rows.get(("f", it), {}).get("status") != "OK":
            raise JudgeMetricsError(f"final judge f:{it} is not OK")
        per[it] = load_clips(cdir / CLIPS.format(run=run, kind="f", it=it))
        jd = json.loads((cdir / CLIPS.format(run=run, kind="f", it=it)).read_text(encoding="utf-8"))
        pen[it] = jd.get("arm_pen")
        if jd.get("park_excl_a"):
            xa_f[str(it)] = jd["park_excl_a"]
    names = set(per[finals[0]])
    if any(set(v) != names for v in per.values()):
        raise JudgeMetricsError("the final judges' clip sets differ")
    clips = {}
    for c in sorted(names):
        _it, cc, tt = _pick([(it, *per[it][c]) for it in finals])
        clips[c] = [cc, tt, int(any(passes(*per[it][c]) for it in finals))]
    ten = {
        "k": sum(v[2] for v in clips.values()),
        "n": len(clips),
        "evals": min(v[1] for v in clips.values()),
        "clips": clips,
    }
    fins = [
        {
            "iter": it,
            "agg_pct": float(rows[("f", it)]["agg_pct"]),
            "pass10": [sum(passes(*v) for v in per[it].values()), len(per[it])],
        }
        for it in finals
    ]
    out: dict = {
        "ten_pct": ten,
        **best_last(rows, cdir, run, finals),
        "finals": fins,
        **({} if judge is None else {"judge": {"ne": judge[0], "epc": judge[1]}}),
        **({"arm_pen": {str(it): v["pairs"] for it, v in pen.items()}} if all(pen.values()) else {}),
        **({"park_excl_a": xa_f} if xa_f else {}),
    }
    bl = " ".join(f"{k}={out[k]['iter']}:{out[k]['agg_pct']:.2f}" for k in ("best", "last"))
    return out, f"pass10={ten['k']}/{ten['n']} {bl}"


def write_json(path: Path, d: dict) -> None:
    """Write JSON atomically (temporary file + rename)."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _ints(s: str) -> list[int]:
    return [int(t) for t in s.split()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shape")
    p.add_argument("--train-ne", type=int, required=True)
    p.add_argument("--n-clips", type=int, required=True)
    sub.add_parser("finals").add_argument("--mi", type=int, required=True)
    md = sub.add_parser("mids")
    md.add_argument("--mi", type=int, required=True)
    md.add_argument("--quick-every", type=int, required=True)
    sub.add_parser("rule")
    w = sub.add_parser("row")
    w.add_argument("--log", required=True)
    w.add_argument("--n-clips", type=int, required=True)
    w.add_argument("--standard", action="store_true")
    w.add_argument("--save-dir", default=None)
    x = sub.add_parser("extras")
    for k in ("--tsv", "--dir", "--run", "--finals", "--out"):
        x.add_argument(k, required=True)
    x.add_argument("--judge-ne", type=int, required=True)
    x.add_argument("--judge-epc", type=int, required=True)
    sub.add_parser("picks").add_argument("--extras", required=True)
    t = sub.add_parser("traj-name")
    t.add_argument("--run", required=True)
    t.add_argument("--kind", choices=("f",), required=True)
    t.add_argument("--it", type=int, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "shape":
            print(*judge_shape(a.train_ne, a.n_clips))
        elif a.cmd == "finals":
            print(*final_iters(a.mi))
        elif a.cmd == "mids":
            print(*mid_iters(a.mi, a.quick_every))
        elif a.cmd == "rule":
            print(rule_text())
        elif a.cmd == "row":
            text = Path(a.log).read_text(errors="replace")
            k, n, pc = check_row(text, a.n_clips, a.standard, Path(a.log).name)
            if a.save_dir is not None:
                save_clips(
                    Path(a.log),
                    Path(a.save_dir),
                    pc,
                    parse_park_e(text),
                    parse_park_e(text, ARM_PEN_TAG),
                    parse_excl_a(text, Path(a.log).name),
                )
            print(f"{k}/{n}")
        elif a.cmd == "traj-name":
            print(TRAJ.format(run=a.run, kind=a.kind, it=a.it))
        elif a.cmd == "picks":
            print("\n".join(picks_rows(json.loads(Path(a.extras).read_text(encoding="utf-8")))))
        else:
            d, line = build_extras(Path(a.tsv), Path(a.dir), a.run, _ints(a.finals), (a.judge_ne, a.judge_epc))
            write_json(Path(a.out), d)
            print(line)
    except (JudgeMetricsError, OSError, ValueError, KeyError) as e:
        print(f"[JUDGE_METRICS] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
