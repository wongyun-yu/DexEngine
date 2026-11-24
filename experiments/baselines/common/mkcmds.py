#!/usr/bin/env python3
"""Row queues of the Tab. 2 baselines, derived from the per-clip method command files.

The method commands have one owner each: ``common/gen_cmds.sh`` (SPIDER, CHORD, EgoEngine) and
``maniptrans/gen_cmds.sh``
(ManipTrans) write ``<workspace>/run/*_cmds.txt``. This module never re-types a method command: a queue row is the
clip's
line from that file, with only the variant's inputs and outputs swapped. Every swap is asserted to hit exactly the
expected
number of places, so a changed command line stops the build instead of producing a silently different row.

A queue is a TSV with one row per job: ``name · cap_s · clip · result file · check · cmd``. ``check`` is
``<file>|<regex>`` that must match after the run (``-`` = none). A row passes when its result file is non-empty, its log
carries a ``[SCORE]`` line, the check matches and no time cap was hit (``common/rowlib.sh``).

Per-clip queues (``--queue <kind> <out.tsv> [--first N] [--clips <file>]``, rows in the order of
``experiments/scoring/clips/E123_CLIPS_v1.tsv``):

    spider_a · spider_a_nocheck · spider_a_nocheck_m0p2   SPIDER on the start-state (a) scenes (``prep_start_a.py``);
                                                          ``nocheck`` = SPIDER's own start-state check off; ``m0p2`` =
                                                          every object at our fixed object mass (``prep_mass_0p2.py``)
    egoengine_a · egoengine_a_m0p2[_r1|_r2]               EgoEngine on the same scenes; r1/r2 = further runs in one
    tree
    maniptrans · maniptrans_m0p2                          ManipTrans residual + recorder + score per clip
    chord                                                 FlashCHORD (its own start scene and mass)

Clip-independent ManipTrans queues: ``--queue maniptrans_imitator <out>`` (one imitator per hand),
``--queue maniptrans_retarget_pool <out> <P> <per row> <a-b>`` (the imitator pool's retargets, P at once on one GPU) and
``--queue maniptrans_retarget_repair <out> <P> <tag> <row name>...`` (the same rows again; finished items are skipped).

Inputs: ``$DEXENGINE_E123_ROOT/run/*_cmds.txt``, the clip lists under ``experiments/scoring/clips``, the ManipTrans
checkout
(motion frames of the CHORD parquets for the retarget caps). Output: the queue TSV given.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from experiments import workspace
from experiments.baselines.common.method_cmds import (
    chord_cmd,
    egoengine_a_cmd,
    fail,
    line_for,
    maniptrans_clip_cmd,
    maniptrans_m0p2_cmd,
    mk_dirs,
    root,
    spider_a_cmd,
)

HERE = Path(__file__).resolve().parent
CLIPS = workspace.REPO / "experiments/scoring/clips/E123_CLIPS_v1.tsv"
RT_BATCH = workspace.REPO / "experiments/baselines/maniptrans/retarget_batch.sh"

IMIT_END = "^(MAX EPOCHS NUM!|EARLY STOP!)$"  # rl_games' own end lines (lib/rl/base.py); a cap hit reads TRIM instead
# Per-item retarget cap = max(the production line's cap, RT_S_PER_FRAME x the clip's motion frames): a pool row runs P
# retargets at once, so the solo cap is too short for long clips (slowest measured 7.66 s per frame; 12 = 1.5 x that).
RT_S_PER_FRAME = 12


# kind -> (row cap s, command builder, result file, check '<file>|<regex>' or '-'); paths relative to the workspace root
QUEUES = {
    "spider_a": (5400, spider_a_cmd, "results/spider/{c}.start_a.json", "logs/spider/{c}.start_a.log|^rc=0$"),
    # the recorder's guard line proves both retargets loaded (else it dies)
    "maniptrans": (
        13800,
        maniptrans_clip_cmd,
        "results/maniptrans/{c}.json",
        "logs/maniptrans/{c}.eval.log|^\\[MT_RETARGET\\] ResDexHand BiH loaded=2 missing=0",
    ),
    # the check = the recorder's env applied the mass (its [MT_MASS] line for the second object)
    "maniptrans_m0p2": (
        13800,
        maniptrans_m0p2_cmd,
        "results/maniptrans/{c}.m0p2.json",
        "logs/maniptrans/{c}.m0p2.eval.log|^\\[MT_MASS\\] applied lh mass [0-9.]+ -> 0\\.200000 kg",
    ),
    # the check line proves SPIDER saved sanity_check_seconds: 0.0
    "spider_a_nocheck": (
        5400,
        lambda c: spider_a_cmd(c, "off"),
        "results/spider/{c}.start_a_nocheck.json",
        "logs/spider/{c}.start_a_nocheck.log|^\\[SPIDER_STARTCHECK\\] off rc=0 applied sanity_check_seconds: 0\\.0$",
    ),
    "egoengine_a": (
        7200,
        egoengine_a_cmd,
        "results/egoengine/{c}.start_a.json",
        "logs/egoengine/{c}.start_a.log|^rc=0$",
    ),
    "spider_a_nocheck_m0p2": (
        5400,
        lambda c: spider_a_cmd(c, "off", mass="m0p2"),
        "results/spider/{c}.start_a_nocheck_m0p2.json",
        "logs/spider/{c}.start_a_nocheck_m0p2.log|^\\[SPIDER_STARTCHECK\\] off rc=0 applied "
        "sanity_check_seconds: 0\\.0$",
    ),
    "egoengine_a_m0p2": (
        7200,
        lambda c: egoengine_a_cmd(c, mass="m0p2"),
        "results/egoengine/{c}.start_a_m0p2.json",
        "logs/egoengine/{c}.start_a_m0p2.log|^rc=0$",
    ),
    "egoengine_a_m0p2_r2": (
        7200,
        lambda c: egoengine_a_cmd(c, mass="m0p2", run="r2"),
        "results/egoengine/{c}.start_a_m0p2_r2.json",
        "logs/egoengine/{c}.start_a_m0p2_r2.log|^rc=0$",
    ),
    "egoengine_a_m0p2_r1": (
        7200,
        lambda c: egoengine_a_cmd(c, mass="m0p2", run="r1"),
        "results/egoengine/{c}.start_a_m0p2_r1.json",
        "logs/egoengine/{c}.start_a_m0p2_r1.log|^rc=0$",
    ),
    # FlashCHORD: train <= 21,600 s, eval <= 3,600 s, score
    "chord": (18600, chord_cmd, "results/chord/{c}/score.json", "-"),
}


def imitator_line(side: str) -> str:
    st = [
        ln
        for ln in (workspace.root() / "run/maniptrans_stage1_imitator_cmds.txt").read_text().splitlines()
        if ln and not ln.startswith("#")
    ]
    hit = [ln for ln in st if f" side={side.upper()} " in ln]
    if len(st) != 2 or len(hit) != 1:
        fail(f"stage1 lines {len(st)} / side {side} {len(hit)}")
    return hit[0]


def imitator_pool() -> list:
    """The imitator's training pool = its own dataIndices (the stage-1 lines own it); both hands must name the same
    set.
    """
    pools = []
    for sd in ("rh", "lh"):
        m = re.findall(r" dataIndices=\[([^\]]*)\] ", imitator_line(sd))
        if len(m) != 1:
            fail(f"imitator {sd}: dataIndices {len(m)}")
        pools.append(m[0].split(","))
    if sorted(pools[0]) != sorted(pools[1]) or len(set(pools[0])) != len(pools[0]):
        fail("imitator pools differ between hands or repeat a clip")
    return sorted(pools[0])


def retarget_line() -> tuple:
    """(iter, timeout, python, code dir) of the production retarget line (run/maniptrans_cmds.txt)."""
    ln = line_for("maniptrans_cmds.txt", clips_v1()[0])
    cd = re.findall(r"cd (\S+/code/maniptrans) && export PATH=", ln)
    m = re.findall(
        r"timeout (\d+) (\S+python) main/dataset/mano2dexhand\.py --data_idx \S+ --dexhand sharpa_wave "
        r"--side (right|left) --headless --iter (\d+) ",
        ln,
    )
    if len(set(cd)) != 1 or sorted(x[2] for x in m) != ["left", "right"] or len({(x[0], x[1], x[3]) for x in m}) != 1:
        fail(f"retarget line form in run/maniptrans_cmds.txt (cd {len(set(cd))} · lines {len(m)})")
    to, py, _, it = m[0]
    return it, to, py, cd[0]


_FRAMES: dict = {}


def rt_frames() -> dict:
    """{clip: motion frames} of every CHORD TACO parquet with the MANO columns (data/chord_hf_all_taco, the ManipTrans
    loader's data_dir; files without MANO columns are never pool items)."""
    if not _FRAMES:
        _, _, py, _ = retarget_line()
        r = subprocess.run(
            [
                py,
                "-c",
                "import glob,os,sys,pyarrow.parquet as pq\n"
                "for f in sorted(glob.glob(sys.argv[1]+'/*.parquet')):\n"
                " n = pq.ParquetFile(f).schema_arrow.names\n"
                " if 'mano_right_trans' in n: print(os.path.basename(f)[:-8], len(pq.read_table(f, "
                "columns=['mano_right_trans']).column(0)[0]))",
                str(workspace.data("chord_hf_all_taco")),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        for ln in r.stdout.splitlines():
            c, n = ln.split()
            _FRAMES[c] = int(n)
        if r.returncode or not _FRAMES:
            fail(f"motion frames from data/chord_hf_all_taco: rc {r.returncode} {r.stderr[-300:]}")
    return _FRAMES


def item_cap(to: int, clip: str) -> int:
    fr = rt_frames()
    if clip not in fr:
        fail(f"no motion parquet for retarget item {clip}")
    return max(int(to), -(-RT_S_PER_FRAME * fr[clip] // 1))


def rt_row(name: str, P: int, items: list, ld: str, cap: int | None = None) -> tuple:
    """items = side:clip; each gets :<cap s> (item_cap); the row cap = the sum over its waves (P items at a time, in
    order) of the wave's largest item cap + 600 s."""
    it, to, py, code = retarget_line()
    lg = f"{ld}.log"
    caps = [item_cap(int(to), x.split(":")[1]) for x in items]
    items = [f"{x}:{c}" for x, c in zip(items, caps, strict=True)]
    cap = cap or sum(max(caps[k : k + P]) for k in range(0, len(caps), P)) + 600
    cmd = (
        f"{mk_dirs()}; mkdir -p {ld}; bash {RT_BATCH} {P} {ld} {it} {to} {py} {code} {' '.join(items)} > {lg} 2>&1; "
        f"echo \"rc=$? $(grep -a '^\\[MT_RTBATCH\\]' {lg} | tail -1)\" | sed 's/^/[SCORE] /'"
    )
    return (name, cap, name, "-", f"{lg}|^\\[MT_RTBATCH\\] P={P} n={len(items)} .* missing=0 ", cmd)


def pool_items() -> list:
    return [f"{sd}:{i}" for i in imitator_pool() for sd in ("right", "left")]


def write_retarget_pool(out: str, P: int, per: int, part: str) -> None:
    """The imitator pool's retargets (every pool clip x both hands, before the imitator). Rows of <per> items, each row
    =
    one GPU running P at once. part = 'a-b' row range (0-based, half-open); 'b-a' with b > a = the same rows last-first.
    Rows keep their global number b<NNN> and carry their size in the name, so two parts never name the same row."""
    B = root()
    items = pool_items()
    rows = [items[k : k + per] for k in range(0, len(items), per)]
    a, b = (int(x) for x in part.split("-"))
    lo, hi = min(a, b), max(a, b)
    if not (0 <= lo < hi <= len(rows)):
        fail(f"--part {part} of {len(rows)} rows")
    out_rows = []
    for k in range(lo, hi) if a < b else range(hi - 1, lo - 1, -1):
        r = rt_row(
            f"mt_retarget__p{per}b{k:03d}", P, rows[k], f"{B}/logs_short/maniptrans/retarget_pool/p{per}b{k:03d}"
        )
        if "\t" in r[5] or "\n" in r[5]:
            fail(f"tab/newline in {r[0]}")
        out_rows.append("\t".join(map(str, r)) + "\n")
    n_items = sum(len(rows[k]) for k in range(lo, hi))
    Path(out).write_text("".join(out_rows))
    print(
        f"[MKCMDS] QUEUE maniptrans_retarget_pool rows {part} ({'last-first' if a > b else 'in order'}) of {len(rows)} "
        f"({n_items} of {len(items)} items · P={P} · per={per}) → {out}"
    )


def write_retarget_repair(out: str, P: int, names: list, tag: str) -> None:
    """Named pool rows again under <name><tag>: the batch skips every item whose pkl exists, so only missing ones
    run.
    """
    B = root()
    items = pool_items()
    out_rows = []
    for nm in names:
        m = re.fullmatch(r"mt_retarget__p(\d+)b(\d{3})", nm)
        if not m:
            fail(f"repair row name {nm}")
        per, k = int(m[1]), int(m[2])
        r = rt_row(
            f"{nm}{tag}",
            P,
            items[k * per : (k + 1) * per],
            f"{B}/logs_short/maniptrans/retarget_pool/p{per}b{k:03d}{tag}",
        )
        out_rows.append("\t".join(map(str, r)) + "\n")
    Path(out).write_text("".join(out_rows))
    print(f"[MKCMDS] QUEUE maniptrans_retarget_repair {len(out_rows)} rows ({tag}) P={P} → {out}")


def imitator_rows() -> list:
    """One imitator row per hand at the author scale: converge or the 36 h cap."""
    B = root()
    return [
        (
            f"maniptrans_imitator_{s}",
            130500,
            f"imitator_{s}",
            "-",
            f"{B}/logs/maniptrans/imitator_{s}.log|{IMIT_END}",
            f"{mk_dirs()}; {imitator_line(s)}",
        )
        for s in ("rh", "lh")
    ]


def clips_v1() -> list:
    rows = [ln.split("\t") for ln in CLIPS.read_text().splitlines() if ln and not ln.startswith("#")]
    if rows[0][:2] != ["idx", "clip"] or len(rows) != 65:
        fail(f"{CLIPS.name} header/rows {rows[0][:2]} {len(rows)}")
    return [r[1] for r in rows[1:]]


def write_queue(kind: str, out: str, first: int = 64, subset: str | None = None) -> None:
    """first = the first N clips of the frozen list (its order); subset = a file of clip names (each must be in the
    list;
    rows keep the list's order)."""
    B = root()
    cap, mkcmd, res, chk = QUEUES[kind]
    if not 1 <= first <= 64:
        fail(f"--first {first}")
    sel = clips_v1()[:first]
    if subset is not None:
        with open(subset) as fh:
            want = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        bad = sorted(set(want) - set(clips_v1()))
        if bad or len(set(want)) != len(want):
            fail(f"--clips {subset}: not in the list {bad} or duplicates")
        sel = [c for c in sel if c in set(want)]
        first = len(sel)
    rows = []
    for c in sel:
        cmd = mkcmd(c)
        if "\t" in cmd or "\n" in cmd:
            fail(f"tab/newline in {kind} {c}")
        ck = "-" if chk == "-" else f"{B}/{chk.format(c=c)}"  # '-' stays '-': the worker greps no file then
        rows.append(f"{kind}__{c}\t{cap}\t{c}\t{B}/{res.format(c=c)}\t{ck}\t{cmd}\n")
    if len(rows) != first or len({r.split(chr(9))[0] for r in rows}) != first:
        fail(f"queue {kind}: {len(rows)} rows")
    Path(out).write_text("".join(rows))
    how = f"subset {subset}" if subset else f"first {first} of clips/E123_CLIPS_v1.tsv"
    print(f"[MKCMDS] QUEUE {kind} {first} rows ({how}) → {out}")


def main(argv: list | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] != ["--queue"] or len(args) < 3:
        sys.exit(__doc__)
    kind, out = args[1], args[2]
    if kind == "maniptrans_imitator" and len(args) == 3:
        rows = imitator_rows()
        Path(out).write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
        print(f"[MKCMDS] QUEUE {kind} {len(rows)} rows → {out}")
    elif kind == "maniptrans_retarget_repair" and len(args) >= 5:
        write_retarget_repair(out, int(args[3]), args[5:], args[4])
    elif kind == "maniptrans_retarget_pool":
        if len(args) != 6:
            fail("usage --queue maniptrans_retarget_pool <out.tsv> <P> <per row> <a-b>")
        write_retarget_pool(out, int(args[3]), int(args[4]), args[5])
    elif kind in QUEUES:
        ap = argparse.ArgumentParser(prog="mkcmds.py --queue <kind> <out.tsv>")
        ap.add_argument("--first", type=int, default=64)
        ap.add_argument("--clips", default=None)
        a = ap.parse_args(args[3:])
        write_queue(kind, out, a.first, a.clips)
    else:
        fail(
            "usage --queue "
            f"{{{'|'.join(QUEUES)}|maniptrans_imitator|maniptrans_retarget_pool|maniptrans_retarget_"
            "repair} <out.tsv> …"
        )


if __name__ == "__main__":
    main()
