"""Clip-owner ledger writer: which kept checkpoint owns each clip of one hand, over the bundles of its RL campaigns.

The ledger (`rl_clip_owner.tsv`, schema acceptance.SCHEMA, head acceptance.HEAD) is what recording takes: record
`drive`, record.bundle and the augment run read it with acceptance.read_ledger / use_rows, and the launcher measures
its sha256 (record.sources canon). This module writes it from finished campaign cells.

  bundles     every bundle of the named campaigns' frozen manifest copies (acceptance.manifest_of); a clip may sit in
              several bundles (a dup, or another campaign that trained it again).
  candidates  a judged bundle's kept checkpoints with a per-clip record: the judge
              table's OK final rows (kind f) and its special-tag checkpoints (the judge record's `special`, each the
              table's one OK mid row of that checkpoint). Per clip [complete, total] = that judge's clips file
              (judge.metrics.CLIPS). Oracles: every candidate is in FINALIZED.json `kept` · each clips file names
              exactly the bundle's clips. Judged = cell state done with the judge record (ten_pct) and FINALIZED.json.
  owner_of    per clip over the candidates of the judged bundles holding it: the highest completion rate
              complete/total (exact Fraction) · a tie → the later iteration · the bundle id first (RULE).
  bundles     the `bundles` / `waiting` columns list every bundle of the named campaigns that holds the clip / is not
              judged yet (bundle ids).
  states      canon (a judged bundle holds the clip — owner_of) · pending (none judged yet). That is all this writer
              knows: the states regen_wait · held · dropped and the generation columns (gen, old, turn,
              train_gen, check_gen) come from a plugin's turnover records, and this writer writes them as "-".
  replace     the file is written whole (temp + replace) under --owner-root (no default: the caller names where).
              An existing ledger holding a state or a generation column this writer does not produce is never
              replaced (exit 3) — such a ledger stays its own writer's.

Usage:
  python -m dexengine.rl.owner_ledger --hand <hand> --campaigns <c1>[,<c2>…] --owner-root DIR [--dry-run]
      [--data-root DIR]
  Marker: `[CLIP_OWNER] <hand> clips <n> · canon <k> · pending <p> · checkpoints <c> → <file>`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

from dexengine.core import config
from dexengine.core import paths as P
from dexengine.rl import acceptance as A
from dexengine.rl import cell_state as CH
from dexengine.rl import cells as CL
from dexengine.rl.judge import metrics as JM
from dexengine.rl.judge import record as CX

RULE = "highest complete/total among kept judged checkpoints · tie: later iteration · bundle id first"
#: The states this writer produces (acceptance.STATES keys).
WRITES = ("canon", "pending")
#: The columns a plugin's turnover records fill; this writer writes "-".
SITE_COLS = ("gen", "old", "turn", "static_add", "train_gen", "check_gen")


def die(msg: str, rc: int = 2):
    print(f"{A.TAG} ✗ {msg}", file=sys.stderr, flush=True)
    raise SystemExit(rc)


def candidates(b: dict, run: str | None, sdir: Path | None, state: str | None) -> list[dict] | None:
    """[{bundle, iter, kind f|q, tag, ckpt, kept, clips {clip: (complete, total)}}] of a judged bundle (module doc),
    None = not judged yet. A record that breaks an oracle dies."""
    if state != "done" or not run or run == "-" or sdir is None:
        return None
    x = CX.judge_extras(run, sdir)
    fin_f, tsv = CH.cell_file(sdir, "finalized"), CH.judge_files(run, sdir)[1]
    if x.ten_pct is None or not fin_f.is_file() or not tsv.is_file():
        return None
    fin = json.loads(fin_f.read_text(encoding="utf-8"))
    if not isinstance(fin.get("kept"), list):
        die(f"{b['id']}: FINALIZED.json has no kept list — no owner without the kept checkpoints")
    kept = {Path(p).name: p for p in fin["kept"]}
    rows = JM._rows(tsv)
    picks = [(k, it, None) for (k, it), r in sorted(rows.items()) if k == "f" and r.get("status") == "OK"]
    if not picks:
        die(f"{b['id']}: a judge record but no OK final row in the judge table")
    for e in x.special or []:
        hit = [
            (k, it)
            for (k, it), r in rows.items()
            if k == "q" and r.get("status") == "OK" and r.get("ckpt") == e["ckpt"]
        ]
        if len(hit) != 1:
            die(f"{b['id']}: special tag {e['tag']}: {len(hit)} OK mid rows for checkpoint {e['ckpt']} (one expected)")
        picks.append((*hit[0], e["tag"]))
    want = {P.clip_name(c) for c in b["clips"]}
    out, seen = [], set()
    for k, it, tag in picks:
        ck = rows[(k, it)].get("ckpt", "")
        if ck in seen:
            die(f"{b['id']}: checkpoint {ck} is a candidate twice")
        seen.add(ck)
        if ck not in kept:
            die(
                f"{b['id']}: candidate {k}{it} checkpoint {ck!r} is not in FINALIZED kept — a pruned checkpoint "
                "cannot own a clip"
            )
        f = sdir / JM.CLIPS.format(run=run, kind=k, it=it)
        if not f.is_file():
            die(f"{b['id']}: judge {k}{it} has no per-clip record {f.name}")
        raw = JM.load_clips(f)
        pc = {P.clip_name(c): v for c, v in raw.items()}
        if len(pc) != len(raw) or set(pc) != want:
            die(
                f"{b['id']}: {f.name} clips {len(raw)} ≠ bundle clips {len(want)} — outside "
                f"{sorted(set(pc) - want)[:3]} · missing {sorted(want - set(pc))[:3]}"
            )
        if bad := [c for c, (cc, tt) in pc.items() if not (type(cc) is int and type(tt) is int and 0 <= cc <= tt > 0)]:
            die(f"{b['id']}: {f.name} [complete, total] of the wrong form {bad[:3]}")
        out.append({"bundle": b["id"], "iter": it, "kind": k, "tag": tag, "ckpt": ck, "kept": kept[ck], "clips": pc})
    return out


def owner_of(cs: list[dict]) -> dict:
    """The owner among one clip's candidates [{bundle, iter, c, t, …}] — RULE. max keeps the first of equal keys, so
    the bundle-id order goes in before it."""
    return max(sorted(cs, key=lambda x: x["bundle"]), key=lambda x: (Fraction(x["c"], x["t"]), x["iter"]))


def _bl(x: dict | None) -> str:
    return "-" if not x else f"{x['iter']}:{x['ckpt']}"


def rows_of(hand: str, campaigns: list[str], root: Path) -> tuple[list[dict], dict]:
    """The ledger rows (acceptance.HEAD, one per unique (axis, clip), sorted) and counts over the campaigns' bundles."""
    per: dict[tuple[str, str], list[tuple]] = collections.defaultdict(list)
    res, judged, ds = {}, {}, {}
    for camp in campaigns:
        man = A.manifest_of(hand, camp, root)
        for b in man["bundles"]:
            b = dict(b, hand=hand)
            key = (camp, b["id"])
            if key in res:
                die(f"bundle {b['id']} twice in campaign {camp}")
            r = res[key] = A.bundle_result(hand, camp, b, root)
            judged[key] = candidates(b, r["run"], r["cell"], r["state"])
            for c in b["clips"]:
                k = (b["axis"], P.clip_name(c))
                per[k].append(key)
                ds.setdefault(k, b["dataset"])
    out = []
    for k, keys in sorted(per.items()):
        cs = [
            dict(x, c=x["clips"][k[1]][0], t=x["clips"][k[1]][1], camp=key[0])
            for key in sorted(keys)
            for x in judged[key] or []
        ]
        ids = ",".join(sorted({b for _c, b in keys}))  # bundle ids, comma-joined
        wait = ",".join(sorted({b for c, b in keys if not judged[(c, b)]})) or "-"
        r = dict(axis=k[0], clip=k[1], dataset=ds[k], bundles=ids, waiting=wait, **dict.fromkeys(SITE_COLS, "-"))
        if cs:
            w = owner_of(cs)
            ob = res[(w["camp"], w["bundle"])]
            r.update(
                state="canon",
                bundle=w["bundle"],
                iter=w["iter"],
                kind="final" if w["kind"] == "f" else "special",
                tag=w["tag"] or "-",
                ckpt=w["ckpt"],
                complete=w["c"],
                total=w["t"],
                rate_pct=f"{100 * w['c'] / w['t']:.2f}",
                campaign=w["camp"],
                run=ob["run"],
                ckpt_rel=os.path.relpath(Path(ob["cell"]) / w["kept"], root),
                best=_bl(ob["best"]),
                last=_bl(ob["last"]),
            )
        else:
            r.update(
                state="pending",
                **dict.fromkeys(
                    (
                        "bundle",
                        "iter",
                        "kind",
                        "tag",
                        "ckpt",
                        "complete",
                        "total",
                        "rate_pct",
                        "campaign",
                        "run",
                        "ckpt_rel",
                        "best",
                        "last",
                    ),
                    "-",
                ),
            )
        out.append(r)
    cnt = collections.Counter(r["state"] for r in out)
    ck = {(r["campaign"], r["bundle"], r["ckpt"]) for r in out if r["state"] == "canon"}
    return out, {"clips": len(out), "canon": cnt["canon"], "pending": cnt["pending"], "checkpoints": len(ck)}


def foreign(path: Path) -> list[str]:
    """Why an existing ledger is not this writer's to replace: a state it does not write or a plugin column filled."""
    rows = A.read_ledger(path)
    bad = sorted({r["state"] for r in rows} - set(WRITES))
    filled = sorted({c for r in rows for c in SITE_COLS if r[c] != "-"})
    return ([f"states {bad}"] if bad else []) + ([f"site columns filled {filled}"] if filled else [])


def write(hand: str, rows: list[dict], owner_root: Path) -> Path:
    """The ledger under owner_root (acceptance.ledger_path), whole file, temp + replace; read back and counted."""
    f = A.ledger_path(hand, owner_root)
    if f.is_file() and (why := foreign(f)):
        die(f"{f} holds {' · '.join(why)} — a plugin's ledger is not replaced by this writer", rc=3)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.tmp")
    tmp.write_text(
        "\t".join(A.HEAD) + "\n" + "".join("\t".join(str(r[k]) for k in A.HEAD) + "\n" for r in rows), encoding="utf-8"
    )
    os.replace(tmp, f)
    back = A.read_ledger(f)
    if [(r["axis"], r["clip"], r["state"]) for r in back] != [(r["axis"], r["clip"], r["state"]) for r in rows]:
        die(f"{f}: the ledger read back differs from the rows written")
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dexengine.rl.owner_ledger", description=__doc__.split("\n")[0], epilog="Rule: " + RULE
    )
    ap.add_argument("--hand", required=True)
    ap.add_argument("--campaigns", required=True, help="comma-separated campaign ids of the hand")
    ap.add_argument(
        "--owner-root",
        type=Path,
        required=True,
        help="where the ledger goes: <owner root>/records/hands/<hand>/tables/rl_clip_owner.tsv "
        "(readers find it with python -m dexengine.core.paths --what owner-root)",
    )
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    camps = [c for c in a.campaigns.split(",") if c]
    for c in camps:
        if not CL.campaign_dir(a.hand, c, P.main_root()).is_dir():
            die(f"no campaign {c} of hand {a.hand} under {P.main_root()}")
    rows, n = rows_of(a.hand, camps, P.main_root())
    dest = "dry run, nothing written" if a.dry_run else write(a.hand, rows, a.owner_root)
    print(
        f"{A.TAG} {a.hand} clips {n['clips']} · canon {n['canon']} · pending {n['pending']} · checkpoints "
        f"{n['checkpoints']} → {dest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
