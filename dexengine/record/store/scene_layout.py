#!/usr/bin/env python3
"""Scene layout sidecar `meta/scene_layout.jsonl` of every final dataset: each episode's scene rebuilt from its
metadata only (desk pose and height, torso joints and height, robot base; one frame, no relative fields).

Per-line oracles (a failing line is missing, never filled): desk_stretch top, torso dz, support-disc stamps vs render
centres, and the ego left eye of the rebuilt body vs the recorded camera_pose (CAM_ORACLE_TOL_M). Augmented variant
lines read their variant tree from the piece archive (collect_job_archive_roots('aug')); other lines read the original
tree. Unknown methods, scenes, sides or another round's episodes are errors.
sync = compute missing / stale lines (key episode_index, clip, root_z_offset_m, provisional mark) and rewrite the
sidecar and the provisional statement under piece.final_lock, on the machine that holds the finals; check = every
episode has a fresh line. Marker [SCENE_LAYOUT].
Usage: python -m dexengine.record.store.scene_layout sync|check --arm rby1 --hands <hand>... [--datasets ...] [--dry]
       [--all] [--dump <jsonl>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from dexengine.core import config
from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.record.store.scene_episode import MARK, _tree_at, _variant_candidates, aug_variant, die, episode_line

SIDECAR = "scene_layout.jsonl"  # under meta/
DISC_ON_Z = -5.0  # a support disc stands when its top > -5 (object_pin.support_disc_rows)


# ---- final datasets -------------------------------------------------------------------------------------------
def _eps(f: Path) -> dict:
    return {
        e["episode_index"]: e
        for e in map(json.loads, filter(str.strip, (f / "meta/episodes.jsonl").read_text().splitlines()))
    }


def _side(f: Path) -> dict:
    p = f / "meta" / SIDECAR
    return (
        {r["episode_index"]: r for r in map(json.loads, filter(str.strip, p.read_text().splitlines()))}
        if p.is_file()
        else {}
    )


def _key(i, clip, dz) -> list:
    return [i, clip, float(dz)]


def _pmark(prov: dict | None):
    return (prov or {}).get("mark")


def _fresh(i: int, e: dict, s: dict | None, prov: dict | None) -> bool:
    return (
        s is not None
        and s.get("provisional") == _pmark(prov)
        and _key(i, s["clip"], s["root_z_offset_m"]) == _key(i, e["clip"], e["body"]["root_z_offset_m"])
    )


def _mark_file(f: Path):
    p = f / "meta" / P.data_layout.PROVISIONAL_FILE
    return json.loads(p.read_text()) if p.is_file() else None


@lru_cache(maxsize=16)
def prov_of(hand: str) -> dict | None:
    """Provisional-hand statement (data_layout.hand_provisional), None for a canonical hand. Lines carry its mark in
    `provisional`; the final's meta/<PROVISIONAL_FILE> holds the statement itself."""
    DL = P.data_layout
    root = P.main_root()
    own = root / DL.render_run("run_root", hand) / DL.HAND_OWNER_FILE
    if not own.is_file():  # without the hand root the provisional state is unknown
        die(
            input_missing(
                f"hand {hand}: no hand-root marker {own} — cannot tell whether the hand is provisional (run "
                "from the main checkout)"
            )
        )
    d = DL.hand_provisional(hand, str(root))
    if d is None:
        return None
    return {
        "hand": d["hand"],
        "mark": d["mark"],
        "from": d["from"],
        "expiry": d["expiry"],
        "declared": DL.render_run("run_hand_carry", hand),
    }


def finals(arm: str, hands: list[str], datasets: list[str]) -> list[tuple]:
    """[(ds, combo, hand, round, final folder)] for every collect round x hand x dataset (orig finals: the collect
    layer of the data root; other rounds: the round root)."""
    DL = P.data_layout
    out = []
    for rnd, root in sorted(DL.collect_round_roots().items()):
        for h in hands:
            combo = P.combo_id(arm, h)
            for ds in datasets:
                f = (
                    Path(root) / DL.render("collect", ds, combo=combo)
                    if root
                    else Path(P.layer_path("collect", ds, combo=combo))
                )
                out.append((ds, combo, h, rnd, f))
    return out


def extract(f: Path, rnd: str, every: bool, prov: dict | None) -> dict:
    """Read one final: counts and the records of missing / stale lines (all lines with every=True)."""
    if not (f / "meta/episodes.jsonl").is_file():
        return {"absent": True}
    E, S = _eps(f), _side(f)
    todo, have, wrong = [], 0, 0
    for i, e in sorted(E.items()):
        b, fresh = e["body"], _fresh(i, e, S.get(i), prov)
        have += fresh
        wrong += e.get("round") != rnd
        if fresh and not every:
            continue
        sv = (e.get("render_look") or {}).get("supports") or {}
        todo.append(
            {
                "i": i,
                "clip": e["clip"],
                "side": e["side"],
                "scene": (e.get("framing") or {}).get("scene"),
                "body": b,
                "n_on": sum(1 for v in (e.get("supports") or {}).values() if v[2] > DISC_ON_Z),
                "discs": [{"slot": d["slot"], "center": d["center"]} for d in sv.get("discs") or []],
                "head_joints": e.get("head_joints"),
                "cam_left": ((e.get("camera_pose") or {}).get("ego_left") or {}).get("pos"),
                "object_present": e.get("object_present"),
                "round": e.get("round"),
                "piece_id": e.get("piece_id"),
            }
        )
    return {
        "n": len(E),
        "have": have,
        "extra": len(S) - have,
        "todo": todo,
        "wrong_round": wrong,
        "prov_ok": _mark_file(f) == prov,
    }


def write_sidecar(f: Path, new: list[dict], prov: dict | None) -> dict:
    """Merge new lines into the sidecar and rewrite it, and the provisional statement file, under piece.final_lock
    (keeps only fresh lines of the current provisional mark; a canonical hand removes the statement file)."""
    from dexengine.record import piece as PC

    with PC.final_lock(f):
        E, S = _eps(f), _side(f)
        S.update({r["episode_index"]: r for r in new})
        ok = {i: r for i, r in S.items() if i in E and _fresh(i, E[i], r, prov)}
        body = "".join(json.dumps(ok[i], ensure_ascii=False) + "\n" for i in sorted(ok))
        mark = json.dumps(prov, ensure_ascii=False, indent=1) + "\n" if prov else None
        for p, text in ((f / "meta" / SIDECAR, body), (f / "meta" / P.data_layout.PROVISIONAL_FILE, mark)):
            if text is None:
                p.unlink(missing_ok=True)
                continue
            tmp = p.with_name(p.name + f".part.{os.getpid()}")
            tmp.write_text(text)
            os.replace(tmp, p)
    if _mark_file(f) != prov:
        die(f"{f}: provisional statement file differs from the statement after writing")
    return {"episodes": len(E), "lines": len(ok), "dropped": len(S) - len(ok)}


def variant_trees(arm: str, todo: list[dict], hand: str, ds: str) -> None:
    """Set rec['vtree'] = (variant tree data | None, path | reason) on every aug variant record."""
    roots = P.data_layout.collect_job_archive_roots("aug")
    found = dict.fromkeys(range(len(roots)), 0)
    n = 0
    for rec in todo:
        if not aug_variant(rec):
            continue
        n += 1
        cands = _variant_candidates(arm, hand, ds, rec)
        if cands is None:
            rec["vtree"] = (None, f"aug line has no piece_id ({rec.get('piece_id')!r}) — variant tree unknown")
            continue
        at = next((i for i, c in enumerate(cands) if (Path(c) / "task_info.json").is_file()), None)
        if at is None:
            rec["vtree"] = (None, f"no variant tree ({len(cands)} places tried: {cands[0]} ...)")
            continue
        found[at] += 1
        rec["vtree"] = (_tree_at(cands[at]), cands[at])
    if n:
        print(
            f"{MARK} variant trees found {sum(found.values())} / {n} · per root "
            + " · ".join(f"{roots[i]} {v}" for i, v in found.items()),
            flush=True,
        )


def sync(a) -> int:
    tot = {"finals": 0, "eps": 0, "lines": 0, "new": 0, "missing": 0}
    why: dict[str, int] = {}
    for ds, combo, hand, rnd, f in finals(a.arm, a.hands, a.datasets):
        prov = prov_of(hand)
        x = extract(f, rnd, a.all, prov)
        if x.get("absent"):
            continue
        if x["wrong_round"]:
            die(f"{ds}/{combo}: the {rnd} final has {x['wrong_round']} episodes of another round")
        variant_trees(a.arm, x["todo"], hand, ds)
        tot["finals"] += 1
        tot["eps"] += x["n"]
        lines, miss = [], 0
        for rec in x["todo"]:
            ln, w = episode_line(rec, a.arm, hand, ds)
            if ln is None:
                miss += 1
                k = f"{ds}: {w[:90]}"
                why[k] = why.get(k, 0) + 1
            else:
                lines.append(dict(ln, provisional=_pmark(prov)))
        tot["new"] += len(lines)
        tot["missing"] += miss
        if a.dump:
            with open(a.dump, "a", encoding="utf-8") as fh:
                fh.writelines(json.dumps(dict(ln, dataset=ds, combo=combo), ensure_ascii=False) + "\n" for ln in lines)
        if a.dry or not (lines or x["extra"] or not x["prov_ok"]):
            got = x["have"] + (len(lines) if a.dry and not a.all else 0)
        else:
            got = write_sidecar(f, lines, prov)["lines"]
        tot["lines"] += got
        print(
            f"{MARK} {ds}/{combo}[{rnd}] episodes {x['n']} · existing lines {x['have']} · new {len(lines)} · "
            f"missing {miss} · lines after write {got} · provisional {'mark ' + prov['mark'] if prov else 'no'}"
            f"{' (dry run, nothing written)' if a.dry else ''}",
            flush=True,
        )
    for k, n in sorted(why.items(), key=lambda kv: -kv[1])[:10]:
        print(f"{MARK} missing {n} × {k}", flush=True)
    print(
        f"{MARK} end finals {tot['finals']} · episodes {tot['eps']} · lines {tot['lines']} · new {tot['new']} · "
        f"missing {tot['missing']}",
        flush=True,
    )
    return 0 if tot["missing"] == 0 and tot["lines"] == tot["eps"] else 1


def check(a) -> int:
    bad = n = 0
    for ds, combo, hand, rnd, f in finals(a.arm, a.hands, a.datasets):
        x = extract(f, rnd, False, prov_of(hand))
        if x.get("absent"):
            continue
        n += 1
        ok = not x["todo"] and not x["extra"] and x["have"] == x["n"] and x["prov_ok"] and not x["wrong_round"]
        bad += not ok
        print(
            f"{MARK} {'✓' if ok else '✗'} {ds}/{combo}[{rnd}] episodes {x['n']} · fresh lines {x['have']} · "
            f"missing or stale {len(x['todo'])} · extra {x['extra']} · provisional file "
            f"{'ok' if x['prov_ok'] else 'mismatch'} · other round {x['wrong_round']}",
            flush=True,
        )
    print(f"{MARK} end check finals {n} · mismatched {bad}", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("sync", "check"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hands", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=sorted(P.data_layout.DATASETS))
    ap.add_argument("--dry", action="store_true", help="sync: compute only, write nothing")
    ap.add_argument("--all", action="store_true", help="sync: recompute fresh lines too")
    ap.add_argument("--dump", help="sync: also append the new lines to a local jsonl")
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    if unknown := sorted(set(a.datasets) - set(P.data_layout.DATASETS)):
        die(f"unknown datasets {unknown}")
    return sync(a) if a.cmd == "sync" else check(a)


if __name__ == "__main__":
    sys.exit(main())
