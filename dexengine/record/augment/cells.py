#!/usr/bin/env python3
"""Judged cell bundle runner (judged-cell contract v2.1), one RL bundle:
judge mode D (region dump) -> cell choice (augment.cells_pick) -> judge mode L (list judgement). Place =
paths.run_path("run_aug_cells", hand, root=main_root, …) = the place the judging host's results are fetched to,
read by augment.stage judged_bundle
(the data root, also from a worktree).
Inputs: clips = driver_check.bundle_queue (the same function as the record driver) · judge input = the adopted rows
of the final ledger -> run.env (combo) of the source chunk (source_tag) · judge.log [ACTIVE_SIDES] mode= · judge form
= the judge line of scripts/stage2_ground.sh (canon_env.sh · dz_policy · src-layer · pool_root · f0_axis) +
--aug-cells-*. The judge mode checks the generation and the canonical dz.
Stage markers · end count equation bundle = region = list = judged clips + the augment.judged.load check ->
RUN_DONE.json · end line [AUG_CELLS_RUN] DONE. Finished stages are skipped only with the same fingerprint (a mismatch
dies; reuse refused). Re-selection margin · user holds (--hold-clips) · f0 shadow check = augment.cells_cond.
Unsupported (dies): chunk judge_extra · retry_budget · exh_resume (variant axis = judge aug_cells axis_combo).
Usage: cells.py --hand <hand> --campaign <campaign> --bundle <bundle> --owner <ledger> --owner-sha <sha> --gpu <n>
       [--verify-only (state only, no GPU)] [--min-rate] [--mem 64G] [--fov-records a.json,…] [--hold-clips t.tsv]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.placement.judge import aug_cells as JAC
from dexengine.placement.judge import input_record as JFP
from dexengine.record import bundle as C
from dexengine.record import sources as OWN
from dexengine.record.augment import cells_cond as ACC
from dexengine.record.augment import cells_pick as ACP
from dexengine.record.augment.cells_steps import ROOT, d_state, die, final_count, l_state, run_judge, run_pick, say
from dexengine.record.driver_check import adopted, bundle_queue
from dexengine.rl import acceptance as CO

#: judge process memory cap (capped.sh; going over kills that command only)
MEM_DEFAULT = "64G"
#: judge records in a chunk run.env this runner does not take yet; a value dies
UNSUPPORTED_ENV = ("judge_extra", "retry_budget", "exh_resume")
ACTIVE_RE = re.compile(r"^\[ACTIVE_SIDES\] mode=(\S+)", re.M)


# ── inputs
def bundle_clips(hand: str, owner: Path, campaign: str, bundle: str, min_rate: float) -> tuple[str, list[str]]:
    """The bundle's record target clips: the record driver's queue function (bundle_queue) as it is (only one bundle
    taken; its order is not used)."""
    order = sorted({r["dataset"] for r in CO.read_ledger(owner)})
    q = [
        b
        for b in bundle_queue(owner, order, min_rate, adopted(hand))
        if (b["campaign"], b["bundle"]) == (campaign, bundle)
    ]
    if len(q) != 1:
        die(
            f"ledger {owner} queue has {campaign} / {bundle} {len(q)} times (must be 1: canonical ∧ completion rate >= "
            f"{min_rate})"
        )
    clips = [P.clip_name(c) for c in q[0]["clips"]]
    if not clips or len(set(clips)) != len(clips):
        die(f"{bundle}: clips {len(clips)} · duplicates {len(clips) - len(set(clips))}")
    return q[0]["dataset"], clips


def adopted_ledgers(hand: str, dataset: str) -> list[tuple[Path, str, str]]:
    """The final ledgers adopted_rows reads -> [(file, emb, variant)]: the hand's ledger files (ledger_files under the
    data root) whose axis is `dataset`. The record job input statement carries exactly these (one owner)."""
    RR = JFP._rr()
    out = []
    for f in RR.ledger_files(hand=hand, root=P.main_root()):
        ds, emb, var = RR.ledger_axis_parts(RR.ledger_axis(f))
        if ds == dataset:
            out.append((Path(f), emb, var))
    return out


def adopted_rows(hand: str, dataset: str, clips: list[str]) -> dict:
    """Clip -> adopted row of the final ledger (+ axes _emb · _variant · _file): exactly one over all dataset ledgers
    (adopted_ledgers) ∧ ADOPTED."""
    RR = JFP._rr()
    from dexengine.placement import final_ledger as FL  # noqa: PLC0415

    got: dict = {c: [] for c in clips}
    for f, emb, var in adopted_ledgers(hand, dataset):
        for r in FL.read_final_rows(f):
            if r.get("clip") in got:
                got[r["clip"]].append(dict(r, _emb=emb, _variant=var, _axis=RR.ledger_axis(f), _file=f.name))
    bad = {
        c: [(r["_file"], r.get("status")) for r in v]
        for c, v in got.items()
        if len(v) != 1 or v[0].get("status") != "ADOPTED" or not v[0].get(JFP.COLUMN)
    }
    if bad:
        die(
            input_missing(
                f"adopted row not unique or without judge input fingerprint {dict(list(bad.items())[:3])}: "
                "canonical adopted clips only"
            )
        )
    return {c: v[0] for c, v in got.items()}


def read_run_env(path: Path) -> dict:
    """run.env -> {key: value} (first value; 'a=1 b=2' lines split)."""
    kv: dict = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        if " " in v and all("=" in t for t in v.split()[1:]):  # dataset=hocap emb=bimanual
            toks = [f"{k}={v.split()[0]}"] + v.split()[1:]
            for t in toks:
                kk, vv = t.split("=", 1)
                kv.setdefault(kk, vv)
            continue
        kv.setdefault(k, v)
    return kv


def chunk_record(hand: str, tag: str) -> dict:
    """Judge record of the source chunk (adopted row source_tag): combo · active sides · no unsupported records."""
    d = P.run_path("run_chunk", hand, root=P.main_root(), tag=tag)
    env_f, log_f = d / "run.env", d / "judge.log"
    if not env_f.is_file() or not log_f.is_file():
        die(input_missing(f"source chunk {d} has no run.env · judge.log: judge conditions are not invented"))
    kv = read_run_env(env_f)
    extra = {k: kv[k] for k in UNSUPPORTED_ENV if kv.get(k, "").strip()}
    if extra:
        die(f"source chunk {tag} run.env {extra}: judge records this runner does not take yet")
    act = sorted(set(ACTIVE_RE.findall(log_f.read_text(encoding="utf-8", errors="replace"))))
    if len(act) != 1:
        die(f"source chunk {tag} judge.log [ACTIVE_SIDES] mode= values {act}: must be one")
    if not kv.get("combo"):
        die(input_missing(f"source chunk {tag} run.env has no combo="))
    return {
        "tag": tag,
        "combo": kv["combo"],
        "active": act[0],
        "dataset": kv.get("dataset"),
        "emb": kv.get("emb"),
        "commit": kv.get("commit", ""),
        "reselect": ACC.reselect_of(d, kv.get("place_reselect", ""), tag),
    }


def conditions(hand: str, dataset: str, row: dict, ck: dict) -> dict:
    """Adopted row + source chunk -> judge conditions (values of the stage2 judge line; owner functions as they are)."""
    from dexengine.placement import search_policy as dz_policy

    RR = JFP._rr()
    arm, hand_c, var = P.split_combo(ck["combo"], P.ARM_IDS, P.HAND_IDS)
    emb = row["_emb"]
    # the axis check uses the ledger axis key (ledger_key): the profile default variant (arctic = _artic) comes back as
    #   '' from ledger_axis_parts
    if (hand_c, ck["dataset"], ck["emb"]) != (hand, dataset, emb) or RR.ledger_key(dataset, emb, var) != row["_axis"]:
        die(
            f"source chunk {ck['tag']} (hand {hand_c} · variant {var!r} · {ck['dataset']} · {ck['emb']}) != ledger "
            f"axis {row['_axis']} ({hand} · {dataset} · {emb})"
        )
    ax = RR.f0_axis(dataset, var)
    scene = ax["scene"]
    f0 = Path(ax["node"]) if Path(ax["node"]).is_absolute() else P.output_root() / ax["node"]
    return ACC.with_reselect(
        {
            "arm": arm,
            "hand": hand,
            "combo": ck["combo"],
            "dataset": dataset,
            "emb": emb,
            "variant": var,
            "scene": scene,
            "f0": str(f0 / ax["tail"]),
            "dz": dz_policy.dz_cells_for(dataset, scene),
            "src_layer": P.default_src_layer(dataset),
            "pool": str(P.pool_root()),
            "active": ck["active"],
        },
        ck,
        row,
    )


def remote_cmd(a) -> str:
    return (
        f"cd <code tree> && DEXENGINE_RL_QUEUE_ROOT=<queue root> python3 dexengine/record/augment/cells.py "
        f"--hand {a.hand} --campaign {a.campaign} --bundle {a.bundle} --owner <sent ledger copy> "
        f"--owner-sha {a.owner_sha} --gpu <n> --mem {a.mem}"
    )


def _parser():
    """Command line of the augment cell stages."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument(
        "--owner", required=True, type=Path, help="canonical owner ledger (path of record.sources canon --hand)"
    )
    ap.add_argument("--owner-sha", required=True, help="ledger sha256 measured at launch (record.sources canon)")
    ap.add_argument("--gpu", default="", help="judge GPU (required unless --verify-only; no default)")
    ap.add_argument("--min-rate", type=float, default=C.MIN_RATE)
    ap.add_argument("--mem", default=MEM_DEFAULT, help="judge process memory cap (capped.sh -m)")
    ap.add_argument("--verify-only", action="store_true", help="read the stage state only, without GPU")
    ap.add_argument("--fov-records", default="", help="list of off-screen render_clip.json")
    ap.add_argument("--hold-clips", default="", help="user hold table (augment.cells_cond.user_holds; default off)")
    return ap


def _run_stages(bundle, d_todo, a, dataset, clips, held, groups, live, rows, d_done, conds, st_l):
    """Stages D, pick and L of one bundle, then RUN_DONE.json and the remote command line."""
    bundle.mkdir(parents=True, exist_ok=True)
    ACC.drop_stale_cells(bundle, d_todo)
    t_all = time.time()
    prev_f = bundle / "RUN_DONE.json"
    prev = json.loads(prev_f.read_text()).get("stages", {}) if prev_f.is_file() else {}
    rec = {
        "campaign": a.campaign,
        "bundle": a.bundle,
        "dataset": dataset,
        "hand": a.hand,
        "clips": clips,
        "held": held,
        "judge_commit": JAC.head_commit(),
        "tree": str(ROOT),
        "host": socket.gethostname(),
        "gpu": a.gpu,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "stages": {},
    }
    if d_todo:
        rec["stages"]["D"] = [
            run_judge(json.loads(g), "D", bundle, [c for c in cs if c in d_todo], a.gpu, "", a.mem)
            for g, cs in groups.items()
            if any(c in d_todo for c in cs)
        ]
        _d_done2, d_todo2 = d_state(bundle, live, rows, dataset)
        if d_todo2:
            die(input_missing(f"no region dumps even after judge D {d_todo2[:3]}"))
    else:
        say(f"D skipped: {len(d_done)} region dumps with the same fingerprint")
        rec["stages"]["D"] = {"skipped": "same fingerprint", "before": prev.get("D")}  # data (RUN_DONE.json)
    if a.fov_records:
        rec["fov"] = ACP.put_fov(bundle, a.fov_records.split(","), clips, a.hand, dataset, conds)
    rec["stages"]["pick"] = run_pick(a.hand, a.campaign, a.bundle, dataset, live, bundle)
    if st_l == "absent":
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        rec["stages"]["L"] = [
            run_judge(json.loads(g), "L", bundle, cs, a.gpu, f"{stamp}g{i}", a.mem)
            for i, (g, cs) in enumerate(groups.items())
        ]
    else:
        say("L skipped: judge chunks equal the list and fingerprint")
        rec["stages"]["L"] = {"skipped": "same fingerprint", "before": prev.get("L")}  # data (RUN_DONE.json)
    l_state(bundle, live)
    cnt = final_count(bundle, a.hand, dataset, live, conds)
    rec.update(
        count=cnt,
        sec=round(time.time() - t_all, 1),
        finished=dt.datetime.now().isoformat(timespec="seconds"),
        remote_cmd=remote_cmd(a),
        conditions={c: conds[c] for c in clips},
    )
    tmp = bundle / ".RUN_DONE.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    tmp.replace(bundle / "RUN_DONE.json")
    say(
        f"DONE clips {cnt['clips']} · region {cnt['region']} · list {cnt['cells_files']} · judged {cnt['judged']} · "
        f"cells {cnt['cells']} · passed {cnt['passed']} · ikc {cnt['ikc']} · chunks {len(cnt['chunks'])} · "
        f"{rec['sec']:.0f} s · {bundle / 'RUN_DONE.json'}"
    )
    say(f"remote: {remote_cmd(a)}")
    return 0


def main(argv=None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    env_hand = os.environ.get("DEXENGINE_HAND", "")
    if env_hand and env_hand != a.hand:
        die(f"--hand {a.hand} != env DEXENGINE_HAND {env_hand}: one hand")
    os.environ["DEXENGINE_HAND"] = a.hand  # owner of the hand = --hand (also for rule reads in this process)
    if not a.verify_only and not a.gpu:
        die(input_missing("no --gpu: the judge GPU is an explicit input"))
    pkg = str(ROOT)  # the folder that holds this tree's dexengine package
    if P.main_root().resolve() != ROOT and pkg not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        die(
            input_missing(
                f"code tree {ROOT} != data root {P.main_root()} but PYTHONPATH lacks {pkg}: it would import "
                "another tree's package"
            )
        )
    OWN.verify_src(a.owner, a.owner_sha)
    dataset, clips = bundle_clips(a.hand, a.owner, a.campaign, a.bundle, a.min_rate)
    rows = adopted_rows(a.hand, dataset, clips)
    conds = {c: conditions(a.hand, dataset, rows[c], chunk_record(a.hand, rows[c]["source_tag"])) for c in clips}
    bundle = P.run_path("run_aug_cells", a.hand, root=P.main_root(), campaign=a.campaign, bundle=a.bundle)
    live, held = ACC.holds(a.hold_clips, list(clips), {}, bundle, a.verify_only)
    groups: dict = {}
    for c in live:
        groups.setdefault(json.dumps(conds[c], sort_keys=True), []).append(c)
    say(
        f"bundle {a.campaign} / {a.bundle} · {dataset} · clips {len(clips)} (holds {len(held)}) · condition "
        f"groups {len(groups)} · place {bundle} · tree {ROOT}"
    )
    for g, cs in groups.items():
        say(f"condition {g} <- {len(cs)} clips (source chunks {sorted({rows[c]['source_tag'] for c in cs})})")
    d_done, d_todo = d_state(bundle, live, rows, dataset)
    if not a.verify_only:  # f0 built again → judged again
        d_done, d_todo = ACC.f0_again(bundle, d_done, d_todo, conds, a.hand)
    st_l = l_state(bundle, live) if not d_todo else "absent"
    if d_todo and (bundle / "judged").is_dir() and any((bundle / "judged").iterdir()):
        die(
            f"reuse refused: region dumps are short (clips to do {d_todo[:3]}) but judge chunks exist: clear "
            f"{bundle}/judged · ikc and run again"
        )
    say(f"state D done {len(d_done)} · to do {len(d_todo)} · L {st_l}")
    if a.verify_only:
        say(f"VERIFY DONE D done {len(d_done)}/{len(live)} · held {len(held)} · L {st_l} · remote: {remote_cmd(a)}")
        return 0
    return _run_stages(bundle, d_todo, a, dataset, clips, held, groups, live, rows, d_done, conds, st_l)


if __name__ == "__main__":
    sys.exit(main())
