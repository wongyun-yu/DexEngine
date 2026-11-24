#!/usr/bin/env python3
"""Identity record of the record work pieces (run layer run_collect_work; one bundle = one bundle --out) and the call
of the final-place merge. A part of the driver (record.driver); the rules are in its module doc 'output'. The merge
itself is owned by MERGE_MODULE (this file only calls it: pre-start gate merge_gate, dataset lock merge_step, result
field merge_into).

Rounds (record.episodes.ROUNDS: orig, aug): the bundle place (<piece>) is one for all rounds. They share the stage-1
record (stage1; the only source of the chosen trajectory: a stage-1 rerun is not a bit-exact reproduction and the text
describes that trajectory), the augment physics (aug/) and the clip render (stage2c) caches. Round outputs (dataset,
preview) go to the round place round_dir = <piece>/rounds/<round>: one piece = (bundle, round) · identity record =
piece.json of the round place (bundle identity + round) · merge key piece_id = <bundle place name>.<round> (the orig
and aug pieces of one bundle are different pieces, merged into the same final in turn).
Owner of the round place form = round_dir alone. Specs splitting a bundle into parts (nparts > 1) split the round
place per part (<round>.p<k>of<n>, part_key): sibling parts writing the same bundle place at once do not overwrite
each other's dataset, preview or cleanup records. The bundle place (stage1, stage2c caches) stays shared. nparts = 1
keeps the place name as it is."""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.record import episodes as EPI  # owner of the rounds (ROUNDS)
from dexengine.record.driver_check import die, say, write_atomic

#: identity of the work piece (hand, combo, dataset, campaign, bundle); the piece place (run_collect_work) has no
#: dataset
#: field. Writer piece_dir (bundle start; another identity dies).
PIECE_FILE = "piece.json"
ROUNDS_DIR = "rounds"  # parent of the round places (under the bundle place); form owner round_dir


PART_RE = re.compile(r"p([1-9][0-9]*)of([1-9][0-9]*)")  # form of the round place part tail (built by part_key)


def part_key(spec: dict) -> str:
    """Round place part tail of a spec: '' for nparts 1 (place as it is), else 'p<part>of<nparts>'. A missing field, a
    non-integer or a part outside 1 <= part <= nparts dies (no sibling place built from a spec that does not know its
    part)."""
    try:
        k, n = int(spec["part"]), int(spec["nparts"])
    except (KeyError, TypeError, ValueError):
        die(
            input_missing(
                f"spec {spec.get('spec_id', '?')}: part or nparts missing or not an integer: the round place "
                "cannot be built"
            )
        )
    if not 1 <= k <= n:
        die(f"spec {spec.get('spec_id', '?')}: part {k} · nparts {n}: outside 1 <= part <= nparts")
    return "" if n == 1 else f"p{k}of{n}"


def round_dir(piece: Path, rnd: str, part: str = "") -> Path:
    """Output place of round rnd (<piece>/rounds/<round>[.<part>]: dataset, preview). part = part_key(spec); '' keeps
    the
    place name. Unknown rounds or parts outside the form die."""
    if part and not PART_RE.fullmatch(part):
        die(f"round place part {part!r}: outside the form p<k>of<n> (built by part_key)")
    try:
        return Path(piece) / ROUNDS_DIR / (EPI.check_round(rnd) + (f".{part}" if part else ""))
    except ValueError as e:
        die(str(e))


def round_name(name: str) -> tuple[str, str] | None:
    """Round place name -> (round, part); None unless it is '<round>' or '<round>.p<k>of<n>'."""
    rnd, _sep, part = name.partition(".")
    if rnd not in EPI.ROUNDS or (part and not PART_RE.fullmatch(part)):
        return None
    return rnd, part


def round_dirs(piece: Path, rnd: str) -> list[Path]:  # noqa: V103 — public API for plugins (augment lane)
    """Every round rnd place of the bundle place (per part, existing only), so readers of the round (text track,
    diagnostics) do not miss a part."""
    root = Path(piece) / ROUNDS_DIR
    return (
        sorted(d for d in root.iterdir() if d.is_dir() and (round_name(d.name) or ("", ""))[0] == rnd)
        if root.is_dir()
        else []
    )


def bundle_of(ds: Path) -> Path | None:
    """Round dataset place (round_dir(<piece>, <round>[, part])/dataset) -> bundle place (<piece>: where the record's
    seq1* and the caches live). None when not of that form (a final dataset, another place; the caller states the
    reason)."""
    ds = Path(ds)
    if ds.name == "dataset" and ds.parent.parent.name == ROUNDS_DIR and round_name(ds.parent.name):
        return ds.parents[2]
    return None


def round_gate(a) -> None:
    """Driver round gate (check_env, before the claim): default bundle time cap (a.bundle_timeout_per_clip None) = 240 +
    the round's augment share (chain.extra_s) + the clip render share (render.clip) · an explicit env
    DEXENGINE_COLLECT_AUG that disagrees with the round dies (chain.round_aug)."""
    from dexengine.record import chain as CANON
    from dexengine.record.render import clip as CCL

    extra = CANON.extra_s(a.round)
    if a.bundle_timeout_per_clip is None:
        a.bundle_timeout_per_clip = 240 + extra + CCL.RENDER_TIMEOUT_PER_CLIP_S


def round_claim(meta: dict, rnd: str, pj: Path) -> None:
    """One operations place (tag) = one round; a round different from pilot.json's dies (skipping DONE would mix
    rounds)."""
    if meta and meta.get("round") != rnd:
        die(f"{pj}: this tag is round {meta.get('round')!r}: --round {rnd} needs a new tag")


EMPTY_FILE = "empty.json"  # empty round record (round place: 0 episodes to draw, a reason per clip)
#: view records of clips whose orig render was skipped as off-screen (round place: <tree name>.json = {clip, orig_fov};
#: the render_clip.json form as it is, received by the judged cell put_fov). Writer render.clip_all.write_fov_records
#: · the rule that carries them in the piece belongs to the carry constant (CARRY) (job scratch places are removed).
ORIG_FOV_DIR = "orig_fov"
SKIP_EMPTY = "skip-empty"  # merge.mode of an empty round result; read by the packer (empty_round) and the merge


def empty_round(res: dict) -> bool:  # noqa: V103 — public API for plugins (collect lanes)
    """Is the placed piece's result (LANE_MANIFEST result) an empty round? There is no dataset to merge, so the merge
    is skipped (it never enters the final) and the reasons are in result.excluded. Read by the failure count (it
    writes exclusion lines). The merge reads the same rule through SKIP_EMPTY of the piece's commit tree (old commit
    trees do not have this function)."""
    skip = (res.get("merge") or {}).get("mode") == SKIP_EMPTY
    if skip and res.get("episodes") != 0:
        die(
            input_missing(
                f"empty round statement is half done: merge.mode {SKIP_EMPTY} but episodes {res.get('episodes')!r}"
            )
        )
    return skip


def write_empty(rd: Path, rnd: str, round_excluded: dict, base_excluded: dict) -> dict:
    """Reason record of an empty round (0 episodes for the whole bundle, render.clip_all.render_all) -> {clip:
    reason}."""
    why = {**base_excluded, **round_excluded}
    Path(rd).mkdir(parents=True, exist_ok=True)
    write_atomic(Path(rd) / EMPTY_FILE, json.dumps({"round": rnd, "excluded": why}, ensure_ascii=False, indent=1))
    return why


def empty_result(rd: Path, clips: list[str]) -> dict | None:
    """Driver bundle result: with an empty round record, DONE + an alert (no merge, no cleanup) when the targets equal
    the reasoned exclusions (sets); FAILED when a target has no reason. None without the record (normal check path)."""
    f = Path(rd) / EMPTY_FILE
    if not f.is_file():
        return None
    e = json.loads(f.read_text())
    want = {P.tree_name(c) for c in clips}
    miss = sorted(want - set(e["excluded"]))
    if miss:
        return {
            "state": "FAILED",
            "reason": f"empty round {e['round']} but {len(miss)} targets without a reason: {miss[:3]}",
        }
    alert = f"[COLLECT_ALERT] empty round {e['round']}: 0 episodes · all {len(want)} targets are reasoned exclusions"
    say(alert)
    return {
        "state": "DONE",
        "reason": "",
        "alert": alert,
        "excluded": e["excluded"],
        "episodes": 0,
        "clips": 0,
        "target": len(want),
        "merge": {"mode": SKIP_EMPTY},
        "cleanup": {"done": False, "reason": "empty round"},
    }


def piece_id(piece: Path, rnd: str, part: str = "") -> str:
    """Merge key: <bundle place name>.<round place name> (bundle place name = <campaign>__<bundle>, owned by the run
    layer run_collect_work · round place name = <round>[.<part>])."""
    return f"{Path(piece).name}.{round_dir(piece, rnd, part).name}"


def _claim(f: Path, rec: dict) -> None:
    if f.is_file() and (old := json.loads(f.read_text())) != rec:
        die(f"{f}: piece identity {old} != {rec}: two writers on one place")
    f.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(f, json.dumps(rec, ensure_ascii=False) + "\n")


def work_root(main: Path) -> Path:
    """This job's work root: env data_layout.COLLECT_WORK_ROOT_ENV (local jobs = that job tree; not a directory
    dies) · without it main (remote hosts, clusters: the data root is the job tree). One job owns the work place."""
    env = P.data_layout.COLLECT_WORK_ROOT_ENV
    w = P.data_layout.collect_work_root_env()
    if w and not Path(w).is_dir():
        die(f"{env}={w} is not a directory")
    return Path(w) if w else Path(main)


def piece_dir(hand: str, combo: str, main: Path, b: dict, rnd: str, part: str = "") -> Path:
    """Work piece place of bundle b (run layer run_collect_work, hand root write guard) + the bundle identity record +
    the identity record of round rnd (· part) (round_dir/piece.json = bundle identity + round (+ part); another
    identity dies). Returns the bundle place (record.bundle
    --out)."""
    root = work_root(main)
    out = P.run_path("run_collect_work", hand, root=root, campaign=b["campaign"], bundle=b["bundle"])
    out.mkdir(parents=True, exist_ok=True)
    rec = {"hand": hand, "combo": combo, "dataset": b["dataset"], "campaign": b["campaign"], "bundle": b["bundle"]}
    _claim(out / PIECE_FILE, rec)
    _claim(
        round_dir(out, rnd, part) / PIECE_FILE,
        dict(rec, round=rnd, piece_id=piece_id(out, rnd, part), **({"part": part} if part else {})),
    )
    return out


#: owner module of the piece -> final (hand x one dataset) merge; the driver only calls it (contract
#: measure_piece(piece_dataset: Path) -> dict, outside the final lock, then merge_piece(final_dir: Path,
#: piece_dataset: Path, piece_id: str, measured: dict) -> dict under it; idempotence for the same piece_id is that
#: module's job).
MERGE_MODULE = "collect_merge"


def merge_gate(mode: str):
    """Gate of --merge (before start, ahead of the claim): off -> None (not called) · on -> the merge module with its
    two calls measure_piece · merge_piece (unreadable or missing dies)."""
    if mode == "off":
        return None
    if mode != "on":
        die(input_missing(f"--merge {mode!r}: on | off only (no default)"))
    import importlib

    try:
        mod = importlib.import_module(MERGE_MODULE)
    except ImportError as e:
        die(f"--merge on but the merge module {MERGE_MODULE} cannot be read ({e}): not starting")
    gone = [n for n in ("measure_piece", "merge_piece") if not callable(getattr(mod, n, None))]
    if gone:
        die(input_missing(f"--merge on but {MERGE_MODULE} has no {gone}: not starting"))
    say(f"merge on: {MERGE_MODULE}.measure_piece · merge_piece (checked pieces -> final dataset)")
    return mod


@contextlib.contextmanager
def final_lock(final_dir: Path):
    """Owner of the final dataset lock: flock on the parent folder of that dataset place (data/<ds>/collect). The two
    writers of the final place (driver merge merge_step, text track reload retext) write only under the same lock (one
    at a time across workers, drivers and tracks)."""
    import fcntl

    Path(final_dir).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(Path(final_dir).parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def merge_step(merger, final_dir: Path, piece: Path, rnd: str, part: str = "") -> dict:
    """Merge one checked piece (round rnd · part of the bundle place piece) into the final place: measure_piece before
    final_lock (the measurement never blocks another merge), merge_piece under it. Piece id = piece_id(bundle, round,
    part). Result = result.json merge field (the merger's return lives inside the result; its form is that module's)."""
    import time

    pid = piece_id(piece, rnd, part)
    ds = round_dir(piece, rnd, part) / "dataset"
    t0 = time.time()
    measured = merger.measure_piece(ds)
    with final_lock(final_dir):
        r = merger.merge_piece(final_dir, ds, pid, measured)
    if not isinstance(r, dict):
        raise TypeError(f"{MERGE_MODULE}.merge_piece returned {type(r).__name__}: must be a dict")
    return {
        "mode": "on",
        "final": str(final_dir),
        "piece_id": pid,
        "round": rnd,
        "sec": round(time.time() - t0, 1),
        "result": r,
    }


def merge_into(res: dict, merger, final_dir: Path, piece: Path, rnd: str, part: str = "") -> None:
    """merge field of the bundle result (res, after the check and before the cleanup): off (fn None) does not call ·
    only checked (DONE) bundles merge · a merge failure = FAILED (reason field); that bundle is not cleaned so the
    piece stays (--retry-failed merges again)."""
    res["merge"] = {"mode": "off" if merger is None else "on"}
    if merger is None or res.get("state") != "DONE":
        return
    try:
        res["merge"] = merge_step(merger, final_dir, piece, rnd, part)
    except Exception as e:  # noqa: BLE001 — a merge defect is that bundle's failure reason too
        res["state"], res["reason"] = "FAILED", f"merge failed {type(e).__name__}: {e}"[:300]
        res["merge"].update(error=res["reason"])
