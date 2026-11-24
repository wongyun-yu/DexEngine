#!/usr/bin/env python3
"""TACO hand↔object assignment producer: per sequence, **measures** from contact and sustained grasps between the source
MANO hand surface and the tool/target object surfaces, and writes `Object_Poses/<triplet>/<sid>/side_assign.json`
(consumer taco_convert_io._read_side_assign).

Why: the TACO source does not say which hand held the tool, and the old conversion filled in 'tool = right hand'
(in sequences where the left hand holds the tool, per-hand contact sensors and IK contact weights flip).

Rules (verdicts read the shared owners; no second verdict function):
  contact  frames where hand surface (MANO vertices) ↔ object surface ≤ scene_requirement.TOUCH_M
           touched = scene_requirement.touched (that hand's contact time ≥ TOUCH_MIN_S)
  grasp    per hand and object, runs of consecutive contact → grasp_runs.grasp_runs → grasp_runs.sustained_runs
           (D36 sustained grasp: holding together and brushes below the minimum length are dropped)
  assign   (canonical: D36 sustained grasp · D6-b time-varying, function assign)
           each hand's assigned object = its sustained-grasp object. Brushes and assists are no switch
           (the previous sustained-grasp object is kept · without any grasp the hand is idle).
           A run where both hands hold one object together = carry (shared); when one hand then grasps another
           object, only that hand changes.
           fixed slots (right/left) = each hand's longest sustained-grasp object · idle hand = the remaining object.
           ties within a hand (grasp difference between two objects < candidate_rules.MIN_DURATION_S) =
               grasp_runs.break_tie
           (smaller mean hand surface–object distance → earlier grasp start).
           both hands' main object equal = D6-a static sharing: right = left = that object · other objects are extra (no
               duplication).
           branches: separate · carry (shared carrying runs) · graze (touched but not a sustained grasp) · shared (D6-a)
               ·
           owned (the hand gripping a manipulated object that would stay extra owns it; assign)
  excluded sequences in the source defect ledger (data.rules.retarget_excluded,
      configs/datasets/taco/excluded_clips.txt)
           are not measured; {"excluded": reason} is written (the converter skips them; no_contact etc.)
  failure  no contact (when not in the ledger) · no sustained grasp · tie-break still equal
           → an error for that sequence (reason recorded, rc 1; never filled with a default)
Hand model: smplx MANO (flat_hand_mean · joint 0 = hand_trans, the same convention as the TACO conversion) ·
  left hand = mano_conventions (TACO, no correction).
Source reading = the converter's readers (dexengine.data.readers.taco_convert: _load_pickle ·
_validate_matching_frame_keys ·
  _stack_hand_frames · _load_hand_shape · _find_single_object_pose · _load_object_trajectory).
Marker [SIDE_ASSIGN_MEASURE] · --report <json> (per-sequence result · runs · whether equal to the old assumption) ·
  done = rows == sequences. retarget/batch.py matches the summary line text, so it stays as written; the
  side_assign.json `method` text is stored data and stays as written too.
Usage: $(paths.py --what retarget-py) dexengine/data/readers/taco_side_assign.py --raw <source root> [--only <key file>]
       [--jobs 16] [--report <json>] [--write]     # without --write, measure only (no writes)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dexengine.core import capacity
from dexengine.core import (
    paths as _P,  # noqa: E402 — owner of the side tuple (qpos_layout.SIDES) and the source readers (converter_module)
)
from dexengine.core.errors import InputMissingError
from dexengine.data import candidate_rules as TC  # noqa: E402 — owner of the minimum length MIN_DURATION_S
from dexengine.data.objects import scene_requirement as SR  # noqa: E402 — owner of the contact distance and touch rule
from dexengine.data.readers.taco_side_geom import (
    FPS,
    LazyGap,
    hand_vertices,
    intervals,
    moved,
    obj_motion,
    surface,
    surface_gap,
)
from dexengine.data.segment import grasp_runs as GR  # noqa: E402 — owner of grasp runs and sustained grasps (D36)

SIDES = tuple(_P.qpos_layout.SIDES)  # owner of the side tuple (qpos_layout)


class SeqError(RuntimeError):
    pass


def _sustained_grasps(case: dict, touch_s: dict, objs: list, fps: float, mins: float) -> tuple[dict, dict]:
    """(runs, dur): each hand's sustained grasp runs and its grasp time per object (s). SeqError when no hand touched
    an object, or no object was held long enough."""
    if not any(SR.touched({h: touch_s[(h, o)] for h in SIDES}, SR.TOUCH_MIN_S) for o in objs):
        raise SeqError(f"no contact — no hand touched an object for {SR.TOUCH_MIN_S}s or more")
    runs = {h: GR.sustained_runs(GR.grasp_runs(case, h, fps), fps, mins) for h in SIDES}
    dur = {h: {o: sum(b - a for a, b, x in runs[h] if x == o) / fps for o in objs} for h in SIDES}
    if not any(runs[h] for h in SIDES):
        raise SeqError(
            f"no sustained grasp — touches exist but no object was held for {mins}s or more (brushes and assists only)"
        )
    return runs, dur


def _primary_objects(objs: list, runs: dict, dur: dict, mean_gap: dict, mins: float) -> tuple[dict, dict]:
    """(primary, tie_break): each hand's longest sustained-grasp object (None without a grasp); a tie within a hand
    is broken by grasp_runs.break_tie and recorded, and SeqError when it stays tied."""
    primary, tie_break = {}, {}
    for h in SIDES:
        if not runs[h]:
            primary[h] = None
            continue
        top = sorted(objs, key=lambda o: -dur[h][o])
        tied = [o for o in top if dur[h][o] > 0 and dur[h][top[0]] - dur[h][o] < mins]
        if len(tied) > 1:
            first = {o: min(a for a, _b, x in runs[h] if x == o) for o in tied}
            win = GR.break_tie([{"obj": o, "mean_gap_m": mean_gap[(h, o)], "start": first[o]} for o in tied])
            if win is None:
                raise SeqError(
                    f"tie — {h} hand objects {tied}: sustained grasp difference < {mins}s, equal mean "
                    "distance and grasp start"
                )
            tie_break[h] = {
                "tied": tied,
                "won": win,
                "mean_gap_mm": {o: round(mean_gap[(h, o)] * 1e3, 2) for o in tied},
                "start": first,
            }
            primary[h] = win
            continue
        primary[h] = top[0]
    return primary, tie_break


def _give_manipulated(
    objs: list,
    extra: list,
    primary: dict,
    slots: dict,
    dur: dict,
    moving: dict,
    runs: dict,
    mean_gap: dict,
    mins: float,
) -> tuple[dict, list]:
    """(owned, extra): a manipulated object (moving and held) never stays extra — it goes into the slot of the hand
    that held it longest (``slots`` is updated); SeqError on a tie or more than one such object."""
    owned = {}
    manip = [o for o in extra if moving[o] and any(dur[h][o] > 0 for h in SIDES)]
    if len(manip) > 1:
        raise SeqError(f"more than one manipulated object {manip} would be extra — no hand left for the shared object")
    for o in manip:  # manipulated object = slot of the hand that held and moved it
        g = {h: dur[h][o] for h in SIDES}
        own = max(SIDES, key=lambda h: g[h])
        if abs(g["right"] - g["left"]) < mins:
            start = {h: min((a for a, _b, x in runs[h] if x == o), default=np.inf) for h in SIDES}
            own = GR.break_tie([{"obj": h, "mean_gap_m": mean_gap[(h, o)], "start": start[h]} for h in SIDES])
            if own is None:
                raise SeqError(
                    f"tie on the hand owning manipulated object {o} — sustained grasp {g}, equal mean "
                    "distance and grasp start"
                )
        slots[own] = o
        owned[o] = {"owner": own, "grip_s": {h: round(g[h], 3) for h in SIDES}, "was": primary[own]}
        extra = [x for x in objs if x not in slots.values()]
    return owned, extra


def _grazes(objs: list, runs: dict, touch_s: dict) -> list[str]:
    """``hand/object`` pairs touched long enough but never in a sustained grasp of that hand."""
    return sorted(
        {
            f"{h}/{o}"
            for h in SIDES
            for o in objs
            if touch_s[(h, o)] >= SR.TOUCH_MIN_S and not any(x == o for _a, _b, x in runs[h])
        }
    )


def assign(case: dict, touch_s: dict, mean_gap: dict, moving: dict) -> dict:
    """Canonical hand assignment (D36 sustained grasp · D6-b time-varying).
    case = {"grasp_s": {object: {hand: [[t0, t1] s]}}} · touch_s = {(hand, object): s} · mean_gap = {(hand, object): m}.
      Each hand's assigned object = its sustained grasp (grasp_runs.sustained_runs: ≥ MIN_DURATION_S · not an assist).
      Between runs (brushes, assists, no grasp) the previous sustained-grasp object is kept · before the first grasp the
          hand is idle (None).
      A run where both hands hold the same object together for ≥ MIN_DURATION_S = carry (shared); when one hand then
      grasps another object, only that hand changes.
      Fixed slots (converter side_assign.json right/left) = each hand's longest sustained-grasp object · the idle hand
          gets the remaining object.
      Ties within a hand (grasp difference between two objects < the minimum grasp) are broken by grasp_runs.break_tie
          (mean distance → grasp start).
      Both hands' main object equal = D6-a static sharing (shared): both slots hold that object · other objects are
          extra (they stay in the scene).
      Manipulated objects (moving · sustained grasp) never stay extra: the hand with the longest sustained grasp
      (difference < minimum grasp = break_tie) owns it · the other hand gets the shared object (owned). Motion from
      touching only is a brush; sharing stays.
    Returns {case, primary, slots, extra, segments, shared, idle, graze, tie_break, owned}. SeqError when undecidable
      (with a reason): no contact · no sustained grasp · tie-break still equal · tie on owning a manipulated object ·
          more than one manipulated object."""
    fps, mins = FPS, TC.MIN_DURATION_S
    objs = sorted(case["grasp_s"])
    runs, dur = _sustained_grasps(case, touch_s, objs, fps, mins)
    primary, tie_break = _primary_objects(objs, runs, dur, mean_gap, mins)
    shared = [
        [max(a1, a2) / fps, min(b1, b2) / fps, o1]
        for a1, b1, o1 in runs["right"]
        for a2, b2, o2 in runs["left"]
        if o1 == o2 and min(b1, b2) - max(a1, a2) >= int(mins * fps)
    ]
    idle = [h for h in SIDES if primary[h] is None]
    slots, extra = dict(primary), []
    if primary["right"] is not None and primary["right"] == primary["left"]:
        extra = [o for o in objs if o != primary["right"]]  # D6-a static sharing; other objects stay in the scene
    owned, extra = _give_manipulated(objs, extra, primary, slots, dur, moving, runs, mean_gap, mins)
    for h in idle:
        slots[h] = next(o for o in objs if o != primary[next(x for x in SIDES if x != h)])
    graze = _grazes(objs, runs, touch_s)
    case_kind = "owned" if owned else "shared" if extra else ("carry" if shared else ("graze" if graze else "separate"))
    seg = {h: [[a / fps, b / fps, o] for a, b, o in runs[h]] for h in SIDES}
    return {
        "case": case_kind,
        "primary": primary,
        "slots": slots,
        "extra": extra,
        "segments": seg,
        "shared": shared,
        "idle": idle,
        "graze": graze,
        "tie_break": tie_break,
        "owned": owned,
        "grip_s": {f"{h}/{o}": round(dur[h][o], 3) for h in SIDES for o in objs},
    }


_EXCL: dict = {}


def _excluded() -> dict:
    """TACO source defect ledger {clip: reason}, owned by data.rules.retarget_excluded
    (configs/datasets/taco/excluded_clips.txt)."""
    if not _EXCL:
        from dexengine.data import rules as RR  # noqa: PLC0415

        _EXCL.update(RR.retarget_excluded("taco"))
        _EXCL.setdefault("", "")
    return _EXCL


def _load_source(raw: Path, trip: str, sid: str) -> tuple[int, dict, dict, dict]:
    """(T, hand pose/trans per side, hand shape per side, {role: (object id, (T, 4, 4) trajectory)}) of one source
    sequence, read with the converter's readers; SeqError when tool and target are the same object."""
    from dexengine.data.readers import taco_convert as T2S  # noqa: PLC0415  (imports torch)

    hd, od = raw / "Hand_Poses" / trip / sid, raw / "Object_Poses" / trip / sid
    data = {s: T2S._load_pickle(hd / f"{s}_hand.pkl") for s in ("right", "left")}
    frames = T2S._validate_matching_frame_keys(data["right"], data["left"])
    hp = {
        s: tuple(x.detach().cpu().numpy() for x in T2S._stack_hand_frames(data[s], frames, s.capitalize()))
        for s in ("right", "left")
    }
    T = len(frames)
    shp = {
        s: np.asarray(T2S._load_hand_shape(hd / f"{s}_hand_shape.pkl", s.capitalize()), float)
        for s in ("right", "left")
    }
    objs = {}
    for role in ("tool", "target"):
        oid, f = T2S._find_single_object_pose(od, role)
        objs[role] = (oid, np.asarray(T2S._load_object_trajectory(f, T, 1.0), float))
    if objs["tool"][0] == objs["target"][0]:
        raise SeqError(
            f"tool and target are the same object {objs['tool'][0]} — objects cannot split the hand assignment"
        )
    return T, hp, shp, objs


def _hand_contact(objs: dict, hp: dict, shp: dict, trees: dict) -> tuple[dict, dict, dict]:
    """(touch_s, case, Vhs): per (hand, role) the touch time (s), the grasp intervals per role and hand, and each
    hand's MANO vertices."""
    touch_s, case, Vhs = {}, {"grasp_s": {}}, {}
    for side in SIDES:
        Vhs[side] = Vh = hand_vertices(side, np.asarray(hp[side][0], float), np.asarray(hp[side][1], float), shp[side])
        for role, (_oid, tr) in objs.items():
            c = surface_gap(Vh, tr, trees[role], bound=SR.TOUCH_M * 4) <= SR.TOUCH_M
            touch_s[(side, role)] = float(c.sum() / FPS)
            case["grasp_s"].setdefault(role, {})[side] = intervals(c)
    return touch_s, case, Vhs


def _side_assign_record(A: dict, objs: dict, mot: dict, moving: dict) -> dict:
    """The side_assign.json body of one measured sequence: the assignment with roles turned into object ids."""
    oid = {r: objs[r][0] for r in objs}
    rec = {
        "right": oid[A["slots"]["right"]],
        "left": oid[A["slots"]["left"]],
        "extra": [oid[o] for o in A["extra"]],
        "method": (
            f"measured[MANO contact ≤ {SR.TOUCH_M * 1e3:.0f} mm · main object of each hand's sustained grasp"
            f"(grasp_runs.sustained_runs ≥ {TC.MIN_DURATION_S}s) · {A['case']}]"
        ),
        "case": A["case"],
        "idle": A["idle"],
        "segments": {h: [[round(a, 3), round(b, 3), oid[o]] for a, b, o in A["segments"][h]] for h in SIDES},
        "shared": [[round(a, 3), round(b, 3), oid[o]] for a, b, o in A["shared"]],
        "owned": {oid[o]: {**v, "was": oid[v["was"]] if v["was"] else None} for o, v in A["owned"].items()},
        "motion": {oid[r]: {**{k: round(v, 4) for k, v in mot[r].items()}, "moved": moving[r]} for r in objs},
        "tie_break": {
            h: {
                "tied": [oid[o] for o in t["tied"]],
                "won": oid[t["won"]],
                "mean_gap_mm": {oid[o]: v for o, v in t["mean_gap_mm"].items()},
            }
            for h, t in A["tie_break"].items()
        },
    }
    return rec


def measure(job) -> dict:
    raw, key = job
    raw = Path(raw)
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.readers import taco_convert as T2S  # noqa: PLC0415  (imports torch)

    trip, sid = key.split("/", 1)
    clip = P.canonical_clip(T2S.taco_task_name(trip, sid), "taco")
    excl = _excluded().get(clip)
    if excl is not None:  # exclusion owned by the source defect ledger (data.rules.retarget_excluded)
        return {"key": key, "clip": clip, "excluded": excl, "assign": {"excluded": excl}}
    try:
        T, hp, shp, objs = _load_source(raw, trip, sid)
        trees = {r: surface(raw, oid) for r, (oid, _tr) in objs.items()}
        touch_s, case, Vhs = _hand_contact(objs, hp, shp, trees)
        mean_gap = LazyGap(lambda h, r: surface_gap(Vhs[h], objs[r][1], trees[r]).mean())
        mot = {r: obj_motion(tr) for r, (_oid, tr) in objs.items()}
        moving = {r: moved(mot[r], {h: touch_s[(h, r)] for h in SIDES}) for r in objs}
        A = assign(case, touch_s, mean_gap, moving)
        rec = _side_assign_record(A, objs, mot, moving)
        return {
            "key": key,
            "clip": clip,
            "T": T,
            "assign": rec,
            "case": A["case"],
            "same_as_old": (A["slots"]["right"] == "tool"),
            "tie_break": bool(A["tie_break"]),
            "idle": A["idle"],
            "graze": A["graze"],
            "grip_s": A["grip_s"],
            "touch_s": {f"{a}/{b}": round(v, 3) for (a, b), v in touch_s.items()},
        }
    except Exception as e:  # noqa: BLE001 — one sequence's failure goes into its row, is counted, and gives rc 1
        return {"key": key, "error": f"{type(e).__name__}: {e}"[:400]}


def check_processed(raw: Path, mano: Path) -> int:
    """Gate of the rebuild chain: for every MANO layer clip, task_info's right/left objects must equal the sequence's
    measured assignment (side_assign.json). No measured assignment (unmeasurable, producer not run) or a difference
    (layer converted under the old assumption) fails with rc 1; then rerun conversion (dexengine.retarget.batch --phase
    convert, which calls the producer first) → kinematic → decompose → detect_contact. Marker [SIDE_ASSIGN_CHECK] same ·
    differs · no assignment · no layer (counts)."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.readers import taco_convert as T2S  # noqa: PLC0415

    keys = sorted(f"{p.parent.name}/{p.name}" for p in (raw / "Hand_Poses").glob("*/*") if p.is_dir())
    if not keys:
        raise InputMissingError(f"[SIDE_ASSIGN_CHECK] ✗ no sequences in {raw}/Hand_Poses")
    out = {"same": [], "differs": [], "no_assign": [], "no_layer": []}
    for k in keys:
        trip, sid = k.split("/", 1)
        f = raw / "Object_Poses" / trip / sid / "side_assign.json"
        raw_name = T2S.taco_task_name(
            trip, sid
        )  # converter output name · the canonical layer uses the name after the naming convention
        ti = next(
            (
                mano / n / "task_info.json"
                for n in (P.canonical_clip(raw_name, "taco"), raw_name)
                if (mano / n / "task_info.json").is_file()
            ),
            mano / raw_name / "task_info.json",
        )
        if not ti.is_file():
            out["no_layer"].append(k)
            continue
        if not f.is_file():
            out["no_assign"].append(k)
            continue
        want = json.loads(f.read_text())
        if want.get("excluded"):
            out.setdefault("excluded", []).append(k)
            continue
        t = json.loads(ti.read_text())
        got = {h: Path(str(t.get(f"{h}_object_mesh_dir") or "")).name for h in SIDES}
        if (
            want["right"] == want["left"]
        ):  # D6-a sharing: the retarget scene has no left object and both hands use right_object
            got["left"] = got["left"] or got["right"]
        got_extra = sorted(str(e.get("obj_id")) for e in t.get("extra_objects") or [])
        same = all(got[h] == str(want[h]) for h in SIDES) and got_extra == sorted(
            str(e) for e in want.get("extra") or []
        )
        out["same" if same else "differs"].append(k)
    print(
        f"[SIDE_ASSIGN_CHECK] sequences {len(keys)} · same {len(out['same'])} · differs {len(out['differs'])} · "
        f"no assignment {len(out['no_assign'])} · excluded {len(out.get('excluded', []))} · no layer "
        f"{len(out['no_layer'])} "
        f"({mano})",
        flush=True,
    )
    for name in ("differs", "no_assign"):
        for k in out[name][:5]:
            print(f"[SIDE_ASSIGN_CHECK] ✗ {name} {k}", flush=True)
    n_exc = len(out.get("excluded", []))
    return (
        0
        if (out["same"] and len(out["same"]) + len(out["no_layer"]) + n_exc == len(keys) and not out["differs"])
        else 1
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--raw", required=True, help="TACO source root (Hand_Poses · Object_Poses · object_models_released)"
    )
    ap.add_argument(
        "--only",
        default="",
        help="file with one <triplet>/<sid> per line (names contain commas and spaces); all when omitted",
    )
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--report", default="")
    ap.add_argument("--write", action="store_true", help="write side_assign.json for measured sequences (temp+mv)")
    ap.add_argument(
        "--check-processed",
        default="",
        metavar="MANO_DIR",
        help="compare only, without measuring: whether each hand's object in the MANO layer "
        "(processed/taco/mano/bimanual) = side_assign.json",
    )
    a = ap.parse_args()
    raw = Path(a.raw).resolve()
    if a.check_processed:
        return check_processed(raw, Path(a.check_processed))
    keys = sorted(f"{p.parent.name}/{p.name}" for p in (raw / "Hand_Poses").glob("*/*") if p.is_dir())
    if a.only:
        want = [k.strip() for k in Path(a.only).read_text().splitlines() if k.strip()]
        miss = sorted(set(want) - set(keys))
        if miss:
            raise SystemExit(f"[SIDE_ASSIGN_MEASURE] ✗ sequences missing from the source {miss[:5]}")
        keys = want
    if not keys:
        raise InputMissingError(f"[SIDE_ASSIGN_MEASURE] ✗ no sequences in {raw}/Hand_Poses")
    t0 = time.time()
    step = "dexengine.data.readers.taco_side_assign"  # capacity.NEEDS key (this module)
    with capacity.sized(step, len(keys), a.jobs) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        rows = list(pool.imap_unordered(measure, [(str(raw), k) for k in keys], chunksize=1))
    rows.sort(key=lambda r: r["key"])
    exc = [r for r in rows if "excluded" in r]
    ok = [r for r in rows if "error" not in r and "excluded" not in r]
    err = [r for r in rows if "error" in r]
    wrote = 0
    if a.write:
        for r in ok + exc:
            f = raw / "Object_Poses" / r["key"] / "side_assign.json"
            tmp = f.with_name(f".{f.name}.tmp")
            tmp.write_text(json.dumps(r["assign"], indent=1, ensure_ascii=False))
            os.replace(tmp, f)
            wrote += 1
    same = sum(r["same_as_old"] for r in ok)
    print(
        f"[SIDE_ASSIGN_MEASURE] sequences {len(rows)}/{len(keys)} · measured {len(ok)} "
        f"(same as the old assumption 'tool = right hand' {same} · "
        f"different {len(ok) - same}) · excluded (ledger) {len(exc)} · not measurable {len(err)} · written {wrote} · "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )
    if a.report:
        Path(a.report).write_text(
            json.dumps(
                {
                    "raw": str(raw),
                    "n": len(rows),
                    "measured": len(ok),
                    "same_as_old": same,
                    "differs": len(ok) - same,
                    "excluded": len(exc),
                    "errors": len(err),
                    "rows": rows,
                },
                indent=1,
                ensure_ascii=False,
            )
        )
    for r in err[:10]:
        print(f"[SIDE_ASSIGN_MEASURE] ✗ {r['key']}: {r['error']}", flush=True)
    return 0 if len(rows) == len(keys) and not err and (not a.write or wrote == len(ok) + len(exc)) else 1


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.readers.taco_side_assign")
    sys.exit(main())
