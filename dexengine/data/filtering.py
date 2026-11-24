#!/usr/bin/env python3
"""Check the base trees' hand–object assignment against each dataset's **source reference material** and write
`reference` into task_info.

Reference material differs per dataset; each uses what it has instead of one forced rule:
  taco     measured grasp assignment of the V1 frozen base (roster_root(V1, base)/<clip>/task_info.json
      <side>_object_mesh_dir) ↔ our contact-based assignment
  oakink2  source program ((lh_range, rh_range) keys of program_extension/extension/<seq>.json = hands **active** in
      that primitive, seg_desc of desc_info)
           ↔ our per-hand contact (active hands should touch, inactive hands should not). The clip key and `_sNNN`
           are decided by the owner of the intake rule (oakink2.raw_objects.intake_clip · SEQ_KEY: segments and pieces
           map to the source clip · `_sNNN` = program_info key order); extension (ranges) and desc_info (descriptions)
               are looked up by that key **name**.
  grab/arctic  intent label in the clip name (lift/pass/offhand…, use/grab, inspecting/…) ↔ number of touching hands
  hrdexdb  object in the clip name (apple__0) ↔ assigned object id
  h2o      official action_labels (github taeinkwon/h2odataset, public; 30 fps ranges · action ids 0–36) ↔ expected
      contact (actions other than apply/background overlapping means contact) + length check
  hocap/dexcanvas  no usable labels (text_annotations are like 'Task: s02') → reference.source="none"
Result: task_info["reference"] = {source, label/desc, expected, observed, agree} · ledger
data/<ds>/INTAKE/reference_check_<spec>.json (counts, mismatch list).
Usage: <retarget py> dexengine/data/filtering.py --hand <hand> --dataset <ds>|all; the hand is required (no default).
Producer of the quality step (data.stages `4b quality D12-c`); writes `quality.flags` · `train_candidate` ·
`keypoint_glitch_frames` · `fingertip_mm_window` · `reference`.
Hand-independent (human hand and object): no_object_contact · object_moves_without_contact
(assignment record rules.ASSIGN.quality: MANO contact in the tree window) · keypoint_glitch (source MANO) ·
object_gt_gap (long source GT gap of the main object, the mano layer `object_gt_gap` measured by the reader).
Hand-dependent: retarget_error (`fingertip_mm_window` = fingertip error of this hand's retargeting in the **tree
window**, dexengine.retarget.quality).
Writes go through tree_io and are skipped when the content is unchanged. Ledger:
data/<ds>/INTAKE/reference_check_<spec>.json (per hand).

Input fingerprint: each verdict writes `task_info.rules.D12c = {version, input_fp}` (hash of the inputs the verdict
read), and the stage-1 table (data.stages `4b`) counts whether this fingerprint equals the current inputs
(`freshness`) rather than whether the result keys exist; with keys alone, roles (4) changing later or pieces copying
the parent's verdict would still look "done". It sits under `rules` because the f0 input key
(`dexengine.retarget.f0.shadow_key.TI_UNREAD_TOP`) leaves out `rules`, and segment.split drops `rules` from pieces.

The "no source" reasons returned by the source readers (filtering_refs.SOURCE_REASONS) and the `note` / `label` texts
enter the stored reference and the input fingerprint.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data import filtering_refs as FR  # owner of the run state HAND (set below from --hand)
from dexengine.data.cases import (
    # keys and read entry of the assignment record (per-hand contact · no contact · motion without contact)
    assign_version as AR,
)
from dexengine.data.filtering_refs import (
    H2O_ACT,
    H2O_NO_CONTACT_VERBS,
    INTENT_HANDS,
    _h2o_entry,
    _oak_entry,
    _taco_legacy,
    dl,
    mains_by_hand,
    ref_h2o,
    ref_hrdexdb,
    ref_intent,
    ref_oakink2,
    ref_taco,
    root,
)
from dexengine.data.tree_io import write_json_if_changed
from dexengine.retarget import quality as RQA  # owner of the tree-window fingertip error (retarget_error input)

#: Version of the quality step; part of the fingerprint (a new version makes every tree stale → rerun from 4b). Bump it
#: when the verdict code and flags (this file) change, or when verdict inputs outside it (dexengine.retarget.quality
#: window fingertip error and diagnostics) change.
VERSION = "D12c-2026-10-12"
#: Location `task_info.rules.<RULE_KEY>` of the input fingerprint and its form (changing the version makes every old
#: fingerprint differ).
RULE_KEY, FP_SCHEMA = "D12c", "d12c1"
#: task_info fields the verdict reads; **the only statement**. Verdict functions (ref_* · mano_glitches · object_gt_gaps
#: ·
#: flags) receive only this projection: reading a new field requires listing it here (otherwise KeyError), and listing
#: it puts it into the fingerprint automatically.
#: None = the whole value · tuple = only those keys of a dict (or list of dicts)
INPUT_FIELDS = {
    "clip_id": None,
    "parent_clip": None,
    "task": None,
    "hands": None,
    "ref_dt": None,
    "objects": ("obj_id", "role", "side"),
    "rules": (AR.ASSIGN_KEY,),  # mains_by_hand · contact = assignment record
    "segment": ("start", "end", "T", "frame_map"),
    "provenance": ("mano_dir", "T_full", "retarget_dir"),
}


#: Window fields this step **owns** (top level · under quality; the fingerprint under rules also belongs to this tree
#: window); D36 pieces do not inherit them (segment.pieces.WINDOW_DECL). Candidate selection fails new trees that
#: skipped 4b.
WINDOW_TOP = ("reference",)
WINDOW_QUALITY = ("flags", "train_candidate", "keypoint_glitch_frames", "object_gt_gap", "fingertip_mm_window")


def input_view(ti: dict) -> dict:
    """task_info → deep copy keeping only the fields the verdict reads (no INTAKE ledger or written fields)."""
    v = {}
    for k, sub in INPUT_FIELDS.items():
        if k not in ti:
            continue
        x = ti[k]
        if sub is not None and isinstance(x, list):
            x = [{s: e[s] for s in sub if s in e} if isinstance(e, dict) else e for e in x]
        elif sub is not None and isinstance(x, dict):
            x = {s: x[s] for s in sub if s in x}
        v[k] = x
    return json.loads(json.dumps(v))


def trees(ds, hand=None):
    for emb in dl.EMBODIMENTS:
        d = root() / dl.render("base", ds, hand=hand or FR.HAND, emb=emb)
        if d.is_dir():
            yield from sorted(x for x in d.iterdir() if (x / "task_info.json").exists())


GLITCH_MM, GLITCH_WRIST_DEG, GLITCH_MIN_FRAMES, TIP_ERR_MM = 30.0, 5.0, 3, 20.0
# Keypoint defect = one finger jumps more than 30 mm relative to the wrist on a frame where the wrist turns less than 5°
# (dexcanvas 038: only the thumb, 64–71 mm, wrist 1–3°). The whole hand moving while the wrist turns 11°+ (fast wrist
# rotation in grab/arctic) is motion, not a defect; the measurement tells them apart.


def mano_glitches(ti) -> dict:
    """Frame-to-frame jumps of the source MANO fingertips (relative to the wrist); breaks in the **source**, not the
    retargeting. Only window frames count."""
    import numpy as np

    mf = root() / ti["provenance"]["mano_dir"] / "0/trajectory_keypoints.npz"
    mano = np.load(mf)
    a, T, out = int(ti["segment"]["start"]), int(ti["segment"]["T"]), {}
    for s in ti["hands"]:
        if f"qpos_finger_{s}" not in mano.files:
            # this hand is in the tree but the source keypoints are missing; the jumps were not measured, which is not
            # zero jumps (passing silently would let a defect through)
            raise InputMissingError(
                f"[ref] ✗ {ti['clip_id']}: source MANO keypoints {mf} lack qpos_finger_{s} — "
                f"tree hands {ti['hands']} · keys {sorted(mano.files)}"
            )
        from scipy.spatial.transform import Rotation as Rot

        tips = mano[f"qpos_finger_{s}"][a : a + T, :, :3]
        w7 = mano[f"qpos_wrist_{s}"][a : a + T]
        wr = w7[:, :3]
        rel = np.linalg.norm(np.diff(tips - wr[:, None, :], axis=0), axis=2) * 1000  # (T-1,5)
        q = w7[:, 3:7]
        dq = np.degrees(
            (Rot.from_quat(q[1:][:, [1, 2, 3, 0]]) * Rot.from_quat(q[:-1][:, [1, 2, 3, 0]]).inv()).magnitude()
        )
        single = (np.sort(rel, axis=1)[:, -1] > GLITCH_MM) & (
            np.sort(rel, axis=1)[:, -2] < GLITCH_MM / 2
        )  # only one finger jumps
        out[s] = int((single & (dq < GLITCH_WRIST_DEG)).sum())
    return out


#: Datasets whose reader **measures** source object GT gaps = the range of rule D12-g, which the rules registry derives
#: from the readers (every dataset whose reader calls the gap fill `hot3d._fill_gt_gaps`, rules.READER_CALL_SCOPE).
#: Without the `object_gt_gap` key in this dataset's mano layer the output came from an old reader (nearest sample, gaps
#: unmarked); it fails rather than silently reading "no gap". The reader applies the gap thresholds (interpolation
#: length, rest test); this only reads what the reader marked `long`.
from dexengine.data import rules as RR  # noqa: E402

GT_GAP_DATASETS = RR.rule_datasets("D12-g")


def object_gt_gaps(ds: str, ti) -> list:
    """Long source GT gaps of the **main object** overlapping this tree's window [[obj_id, t0_s, t1_s, gap_s]] (those
    the reader did not interpolate)."""
    if ds not in GT_GAP_DATASETS:
        return []
    mf = root() / ti["provenance"]["mano_dir"] / "task_info.json"
    g = json.loads(Path(mf).read_text()).get("object_gt_gap")
    if g is None:
        raise InputMissingError(
            f"[ref] {ds}: {mf} has no object_gt_gap — output of an old reader (gaps unmarked). "
            "Rebuild the mano layer with the new reader"
        )
    seg = ti["segment"]
    dt = float(ti["ref_dt"])
    a = int(seg["start"]) + int((seg.get("frame_map") or {}).get("offset") or 0)
    w0, w1 = a * dt, (a + int(seg["T"]) - 1) * dt
    mains = set(mains_by_hand(ti).values())
    return [
        [name, t0, t1, gs]
        for name, rows in sorted(g["objects"].items())
        if name in mains
        for t0, t1, gs in rows["long"]
        if t0 <= w1 and t1 >= w0
    ]


def _ref(ds):
    """Dataset → (verdict function, reference source reader | None = clip name only); main and the input fingerprint use
    the same table."""
    table = {
        "taco": (ref_taco, _taco_legacy),
        "oakink2": (ref_oakink2, _oak_entry),
        "hrdexdb": (ref_hrdexdb, None),
        "h2o": (ref_h2o, _h2o_entry),
    }
    if ds in table:
        return table[ds]
    if ds in ("grab", "arctic"):
        return (lambda t: ref_intent(ds, t)), None
    return (lambda t: {"source": "none", "note": "no usable source labels"}), None


def _consts():
    """Verdict thresholds and tables; changing a value changes the verdict (part of the fingerprint)."""
    return [
        GLITCH_MM,
        GLITCH_WRIST_DEG,
        GLITCH_MIN_FRAMES,
        TIP_ERR_MM,
        INTENT_HANDS,
        list(H2O_NO_CONTACT_VERBS),
        H2O_ACT,
    ]


def input_fingerprint(ds: str, ti: dict, view: dict | None = None) -> str:
    """Fingerprint of the inputs the 4b verdict reads = version, thresholds · projection (`input_view`) · reference
    source (`_ref`) · MANO keypoint file identity (size, mtime: the dexengine.placement.train_ready T5 input key
    convention) · (D12-g) the mano layer GT gap record. main and freshness call only this."""
    iv = input_view(ti) if view is None else view
    _fn, source = _ref(ds)
    mano = root() / str((iv.get("provenance") or {}).get("mano_dir") or "")
    try:
        st = (mano / "0/trajectory_keypoints.npz").stat()
        kp = [st.st_size, st.st_mtime_ns]
    except OSError:
        kp = None
    gap = None
    if ds in GT_GAP_DATASETS:
        try:
            gap = json.loads((mano / "task_info.json").read_text()).get("object_gt_gap")
        except (OSError, ValueError):
            gap = None
    payload = [
        FP_SCHEMA,
        VERSION,
        _consts(),
        ds,
        ds in GT_GAP_DATASETS,
        iv,
        source(iv) if source else None,
        kp,
        gap,
        RQA.diag_ident(root(), iv),
    ]
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    return f"{FP_SCHEMA}:{h[:32]}"


def tree_fresh(ds: str, spec_dir: str, tree: Path) -> bool:
    """Whether one tree was judged on the current inputs: marker and verdict present and the fingerprint equal.
    `freshness`
    (the 4b column) and the piece to-do (stage1.status.tree_todo) call this one test. No fingerprint (older version, a
    piece that copied its parent) · different · unreadable = stale."""
    try:
        ti = json.loads((tree / "task_info.json").read_text())
        rec = (ti.get("rules") or {}).get(RULE_KEY) or {}
        return bool(
            "train_candidate" in (ti.get("quality") or {})
            and "reference" in ti
            and rec.get("input_fp") == input_fingerprint(ds, ti)
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def freshness(ds: str, spec_dir: str) -> tuple[int, int]:
    """(base trees judged on the current inputs, total); owner of the data.stages `4b` column (tree verdict =
    tree_fresh).
    One stale tree makes stage 1 rerun from 4b."""
    ts = list(trees(ds, spec_dir))
    return sum(tree_fresh(ds, spec_dir, tp) for tp in ts), len(ts)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--hand", required=True, help="hand id; checks this hand's base layer and writes the markers (no default)"
    )
    ap.add_argument(
        "--check-fresh",
        action="store_true",
        help="compare input fingerprints only, no writes (`freshness`); per dataset `same/total`, rc 1 if any is stale",
    )
    return ap


def main():
    a = _parser().parse_args()
    FR.HAND = P.hand_spec_dir(a.hand)
    print(f"[ref] hand={a.hand} spec={FR.HAND}", flush=True)  # marker: which hand's base was checked
    dss = (
        sorted(d for d in dl.DATASETS if (root() / dl.render("intake", d) / f"base_{FR.HAND}.json").exists())
        if a.dataset == "all"
        else [a.dataset]
    )
    if a.check_fresh:
        return _check_fresh(dss)
    for ds in dss:
        _mark_dataset(ds, a.hand)
    return 0


def _check_fresh(dss: list[str]) -> int:
    """--check-fresh: print `same/total` input fingerprints per dataset; 1 when any tree is stale."""
    rc = 0
    for ds in dss:
        f, t = freshness(ds, FR.HAND)
        print(f"[ref] fingerprint {ds}: same {f}/{t}" + ("" if f == t else f" · stale {t - f}"), flush=True)
        rc |= int(f != t)
    return rc


def _mark_dataset(ds: str, hand: str) -> None:
    """Write the reference verdict and quality markers into every tree of `ds`, then the dataset ledger."""
    n, src, dis, n_wr = 0, collections.Counter(), [], 0
    for tp in trees(ds):
        n += 1
        n_wr += _mark_tree(ds, tp, src, dis)
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "dexengine/data/filtering.py",
        "dataset": ds,
        "hand": hand,
        "hand_spec": FR.HAND,
        "n_trees": n,
        "by_source": dict(src),
        "n_disagree": len(dis),
        "disagree": dis,
    }
    _led = root() / dl.render("intake", ds) / f"reference_check_{FR.HAND}.json"
    _led.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(
        f"[ref] {ds}: trees {n} · sources {dict(src)} · disagree {len(dis)} · rewritten {n_wr} · unchanged "
        f"{n - n_wr}" + (f" e.g. {dis[0]['clip'][:40]}" if dis else "")
    )


def _mark_tree(ds: str, tp: Path, src: collections.Counter, dis: list) -> int:
    """One tree: reference verdict + quality markers + input fingerprint; counts into `src`, disagreements into
    `dis` -> 1 when task_info.json was rewritten."""
    ti = json.loads((tp / "task_info.json").read_text())
    # the verdict reads only the **verdict input projection** (input_view), not task_info; the same inputs as
    # the fingerprint.
    iv = input_view(ti)
    fp = input_fingerprint(ds, ti, iv)
    r = _ref(ds)[0](iv)
    # the time is stamped anew **only when the content changes**; rerunning with the same result gives the same
    # bytes (no rewrite · the f0 input key is unchanged).
    _old = dict(ti.get("reference") or {})
    _t_old = _old.pop("time", None)
    r["time"] = _t_old if (_t_old and _old == r) else time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ti["reference"] = r
    src[r["source"]] += 1
    q = ti["quality"]
    _mark_quality(ds, iv, q, src)
    # input fingerprint; the stage-1 table counts 'rerun or not' from it (not from result keys). It lives under
    # rules (outside the f0 key).
    ti.setdefault("rules", {})[RULE_KEY] = {"version": VERSION, "input_fp": fp}
    src["train_candidate" if q["train_candidate"] else "flagged"] += 1
    if r.get("agree") is False:
        dis.append({"clip": ti["clip_id"], **{k: x for k, x in r.items() if k not in ("time", "ranges")}})
    return int(write_json_if_changed(tp / "task_info.json", ti))


def _mark_quality(ds: str, iv: dict, q: dict, src: collections.Counter) -> None:
    """Fill the tree's quality block `q`: glitch frames, object GT gaps, fingertip error, flags, train_candidate."""
    q["keypoint_glitch_frames"] = mano_glitches(iv)
    for _legacy in ("mano_tip_glitch_frames", "usable", "unusable_reasons"):
        q.pop(_legacy, None)
    gaps = object_gt_gaps(ds, iv)
    if ds in GT_GAP_DATASETS:
        q["object_gt_gap"] = gaps
    if gaps:
        src["object_gt_gap"] += 1
    aq, q["fingertip_mm_window"] = (
        AR.record(iv, FR.HAND)["quality"],
        round(RQA.window_fingertip_mm(root(), iv), 3),
    )
    flags = _quality_flags(aq, q, gaps)
    q["flags"] = flags
    q["train_candidate"] = not any(
        f in flags for f in ("no_object_contact", "object_moves_without_contact", "keypoint_glitch", "retarget_error")
    )


def _quality_flags(aq: dict, q: dict, gaps) -> list[str]:
    """Flags describe the data's character, not "unusable". Public datasets are valid as they are; these are
    conditions our pipeline (manipulation learning) needs."""
    return [
        k
        for k, x in (
            # no hand touches an object in the window (e.g. h2o applying lotion); unsuitable for manipulation
            # learning, the data is fine
            ("no_object_contact", aq["no_contact"]),
            # object moves without a hand; label/pose mismatch suspected
            ("object_moves_without_contact", aq["moved_without_contact"]),
            # source keypoint defect (one finger teleports)
            ("keypoint_glitch", sum(q["keypoint_glitch_frames"].values()) >= GLITCH_MIN_FRAMES),
            # long source GT gap of the main object (marker occlusion); no object pose meanwhile (the reader
            # interpolates only short ones)
            ("object_gt_gap", bool(gaps)),
            # retargeting fingertip error above the cap; to be retargeted again
            ("retarget_error", q["fingertip_mm_window"] > TIP_ERR_MM),
        )
        if x
    ]


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.filtering")
    raise SystemExit(main())
