#!/usr/bin/env python3
"""OakInk2 desk height (desk_z) recovery: writes per-clip desk_z from the source mocap annotations (HF
kelvin34501/OakInk-v2 anno_preview).

Version = RULE_VERSION.

Rules (apply unchanged to unseen sequences and clips; when they cannot be applied, a clip raises ClipError and a session
SessionError):
  ① transform: delivered layer = ours = Rx(+90°)·raw + t, with one t per clip (rotation fixed).
     t = median_k(ours_origin(k) − Rx·raw_origin(k)) over the main-object origin trajectory, raw 120 Hz interpolated to
         our 50 Hz times.
     Oracle (ClipError when exceeded): |t_x|,|t_y| ≤ T_XY_MAX_M · residual p95 ≤ RES_P95_MAX_M · median pose difference
         ≤ ANG_MED_MAX_DEG.
  ② source-world desk (z of the rotated raw = raw y): per object, rest intervals (centre speed < V_REST_MPS · angular
     speed < W_REST_DPS for at least MIN_REST_S, 30 Hz); interval level = median collision hull bottom (the hull is the
     geometry the simulation puts against the desk, objects.hull).
     Sequence G = duration-weighted median of the interval levels within the window DESK_WINDOW_M (no G when none fall
     inside). The window is a constant derived from the rest level distribution over all sequences.
  ③ session = (scene, date), read from the sequence name. Session desk = median of the session's G values. A session
     without any G raises SessionError (never falls back to a global value).
  ④ desk_z(clip) = raw_desk(seq) + t_z(clip), with raw_desk's policy = DESK_POLICY.
Oracles (markers only): lowest hand < desk_z − HAND_ORACLE_M → hand_below · |G − session| > SESSION_DEV_FLAG_M →
seq_dev_flag.
Write (--write-intake): desk_mode=table_recovered · desk_z · desk_z_confidence=raw_<source> · desk_rule{version, policy,
…}
  into clips[*] of INTAKE/support_audit.json (base_adapter copies them into the tree's support, marking trees rebuilt
  under this version). Fields of the old version (OLD_FIELDS) are removed. A clip recovered before that fails under
  this version makes the run fail without writing.
  Note: rerunning dexengine/data/support/plane.py on oakink2 erases these values (do not rerun).
  Per-clip error texts are stored in INTAKE (desk_rule.error) and stay as written.
Mapping: our clips `…_<hash12>_s<NN>` ↔ raw seq (hash prefix), NN = program_info key order. Interpreter = retarget venv
(torch for the pkl · pxr for USD).
Usage: $(python3 dexengine/core/paths.py --what retarget-py) dexengine/data/support/oakink2.py --out <dir>
[--write-intake] [--workers 6]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dexengine.core import capacity  # noqa: E402
from dexengine.core import paths as P  # noqa: E402
from dexengine.data.oakink2 import parts as OPT  # noqa: E402
from dexengine.data.objects import hull as OH  # noqa: E402
from dexengine.data.support.oakink2_rule import (
    ANG_MED_MAX_DEG,
    CLIP_RE,
    DESK_POLICY,
    DESK_WINDOW_M,
    HAND_ORACLE_M,
    MIN_OVERLAP,
    MIN_REST_S,
    MOCAP_HZ,
    OLD_FIELDS,
    POLICIES,
    RES_P95_MAX_M,
    RULE_VERSION,
    RX,
    SESSION_DEV_FLAG_M,
    STRIDE,
    T_XY_MAX_M,
    USD_PREFIXES,
    V_REST_MPS,
    W_REST_DPS,
    ClipError,
    fixed_rx_t,
    raw_desks,
    rest_intervals,
    rest_mask,
    seq_desk,
    session_of,
)
from dexengine.data.support.plane import DESK_RECOVERED  # noqa: E402 — owner of the desk_mode vocabulary

STEP = "dexengine.data.support.oakink2"  # capacity.NEEDS key (this module)

# Source reading.


def quat_wxyz(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot

    return Rot.from_quat(np.asarray(q, float)[:, [1, 2, 3, 0]]).as_matrix()


def hull_for_rid(rid: str) -> tuple[np.ndarray, str] | tuple[None, None]:
    """raw rid → (hull vertices, USD id). Without any USD copy (None, None): the object drops out of the desk samples
    (count recorded)."""
    k = rid.lower().replace("@", "_")
    for pre in USD_PREFIXES:
        mid = f"{pre}_{k}"
        if OPT.rid_of(mid) != rid:  # a mismatch with the identity rule's owner (oakink2.parts) fails
            raise ClipError(f"USD id {mid} → rid {OPT.rid_of(mid)} ≠ {rid}")
        try:
            return OH.hull_points("oakink2", mid, False), mid
        except OH.HullMissing:
            continue
    return None, None


def frames_of(fr: dict) -> tuple[np.ndarray, np.ndarray]:
    ks = np.array(sorted(fr), dtype=np.int64)
    T = np.stack([np.asarray(fr[k], dtype=np.float64)[:3, :4] for k in ks])
    return ks, T


def seq_job(args) -> dict:
    """One sequence: per-object rest intervals (hull bottoms) · G · t of the sequence's clips."""
    pkl, seq, clips, raw, mano_root = args
    out = {"seq": seq, "objs": {}, "clips": {}, "no_hull": []}
    try:
        with open(pkl, "rb") as fh:
            ot = pickle.load(fh)["obj_transf"]
        keys = list(json.loads((Path(raw) / "program/program_info" / f"{seq}.json").read_text()))
    except Exception as e:  # noqa: BLE001 — a source read failure is a ClipError for every clip of the sequence
        out["error"] = f"raw read failed: {type(e).__name__}: {e}"[:200]
        return out
    levels = []
    hz = MOCAP_HZ / STRIDE
    for rid, fr in ot.items():
        ks, T = frames_of(fr)
        T = T[::STRIDE]
        ok = np.isfinite(T).all((1, 2)) & (np.abs(T[:, :, 3]).max(1) < 10)
        if ok.sum() < MIN_OVERLAP:
            continue
        hull, mid = hull_for_rid(rid)
        if hull is None:
            out["no_hull"].append(rid)
            continue
        R, t = T[:, :, :3], T[:, :, 3]
        with np.errstate(invalid="ignore"):  # source gaps (NaN frames): comparisons are false, so not at rest
            bot = (R[:, 1, :] @ hull.T).min(1) + t[:, 1]  # raw y-up bottom = z after rotation
            rest = ok & rest_mask(t, R, hz)
        ints = rest_intervals(rest, bot, hz)
        out["objs"][rid] = {
            "usd": mid,
            "ints": [{"dur_s": round(x["dur_s"], 2), "level": round(x["level"], 5)} for x in ints],
        }
        levels += [(x["level"], x["dur_s"]) for x in ints]
    out["G"] = seq_desk(levels)
    out["n_win"] = sum(1 for z, _ in levels if DESK_WINDOW_M[0] <= z <= DESK_WINDOW_M[1])
    for idx, c in clips:
        out["clips"][c] = _clip_entry(ot, keys, idx, Path(mano_root) / c)
    return out


def _clip_entry(ot: dict, keys: list, idx: int, mdir: Path) -> dict:
    """clip_t of one clip; a ClipError becomes the clip's error entry."""
    try:
        return clip_t(ot, keys, idx, mdir)
    except ClipError as e:
        return {"error": str(e)[:240]}


def clip_t(ot: dict, keys: list, idx: int, mdir: Path) -> dict:
    if idx >= len(keys):
        raise ClipError(f"interval {idx} ≥ program_info interval count {len(keys)}")
    iv = ast.literal_eval(keys[idx])
    ivs = [x for x in iv if x]
    a0 = min(x[0] for x in ivs)
    ti = json.loads((mdir / "task_info.json").read_text())
    side = "right" if ti.get("right_object_mesh_dir") else "left"
    oid = Path(ti[f"{side}_object_mesh_dir"]).name
    rid = OPT.rid_of(oid)
    if rid not in ot:
        raise ClipError(f"main object {rid} not in the raw obj_transf")
    with np.load(mdir / "0/trajectory_keypoints.npz") as kp:
        o = kp[f"qpos_obj_{side}"]
        hz = []
        for s in ("right", "left"):
            w, f = kp.get(f"qpos_wrist_{s}"), kp.get(f"qpos_finger_{s}")
            if w is not None and np.abs(w[:, :3]).sum() > 0:
                hz.append(float(min(w[:, 2].min(), f[..., 2].min())))
    ks, T = frames_of(ot[rid])
    good = np.isfinite(T).all((1, 2))
    ks, T = ks[good], T[good]
    if len(ks) < MIN_OVERLAP:
        raise ClipError("too few raw main-object frames")
    rec = fixed_rx_t(
        (ks - a0) / MOCAP_HZ, T[:, :, 3], T[:, :, :3], np.arange(len(o)) * 0.02, o[:, :3], quat_wxyz(o[:, 3:7])
    )
    rec.update({"stage": idx, "obj": rid, "side": side, "hand_zmin": (min(hz) if hz else None)})
    return rec


# Run.


def _seq_jobs(raw: Path, max_seq: int) -> tuple[list, dict]:
    """One job per raw sequence whose hash has mano clips (first max_seq when > 0), and the clips whose hash has no
    raw pkl."""
    mano_root = P.layer_path("mano", "oakink2", emb="bimanual")
    by_hash = collections.defaultdict(list)
    for c in sorted(os.listdir(mano_root)):
        m = CLIP_RE.match(c)
        if m:
            by_hash[m.group(1)].append((int(m.group(2)), c))
    pk = {}
    for f in sorted((raw / "anno_preview").glob("*.pkl")):
        m = re.search(r"seq__([0-9a-f]+)__", f.stem)
        if m and m.group(1)[:12] in by_hash:
            pk[m.group(1)[:12]] = f
    jobs = [
        (str(f), f.stem, sorted(by_hash[h]), str(raw), str(mano_root))
        for h, f in sorted(pk.items(), key=lambda x: x[1].stem)
    ]
    if max_seq:
        jobs = jobs[:max_seq]
    missing = {h: c for h, c in by_hash.items() if h not in pk}
    return jobs, missing


def _run_seqs(jobs: list, workers: int | None) -> dict:
    """seq_job over the jobs in a process pool sized by capacity (`workers` = an explicit option), with a progress
    line every 50 sequences."""
    seqs, t0 = {}, time.time()
    with (
        capacity.sized(STEP, len(jobs), workers) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        for i, r in enumerate(ex.map(seq_job, jobs, chunksize=1), 1):
            seqs[r["seq"]] = r
            if i % 50 == 0:
                print(f"  … seq {i}/{len(jobs)} · {time.time() - t0:.0f}s", flush=True)
    return seqs


def _clip_record(s: str, rec: dict, desks: dict, seq_G: dict, sess: dict) -> dict:
    """Desk record of one recovered clip: desk_z under the policy, sources, residuals and flags."""
    rd, src = desks[DESK_POLICY][s]
    dz = rd + rec["t"][2]
    k = session_of(s)
    g = seq_G[s]
    smed = float(np.median(sess[k])) if sess[k] else None
    return {
        "seq": s,
        "session": f"{k[0]}|{k[1]}",
        "stage": rec["stage"],
        "obj": rec["obj"],
        "side": rec["side"],
        "desk_z": round(dz, 5),
        "raw_desk": round(rd, 5),
        "raw_desk_source": src,
        "seq_G": None if g is None else round(g, 5),
        "t": [round(x, 5) for x in rec["t"]],
        "res_p95_mm": round(rec["res_p95_m"] * 1000, 2),
        "ang_med_deg": round(rec["ang_med_deg"], 3),
        "hand_zmin": rec["hand_zmin"],
        "hand_below": bool(rec["hand_zmin"] is not None and rec["hand_zmin"] < dz - HAND_ORACLE_M),
        "seq_dev_flag": bool(g is not None and smed is not None and abs(g - smed) > SESSION_DEV_FLAG_M),
        "by_policy": {p: round(desks[p][s][0] + rec["t"][2], 5) for p in POLICIES},
    }


def _clip_records(seqs: dict, desks: dict, seq_G: dict, sess: dict, missing: dict) -> dict:
    """Per-clip desk records; sequence or clip errors and clips without a raw pkl become error entries."""
    clips = {}
    for s, r in seqs.items():
        for c, rec in r["clips"].items():
            if "error" in r or "error" in rec:
                clips[c] = {"seq": s, "error": r.get("error") or rec["error"]}
                continue
            clips[c] = _clip_record(s, rec, desks, seq_G, sess)
    for h, cl in missing.items():
        for _, c in cl:
            clips[c] = {"error": f"raw pkl missing (hash {h})"}
    return clips


def _summary(seqs: dict, seq_G: dict, sess: dict, clips: dict, ok: list) -> dict:
    """Run summary: counts, sources, no-hull objects and the rule constants."""
    S = {
        "rule_version": RULE_VERSION,
        "policy": DESK_POLICY,
        "seq": len(seqs),
        "seq_G": sum(g is not None for g in seq_G.values()),
        "seq_error": sum("error" in r for r in seqs.values()),
        "sessions": len(sess),
        "clips": len(clips),
        "recovered": len(ok),
        "errors": len(clips) - len(ok),
        "hand_below": sum(v["hand_below"] for v in ok),
        "seq_dev_flag_seqs": len({v["seq"] for v in ok if v["seq_dev_flag"]}),
        "no_hull_objects": sorted({x for r in seqs.values() for x in r.get("no_hull", [])}),
        "sources": dict(collections.Counter(v["raw_desk_source"] for v in ok)),
        "consts": {
            "DESK_WINDOW_M": DESK_WINDOW_M,
            "V_REST_MPS": V_REST_MPS,
            "W_REST_DPS": W_REST_DPS,
            "MIN_REST_S": MIN_REST_S,
            "STRIDE": STRIDE,
            "T_XY_MAX_M": T_XY_MAX_M,
            "RES_P95_MAX_M": RES_P95_MAX_M,
            "ANG_MED_MAX_DEG": ANG_MED_MAX_DEG,
        },
    }
    return S


def run(raw: Path, workers: int | None, max_seq: int) -> dict:
    """Desk height of every OakInk2 clip from the raw object transforms (per-sequence rest levels → desk G)."""
    jobs, missing = _seq_jobs(raw, max_seq)
    seqs = _run_seqs(jobs, workers)
    seq_G = {s: r.get("G") for s, r in seqs.items() if "error" not in r}
    seq_lv = {
        s: [(x["level"], x["dur_s"]) for o in r["objs"].values() for x in o["ints"]]
        for s, r in seqs.items()
        if "error" not in r
    }
    desks = {p: raw_desks(seq_G, p, seq_lv) for p in POLICIES}
    sess = collections.defaultdict(list)
    for s, g in seq_G.items():
        if g is not None:
            sess[session_of(s)].append(g)
    clips = _clip_records(seqs, desks, seq_G, sess, missing)
    ok = [v for v in clips.values() if "desk_z" in v]
    S = _summary(seqs, seq_G, sess, clips, ok)
    seqout = {
        s: {"G": r.get("G"), "n_win": r.get("n_win"), "error": r.get("error"), "objs": r["objs"]}
        for s, r in seqs.items()
    }
    return {"summary": S, "seqs": seqout, "clips": clips}


def _refuse_partial(res: dict, SA: dict) -> None:
    """Refuse the INTAKE write (SystemExit) when a clip recovered before fails now or a clip is not in INTAKE."""
    lost = [
        c
        for c, r in res["clips"].items()
        if "desk_z" not in r and (SA["clips"].get(c) or {}).get("desk_mode") == DESK_RECOVERED
    ]
    if lost:
        raise SystemExit(
            f"[desk_recover] ✗ {len(lost)} clips recovered before fail under this version ({RULE_VERSION}) — "
            f"not writing (e.g. {lost[:3]}: "
            f"{res['clips'][lost[0]].get('error')})"
        )
    unknown = sorted(
        {P.clip_name(c) for c in res["clips"]} - {P.clip_name(c) for c in SA["clips"]}
    )  # both are parent clip names (normalisation is formal only)
    if unknown:
        raise SystemExit(
            f"[desk_recover] ✗ clips missing from INTAKE {len(unknown)} (e.g. {unknown[:3]}) — not "
            "support_audit targets"
        )


def write_intake(res: dict) -> int:
    """Write the recovered desk heights (and the error of each failed clip) into INTAKE support_audit.json."""
    ip = P.layer_path("intake", "oakink2") / "support_audit.json"
    SA = json.loads(ip.read_text())
    _refuse_partial(res, SA)
    n = 0
    for c, r in res["clips"].items():
        rec = SA["clips"][c]
        for k in OLD_FIELDS:
            rec.pop(k, None)
        if "desk_z" not in r:
            rec["desk_rule"] = {"version": RULE_VERSION, "error": r["error"]}
            continue
        rec.update(
            {
                "desk_mode": DESK_RECOVERED,
                "desk_z": r["desk_z"],
                "desk_z_confidence": f"raw_{r['raw_desk_source']}",
                "raw_seq": r["seq"],
                "desk_rule": {
                    "version": RULE_VERSION,
                    "policy": DESK_POLICY,
                    "raw_desk": r["raw_desk"],
                    "raw_desk_source": r["raw_desk_source"],
                    "session": r["session"],
                    "seq_G": r["seq_G"],
                    "t": r["t"],
                    "res_p95_mm": r["res_p95_mm"],
                    "ang_med_deg": r["ang_med_deg"],
                    "hand_below": r["hand_below"],
                    "seq_dev_flag": r["seq_dev_flag"],
                },
            }
        )
        n += 1
    SA.setdefault("recovery", {})["oakink2_desk_recover"] = {
        "tool": "dexengine/data/support/oakink2.py",
        "updated": n,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **res["summary"],
    }
    tmp = ip.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(SA, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, ip)
    return n


def clip_transform(clip: str) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of capture = R·raw + t: reads the clip t this version wrote into INTAKE (owner = this file's desk_rule).
    Called by the raw supplement objects (oakink2.raw_objects). Missing or an old version fails (never falls back to old
    Kabsch values)."""
    rec = _intake_clips().get(clip) or {}
    dr = rec.get("desk_rule") or {}
    if dr.get("version") != RULE_VERSION or "t" not in dr:
        raise ClipError(
            f"{clip}: INTAKE desk_rule version {dr.get('version')!r} ≠ {RULE_VERSION} or no t — "
            "run dexengine/data/support/oakink2.py --write-intake first"
        )
    return RX.copy(), np.asarray(dr["t"], dtype=np.float64)


def clip_fit_residual(clip: str) -> float:
    """Transform residual p95 this version measured on the clip (m, INTAKE desk_rule.res_p95_mm): the oracle when the
    tree has no delivered pose to compare."""
    clip_transform(clip)  # version and presence check (ClipError when missing or old)
    return float(_intake_clips()[clip]["desk_rule"]["res_p95_mm"]) / 1000.0


_SA_CLIPS: dict | None = None


def _intake_clips() -> dict:
    global _SA_CLIPS
    if _SA_CLIPS is None:
        _SA_CLIPS = json.loads((P.layer_path("intake", "oakink2") / "support_audit.json").read_text())["clips"]
    return _SA_CLIPS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default=str(P.layer_path("raw", "oakink2") / "hf"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-intake", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--max-seq", type=int, default=0, help="for tests (fails together with writing)")
    a = ap.parse_args()
    if a.write_intake and a.max_seq:
        raise SystemExit("[desk_recover] --max-seq and --write-intake are not used together (no partial writes)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run(Path(a.raw), a.workers, a.max_seq)
    (out / "oakink2_desk_recover.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    S = res["summary"]
    dz = np.array([v["desk_z"] for v in res["clips"].values() if "desk_z" in v])
    print(
        f"[DESK_RULE] version={RULE_VERSION} policy={DESK_POLICY} seq={S['seq']} seq_G={S['seq_G']} sessions="
        f"{S['sessions']} "
        f"clips={S['clips']} recovered={S['recovered']} errors={S['errors']} hand_below={S['hand_below']} "
        f"seq_dev_flag_seqs={S['seq_dev_flag_seqs']} no_hull_objects={len(S['no_hull_objects'])}"
        + (
            f" desk_z_p05/50/95={np.percentile(dz, 5):.4f}/{np.median(dz):.4f}/{np.percentile(dz, 95):.4f}"
            if dz.size
            else ""
        ),
        flush=True,
    )
    if a.write_intake:
        n = write_intake(res)
        print(
            f"[DESK_RULE] INTAKE updated {n} clips → {P.layer_path('intake', 'oakink2') / 'support_audit.json'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
