"""FlashDexRetarget rollout → the shared scorer (experiments/scoring/score.py, used as a library — its rules are not
copied here). One clip: the recorder's raw npz (fdr/eval_record.py · the authors' z-up world · motion frames) →
score.py's 'motion' form in CHORD world — t (K,) original-motion seconds · obj_pos_<tool|target> (N,K,3) ·
obj_quat_<body> (N,K,4 wxyz)
· the reference the method was given (ref_t · ref_obj_*) · horizon_t — then score.adapter_motion (independent
oracle: that reference
= the original motion parquet within 1 mm · 0.01 rad, else not scored) and score.score_all with the vertices of the
mesh FDR simulates (its objects/<id>/visual.obj, metres × the motion.pt scale).
Frame map (each part is proven per clip by the oracle, never trusted): world = the authors' TACO import rotation
to_world = diag(-1,-1,1) (code/flashdexretarget/src/retarget/human_demo/sources/taco.py · it is its own inverse) with
no height shift (config/retarget/source/taco.yaml table_z null) · time τ = (motion frame + head) / fps, head = frames
the authors' pipeline dropped before motion frame 0 (measured by the recorder against the imported clip).
A clip the authors' rules dropped (no rollout exists): --dropped <rule> writes its 0 record (fdr_dropped · steps_scored
0). usage (the FlashCHORD venv python, this repository on PYTHONPATH):
       score.py --raw <raw npz> --parquet <motion_v1 parquet> --objects <dataset>/objects --authors <authors_eval.json>
                --out <json>   |   score.py --dropped <rule> --clip <clip> --worlds N --out <json>"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from experiments.scoring import score as S

TO_WORLD = np.diag([-1.0, -1.0, 1.0])  # CHORD world ← the authors' z-up world (taco.py to_world; its own inverse)


def to_chord(pos, rot):
    """positions (...,3) and rotation matrices (...,3,3) of the authors' world → CHORD world (pos, quat wxyz)."""
    p = pos @ TO_WORLD.T
    q = R.from_matrix(TO_WORLD @ rot.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(rot.shape[:-2] + (4,))
    return p, q


def build(raw, parquet, tmpdir):
    d = np.load(raw, allow_pickle=True)
    ids = S.chord_object_ids(parquet)
    by_id = {v: k for k, v in ids.items()}
    if len(by_id) != len(ids):
        raise SystemExit(f"[FDR_SCORE] FAIL two bodies share an object id {ids}")
    head, fps = int(d["head"]), float(d["fps"])
    t = (np.asarray(d["t_frame"], float) + head) / fps
    out = {"t": t, "ref_t": t, "horizon_t": np.float64(t[-1])}
    bodies = {}
    for side in ("right", "left"):
        oid = str(d[f"obj_id_{side}"])
        if oid not in by_id:
            raise SystemExit(f"[FDR_SCORE] FAIL {side} object {oid} not in the original motion's {ids}")
        b = by_id[oid]
        bodies[b] = (oid, float(d[f"obj_scale_{side}"]))
        sq = np.asarray(d[f"sim_quat_{side}"], float)  # (N,K,4) wxyz (Isaac Lab)
        sp, sqw = to_chord(
            np.asarray(d[f"sim_pos_{side}"], float),
            R.from_quat(sq.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(sq.shape[:-1] + (3, 3)),
        )
        rp, rqw = to_chord(np.asarray(d[f"ref_pos_{side}"], float), np.asarray(d[f"ref_rotmat_{side}"], float))
        out[f"obj_pos_{b}"], out[f"obj_quat_{b}"] = sp, sqw
        out[f"ref_obj_pos_{b}"], out[f"ref_obj_quat_{b}"] = rp, rqw
    if sorted(bodies) != sorted(ids):
        raise SystemExit(
            f"[FDR_SCORE] FAIL the rollout's bodies {sorted(bodies)} ≠ the original motion's {sorted(ids)}"
        )
    f = os.path.join(tmpdir, "motion_form.npz")
    np.savez(f, **out)
    return f, bodies, d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--parquet")
    ap.add_argument("--objects")
    ap.add_argument("--authors")
    ap.add_argument("--dropped")
    ap.add_argument("--clip")
    ap.add_argument("--worlds", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.dropped:
        res = {
            "fdr_dropped": a.dropped,
            "clip": a.clip,
            "add_auc": 0.0,
            "add_auc_best": 0.0,
            "add_auc_best_world": None,
            "spider_sr": 0.0,
            "maniptrans_sr": 0.0,
            "worlds": int(a.worlds),
            "steps_scored": 0,
            "steps_missing": 0,
            "note": "dropped by the authors' own rules before training — no rollout exists; counted 0",
        }
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("[SCORE] " + json.dumps(res))
        return 0
    with tempfile.TemporaryDirectory() as td:
        f, bodies, d = build(a.raw, a.parquet, td)
        roll, ref = S.adapter_motion(f, a.parquet, 1.0)
        verts = {}
        for b, (oid, sc) in bodies.items():
            import trimesh

            m = trimesh.load(
                os.path.join(a.objects, oid, "visual.obj"), force="mesh", process=False, skip_materials=True
            )
            verts[b] = np.asarray(m.vertices, float) * sc
        res = S.score_all(roll, ref, verts)
    name = os.path.basename(a.raw)[:-4]
    au = json.loads(Path(a.authors).read_text())["per_clip"].get(name) if a.authors else None
    res["fdr"] = {
        "motion": name,
        "head_frames": int(d["head"]),
        "t0_s": round(float((int(d["t_frame"][0]) + int(d["head"])) / float(d["fps"])), 4),
        "replicas": int(d["replicas"]),
        "frame0_gap_mm": [round(float(x) * 1e3, 3) for x in d["frame0_gap_m"]],
        "objects": {b: oid for b, (oid, _) in bodies.items()},
        "vertices": "objects/<id>/visual.obj (the mesh FDR simulates)",
        "authors_eval": au,
        "frame_map": "CHORD world = diag(-1,-1,1) · authors' z-up world (taco.py to_world) · τ = (frame + head)/fps — "
        "proven by score.adapter_motion's oracle (method reference = original motion ≤ 1 mm · 0.01 rad)",
    }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    S.save_scored(a.out, roll)
    print("[SCORE] " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
