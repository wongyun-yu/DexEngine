"""ManipTrans SharpA relative_rotation / relative_translation derived numerically from CHORD's released parquets (CPU).
R_rel = R_mano_wrist^T · R_robot_wrist per frame -> mean over all sequences and frames (chordal L2), spread reported as
the max angle. t_rel = R_mano_wrist^T · (p_robot - p_mano_wrist). Also checks the fingertip order: joints 16-20 must
be the closest to each finger's third joint.
usage: calib_from_chord.py <CHORD taco processed dir> <out json>   (the ManipTrans port reads it as
e123_inputs.CALIB_JSON:
       data/maniptrans/maniptrans_calib.json)"""

import glob
import json
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rr


def rot(wxyz):
    return Rr.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def main(src: str, out_path: str) -> None:
    out = {}
    for side in ("right", "left"):
        Rs, ts = [], []
        tip_ok = 0
        n = 0
        skipped = []  # parquets without this side's columns (or unreadable): left out, counted in one marker line
        for p in sorted(glob.glob(src + "/*/robot_name=sharpa_wave/motion_v1.parquet")):
            cols = [
                f"mano_{side}_joints",
                f"mano_{side}_joints_wxyz",
                f"robot_{side}_wrist_position",
                f"robot_{side}_wrist_wxyz",
            ]
            try:
                r = pq.read_table(p, columns=cols).to_pylist()[0]
            except (KeyError, ValueError, OSError) as e:  # pyarrow: ArrowKeyError / ArrowInvalid / ArrowIOError
                skipped.append(f"{p}: {type(e).__name__}")
                continue
            J = np.asarray(r[cols[0]], float)
            Q = np.asarray(r[cols[1]], float)
            Rm = rot(Q[:, 0])
            Rb = rot(np.asarray(r[cols[3]], float))
            pb = np.asarray(r[cols[2]], float)
            Rs.append((Rm.inv() * Rb).as_matrix())
            ts.append(Rm.inv().apply(pb - J[:, 0]))
            # fingertip order: which of joints 16-20 is closest to each finger's last joint (3, 6, 9, 12, 15)
            d = np.linalg.norm(J[:, 16:21, None] - J[:, None, [3, 6, 9, 12, 15]], axis=-1).mean(0)
            tip_ok += int((d.argmin(1) == np.arange(5)).all())
            n += 1
        if skipped:
            print(f"[MT_CALIB] {side}: {len(skipped)} parquet(s) skipped (no {side}-hand columns), e.g. {skipped[0]}")
        R = np.concatenate(Rs)
        t = np.concatenate(ts)
        U, _, Vt = np.linalg.svd(R.mean(0))
        Rmean = U @ Vt
        ang = np.degrees(Rr.from_matrix(np.einsum("ji,njk->nik", Rmean, R)).magnitude())
        out[side] = {
            "relative_rotation": Rmean.round(6).tolist(),
            "relative_translation": t.mean(0).round(5).tolist(),
            "rot_spread_deg_p50_p99_max": [
                float(np.percentile(ang, 50)),
                float(np.percentile(ang, 99)),
                float(ang.max()),
            ],
            "trans_std_m": t.std(0).round(5).tolist(),
            "frames": int(len(R)),
            "seqs": n,
            "tips_order_thumb_index_middle_ring_pinky_ok": f"{tip_ok}/{n}",
        }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[MT_CALIB] {sorted(out)} → {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
