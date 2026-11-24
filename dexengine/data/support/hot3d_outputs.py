"""HOT3D desk height recovery — the run's outputs: the recovery record and marker lines (``report``) and the
high/medium-confidence desk heights written into INTAKE support_audit.json (``write_intake``). The run itself is
dexengine/data/support/hot3d.py, which stays the tool named in both outputs."""

from __future__ import annotations

import json
import time

import numpy as np

from dexengine.data.support import work_surface as WS
from dexengine.data.support.plane import DESK_RECOVERED  # noqa: E402 — owner of the desk_mode vocabulary


def report(out, S, per_seq, results, t0) -> None:
    """hot3d_desk_recover.json in --out and the [WORK_SURFACE] / [hot3d_desk_recover] marker lines."""
    S["conf"] = dict(S["conf"])
    dzs = np.array([r["desk_z"] for r in results.values()])
    with open(out / "hot3d_desk_recover.json", "w") as fh:
        json.dump(
            {
                "summary": S,
                "per_seq": {k: {"table_z_world": v[2], "R_to_z_up": v[4].tolist(), **v[3]} for k, v in per_seq.items()},
                "clips": results,
            },
            fh,
            ensure_ascii=False,
            indent=1,
        )
    print(
        f"[WORK_SURFACE] branch A {S['branch_' + WS.BRANCH_PLACED]} · B {S['branch_' + WS.BRANCH_GIVEN]}"
        f"(lowered {S['given_lowered']}) · footprint {S['footprint']}(none {S['footprint_none']}) · "
        f"conflicts object {S['conflict_obj']} · hand {S['conflict_hand']} (owner "
        "dexengine/data/support/work_surface.py)",
        flush=True,
    )
    print(
        f"[hot3d_desk_recover] {S} · desk_z p05/p50/p95 {np.percentile(dzs, 5):.3f}/{np.median(dzs):.3f}/"
        f"{np.percentile(dzs, 95):.3f} "
        f"· seq {len(per_seq)} ({time.time() - t0:.0f}s)",
        flush=True,
    )


def write_intake(SA, clips, results, S, ip) -> None:
    """--write-intake: high/medium-confidence desk heights into INTAKE support_audit.json (atomic replace)."""
    n = 0
    for c, r in results.items():
        if r["confidence"] in ("high", "medium"):
            clips[c].update(
                {
                    "desk_mode": DESK_RECOVERED,
                    "desk_z": r["desk_z"],
                    "desk_z_confidence": r["confidence"],
                    "raw_seq": r["seq"],
                    "raw_table_z_world": r["table_z_world"],
                    "raw_dz": r["dz"],
                    "work_surface_branch": r["branch"],
                    "work_footprint_raw": r["work_footprint_raw"],
                    "work_footprint_basis": {
                        "rests": r["footprint"]["n_rests"],
                        "objects": r["footprint"]["n_objects"],
                        "z": r["footprint"]["z"],
                    },
                    "recover_source": "dexengine/data/support/hot3d.py",
                }
            )
            n += 1
    SA.setdefault("recovery", {})["hot3d_desk_recover"] = {
        "tool": "dexengine/data/support/hot3d.py",
        "updated": n,
        **S,
    }
    tmp = ip.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(SA, ensure_ascii=False, indent=1, default=str))
    tmp.replace(ip)
    print(f"[hot3d_desk_recover] INTAKE updated {n} clips → {ip}", flush=True)
