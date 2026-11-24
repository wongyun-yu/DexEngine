"""Re-judge P&P runs with the P&P judge (pnp-a2nr) — one JSON line per run, the input of the level counter.

The judge has no release condition, so it reads no data-force mirror. Errors are reported as a line with an `error`
field (the counter treats that scene as unjudgeable), never by dropping the run.
CLI: `python -m experiments.sim_eval.judges.pnp_batch --data-root <data root> --home-map <built HOME>=<this HOME>
<run>…`
"""

from __future__ import annotations

import argparse
import json
import sys

from experiments.sim_eval.judges import pnp as J


def batch_tag(run: str) -> str:
    """Batch tag of a run folder `<runs>/<tag>_r1_<scene>` → `<tag>_r1` (policy mode = `policy-<tag>_r1`)."""
    return run.rstrip("/").split("/")[-1].split("_r1_")[0] + "_r1"


def judge_line(run: str, data_root: str, home_map: str, ctx) -> dict:
    """One P&P judge line (the field set the counter and the tables read)."""
    tag = batch_tag(run)
    try:
        r = J.judge_pnp(run, f"policy-{tag}", data_root, home_map, _ctx=ctx)
        objs = {
            k: {
                "d": v["d_end_m"],
                "d_org": v["d_org_end_m"],
                "dz": v["dz_end_m"],
                "dtilt": v["dtilt_deg"],
                "f": v["force_end_N"],
                "why": v["why"],
                "disp": v["disp_ref_m"],
            }
            for k, v in r["objs"].items()
        }
        return {
            "scene": run.rstrip("/").split("_r1_")[1],
            "tag": tag,
            "success": r["success"],
            "lift": r["lift_success"],
            "after": r["after"],
            "why": r["why"],
            "objs": objs,
            "judge": r["judge"],
            "end": r["end_row"],
            "lrow": r["lift_row"],
            "fit": r["frame_fit_m"],
        }
    except SystemExit as ex:
        return {"scene": run, "error": str(ex)[:200]}
    except Exception as ex:  # noqa: BLE001 — one bad run must not hide the others
        return {"scene": run, "error": f"{type(ex).__name__}: {ex}"[:200]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--home-map", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    ctx = J._load()
    for run in a.runs:
        print(json.dumps(judge_line(run, a.data_root, a.home_map, ctx), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
