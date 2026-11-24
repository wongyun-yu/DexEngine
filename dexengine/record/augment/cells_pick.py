#!/usr/bin/env python3
"""Placement augment cell choice entry: judge region dump (region/region_<clip>.npz, judge mode D) -> cell list
(cells/cells_<clip>.json, the input of judge mode L). Every rule is policy's (canonical cell canon_index, choice
pick_cells, seed clip_seed); this file only reads and writes. Contract: the judged-cell contract (v2);
bundle folder = paths.run_path("run_aug_cells", hand,
campaign=…, bundle=…).
cells json = {"clip", "judge_fp", "seed", "rule", "n_region", "cells": [[ky, kx, kyy, dz], …]}; first row = the
  canonical cell · list content is compared by list_part (outside identity and record fields; the judge_code_fp record
  field of old lists drops out by the same rule).
A canonical cell missing from the region, or missing grid, canon or judge identity fields, die.
Off-screen clips (policy.CELL_RULE_FOV): only clips with a bundle fov/fov_<clip>.json (put_fov; augment.cells
--fov-records puts the orig round render_clip.json orig_fov.view and Src.object_xy_frame0 there): the visibility of
each cell's variant (abs_to_variant) is predicted with render.fov.predict and SAMPLE_FOV_CELLS cells are chosen by
the key policy.fov_key (cells json rule · fov field).
A list that differs from the existing one dies without writing anything (write_cells: all or nothing).
Markers: [AUG_CELLS_PICK] <clip> region N · chosen 1 + n[ · off-screen rule …] ·
end line DONE_HEAD c · written a · unchanged b
Usage: cells_pick.py --hand <hand> --campaign <campaign> --bundle <bundle> --dataset <ds> [--clips a,b]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.record.augment import judged as AJ
from dexengine.record.augment import judged_region as JR
from dexengine.record.augment import policy as AP

MARK = "[AUG_CELLS_PICK]"
DONE_HEAD = f"{MARK} DONE clips "  # end line head; augment.cells checks it with the clip count
# off-screen clip material place in the bundle (owner augment.judged; load reads the same place)
fov_path = AJ.fov_path


def put_fov(root: Path, records: list, clips: list[str], hand: str, dataset: str, conds: dict) -> list[str]:
    """orig round render_clip.json (originally off-screen, orig_fov.view) -> bundle fov/fov_<clip>.json (+ q_xy_tree =
    placement Src.object_xy_frame0; conds = the augment.cells clip conditions emb · arm · scene). Rule, visibility or
    bundle clip mismatches, or a difference from an existing file, die (no overwrite)."""
    import hashlib

    from dexengine.record import episodes as EPI
    from dexengine.record.augment.place_src import Src
    from dexengine.record.render import fov as FOV

    def q_tree(c: str):
        cd = conds[c]
        return Src(
            P.main_root(), hand, dataset, cd["emb"], c, arm=cd["arm"], aerial=cd["scene"] == "aerial"
        ).object_xy_frame0()

    out = []
    for f in [x for x in records if x]:
        raw = Path(f).read_bytes()
        d = json.loads(raw)
        of, clip = d.get("orig_fov") or {}, P.clip_name(d["clip"])
        if of.get("rule") != EPI.ORIG_FOV_RULE or of.get("orig_kept") is not False or clip not in clips:
            raise SystemExit(
                f"{MARK} {f}: not off-screen cell material (rule {of.get('rule')} · "
                f"orig_kept {of.get('orig_kept')} · bundle clip {clip in clips})"
            )
        q = [float(v) for v in q_tree(clip)[:2]]
        rec = {
            "clip": clip,
            "source": str(f),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "view": FOV.check(of["view"]),
            "q_xy_tree": q,
        }
        dst = fov_path(root, clip)
        if dst.is_file():
            old = json.loads(dst.read_text())
            if (old["source_sha256"], old["q_xy_tree"]) != (rec["source_sha256"], q):
                raise SystemExit(f"{MARK} {dst} is already other material: no overwrite (clear it and judge again)")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(rec, indent=1))
        out.append(clip)
        print(f"{MARK} off-screen material {clip} <- {f} · q_tree {q}", flush=True)
    return out


def fov_scores(R: dict, cells: list, rec: dict) -> dict:
    """Per cell (predicted visibility, distance from the image centre): judge cell (absolute) -> variant
    (policy.abs_to_variant) -> render.fov.predict."""
    from dexengine.record.render import fov as FOV

    view = FOV.check(rec["view"])
    return {
        c: FOV.predict(view, *AP.abs_to_variant(float(th), d, R["canon"], rec["q_xy_tree"]))
        for c, th, d in zip(cells, R["theta_abs_deg"], R["d_abs"], strict=True)
    }


AGAIN = "again"  # reuse_why: judge again


def canon_i(R: dict) -> int:
    """Canonical cell index of the region dump (policy.canon_index); the choice and the reuse gate (reuse_why) use the
    same function."""
    return AP.canon_index(
        list(R["theta_abs_deg"]), [tuple(d) for d in R["d_abs"]], list(R["dz"]), R["canon"], R["grid"]
    )


def reuse_why(name: str, R: dict, verdict: str, comps, head: str) -> str:
    """Region dump reuse gate (augment.cells d_state): '' = done · AGAIN = judge again · else the refusal reason.
    Even with the same judge input the choice cannot use a dump whose canonical cell is not on the grid: a dump of an
    earlier commit is judged again and one of the current commit (head) is refused."""
    if verdict != "same":
        return f"{name}(judge input {verdict}: {','.join(comps)}"
    try:
        canon_i(R)
    except ValueError as e:
        if R["judge_commit"] == head:
            return f"{name}(dump of the current commit but {str(e)[:160]})"
        print(
            f"[AUG_CELLS_RUN] {name}: dump of an earlier commit ({R['judge_commit'][:9]}) without the canonical cell "
            "on the grid -> judge again",
            flush=True,
        )
        return AGAIN
    return ""


def pick_one(region_npz: Path, dataset: str, n: int | None = None, fov: dict | None = None) -> dict:
    """One region dump -> cells json dict (rules policy). n = cells outside the canonical one (None = canonical
    policy.SAMPLE; off-screen clips SAMPLE_FOV_CELLS, experiments only) · fov = the bundle fov file (CELL_RULE_FOV
    when present)."""
    clip = JR.clip_of_region(region_npz)
    R = JR.read_region(region_npz)
    cells = [(int(c[0]), int(c[1]), int(c[2]), float(z)) for c, z in zip(R["cells"], R["dz"], strict=True)]
    i0 = canon_i(R)
    seed = AP.clip_seed(dataset, clip)
    if fov is None:
        got = AP.pick_cells(cells, cells[i0], seed, n)
        return {
            "clip": clip,
            "judge_fp": R["judge_fp"],
            "seed": seed,
            "rule": AP.CELL_RULE,
            "n_region": len(cells),
            "cells": [list(c) for c in got],
        }
    from dexengine.record.render import framing as CF

    sc = fov_scores(R, cells, fov)
    got = AP.pick_cells(cells, cells[i0], seed, AP.SAMPLE_FOV_CELLS if n is None else n, key=AP.fov_key(seed, sc))
    vis = sum(sc[c][0] >= CF.VIS_RULE["in_min"] for c in got[1:])
    return {
        "clip": clip,
        "judge_fp": R["judge_fp"],
        "seed": seed,
        "rule": AP.CELL_RULE_FOV,
        "n_region": len(cells),
        "cells": [list(c) for c in got],
        "fov": {
            "source_sha256": fov["source_sha256"],
            "pred_visible": int(vis),
            "pred": [[round(sc[c][0], 3), round(min(sc[c][1], 1e6), 1)] for c in got],
        },
    }


#: content fields of a cell list: comparing an existing list with a re-chosen one looks at these only (the rest =
#: record fields, outside the comparison).
LIST_KEYS = ("clip", "judge_fp", "seed", "rule", "n_region", "cells", "fov", "ext_of", "level", "exhausted")


def list_part(cj: dict) -> dict:
    return {k: cj.get(k) for k in LIST_KEYS}


def write_cells(root: Path, picks: list[dict]) -> tuple[int, int]:
    """Writes the chosen lists to cells/cells_<clip>.json, all or nothing: first the content (list_part) is compared
    with existing files in memory -> any difference writes nothing and dies (the existing list stays: judge chunks are
    bound to it; to change it, delete that clip's cells and judge chunks together with the ledger first). Equal files
    are not rewritten · only missing ones go through tmp -> rename. -> (written, unchanged)."""
    cdir = root / "cells"
    body = {d["clip"]: json.dumps(d, indent=1) for d in picks}
    have = {c: (cdir / f"cells_{c}.json").read_text() for c in body if (cdir / f"cells_{c}.json").is_file()}
    changed = sorted(c for c, t in have.items() if list_part(json.loads(t)) != list_part(json.loads(body[c])))
    if changed:
        raise SystemExit(
            f"{MARK} ✗ cell lists changed on re-choosing {len(changed)} (e.g. {changed[:3]}): "
            "nothing written, existing lists kept (they would split from the judge chunks; the choice is "
            "not deterministic or the region or rule material changed)"
        )
    cdir.mkdir(parents=True, exist_ok=True)
    for c in sorted(set(body) - set(have)):
        out = cdir / f"cells_{c}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(body[c])
        tmp.replace(out)
    return len(body) - len(have), len(have)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", required=True, help="dataset of the clip seed (policy.clip_seed)")
    ap.add_argument("--clips", default="", help="comma list (default = all of region/)")
    a = ap.parse_args()
    # bundle folder = the data root side (the fetch place, the same form as augment.stage judged_bundle), even when run
    #   in a worktree
    root = P.run_path("run_aug_cells", a.hand, root=P.main_root(), campaign=a.campaign, bundle=a.bundle)
    regs = sorted((root / "region").glob("region_*.npz"))
    if a.clips:
        want = P.read_clip_list([c for c in a.clips.split(",") if c], dataset=a.dataset, label="--clips")
        regs = [root / "region" / f"region_{c}.npz" for c in want]
    if not regs:
        raise SystemExit(f"{MARK} 0 region dumps: {root / 'region'}")
    picks = []
    for f in regs:
        fp = fov_path(root, JR.clip_of_region(f))
        d = pick_one(f, a.dataset, fov=json.loads(fp.read_text()) if fp.is_file() else None)
        tail = (
            f" · off-screen rule {d['rule']} · predicted visible {d['fov']['pred_visible']}/{len(d['cells']) - 1}"
            if "fov" in d
            else ""
        )
        print(f"{MARK} {d['clip']} region {d['n_region']} · chosen 1 + {len(d['cells']) - 1}{tail}", flush=True)
        picks.append(d)
    new, same = write_cells(root, picks)
    print(f"{DONE_HEAD}{len(regs)} · written {new} · unchanged {same}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
