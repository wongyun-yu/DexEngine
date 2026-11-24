#!/usr/bin/env python3
"""Placement-search ablation (paper Sec. E.5): the data-root-relative input set the full stage-2 chain (f0 seed → judge
→ bake → retime → gate) reads for the drawn clips, so a self-contained data root can be staged on a judge host.

The set was measured, not guessed: strace of a CPU judge (1 TACO clip, boot + clip load) and of a sandboxed sim
harvest replay (f0-only + skip-judge bake/retime/gate, 1 clip). Generalised by layer:
  per clip     base/<clip>/ · mano/<clip>/ · retarget/<clip>/ · contact_base/<clip>_* · contact_base_own/<clip>_* ·
               wrist_base/wrist_poses_{left,right}/<clip>.npz · f0 shadow <f0 node>/processed/<ds>/<hand>/<emb>/<clip>/
               + its wrist_poses_{left,right}/<clip>.npz
  per dataset  assets_objects/<obj> · usd/<obj> · meshes/<obj> for every object id the clips' task_info names ·
               assets_objects/../robots/<hand>/ ·
               data/<ds>/GENERATION · precompute/<ds>/diag/excluded_clips.txt
  global       data/_stage/ (every hand — the collider digest reads all) ·
  precompute/taco/ikpool/<arm>_<hand>_{left,right}/ ·
               EXTRA_GLOBAL below (from the bake trace)
A missing per-clip input dies; a missing optional layer entry (contact_base_own, wrist_base) is skipped and counted.

Usage: python -m experiments.ablations.placement.stage_list --root <data root> --hand wuji2 --clips <tsv with
dataset,axis,emb,clip cols> --out <list>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.data import rules as RR

_VARIANT = re.compile(r"^(left|right|raw)_")


def _family(name: str) -> str:
    """The object id without its variant prefix (left_ · right_ · raw_) — the key of one object's folder family."""
    return _VARIANT.sub("", name)


def _strings(x) -> set:
    """Every string leaf of a JSON value."""
    if isinstance(x, str):
        return {x}
    if isinstance(x, dict):
        return set().union(*(_strings(v) for v in x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*(_strings(v) for v in x)) if x else set()
    return set()


#: global inputs seen in the traces outside the per-clip / per-dataset layers (data-root relative).
#: setup inputs (coords.setup_inputs) a host fingerprint check found missing: the h2o desk plane and the OakInk2
#: official part tree. Part-object meshes named only through that tree are passed with --extra.
EXTRA_GLOBAL: list[str] = ["data/h2o/INTAKE/h2o_table_plane.json", "data/oakink2/raw/hf/object_affordance"]


def die(msg):
    raise SystemExit(f"[PLACE_ABL_STAGE] ✗ {msg}")


def _parse_args():
    """Parse the command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--arm", default="rby1")
    ap.add_argument("--clips", required=True, nargs="+")
    ap.add_argument("--extra", default="", help="file of extra data-root-relative paths (one per line)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return a


def _read_rows(a):
    """Clip rows of the given clip lists, keyed by (dataset, clip)."""
    rows = {}
    for f in a.clips:
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                rows[(r["dataset"], r["clip"])] = r
    return rows


def _stage_clips(rows, hd, need, out, rel, opt, a, root):
    """Stage each clip (base, ancestor mano/retarget, contact and wrist sidecars, f0 shadow).

    Returns the embs and the object names per dataset."""
    dsets = {}
    objs: dict[str, set] = {}
    for (ds, clip), r in sorted(rows.items()):
        emb = r["emb"]
        var = RR.ledger_axis_variant(r["axis"])
        dsets.setdefault(ds, set()).add(emb)
        ax = {"hand": hd, "emb": emb}
        base = Path(P.layer_path("base", ds, **ax)) / clip
        need(base, f"base {ds}/{clip}")
        ti = json.loads((base / "task_info.json").read_text(encoding="utf-8"))
        # object names: the objects/<x> paths and every bare string value (task_info names the moving object as a bare
        # id, e.g. "moving": "left_o02_0045_00002"); only strings that match an object folder (or its
        # variant family) are staged below, so other strings cost nothing
        objs.setdefault(ds, set()).update(re.findall(r"objects/([^/'\"\s]+)", json.dumps(ti)))
        objs[ds].update(_strings(ti))
        # segment / variant clips read an ancestor's mano · retarget folder (any emb folder): the clip itself, its
        # parent_clip, then the names left after stripping a trailing _t<k> / _seg<k> — every one that exists is staged
        names = [clip, ti.get("parent_clip") or clip]
        n = clip
        while True:
            m = re.sub(r"_(t|seg)\d+$", "", n)
            if m == n:
                break
            names.append(m)
            n = m
        for L in ("mano", "retarget"):
            hits = sorted(
                {h for nm in names for h in glob.glob(str(Path(P.layer_path(L, ds, **ax)).parent / "*" / nm))}
            )
            if not hits:
                die(f"{L} {ds}/*/{names} missing")
            out.update(rel(h) for h in hits)
        opt(Path(P.layer_path("contact_base", ds, **ax)) / f"{clip}_*")
        opt(Path(P.layer_path("contact_base_own", ds, **ax)) / f"{clip}_*")
        wb = Path(P.layer_path("wrist_base", ds, **ax))
        for s in ("left", "right"):
            opt(wb / f"wrist_poses_{s}" / f"{clip}.npz")
        fa = RR.f0_axis(ds, var, a.hand)
        node = root / fa["node"]
        sh = glob.glob(str(node / "processed" / ds / hd / emb / clip))
        if len(sh) != 1:
            die(f"f0 shadow {node}/processed/{ds}/{hd}/{emb}/{clip}: {len(sh)} hits")
        out.add(rel(sh[0]))
        for s in ("left", "right"):
            opt(node / "processed" / ds / hd / f"wrist_poses_{s}" / f"{clip}.npz")
    return dsets, objs


def _stage_datasets(dsets, hd, need, objs, out, rel, root):
    """Stage per dataset: hand assets, the named objects and their variant families, GENERATION/excluded lists and
    the sidecar registries."""
    for ds, _embs in sorted(dsets.items()):
        rob = Path(P.layer_path("assets_objects", ds)).parent / "robots" / hd  # the base scene.xml's hand meshes
        need(rob, f"hand assets {ds}")
        for L in ("assets_objects", "usd", "meshes"):  # the clips' objects only (ids read from task_info)
            p = Path(P.layer_path(L, ds))
            ids = objs.get(ds, set())
            # OakInk2 articulated objects: the joint USD is its own folder <a>__<b>__<joint> naming the parts — every
            # folder that names one of the clip's objects goes. Parts named only inside the part tree come through
            # --extra.
            # A named object stages its whole variant family: OakInk2 keeps one object as <id> · left_<id> ·
            # right_<id>
            # · raw_<id>, and the bake opens the siblings the task_info does not name (a missing sibling →
            # BAKE_ERROR). Datasets without prefixes are unchanged.
            fam = {_family(i) for i in ids}
            if p.is_dir():
                for d in sorted(os.listdir(p)):
                    if (
                        d in ids
                        or _family(d) in fam
                        or (
                            "__" in d
                            and any(i in d.split("__") or _family(i) in map(_family, d.split("__")) for i in ids)
                        )
                    ):
                        out.add(rel(p / d))
        for p in (root / "data" / ds / "GENERATION", root / "precompute" / ds / "diag" / "excluded_clips.txt"):
            if p.exists():
                out.add(rel(p))
        # the per-hand sidecar registries (every hand of both own layers): without them the resolver reads a
        # registered clip's sidecar from the shared folder, i.e. another hand's (a stale sidecar fingerprint →
        # BAKE_ERROR)
        reg = P.data_layout.SIDECAR_OWN_REGISTRY
        for L in P.data_layout.SIDECAR_OWN_LAYER.values():
            out.update(rel(h) for h in glob.glob(str(Path(P.layer_path(L, ds, hand="*")) / reg)))


def _stage_global(need, root, a):
    """Stage the hand stage, the IK pools and the extra paths."""
    need(root / "data" / "_stage", "hand stage")
    for s in ("left", "right"):
        need(root / "precompute" / "taco" / "ikpool" / f"{a.arm}_{a.hand}_{s}", "IK pool")
    extra = list(EXTRA_GLOBAL)
    if a.extra:
        with open(a.extra, encoding="utf-8") as fh:
            extra += [ln.strip() for ln in fh if ln.strip()]
    for e in extra:
        need(root / e, "extra")


def main() -> int:
    a = _parse_args()
    root = Path(a.root).resolve()
    hd = P.hand_spec_dir(P.hand_id(a.hand))
    rows = _read_rows(a)
    out: set[str] = set()
    skipped = 0

    def rel(p) -> str:
        return os.path.relpath(str(p), str(root))

    def need(p, what):
        if not os.path.exists(p):
            die(f"{what} missing: {p}")
        out.add(rel(p))

    def opt(pattern):
        nonlocal skipped
        hits = glob.glob(str(pattern))
        if not hits:
            skipped += 1
        out.update(rel(h) for h in hits)

    dsets, objs = _stage_clips(rows, hd, need, out, rel, opt, a, root)
    _stage_datasets(dsets, hd, need, objs, out, rel, root)
    _stage_global(need, root, a)
    bad = [p for p in out if p.startswith("..")]
    if bad:
        die(f"paths outside the data root: {bad[:3]}")
    Path(a.out).write_text("\n".join(sorted(out)) + "\n", encoding="utf-8")
    print(
        f"[PLACE_ABL_STAGE] clips={len(rows)} datasets={len(dsets)} entries={len(out)} optional_absent={skipped} "
        f"-> {a.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
