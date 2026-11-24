#!/usr/bin/env python3
"""Write object texture packs into the canonical tex_packs layer.

Copies prepared per-object textures (`<src>/<ds>__<id>/{tex.png, st.npy, meta.json}`) to the dexengine.core.layout
`tex_packs` layer (`data/<ds>/tex_packs/<id>/`). Rules and fields: textures.rules.

Checks (an object that fails any of them is not written and counts as failed; rc 1 at the end):
  · meta.tex_source equals the dataset's policy (PACK_POLICY). Uniform colour is not a pack: a dataset whose policy is
    uniform (GRAB) has nothing to write and fails (its colour is per object, textures.colors.object_color).
  · st.npy is finite float32 (3F,2), F = face count of the current training USD visual mesh; tex.png opens.
  · sha256_tex and sha256_st are recomputed (the source meta values are not trusted; when present they are compared).
  · sha256_usd_points and sha256_usd_fvi = the current training USD visual mesh (points float32, faceVertexIndices
    int32). The reader (render overlay `--packs`) checks the training USD identity against them.
Writing goes to a temporary folder, then a rename (identical content is skipped; different content fails without
`--replace`). This tool does not make textures; the source meta fields model, source and producer are carried over.

Usage (retarget venv or Isaac python; needs pxr; CPU):
  python dexengine/robots/textures/pack_write.py --src <dir> --dataset hocap,taco [--ids a,b] [--replace] [--dry-run]
Markers: per object `[TEX_PACK] <ds>/<id> <written|same|fail> …`; at the end `[TEX_PACK_DONE] n=<targets> written=W
same=S failed=F`; per dataset `[TEX_PACK_COVER] ds=… used=… with_pack=… missing=…` (used objects =
textures.rules.used_objects). rc 0 when nothing failed and W + S = number of targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from dexengine.core import artifact_paths
from dexengine.robots.textures import rules as OT

TP = artifact_paths


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def usd_mesh_sha(train_usd: str) -> tuple[int, str, str]:
    """Training USD visual mesh (/Object/base_link/visuals/mesh) -> (face-vertex count, sha(points), sha(fvi))."""
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(train_usd)
    p = st.GetPrimAtPath("/Object/base_link/visuals/mesh")
    if not p.IsValid():
        raise ValueError(f"no visual prim: {train_usd}")
    m = UsdGeom.Mesh(p)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float32)
    fvi = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    return int(fvi.size), _sha(pts.tobytes()), _sha(fvi.tobytes())


def _install_pack(dry, root, oid, src_dir, meta, dst):
    """Copy tex.png/st.npy and write meta.json in a temp folder, then move it into place (dry: nothing written)."""
    if dry:
        return "written", "dry"
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".{oid}.", dir=root)
    for f in ("tex.png", "st.npy"):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(tmp, f))
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if os.path.isdir(dst):
        shutil.rmtree(dst)  # only with --replace (one pack per object, fixed name)
    os.replace(tmp, dst)
    return "written", ""


def write_one(ds: str, oid: str, src_dir: str, root: str, replace: bool, dry: bool) -> tuple[str, str]:
    """-> (result written | same | fail, reason)."""
    if not all(os.path.isfile(os.path.join(src_dir, f)) for f in OT.PACK_FILES):
        return "fail", f"source files missing {src_dir}"
    with open(os.path.join(src_dir, "meta.json"), encoding="utf-8") as f:
        src = json.load(f)
    kind = src.get("tex_source")
    pol = OT.PACK_POLICY.get(ds, "none")
    if pol == "none":
        return "fail", f"{ds} is not a pack dataset (PACK_POLICY)"
    if pol == OT.SRC_UNIFORM:
        return "fail", f"{ds} policy is uniform colour, not a pack (nothing to write; colour: colors.object_color)"
    if kind != pol:
        return "fail", f"tex_source {kind!r} ≠ policy {pol!r}"
    st = np.load(os.path.join(src_dir, "st.npy"))
    n, sp, sf = usd_mesh_sha(TP.get(ds).usd_file(oid, False))
    if st.dtype != np.float32 or st.shape != (n, 2) or not np.isfinite(st).all():
        return "fail", f"st {st.dtype} {st.shape} ≠ float32 ({n}, 2)"
    from PIL import Image

    with Image.open(os.path.join(src_dir, "tex.png")) as im:
        im.verify()
    sha_tex = OT._sha_file(os.path.join(src_dir, "tex.png"))
    sha_st = OT._sha_file(os.path.join(src_dir, "st.npy"))
    for k, v in (("sha256_tex", sha_tex), ("sha256_st", sha_st)):
        if src.get(k) and src[k] != v:
            return "fail", f"source meta {k} ≠ actual file"
    meta = {k: src.get(k) for k in ("tex_source", "model", "source", "gen_fraction", "dataset_license", "tool_license")}
    miss = [k for k, v in meta.items() if v is None]
    if miss:
        return "fail", f"source meta fields missing {miss}"
    meta.update(
        sha256_tex=sha_tex,
        sha256_st=sha_st,
        sha256_usd_points=sp,
        sha256_usd_fvi=sf,
        obj=f"{ds}__{oid}",
        origin={
            k: src.get(k)
            for k in ("method", "prompt", "cond", "extra", "fallback_from")
            if src.get(k) not in (None, "")
        },
        writer="dexengine/robots/textures/pack_write.py",
    )
    dst = os.path.join(root, oid)
    if os.path.isdir(dst):
        try:
            with open(os.path.join(dst, "meta.json"), encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = {}
        keys = ("sha256_tex", "sha256_st", "sha256_usd_points", "sha256_usd_fvi", "tex_source")
        if all(old.get(k) == meta[k] for k in keys):
            return "same", ""
        if not replace:
            return "fail", "a different pack already exists (not overwritten without --replace)"
    return _install_pack(dry, root, oid, src_dir, meta, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="per-object texture folder (<ds>__<id>/ …)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument(
        "--pack-data-root",
        default="",
        help="write under this root with the same template instead of the data root (preview staging; "
        "the canonical place is untouched)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dss = [d for d in a.dataset.split(",") if d]
    want = {i for i in a.ids.split(",") if i}
    rows = []
    for ds in dss:
        if ds not in OT.PACK_POLICY:
            raise SystemExit(f"[TEX_PACK] {ds} is not a pack dataset — {sorted(OT.PACK_POLICY)}")
        root = OT.pack_root(ds, a.pack_data_root or None)
        print(f"[TEX_PACK] {ds} root {root}", flush=True)
        objs = sorted(n.split("__", 1)[1] for n in os.listdir(a.src) if n.startswith(f"{ds}__"))
        objs = [o for o in objs if not want or o in want]
        for oid in objs:
            try:
                res, why = write_one(ds, oid, os.path.join(a.src, f"{ds}__{oid}"), root, a.replace, a.dry_run)
            except Exception as e:  # noqa: BLE001 — one object's failure is counted; rc 1 at the end
                res, why = "fail", f"{type(e).__name__}: {e}"
            rows.append((ds, oid, res))
            print(f"[TEX_PACK] {ds}/{oid} {res} {why}".rstrip(), flush=True)
    n = len(rows)
    w = sum(r[2] == "written" for r in rows)
    s = sum(r[2] == "same" for r in rows)
    f = sum(r[2] == "fail" for r in rows)
    print(f"[TEX_PACK_DONE] n={n} written={w} same={s} failed={f} dry={int(a.dry_run)}", flush=True)
    for ds in dss:  # used objects (rules.used_objects) vs packs
        used = {aid for aid, _art in OT.used_objects(ds)}
        root = OT.pack_root(ds, a.pack_data_root or None)
        have = {
            o
            for o in (os.listdir(root) if os.path.isdir(root) else [])
            if all(os.path.isfile(os.path.join(root, o, x)) for x in OT.PACK_FILES)
        }
        miss = sorted(used - have)
        print(
            f"[TEX_PACK_COVER] ds={ds} used={len(used)} with_pack={len(used & have)} missing={len(miss)} "
            f"e.g.={miss[:12]} packs_not_used={len(have - used)}",
            flush=True,
        )
    return 0 if f == 0 and w + s == n and n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
