#!/usr/bin/env python3
"""Intake tool for delivered retargeting handoffs — verify, and admit only what passes into the
canonical layout.

Handoff contract (what a delivery has to satisfy):
    <root>/
      MANIFEST.sha256                       required. sha256 of every delivered file (paths relative to root)
      README.md · EXCLUDED.{md,json}         recommended
      <dataset>/
        mano/<emb>/<task>/{task_info.json, 0/trajectory_keypoints.npz}
        <hand>/<emb>/<task>/{task_info.json, scene.xml, scene_eq.xml, 0/{trajectory_kinematic.npz,
            trajectory_ikrollout.npz, retarget_diagnostics.*}}
        assets/objects/<obj>/{visual.obj, convex/}      every object task_info points at
        assets/robots/<hand>/{right,left,bimanual}.xml + assets/   the hash has to be registered in
            dexengine.core.layout.RETARGET_HAND_ASSETS
        visualization/<emb>/<task>/0.mp4 (recommended)

Check order and verdict (one ACCEPT / REJECT per clip; a dataset-level REJECT blocks every clip in it):
    D1 manifest — absent → dataset REJECT. Present → re-hash every file of that dataset and compare
              (mismatch or missing → REJECT).
    D2 hand asset — <ds>/assets/robots/<hand> exists, its xml3 hash is registered, status != rejected.
              Otherwise dataset REJECT.
    D3 scene load — after intake the meshdir and robots/<hand>/ references of scene*.xml are rewritten to fit
              the layout (fix_scene_refs), and --fix-scene-refs --verify-load really opens every clip with mujoco.
    C1 task_info — required keys, dataset_name/robot_type/embodiment_type agree with the path, ref_dt=0.02.
    C2 objects — the right/left/extra object directories of task_info are inside the handoff and have visual.obj.
    C3 scene.xml — every referenced file="…" exists relative to meshdir.
    C4 trajectory — qpos of trajectory_kinematic.npz is a finite (T, nq), nq == the nq of scene.xml compiled by MuJoCo
              (without mujoco this drops to WARN plus a joint-count heuristic).
    C5 mano — the mano tree for the same (emb, task) exists and its keypoints T is within ±MAX_T_DIFF of kinematic T.
    C6 clip_id — derived by dexengine.core.layout.clip_id_from_task, syntax-checked, unique inside the dataset
              (collision → both REJECT, a human decides).
    V1 visualization — absent → WARN.

Intake (--ingest): only ACCEPT clips are **hard-link copied** into the canonical layout (dexengine.core.layout.render)
    (same filesystem; a plain copy otherwise). The directory name is the clip_id, the contents (the task in
    task_info.json and so on) stay exactly as delivered (the input is immutable). The mapping is held by the
    INTAKE ledger:
      data/<ds>/INTAKE/<handoff_id>.json     check results · hashes · source_task→clip_id · rejection reasons
      data/<ds>/INTAKE/<handoff_id>/          copies of the handoff's README/EXCLUDED/reports (provenance kept)
    If the same clip_id is already there with different contents → REJECT (nothing is overwritten without --force).

Usage:
    python3 dexengine/data/intake.py <delivery dir>                    # verify only, print the report
    python3 dexengine/data/intake.py <delivery dir> --ingest           # verify + intake
    python3 dexengine/data/intake.py <delivery dir> --datasets grab --report <report dir>
Exit codes: 0 all ACCEPT · 3 some REJECT (the report is written anyway) · 2 argument/structure error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dexengine.core import config  # noqa: E402

#: `--help` is operator-facing and the docstring above is for whoever edits this file; wiring argparse to `__doc__`
#: would merge the two — hence this constant.
HELP = (
    "Intake tool for delivered retargeting handoffs: verifies them and brings only what passes into the "
    "canonical layout."
)


from dexengine.data import intake_ingest as II  # owner of the run state DEST · ARCHIVE (set below from the CLI)
from dexengine.data.handoff.checks import (
    Report,
    check_clip,
    dl,
    sha256_file,
    summarize,
    verify_archive,
    verify_hand_asset,
    verify_scene_loads,
)
from dexengine.data.intake_ingest import _dest, dest_root, fix_scene_refs, ingest, write_ledgers

NON_HAND_DIRS = {"mano", "assets", "visualization", "reports", "logs"}
#: head of the ingest summary line (the stage-1 driver shows it in its step line)
INGESTED = "[intake] ingested:"


def verify_manifest(root: Path, ds: str, rep: Report) -> dict:
    mf = root / "MANIFEST.sha256"
    if not mf.exists():
        if II.ARCHIVE is not None:
            rep.warnings.append(
                f"[{ds}] no MANIFEST.sha256 — single-archive integrity (zstd checksum ok, sha256 "
                f"{II.ARCHIVE['sha256'][:12]}…) recorded in the ledger instead"
            )
            return {"present": False, "archive": II.ARCHIVE}
        rep.ds_reject[ds].append(
            "MANIFEST.sha256 missing — cannot verify the handoff integrity (contract §3: a per-file manifest "
            "or a single --archive archive)"
        )
        return {"present": False}
    want = {}
    for line in mf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        rel = rel.strip().lstrip("*")
        if rel.startswith(ds + "/"):
            want[rel] = digest.strip()
    bad, missing = [], []
    for rel, digest in want.items():
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        if sha256_file(p) != digest:
            bad.append(rel)
    # files absent from the manifest (present in the handoff but unlisted) — outside the integrity
    # evidence, so they are excluded from intake
    listed = set(want)
    extra = [
        str(p.relative_to(root))
        for p in (root / ds).rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in listed
    ]
    if not want:
        rep.ds_reject[ds].append(f"MANIFEST.sha256 has no {ds}/ entries")
    if bad or missing:
        rep.ds_reject[ds].append(f"manifest mismatch {len(bad)}·missing {len(missing)} (e.g. {(bad + missing)[:3]})")
    if extra:
        rep.warnings.append(f"[{ds}] {len(extra)} files not in the manifest — not taken in (e.g. {extra[:2]})")
    return {"present": True, "listed": len(want), "bad": len(bad), "missing": len(missing), "unlisted": len(extra)}


def check_dataset(root: Path, ds: str, rep: Report, only_hands: tuple[str, ...] = ()) -> dict:
    dsdir = root / ds
    try:
        spec = dl.dataset(ds)
    except KeyError as e:
        rep.ds_reject[ds].append(str(e))
        return {}
    hands = sorted(d.name for d in dsdir.iterdir() if d.is_dir() and d.name not in NON_HAND_DIRS)
    if only_hands:
        hands = [h for h in hands if h in only_hands]
    info = {"dataset": ds, "hands": hands, "manifest": verify_manifest(root, ds, rep), "assets": {}}
    if not hands:
        rep.ds_reject[ds].append("no hand retarget directory (0 directories besides mano/assets/visualization)")
    for h in hands:
        info["assets"][h] = verify_hand_asset(dsdir, h, ds, rep)
    for h in hands:
        for embdir in sorted(p for p in (dsdir / h).iterdir() if p.is_dir()):
            emb = embdir.name
            for tdir in sorted(p for p in embdir.iterdir() if p.is_dir()):
                c = check_clip(dsdir, ds, h, emb, tdir.name, rep)
                c["hand_dir"] = h
                c["hand"] = info["assets"].get(h, {}).get("spec_dir", h)
    # C6 uniqueness (inside the dataset)
    by = defaultdict(list)
    for k, c in rep.clips.items():
        if c["dataset"] == ds and c.get("clip_id"):
            by[c["clip_id"]].append(k)
    for cid, ks in by.items():
        if len(ks) > 1:
            for k in ks:
                rep.clips[k]["reject"].append(
                    f"clip_id collision {cid!r}: {ks} — same name after normalisation; a person decides"
                )
    if spec.status == "canon":
        rep.warnings.append(
            f"[{ds}] taking a handoff into a canonical dataset — check that it does not mix with existing trees"
        )
    return info


def ingested_hand_map(ds: str, hands: tuple[str, ...]) -> dict[str, str]:
    """{delivered directory name: spec directory} for an already-ingested dataset, read from hand_assets in
    the INTAKE ledger. With no ledger (TACO re-retargeting) the map is the identity."""
    m: dict[str, str] = {}
    idir = _dest("intake", ds)
    for lp in sorted(idir.glob("*.json")) if idir.is_dir() else []:
        try:
            L = json.loads(lp.read_text())
        except json.JSONDecodeError:
            continue
        for dlv, a in (L.get("hand_assets") or {}).items():
            if a.get("registered") and a.get("spec_dir"):
                m[dlv] = a["spec_dir"]
    for h in hands:
        m.setdefault(h, h)
    return m


def fix_ingested(ds: str, hands: tuple[str, ...], verify: bool, workers: int | None) -> dict:
    hand_map = ingested_hand_map(ds, hands)
    specs = sorted(set(hand_map.values()) & set(hands or hand_map.values()))
    clip_dirs: list[Path] = []
    for spec in specs:
        for emb in dl.EMBODIMENTS:
            d = _dest("retarget", ds, hand=spec, emb=emb)
            if d.is_dir():
                clip_dirs += sorted(p for p in d.iterdir() if p.is_dir() and (p / "scene.xml").exists())
    changed = 0
    for cd in clip_dirs:
        changed += fix_scene_refs(cd, ds, cd.parent.parent.name, hand_map)["changed"]
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "dexengine/data/intake.py --fix-scene-refs",
        "hand_map": hand_map,
        "n_clips": len(clip_dirs),
        "xml_changed": changed,
    }
    if verify:
        rec["load"] = verify_scene_loads(clip_dirs, workers)
    idir = _dest("intake", ds)
    idir.mkdir(parents=True, exist_ok=True)
    (idir / "scene_refs.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "root",
        nargs="?",
        default="",
        help="handoff root (<root>/<dataset>/… · <root>/MANIFEST.sha256). Omitted in --fix-scene-refs mode",
    )
    ap.add_argument(
        "--fix-scene-refs",
        action="store_true",
        help="repair the scene*.xml references (meshdir · robots/<hand>/) of already ingested --datasets to match "
        "the layout and write the ledger INTAKE/scene_refs.json",
    )
    ap.add_argument(
        "--verify-load",
        action="store_true",
        help="(--fix-scene-refs) after repair, open every clip's scene*.xml with mujoco; needs the retarget python",
    )
    ap.add_argument("--workers", type=int, default=None, help="scene-load pool size (default: capacity.plan)")
    ap.add_argument(
        "--datasets", default="", help="comma list (default: every registered dataset directory under root)"
    )
    ap.add_argument("--hands", default="", help="comma list; only these hand directories (default all)")
    ap.add_argument(
        "--ingest", action="store_true", help="bring ACCEPT clips into the canonical layout (default: verify only)"
    )
    ap.add_argument("--force", action="store_true", help="replace other content already taken in")
    ap.add_argument(
        "--supersede-label",
        default="",
        help="when other content already taken in is **this hand's retarget layer**, move it to "
        "data/<ds>/_superseded/<label>/ instead of deleting,"
        " then replace (the explicit choice of stage-1 regeneration · the shared mano layer is not moved · "
        "not used with --force)",
    )
    ap.add_argument("--handoff-id", default="", help="ledger name (default <root name>_<YYYYMMDD>)")
    ap.add_argument("--report", default="", help="also write the report JSON into this directory")
    ap.add_argument(
        "--dest", default="", help="intake destination root (default: the data root paths.main_root). For tests"
    )
    ap.add_argument(
        "--archive",
        default="",
        help="the file when the handoff came as a single .tar.zst; the zstd checksum check replaces MANIFEST",
    )
    config.add_data_root_arg(ap)
    return ap


def main() -> int:
    a = _parser().parse_args()
    config.apply_data_root(a)
    if a.archive:
        II.ARCHIVE = verify_archive(Path(a.archive).resolve())
        print(f"[intake] archive integrity ok: {II.ARCHIVE['bytes'] / 1e9:.2f} GB sha256 {II.ARCHIVE['sha256'][:16]}…")
    if a.dest:
        II.DEST = Path(a.dest).resolve()
    if a.ingest:
        print(f"[intake] destination root: {dest_root()}")
    if a.fix_scene_refs:
        return _fix_scene_refs_mode(a)
    return _intake_mode(a)


def _fix_scene_refs_mode(a: argparse.Namespace) -> int:
    """--fix-scene-refs: repair the scene references of ingested datasets (exit 3 when a scene fails to load)."""
    dss = [d for d in a.datasets.split(",") if d]
    if not dss:
        print("[intake] --fix-scene-refs needs --datasets", file=sys.stderr)
        return 2
    bad = 0
    for ds in dss:
        rec = fix_ingested(ds, tuple(h for h in a.hands.split(",") if h), a.verify_load, a.workers)
        ld = rec.get("load") or {}
        nf = len(ld.get("fails", {}))
        bad += nf
        print(
            f"[scene_refs] {ds}: clips {rec['n_clips']} · xml changed {rec['xml_changed']} · hand_map "
            f"{rec['hand_map']}" + (f" · loaded {ld['n_ok']}/{ld['n_xml']} failed {nf}" if ld else "")
        )
        for x, r in list(ld.get("fails", {}).items())[:3]:
            print(f"    {x}: {r}")
    return 3 if bad else 0


def _intake_mode(a: argparse.Namespace) -> int:
    """Verify a handoff, optionally ingest it, and write the ledgers (exit 3 when anything was rejected)."""
    if not a.root:
        print("[intake] root is required", file=sys.stderr)
        return 2
    root = Path(a.root).resolve()
    if not root.is_dir():
        print(f"[intake] root does not exist: {root}", file=sys.stderr)
        return 2
    dss = _handoff_datasets(root, a.datasets)
    if not dss:
        print("[intake] no datasets to check", file=sys.stderr)
        return 2
    hid = a.handoff_id or f"{root.name}_{time.strftime('%Y%m%d')}"
    rep = Report()
    info = {}
    for ds in dss:
        info[ds] = check_dataset(root, ds, rep, tuple(h for h in a.hands.split(",") if h))
    if a.supersede_label and (a.force or not re.fullmatch(r"[A-Za-z0-9_.-]+", a.supersede_label)):
        raise SystemExit(
            "[intake] --supersede-label is not used with --force, and the label may only contain [A-Za-z0-9_.-]"
        )
    ing = ingest(root, rep, info, hid, a.force, a.supersede_label) if a.ingest else None
    if a.supersede_label and ing is not None:
        # applied marker, printed even when nothing moved (stage 1 checks this marker to see that the switch took
        # effect)
        print(
            f"[INTAKE_SUPERSEDE] label={a.supersede_label} moved {ing['counts'].get('clip_superseded', 0)} → "
            f"data/<ds>/_superseded/{a.supersede_label}/ ({','.join(dss)})",
            flush=True,
        )
    files = write_ledgers(root, rep, info, hid, ing, Path(a.report).resolve() if a.report else None)
    print(summarize(rep, info))
    if ing:
        print(f"{INGESTED} {ing['counts']} · ledgers {list(ing['ledgers'].values())}")
    for f in files:
        print(f"[intake] report → {f}")
    any_reject = any(rep.ds_reject.get(ds) for ds in dss) or any(v["reject"] for v in rep.clips.values())
    return 3 if any_reject else 0


def _handoff_datasets(root: Path, datasets: str) -> list[str]:
    """The datasets to check: --datasets, else every registered dataset directory under root (unregistered ones are
    named and ignored)."""
    dss = [d for d in datasets.split(",") if d] or sorted(
        d.name for d in root.iterdir() if d.is_dir() and d.name in dl.DATASETS
    )
    unknown = sorted(
        d.name for d in root.iterdir() if d.is_dir() and d.name not in dl.DATASETS and d.name not in ("reports", "logs")
    )
    if unknown:
        print(
            f"[intake] unregistered dataset directories (ignored): {unknown} — register them in "
            "dexengine.core.layout.DATASETS to take them in"
        )
    return dss


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.intake")
    sys.exit(main())
