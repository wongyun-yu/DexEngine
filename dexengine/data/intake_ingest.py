"""Intake destination and ingest: the destination root (DEST, set by intake's --dest), scene reference
repair, ingest of ACCEPT clips into the canonical layout and the intake ledgers (ARCHIVE = single-archive integrity
evidence, set by intake's --archive). Driven by dexengine/data/intake.py."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from dexengine.core import paths as P
from dexengine.data.handoff.checks import (
    RE_ROBOTS,
    SCENE_XMLS,
    Report,
    dl,
    fixed_scene,
    link_tree,
    tree_fingerprint,
)

DEST: Path | None = None  # intake destination: None = the data root at call time (paths.main_root); --dest sets it


def dest_root() -> Path:
    """The intake destination root: --dest, else the data root (read when used, so --data-root applies)."""
    return DEST if DEST is not None else P.main_root()


def _dest(layer: str, ds: str, **axes) -> Path:
    return dest_root() / dl.render(layer, ds, **axes)


# ── D1 manifest ──────────────────────────────────────────────────────────────────────
ARCHIVE: dict | None = (
    # --archive: integrity evidence for a single-archive handoff (zstd frame checksum + sha256). Stands in for the
    # per-file manifest.
    None
)


def scene_assets_root(ds: str, hand: str) -> Path:
    """The asset root a scene xml's meshdir has to point at = the grandparent (…/assets) of the assets_robot
    layer (…/assets/robots/<hand>). The objects layer has to sit beneath it (…/assets/objects) — a table invariant.
    Otherwise the scene's objects/ references cannot be resolved, so raise."""
    root = _dest("assets_robot", ds, hand=hand).parent.parent
    objs = _dest("assets_objects", ds)
    if objs != root / "objects":  # the table must name the same root directly (text comparison, no links)
        raise RuntimeError(f"[{ds}] layout table invariant violated: assets_objects={objs} is not {root}/objects")
    return root


def _ingested_fingerprint(src: Path, dst: Path, ds: str, hand: str, hand_map: dict[str, str]) -> str:
    """Fingerprint of src **after it is taken in** at dst (the same form as tree_fingerprint); scene*.xml files are
    counted at the size fix_scene_refs will rewrite them to. Measuring src as is would call an already ingested
    identical result 'different' by the size of the meshdir fix (reruns would not be idempotent)."""
    rel = os.path.relpath(scene_assets_root(ds, hand), dst) + "/"
    h = hashlib.sha256()
    for q in sorted(x for x in src.rglob("*") if x.is_file()):
        size = q.stat().st_size
        if q.parent == src and q.name in SCENE_XMLS:
            s0 = q.read_text()
            s1 = fixed_scene(s0, rel, hand_map)
            size = len(s1.encode()) if s1 != s0 else size
        h.update(str(q.relative_to(src)).encode())
        h.update(str(size).encode())
    return h.hexdigest()[:16]


def fix_scene_refs(clip_dir: Path, ds: str, hand: str, hand_map: dict[str, str]) -> dict:
    """Bring the scene*.xml references of an ingested clip in line with our layout
    (every reference is swept).
    ① meshdir/texturedir = relpath(asset root, clip directory) — the scene builder writes ../../../assets/ relative to
       processed/<ds>/<hand>/<emb>/<clip>, but our retarget/base layer is one level deeper
       (../../../../assets/), so leaving it alone stops mujoco from opening the meshes.
    ② robots/<delivered directory name>/ → robots/<spec directory>/ (when intake renamed it, the xml
       reference is renamed too).
    An unknown robots/<x>/ left over raises. Changed files are written with tmp+replace, which breaks the
    hard link to the delivered original."""
    root = scene_assets_root(ds, hand)
    rel = os.path.relpath(root, clip_dir) + "/"
    out = {"changed": 0, "meshdir": rel}
    for name in SCENE_XMLS:
        xp = clip_dir / name
        if not xp.exists():
            continue
        s0 = xp.read_text()
        s = fixed_scene(s0, rel, hand_map)
        unknown = set(RE_ROBOTS.findall(s)) - set(hand_map.values())
        if unknown:
            raise RuntimeError(f"{xp}: unknown hand reference robots/{sorted(unknown)} — hand_map={hand_map}")
        if s != s0:
            tmp = xp.with_name(xp.name + ".tmp")
            tmp.write_text(s)
            os.replace(tmp, xp)
            out["changed"] += 1
    return out


def _superseded(ds: str, dst: Path, label: str) -> Path:
    """Quarantine location of an earlier generation's intake: data/<ds>/_superseded/<label>/<path relative to
    data/<ds>>.
    It lies outside the layer paths, so no layer glob picks it up (the canonical-axis hand's output is never
    deleted)."""
    base = dest_root() / "data" / ds
    return base / "_superseded" / label / dst.relative_to(base)


def ingest(root: Path, rep: Report, info: dict, handoff_id: str, force: bool, supersede: str = "") -> dict:
    """Bring ACCEPT clips into the canonical layout. Other content already taken in: default = reject · force = delete
    and replace · supersede(label) = move **only this hand's retarget layer** to the quarantine location and replace
    (the explicit choice of stage-1 regeneration). The hand-independent shared layer (mano) is read by other hands, so
    supersede never moves it; a difference is rejected."""
    done = Counter()
    out = {}
    for ds, dinfo in info.items():
        if rep.ds_reject.get(ds):
            continue
        dsdir = root / ds
        _ingest_robot_assets(dsdir, ds, dinfo, rep, done)
        if rep.ds_reject.get(ds):
            continue
        _ingest_object_assets(dsdir, ds, rep, force, done)
        # clips
        hmap = {
            h: a["spec_dir"] for h, a in dinfo["assets"].items() if a.get("registered")
        }  # the same table as fix_scene_refs and the fingerprint
        for c in rep.clips.values():
            if c["dataset"] != ds or c["reject"]:
                continue
            _ingest_clip(c, dsdir, ds, hmap, force, supersede, done)
        out[ds] = _copy_provenance(root, ds, handoff_id)
    return {"counts": dict(done), "ledgers": out}


def _ingest_robot_assets(dsdir: Path, ds: str, dinfo: dict, rep: Report, done: Counter) -> None:
    """Hand assets (one per dataset): if already present, the content fingerprint has to match."""
    for h, a in dinfo["assets"].items():
        if not a.get("registered"):
            continue
        src = dsdir / "assets" / "robots" / h
        dst = _dest("assets_robot", ds, hand=a["spec_dir"])
        if dst.exists():
            if dl.asset_hash(str(dst)) != a["sha256_xml3"]:
                rep.ds_reject[ds].append(
                    f"hand asset hash of the existing {dst} differs from the handoff — check by hand"
                )
                break
        else:
            link_tree(src, dst)
            done["asset_robot"] += 1


def _ingest_object_assets(dsdir: Path, ds: str, rep: Report, force: bool, done: Counter) -> None:
    """Object assets: an existing object with different content is skipped with a warning (--force replaces)."""
    objdst = _dest("assets_objects", ds)
    for od in (
        sorted(p for p in (dsdir / "assets" / "objects").resolve().iterdir() if p.is_dir())
        if (dsdir / "assets" / "objects").is_dir()
        else []
    ):
        dst = objdst / od.name
        if dst.exists():
            if tree_fingerprint(dst) != tree_fingerprint(od) and not force:
                rep.warnings.append(
                    f"[{ds}] object {od.name} already exists with different content — skipped (--force replaces)"
                )
            continue
        link_tree(od, dst)
        done["asset_object"] += 1


def _ingest_clip(c: dict, dsdir: Path, ds: str, hmap: dict, force: bool, supersede: str, done: Counter) -> None:
    """One ACCEPT clip: plan its two layers (hand retarget, shared mano), then write them, the scene references and
    the visualization. A rejected pair leaves the clip untouched."""
    emb, task, cid, hand = c["emb"], c["source_task"], c["clip_id"], c["hand"]
    pairs = [
        (dsdir / c["hand_dir"] / emb / task, _dest("retarget", ds, hand=hand, emb=emb, clip=cid), True),
        (dsdir / "mano" / emb / task, _dest("mano", ds, emb=emb, clip=cid), False),
    ]
    plan, skipped = _clip_plan(c, pairs, ds, hand, hmap, force, supersede)
    if c["reject"]:
        return
    for src, dst, old in plan:
        if old == "rm":
            shutil.rmtree(dst)
        elif old is not None:
            old.parent.mkdir(parents=True, exist_ok=True)
            os.rename(dst, old)
            c["superseded_to"] = str(old)
            done["clip_superseded"] += 1
        link_tree(src, dst)
    fix_scene_refs(_dest("retarget", ds, hand=hand, emb=emb, clip=cid), ds, hand, hmap)
    _link_visualization(dsdir, c, ds, hand, emb, task, cid)
    c["ingested_to"] = str(_dest("retarget", ds, hand=hand, emb=emb, clip=cid))
    done["clip_skipped_same" if skipped else "clip"] += 1


def _clip_plan(c: dict, pairs: list, ds: str, hand: str, hmap: dict, force: bool, supersede: str) -> tuple[list, bool]:
    """(plan [(src, dst, None | quarantine path | "rm")], skipped: a layer already holds the same content).
    Decide first, write afterwards: a rejected pair appends to c["reject"] and stops the plan."""
    skipped = False
    plan = []
    for src, dst, hand_layer in pairs:
        if not dst.exists():
            plan.append((src, dst, None))
            continue
        want = _ingested_fingerprint(src, dst, ds, hand, hmap) if hand_layer else tree_fingerprint(src)
        if tree_fingerprint(dst) == want:
            skipped = True
            continue
        if supersede and hand_layer:
            arch = _superseded(ds, dst, supersede)
            if arch.exists() or arch.is_symlink():
                c["reject"].append(f"quarantine slot already exists {arch} — not overwritten")
                break
            plan.append((src, dst, arch))
        elif force:
            plan.append((src, dst, "rm"))
        else:
            note = "" if hand_layer else " · the hand-independent shared layer is not moved even with --supersede-label"
            c["reject"].append(
                f"content differs from the already accepted {dst} (not overwritten without --force{note})"
            )
            break
    return plan, skipped


def _link_visualization(dsdir: Path, c: dict, ds: str, hand: str, emb: str, task: str, cid: str) -> None:
    """Hard-link (else copy) the clip's preview video into the visualization layer when one exists."""
    vis = dsdir / "visualization" / emb / task / "0.mp4"
    if not vis.exists():
        alt = sorted((dsdir / c["hand_dir"] / emb / task / "0").glob("visualization_kinematic.mp4"))
        vis = alt[0] if alt else vis
    if vis.exists():
        vdst = _dest("visualization", ds, hand=hand, emb=emb) / f"{cid}.mp4"
        if not vdst.exists():
            vdst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(vis, vdst)
            except OSError:
                shutil.copy2(vis, vdst)


def _copy_provenance(root: Path, ds: str, handoff_id: str) -> str:
    """Ledger + provenance copies of the handoff under the dataset's intake dir -> the ledger path."""
    idir = _dest("intake", ds)
    idir.mkdir(parents=True, exist_ok=True)
    prov = idir / handoff_id
    prov.mkdir(exist_ok=True)
    for name in ("README.md", "EXCLUDED.md", "EXCLUDED.json", "MANIFEST.sha256", "MANIFEST.json"):
        if (root / name).exists():
            shutil.copy2(root / name, prov / name)
    if (root / "reports").is_dir():
        shutil.copytree(root / "reports", prov / "reports", dirs_exist_ok=True)
    return str(idir / f"{handoff_id}.json")


def write_ledgers(
    root: Path, rep: Report, info: dict, handoff_id: str, ingested: dict | None, report_dir: Path | None
) -> list[Path]:
    written = []
    for ds, dinfo in info.items():
        clips = {k: v for k, v in rep.clips.items() if v["dataset"] == ds}
        acc = [k for k, v in clips.items() if not v["reject"]]
        rec = {
            "handoff_id": handoff_id,
            "source_root": str(root),
            "dataset": ds,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": "dexengine/data/intake.py",
            "layout_module": "dexengine/core/layout.py",
            "dataset_reject": rep.ds_reject.get(ds, []),
            "manifest": dinfo.get("manifest"),
            "archive": ARCHIVE,
            "hand_assets": dinfo.get("assets"),
            "n_clips": len(clips),
            "n_accept": len(acc),
            "n_reject": len(clips) - len(acc),
            "by_emb": dict(Counter(v["emb"] for v in clips.values())),
            "ingested": (ingested or {}).get("counts") if ingested else None,
            "clips": clips,
        }
        targets = []
        if ingested and not rep.ds_reject.get(ds):
            targets.append(_dest("intake", ds) / f"{handoff_id}.json")
        if report_dir:
            targets.append(report_dir / f"{handoff_id}.{ds}.json")
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            written.append(t)
    return written
