"""Checks of a delivered retargeting handoff: file digests, the archive, hand assets, every clip (keys, files, frame
counts, scene references), scene loading, and the summary of a check run."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from dexengine.core import paths as P  # noqa: E402  — the owner of clip_id syntax and normalisation
from dexengine.core.hashing import sha256_file

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import mujoco
except ImportError:
    mujoco = None


dl = P.data_layout  # noqa: E402  — the owner of the layout · dataset · hand-asset tables
MAX_T_DIFF = 5  # allowed mano ↔ kinematic frame-count gap (delivery README: ~3, from filters)
REQ_TI_KEYS = ("task", "dataset_name", "robot_type", "embodiment_type", "data_id", "ref_dt")
REQ_KIN_FILES = ("task_info.json", "scene.xml", "0/trajectory_kinematic.npz")
_FILE_RE = re.compile(r'file="([^"]+)"')
_MESHDIR_RE = re.compile(r'meshdir="([^"]+)"')


class Report:
    def __init__(self):
        self.ds_reject: dict[str, list[str]] = defaultdict(list)  # dataset-level rejection reasons
        self.clips: dict[str, dict] = {}  # key "<ds>/<emb>/<task>"
        self.warnings: list[str] = []

    def clip(self, ds, emb, task) -> dict:
        k = f"{ds}/{emb}/{task}"
        if k not in self.clips:
            self.clips[k] = {"dataset": ds, "emb": emb, "source_task": task, "reject": [], "warn": []}
        return self.clips[k]


def verify_archive(path: Path) -> dict:
    """If the handoff arrived as **one .tar.zst**, the zstd frame checksum (`zstd -t`) is itself the transfer-integrity
    evidence — equivalent to a per-file MANIFEST. A failed check raises (it is never passed over silently).
    The archive sha256 stays in the ledger as provenance."""
    r = subprocess.run(["zstd", "-t", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[intake] --archive check failed: {path} — {r.stderr.strip()[:200]}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "zstd_test": "ok"}


# ── D2 hand asset ────────────────────────────────────────────────────────────────────
def verify_hand_asset(dsdir: Path, hand: str, ds: str, rep: Report) -> dict:
    adir = dsdir / "assets" / "robots" / hand
    if not adir.is_dir():
        rep.ds_reject[ds].append(
            f"hand asset missing: {ds}/assets/robots/{hand} — cannot resolve robots/{hand}/assets that "
            "scene.xml points to"
        )
        return {"present": False}
    try:
        a = dl.identify_asset(str(adir))
    except (KeyError, FileNotFoundError) as e:
        rep.ds_reject[ds].append(str(e))
        return {
            "present": True,
            "registered": False,
            "sha256_xml3": dl.asset_hash(str(adir))
            if all((adir / n).exists() for n in ("right.xml", "left.xml", "bimanual.xml"))
            else None,
        }
    if a.status == "rejected":
        rep.ds_reject[ds].append(
            f"hand asset {hand} hashes equal to a **rejected asset** '{a.spec_dir}': {a.provenance}"
        )
    if a.spec_dir != hand:
        rep.warnings.append(
            f"[{ds}] hand directory name {hand!r} ≠ registered name {a.spec_dir!r} (identified by hash) — "
            "intake uses the registered name"
        )
    return {
        "present": True,
        "registered": True,
        "spec_dir": a.spec_dir,
        "status": a.status,
        "sha256_xml3": a.sha256_xml3,
        "isaac_hand_id": a.isaac_hand_id,
    }


# ── C1~C6 clip checks ────────────────────────────────────────────────────────────────
def _scene_nq(scene: Path) -> tuple[int | None, str]:
    """The nq of scene.xml. Counts only the joints **under worldbody** — `<default><joint …/>` is a default,
    not a joint (string counting would put every clip +1 because of that one line). hinge/slide 1 · ball 4 ·
    free(freejoint) 7.
    If mujoco imports, compare against the compiled nq and return the discrepancy when they differ (missing assets
    are caught by C3)."""
    import xml.etree.ElementTree as ET

    try:
        wb = ET.parse(scene).getroot().find("worldbody")
    except ET.ParseError as e:
        return None, f"xml parse failed: {e}"
    if wb is None:
        return None, "no worldbody"
    n = 0
    for j in wb.iter("joint"):
        t = j.get("type", "hinge")
        n += 7 if t == "free" else 4 if t == "ball" else 1
    n += 7 * sum(1 for _ in wb.iter("freejoint"))
    how = "xml"
    if mujoco is not None:
        try:
            m = int(mujoco.MjModel.from_xml_path(str(scene)).nq)
            if m != n:
                return None, f"xml nq {n} ≠ mujoco nq {m}"
            how = "xml+mujoco"
        except Exception as e:  # noqa: BLE001  any compile error only downgrades the check to the xml count
            how = f"xml (mujoco compile failed: {str(e)[:60]})"
    return n, how


def _check_task_info(c: dict, ti: dict, ds: str, hand: str, emb: str) -> None:
    # C1
    for k in REQ_TI_KEYS:
        if k not in ti:
            c["reject"].append(f"task_info key missing: {k}")
    if ti.get("dataset_name") != ds:
        c["reject"].append(f"task_info.dataset_name={ti.get('dataset_name')!r} ≠ path {ds!r}")
    if ti.get("robot_type") != hand:
        c["reject"].append(f"task_info.robot_type={ti.get('robot_type')!r} ≠ hand directory {hand!r}")
    if ti.get("embodiment_type") != emb:
        c["reject"].append(f"task_info.embodiment_type={ti.get('embodiment_type')!r} ≠ path {emb!r}")
    if emb not in dl.EMBODIMENTS:
        c["reject"].append(f"embodiment {emb!r} must be one of {dl.EMBODIMENTS}")
    if abs(float(ti.get("ref_dt", 0)) - 0.02) > 1e-9:
        c["reject"].append(f"ref_dt={ti.get('ref_dt')} ≠ 0.02 (50 Hz contract)")


def _check_objects(c: dict, ti: dict, dsdir: Path, ds: str, emb: str) -> None:
    # C2 objects
    objs = []
    for side in ("right", "left"):
        d = ti.get(f"{side}_object_mesh_dir")
        if d:
            objs.append((side, d))
    objs.extend(("extra", ex.get("mesh_dir", "")) for ex in ti.get("extra_objects") or [])
    obj_ids = []
    for side, d in objs:
        m = re.fullmatch(rf"processed/{re.escape(ds)}/assets/objects/([^/]+)/?", d or "")
        if not m:
            c["reject"].append(
                f"{side} object path outside the convention: {d!r} (processed/{ds}/assets/objects/<obj>)"
            )
            continue
        oid = m.group(1)
        obj_ids.append(oid)
        if not (dsdir / "assets" / "objects" / oid / "visual.obj").exists():
            c["reject"].append(f"object asset missing: {ds}/assets/objects/{oid}/visual.obj")
    c["objects"] = obj_ids
    if emb == "bimanual" and not ti.get("right_object_mesh_dir") and not ti.get("left_object_mesh_dir"):
        c["warn"].append("bimanual but neither main object is present")


def _check_scene(c: dict, kdir: Path) -> tuple[Path, list]:
    # C3 scene.xml references
    scene = kdir / "scene.xml"
    xml = scene.read_text(errors="replace")
    md = _MESHDIR_RE.search(xml)
    meshdir = (scene.parent / md.group(1)).resolve() if md else scene.parent
    missing = [f for f in _FILE_RE.findall(xml) if not (meshdir / f).exists()]
    if missing:
        c["reject"].append(f"scene.xml references missing: {len(missing)} (e.g. {missing[:2]}) — meshdir={meshdir}")
    return scene, missing


def _check_trajectory(c: dict, kdir: Path, scene: Path, missing: list):
    # C4 trajectory
    T = None
    if np is not None:
        try:
            z = np.load(kdir / "0/trajectory_kinematic.npz")
            q = z["qpos"]
            if q.ndim != 2:
                c["reject"].append(f"qpos ndim={q.ndim}")
            else:
                T = int(q.shape[0])
                c["T"] = T
                c["nq"] = int(q.shape[1])
                if not np.isfinite(q).all():
                    c["reject"].append("non-finite values in qpos")
                if T < 2:
                    c["reject"].append(f"frame count {T}")
                if not missing:
                    nq, how = _scene_nq(scene)
                    if nq is None:
                        c["warn"].append(f"nq cannot be checked: {how}")
                    elif nq != q.shape[1]:
                        c["reject"].append(f"qpos width {q.shape[1]} ≠ scene.xml nq {nq} ({how})")
                    c["nq_check"] = how
        except Exception as e:  # noqa: BLE001  any read error of a delivered file rejects the clip with its text
            c["reject"].append(f"trajectory_kinematic.npz read failed: {e}")
    return T


def _check_mano(c: dict, dsdir: Path, emb: str, task: str, T) -> None:
    # C5 mano
    mdir = dsdir / "mano" / emb / task
    if not (mdir / "0/trajectory_keypoints.npz").exists():
        c["reject"].append(f"mano tree missing: mano/{emb}/{task}/0/trajectory_keypoints.npz")
    elif np is not None and T is not None:
        try:
            kp = np.load(mdir / "0/trajectory_keypoints.npz")
            key = "qpos_wrist_right" if "qpos_wrist_right" in kp.files else kp.files[0]
            Tm = int(kp[key].shape[0])
            c["T_mano"] = Tm
            if abs(Tm - T) > MAX_T_DIFF:
                c["reject"].append(f"mano T={Tm} vs kinematic T={T} differ by {abs(Tm - T)} > {MAX_T_DIFF}")
        except Exception as e:  # noqa: BLE001  any read error of a delivered file rejects the clip with its text
            c["reject"].append(f"trajectory_keypoints.npz read failed: {e}")


def _check_visualization(c: dict, dsdir: Path, kdir: Path, emb: str, task: str) -> None:
    # V1
    vis = dsdir / "visualization" / emb / task / "0.mp4"
    if not vis.exists():  # alternative delivery layout: 0/visualization_*.mp4 in clip
        alt = sorted((kdir / "0").glob("visualization_*.mp4"))
        vis = alt[0] if alt else vis
    c["visualization"] = vis.exists()
    if not vis.exists():
        c["warn"].append("no visualization")


def _check_clip_id(c: dict, task: str) -> None:
    # C6 clip_id
    try:
        c["clip_id"] = dl.clip_id_from_task(task, c.get("data_id", 0), P.normalize_field)
        if not P.is_clip_id(c["clip_id"]):
            c["reject"].append(f"derived clip_id breaks the grammar: {c['clip_id']!r}")
    except ValueError as e:
        c["reject"].append(f"clip_id derivation failed: {e}")


def check_clip(dsdir: Path, ds: str, hand: str, emb: str, task: str, rep: Report) -> dict:
    c = rep.clip(ds, emb, task)
    kdir = dsdir / hand / emb / task
    for rel in REQ_KIN_FILES:
        if not (kdir / rel).exists():
            c["reject"].append(f"required file missing: {hand}/{emb}/{task}/{rel}")
    if c["reject"]:
        return c
    ti = json.loads((kdir / "task_info.json").read_text())
    _check_task_info(c, ti, ds, hand, emb)
    c["data_id"] = int(ti.get("data_id", 0))
    _check_objects(c, ti, dsdir, ds, emb)
    scene, missing = _check_scene(c, kdir)
    T = _check_trajectory(c, kdir, scene, missing)
    _check_mano(c, dsdir, emb, task, T)
    _check_visualization(c, dsdir, kdir, emb, task)
    _check_clip_id(c, task)
    return c


# ── Intake ──────────────────────────────────────────────────────────────────────────
def link_tree(src: Path, dst: Path) -> None:
    """Hard-link copy (same FS) — a plain copy when that fails. dst must not already exist."""
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # the stage (re-retargeting) mano/assets are symlinks back into the handoff — admit the **real files**, not the link
    src = src.resolve()
    r = subprocess.run(["cp", "-al", str(src), str(dst)], capture_output=True, text=True)
    if r.returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=False)


def tree_fingerprint(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


SCENE_XMLS = ("scene.xml", "scene_eq.xml")
RE_ROBOTS = re.compile(r"robots/([A-Za-z0-9_\-]+)/")
_RE_DIRATTR = re.compile(r'(meshdir|texturedir)="([^"]*)"')


def fixed_scene(s0: str, rel: str, hand_map: dict[str, str]) -> str:
    """Text rewrite of fix_scene_refs (pure): meshdir/texturedir = rel · robots/<delivered name>/ → robots/<spec>/."""
    s = _RE_DIRATTR.sub(lambda m: f'{m.group(1)}="{rel}"', s0)
    for dlv, spec in hand_map.items():
        if dlv != spec:
            s = s.replace(f"robots/{dlv}/", f"robots/{spec}/")
    return s


def _load_scene(xml: str) -> tuple[str, str]:
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(xml)
        return xml, f"ok nq={m.nq}"
    except Exception as e:  # noqa: BLE001 — the failure reason is kept in the ledger
        return xml, f"FAIL {str(e)[:240]}"


def verify_scene_loads(clip_dirs: list[Path], workers: int | None) -> dict:
    """Really open scene.xml and scene_eq.xml of every clip with mujoco (D3) on a pool sized by capacity (`workers` =
    an explicit option). Passes on an exact count match only."""
    from multiprocessing import Pool  # noqa: PLC0415

    from dexengine.core import capacity  # noqa: PLC0415

    xmls = [str(d / n) for d in clip_dirs for n in SCENE_XMLS if (d / n).exists()]
    step = "dexengine.data.handoff.checks"  # capacity.NEEDS key (this module)
    with capacity.sized(step, len(xmls), workers) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        res = pool.map(_load_scene, xmls, chunksize=8)
    fails = {x: r for x, r in res if not r.startswith("ok")}
    return {"n_xml": len(xmls), "n_ok": len(xmls) - len(fails), "fails": fails}


def summarize(rep: Report, info: dict) -> str:
    lines = []
    for ds, dinfo in info.items():
        clips = {k: v for k, v in rep.clips.items() if v["dataset"] == ds}
        acc = sum(1 for v in clips.values() if not v["reject"])
        emb = Counter(v["emb"] for v in clips.values())
        head = (
            f"[{ds}] clips {len(clips)} (ACCEPT {acc} · REJECT {len(clips) - acc}) · emb {dict(emb)} · hands "
            f"{dinfo.get('hands')}"
        )
        if rep.ds_reject.get(ds):
            head += "  ★ dataset REJECT"
        lines.append(head)
        lines.extend(f"    ★ {r}" for r in rep.ds_reject.get(ds, []))
        m = dinfo.get("manifest") or {}
        if m.get("present"):
            lines.append(
                f"    manifest: listed {m['listed']} · mismatched {m['bad']} · missing {m['missing']} · "
                f"unlisted {m['unlisted']}"
            )
        for h, a in (dinfo.get("assets") or {}).items():
            lines.append(
                f"    hand asset {h}: "
                + (
                    f"{a.get('status')} → {a.get('spec_dir')} (Isaac id {a.get('isaac_hand_id')}) sha "
                    f"{str(a.get('sha256_xml3'))[:12]}"
                    if a.get("registered")
                    else f"unregistered/missing {a}"
                )
            )
        reasons = Counter(r.split(":")[0].split("(")[0].strip() for v in clips.values() for r in v["reject"])
        for r, n in reasons.most_common(8):
            lines.append(f"    reject reason ×{n}: {r}")
        warns = Counter(w.split(":")[0] for v in clips.values() for w in v["warn"])
        for w, n in warns.most_common(4):
            lines.append(f"    warning ×{n}: {w}")
    lines.extend(f"  ! {w}" for w in rep.warnings[:10])
    return "\n".join(lines)
