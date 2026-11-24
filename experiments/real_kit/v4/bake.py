"""v4 gen's bake = the v2 bake files used as they are — no re-bake.

v4 variants have the placements of variants the v2 dense chain already baked (same judge cell, dx, dy, yaw). A re-bake
gives IK floating-point differences between machines (up to 9 mrad, wrist 2.2 mm) and departs from the reference the
policy saw, so the files the bake builds (BAKE_TREE_FILES, arm IK, wrist cache of both hands) are copied byte for byte
from the v2 pair. v2 pair (bake_pair): k ≠ 0 = the same name in the v2 shadow (--bake-from) · k = 0 (canonical cell =
identity) = <clip>_pl_rt of the canonical root (MAIN) (the v2 dense chain does not bake k 0). A missing pair dies (no
fallback to a re-bake). Support stamping (stamp) and aliases (alias_base) follow the bake path (task_info as in v2); the
contact time axis is a later step (after_bake). Oracle (sha_oracle, called by `run gen`): the chosen variants' files
have
the v2 pair's sha256 — else rc 7. Marker [AUG_V4_BAKE] taken · oracle."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.record.augment import bake as AB
from dexengine.record.augment import bake_run as ABR

TAG = "[AUG_V4_BAKE]"
#: files the bake steps (bake.step_specs body, xr, retime) build in variant tree 0/ · contacts (contact_mano_*) come in
#: after_bake
BAKE_TREE_FILES = (
    "body_positions.npz",
    "trajectory_ikrollout.npz",
    "trajectory_kinematic.npz",
    "trajectory_kinematic.pre_rt.npz",
    "trajectory_kinematic.pre_xr.npz",
    "trajectory_kinematic.pre_xr.meta.json",
    "xr_report.json",
)
#: must exist — clips that skipped xr have no pre_xr or xr_report (only what the pair has is taken)
BAKE_TREE_MUST = (
    "body_positions.npz",
    "trajectory_ikrollout.npz",
    "trajectory_kinematic.npz",
    "trajectory_kinematic.pre_rt.npz",
)


def bake_pair(src, shadow: Path, v2: Path, main: Path, name: str, k: int) -> list[tuple[Path, Path]]:
    """One variant's bake files → [(v4 shadow path, v2 pair path)]. A missing pair file dies (required, both hands'
    caches).
    """
    clip = src.tree_name
    sname = clip if int(k) == 0 else name
    base = Path(main) if int(k) == 0 else Path(v2)
    out = []
    tsrc = base / src.trees_rel / sname / "0"
    for fn in BAKE_TREE_FILES:
        f = tsrc / fn
        if not f.is_file():
            if fn in BAKE_TREE_MUST:
                raise SystemExit(f"{TAG} {name}: v2 pair {f} missing — never baked again")
            continue
        out.append((Path(shadow) / src.trees_rel / name / "0" / fn, f))
    for s in ("right", "left"):
        for rel in (src.ik_rel[s], src.wrist_rel[s]):
            f = base / rel / f"{sname}.npz"
            if not f.is_file():
                raise SystemExit(f"{TAG} {name}: v2 pair {f} missing — never baked again")
            out.append((Path(shadow) / rel / f"{name}.npz", f))
    return out


def adopt_all(src, shadow: Path, recs: list[dict], v2: Path, main: Path, log_dir: Path) -> dict:
    """For every variant with status written, copy the bake files from the v2 pair → status baked (bake = the taken
    record). In place of bake_run.bake_all (v4 only, place_gen --bake-from)."""
    from dexengine.placement.bake import chain as SR

    todo = [r for r in recs if r["status"] == "written"]
    if not todo:
        return {
            "boot_s": 0.0,
            "n": 0,
            "adopted": str(v2),
            "active": AB.clip_ctx(src, main)["active"],
            "log_dir": str(log_dir),
        }
    t0 = time.time()
    for r in todo:
        ABR.clean_outputs(src, shadow, r["name"])
        r["alias"] = AB.alias_base(src, shadow, r["name"])
    from dexengine.record.augment import bake_post as ABP

    ctx = AB.clip_ctx(src, main)
    troot = Path(shadow) / src.trees_rel
    ABP.worker(AB.bake_env(src, ctx["active"], shadow), "stamp", [str(troot / r["name"]) for r in todo], log_dir)
    nf = 0
    for r in todo:
        pairs = bake_pair(src, shadow, v2, main, r["name"], int(r["k"]))
        for dst, f in pairs:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or (dst.exists() and not dst.is_file()):
                raise SystemExit(f"{TAG} {dst} is not a real file location (canonical symlink — name clash)")
            shutil.copyfile(f, dst)
        nf += len(pairs)
        tree = troot / r["name"]
        ikf = {s: Path(shadow) / src.ik_rel[s] / f"{r['name']}.npz" for s in ("right", "left")}
        nr, nl, nq = SR.halfwrite_lengths(ikf["right"], ikf["left"], tree / "0/trajectory_kinematic.npz")
        b = {"adopted": "canonical" if int(r["k"]) == 0 else str(v2), "files": len(pairs), "T": nq}
        if min(nr, nl, nq) < 0 or not nr == nl == nq:
            r.update(status="reject", why="length", detail=f"R={nr} L={nl} qpos={nq}", bake=b)
        else:
            r.update(status="baked", bake=b)
    wall = round(time.time() - t0, 1)
    print(
        f"{TAG} imported {src.tree_name}: variants {len(todo)} · files {nf} · source v2 {v2} (k 0 = "
        f"canonical {main}) · "
        f"passed {sum(r['status'] == 'baked' for r in todo)} · {wall} s — not baked again",
        flush=True,
    )
    return {
        "boot_s": 0.0,
        "wall_s": wall,
        "n": len(todo),
        "adopted": str(v2),
        "files": nf,
        "active": ctx["active"],
        "chunk": ctx["tag"],
        "log_dir": str(log_dir),
    }


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_oracle(src, shadow: Path, v2: Path, main: Path, m: dict, ks: list) -> tuple[int, list[str]]:
    """Independent oracle: are the chosen variants' (ks, ok | spare) bake files sha256-equal to the v2 pair → (equal
    file count, differences).
    """
    same, bad = 0, []
    for v in m["variants"]:
        if int(v["k"]) not in ks or v["status"] not in ("ok", "spare"):
            continue
        for dst, f in bake_pair(src, shadow, v2, main, v["name"], int(v["k"])):
            if not dst.is_file():
                bad.append(f"{dst.relative_to(shadow)} missing")
            elif _sha(dst) != _sha(f):
                bad.append(f"{dst.relative_to(shadow)} ≠ {f}")
            else:
                same += 1
    return same, bad


def layout(hand: str, ds: str, emb: str, clip: str, arm: str):
    """The pair path layout on the oracle side (run.py) — only the layer names of place_gen.Src (rel_layers,
    tree_combo).
    """
    import types

    from dexengine.record.augment import place_src as APS
    from dexengine.record.augment import scene as AA

    trees_rel, wrist_rel, ik_rel, _pl = APS.rel_layers(ds, AA.tree_combo(arm, hand, ds, emb), emb)
    return types.SimpleNamespace(tree_name=P.tree_name(clip), trees_rel=trees_rel, wrist_rel=wrist_rel, ik_rel=ik_rel)
