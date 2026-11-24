"""Stage 2e input identity and freshness: the trees a run covers, the qpos hash, the input fingerprint (rule version and
constants, desk tolerance, arm colliders, palm offset, finger-range rule source, contact sidecars), the resumable
pieces and the freshness count. The stage (repair/wrist_desk_stage.py) writes; this module only reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from dexengine.core.repo import REPO_ROOT
from dexengine.placement.repair import wrist_desk as WDS

_FP_CACHE: dict = {}


def stage_trees(ds: str, spec_dir: str, ls=None) -> list:
    from dexengine.data.support import gap as SG  # noqa: PLC0415 — the base-tree list of the stage-1 columns

    return SG._base_trees(ds, spec_dir, ls)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def qpos_sha(tree: Path) -> str:
    with np.load(Path(tree) / "0/trajectory_kinematic.npz") as z:
        return _sha_bytes(np.ascontiguousarray(z["qpos"]).tobytes())[:24]


def contact_file(ds: str, ti: dict, clip: str, side: str, spec: str):
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.contact import fingerprint as SF  # noqa: PLC0415

    if not SF.objects_for(ti, side, SF.LAYER_ALL):
        return None
    return P.sidecar_file(SF.LAYER_ALL, ds, clip, side, spec)


def input_fp(ds: str, tree: Path, ti: dict, arm: str, hand: str) -> str:
    """What the result depends on besides the original qpos: rule version and constants, the judge's desk tolerance,
    the arm colliders (sim_collider digest), the palm offset, the finger-range rule source (prior.py · prior yaml), the
    contact sidecars' bytes."""
    from dexengine.core import hands as HS  # noqa: PLC0415
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.stage1.plan import prior_file  # noqa: PLC0415 — the prior file's owner
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import sim_collider as SC  # noqa: PLC0415

    k = (arm, hand)
    if k not in _FP_CACHE:
        rule = [REPO_ROOT / "dexengine/retarget/ik/prior.py", prior_file(P.hand_spec_dir(hand))]
        _FP_CACHE[k] = {
            "v": WDS.CODE_VERSION,
            "c": WDS.CONSTS,
            "tol": SCL.DESK_SURF_PEN_MM,
            "arm": SC.measured_digest(arm),
            "palm_off": list(HS.HAND_REGISTRY[hand].palm_frame_offset_pos),
            "range": [_sha_bytes(f.read_bytes())[:16] if f.is_file() else "ABSENT" for f in rule],
        }
    cs = {}
    for side in ti.get("hands") or []:
        f = contact_file(ds, ti, Path(tree).name, side, P.hand_spec_dir(hand))
        cs[side] = "-" if f is None else (_sha_bytes(f.read_bytes())[:16] if f.is_file() else "ABSENT")
    return _sha_bytes(json.dumps({**_FP_CACHE[k], "contact": cs}, sort_keys=True).encode())[:24]


def pieces(ds: str, spec_dir: str, keep=None) -> dict:
    """The segment pieces present in this hand's base layer -> {piece tree: parent tree} (the parent may be absent).
    The statement is the units registry (owner data.segment.split, address paths.units_registry) — no name rule here.
    `keep(path)` (a caller's scope) limits which registries are read and which piece trees are looked at."""
    from dexengine.core import paths as P  # noqa: PLC0415

    out = {}
    for emb in P.data_layout.EMBODIMENTS:
        reg, root = P.units_registry(ds, emb), P.layer_path("base", ds, hand=spec_dir, emb=emb)
        if keep is not None and not keep(reg):
            continue
        for clip, units in (json.loads(reg.read_text())["parents"] if reg.is_file() else {}).items():
            out.update(
                {
                    root / u["name"]: root / clip
                    for u in units
                    if (keep is None or keep(root / u["name"])) and (root / u["name"] / "task_info.json").is_file()
                }
            )
    return out


def read_record(d: Path) -> dict:
    return (json.loads((d / "task_info.json").read_text()).get("rules") or {}).get(WDS.RULE_KEY) or {}


def fresh_record(ds: str, d: Path, hand: str, arm_env: str) -> dict | None:
    """A solved tree's stamp when it matches the current inputs and qpos, else None."""
    try:
        ti = json.loads((d / "task_info.json").read_text())
        rec = (ti.get("rules") or {}).get(WDS.RULE_KEY) or {}
        arm = rec.get("arm", "")
        ok = (
            arm
            and (not arm_env or arm_env == arm)
            and rec.get("fp") == input_fp(ds, d, ti, arm, hand)
            and rec.get("qpos_sha") == qpos_sha(d)
        )
        return rec if ok else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        return None


def follow_pieces(ds: str, hand: str, pcs: dict, solved: dict) -> tuple[int, int, list]:
    """The pieces of the solved parents follow them (write mode): the piece owner re-cuts every piece that is no longer
    its parent's window (split.reslice_stale — quarantine + units_replaced), then each piece takes its parent's stamp.
    ``solved`` = {parent tree: True when its stamp was written} -> (re-cut, stamped, [(piece, why)] unstamped)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    if os.environ.setdefault("DEXENGINE_HAND", hand) != hand:  # data.segment.split reads the hand at import
        raise SystemExit(f"[WRIST_SLACK] ✗ DEXENGINE_HAND={os.environ['DEXENGINE_HAND']} != --hand {hand}")
    from dexengine.data.segment import (
        split as SS,  # noqa: PLC0415 — the piece owner (registry · re-cut · units_replaced)
    )

    mine = {d: par for d, par in pcs.items() if par in solved}
    label, recut = f"wrist_slack_{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 10**9:09d}", 0
    for emb in P.data_layout.EMBODIMENTS:
        reg = P.units_registry(ds, emb)
        parents = {par.name for d, par in mine.items() if d.parent.name == emb and solved[par]}
        table = {
            c: u for c, u in (json.loads(reg.read_text())["parents"] if reg.is_file() else {}).items() if c in parents
        }
        if table:
            recut += SS.reslice_stale(ds, emb, table, write=True, label=label)["resliced"]
    from dexengine.data.tree_io import write_json_if_changed  # noqa: PLC0415 — base task_info writer

    stamped, missed = 0, []
    for d, par in pcs.items():
        if par not in solved:
            if not (par / "task_info.json").is_file():
                missed.append((d.name, f"parent {par.name} is not in the base layer"))
            continue
        if not solved[par]:
            missed.append((d.name, f"parent {par.name} failed"))
            continue
        ti = json.loads((d / "task_info.json").read_text())
        ti.setdefault("rules", {})[WDS.RULE_KEY] = read_record(par)
        write_json_if_changed(d / "task_info.json", ti)
        stamped += 1
    return recut, stamped, missed
