"""SPIDER + arm rows (spider_arm.data): one variant tree's files rewritten with SPIDER's rows — trajectory,
ikrollout, the human contact sidecars on the SPIDER time axis and task_info e5_spider."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.policies.spider_arm.spider_load import FREQ_REF, MARK

# ── one variant
# tree
# ──────────────────────────────────────────────────────────────────────────────────────────────────
STRIP_KEYS = ("retimed_from", "retime_safety", "retime_ik_suffix", "retime_v2", "trimmed_tail_frames", "f0_policy")


# ↑ statements about the kin retime/trim this tree no longer carries (their readers are bake-time tools). KEPT: the
# placement
#   record xy_shift_traj · yaw_deg · pivot_traj (coords.tree_placement — the tree frame; SPIDER rows get the same
# rigid_pl).
# f0_prepad is NOT stripped but set to 0: coords.tree_f0_pad falls back to F0_REPAIR.json when the key is absent


def contacts_from_orig(orig: Path, tdir: Path, T: int) -> list:
    """Human (MANO) contact sidecars on the SPIDER time axis: SPIDER frame k = base frame k = the orig tree's pre-retime
    index k + its f0 pad P → the orig (retimed) tree row whose retime_src_idx is nearest
    (dexengine.record.augment.bake_post.nearest_frames
    —
    the owner of that rule). Rows of every per-frame array are re-indexed, as
    dexengine.record.augment.bake_post.remap_contacts does."""
    from dexengine.placement.coords import core as CO  # noqa: PLC0415
    from dexengine.record.augment import bake_post as ABP  # noqa: PLC0415

    zo = np.load(orig / "0/trajectory_kinematic.npz")
    Tc = len(zo["qpos"])
    P_ = int(CO.tree_f0_pad(orig, json.loads((orig / "task_info.json").read_text())))
    b = np.arange(T, dtype=np.float64) + P_
    j = (
        ABP.nearest_frames(zo["retime_src_idx"], b)
        if "retime_src_idx" in zo.files
        else np.clip(b.astype(np.int64), 0, Tc - 1)
    )
    done = []
    for f in sorted((orig / "0").glob("contact_mano_*.npz")):
        if f.name.count(".") > 1:
            continue
        cz = dict(np.load(f, allow_pickle=False))
        rows = [k for k, x in cz.items() if np.ndim(x) >= 1 and np.shape(x)[0] == Tc]
        if "contact" not in rows:
            raise SystemExit(f"{MARK} {f}: contact row count ≠ source T {Tc}")
        for k in rows:
            cz[k] = np.asarray(cz[k])[j]
        np.savez(tdir / "0" / f.name, **cz)
        done.append(f.name)
    return done


def _spider_npz(name: str, ti: dict, zk: dict, qv: np.ndarray, lay_t, T: int) -> dict:
    """The tree's trajectory npz: qpos = qpos_rollout = SPIDER, articulation streams, zero qvel and contact arrays."""
    out = {
        "qpos": qv.astype(zk["qpos"].dtype),
        "qpos_rollout": qv.astype(zk["qpos"].dtype),
        "frequency": np.float64(FREQ_REF),
    }
    from dexengine.record.store import meta as CM  # noqa: PLC0415  — owner of the xr repair trace prefix

    # articulation angle stream (npz key = the object's articulation.npz_key — owner base_adapter) = its qpos column
    arti = {}
    for o in ti.get("objects") or []:
        art = o.get("articulation")
        if art:
            if not art.get("npz_key"):
                raise SystemExit(f"{MARK} {name}: articulated object {o.get('name')} has no npz_key")
            arti[art["npz_key"]] = int(art["qpos_index"])
    for k in zk:
        if k in arti:
            out[k] = qv[:, [arti[k]]].astype(np.asarray(zk[k]).dtype)
        elif k.startswith((CM.XR_SHIFT_PREFIX, "retime")):
            pass  # our xr repair / retime traces — neither is applied to SPIDER rows, so they are not carried
        elif k not in out and k not in ("qvel", "contact", "contact_pos"):
            raise SystemExit(f"{MARK} {name}: unknown key {k} in the kin tree npz — no rule says whether to carry it")
    if set(arti) - set(zk):
        raise SystemExit(
            f"{MARK} {name}: articulated-object npz keys {sorted(set(arti) - set(zk))} are not in the kin tree"
        )
    nv = int(lay_t.nv)
    out["qvel"] = np.zeros((T, nv))
    out["contact"] = np.zeros((T,) + tuple(np.asarray(zk["contact"]).shape[1:])) if "contact" in zk else None
    out["contact_pos"] = (
        np.zeros((T,) + tuple(np.asarray(zk["contact_pos"]).shape[1:])) if "contact_pos" in zk else None
    )
    return {k: v for k, v in out.items() if v is not None}


def write_spider_tree(name, tdir, orig, ti, zk, qv, lay_t, T, sp, sp_sides, other, d16) -> None:
    """Tree files: trajectory (qpos = qpos_rollout = SPIDER) · ikrollout · drop the kin pre_* copies and kin-time
    sidecars · task_info e5_spider."""
    out = _spider_npz(name, ti, zk, qv, lay_t, T)
    for fn in (
        "0/trajectory_kinematic.pre_xr.npz",
        "0/trajectory_kinematic.pre_rt.npz",
        "0/trajectory_kinematic.pre_xr.meta.json",
        "0/xr_report.json",
        "0/contact_mano_left.npz",
        "0/contact_mano_right.npz",
        "0/body_positions.npz",
        "F0_REPAIR.json",
    ):
        p = tdir / fn
        if p.is_symlink() or p.is_file():
            p.unlink()
    np.savez(tdir / "0/trajectory_kinematic.npz", **out)
    np.savez(tdir / "0/trajectory_ikrollout.npz", qpos=out["qpos_rollout"])
    for k in STRIP_KEYS:
        ti.pop(k, None)
    ti["f0_prepad"] = 0
    cdone = contacts_from_orig(orig, tdir, T)
    ti["e5_spider"] = {
        "src": sp["meta"],
        "spider_sides": sp_sides,
        "kin_idle_block_sides": other,
        "d16_idle_sides_spider_had": d16,
        "T": T,
        "kin_T": len(zk["qpos"]),
        "transforms": [
            "make_combo_variants.rigid_pl(adopted dx dy yaw pivot of " + orig.name + ")",
            "aug_scene.rigid_any(scene_to_tree_vec(aug_place dx/dy_scene), yaw_deg, pivot_tree)",
        ],
        "not_applied": ["xr cross repair", "retime"],
        "kept_as_kin": ["D16 idle side (task_info idle_sides): kin constant idle hand block + idle arm"],
        "qvel": "zeros (kinematic replay reads positions only)",
        "contact": f"npz contact arrays zeros; sidecars {cdone} = the orig tree's human contacts on the SPIDER "
        "time axis (contacts_from_orig)",
        "f0_prepad": "0 (SPIDER starts at the human frame 0)",
        "code": "experiments/policies/spider_arm/data.py",
    }
    (tdir / "task_info.json").write_text(json.dumps(ti, indent=1, ensure_ascii=False))
