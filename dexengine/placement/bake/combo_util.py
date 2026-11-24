"""Stateless helpers of the bake stage (`bake/combo.py`): the f0 pad marker, reading one source clip, the planar
placement transform, scene file rewriting and the tree task_info / markers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import qpos_layout
from dexengine.data.objects import scene_parts as SP
from dexengine.placement.retime import warp as _rt

_XML_DIR_ATTR = re.compile(r'\b(meshdir|texturedir)="([^"]*)"')


def f0_pad_of(src: Path) -> int:
    """The pad P the f0 repair prepended to this staging (0 = no f0 repair in the staging).

    f0 runs before judging, so the staging the bake reads is P frames longer than the source (front pad). P is read
    only from the marker the f0 step leaves — an array-length difference would mix with the tail-trim tolerance.
    A marker that exists but cannot be read dies (read as 0 it would shift the time axis by P frames).
    """
    for name, key in (
        ("F0_PREJUDGE.json", "P"),
        ("F0_REPAIR_INPLACE.json", "P"),
        ("F0_REPAIR.json", "n_blend_effective"),
    ):
        p = src / name
        if p.exists():
            return int(json.loads(p.read_text()).get(key, 0) or 0)
    return 0


def _load_staging(hand: str, base: str, processed_root: Path, emb: str):
    """(src, z, qpos, hqd, pad) — one base-layer source clip (the reference is the clip itself).

    The hand width is owned by the qpos_layout reader (the recorded value of the base layer)."""
    src = processed_root / hand / emb / base
    z = dict(np.load(src / "0" / "trajectory_kinematic.npz"))
    qpos = z["qpos"].astype(np.float64)
    D = qpos.shape[1]
    hqd = qpos_layout.read(str(src), nq=D).hand_qpos_dim
    pad = f0_pad_of(src)  # f0 pre-repair pad (at the front), 0 if absent
    return src, z, qpos, hqd, pad


def _tf_points(p: np.ndarray, dx, dy, yaw_deg, pivot):
    """The planar placement transform of points (…, 3): rotate by yaw about the vertical axis through `pivot`
    (xy), then shift by (dx, dy); z unchanged."""
    Rz = Rot.from_euler("z", np.deg2rad(yaw_deg)).as_matrix()
    piv = np.array([pivot[0], pivot[1], 0.0])
    d3 = np.array([dx, dy, 0.0])
    shp = p.shape
    q = (p.reshape(-1, 3) - piv) @ Rz.T + piv + d3
    return q.reshape(shp)


def _abs_dirs_text(text: str, src_dir: Path) -> str:
    """Same rule as copy_xml_abs_dirs applied to **text** (an injected scene): meshdir/texturedir → absolute from
    src. The object convex part list is checked against the current decomposition first (scene_parts.verify_text)."""
    SP.verify_text(text, src_dir, str(src_dir))
    import re as _re

    def _fix(m):
        rel = m.group(2)
        absd = (src_dir / rel).resolve() if not os.path.isabs(rel) else Path(rel)
        return f'{m.group(1)}="{absd}/"'

    return _re.sub(r'(meshdir|texturedir)="([^"]*)"', _fix, text)


def copy_xml_abs_dirs(src_xml: Path, dst_xml: Path) -> None:
    """Copy scene.xml into the tree, rewriting `<compiler meshdir/texturedir>` as **absolute** paths.

    A relative meshdir resolves from the file's location and breaks at every tree depth, so it is pinned to an
    absolute path. An already absolute value (the f0 shadow) is left as is. The object convex part list is checked
    against the current decomposition first.
    """
    txt = src_xml.read_text()
    SP.verify_text(txt, src_xml.parent, str(src_xml))
    m = re.search(r"<compiler\b[^>]*>", txt)
    if m is None:
        raise InputMissingError(f"[bake] {src_xml}: no <compiler> element — meshdir cannot be pinned")

    def _abs(mm):
        val = mm.group(2)
        pth = Path(val) if os.path.isabs(val) else (src_xml.parent / val)
        return f'{mm.group(1)}="{os.path.normpath(pth)}/"'

    comp = _XML_DIR_ATTR.sub(_abs, m.group(0))
    dst_xml.write_text(txt[: m.start()] + comp + txt[m.end() :])


def _write_base_task_info(
    a,
    tio: dict,
    src: Path,
    dst: Path,
    base: str,
    hqd: int,
    pad: int,
    tf,
    adopted_p: Path,
    pl_name: str,
    retime_meta: dict,
) -> None:
    """task_info of a base-layer tree — assignment, slots, meshes, qpos_layout, objects[], support and quality come
    from the **source task_info** (tio is its copy), plus the placement and bake fields.
    The slots the tree states are read back through the reader only to check they are not self-contradictory."""
    dx, dy, yaw, pivot = tf
    # Slot check: <side>_object_slot = the rank of that side's main object block among the object blocks sorted by
    # address (joint columns not counted), as written by base_adapter.build_objects(). `2*H + 7*slot` arithmetic
    # would miss a 1-column joint between two object blocks, so the rank is looked up in lay.object_starts()
    # (qpos_layout). For an injected tree the statement about to be written (tio) is read instead of the source
    # (which has one hand and another nq).
    if tio.get("idle_hand_injected"):
        lay = qpos_layout.from_layout(str(src), tio)
    else:
        lay = qpos_layout.read(str(src), nq=int(tio.get("nq") or 0) or None)
    _obj_starts = lay.object_starts()  # start columns of the main ∪ extra blocks sorted by address
    for sd in ("right", "left"):
        st = tio.get(f"{sd}_object_slot")
        if st is not None and lay.main.get(sd) is not None:
            st_i = int(st)
            if st_i >= len(_obj_starts) or _obj_starts[st_i] != lay.main[sd]:
                raise SystemExit(
                    f"[bake] {base}: {sd}_object_slot={st} differs from qpos_layout main[{sd}]={lay.main[sd]}"
                    f"(object_starts={_obj_starts})"
                )
    tio.update(
        dict(
            xy_shift_traj=[dx, dy],
            yaw_deg=yaw,
            pivot_traj=list(pivot),
            placement_v2=True,
            placement_source=str(adopted_p.name),
            retimed_from=pl_name,
            retime_safety=a.warp_safety,
            retime_ik_suffix="",
            **retime_meta,
            combo=a.combo,
            combo_variant_of=base,
            hand_qpos_dim=hqd,
            trimmed_tail_frames=0,
            f0_policy=("v6_repaired_prejudge" if pad else "v6_repair_pending"),
            f0_prepad=int(pad),
            src_layer="base",
            dataset_name=str(tio.get("dataset_name") or a.dataset),
            embodiment_type=a.emb,
            baked_from=str(src),
        )
    )
    (dst / "task_info.json").write_text(json.dumps(tio, indent=1))
    _stamp_scene(dst)
    _write_f0_marker(dst, base, pad)


def _stamp_scene(dst: Path) -> None:
    """Support stamp — owner coords.stamp_scene (every step that writes a tree trajectory calls it right after)."""
    from dexengine.placement.coords import core as _vc  # noqa: PLC0415

    _vc.stamp_scene(dst)


def _write_f0_marker(dst: Path, base: str, pad: int) -> None:
    """F0_REPAIR.json of a padded tree: contact derivation and the render overlay read the pad from
    `n_blend_effective`. Length invariant (dexengine.data.contact.derive): T before retiming == T_base − trimmed + P."""
    if not pad:
        return
    with open(dst / "F0_REPAIR.json", "w") as fh:
        json.dump(
            {
                "source": base,
                "design": "prejudge_prepend_v1(C109)",
                "n_blend_effective": int(pad),
                "pad_new": int(pad),
                "pad_baseline": 0,
                "dt": _rt.DT,
                "note": "f0 was done by f0_prejudge.py in shadow staging before the judge. "
                "The bake only copies that pad.",
            },
            fh,
            indent=1,
        )
