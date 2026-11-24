"""Legacy trees (no `qpos_layout` key in task_info — the frozen V1 generation): the address table rebuilt with the
old slot arithmetic, byte-identical, and the self-test that compares it with the base-layer reader. Owner of the
addresses stays dexengine.core.qpos_layout; it calls `from_legacy` for these trees."""

from __future__ import annotations

import json
import os
import re

from dexengine.core.qpos_layout import DESK_SUPPORT_KEY, SIDES, Layout, read

_RE_LEGACY_OBJ = re.compile(r"objects/(\d+)/visual\.obj")


def from_legacy(
    tree_dir: str, ti: dict, require_layout: bool, nq: int | None = None, expect_H: int | None = None
) -> Layout:
    """A tree **without** qpos_layout (legacy TACO·gen_v2). Reproduces the loader arithmetic exactly.

    The decision rests on a measurement ("no key"), not a silent default. Entry points for new datasets forbid this
    path with require_layout=True.
    """
    if require_layout:
        raise ValueError(
            f"{tree_dir}: task_info has no qpos_layout (legacy tree). "
            "This entry point accepts only trees that record a layout — rebuild it with the base adapter "
            "or state require_layout=False."
        )
    # Hand width source order: tree record → derived from the actual width → caller statement; legacy 18 only last.
    H = ti.get("hand_qpos_dim")
    H_src = "task_info"
    if not H:
        if nq is not None and int(nq) >= 14 and (int(nq) - 14) % 2 == 0:
            H, H_src = (int(nq) - 14) // 2, "derived_from_nq"
        elif expect_H:
            H, H_src = int(expect_H), "caller"
        else:
            H, H_src = 18, "legacy_default_18"
    H = int(H)
    if expect_H and int(expect_H) != H and H_src in ("derived_from_nq", "task_info"):
        raise ValueError(f"{tree_dir}: hand widths diverge — tree/derived {H} ({H_src}) vs caller {expect_H}")
    nq = int(nq or ti.get("nq") or 0)
    n_extra = None
    r, left_slot, src = legacy_slots(tree_dir, ti)
    off = 2 * H + 7 * 0  # n_extra is an npz fact unknown here (caller: with_npz_extras)
    main = {"right": off + 7 * r, "left": off + 7 * left_slot}
    # legacy identity = the basename of `<side>_object_mesh_dir`. A side without the key stays empty.
    obj_ids, mesh_dirs = {}, {}
    for side in SIDES:
        v = ti.get(f"{side}_object_mesh_dir")
        if v:
            mesh_dirs[side] = str(v)
            obj_ids[side] = str(v).rstrip("/").split("/")[-1]
    return Layout(
        tree_dir=tree_dir,
        nq=nq or (off + 14),
        hand_qpos_dim=H,
        hands={"right": 0, "left": H},
        main=main,
        extras=[],
        artic={},
        objects=[],
        source=f"{src}|H:{H_src}",
        shared_object=False,
        n_extra_npz=n_extra,
        obj_ids=obj_ids,
        mesh_dirs=mesh_dirs,
        desk_support=ti.get(DESK_SUPPORT_KEY),
    )  # legacy (desk-scene) trees use the same stamped field


def legacy_slots(tree_dir: str, ti: dict) -> tuple[int, int, str]:
    """**The same decision order** as resolve_object_slots: slot keys (both) → scene.xml object ids → (0,1)."""
    if "right_object_slot" in ti and "left_object_slot" in ti:
        return int(ti["right_object_slot"]), int(ti["left_object_slot"]), "legacy_slots"
    xml_p = os.path.join(tree_dir, "scene.xml")
    if os.path.isfile(xml_p):
        with open(xml_p) as f:  # unreadable scene → dies (never the default slots)
            xml = f.read()
        ids: list[str] = []
        for m in _RE_LEGACY_OBJ.finditer(xml):
            if m.group(1) not in ids:
                ids.append(m.group(1))
        if len(ids) == 2:

            def _slot(key):
                v = ti.get(key)
                if not v:
                    return None
                mid = str(v).rstrip("/").split("/")[-1]
                return ids.index(mid) if mid in ids else None

            _r, _l = _slot("right_object_mesh_dir"), _slot("left_object_mesh_dir")
            if _r is not None and _l is not None and _r != _l:
                return _r, _l, "legacy_scene_xml"
    return 0, 1, "legacy_default"


# ── Self-test: compare old and new results for every tree ──────────────────────────
def selftest(root: str, limit: int = 0) -> int:
    """Scan <root>/<clip>/ and count whether (a) legacy trees give the same columns as the old arithmetic and (b)
    base trees pass check()."""
    import glob as _g

    trees = sorted(p for p in _g.glob(os.path.join(root, "*")) if os.path.isfile(os.path.join(p, "task_info.json")))
    if limit:
        trees = trees[:limit]
    src = {}
    bad = []
    for t in trees:
        try:
            lay = read(t, use_cache=False)
        except Exception as e:  # noqa: BLE001
            bad.append((t, f"{type(e).__name__}: {e}"))
            continue
        src[lay.source] = src.get(lay.source, 0) + 1
        if lay.source != "task_info.layout":
            with open(os.path.join(t, "task_info.json")) as f:
                ti = json.load(f)
            H = int(ti.get("hand_qpos_dim") or 18)
            r_old, l_old = legacy_slots(t, ti)[:2]
            if lay.main["right"] != 2 * H + 7 * r_old or lay.main["left"] != 2 * H + 7 * l_old:
                bad.append((t, "legacy column mismatch"))
    print(f"[qpos_layout] trees {len(trees)} · source {src} · mismatches/errors {len(bad)}")
    for t, why in bad[:5]:
        print(f"    {os.path.basename(t)}: {why}")
    return 0 if not bad else 3
