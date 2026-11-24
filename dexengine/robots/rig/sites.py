"""Vendor asset lookup and site rules of the hand-rig builder (``dexengine/robots/hand_rig.py``): palm site offset
and orientation, fingertip sites and the fingertip pad check."""

from __future__ import annotations

import pathlib

import numpy as np

from dexengine.core import paths as _P
from dexengine.core.errors import InputMissingError
from dexengine.robots.rig.urdf_kin import Urdf, _origin_of

#: Overridden by --out-name; empty means "use the registry's retarget_dirname".
_OUT_NAME = ""


def load_specs():
    """The hand registry module (`dexengine.core.hands`)."""
    from dexengine.core import hands

    return hands


def vendor_of(hand: str, specs) -> dict:
    """Vendor asset locations for one hand, read from its HandSpec.

    Registration is one place: a separate vendor table here would need a second edit per hand and go stale.
    The location lives with the hand.
    """
    hs = specs.HAND_REGISTRY.get(hand)
    if hs is None:
        raise SystemExit(
            f"[hand_rig] unregistered hand {hand!r} — "
            f"registered: {sorted(k for k, v in specs.HAND_REGISTRY.items() if v)}. "
            f"Register it first."
        )
    urdf = getattr(hs, "vendor_urdf", "") or ""
    if not urdf:
        raise InputMissingError(
            f"[hand_rig] hand {hand!r}: its HandSpec has no vendor_urdf "
            f"— state the vendor URDF location there; it is not guessed here."
        )
    out = _OUT_NAME or getattr(hs, "retarget_dirname", "") or hand
    return {"urdf": urdf, "meshes": getattr(hs, "vendor_meshes", "") or "", "out": out}


def vendor_paths(vend: dict, side: str) -> tuple[pathlib.Path, pathlib.Path]:
    """(vendor URDF, mesh folder) of ``side`` for ``vendor_of``'s result. Paths are placed by the one repo-relative rule
    (``paths.repo_rel_path``: a data-layer head such as ``data/_stage/<hand>`` lives under the data root, anything
    else is a tracked asset of this tree). An empty mesh folder means the URDF's own folder (the common vendor
    layout)."""
    urdf_path = _P.repo_rel_path(vend["urdf"].format(side=side))
    meshdir = _P.repo_rel_path(vend["meshes"].format(side=side)) if vend["meshes"] else urdf_path.parent
    return urdf_path, meshdir


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


#: Position mismatch (m) allowed when the left rig is read as the mirror of the right one. A vendor that ships the
#: left hand as the mirror of the right gives 0 in theory, but drawing rounding remains — measured on DG-5F-S at most
#: 0.40 mm (ring distal `link_4_4`: right x=13.9 mm vs left x=14.3 mm). 1 mm passes that rounding and stays an order of
#: magnitude below the amount mirrored (the Y component of palm_site_offset, 4.6 mm on DG-5F-S). A hand off by more is
#: not a mirror and must be measured per side.
_MIRROR_TOL_M = 1e-3


def _palm_site_offset(spec, side: str, urdf_of_side, vname_of_side) -> np.ndarray:
    """Map HandSpec.palm_site_offset (declared for the right side) to an offset in this side's rig frame.

    Y may be flipped only if the left rig is the Y-mirror of the right one. That assumption is measured, not
    trusted: the zero-angle FK positions of the links the spec names (palm + 5 tips) in both vendor URDFs must satisfy
    `p_left == diag(1,-1,1) @ p_right`, otherwise this fails. A hand that is not a mirror cannot use this field —
    stopping beats emitting a silently wrong palm.
    """
    off = np.array([float(x) for x in getattr(spec, "palm_site_offset", (0.0, 0.0, 0.0))], dtype=float)
    if side == "right" or not off.any():
        return off

    right_urdf = urdf_of_side("right")
    left_urdf = urdf_of_side("left")
    checked = []
    for canon in (spec.palm_fk_body_right, *spec.tip_body_names_right):
        rn, ln = vname_of_side(canon, "right"), vname_of_side(canon, "left")
        if rn not in right_urdf.links or ln not in left_urdf.links:
            raise ValueError(
                f"palm_site_offset mirror check: declared link {canon!r} is not in the vendor URDF "
                f"(right {rn!r} · left {ln!r})"
            )
        pr = right_urdf.fk_world(rn)[:3, 3]
        pl = left_urdf.fk_world(ln)[:3, 3]
        gap = float(np.max(np.abs(pl - pr * np.array([1.0, -1.0, 1.0]))))
        checked.append((canon, gap))
        if gap > _MIRROR_TOL_M:
            raise ValueError(
                f"palm_site_offset can be mapped only when the left rig is the Y-mirror of the right one. "
                f"Mismatch at {canon!r} {gap * 1000:.3f} mm > {_MIRROR_TOL_M * 1000:.3f} mm "
                f"(right {np.round(pr, 6)} · left {np.round(pl, 6)}). "
                f"This hand needs palm_site_offset measured per side."
            )
    worst = max(checked, key=lambda kv: kv[1])
    print(
        f"  [left] palm_site_offset mirror check OK — {len(checked)} links, "
        f"max mismatch {worst[1] * 1000:.4f} mm ({worst[0]})"
    )
    return off * np.array([1.0, -1.0, 1.0])


#: rrc MANO wrist axis convention (the axes of `<side>_wrist_frame`, the MANO wrist frame of the hardware setup, used as
#: the retargeting root and the Pink control frame): distal = right −X, left +X; palm normal (out of the palm) = −Y on
#: both sides. Source: rrc `WRIST_CONVENTIONS["rby1m"]`.
RRC_MANO_AXES = {
    "right": {"distal": (-1.0, 0.0, 0.0), "palm_normal": (0.0, -1.0, 0.0)},
    "left": {"distal": (1.0, 0.0, 0.0), "palm_normal": (0.0, -1.0, 0.0)},
}
#: rig wrist axis convention (PATH A): distal = +Z, palm normal = +X, Y = Z × X — the axes of the MANO layer's
#: `qpos_wrist_{side}` (HandSpec.palm_site_offset docs; measured on h2o MANO keypoints: right distal +z, thumb +y,
#: flexion +x, left thumb −y).


def _palm_site_quat(spec, side: str, palm_link: str, vname_of_side) -> np.ndarray | None:
    """HandSpec.palm_site_mano_frame → orientation of this side's `{side}_palm` site (palm_link-local, wxyz). None
    when unset (the axes are inherited).

    Rule: when the hand root and the MANO wrist site are fixed children of the same parent in the robot document, read
    the relative rotation R_root←mano and map the rrc axes (distal, palm normal) to the rig axes (+Z, +X). A
    different parent or a non-fixed joint is an error."""
    src = getattr(spec, "palm_site_mano_frame", ()) or ()
    if not src:
        return None
    doc_rel, mano_right = src
    doc = Urdf(_P.repo_rel_path(doc_rel))
    mano_link = vname_of_side(mano_right, side)
    joints = {}
    for role, link in (("root", palm_link), ("mano", mano_link)):
        j = doc.parent_joint.get(link)
        if j is None or j.get("type") != "fixed":
            raise ValueError(f"palm_site_mano_frame: in {doc_rel} the parent joint of {link!r} is not fixed ({role})")
        joints[role] = j
    parents = {r: j.find("parent").get("link") for r, j in joints.items()}
    if parents["root"] != parents["mano"]:
        raise ValueError(
            f"palm_site_mano_frame: hand root {palm_link!r} and MANO site {mano_link!r} have different "
            f"parents ({parents}) — the relative pose cannot be read from two fixed joints"
        )
    r_root_mano = _origin_of(joints["root"])[:3, :3].T @ _origin_of(joints["mano"])[:3, :3]
    ax = RRC_MANO_AXES[side]
    z = r_root_mano @ np.array(ax["distal"])
    x = r_root_mano @ np.array(ax["palm_normal"])
    r_site = np.column_stack([x, np.cross(z, x), z])
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(r_site).as_quat()
    q = np.array([xyzw[3], *xyzw[:3]])
    q = q if q[0] >= 0 else -q
    ang = np.degrees(Rotation.from_matrix(r_site).magnitude())
    print(
        f"  [PALM_SITE_FRAME] {side}: {doc_rel} {palm_link}→{mano_link} (parent {parents['root']}) "
        f"→ site quat(wxyz) {_fmt(np.round(q, 7))} · {ang:.2f}° from the root axes"
    )
    return q


def _site_offsets(spec, side: str) -> list[np.ndarray]:
    """HandSpec.fingertip_site_offsets for ``side`` as one 3-vector per tip, tip-local.

    The parser lives in ``hands.fingertip_site_offsets_for`` (the one owner, shared with the Isaac env's naming
    helpers) — a second copy here would have no notion of side and give every hand's LEFT rig the RIGHT tip-local
    offsets.
    """
    return [np.array(v, float) for v in load_specs().fingertip_site_offsets_for(spec, side)]


def vendor_link_name(spec, name: str, side: str) -> str:
    """HandSpec name (MJCF right-side spelling) → the ``side`` vendor URDF spelling. Spelling differs per hand and is
    mapped with the declared prefix (wuji: `{side}_`, DG-5F-S: none — measured 26/27 vs 0/27)."""
    pref = (getattr(spec, "vendor_link_prefix", "{side}_") or "").format(side=side)
    return pref + (name[len("right_") :] if name.startswith("right_") else name)


#: Fingertip site = the pad (a geometric rule, not a naming convention): the site must lie within [0, _PAD_GAP_MAX_M]
#: inside the far end (t_max) to which the collision meshes of the tip link and its fixed children reach along the site
#: direction, and laterally within the hand's declared `HandSpec.fingertip_pad_lateral_max_m` (per hand, no default)
#: of the centre of the far-end cap _PAD_CAP_M thick. Measured gaps on the registered hands: 0.7–4.8 mm.
#: A wrong lateral sign is caught by left/right agreement: both sides mirror one hand, so the pad is at the same place
#: and the per-side (gap, lateral) must agree within _PAD_SIDE_TOL_M. Measured left/right difference at most 0.12 mm
#: (Wuji lateral); a value flipped on one side only gives 4.9 mm (DG-5F-S) and 62 mm (XHand). A direction rule such as
#: "palm side is fine" is not used — a DG-5F-S mirror error lies +5.0 mm on the palm side, so it would detect nothing.
_PAD_GAP_MAX_M = 6e-3
_PAD_CAP_M = 3e-3
_PAD_SIDE_TOL_M = 1e-3


def _tip_collider_points(urdf: Urdf, meshdir: pathlib.Path, tip: str) -> np.ndarray:
    """(N,3) collision-mesh vertices of ``tip`` and every fixed-joint descendant, in ``tip``'s frame.
    Mesh files resolve as the rig compiler does (``meshdir`` / basename); a collision without a mesh file dies."""
    import trimesh

    links, stack = [tip], [tip]
    while stack:
        cur = stack.pop()
        for j in urdf.root.findall("joint"):
            if j.get("type") == "fixed" and j.find("parent").get("link") == cur:
                c = j.find("child").get("link")
                links.append(c)
                stack.append(c)
    pts = []
    for ln in links:
        T_ln = urdf.transform(ln, tip)
        for col in urdf.links[ln].findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            f = meshdir / pathlib.Path(mesh.get("filename")).name
            if not f.is_file():
                raise FileNotFoundError(f"[TIP_SITE] {tip}: collision mesh {f} missing (meshdir={meshdir})")
            sc = np.array([float(x) for x in (mesh.get("scale") or "1 1 1").split()])
            v = np.asarray(trimesh.load(f, force="mesh").vertices) * sc
            pts.append((np.c_[v, np.ones(len(v))] @ (T_ln @ _origin_of(col)).T)[:, :3])
    if not pts:
        raise ValueError(
            f"[TIP_SITE] {tip}: no collision mesh on the tip link or its fixed children — cannot measure the pad"
        )
    return np.concatenate(pts)


def pad_gap(pts: np.ndarray, off: np.ndarray) -> tuple[float, float]:
    """(gap, lateral) m — the site ``off`` measured against the collider: gap along the site axis to the far end,
    lateral offset from the far-end cap centre. The rule's one measurement."""
    a = off / np.linalg.norm(off)
    t = pts @ a
    t_max = float(t.max())
    cap = pts[t >= t_max - _PAD_CAP_M].mean(axis=0)
    d = off - cap
    return t_max - float(off @ a), float(np.linalg.norm(d - (d @ a) * a))


def pad_lateral_max(spec) -> float:
    """The lateral tolerance (m) the hand declares. Missing is an error — there is no global default (see the HandSpec
    field docstring)."""
    v = getattr(spec, "fingertip_pad_lateral_max_m", None)
    if v is None or not float(v) > 0.0:
        raise ValueError(
            f"[TIP_SITE] {spec.name}: HandSpec.fingertip_pad_lateral_max_m is missing ({v!r}) — "
            "declare the current lateral maximum measured on the hand's vendor URDF + 1 mm (field "
            "docstring). There is no global default."
        )
    return float(v)


def tip_site_ctx(hand: str, spec, side: str, specs=None):
    """(urdf, tip link names, tip offsets, mesh folder) — the ``side`` inputs of the fingertip check. The source is
    HandSpec.vendor_urdf alone, the same document the rig and the merged URDF are built from."""
    urdf_path, meshdir = vendor_paths(vendor_of(hand, specs or load_specs()), side)
    tips = [vendor_link_name(spec, n, side) for n in spec.tip_body_names_right]
    return Urdf(urdf_path), tips, _site_offsets(spec, side), meshdir


def check_tip_sites(spec, ctxs: dict) -> None:
    """Measure per side whether each fingertip site sits on the pad (far end) of the vendor URDF tip collider, and
    check that both sides measure the same — rules in the constant comments above. ``ctxs`` = {side:
    tip_site_ctx(...)}. All tips are measured and every violation listed before failing (stopping at the first one
    would hide the other tips with the same cause). Marker `[TIP_SITE]`."""
    lat_max = pad_lateral_max(spec)
    meas: dict[str, list[tuple[str, float, float]]] = {}
    bad: list[str] = []
    for side, (urdf, tip_links, offsets, meshdir) in ctxs.items():
        meas[side] = []
        for tip, off in zip(tip_links, offsets, strict=True):
            if not np.linalg.norm(off) > 1e-6:
                raise ValueError(
                    f"[TIP_SITE] {side} {tip}: the tip offset is 0 — no site direction, so the pad cannot be measured"
                )
            gap, lat = pad_gap(_tip_collider_points(urdf, meshdir, tip), np.asarray(off, float))
            if not (0.0 <= gap <= _PAD_GAP_MAX_M and lat <= lat_max):
                bad.append(
                    f"{side} {tip}: offset {np.round(np.asarray(off) * 1000, 3).tolist()} mm is not on "
                    f"the pad — {gap * 1000:.1f} mm to the far end (allowed 0–{_PAD_GAP_MAX_M * 1000:.0f}) · "
                    f"lateral {lat * 1000:.1f} mm (allowed ≤{lat_max * 1000:.1f}, declared by {spec.name})"
                )
            meas[side].append((tip, gap, lat))
    dmax = (0.0, 0.0)
    if {"right", "left"} <= set(meas):
        for (tr, gr, lr), (tl, gl, ll) in zip(meas["right"], meas["left"], strict=True):
            dg, dl = abs(gr - gl), abs(lr - ll)
            if dg > _PAD_SIDE_TOL_M or dl > _PAD_SIDE_TOL_M:
                bad.append(
                    f"{tr} ↔ {tl}: left and right measure differently — far-end gap {gr * 1000:.1f} ↔ "
                    f"{gl * 1000:.1f} · lateral {lr * 1000:.1f} ↔ {ll * 1000:.1f} mm (allowed left/right "
                    f"difference ≤{_PAD_SIDE_TOL_M * 1000:.0f}) — the two sides mirror one hand, so the pad is "
                    "at the same place on both. One side's value (left key or right value) carries a wrong "
                    "lateral sign"
                )
            dmax = (max(dmax[0], dg), max(dmax[1], dl))
    if bad:
        raise ValueError(
            f"[TIP_SITE] {spec.name}: {len(bad)} fingertip site violations —\n  " + "\n  ".join(bad) + "\n"
            "Fix: first check that the measured document (HandSpec.vendor_urdf) is the rig/USD build source. If it is, "
            "measure the pad (far-end collider cap) on that side's vendor URDF and fix HandSpec.fingertip_site_offsets "
            "(a left key when the sides differ) — do not guess with a mirror of the right value (y·z sign flip)."
        )
    allm = [m for v in meas.values() for m in v]
    print(
        f"  [TIP_SITE] {spec.name} {'+'.join(meas)}: tips {len(allm)}/{len(allm)} = collider pad "
        f"(max to far end {max(m[1] for m in allm) * 1000:.1f} mm ≤ {_PAD_GAP_MAX_M * 1000:.0f} · max lateral "
        f"{max(m[2] for m in allm) * 1000:.1f} mm ≤ {lat_max * 1000:.1f} (declared) · "
        f"max left/right difference gap {dmax[0] * 1000:.2f} · "
        f"lateral {dmax[1] * 1000:.2f} mm ≤ {_PAD_SIDE_TOL_M * 1000:.0f})"
    )
