"""The render robot's head camera = one ZED Mini at the calibrated mount — the rule, its authoring and its check
(pxr only, no Kit; numpy · pxr · trimesh are imported inside functions so Kit processes can read the constants).

Why (only the mesh attachment changes): the head of every training body carries the camera link of the URDF it
was baked from — the ZED Mini mesh at the head mount (``zed_camera_link.xml``; bodies baked by earlier code carry
a larger stand-in box or a ``zedm.stl`` at the nominal URDF pose). The public data is filmed by a ZED Mini, so the
**render body** (the repaint ``render_body.py``
makes — never a training or judging body) gets exactly one ZED Mini mesh at the calibrated mount and loses the camera
link's visual. Colliders, masses and joints stay as trained (``render_plate.parity`` checks the rest byte-equal).

Owner of: the mount (``ZEDM_MOUNT`` — the physical robot's calibrated chain), the mesh file, the root the generator
authors (``ZEDM_ROOT``), which visual child is the camera link's — read, never listed by name (a camera mesh renamed in
the URDF renames the visual: ``camera.stl`` → ``head_camera.stl`` did, 2026-11):
  stated   ``stated_camera_visuals`` — the URDF the body was baked from (its converter record, ``config.yaml``) and the
           arm skeleton's head camera link (``camera_link``: the one ``extra_links`` entry fixed to the swap link):
           that joint's child link's ``<visual>`` mesh file stems (Isaac's URDF importer names a merged visual after
           its mesh file);
  measured ``camera_visual_names`` — the visual children of ``link_head_2/visuals`` that have a same-named collider
           child in ``link_head_2/collisions`` (the camera link is the only head link with a collider).
The generator (``swap``) dies unless the two agree on exactly one child; the parity check (``render_plate.parity``)
uses the measured rule on the training plate. ``problems`` is the positive check of render_body after the edit.
Marker: ``[RENDER_HEAD_CAM]``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

from dexengine.core.repo import REPO_ROOT as REPO

MARK = "[RENDER_HEAD_CAM]"
POSE_TOL = 1e-6  # m · quaternion components — authored values are the rule's, not a fit
_CAL = "head ZED Mini of the physical robot, simple-easyhec calibration"  # citation of the calibration source


class Src(NamedTuple):
    """One ZEDM_MOUNT row — (value, why): `why` is the value's source citation (a reason sentence, not a value)."""

    value: object
    why: str


# The ZED Mini mesh is the head camera's, the same for every hand (Stereolabs zed-ros2-description meshes/zedm.stl,
# fetched into the merged-robot mesh folder; configs/robots/vendor/) — an asset location, not a hand choice.
ZEDM_MESH = "assets/robots/full_robot_gen/_sanitized_meshes/zedm.stl"
# The physical robot's calibrated chain (the simple-easyhec recalibration). The uncalibrated nominal mount is
# xyz (0.0125, -0.007, 0.058) rpy 0 (the pose sharpa_wave's old body carries). head_0 · head_1 origins/axes are equal
# in both URDFs, so link_head_2 is the same frame (checked).
ZEDM_MOUNT = {
    "parent_link": Src("link_head_2", why=f"{_CAL} — calibrated chain link_head_2 -> ego_zed_m_camera"),
    "mount_xyz": Src(
        (0.0121367242, -0.0153970643, 0.0572624549), why="same comment — link_head_2 -> ego_zed_m_camera xyz"
    ),
    "mount_rpy": Src((0.0227371037, 0.0012451888, -0.0177429433), why="same comment — rpy (URDF fixed-axis XYZ)"),
    "left_xyz": Src((0.015, 0.0315, 0.0), why="same comment — ego_zed_m_camera -> left optical xyz"),
    "left_rpy": Src((-math.pi / 2, 0.0, -math.pi / 2), why="same comment — left optical rpy (ROS optical convention)"),
    "right_xyz": Src(
        (0.015, -0.0313986191, 0.0), why="same comment — ego_zed_m_camera -> right optical xyz (rpy = left's)"
    ),
    "left_pos_check": Src(
        (0.0276939720886537, 0.015823691002036552, 0.05795993357446777),
        why=f"{_CAL} — left optical centre in link_head_2, stated separately (independent oracle)",
    ),
    "mesh": Src(
        ZEDM_MESH,
        why="Stereolabs zedm.stl at the ego_zed_m_camera frame (origin 0 · scale 1 · black)",
    ),
}
BLACK = {"diffuse": (0.039, 0.039, 0.039), "metallic": 0.35, "roughness": 0.5}  # = render_body BLACK (camera parts)
#: The visual root the generator authors under the swap link's visual prototype (the ZED Mini) — the only root a render
#: plate has that its training plate lacks, besides the pair's declared render parts.
ZEDM_ROOT = "ZedMini"


def rpy_to_mat(r, p, y):
    """URDF rpy (fixed axes X→Y→Z) → R = Rz(y) Ry(p) Rx(r)."""
    import numpy as np

    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def mat_to_quat_wxyz(m):
    import numpy as np

    m = np.asarray(m, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (m[j, i] + m[i, j]) / s
        v[k] = (m[k, i] + m[i, k]) / s
        q = ((m[k, j] - m[j, k]) / s, *v)
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return q if q[0] >= 0 else tuple(-c for c in q)


def zedm_rig() -> dict:
    """link_head_2 → ZED Mini body (mesh) · left eye · right eye (pos, quat wxyz — eyes in the ROS optical convention),
    both eyes from the calibrated chain; ``baseline_m`` = |right − left|. Dies when the composed left eye differs from
    the separately stated left optical centre (left_pos_check) by > 0.01 mm or the right eye is not on the left
    optical +x axis."""
    import numpy as np

    mt = ZEDM_MOUNT
    Rm = rpy_to_mat(*mt["mount_rpy"][0])
    tm = np.array(mt["mount_xyz"][0])
    Rl = Rm @ rpy_to_mat(*mt["left_rpy"][0])
    tl = tm + Rm @ np.array(mt["left_xyz"][0])
    if np.max(np.abs(tl - np.array(mt["left_pos_check"][0]))) > 1e-5:
        raise SystemExit(f"{MARK} ZED Mini left eye {tl.tolist()} ≠ left_pos_check {mt['left_pos_check'][0]}")
    tr = tm + Rm @ np.array(mt["right_xyz"][0])
    baseline_m = float(np.linalg.norm(tr - tl))
    if np.max(np.abs(tr - (tl + baseline_m * Rl[:, 0]))) > 1e-9:
        raise SystemExit(f"{MARK} ZED Mini right eye {tr.tolist()} is off the left optical +x axis")
    ql = mat_to_quat_wxyz(Rl)
    return {
        "parent": mt["parent_link"][0],
        "body_pos": tuple(tm.tolist()),
        "body_quat": mat_to_quat_wxyz(Rm),
        "left_pos": tuple(tl.tolist()),
        "left_quat": ql,
        "right_pos": tuple(tr.tolist()),
        "right_quat": ql,
        "baseline_m": baseline_m,
    }


def swap_rule() -> tuple[str, str]:
    """(the swap link = the ZED Mini mount's parent link, the root this generator authors there)."""
    return ZEDM_MOUNT["parent_link"][0], ZEDM_ROOT


def camera_link(arm: str):
    """The arm skeleton's head camera link: the one ``extra_links`` entry (core.arms.ArmSkeleton — the URDF merge adds
    it to the vendor body) fixed to the swap link. None when the arm declares none; more than one dies."""
    from dexengine.core import arms as AS

    spec = AS.ARM_REGISTRY.get(arm)
    sk = getattr(spec, "skeleton", None)
    link = swap_rule()[0]
    got = [x for x in (sk.extra_links if sk is not None else ()) if x.parent == link]
    if len(got) > 1:
        raise SystemExit(f"{MARK} {arm}: {len(got)} skeleton extra links on {link} — one head camera expected")
    return got[0] if got else None


def body_urdf(usd_path) -> Path:
    """The URDF a body was baked from: its path is the converter's (``config.yaml`` asset_path,
    ``selection.baked_urdf_path``) and its bytes are the bake record's (``build.BAKE_RECORD``, checked by
    ``build.baked_from`` — the URDF sha256, the converter record and this USD's md5 as the bake wrote it). The camera
    visual is read from that URDF only when all of them still hold; a missing record or file, another USD, other URDF
    bytes or a USD written after the bake (record stale) dies — re-bake the body. No fallback."""
    from dexengine.robots import selection as SEL
    from dexengine.robots.usd import build as B

    p = SEL.baked_urdf_path(usd_path)
    if p is None or not p.is_file():
        raise SystemExit(
            f"{MARK} {usd_path}: the URDF it was baked from is unknown or missing ({SEL.bake_meta_file(usd_path)} "
            f"asset_path → {p}) — the head camera visual is read from that URDF; re-bake the body"
        )
    got = B.baked_from(Path(usd_path).parent, p, usd=Path(usd_path))
    if got != "same":
        raise SystemExit(
            f"{MARK} {usd_path}: bake record ({B.BAKE_RECORD}) vs {p} — {got}; re-bake the body "
            "(scripts/build_robot_usd.py) before its render body"
        )
    return p


def stated_camera_visuals(urdf, arm: str) -> list[str]:
    """The camera visual names the URDF states: the arm's head camera joint (``camera_link``) → its child link →
    the file stem of each ``<visual>`` mesh (the importer's visual name). A URDF without that joint, a joint on
    another parent, or a link without a visual mesh dies."""
    import xml.etree.ElementTree as ET

    x = camera_link(arm)
    link = swap_rule()[0]
    if x is None:
        raise SystemExit(f"{MARK} {arm}: no head camera link on {link} in the arm skeleton — nothing to swap")
    root = ET.parse(str(urdf)).getroot()
    js = [j for j in root.findall("joint") if j.get("name") == x.joint]
    if len(js) != 1 or js[0].find("parent") is None or js[0].find("parent").get("link") != link:
        raise SystemExit(f"{MARK} {urdf}: joint {x.joint} on {link} — {len(js)} found (one expected)")
    child = js[0].find("child").get("link")
    links = [lk for lk in root.findall("link") if lk.get("name") == child]
    stems = [Path(m.get("filename", "")).stem for lk in links for v in lk.findall("visual") for m in v.iter("mesh")]
    if len(links) != 1 or not stems:
        raise SystemExit(f"{MARK} {urdf}: camera link {child!r} — {len(links)} links · visual meshes {stems}")
    return stems


def mesh_file() -> Path:
    return REPO / ZEDM_MESH


def _mesh():
    import trimesh

    m = trimesh.load(str(mesh_file()), force="mesh")
    return m.vertices.astype("float32"), m.faces.astype("int64")


def _protos(stage) -> tuple:
    """(visual prototype prim, collider prototype prim) the default prim's link_head_2/{visuals,collisions} reference —
    each referenced by that one prim only (an edit there must not reach another link; the rule's owner is
    dexengine.robots.usd.render_parts.single_user_prototype, imported in the call)."""
    from dexengine.robots.usd import render_parts as RVP

    root, link = stage.GetDefaultPrim().GetPath(), swap_rule()[0]
    return tuple(
        RVP.single_user_prototype(stage, str(root.AppendPath(f"{link}/{sub}")), MARK)
        for sub in ("visuals", "collisions")
    )


def camera_visual_names(stage) -> list[str]:
    """The camera link's visual children: visual children of link_head_2/visuals with a same-named collider child."""
    vis, col = _protos(stage)
    # all children — the prototypes are authored as `over` (a flattened instancing layer): the default predicate
    # hides them
    cols, mine = {c.GetName() for c in col.GetAllChildren()}, ZEDM_ROOT
    return [c.GetName() for c in vis.GetAllChildren() if c.GetName() in cols and c.GetName() != mine]


def swap(stage, stated: list[str]) -> str:
    """The generator's edit (render body only): remove the camera link's visual child, author ``ZEDM_ROOT`` at the
    mount with the zedm.stl mesh (meters, scale 1, no collider) and a black OmniPBR. ``stated`` = the URDF's camera
    visuals (``stated_camera_visuals``); the stage's (``camera_visual_names``) must be the same one child — else die.
    Returns the marker line."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    vis, _ = _protos(stage)
    mine = ZEDM_ROOT
    old = camera_visual_names(stage)
    if len(old) != 1 or sorted(old) != sorted(stated):
        raise SystemExit(
            f"{MARK} camera link visual children {old} on the stage (rule: same-named collider) ≠ {stated} stated by "
            "the URDF the body was baked from — exactly one, the same, expected"
        )
    if vis.GetChild(mine):
        raise SystemExit(f"{MARK} {vis.GetPath()}/{mine} already exists — the body is not a fresh copy")
    stage.RemovePrim(vis.GetPath().AppendChild(old[0]))
    rig = zedm_rig()  # body pose only — the eyes are the collect cameras'
    x = UsdGeom.Xform.Define(stage, vis.GetPath().AppendChild(mine))
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*rig["body_pos"]))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*rig["body_quat"]))
    x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    v, f = _mesh()
    m = UsdGeom.Mesh.Define(stage, x.GetPath().AppendChild("mesh"))
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(f.reshape(-1).tolist()))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mat = UsdShade.Material.Define(stage, x.GetPath().AppendPath("Looks/ZedMiniBlack"))
    sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*BLACK["diffuse"]))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(BLACK["metallic"]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(BLACK["roughness"]))
    outp = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
        o.ConnectToSource(outp)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    pos, quat = (tuple(round(float(c), 6) for c in rig[k]) for k in ("body_pos", "body_quat"))
    return (
        f"{MARK} {vis.GetPath()}: removed camera visual '{old[0]}' · {mine} at {pos} q={quat} · "
        f"mesh {len(v)}v/{len(f)}f {ZEDM_MOUNT['mesh'][0]}"
    )


def problems(stage) -> list[str]:
    """What keeps a render body/plate from the rule (empty = it holds): no camera-link visual left · exactly one
    generator root (``swap_rule``) under the visual prototype at the mount (translate · orient · unit scale) ·
    its mesh = zedm.stl points ·
    no collider API anywhere in that subtree."""
    import numpy as np

    from pxr import Usd, UsdGeom, UsdPhysics

    vis, _ = _protos(stage)
    mine = ZEDM_ROOT
    out = []
    left = camera_visual_names(stage)
    if left:
        out.append(f"camera link visual still there: {left}")
    z = vis.GetChild(mine)
    if not z:
        return out + [f"{vis.GetPath()}/{mine} missing"]
    rig = zedm_rig()  # body pose only
    ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(z).GetOrderedXformOps()}
    t, q, s = ops.get("xformOp:translate"), ops.get("xformOp:orient"), ops.get("xformOp:scale")
    if t is None or np.max(np.abs(np.array(t) - np.array(rig["body_pos"]))) > POSE_TOL:
        out.append(f"{mine} translate {t} ≠ mount {rig['body_pos']}")
    qa = None if q is None else np.array([q.GetReal(), *q.GetImaginary()])
    if qa is None or np.max(np.abs(qa - np.array(rig["body_quat"]))) > POSE_TOL:
        out.append(f"{mine} orient {qa} ≠ mount {rig['body_quat']}")
    if s is None or np.max(np.abs(np.array(s) - 1.0)) > POSE_TOL:
        out.append(f"{mine} scale {s} ≠ 1")
    meshes = [p for p in Usd.PrimRange(z) if p.IsA(UsdGeom.Mesh)]
    v, _f = _mesh()
    pts = np.asarray(meshes[0].GetAttribute("points").Get() or [], float) if len(meshes) == 1 else None
    if pts is None or pts.shape != v.shape or np.max(np.abs(pts - v)) > POSE_TOL:
        out.append(f"{mine} mesh ≠ {ZEDM_MOUNT['mesh'][0]} ({len(meshes)} meshes)")
    if any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(z)):
        out.append(f"{mine} carries a collider")
    return out
