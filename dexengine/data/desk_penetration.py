"""Single owner of the version and the measurement of the D24 input (main object–desk penetration); the module head has
no dependencies (measurement imports stay inside functions).

Used by: object_roles (calls `measure` after deciding the main object and writes the marker `rules.D24`) ·
dexengine.data.candidates (compares the tree marker with this version) · hand_pipeline (the step-4 column counts this
version's marker).
The version is read by import, not by parsing the source.

Measurement (collision hull basis): the lowest z (capture frame) over **every frame** of this window's trajectory and
**every vertex of the main object's collision hull**, the quantity the simulation puts against the desk (read through
objects.hull: rigid = the authored hull used by PARK lifting `object_pin` · articulated root link = the cooked hull from
the bake). Every tree, no gate.
  support.obj_zmin_hull_traj = that value · support.obj_zmin_traj = lowest main-object origin z (information, not a
      verdict input).
  basis ∈ {hull, hull_artic_root (articulated: only the root link cooked hull,
  dexengine.data.objects.hull.root_link_points), no_usd (no hull = no USD, rigid body or collision mesh; no value
  written, dexengine.data.candidates drops it as
      `d24_unmeasured`),
  no_main (no main object)}. When an articulated object's cooking dump is missing or stale while the USD has collision
  meshes, it fails instead of reporting no_usd (dexengine.data.objects.hull.CookedHullInvalid, a bake order defect).
Trees on an older version are remeasured from step 4. Bump the version when this file, the hull reading
(dexengine.data.objects.hull) or the input outside the code, the **object collision shapes** (hull generation), change.
"""

# D24 input = every collision hull vertex · every frame · no gate · articulated root = cooked hull
D24_VERSION = "D24-hull-assetid-1"
#: Fields the D24 input writes into the tree support; measured on the tree trajectory (window), so D36 pieces do not
#: inherit them (segment.pieces.WINDOW_DECL).
D24_SUPPORT_FIELDS = ("obj_zmin_traj", "obj_zmin_hull_traj")
#: Retired old input (256 surface points · origin gate); the measurement removes it.
D24_RETIRED_FIELDS = ("obj_zmin_mesh_traj",)


def hull_zmin(ti: dict, q, mains: set) -> tuple:
    """Lowest z (capture frame) over every trajectory frame of the main objects' collision hulls · basis. (None,
    "no_usd")
    when any main object's hull cannot be read; a value from only some main objects is not the tree's D24. Articulated =
    root link hull only. A missing or stale articulated dump (with collision meshes in the USD) raises
    dexengine.data.objects.hull.CookedHullInvalid."""
    from scipy.spatial.transform import Rotation as Rot  # noqa: PLC0415

    from dexengine.data.objects import (
        hull as OH,  # noqa: PLC0415 — collision hull reader (owners object_pin · artifact_paths)
    )

    ds = ti.get("dataset_name")
    best, artic = None, False
    for o in ti["objects"]:
        if o["name"] not in mains:
            continue
        art = bool(o.get("articulation"))
        try:
            H = OH.root_link_points(ds, OH.asset_id(o)) if art else OH.hull_points(ds, OH.asset_id(o), False)
        except OH.HullMissing:
            return None, "no_usd"
        s0 = int(o["qpos_start"])
        R = Rot.from_quat(q[:, s0 + 3 : s0 + 7][:, [1, 2, 3, 0]]).as_matrix()
        m = float(OH.bottoms(H, R, q[:, s0 : s0 + 3]).min())
        best = m if best is None else min(best, m)
        artic = artic or art
    return best, ("hull_artic_root" if artic else "hull")


def measure(ti: dict, tree_dir: str, mains: set) -> str:
    """Measure this tree's D24 input with the main object name set `mains`, write it into `ti["support"]` and return the
    basis.
    The caller writes the marker (rules.D24)."""
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    sup = ti.get("support")
    if isinstance(sup, dict):
        for k in (
            D24_RETIRED_FIELDS + D24_SUPPORT_FIELDS
        ):  # this measurement rewrites them (nothing is left when it cannot measure)
            sup.pop(k, None)
    npz = glob.glob(os.path.join(tree_dir, "*", "trajectory_kinematic.npz")) or glob.glob(
        os.path.join(tree_dir, "trajectory_kinematic.npz")
    )
    if not (npz and mains):
        return "no_main"
    q = np.load(npz[0])["qpos"]
    sup = ti.setdefault("support", {})
    sup["obj_zmin_traj"] = round(
        min(float(q[:, int(o["qpos_start"]) + 2].min()) for o in ti["objects"] if o["name"] in mains), 4
    )
    hz, basis = hull_zmin(ti, q, mains)
    if hz is not None:
        sup["obj_zmin_hull_traj"] = round(hz, 4)
    return basis
