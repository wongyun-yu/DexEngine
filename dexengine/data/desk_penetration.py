"""Single owner of the version and the measurement of the D24 input (main object–desk penetration); the module head has
no dependencies (measurement imports stay inside functions).

Used by: object_roles (calls `measure` after deciding the main object and writes the marker `rules.D24`) ·
dexengine.data.candidates (compares the tree marker with this version) · data.stages (the step-4 column counts this
version's marker) · the judge's desk check (placement/scene/object_desk reads every frame's hull vertices,
`hull_points_traj`; in an interpreter without pxr through this file's CLI in the retarget python).
The version is read by import, not by parsing the source.

Measurement (collision hull basis): the lowest z (capture frame) over **every frame** of this window's trajectory and
**every vertex of the main object's collision hull**, the quantity the simulation puts against the desk (read through
objects.hull: rigid = the authored hull used by PARK lifting `object_pin` · articulated root link = the cooked hull from
the bake). Every tree, no gate.
  support.obj_zmin_hull_traj = that value · support.obj_zmin_traj = lowest main-object origin z (information, not a
      verdict input).
  basis ∈ {hull, hull_artic_root (articulated: only the root link cooked hull, objects.hull.root_link_points),
  no_usd (the USD exists but has no rigid body or collision mesh; no value written, dexengine.data.candidates drops it
  as
      `d24_unmeasured`),
  no_main (no main object)}. A main object whose USD file does not exist fails (ObjectUsdMissing, naming the USD and
  its bake): the object USD bake runs before this step. When an articulated object's cooking dump is missing or stale
  while the USD has collision meshes, it fails too (objects.hull.CookedHullInvalid, a bake order defect).
Trees on an older version are remeasured from step 4. Bump the version when this file, the hull reading (objects.hull)
or the input outside the code, the **object collision shapes** (hull generation), change.

  <retarget py> dexengine/data/desk_penetration.py --deep <tree> --mains <names> --below <m> --out <npz>
  writes the judge's deep vertices (`deep_vertices`) of one tree (key `deep`, capture xyz (N, 3)) and prints one line.
"""

# D24 input = every collision hull vertex · every frame · no gate · articulated root = cooked hull
D24_VERSION = "D24-hull-assetid-1"
#: Fields the D24 input writes into the tree support; measured on the tree trajectory (window), so D36 pieces do not
#: inherit them (segment.pieces.WINDOW_DECL).
D24_SUPPORT_FIELDS = ("obj_zmin_traj", "obj_zmin_hull_traj")
#: Retired old input (256 surface points · origin gate); the measurement removes it.
D24_RETIRED_FIELDS = ("obj_zmin_mesh_traj",)


class ObjectUsdMissing(RuntimeError):
    """Main objects of a tree have no USD file (`missing_usds` rows): the object USD bake has not run for them. Not a
    HullMissing, so no caller that accepts an unmeasurable hull can take it for `no_usd`."""

    def __init__(self, ds: str, missing: list[dict]):
        names = ", ".join(f"{m['asset']} ({m['usd']})" for m in missing)
        super().__init__(f"{ds}: main objects without a USD: {names}")
        self.ds, self.missing = ds, missing


def missing_usds(ti: dict, mains: set) -> list[dict]:
    """Main objects (names in `mains`) whose USD file does not exist: [{name, asset, articulated, usd}] in tree order.
    The USD path is the one training spawns (objects.hull.usd_file)."""
    from dexengine.data.objects import hull as OH  # noqa: PLC0415 — collision hull reader (owner of the USD path)

    out = []
    for o in ti["objects"]:
        if o["name"] not in mains:
            continue
        art, asset = bool(o.get("articulation")), OH.asset_id(o)
        f = OH.usd_file(ti.get("dataset_name"), asset, art)
        if not f.is_file():
            out.append({"name": o["name"], "asset": asset, "articulated": art, "usd": str(f)})
    return out


def _main_hulls(ti: dict, q, mains: set) -> tuple:
    """The one hull reading of D24: ([(hull vertices (V, 3) body frame, R (T, 3, 3), t (T, 3)) per main object in tree
    order], basis); ([], "no_usd") when a main object's USD has no readable hull (a value from only some main objects
    is not the tree's D24). Articulated = root link hull only. A USD file that does not exist raises
    objects.hull.UsdMissing; a missing or stale articulated dump (with collision meshes in the USD) raises
    objects.hull.CookedHullInvalid."""
    from scipy.spatial.transform import Rotation as Rot  # noqa: PLC0415

    from dexengine.data.objects import (
        hull as OH,  # noqa: PLC0415 — collision hull reader (owners object_pin · artifact_paths)
    )

    ds = ti.get("dataset_name")
    out, artic = [], False
    for o in ti["objects"]:
        if o["name"] not in mains:
            continue
        art = bool(o.get("articulation"))
        try:
            H = OH.root_link_points(ds, OH.asset_id(o)) if art else OH.hull_points(ds, OH.asset_id(o), False)
        except OH.UsdMissing:
            raise
        except OH.HullMissing:
            return [], "no_usd"
        s0 = int(o["qpos_start"])
        out.append((H, Rot.from_quat(q[:, s0 + 3 : s0 + 7][:, [1, 2, 3, 0]]).as_matrix(), q[:, s0 : s0 + 3]))
        artic = artic or art
    return out, ("hull_artic_root" if artic else "hull")


def hull_zmin(ti: dict, q, mains: set) -> tuple:
    """Lowest z (capture frame) over every trajectory frame of the main objects' collision hulls · basis (`_main_hulls`;
    (None, "no_usd") without a readable hull)."""
    from dexengine.data.objects import hull as OH  # noqa: PLC0415

    hulls, basis = _main_hulls(ti, q, mains)
    if basis == "no_usd":
        return None, basis
    best = None
    for H, R, t in hulls:
        m = float(OH.bottoms(H, R, t).min())
        best = m if best is None else min(best, m)
    return best, basis


def deep_vertices(ti: dict, q, mains: set, below_z: float) -> tuple:
    """Every main-object collision-hull vertex at every trajectory frame (`_main_hulls`) whose capture z is below
    `below_z` → (capture xyz (N, 3), basis); (None, "no_usd") without a readable hull."""
    import numpy as np  # noqa: PLC0415

    hulls, basis = _main_hulls(ti, q, mains)
    if basis == "no_usd":
        return None, basis
    pts = [np.einsum("tij,vj->tvi", R, np.asarray(H, float)) + t[:, None, :] for H, R, t in hulls]
    allp = np.concatenate([p.reshape(-1, 3) for p in pts]) if pts else np.zeros((0, 3))
    return allp[allp[:, 2] < below_z], basis


def measure(ti: dict, tree_dir: str, mains: set) -> str:
    """Measure this tree's D24 input with the main object name set `mains`, write it into `ti["support"]` and return the
    basis. A main object without a USD file raises ObjectUsdMissing (nothing is measured). The caller writes the
    marker (rules.D24)."""
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
    missing = missing_usds(ti, mains)
    if missing:
        raise ObjectUsdMissing(ti.get("dataset_name"), missing)
    q = np.load(npz[0])["qpos"]
    sup = ti.setdefault("support", {})
    sup["obj_zmin_traj"] = round(
        min(float(q[:, int(o["qpos_start"]) + 2].min()) for o in ti["objects"] if o["name"] in mains), 4
    )
    hz, basis = hull_zmin(ti, q, mains)
    if hz is not None:
        sup["obj_zmin_hull_traj"] = round(hz, 4)
    return basis


def main(argv=None) -> int:
    """CLI (module doc): the deep vertices of one tree for an interpreter without pxr (the judge's desk check)."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="D24 deep vertices of one tree (the judge's desk check)")
    ap.add_argument("--deep", required=True, help="base tree")
    ap.add_argument("--mains", required=True, help="main object names, comma separated")
    ap.add_argument("--below", type=float, required=True, help="capture z below which a vertex is deep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = Path(a.deep)
    ti = json.loads((t / "task_info.json").read_text(encoding="utf-8"))
    q = np.load(t / "0" / "trajectory_kinematic.npz")["qpos"]
    pts, basis = deep_vertices(ti, q, set(a.mains.split(",")), a.below)
    if pts is None:
        print(f"[D24_DEEP] ✗ {t.name}: basis {basis} — no readable hull")
        return 2
    np.savez(a.out, deep=pts)
    print(f"[D24_DEEP] {t.name} deep vertices {len(pts)} · basis {basis} · below z {a.below} · {a.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
