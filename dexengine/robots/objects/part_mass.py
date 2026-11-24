"""Masses of an articulated object's parts — the one rule every consumer that splits an object's mass over its parts
reads (the USD bake's link masses, which the runtime splits its total by; the MuJoCo retarget scene's part geoms).

Rule: the object's total mass is its consumer's total; it is split over the parts in proportion to the parts'
volumes. A part's volume is its mesh's volume when the mesh is watertight, otherwise the summed volume of its convex
collision pieces (the decomposition its colliders use — an open mesh's own volume is not a volume; weighing each part
on its own, with a per-part cap, lets a small door outweigh a body cut to the cap).

Top-level imports are standard library only: the bake imports this module in the Isaac interpreter.
"""

from __future__ import annotations

import os

#: Marker printed by every consumer that applies the rule.
TAG = "[PART_MASS]"


def part_volume(mesh_file: str) -> tuple[float, str]:
    """(volume m³, source) of one part mesh: the mesh's volume when it is watertight ('watertight'), otherwise the
    summed convex-hull volume of its pieces in the part's convex folder ('convex(<n>)', folder name owner
    decompose.part_convex_dir). A part that is open and has no pieces, or a volume that is not positive, dies."""
    import trimesh

    from dexengine.retarget import decompose as D

    m = trimesh.load(mesh_file, force="mesh", process=False)
    if m.is_watertight:
        v, src = abs(float(m.volume)), "watertight"
    else:
        cdir = D.part_convex_dir(mesh_file)
        files = D.part_files(cdir) if os.path.isdir(cdir) else []
        if not files:
            raise FileNotFoundError(f"{TAG} {mesh_file}: open mesh and no convex pieces in {cdir} — no volume")
        v = sum(
            abs(float(trimesh.load(os.path.join(cdir, f), force="mesh", process=False).convex_hull.volume))
            for f in files
        )
        src = f"convex({len(files)})"
    if not v > 0.0:
        raise ValueError(f"{TAG} {mesh_file}: volume {v} ({src}) is not positive")
    return v, src


def link_volume(mesh_files: list[str]) -> tuple[float, str]:
    """Summed part_volume of the meshes making up one link (its main part and the parts fixed to it)."""
    got = [part_volume(f) for f in mesh_files]
    if not got:
        raise ValueError(f"{TAG} a link without meshes")
    return sum(v for v, _ in got), "+".join(s for _, s in got)


def split_by_volume(total_kg: float, volumes: dict[str, float]) -> dict[str, float]:
    """The total mass split over the parts in proportion to their volumes (sums to the total)."""
    if not total_kg > 0.0:
        raise ValueError(f"{TAG} total mass {total_kg} is not positive")
    if not volumes or any(not v > 0.0 for v in volumes.values()):
        raise ValueError(f"{TAG} part volumes {volumes} — every part needs a positive volume")
    tot = sum(volumes.values())
    return {k: total_kg * v / tot for k, v in volumes.items()}
