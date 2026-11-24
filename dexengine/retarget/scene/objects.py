"""What a clip's `task_info.json` declares about its objects, resolved against the files on disk.

Pure filesystem logic (no MuJoCo): which visual mesh, texture, articulated part meshes and collision parts each
object has. The scene builder (`build.py`) turns these records into MjSpec elements.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field

#: Defaults of `task_info.articulated_object`.
ARTICULATED_DEFAULTS = {
    "joint_axis": [0.0, 0.0, -1.0],
    "joint_range": [0.0, 3.14159],
    "joint_pos": [0.0, 0.0, 0.0],
    "bottom_mesh": "bottom.obj",
    "top_mesh": "top.obj",
}

VISUAL_FILE = "visual.obj"
VISUAL_STEM = "visual"


@dataclass
class Articulation:
    """A one-hinge object: bottom part on the object body, top part on a hinged child body."""

    axis: list[float]
    range: list[float]
    pos: list[float]
    bottom_mesh: str
    top_mesh: str
    bottom_file: str | None = None  # absolute path when the file exists
    top_file: str | None = None


@dataclass
class ObjectAssets:
    """Files of one object as the scene builder needs them (absolute paths)."""

    mesh_dir: str | None
    convex_dir: str | None
    visual_file: str | None = None  # `<mesh_dir>/visual.obj` when it exists
    texture_file: str | None = None  # first `*.png` of `mesh_dir` (directory listing order)
    parts: list[tuple[str, str]] = field(default_factory=list)  # (stem, absolute file) collision parts
    articulation: Articulation | None = None

    @property
    def has_body(self) -> bool:
        return bool(self.parts)


def _abs(dataset_dir: str, rel: str | None) -> str | None:
    """`task_info` paths are relative to the dataset dir; a null value stays null (never the string "None")."""
    if rel is None or rel == "":
        return None
    return os.path.abspath(os.path.join(dataset_dir, rel))


def _first_png(mesh_dir: str) -> str | None:
    found = glob.glob(os.path.join(glob.escape(mesh_dir), "*.png"))
    return found[0] if found else None


def _obj_files(folder: str | None) -> list[tuple[str, str]]:
    """(stem, path) of the `*.obj` files directly in `folder`, sorted lexicographically by file name."""
    if folder is None or not os.path.isdir(folder):
        return []
    names = sorted(n for n in os.listdir(folder) if n.endswith(".obj") and os.path.isfile(os.path.join(folder, n)))
    return [(n[: -len(".obj")], os.path.join(folder, n)) for n in names]


def articulation_for(task_info: dict, side: str) -> Articulation | None:
    """The articulated-object record of `task_info` when it applies to `side` (its `side` is that side or `both`)."""
    art = task_info.get("articulated_object")
    if not art:
        return None
    if art.get("side") not in (side, "both"):
        return None
    vals = {k: art.get(k, v) for k, v in ARTICULATED_DEFAULTS.items()}
    return Articulation(
        axis=[float(x) for x in vals["joint_axis"]],
        range=[float(x) for x in vals["joint_range"]],
        pos=[float(x) for x in vals["joint_pos"]],
        bottom_mesh=str(vals["bottom_mesh"]),
        top_mesh=str(vals["top_mesh"]),
    )


def primary_object(dataset_dir: str, task_info: dict, side: str, visual_as_collision: bool) -> ObjectAssets:
    """Resolve the object held by `side`.

    Collision parts: the single `visual` entry with `visual_as_collision`; else the `*.obj` files of `convex_dir`
    without `visual.obj` and the articulated part names (lexicographic order); an empty list falls back to `visual`
    when the visual mesh exists, and raises `FileNotFoundError` when it does not. A side that declares neither folder
    has no parts (no body).
    """
    mesh_dir = _abs(dataset_dir, task_info.get(f"{side}_object_mesh_dir"))
    convex_dir = _abs(dataset_dir, task_info.get(f"{side}_object_convex_dir"))
    obj = ObjectAssets(mesh_dir=mesh_dir, convex_dir=convex_dir)
    if mesh_dir is None and convex_dir is None:
        return obj
    if mesh_dir is not None:
        visual = os.path.join(mesh_dir, VISUAL_FILE)
        obj.visual_file = visual if os.path.isfile(visual) else None
        if os.path.isdir(mesh_dir):
            obj.texture_file = _first_png(mesh_dir)
    art = articulation_for(task_info, side)
    if art is not None and mesh_dir is not None:
        for attr, name in (("bottom_file", art.bottom_mesh), ("top_file", art.top_mesh)):
            path = os.path.join(mesh_dir, name)
            if os.path.isfile(path):
                setattr(art, attr, path)
            else:
                warnings.warn(f"[SCENE] articulated part mesh missing: {path}", stacklevel=2)
    obj.articulation = art
    if visual_as_collision and obj.visual_file is not None:
        obj.parts = [(VISUAL_STEM, obj.visual_file)]
        return obj
    excluded = {VISUAL_FILE}
    if art is not None:
        excluded |= {art.bottom_mesh, art.top_mesh}
    parts = [(stem, path) for stem, path in _obj_files(convex_dir) if os.path.basename(path) not in excluded]
    if not parts:
        if obj.visual_file is None:
            raise FileNotFoundError(
                f"[SCENE] {side} object has no collision parts in {convex_dir} and no visual mesh in {mesh_dir}"
            )
        warnings.warn(
            f"[SCENE] {side} object: no convex parts in {convex_dir}; colliding the visual mesh (as its convex hull)",
            stacklevel=2,
        )
        parts = [(VISUAL_STEM, obj.visual_file)]
    obj.parts = parts
    return obj


def extra_objects(dataset_dir: str, task_info: dict) -> list[tuple[int, ObjectAssets]]:
    """(list index, assets) of every usable `task_info.extra_objects` entry.

    An entry without a mesh folder or without `visual.obj` is skipped with a warning. Collision parts are all `*.obj`
    of `convex_dir` (default `mesh_dir`), `visual.obj` included, sorted lexicographically; none gives `visual`.
    """
    out = []
    for i, entry in enumerate(task_info.get("extra_objects") or []):
        mesh_dir = _abs(dataset_dir, entry.get("mesh_dir"))
        visual = os.path.join(mesh_dir, VISUAL_FILE) if mesh_dir is not None else None
        if visual is None or not os.path.isfile(visual):
            warnings.warn(f"[SCENE] extra object {i} skipped: no visual mesh ({visual})", stacklevel=2)
            continue
        convex_dir = _abs(dataset_dir, entry.get("convex_dir")) or mesh_dir
        obj = ObjectAssets(
            mesh_dir=mesh_dir, convex_dir=convex_dir, visual_file=visual, texture_file=_first_png(mesh_dir)
        )
        obj.parts = _obj_files(convex_dir) or [(VISUAL_STEM, visual)]
        out.append((i, obj))
    return out


def is_collision_stem(stem: str) -> bool:
    """A part stem that is all digits (a convex piece) or `visual` (hull fallback) makes a collision geom."""
    return stem.isdigit() or stem == VISUAL_STEM
