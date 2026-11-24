"""Phase B obs enrichment: per-fingertip-to-nearest-surface geometry.

Implements `ObjectSurfaceGeometry` observation term that computes, per fingertip:
  - tip_to_surface_vec (3): vector from fingertip to nearest surface point, world frame
  - surface_normal (3): outward normal at nearest surface point, world frame
  - signed_distance (1): sign-positive when tip is outside, negative when inside

Approach:
  - Pre-sampled surface point cloud (N=2048) with per-point normals, computed offline
    by `dexengine/data/object_surface.py`.
  - Each step: transform points to world frame using current obj pose, run GPU cdist
    from each fingertip to all surface points, argmin gives the nearest. Approximation
    error bounded by point spacing (~3-5 mm for the beaker), well under the 2cm hinge
    granularity.

Flags-controlled which slices are included in the obs:
  - obs_add_signed_distance → 5 dims
  - obs_add_surface_vec    → 15 dims (5 tips × 3)
  - obs_add_surface_normal → 15 dims (5 tips × 3)
  - obs_add_inertia        → 3 dims (diagonal inertia)

When all flags are 0, the term returns a 0-dim tensor — but the env_cfg only
registers this term when at least one flag is on, so this fallback never fires
in practice.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass


def load_object_surface_pointcloud(mesh_path: str, device: torch.device | str):
    """Load a pre-sampled surface point cloud + normals from .npz file.

    `mesh_path` is the path to the original .obj used to generate the cloud (or to
    its directory). The cached cloud is expected at `<mesh_dir>/surface_points.npz`.

    Returns:
        points: (N, 3) float32 tensor on device, object-local coordinates.
        normals: (N, 3) float32 tensor on device, object-local coordinates.
    """
    mesh_dir = mesh_path if os.path.isdir(mesh_path) else os.path.dirname(mesh_path)
    npz_path = os.path.join(mesh_dir, "surface_points.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Surface point cloud not found at {npz_path}. "
            f"Run: python dexengine/data/object_surface.py --mesh <mesh.obj>"
        )
    data = np.load(npz_path)
    points = torch.from_numpy(data["points"]).float().to(device)
    normals = torch.from_numpy(data["normals"]).float().to(device)
    return points, normals
