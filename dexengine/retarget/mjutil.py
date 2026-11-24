"""Small retargeting helpers: processed-data path layout and a no-op viewer stand-in.

The path helpers are plain string formatting on purpose: callers compare and store these strings, so they must not
normalise, resolve or otherwise rewrite their inputs.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

#: Absolute path of the `dexengine/retarget` package folder (readers derive their default `--dataset-dir` from it).
ROOT = os.path.dirname(os.path.abspath(__file__))


def processed_data_dir(
    dataset_dir: str, dataset_name: str, robot_type: str, embodiment_type: str, task: str, data_id: int
) -> str:
    """Folder of one processed clip instance: `<dataset_dir>/processed/<ds>/<robot>/<emb>/<task>/<data_id>`.

    `robot_type="mano"` gives the human keypoint folder; a hand name gives the retargeting output folder.
    """
    return f"{dataset_dir}/processed/{dataset_name}/{robot_type}/{embodiment_type}/{task}/{data_id}"


def object_mesh_dir(dataset_dir: str, dataset_name: str, object_name: str) -> str:
    """Folder holding one object's `visual.obj` (and its `convex/` parts) inside a processed dataset."""
    return f"{dataset_dir}/processed/{dataset_name}/assets/objects/{object_name}"


class _NullViewer:
    """Stand-in for an interactive viewer handle: always running, syncing does nothing."""

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        return None


@contextlib.contextmanager
def headless_viewer(model, data, show_viewer: bool = False, **kwargs) -> Iterator[_NullViewer]:
    """Context manager yielding a viewer-like object.

    Only the headless case is supported (no caller opens a window): with `show_viewer=False` it yields an object whose
    `is_running()` is always true and whose `sync()` does nothing. Asking for a real viewer raises.
    """
    if show_viewer:
        raise NotImplementedError("interactive viewer is not supported; run with --no-show-viewer")
    yield _NullViewer()
