"""Trajectory file format: schema version and reader (the writer is ``traj_record.TrajRecorder.finalize``).

A file with another schema version dies on read. Version 2 added failure_best (``fb_*``). RESTORED_SCHEMA_VERSION
marks a file rebuilt from archived finals (parquet, forces, row metadata) rather than recorded: it has the version-2
layout plus ``meta.restored`` (source and missing lists), exposed as ``.restored``. The reader requires the marker and
the version to agree, so neither a restored file posing as an original nor the reverse loads.
"""

from __future__ import annotations

import json
import os

import numpy as np

SCHEMA_VERSION = 2
RESTORED_SCHEMA_VERSION = 102  # restored file (module docstring): version-2 layout + meta.restored


class TrajFile:
    """One trajectory file loaded into arrays (``np.load``, no pickle)."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.path = os.path.abspath(path)
        self.meta = json.loads(str(z["meta"]))
        ver = int(self.meta.get("schema_version", -1))
        if ver not in (SCHEMA_VERSION, RESTORED_SCHEMA_VERSION):
            raise RuntimeError(f"[TRAJ_REC] schema {self.meta.get('schema_version')} ≠ {SCHEMA_VERSION}: {path}")
        self.restored = self.meta.get("restored")
        if (ver == RESTORED_SCHEMA_VERSION) != bool(self.restored):
            raise RuntimeError(
                f"[TRAJ_REC] version {ver} but meta.restored "
                f"{'missing' if not self.restored else 'present'} — restored-version mark "
                f"mismatch: {path}"
            )
        self.entities = json.loads(str(z["entities"]))
        self.clip_names = [str(x) for x in z["clip_names"]]
        self.ep_table, self.ep_score = z["ep_table"], z["ep_score"]
        self.states = z["states"]
        self.kept_offset, self.kept_length = z["kept_offset"], z["kept_length"]
        self.kept_clip, self.kept_rank, self.kept_score = z["kept_clip"], z["kept_rank"], z["kept_score"]
        self.kept_row = z["kept_row"]
        self.sel = z["sel"]
        self.fb_clip, self.fb_offset, self.fb_length = z["fb_clip"], z["fb_offset"], z["fb_length"]
        self.fb_score, self.fb_row = z["fb_score"], z["fb_row"]
        self.d_state = self.entities[-1]["hi"]

    def kept(self, j: int) -> np.ndarray:
        """States [L, D] of kept trajectory j."""
        o, n = int(self.kept_offset[j]), int(self.kept_length[j])
        return self.states[o : o + n]

    def selected(self, clip_name: str) -> np.ndarray | None:
        """Selected trajectory [L, D] of a clip by name; None for a clip without a success."""
        try:
            c = self.clip_names.index(clip_name)
        except ValueError:
            raise KeyError(
                f"[REPLAY_MAP] clip {clip_name!r} is not in the trajectory file's roster — {self.path}"
            ) from None
        j = int(self.sel[c])
        return None if j < 0 else self.kept(j)

    def failure_best(self, clip_name: str) -> np.ndarray | None:
        """Archived failure [L, D] of a no-success clip (not a rendered video); None for a clip with a success or
        without a valid episode."""
        c = self.clip_names.index(clip_name)
        hit = np.nonzero(self.fb_clip == c)[0]
        if hit.size == 0:
            return None
        o, n = int(self.fb_offset[hit[0]]), int(self.fb_length[hit[0]])
        return self.states[o : o + n]


def load(path: str) -> TrajFile:
    """Load a trajectory file."""
    return TrajFile(path)
