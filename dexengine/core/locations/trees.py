"""Tree folders: the identity (combo, hand, dataset, embodiment) of a sequence folder, sync folders, canonical
file counts, and `ClipPaths` (every output path of one clip)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dexengine.core.locations.layers import ik_dir, named_roster_of, placement_dir, robots_dir, trees_dir, wrist_dir
from dexengine.core.locations.modules import data_layout
from dexengine.core.locations.names import PLACEMENT_SFX, TREE_SFX, clip_name, ik_suffix, tree_name
from dexengine.core.locations.roots import f0_root


def _combo_of_tree_root(p: Path) -> str:
    """The combo of a **tree root** folder (the folder whose children are trees), or "" if `p` is not one:
    the layer's trees root `…/trees/<combo>/<emb>`, or a frozen generation's preserved trees root (its roster's combo).
    No folder-name rule for old flat roots (`…/trajectories_gen_v2/<combo>`) — that layout is gone, so such a folder
    has no identity and seq_dir_identity dies on it."""
    if name := named_roster_of(p):
        return data_layout.NAMED_ROSTERS[name]["combo"]
    if p.parent.parent.name == "trees" and p.name in data_layout.EMBODIMENTS:  # canonical: …/trees/<combo>/<emb>
        return p.parent.name
    return ""


def seq_dir_identity(seq_dir: str | os.PathLike) -> tuple[str, bool]:
    """(combo, is_view) of a `--sequence_dir` — the one reader of "which combo's trees is this folder".

    A tree root answers by its own place (`_combo_of_tree_root`). Any other folder is a **view** (a folder of links
    to trees, e.g. `dexengine/placement/train_set/select.py --view-root` or a pilot/canary view): its identity is
    measured, not read from its name — every tree entry is resolved (`realpath`) and all of them must sit in **one**
    tree root. Zero tree entries, entries in two roots, or a root that is not a combo tree raise ValueError (a view with
    no identity cannot be checked against its roster).
    """
    p = Path(seq_dir).resolve()
    combo = _combo_of_tree_root(p)
    if combo:
        return combo, False
    if not p.is_dir():
        raise ValueError(f"no sequence folder: {p}")
    roots = {
        Path(os.path.realpath(p / e)).parent
        for e in os.listdir(p)
        if (p / e / "0" / "trajectory_kinematic.npz").is_file()
    }
    if len(roots) != 1:
        raise ValueError(
            f"the tree entries of view {p} resolve to {len(roots)} tree roots"
            f"({', '.join(sorted(map(str, roots))[:3]) or 'no entries'}) — a view must be a subset of "
            "one combo tree root"
        )
    root = roots.pop()
    combo = _combo_of_tree_root(root)
    if not combo:
        raise ValueError(f"the entries of view {p} point to {root}, which is not a combo tree root")
    return combo, True


def combo_from_seq_dir(seq_dir: str | os.PathLike) -> str:
    """The combo of `--sequence_dir` — a tree root by its place, a view by where its entries resolve
    (`seq_dir_identity`). "" only for an empty argument; a folder whose identity cannot be measured raises.

    The data folder handed to training already determines (arm, hand). Fail to read it and the entry point looks for
    **a different cache with a different robot** (every clip then looks missing, rc=2).
    """
    if not seq_dir:
        return ""
    return seq_dir_identity(seq_dir)[0]


def f0_wrist_dir(hand: str, side: str, dataset: str) -> Path:
    """f0 shadow wrist cache. It is **independent of placement**, so 1× per (hand, clip) is enough;
    that is why it lives outside the combo tree and **has no suffix**."""
    return f0_root(dataset) / hand / f"wrist_poses_{side}"


# ── Per-combo output directories and their canonical-file globs (`--what dirs|counts`) ──
def sync_dirs(combo: str) -> list[tuple[str, str]]:
    """List of (subdirectory name under robots/<combo>, glob of canonical files).

    Why the glob is needed: sidecars live mixed into the same directory. A count check must count
    **only the canonical files** (the same reason we block a hollow 0=0 pass).
    """
    s = ik_suffix()
    return [
        ("placement", f"*{TREE_SFX}{PLACEMENT_SFX}"),
        (f"arm_ik_poses_right{s}", f"*{TREE_SFX}.npz"),
        (f"arm_ik_poses_left{s}", f"*{TREE_SFX}.npz"),
        ("wrist_poses_right", f"*{TREE_SFX}.npz"),
        ("wrist_poses_left", f"*{TREE_SFX}.npz"),
    ]


def canon_count(combo: str, dataset: str) -> list[tuple[str, int]]:
    """**Canonical file count** per subdirectory (sidecars excluded). Uncountable gives -1."""
    out = []
    for sub, pat in sync_dirs(combo):
        d = robots_dir(combo, dataset) / sub
        out.append((sub, len(list(d.glob(pat))) if d.is_dir() else -1))
    return out


# ── Every path of one clip ──────────────────────────────────────────────────
@dataclass(frozen=True)
class ClipPaths:
    """The output paths one clip leaves behind. Suffixes are attached here only.

    `name` is accepted with or without the suffix — freeing the caller from having to care which
    namespace it is in is the whole reason this class exists.
    """

    name: str
    combo: str  # no silent default — the caller states combo, hand, dataset and emb
    hand: str
    dataset: str
    emb: str

    # ── Names ──
    @property
    def clip(self) -> str:
        return clip_name(self.name)

    @property
    def tree(self) -> str:
        return tree_name(self.name)

    # ── 1. Placement judging result ──
    @property
    def placement(self) -> Path:
        return placement_dir(self.combo, self.dataset) / f"{self.tree}{PLACEMENT_SFX}"

    # ── 2. Arm IK ──
    def ik(self, side: str) -> Path:
        return ik_dir(self.combo, side, self.dataset) / f"{self.tree}.npz"

    # ── 3. Wrist cache (combo canonical) ──
    def wrist(self, side: str) -> Path:
        return wrist_dir(self.combo, side, self.dataset) / f"{self.tree}.npz"

    # ── 6. f0 shadow wrist (no suffix) ──
    def f0_wrist(self, side: str) -> Path:
        return f0_wrist_dir(self.hand, side, self.dataset) / f"{self.clip}.npz"

    # ── 7. Combo tree ──
    @property
    def tree_dir(self) -> Path:
        return trees_dir(self.combo, self.dataset, self.emb) / self.tree / "0"

    @property
    def tree_qk(self) -> Path:
        return self.tree_dir / "trajectory_kinematic.npz"

    def contact(self, side: str) -> Path:
        """Contact labels — they live **inside the tree** (not in the combo directory)."""
        return self.tree_dir / f"contact_mano_{side}.npz"

    # ── The 5 kinds completeness (comp_check) needs ──
    def required(self) -> dict[str, Path]:
        return {
            "tree": self.tree_qk,
            "ikR": self.ik("right"),
            "ikL": self.ik("left"),
            "wrR": self.wrist("right"),
            "wrL": self.wrist("left"),
        }

    def missing(self) -> list[str]:
        return [k for k, p in self.required().items() if not p.exists()]
