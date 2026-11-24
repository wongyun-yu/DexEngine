# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central path resolver for TACO generated-data artifacts.

Single source of truth for every TACO generated-data path — precompute IK warmup
(wrist_poses / arm_ik_poses, per side), per-sequence trajectory artifacts
(body_positions, contact, pose-corrections), and per-object artifacts
(surface_points, keypoints, mesh, USD) — replacing the ~60 ad-hoc hardcoded
``precompute/taco…`` / ``data/taco…`` / ``usd_cache/objects_taco…`` literals that
were scattered across env-cfgs, consumers, producers, and launch scripts.

Design tenet — *formalize, don't move*: every method returns the SAME physical path
that exists today, so existing precompute caches load unchanged and the Batch-3
harness sees no path change. No file is moved/renamed. Large-scale expansion (more
sequences/meshes) writes into the same canonical templates via this resolver.

Scope: every dataset whose ``dexengine.core.layout.DATASETS[ds].status == "canon"``.
An intake dataset raises KeyError here — the same gate ``datasets.resolve()`` applies.

Import-safety: this module has no Isaac/torch imports (only os / dataclasses / functools).

Canonical layout (left/right SYMMETRIC — both sides carry a suffix):
  precompute/<dataset>/<type>_<side>/<sequence>.npz      # side = '_right'/'_left' suffix on TYPE DIR
  <traj_root>/<sequence>/0/<file>                        # body_positions, contact_<source>_<side>, …
  <traj_root>/initial_pose_corrections.json
  <meshes layer>/<mesh_id>/{surface_points.npz,keypoints_fps<k>.npy}   # per-asset (object_asset_id) derived
  <assets_objects layer>/<mesh_id>/{visual.obj,convex/}                # per-object source mesh (bake input)
  <usd layer>/<mesh_id>/object.usd                                     # per-object baked USD
                                                                       #   (TACO: data/usd_cache/objects_taco)
  <usd layer>/<mesh_id>/<USD stem>.cooked.npz                          # cooked hull (layout.cooked_hull_file)
Layer roots come from ``dexengine.core.layout.render(layer, ds)`` — the TACO strings are byte-identical to the
former literals.
"""

from __future__ import annotations

import os
import re
from functools import cache

from dexengine.core import dataset_dirs as _datasets
from dexengine.core import legacy_text as _LT
from dexengine.core.artic_assets import (  # noqa: F401 — re-exported
    _ARTIC_KINDS,
    ARTIC_LINKS,
    ARTIC_SELF_COLLISIONS,
    ARTIC_SELF_COLLISIONS_API,
    ARTIC_SELF_COLLISIONS_ATTR,
    ARTIC_SELF_COLLISIONS_BAKED,
    ASSET_SEP,
    artic_link_files,
    artic_self_collisions,
    object_asset_id,
)

_PRECOMPUTE_BASE = "precompute"
_PRECOMPUTE_TYPES = ("wrist_poses", "arm_ik_poses")
_layout = _datasets._layout  # dexengine.core.layout — the ONE owner of layer templates + DATASETS status


def _valid_datasets() -> tuple[str, ...]:
    """Datasets this resolver serves = dexengine.core.layout.DATASETS with status "canon" (read, not copied)."""
    return tuple(sorted(ds for ds, spec in _layout.DATASETS.items() if spec.status == "canon"))


def _abs(p: str) -> str:
    # the data-root rule is dexengine.core.layout.data_abspath alone (env DEXENGINE_MAIN_ROOT, else the working dir)
    return _layout.data_abspath(p)


class TacoPaths:
    """Resolves all TACO generated-data paths for one dataset variant.

    Keyed by (dataset, side, type, sequence) for precompute / per-sequence artifacts
    and (dataset→family, mesh_id, type) for per-object artifacts. The physical
    ``_left``-on-dir / ``_left``-on-filename / shared-geometry differences are
    internal details of the methods below.
    """

    def __init__(self, dataset: str):
        _valid = _valid_datasets()
        if dataset not in _valid:
            _st = _layout.DATASETS[dataset].status if dataset in _layout.DATASETS else "unregistered"
            raise KeyError(
                f"TacoPaths: dataset '{dataset}' is not trainable (status={_st}). Valid (status=canon): {_valid}. "
                "Promote it in dexengine.core.layout.DATASETS after its base/meshes/usd/trees exist."
            )
        self.dataset = dataset
        # Geometry family = the dataset itself (no dataset shares another's meshes/USDs today; a future variant
        # would add a mapping HERE, not a literal at a call site).
        self.family = dataset
        self.bimanual = "bimanual" in _layout.DATASETS[dataset].embodiments
        self._precompute_dataset = dataset
        self._spec = _datasets.resolve(dataset)  # registry: train_data_dir/object_usd/object_mesh_dir (abspath)

    @staticmethod
    def _side_sfx(side: str) -> str:
        if side in ("right", "left"):
            return f"_{side}"  # SYMMETRIC: both sides suffixed
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")

    # ---- precompute IK warmup (per-sequence, side-split via TYPE-DIR suffix) --
    def precompute_dir(self, type: str, side: str = "right") -> str:
        """precompute/<dataset>/<type>_<side> (abspath). type in {wrist_poses, arm_ik_poses}."""
        if type not in _PRECOMPUTE_TYPES:
            raise ValueError(f"precompute type must be in {_PRECOMPUTE_TYPES}, got {type!r}")
        return _abs(os.path.join(_PRECOMPUTE_BASE, self._precompute_dataset, f"{type}{self._side_sfx(side)}"))

    def precompute_file(self, type: str, sequence: str, side: str = "right") -> str:
        """<precompute_dir>/<sequence>.npz (canonical; no legacy-alias fallbacks)."""
        return os.path.join(self.precompute_dir(type, side), f"{sequence}.npz")

    # ---- per-sequence trajectory artifacts (under traj_root) -----------------
    def traj_root(self) -> str:
        """The live trajectory root (registry train_data_dir; abspath). May be a symlink
        — caller should os.path.isdir-guard if existence matters."""
        return self._spec.train_data_dir

    def traj_file(self, sequence: str, file: str, sub: str = "0") -> str:
        """<traj_root>/<sequence>/<sub>/<file>. sub='0' is the per-clip subdir (use sub='' for
        artifacts directly under the sequence dir)."""
        parts = [self.traj_root(), sequence]
        if sub:
            parts.append(sub)
        parts.append(file)
        return os.path.join(*parts)

    # ---- per-object artifacts (keyed by mesh_id, side-agnostic, family-shared) -
    def mesh_root(self) -> str:
        """<meshes layer> (registry object_mesh_dir = layout.render("meshes", ds); abspath). TACO: data/taco/meshes."""
        return self._spec.object_mesh_dir

    def mesh_dir(self, mesh_id) -> str:
        """<mesh_root>/<mesh_id> — fixes the parent-vs-per-id mismatch (datasets.resolve returns
        the PARENT; consumers must append the id rather than defaulting to '120')."""
        return os.path.join(self.mesh_root(), str(mesh_id))

    def surface_points_file(self, mesh_id) -> str:
        return os.path.join(self.mesh_dir(mesh_id), "surface_points.npz")

    def keypoints_file(self, mesh_id, k: int = 2) -> str:
        return os.path.join(self.mesh_dir(mesh_id), f"keypoints_fps{k}.npy")

    # ---- object SOURCE mesh (bake input; assets_objects layer, keyed by mesh_id) -
    def visual_mesh_root(self) -> str:
        """<assets_objects layer> (abspath, physical — layout STORAGE): <id>/visual.obj + <id>/convex/."""
        return _abs(_layout.render("assets_objects", self.family))

    def visual_mesh_dir(self, mesh_id) -> str:
        return os.path.join(self.visual_mesh_root(), str(mesh_id))

    def visual_mesh_file(self, mesh_id) -> str:
        return os.path.join(self.visual_mesh_dir(mesh_id), "visual.obj")

    # ---- object USD (keyed by mesh_id, shared across the geometry family) -----
    def usd_root(self) -> str:
        """<usd layer> (abspath) = dexengine.core.layout.render("usd", family). TACO: data/usd_cache/objects_taco
        (STORAGE row); base datasets: data/<ds>/usd."""
        return _abs(_layout.render("usd", self.family))

    #: **The one owner** of the articulated USD file name (D1). Must equal the name the articulated object bake writes.
    ARTIC_USD_NAME = "object_artic.usd"

    def usd_file(self, mesh_id, articulated: bool = False) -> str:
        """Object USD path. `articulated=True` gives the articulated USD (two links base+moving + one joint) — a
        **different file** from the rigid `object.usd`. Spawning an articulated tree with the rigid USD freezes the lid,
        so the caller must choose from the tree's `objects[].articulation` (never guess from a path string). An
        articulated object's `mesh_id` is the asset id (`object_asset_id(obj_id, articulation)`) — a paired
        articulation lives in a different folder from the object id."""
        return os.path.join(self.usd_root(), str(mesh_id), self.ARTIC_USD_NAME if articulated else "object.usd")

    def __repr__(self) -> str:
        return f"TacoPaths(dataset={self.dataset!r}, family={self.family!r}, bimanual={self.bimanual})"


@cache
def get(dataset: str) -> TacoPaths:
    """Cached TacoPaths factory."""
    return TacoPaths(dataset)


#: Render overlay USD — `<stem>_render.usd` next to the training USD. It loads the training USD as a **sublayer** and
#: only overrides texture materials and UVs on the visual meshes (collision, mass and joints compose to the training
#: version). Generator dexengine/robots/objects/render_overlay.py · texture rules dexengine/robots/textures/rules.py.
#: The naming rule is owned by this constant and render_usd_file alone.
RENDER_USD_SUFFIX = "_render"
#: Where the overlay's texture images live (a folder relative to the overlay).
RENDER_TEX_DIR = "render_tex"


def render_usd_file(train_usd: str) -> str:
    """Training object USD path → its render overlay path (`object.usd` → `object_render.usd` · `object_artic.usd` →
    `object_artic_render.usd`). Dies if it is not `.usd` or already an overlay (no name guessing)."""
    stem, ext = os.path.splitext(str(train_usd))
    if ext != ".usd" or stem.endswith(RENDER_USD_SUFFIX):
        raise ValueError(f"object USD with no definable render overlay: {train_usd!r}")
    return stem + RENDER_USD_SUFFIX + ext


def overlay_assets(overlay: str) -> list[str]:
    """The files a render overlay layer opens besides itself: its asset paths (the training USD sublayer, the
    textures under RENDER_TEX_DIR), resolved next to it. render_overlay writes a text layer; any other form dies."""
    with open(overlay, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("#usda"):
        raise ValueError(f"render overlay {overlay!r} is not a text layer — its assets cannot be stated")
    d = os.path.dirname(os.path.abspath(overlay))
    return sorted({os.path.normpath(os.path.join(d, a)) for a in re.findall(r"@([^@\n]+)@", text)})


#: Overlay root (preview) — when this env is set, the render spawn reads `<root>/<overlay absolute path>` (leading `/`
#: stripped) instead of the overlay next to the training USD (the generator's `--overlay-root` writes with the same
#: rule). A way to render another texture version without overwriting the canonical overlays.
#: Default = unset (byte-identical behaviour). Set but not a folder → dies · marker `[RENDER_OBJ_USD] … <root>`.
OVERLAY_ROOT_ENV = "DEXENGINE_RENDER_OVERLAY_ROOT"


def overlay_root() -> str | None:
    """The overlay root env value (None if unset). Dies if it is set but not a folder."""
    root = os.environ.get(OVERLAY_ROOT_ENV) or None
    if root is not None and not os.path.isdir(root):
        raise FileNotFoundError(f"{OVERLAY_ROOT_ENV}={root!r} is not a folder")
    return root


def overlay_file(train_usd: str, root: str | None) -> str:
    """Overlay place — next to the training USD (render_usd_file) without a root, else the same absolute path under
    that root."""
    r = render_usd_file(os.path.abspath(str(train_usd)))
    return r if root is None else os.path.join(root, r.lstrip("/"))


#: Rendering an object without a render overlay from its training USD (no textures) — off by default: no overlay =
#: death. Only when the caller states '1' explicitly · any other value dies.
RENDER_OBJ_FALLBACK_ENV = "DEXENGINE_RENDER_OBJ_ALLOW_FALLBACK"


def render_fallback_allowed() -> bool:
    v = os.environ.get(RENDER_OBJ_FALLBACK_ENV, "")
    if v not in ("", "0", "1"):
        raise ValueError(f"{RENDER_OBJ_FALLBACK_ENV}={v!r} — only 0 · 1 · unset")
    return v == "1"


def render_spawn_usds(usds: list[str]) -> tuple[list[str], list[str]]:
    """Object spawn list in a render context — swapped for the overlays (render_usd_file). A missing overlay dies (no
    untextured render) — only with RENDER_OBJ_FALLBACK_ENV=1 stated does it spawn the training USD and print the
    marker `[RENDER_OBJ_FALLBACK]`. Second value = those substituted paths (for the caller's `[RENDER_OBJ_USD]`
    marker). The caller (the robot selection's render_context) decides the render context. With an overlay root
    (OVERLAY_ROOT_ENV) only that root's overlays count (no substitution by the canonical-place overlays)."""
    allow = render_fallback_allowed()  # an unknown value dies whether or not overlays exist
    root = overlay_root()
    out, fallback = [], []
    for u in usds:
        r = render_usd_file(u) if root is None else overlay_file(u, root)
        if os.path.isfile(r):
            out.append(r)
        else:
            out.append(u)
            fallback.append(u)
    if fallback and not allow:
        raise FileNotFoundError(
            f"[RENDER_OBJ_USD] render overlay missing {len(fallback)}/{len(usds)} "
            f"{fallback[:3]} — bake with dexengine/robots/objects/render_overlay.py (to render "
            f"the training USD instead, state {RENDER_OBJ_FALLBACK_ENV}=1)"
        )
    if fallback:
        print(
            f"[RENDER_OBJ_FALLBACK] {RENDER_OBJ_FALLBACK_ENV}=1 — rendering {len(fallback)} objects without an "
            f"overlay from the training USD",
            flush=True,
        )
    return out, fallback


#: customLayerData keys of a render overlay — the one home of both names (writer render_overlay.write_overlay,
#: readers overlay_bake_mark and tex_packs). OVERLAY_TEX_KEY holds the per-link texture status, OVERLAY_TEX_SOURCES_KEY
#: the texture-pack sources (written only with --packs). There is no reader of the pre-rename key: overlays baked
#: before the rename read as "no mark" until re-baked (docs/pipeline.md, "Upgrading data made by older code").
OVERLAY_TEX_KEY = "dexengine_render_textures"
OVERLAY_TEX_SOURCES_KEY = "dexengine_render_texture_sources"
#: The overlay's texture marks (per-link values of the OVERLAY_TEX_KEY block) — a value starting with the "none" mark
#: means a shell that sublayers the training USD with no material. Reading the header and the combined mark are owned
#: by overlay_bake_mark alone (the shell count and the canonical manifest generator both read it).
_TEX_TAG = OVERLAY_TEX_KEY
_TEX_VAL_RE = re.compile(r'string\s+(\w+)\s*=\s*"([^"]*)"')


def overlay_bake_mark(path: str) -> tuple[str, dict[str, str]] | None:
    """Per-link texture values of the overlay header (8 KB) → (combined mark, per-link values) · None without a mark
    block. Combined mark = 'none' (any link 'none…' = shell) · 'solid' (every link '…solid…') · 'applied'
    (otherwise)."""
    with open(path, "rb") as fh:
        head = fh.read(8192).decode("utf-8", "replace")
    i = head.find(_TEX_TAG)
    if i < 0:
        return None
    j = head.find("}", i)
    links = dict(_TEX_VAL_RE.findall(head[i : j if j > 0 else i + 2000]))
    if not links:
        return None
    vals = list(links.values())
    if any(_LT.is_legacy(v) for v in vals):  # pre-translation mark: refuse instead of reading it as 'applied'
        raise ValueError(f"{path}: overlay bake mark uses the old (Korean) values — {_LT.MIGRATE_HINT} or re-bake it")
    mark = (
        "none" if any(v.startswith("none") for v in vals) else "solid" if all("solid" in v for v in vals) else "applied"
    )
    return mark, links


def overlay_shells(paths: list[str]) -> list[str]:
    """Spawned overlays that are untextured shells (link mark "none") — measured by actually reading the file
    header (customLayerData). Paths that are not overlays (substituted by the training USD) are not counted. Without a
    mark block (in the 8 KB header) a file is not taken for a shell."""
    out = []
    for p in paths:
        if not str(p).endswith(RENDER_USD_SUFFIX + ".usd") or not os.path.isfile(p):
            continue
        bm = overlay_bake_mark(p)
        if bm is not None and bm[0] == "none":
            out.append(p)
    return out


#: Canonical overlay manifest — md5 and bake mark per overlay of the canonical bake.
#: The statement lives in this one file (written by the generator robots.objects.overlay_manifest · never by hand ·
#: reaches machines through git — kept next to the data it would go stale with stale copies: once 95 stale
#: 611 B shell copies on another machine rendered without textures). The render entry checks the measured md5 of
#: overlays spawned from the canonical root against this manifest (an oracle).
OVERLAY_MANIFEST_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "record", "render_overlay_manifest.json"
    )
)

#: A project's own overlay manifest — a project that bakes its own datasets on its own data root states its list's
#: file here (written by `robots.objects.overlay_manifest` with this env set, outside the DexEngine tree). Unset = the
#: tree's rule below. The render marker names the file in force.
OVERLAY_MANIFEST_ENV = "DEXENGINE_OVERLAY_MANIFEST"

#: A checkout with an operations layer (OPS_DIR) checks the canonical list above; a checkout without it checks
#: the list its own overlay bake wrote (DATA_ROOT_OVERLAY_MANIFEST).
OPS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ops"))

#: The list without OPS_DIR, under the data root: written by the overlay bake (robots.objects.render_overlay →
#: robots.objects.overlay_manifest.record) for the overlays that bake wrote, merged by key — it bakes on its
#: own data root, so the copies between machines that the git list guards against are not there.
DATA_ROOT_OVERLAY_MANIFEST = "data/render_overlay_manifest.json"


def overlay_manifest_rule() -> tuple[str, str]:
    """(rule, list file in force): "project" — OVERLAY_MANIFEST_ENV set (a project's own list) · "canonical" —
    with OPS_DIR: the git list, which must be there · "data-root" — a checkout without OPS_DIR: the list its
    overlay bake wrote under the data root. The check reads this one file and the generator writes it."""
    v = os.environ.get(OVERLAY_MANIFEST_ENV) or None
    if v is not None:
        return "project", os.path.abspath(v)
    if os.path.isdir(OPS_DIR):
        return "canonical", OVERLAY_MANIFEST_FILE
    from dexengine.core.layout import roots as _roots

    return "data-root", _roots.data_abspath(DATA_ROOT_OVERLAY_MANIFEST)


def overlay_manifest_key(path: str) -> str:
    """Overlay path → manifest key = the part after the last `/data/`, as `data/…` (canonical, remote tree and eval
    shadow roots share the key). Dies without `/data/`."""
    p = os.path.abspath(str(path))
    i = p.rfind("/data/")
    if i < 0:
        raise ValueError(f"[OVERLAY_MANIFEST] overlay path has no /data/ layer, so no manifest key: {p}")
    return p[i + 1 :]


def overlay_manifest_check(paths: list[str], root: str | None) -> str:
    """Check the measured md5 of the spawned overlays (`*_render.usd`) against the manifest in force
    (overlay_manifest_rule) — returns the marker tail (`[OVERLAY_MANIFEST] <this>`). With a non-canonical root
    (OVERLAY_ROOT_ENV · deliberately different overlays such as a texture-x version) the check is skipped and the tail
    says so. A missing manifest file (without OPS_DIR: when overlays were spawned), a key not in it or a different md5
    dies (a stale copy, a shell, or a new bake without an updated manifest — copy the canonical overlays again for a
    stale copy; rerun the generator after a new bake; without OPS_DIR, bake the overlays again)."""
    import hashlib
    import json

    ovs = [p for p in paths if str(p).endswith(RENDER_USD_SUFFIX + ".usd")]
    if root is not None:
        return f"skipped — overlay root {root} (not canonical) · overlays {len(ovs)}"
    rule, mf = overlay_manifest_rule()
    if not os.path.isfile(mf):
        if rule == "data-root" and not ovs:
            return f"ok 0/0 · data-root list {mf} (a tree without the deployment layer · no overlays spawned)"
        how = {
            "canonical": "a tree with the deployment layer needs the canonical list from git — "
            "python3 -m dexengine.robots.objects.overlay_manifest",
            "project": f"the project list {OVERLAY_MANIFEST_ENV} names",
            "data-root": "a tree without the deployment layer: the overlay bake (render_overlay) "
            "writes it for the overlays it bakes",
        }[rule]
        raise FileNotFoundError(f"[OVERLAY_MANIFEST] overlay manifest missing {mf} — {how}")
    with open(mf, "rb") as fh:
        raw = fh.read()
    man = json.loads(raw)["overlays"]
    bad = []
    for p in ovs:
        k = overlay_manifest_key(p)
        with open(p, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        want = man.get(k, {}).get("md5")
        if md5 != want:
            bad.append(f"{k}(measured {md5[:8]} {os.path.getsize(p)} B · manifest {want[:8] if want else 'none'})")
    if bad:
        fix = {
            "canonical": "for a stale copy, copy the canonical overlays again; after a new bake, rerun "
            "python3 -m dexengine.robots.objects.overlay_manifest and commit",
            "project": f"rerun the generator for the project list ({OVERLAY_MANIFEST_ENV})",
            "data-root": "bake those overlays again (python3 -m dexengine.robots.objects.render_overlay)",
        }[rule]
        raise RuntimeError(
            f"[OVERLAY_MANIFEST] overlays differing from the manifest {mf} {len(bad)}/{len(ovs)}: "
            f"{'; '.join(bad[:4])}{' …' if len(bad) > 4 else ''} — {fix}"
        )
    own = {"canonical": "", "project": f" · project list {mf}", "data-root": f" · data-root list {mf} (no deployment)"}
    return f"ok {len(ovs)}/{len(ovs)} · manifest md5 {hashlib.md5(raw).hexdigest()[:12]}{own[rule]}"
