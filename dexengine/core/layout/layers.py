"""Data layer templates (`LAYERS`), per-dataset physical storage overrides (`STORAGE`), `render()` and the
object-asset reference grammar stored in task_info.json."""

from __future__ import annotations

import os
import re

from dexengine.core.layout.registry import EMBODIMENTS, dataset


def data_tops() -> frozenset[str]:
    """The top-level folders of the data layers — derived from the tables below (LAYERS ∪ STORAGE), so adding a
    layer widens it."""
    return frozenset(t.split("/", 1)[0] for t in (*LAYERS.values(), *STORAGE.values()))


def _within(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


#: **Articulated object bake generation** — the folder name (under `data/<ds>/usd_artic/`) of the articulated USDs this
#: code bakes and reads. The name carries the bake rule: part masses = the bake total split by part volume
#: (dexengine/robots/objects/part_mass.py · usd_build._link_masses). A change of the articulated bake rule takes a new
#: name, so a new bake lands beside the USDs an older code tree reads and never overwrites them; the rule fingerprint
#: that names it is pinned by tests/robots/test_artic_bake_gen.py. Rigid USDs stay in the usd layer. The asset's
#: surface points and keypoints (dexengine/data/artic/surface.py) live in the same folder, so they never rewrite the
#: meshes-layer files an older tree reads; the name pins the bake rule only — their staleness is surface_problem's.
ARTIC_BAKE_GEN = "pmvol1"

# ── Canonical template per stage (relative to the repo root) ──────────────────────────────────────────
# A trailing `{clip}`/`{tree}` may be omitted (when only the directory is needed).
# Every other placeholder is required.
LAYERS: dict[str, str] = {
    "raw": "data/{ds}/raw",  # source dataset copy (annotations·meshes; RGB optional)
    "mano": "data/{ds}/mano/{emb}/{clip}",
    "retarget": "data/{ds}/retarget/{hand}/{emb}/{clip}",
    "assets_objects": "data/{ds}/assets/objects",
    "assets_robot": "data/{ds}/assets/robots/{hand}",
    "visualization": "data/{ds}/visualization/{hand}/{emb}",
    "intake": "data/{ds}/INTAKE",
    "base": "data/{ds}/base/{hand}/{emb}/{clip}",
    "meshes": "data/{ds}/meshes",
    "usd": "data/{ds}/usd",
    # articulated object USDs (+ cooked hull dumps · render overlays · surface points and keypoints) of ARTIC_BAKE_GEN
    "usd_artic": f"data/{{ds}}/usd_artic/{ARTIC_BAKE_GEN}",
    "trees": "data/{ds}/trees/{combo}/{emb}/{tree}",
    "views": "data/{ds}/views",
    "precompute": "precompute/{ds}/robots/{combo}",
    "f0_root": "precompute/{ds}/f0_stage/processed/{ds}",
    "f0_stage": "precompute/{ds}/f0_stage/processed/{ds}/{hand}",
    "contact_base": "precompute/{ds}/contact_mano_base",
    "contact_all": "precompute/{ds}/contact_mano_all",
    # per-hand sidecar places (only clips in the hand's registry) — resolved by sidecar_rel alone
    "contact_base_own": "precompute/{ds}/contact_mano_base_own/{hand}",
    "contact_all_own": "precompute/{ds}/contact_mano_all_own/{hand}",
    "ikpool": "precompute/{ds}/ikpool",
    # ── stage-2 (grounding) helper layers — same shape as TACO's physical place, so no STORAGE override is needed
    # wrist_base: raw wrist cache (once per hand·clip): …/wrist_poses_{side}/<clip>.npz
    # gate_meta: gate stash·meta (table file = paths.gate_table); the caller adds the variant suffix
    "wrist_base": "precompute/{ds}/robots/{hand}",
    "gate_meta": "precompute/{ds}/robots_v5",
    "contact_derive": "precompute/{ds}/contact_derive/{combo}",  # derived contact-label output
    # ── Public data collection output — the two final places people look at:
    # collect      = one hand × dataset = one LeRobot dataset (this folder is the dataset root — meta/ · data/ ·
    #                videos/ …).
    #                Bundle pieces (one collect --out run) are built in the run layer run_collect_work; the driver's
    #                --merge on merges the pieces that pass checks into here (store.merge.merge_piece). Pieces and
    #                caches never live here.
    # collect_text = hand-agnostic text (texts.jsonl per dataset — the text track's one final table
    #                (run_collect_text/<track>/final) split by dataset, generated output, not a copy · writer
    #                text.pipeline.split_final). Fixed slot name "text" beside the combo slots — it never clashes
    #                with a combo name (rby1_<hand>).
    "collect": "data/{ds}/collect/{combo}",
    "collect_text": "data/{ds}/collect/text",
    # ── Object texture packs — render textures pre-baked in training-mesh face order for objects whose source has no
    # UV·material: `<object id>/{tex.png, st.npy, meta.json}`. Rule and source-kind owner
    # dexengine/robots/textures/rules.py (PACK_*) · writer dexengine/robots/textures/pack_write.py · reader the object
    # render-USD bake with --packs (off by default).
    "tex_packs": "data/{ds}/tex_packs",
    # ── Render-only scene objects for collection — objects in the captured scene that are not in the tree's
    # objects[] (hand-agnostic · measured on the axis hand's base trees): `index.json` ({clip: object count}) plus
    # `<clip>.json` for clips with objects > 0. Rule and writer dexengine/record/scene/render_only_build.py · shape
    # and reading dexengine/record/scene/render_only.py · reader scene.extras.decide (training and judge envs never
    # read it).
    "render_only": "data/{ds}/render_only",
}


# ── Physical storage overrides (per dataset, here only) — hand-agnostic input layers only; hand-axis layers
# (base · trees) have no override ─────
STORAGE: dict[tuple[str, str], str] = {
    ("taco", "raw"): "data/taco/raw_unzipped",
    ("taco", "mano"): "data/taco_retarget/processed/taco/mano/{emb}/{clip}",  # all 2,317 (retarget phase 1)
    ("taco", "retarget"): "data/taco_retarget/processed/taco/{hand}/{emb}/{clip}",
    # The **physical** place of object meshes = the retarget processing folder (same root as assets_robot). Object
    # references stored in task_info are logical names (object_path below resolves them through this row).
    ("taco", "assets_objects"): "data/taco_retarget/processed/taco/assets/objects",
    ("taco", "assets_robot"): "data/taco_retarget/processed/taco/assets/robots/{hand}",
    ("taco", "usd"): "data/usd_cache/objects_taco",
    ("taco", "views"): "data/taco/lad",
}


_PLACEHOLDER_RE = re.compile(r"\{([a-z]+)\}")
_OPTIONAL_TAIL = ("clip", "tree")


def render(layer: str, ds: str, **axes: str) -> str:
    """The repo-relative path of one stage. A missing required axis is a KeyError — no default is invented.

    >>> render("retarget", "hocap", hand="sharpa_wave", emb="bimanual", clip="subject_1__20231022_200657")
    'data/hocap/retarget/sharpa_wave/bimanual/subject_1__20231022_200657'
    >>> render("trees", "taco", combo="rby1_sharpa_wave", emb="bimanual")      # canonical template, tail omitted
    'data/taco/trees/rby1_sharpa_wave/bimanual'
    >>> render("retarget", "taco", hand="sharpa_wave", emb="bimanual")          # STORAGE override (hand-agnostic input)
    'data/taco_retarget/processed/taco/sharpa_wave/bimanual'
    """
    if layer not in LAYERS:
        raise KeyError(f"unknown layer {layer!r} — registered: {sorted(LAYERS)}")
    dataset(ds)  # registration check (KeyError if unknown)
    stray = set(axes) - {n for t in (*LAYERS.values(), *STORAGE.values()) for n in _PLACEHOLDER_RE.findall(t)}
    if stray:  # an axis no data layer has — never drop it silently
        raise KeyError(
            f"{layer}/{ds}: axes {sorted(stray)} appear in no data layer — are they run-layer (render_run) axes?"
        )
    if "emb" in axes and axes["emb"] not in EMBODIMENTS:
        raise ValueError(f"embodiment {axes['emb']!r} — must be one of {EMBODIMENTS}")
    tpl = STORAGE.get((ds, layer), LAYERS[layer])
    parts = []
    for seg in tpl.split("/"):
        names = _PLACEHOLDER_RE.findall(seg)
        if any(n in _OPTIONAL_TAIL and n not in axes for n in names):
            break  # tail (clip/tree) omitted → stop at the directory
        try:
            parts.append(seg.format(ds=ds, **axes))
        except KeyError as e:
            raise KeyError(f"{layer}/{ds}: axis {e.args[0]!r} is required (template {tpl})") from None
    return "/".join(parts)


#: Placement-variant name tail `<clip>_pv<NN>` (NN = variant number · two or more digits · k ≥ 100 = three) — home
#: of the name shape (dexengine.core.paths.PV_SFX_PAT · paths.variant_source re-export it · writer
#: paths.variant_name). The per-hand sidecar registration check reads it: a variant is a rigid move of its source
#: clip, so its base and contact sidecars equal the source's (augment.bake.alias_base) — a registered source puts
#: its variants in the per-hand place too.
PV_SFX_PAT = r"_pv\d{2,}"
_PV_SFX_RE = re.compile(PV_SFX_PAT + "$")


def variant_source(name: str) -> str:
    """Placement variant → the clip it was made from (`X_seg03_pv07` → `X_seg03`); a non-variant is returned as is
    (clip_id input)."""
    return _PV_SFX_RE.sub("", name)


# ── Object asset references — the grammar of task_info.json `objects[].mesh_dir / convex_dir` ─────────────────
# The task_info of outputs (base·trees·f0 shadow) writes object mesh folders as **logical names** = the canonical
# template LAYERS["assets_objects"] (the name before STORAGE) + `/<id>[/…]`. STORAGE decides the physical place, so a
# reader never joins the stored text to the data root directly but resolves it with object_path — a physical move
# (one STORAGE line) changes not one byte of stored output. Every stored object reference uses this grammar.
def _objects_logical_root(ds: str) -> str:
    dataset(ds)  # registration check (KeyError if unknown)
    return LAYERS["assets_objects"].format(ds=ds)


def _clean_rel(rel: str, what: str) -> str:
    """A tail relative to the object layer (`<id>` · `<id>/convex` …) — empty segments·`.`·`..`·absolute paths are
    outside the rule and die."""
    rel = str(rel).rstrip("/")
    if not rel or rel.startswith("/") or any(p in ("", ".", "..") for p in rel.split("/")):
        raise ValueError(f"{what}: object-layer tail {rel!r} is outside the rule (`<id>[/…]`, no empty segment·./..)")
    return rel


def object_ref(ds: str, rel: str) -> str:
    """Writer side — the **logical name** of object-layer tail `rel` (the text stored in task_info).

    >>> object_ref("taco", "116")
    'data/taco/assets/objects/116'
    """
    return f"{_objects_logical_root(ds)}/{_clean_rel(rel, f'object_ref({ds})')}"


def object_path(ds: str, ref: str) -> str:
    """Reader side — the **physical** path (repo-relative, STORAGE applied) of a stored object reference (logical
    name). Anything that is not a logical name is a ValueError — another dataset's reference, an absolute path or
    physical text is never accepted silently.

    >>> object_path("taco", "data/taco/assets/objects/116/convex")
    'data/taco_retarget/processed/taco/assets/objects/116/convex'
    >>> object_path("hocap", "data/hocap/assets/objects/G07_4")
    'data/hocap/assets/objects/G07_4'
    """
    root = _objects_logical_root(ds)
    s = str(ref)
    if not s.startswith(root + "/"):
        raise ValueError(
            f"object reference {s!r} is not a logical name `{root}/<id>[/…]` of dataset {ds} — check task_info"
        )
    return f"{render('assets_objects', ds)}/{_clean_rel(s[len(root) + 1 :], f'object_path({ds})')}"


def object_part_ref(ds: str, mesh_ref: str, part: str) -> str:
    """The **logical name** of an articulated-part file. A stored part path (task_info
    `objects[].articulation.parts.<k>`) is relative to the object reference `mesh_ref` (logical name `<root>/<id>`),
    and the rule has two shapes —
      · a file in the same folder: `visual.obj` · `top.obj` …                  → `<root>/<id>/<file…>`
      · a sibling object folder in the same layer: `../<sibling id>/<file…>` (part pair) → `<root>/<sibling id>/<file…>`
    Anything else (two or more `..` · `..` under a multi-segment reference · empty segment·`.`·absolute path) dies.
    The returned name has no `..`, so object_path accepts it as is — declaring and opening part files both go through
    this one rule.

    >>> object_part_ref("oakink2", "data/oakink2/assets/objects/left_a", "../right_b/visual.obj")
    'data/oakink2/assets/objects/right_b/visual.obj'
    >>> object_part_ref("arctic", "data/arctic/assets/objects/box", "top.obj")
    'data/arctic/assets/objects/box/top.obj'
    """
    root = _objects_logical_root(ds)
    s = str(mesh_ref)
    if not s.startswith(root + "/"):
        raise ValueError(f"object_part_ref({ds}): object reference {s!r} is not a logical name `{root}/<id>`")
    own = _clean_rel(s[len(root) + 1 :], f"object_part_ref({ds}) object reference")
    p = str(part)
    if p.startswith("../"):
        if "/" in own:
            raise ValueError(
                f"object_part_ref({ds}): sibling part {p!r} is in the rule only under a one-segment "
                f"object folder reference (`<id>`) — reference {s!r}"
            )
        rel = p[len("../") :]
    else:
        rel = f"{own}/{p}"
    return f"{root}/{_clean_rel(rel, f'object_part_ref({ds}) part {p!r}')}"


#: Cooked hull dump — one file per object USD, in the same folder (a bake output of the usd layer). Format and
#: reading of its content (the convex hulls PhysX cooked from that USD · source USD sha256 · cooking fingerprint) =
#: dexengine/data/objects/hull.py · generator = dexengine/robots/objects/cooked_hull.py.
COOKED_HULL_EXT = ".cooked.npz"


def cooked_hull_file(usd_file: str) -> str:
    """Object USD path → its cooked hull dump path (same folder · name stem + COOKED_HULL_EXT). Dies on a non-`.usd`.

    >>> cooked_hull_file("data/oakink2/usd_artic/pmvol1/a__b__hinge/object_artic.usd")
    'data/oakink2/usd_artic/pmvol1/a__b__hinge/object_artic.cooked.npz'
    """
    stem, ext = os.path.splitext(str(usd_file))
    if ext != ".usd":
        raise ValueError(f"{usd_file}: not an object USD (.usd) — cannot name its cooked hull dump")
    return stem + COOKED_HULL_EXT
