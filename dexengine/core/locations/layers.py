"""Per-dataset data layers: robots / trees / meta / IK / wrist / placement / gate folders, sidecars, entry-layer
roots and the source layer of a dataset."""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.core.locations.hands import hand_id, hand_spec_dir
from dexengine.core.locations.modules import data_layout
from dexengine.core.locations.names import PLACEMENT_SFX, WT, ik_suffix, tree_name
from dexengine.core.locations.roots import main_root, output_root


def bake_outputs(clip: str, tree_root, robots_dir) -> dict:
    """**Where one clip's bake outputs live** — the gate and the palm oracle find the files from a chunk's bake success
    list (baked_ok) and this function. No per-chunk symlinks to these four places: they would be a second statement
    of the same fact (where it was baked).
      tree_root   baked tree folder (stage 2's TROOT)
      robots_dir  placement / arm IK root (stage 2's W2 = robots/<combo>)
    Returns {tree, placement, ik_right, ik_left} (Paths that may not exist — the reader checks)."""
    t = tree_name(clip)
    rd = Path(robots_dir)
    return {
        "tree": Path(tree_root) / t,
        "placement": rd / "placement" / f"{t}{PLACEMENT_SFX}",
        "ik_right": rd / f"arm_ik_poses_right{ik_suffix()}" / f"{t}.npz",
        "ik_left": rd / f"arm_ik_poses_left{ik_suffix()}" / f"{t}.npz",
    }


# There is **no** default dataset or embodiment here. The functions below take them as required arguments, so a
# caller that forgets an axis fails instead of silently getting one.


def layer_path(layer: str, dataset: str, **axes) -> Path:
    """The **absolute path** of a layer — every layer (`data/…` and `precompute/…`) sits under the data root
    main_root().
    In an isolated run (env DEXENGINE_SANDBOX_ROOT) the layers a run **writes** (dexengine.core.layout.SANDBOX_LAYERS)
    sit under the sandbox root instead — output_root(); the input layers stay on the data root.

    A missing axis raises KeyError — no default is invented. A frozen generation is not a layer: its roots are
    `roster_root(name, kind)` (dexengine.core.layout.NAMED_ROSTERS — the preserved place).
        layer_path("retarget", "show3d", hand="wuji2", emb="bimanual", clip=C)
        layer_path("trees", "taco", combo="rby1_sharpa_wave", emb="bimanual")       # == trees_dir("rby1_sharpa_wave")
    """
    root = output_root() if layer in data_layout.SANDBOX_LAYERS else main_root()  # isolation moves written layers
    p = root / data_layout.render(layer, dataset, **axes)
    if root != WT:  # root outside the code tree — dies if it resolves into another root
        data_layout.confine_layer(str(p), layer, str(root), str(WT))
    return p


#: Stage-1 run guard: a step run (stage1.plan.emit_run) carries its id and the sidecar layers written at or before it.
STAGE1_STEP_ENV, STAGE1_SIDECARS_ENV = "DEXENGINE_STAGE1_STEP", "DEXENGINE_STAGE1_SIDECARS"


def sidecar_file(layer: str, dataset: str, clip: str, side: str, hand: str) -> Path:
    """Absolute path of one contact sidecar file (layer contact_base|contact_all) — the placement rule (shared vs per
    hand) is owned by dexengine.core.layout.sidecar_rel. hand = spec directory ('' if unknown — dies for a clip
    registered per hand). Inside a stage-1 step run (STAGE1_STEP_ENV set) a layer no step at or before it writes dies:
    the step would read a later step's output (absent from an empty root, an older generation's otherwise)."""
    step = os.environ.get(STAGE1_STEP_ENV)
    if step is not None and layer not in os.environ.get(STAGE1_SIDECARS_ENV, "").split(","):
        raise RuntimeError(
            f"[STAGE1_ORDER] stage-1 step {step} opens the {layer} sidecar, which no step at or before it writes "
            f"(written so far: {os.environ.get(STAGE1_SIDECARS_ENV) or 'none'}; stage1.table.check_inputs)"
        )
    root = main_root()
    return root / data_layout.sidecar_rel(root, layer, dataset, clip, side, hand)


def units_registry(dataset: str, emb: str, data_root: Path | str | None = None) -> Path:
    """The hand-agnostic D36 piece registry `data/<ds>/INTAKE/units/<emb>.json` (producer
    segment.split; readers ask here). `data_root` defaults to the data root main_root()."""
    droot = Path(data_root if data_root is not None else main_root())
    return droot / data_layout.render("intake", dataset) / "units" / f"{emb}.json"


# ── Directories (clip-independent) — the functions below are the window onto layer_path (they do not
#    re-type the values). dataset/emb/side are required: no default ─────────────
def robots_dir(combo: str, dataset: str) -> Path:
    """Per-combo output root. **The real thing is always under the repository root** (watch
    out for symlinks)."""
    return layer_path("precompute", dataset, combo=combo)


def trees_dir(combo: str, dataset: str, emb: str) -> Path:
    return layer_path("trees", dataset, combo=combo, emb=emb)


def meta_dir(combo: str, dataset: str) -> Path:
    """Where the promotion ledger (status/bake/universe · PROVENANCE) lives (meta_full_run)."""
    return robots_dir(combo, dataset) / "meta_full_run"


def ik_dir(combo: str, side: str, dataset: str) -> Path:
    return robots_dir(combo, dataset) / f"arm_ik_poses_{side}{ik_suffix()}"


def wrist_dir(combo: str, side: str, dataset: str) -> Path:
    return robots_dir(combo, dataset) / f"wrist_poses_{side}"


def placement_dir(combo: str, dataset: str) -> Path:
    return robots_dir(combo, dataset) / "placement"


# ── Stage 2 auxiliary layers (9-dataset grounding) — the values are owned by
#    dexengine.core.layout.LAYERS ─────────────────────────────────────────────
def wrist_base_dir(hand: str, side: str, dataset: str) -> Path:
    """Wrist RAW cache (1× per (hand, clip); independent of combo and placement):
    <wrist_base>/wrist_poses_<side>/<clip>.npz. `hand` is the retargeting directory name."""
    return layer_path("wrist_base", dataset, hand=hand) / f"wrist_poses_{side}"


def gate_table(meta_root, combo: str, tag: str) -> Path:
    """The gate's one output: the per-chunk table (one row per clip × side, verdict and reason columns). The gate writes
    it, the stage 2 cover check and the status ledger read it, the clip-fate page reads the reason of GUARD_FAIL clips.
    `meta_root` = the gate meta root, `layer_path("gate_meta", dataset)` + the caller's variant suffix."""
    return Path(meta_root) / combo / "meta" / f"gate_{tag}.tsv"


def contact_derive_dir(combo: str, dataset: str) -> Path:
    """Output root of the contact-label derivation."""
    return layer_path("contact_derive", dataset, combo=combo)


def contact_derive_ledger(combo: str, dataset: str) -> Path:
    """The contact derivation's ledger (one row per clip × side) — its completion record. The producer's --apply run
    (dexengine/data/contact/derive.py → dexengine.data.contact.derive_util.finalize_outputs) writes it and re-derives
    fails.txt / zero_contact.txt from it; a DRY run writes the mode-tagged names (mode_record), never this one; the
    bimanual entry's layer-root evidence (layer_root_records) names it."""
    return contact_derive_dir(combo, dataset) / "ledger.tsv"


# ── Layer roots the training entry requires ─────────────────────────────────────────────────────────────────────
#: Layer roots that must EXIST before a training launch, per embodiment — their existence is the entry's evidence
#: that grounding (stage 2) and contact derivation finished for the combo. The launcher's bimanual preflight
#: (dexengine/rl/launch/preflight.py, check_bimanual) reads this through entry_layer_roots (CLI: `--what
#: entry-layers`) and dies on a missing root; the one-hand path has no such check — empty.
#: dexengine/placement/bundle/run_inputs.py states, per required root, the file that carries the root to the GPU host
#: (layer_root_records), so the input audit measures exactly what this check needs (one list, two readers).
ENTRY_LAYER_ROOTS: dict[str, tuple[str, ...]] = {
    "bimanual": ("trees", "contact_derive", "precompute"),
    "right": (),
    "left": (),
}
#: The completion record of a required layer whose root holds no file a run reads (path rule per layer, not a listing:
#: the statement is derived on the GPU host too and must not depend on that machine's directory contents).
_LAYER_RECORD = {"contact_derive": lambda dataset, combo: contact_derive_ledger(combo, dataset)}


def entry_layer_roots(dataset: str, combo: str, emb: str) -> list[tuple[str, Path]]:
    """[(layer, root)] the training entry of `emb` requires (ENTRY_LAYER_ROOTS) for this dataset · combo (with its
    variant suffix — the combo the launch composes) — an unknown embodiment dies (no default)."""
    if emb not in ENTRY_LAYER_ROOTS:
        raise ValueError(f"embodiment {emb!r} — not in the entry layer-root table {sorted(ENTRY_LAYER_ROOTS)}")
    return [(ly, layer_path(ly, dataset, combo=combo, emb=emb)) for ly in ENTRY_LAYER_ROOTS[emb]]


def layer_root_records(dataset: str, combo: str, emb: str, stated) -> list[Path]:
    """Files that carry each root entry_layer_roots requires to the GPU host, beyond `stated` (paths a run already
    reads): a root some stated path lies under is carried by it; any other root by its layer's completion record
    (_LAYER_RECORD). A root with neither raises ValueError (its existence could not be stated)."""
    stated = [Path(p) for p in stated]
    out = []
    for layer, root in entry_layer_roots(dataset, combo, emb):
        if any(p.is_relative_to(root) for p in stated):
            continue
        if layer not in _LAYER_RECORD:
            raise ValueError(
                f"no stated path lies under the entry-required layer root {root} and layer {layer!r} "
                f"has no completion-record rule (registered: {sorted(_LAYER_RECORD)}) — the root cannot "
                f"be carried to the GPU host"
            )
        out.append(_LAYER_RECORD[layer](dataset, combo))
    return out


# ── Stage 2 source layer · reference universe — these two functions are
#    the only owner of the "which layer do we read" decision ──────────────────
SRC_LAYERS = ("base",)  # base is the only source layer


def default_src_layer(dataset: str) -> str:
    """The layer the stage 2 chain (f0, judging, baking) reads clips from: **base, for every dataset**.
    TACO's base layer (data/taco/base/<hand>/bimanual, 2,317 clips, qpos byte-identical to the stage 1
    staging) is the source, so TACO gets `case` (active sides auto) and `support` like the other nine
    datasets. The stage 2 chain reads this layer only."""
    return "base"


def ref_base_dir(dataset: str, hand: str, emb: str, src_layer: str | None = None) -> Path:
    """Root of the stage 2 chain's **reference universe** (children = clip directories) — read by the
    bake's bl, the gate's T_orig and the contact derivation's T_base. The reference is always the base
    layer of the dataset (rule set v2 — the tree compares against itself)."""
    src_layer = src_layer or default_src_layer(dataset)
    if src_layer != "base":
        raise ValueError(f"src_layer={src_layer!r} — must be one of {SRC_LAYERS} (base is the only source layer)")
    hand_dir = hand_spec_dir(hand_id(hand))  # hand id or spec directory → directory; unregistered → KeyError
    return layer_path("base", dataset, hand=hand_dir, emb=emb)


# ── Combo derivation (used by the training entry points) ────────────────────
def roster_root(name: str, kind: str) -> Path:
    """Absolute root `kind` (base · trees) of the frozen generation `name` — dexengine.core.layout.roster_root on the
    data root.
    Every reader of a frozen generation goes through this (never the roster's `origin` text, never a layer path)."""
    return main_root() / data_layout.roster_root(name, kind)


def taco_source_root() -> Path:
    """Absolute location of the TACO source tree — owner dexengine.core.layout.taco_source_root."""
    return main_root() / data_layout.taco_source_root()


def named_roster_of(p: str | os.PathLike) -> str:
    """The named roster whose preserved **trees** root is `p` (realpath), or "" — a frozen generation's trees are
    known by their place, not by a folder name."""
    rp = os.path.realpath(p)
    return next((n for n in data_layout.NAMED_ROSTERS if os.path.realpath(roster_root(n, "trees")) == rp), "")
