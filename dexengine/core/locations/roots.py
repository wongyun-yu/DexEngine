"""Roots: the code checkout, the data root, run roots, cluster trees, staging roots of the data stages, the IK
pool, and the retargeting runtime (interpreter, MANO model)."""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.locations.modules import data_layout
from dexengine.core.locations.names import WT

# ── This repo has four "roots" (one table) ─────────────────────────────────
#   WT                      root of the **code tree** this file belongs to (module constant above) — the worktree
#                           itself in a worktree. Tracked files (code · docs · retarget/ rig · tracked tables) are
#                           read here.
#   DEXENGINE_MAIN_ROOT       the checkout holding the **data tree** — the one root of every untracked data tree
#                           (data/ · precompute/ · logs/ · archive/). Unset = WT. Owner = main_root() below.
#                           In a worktree it names the main checkout, so the worktree runs with **zero symlinks**.
#                           Layer paths · f0 · pool · untracked URDF meshes (urdf_mesh) all live under this root.
#   DEXENGINE_SNAPSHOT_SRC · DEXENGINE_SNAP_DIR   variables of the earlier shell launcher's per-run script copy; the
#                           Python launcher writes no copy (manifest `SNAPSHOT=none`) and drops both from the trainer
#                           env (dexengine.rl.launch.plan).
#   DEXENGINE_SELFTEST_ROOT   (a **boundary**, not a root) the private temp root a fixture-building selftest declares
#                           — while it is set, the sandbox root and the run-layer write places must lie inside it,
#                           and the data root above inside it or inside the read root below. Outside → dies (rule
#                           owner dexengine.core.layout.selftest_confine).
#   DEXENGINE_SELFTEST_READ_ROOT  (part of the boundary) the read root named by a test that must **read** real data
#                           — only data-root resolution goes there; the write owner stays inside the boundary.
#                           Meaningless without the boundary.
# Shell-local `REPO`/`REPO_ROOT`/`WT` variables are **copies** of one of the four above, not a fifth root.


def main_root() -> Path:
    """The **data root** — where every untracked data tree lives (`data/…`, `precompute/…`, `logs/`, `archive/`).
    The repository root in the canonical checkout; only env `DEXENGINE_MAIN_ROOT` makes it differ (a worktree names
    the main checkout and then needs no symlinks). The one owner of "where is the data" — no second heuristic
    (the old `real_repo_root()` guessed it by resolving a `data/taco` symlink; deleted).
    """
    return data_root_of(WT)


def run_root() -> Path:
    """The tree the run layers (data_layout.RUN_LAYERS · RUN_SHARED — the hand roots `overfit/hands/<hand>` ·
    `docs/experiments/hands/<hand>` and the hand-less run layers) and the RL work areas (work_area) live in when a
    caller gives no `root` — run_root_of(main_root(), WT)."""
    return run_root_of(main_root(), WT)


def run_root_of(data_root, code_tree) -> Path:
    """The run tree of a process whose data root is `data_root` and whose code tree is `code_tree` — the one rule, by
    the data root's kind (data_layout.root_kind):
      canon, the code tree itself   → it (the canonical checkout: code, data and runs in one tree)
      canon, another checkout       → the code tree (a dev worktree naming the main checkout keeps its runs apart)
      generation                    → the data root (a chain on a generation root keeps its ledgers · audits ·
                                      campaigns there and never folds into the code checkout's canonical hand root —
                                      hand_root.assert_writable dies on such a write)
      folder                        → the data root (a release or copied data root: runs and logs never land in the
                                      installed code tree)
      shadow                        → the run tree of its base (data_shadow base — a shadow's processes are children
                                      of a run on the base and share its work areas)
    Run layers and work areas share this tree: finalize moves work-area pieces into hand roots with os.rename only
    (root_guard.guarded_import)."""
    r, t = Path(data_root), Path(code_tree)
    kind = data_layout.root_kind(str(r), str(t))
    if kind == "canon":
        return t
    if kind == "shadow":
        return run_root_of(data_layout.data_shadow(str(r))["base"], t)
    return r


def run_out(p, root=None) -> Path:
    """An output path of a run: absolute as given; a relative one is relative to the run tree (`root` when given, else
    run_root()) — never to the cwd (a release runs from its code tree, apart from its data root). Callers: the
    trainer's video folders (--video_dir · --videos_out_dir) and the launcher's default render/replay folders
    (data_layout.RUN_RENDER_OUT)."""
    p = Path(p)
    return p if p.is_absolute() else Path(root if root is not None else run_root()) / p


def work_area(name: str, root=None) -> Path:
    """The place of one RL work area (data_layout.RUN_WORK_AREAS: work_log = trainer logs · launch manifests · summary
    tables; work_ckpt = RUN_CKPT_ROOT, the run folders) — `<run tree>/<area>`, the run tree = `root` when given, else
    run_root(). Never the cwd: a release whose data root is apart from its code tree writes these under the data
    root."""
    return Path(root if root is not None else run_root()) / data_layout.RUN_WORK_AREAS[name]


def run_rel(p) -> Path:
    """`--rel` form of a run-layer path — what a shell in this code tree can use. Run layers in this code tree
    (canonical root · a generation root that is its own code tree) → repo-relative, the form transfer lists need.
    Run layers in a run tree apart from the code tree (a generation or folder data root given by env, run_root() ≠ WT)
    have no repo-relative form: the path under that root comes back absolute, so a `cd <code tree>` shell still
    writes into that root (transfer lists are refused there — data root ≠ this tree). A path under
    neither tree raises ValueError."""
    p = Path(p)
    try:
        return p.relative_to(WT)
    except ValueError:
        r = run_root()
        if r != WT and p.is_relative_to(r):
            return p
        raise


def data_root_of(tree: str | Path) -> Path:
    """The data root a command run **in `tree`** reads — env `DEXENGINE_MAIN_ROOT`, else `tree` itself (rule owner
    `data_layout.data_root`: env name · empty value). `main_root()` is this for the calling code tree; a caller that
    rewrites a command to run in another tree (dexengine.rl.judge.rejudge --relocate) asks for that tree's root, not its
    own."""
    return Path(data_layout.data_root(str(tree))).resolve()


def cluster_tree() -> Path:
    """Where a cluster pod unpacks the tree = this absolute path on the submitting machine (the top name of the code
    tar · the pod's tree variable). It must be the **main checkout** (`.git` is a folder) and that tree must find
    IsaacLab without env — the pod calls the location owner (scripts/_isaac_py.sh) in the unpacked tree without
    ISAACLAB_DIR, so the owner's `isaac_lab_of_tree` answers where (not restated here). Launched from a linked worktree
    (`.git` is a file), the pod would use that path as its tree, miss IsaacLab and die right before training — in a
    worktree, name the main checkout with DEXENGINE_MAIN_ROOT. Otherwise the launcher dies loudly. Called at
    pod launch points (job submission and code packing)."""
    import subprocess  # noqa: PLC0415 — only the emission points need it (the shell's per-clip calls do not)

    root = main_root()
    if not (root / ".git").is_dir():
        raise SystemExit(
            f"[CLUSTER_TREE] {root} is not a main checkout (.git is not a folder) — "
            "launch with DEXENGINE_MAIN_ROOT=<main checkout> (the pod uses this path as its tree)"
        )
    r = subprocess.run(
        ["bash", "-c", '. "$0" && isaac_lab_of_tree "$1"', str(WT / "scripts/_isaac_py.sh"), str(root)],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise SystemExit(
            f"[CLUSTER_TREE] {r.stderr.strip() or f'isaac_lab_of_tree rc {r.returncode}'} — the pod "
            "uses the IsaacLab its tree finds without env (location owner scripts/_isaac_py.sh)"
        )
    return root


def urdf_mesh(urdf, filename: str) -> Path:
    """The file a URDF `<mesh filename>` names (resolved) — the window onto the one rule `data_layout.urdf_mesh`:
    tracked meshes from this code tree, untracked ones (the hand vendor stages `data/_stage/<hand>`) from the data root
    (main_root), so a worktree run with DEXENGINE_MAIN_ROOT needs no `data/_stage` of its own. Every URDF mesh reader
    (sim_collider/wrist_hull_geom · torso_split · dexengine.placement.collision.cross_sets · f0.repair ·
    dexengine.retarget.f0.prejudge · the URDF generators) goes through it. The URDF path is taken as given (absolute,
    symlinks not followed — a fixture repo links `source/`), only the answer is resolved."""
    return Path(data_layout.urdf_mesh(str(urdf), filename, str(WT))).resolve()


def repo_rel_path(rel: str) -> Path:
    """The **place** of one repository-relative path — only a window; the rule is dexengine.core.layout.repo_place
    (a data-layer head → the data root, else a tracked file of this code tree). Readers of the hand vendor URDF
    template (HandSpec.vendor_urdf) resolve through it."""
    return Path(data_layout.repo_place(rel, str(WT)))


def owner_root() -> Path:
    """The owner-ledger root — the per-hand canonical clip-owner ledger is acceptance.ledger_path(hand, this root).
    The layout file's `owner_root` (a site names its ledger place there), else the data root (main_root). Printed by
    `python -m dexengine.core.paths --what owner-root`."""
    return Path(data_layout.layout_config().owner_root or main_root())


def output_root() -> Path:
    """Root of the **written layers** (dexengine.core.layout.SANDBOX_LAYERS) — the sandbox root for an isolated run
    (env DEXENGINE_SANDBOX_ROOT), else the data root (main_root). Input layers always come from main_root (layer_path
    decides). The value's rule is dexengine.core.layout.sandbox_root alone."""
    s = data_layout.sandbox_root()
    return Path(s).resolve() if s else main_root()


def data_rel(p) -> str:
    """The **display/record relative form** of a path — relative to the data root (main_root) when under it, to the
    code tree (WT) when under that, else absolute. In a worktree a data path is not under WT, so `p.relative_to(WT)`
    would raise ValueError."""
    p = Path(p)
    for root in (main_root(), WT):
        if p.is_relative_to(root):
            return str(p.relative_to(root))
    return str(p)


def retarget_stage_root(hand: str) -> Path:
    """Stage root of stage-1 retargeting, `data/_stage/reretarget_<hand>` — the stage-1 writer · the
    stamps · the stage-1 mover all take it from here (the shells never spell it out on their own). hand = the
    run hand id."""
    if not hand:
        raise ValueError("retarget_stage_root: a hand is required (no default)")
    return main_root() / data_layout.STAGE_ROOTS["retarget_stage"].format(handname=hand)


def reread_stage_root(dataset: str) -> Path:
    """Stage root of the source re-read, `data/_stage/reread_<dataset>` — written by the reader · decompose ·
    contact · first-hand retargeting (dexengine.data.read_raw.stage) and read by the intake (dexengine.data.intake
    --ingest). No default dataset."""
    if not dataset:
        raise ValueError("reread_stage_root: a dataset is required (no default)")
    return main_root() / data_layout.STAGE_ROOTS["reread_stage"].format(ds=dataset)


def f0_root(dataset: str) -> Path:
    """f0 shadow staging root (the same rule as `_F0ROOT` in dexengine.placement.judge). If env DEXENGINE_F0_STAGE is
    set, that wins (regardless of dataset). The dataset has no default — the caller names it."""
    if not dataset:
        raise ValueError("f0_root: a dataset is required (no default)")
    return Path(
        os.environ.get("DEXENGINE_F0_STAGE") or (output_root() / data_layout.render("f0_root", dataset))
    )  # written layer: sandbox root if isolated


# Where the canonical IK pool is **stored** — a storage fact, not a dataset choice: the pool is dataset-
# independent (combo-independent, arm-generation dependent; data_layout "ikpool") and lives under the TACO
# precompute root for historical reasons. pool_root() is its only reader.
# It is NOT a default for any dataset input.
POOL_HOME_DATASET = "taco"


def pool_root() -> Path:
    """Canonical IK pool (`precompute/taco/ikpool`, baked with base −0.70 and a chest00-family URDF).

    Judging (judge), baking (stage_dp), placement (place) and the chain (chain_fast/stage2) must
    all look at the **same pool** — a literal per file lets judging and baking look at pools 12.7cm
    apart. Only `DEXENGINE_POOL_ROOT` makes it differ (for pose experiments).
    """
    # The real thing is **one canonical checkout** (the data root, main_root) — a path relative to the
    # worktree lets copies diverge between worktree and main, and the judging seed then depends on which
    # checkout you run from.
    return Path(
        os.environ.get("DEXENGINE_POOL_ROOT") or (main_root() / data_layout.render("ikpool", POOL_HOME_DATASET))
    )


#: Env naming the retargeting interpreter explicitly (wins over the code tree's venv; set but missing = error).
RETARGET_PY_ENV = "RETARGET_PY"
#: The code tree's venv interpreter, relative to the code tree (README install: `python -m venv .venv`).
RETARGET_VENV_PY = ".venv/bin/python"


def retarget_py() -> str:
    """The retargeting python (mujoco·trimesh·smplx·pinocchio…) — the first of retarget_py_candidates().

    A venv belongs to the **code tree**, not the data: env `RETARGET_PY` when set, else `<code tree>/.venv/bin/python`
    (gitignored; WT = the tree this package is imported from). The data root is never searched — a release keeps its
    data apart from the checkout, and a dev worktree without its own .venv names one with RETARGET_PY. A missing file
    makes require_retarget_py die **at once** — not after thousands of clips have failed.
    """
    return retarget_py_candidates()[0]


def mano_model_dir() -> Path:
    """Owner of the MANO hand model place (MANO_LEFT/RIGHT.pkl) — `data/body_models/mano` under the data root.
    Returns the path only, so modules that merely import dexengine.record.viz.mano_render (dexengine.data.contact.common
    · cases.taxonomy …) load without MANO. When it is missing, the actual load
    (dexengine.record.viz.mano_render.get_mano) dies with the path from smplx."""
    return main_root() / "data/body_models/mano"


def retarget_py_candidates() -> list[str]:
    """The places searched, in order — exactly one: an explicit `RETARGET_PY` (set = the only candidate; a set but
    missing value is not replaced by the venv), else the code tree's `.venv`. A list, so the failure message can
    carry it."""
    p = os.environ.get(RETARGET_PY_ENV)
    return [p] if p else [str(WT / RETARGET_VENV_PY)]


def require_retarget_py() -> str:
    """Returns it only after checking that it exists. Missing = immediate InputMissingError (rc 2) naming the places
    tried."""
    p = retarget_py()
    if not os.path.isfile(p):
        env = os.environ.get(RETARGET_PY_ENV)
        tried = (
            f"{RETARGET_PY_ENV}={env} (set — the code tree's {RETARGET_VENV_PY} is not searched)"
            if env
            else f"{WT / RETARGET_VENV_PY} (the code tree's venv; {RETARGET_PY_ENV} unset)"
        )
        raise InputMissingError(
            f"[paths] no retargeting python: {p}\n"
            f"  searched:\n    {tried}\n"
            f"  create the venv in the code tree ({WT}: python -m venv .venv) or set {RETARGET_PY_ENV} to an "
            "interpreter (the data root is not searched)."
        )
    return p
