"""The data root: where the layer templates are joined (configuration, environment, caller default), the identity
check of a root given from outside, the sandbox root of isolated runs, and repository-relative path placement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dexengine.core.layout.generations import GENERATION_FILE, data_generation
from dexengine.core.layout.layers import LAYERS, _within, data_tops
from dexengine.core.layout.registry import DATA_ROOT_ENV, DATASETS
from dexengine.core.layout.shadow import ROOT_TAG, data_shadow, shadow_guard

#: env: a layout file that replaces configs/layout.yaml (dexengine.core.config.ENV_VARS)
LAYOUT_CONFIG_ENV = "DEXENGINE_LAYOUT_CONFIG"


@dataclass(frozen=True)
class LayoutConfig:
    """`configs/layout.yaml` (or the file named by DEXENGINE_LAYOUT_CONFIG). `data_root` = the folder that holds data/,
    precompute/, overfit/ and logs/; "" = the caller's default (the code checkout for scripts, the working directory
    for relative helpers). The record storage roots ("" = inside the data root): `collect_aug_root` = the augmented
    round's final datasets, `collect_traj_archive` = the trajectory archive (stage1/ · jobs/), `collect_archive_old_aug`
    = older read-only places of the augmented piece archive. Deployment: `owner_root` = the root of the per-hand
    RL clip-owner ledgers ("" = the data root; paths.owner_root), `hosts` = the RL campaign hosts {label: hostname}
    (empty = this machine alone; dexengine.rl.cell_state.hosts), `host_gpu_count_cmd` = {label: shell command, run
    from the code tree, that prints the GPU count of that host} for hosts other than the one a process runs on."""

    data_root: str = ""
    collect_aug_root: str = ""
    collect_traj_archive: str = ""
    collect_archive_old_aug: tuple[str, ...] = ()
    owner_root: str = ""
    hosts: dict[str, str] = field(default_factory=dict)
    host_gpu_count_cmd: dict[str, str] = field(default_factory=dict)


_LAYOUT_CFG: list = []
#: Marker of the layout file a process read: `[LAYOUT] <absolute path> pid=<pid>` on stderr, once per read (a process
#: that never needs a layout value prints none; " absent: defaults" follows a missing default file). A launcher
#: can check that its processes name its own file; a process started without
#: DEXENGINE_LAYOUT_CONFIG names configs/layout.yaml.
LAYOUT_MARK = "[LAYOUT]"


def layout_config() -> LayoutConfig:
    """The layout configuration (read once per process): the file named by DEXENGINE_LAYOUT_CONFIG, else
    configs/layout.yaml (a missing default file = the defaults). Prints LAYOUT_MARK with the file on stderr."""
    if not _LAYOUT_CFG:
        import sys  # noqa: PLC0415

        from dexengine.core import config as _config  # noqa: PLC0415 — yaml only when the env does not decide

        given = os.environ.get(LAYOUT_CONFIG_ENV) or None
        _LAYOUT_CFG.append(_config.load(LayoutConfig, "layout", path=given))
        f = os.path.abspath(given or _config.config_path("layout"))
        print(
            f"{LAYOUT_MARK} {f} pid={os.getpid()}" + ("" if os.path.isfile(f) else " absent: defaults"),
            file=sys.stderr,
            flush=True,
        )
    return _LAYOUT_CFG[0]


def configured_data_root() -> str:
    """The data root given from outside: env `DEXENGINE_MAIN_ROOT` (also set by `--data-root`), else
    `configs/layout.yaml` data_root, else ""."""
    return os.environ.get(DATA_ROOT_ENV) or layout_config().data_root


def data_root(default: str) -> str:
    """The data root: `configured_data_root()`, else `default` (the caller's rule: a script passes its code tree).
    A root given from outside is identified (check_data_root): an unregistered root that links into another root's
    data raises."""
    given = configured_data_root()
    root = given or default
    return check_data_root(root, default) if given else root


# ── Identity of a data root — a root that links into another root's data (an overlay root) must be registered ───
# Why: a scratch root built as an overlay of the canonical data (symlinks into the main checkout, folder- or file-level)
# sends every write through a link into the canonical tree, and nothing says so. A root is one of:
#   canon       the caller's own code tree, or a main checkout (`.git` is a folder)
#   generation  it carries GENERATION_FILE (registered — data_generation)
#   shadow      it carries SHADOW_FILE (a registered overlay — data_shadow): its links reach only its recorded base,
#               and a process whose data root it is dies on a write into the base's data (shadow_guard)
#   folder      anything else: a self-contained copy (a delivered bundle folder). It may not link into another
#               data root — an unregistered overlay dies at resolution (check_data_root).
# For generation and folder roots a resolved layer path must physically live in the root, and so must its entries
# (confine_layer — a write through a file-level link lands in the other root); a registered generation may link the
# layers nobody writes (GENERATION_LINK_LAYERS) to read the canonical originals.
GENERATION_LINK_LAYERS = ("raw", "assets_objects")
_CHECKED: set[str] = set()


def owner_root(path: str) -> str:
    """The data root that physically holds `path`: the nearest ancestor of its real path that is a main checkout
    (`.git` is a folder) or a generation root (GENERATION_FILE) — "" when none (storage outside any root)."""
    p = os.path.realpath(path)
    while True:
        if os.path.isdir(os.path.join(p, ".git")) or os.path.isfile(os.path.join(p, GENERATION_FILE)):
            return p
        up = os.path.dirname(p)
        if up == p:
            return ""
        p = up


def root_kind(root: str, code_tree: str = "") -> str:
    """canon · generation · shadow · folder (the table above)."""
    r = os.path.realpath(root)
    if (code_tree and r == os.path.realpath(code_tree)) or os.path.isdir(os.path.join(r, ".git")):
        return "canon"
    if data_generation(r) is not None:
        return "generation"
    return "shadow" if data_shadow(r) is not None else "folder"


def foreign_links(d: str, root: str, depth: int) -> list[str]:
    """`<link> → <real place>` of `d` itself and of the entries under it (`depth` levels; a linked folder is not
    entered) that are symlinks into ANOTHER data root (owner_root ≠ root). A link inside the root or to storage that
    no root owns is not one."""
    r, out = os.path.realpath(root), []

    def foreign(p: str) -> bool:
        real = os.path.realpath(p)
        return not _within(real, r) and owner_root(real) not in ("", r)

    if os.path.islink(d):
        return [f"{d} → {os.path.realpath(d)}"] if foreign(d) else []
    try:
        ents = list(os.scandir(d)) if depth > 0 else []
    except OSError:
        return out
    for e in ents:
        if e.is_symlink():
            if foreign(e.path):
                out.append(f"{e.path} → {os.path.realpath(e.path)}")
        elif depth > 1 and e.is_dir(follow_symlinks=False):
            out += foreign_links(e.path, root, depth - 1)
    return out


def check_data_root(root: str, code_tree: str = "") -> str:
    """A data root given by env states what it is (the table above) → `root`. An unregistered folder whose layer
    folders (the first path parts of LAYERS, two levels down — `data/<ds>/<layer>`) link into another data root is
    an overlay of that root: it dies (register it as a generation or build it from its own copies)."""
    key = os.path.realpath(root)
    if key in _CHECKED:
        return root
    kind = root_kind(root, code_tree)
    if kind == "folder":
        bad = [
            x
            for top in sorted({t.split("/")[0] for t in LAYERS.values()})
            for x in foreign_links(os.path.join(key, top), key, 2)
        ]
        if bad:
            raise SystemExit(
                f"{ROOT_TAG} ✗ {root} is an unregistered overlay root — no {GENERATION_FILE}, yet "
                f"{len(bad)} symlinks into another data root (e.g. {bad[0]}) — writes would land in that "
                "root. Register it as a generation (COMPARE_GENERATIONS) or build it from its own copies "
                "without links"
            )
    if kind == "shadow":
        base = data_shadow(key)["base"]
        bad = [
            x
            for top in sorted({t.split("/")[0] for t in LAYERS.values()})
            for x in foreign_links(os.path.join(key, top), key, 2)
            if not _within(x.rsplit(" → ", 1)[1], base)
        ]
        if bad:
            raise SystemExit(
                f"{ROOT_TAG} ✗ {len(bad)} links of the shadow root {root} go to another root outside "
                f"its base {base} (e.g. {bad[0]})"
            )
        shadow_guard(key, base)
    _CHECKED.add(key)
    return root


def confine_layer(path: str, layer: str, root: str, code_tree: str = "") -> str:
    """A layer path resolved under a generation or folder root physically lives in that root — the path itself and
    the entries of its folder, two levels (a file-level link: a write through it lands in the other root). Else the
    process dies. Unchecked: a canonical root · GENERATION_LINK_LAYERS of a registered generation."""
    kind = root_kind(root, code_tree)
    if kind == "canon" or (kind == "generation" and layer in GENERATION_LINK_LAYERS) or path in _CHECKED:
        return path
    r = os.path.realpath(root)
    real = os.path.realpath(path)
    if _within(real, r):
        bad = foreign_links(real, r, 2)
    else:
        bad = [f"{path} → {real}"] if owner_root(real) not in ("", r) else []
    if bad and kind == "shadow":  # links into the base are reads (shadow_guard blocks writes)
        base = data_shadow(r)["base"]
        bad = [x for x in bad if not _within(x.rsplit(" → ", 1)[1], base)]
    if bad:
        note = f" (layers a generation may link for reading = {GENERATION_LINK_LAYERS})" if kind == "generation" else ""
        raise SystemExit(
            f"{ROOT_TAG} ✗ the place {path} of layer {layer} resolves outside the data root {root} "
            f"({kind}) into another root: {len(bad)} symlinks (e.g. {bad[0]}) — writes would go to that "
            f"(canonical) root. Replace the links with copies{note}"
        )
    _CHECKED.add(path)
    return path


def holds_data(tree: str) -> bool:
    """`tree` carries data layers of its own — an intake folder of a registered dataset (every ingested dataset has
    one). A code checkout that holds data is a canonical data root: its run layers describe that data only."""
    return any(os.path.isdir(os.path.join(tree, LAYERS["intake"].replace("{ds}", ds))) for ds in DATASETS)


def data_path(rel: str) -> str:
    """Join a repository-relative path from a template onto the data root. Without `DEXENGINE_MAIN_ROOT` it comes back
    **as is** (relative to the working folder — byte-identical to the plain string). An absolute path stays as is
    (os.path.join rule)."""
    root = data_root("")
    return os.path.normpath(os.path.join(root, rel)) if root else rel


def data_abspath(rel: str) -> str:
    """Absolute form of data_path — from the working folder when there is no root (same as plain os.path.abspath)."""
    return os.path.abspath(data_path(rel))


def repo_place(rel: str, code_root: str) -> str:
    """Place (absolute) of the repository-relative path `rel` — the data root (data_root(code_root)) when its first
    part is a data-layer head (data_tops), else the code tree `code_root` (tracked files). The same rule as urdf_mesh ②
    — readers of the hand vendor URDF template resolve through it (HandSpec.vendor_urdf holds both the data layer
    `data/_stage/<hand>/…` and tracked assets `source/…`; window dexengine.core.paths.repo_rel_path).
    A path leaving the repository (absolute · `..`) raises ValueError."""
    r = os.path.normpath(str(rel).lstrip("/"))
    if r == "." or r.startswith(".."):
        raise ValueError(f"not a repository-relative path: {rel!r}")
    root = data_root(code_root) if r.split(os.sep, 1)[0] in data_tops() else code_root
    return os.path.normpath(os.path.join(os.path.abspath(root), r))


def urdf_mesh(urdf: str, filename: str, code_root: str) -> str:
    """The **logical path** of a URDF `<mesh filename>` (absolute · normalized as text · symlinks not followed) — the
    only rule of URDF mesh resolution (the window `dexengine.core.paths.urdf_mesh` `.resolve()`s it into the file to
    read) · `urdf` = that URDF (for an output not written yet, where it will be written).
    ① Form: `package://<pkg>/rest` = rest under the URDF folder, else rest under its parent (ROS `<pkg>/urdf/` layout —
       both measured) · `file://rest` · relative = from the URDF folder · absolute = as is · `..` is folded as text
       (folders that do not exist yet).
    ② Place: inside the code tree with a first part that is a data-layer head (data_tops — tracked or not: tracked
       files under it move too · no such URDF reference today) → the same relative place under the **data root**
       (data_root) — code comes from the run tree, data (the vendor stage `data/_stage/<hand>`) from the data root.
       Identity when the data root = the code tree (canonical checkout · no env).
    The path is returned even when the file is missing — the existence check and its named failure belong to the
    caller."""
    code, base = os.path.abspath(code_root), os.path.dirname(os.path.abspath(urdf))
    tops = data_tops()

    def place(rel: str) -> str:
        p = os.path.normpath(os.path.join(base, rel))
        if _within(p, code) and p != code and os.path.relpath(p, code).split(os.sep, 1)[0] in tops:
            p = os.path.normpath(os.path.join(os.path.abspath(data_root(code)), os.path.relpath(p, code)))
        return p

    if filename.startswith("package://"):
        rest = filename[len("package://") :].partition("/")[2]
        cands = [place(rest), place(os.path.join("..", rest))]
        return next((c for c in cands if os.path.isfile(c)), cands[0])
    return place(filename[len("file://") :] if filename.startswith("file://") else filename)


# ── Sandbox root of isolated runs — only the **writes** of test and regression runs go here ───────────────
# The canonical-50 regression and test-tag runs run judge · bake · gate exactly like canonical runs but must not write
# into canonical places (trees · combo cache · gate tables · contact-derived ledger · f0 shadow · ledgers). The scene
# variant `--variant` (aerial · artic) already means a ledger/f0 axis and cannot double as a sandbox name (one flag,
# one meaning) — hence an input with **another name**. When it is set, only the layers below resolve under that root
# (the written layers); input layers (base · retarget · mano · pool · contact canon · wrist RAW …) are still read
# from the data root.
SANDBOX_ENV = "DEXENGINE_SANDBOX_ROOT"
#: Layers **written** by stage 2 (judge → bake → gate) and the training roster — in an isolated run only these go to
#: the sandbox root. A new layer is decided here.
SANDBOX_LAYERS = ("trees", "precompute", "gate_meta", "contact_derive", "f0_root", "f0_stage", "views")


def sandbox_root() -> str:
    """The sandbox root of an isolated run (env DEXENGINE_SANDBOX_ROOT); "" = a canonical run."""
    s = os.environ.get(SANDBOX_ENV, "")
    return s
