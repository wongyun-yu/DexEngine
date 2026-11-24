"""Bake object USDs (rigid or articulated) from the processed object meshes, with the canonical convex collision parts.

A rigid object USD carries:
- the visual mesh (``<obj>/visual.obj``),
- collision = the canonical convex parts in ``<obj>/convex/`` (CoACD, made by the one owner
  ``dexengine/retarget/decompose/`` — parameters, one OpenMP thread, fingerprint record
  ``_decomp.json``),
- physical properties (density, friction, mass cap).

Convex part rules:
- This tool never runs CoACD itself (no inline decomposition). It bakes only when the ``convex/`` record fingerprint
  equals the canonical one — old parts without a record (vertex-decimated) or with other parameters are refused (rc 1).
- An object without ``convex/`` is decomposed through the decomposition window (dexengine/data/objects/decompose.py,
  retarget venv) only with ``--decompose_missing``. A stale ``convex/`` is not decomposed again even with that flag —
  a changed part count breaks the scene.xml that lists the object, so it is migrated separately.
- Articulated parts use ``<obj>/convex_<part stem>/`` under the same rule.
- Datasets the frozen generation reads (layout.frozen_datasets — object USDs shared with that generation) are not
  baked (rc 2).

Usage:
    # Single object (convex parts required):
    ./isaaclab.sh -p dexengine/robots/objects/usd_bake.py --dataset <ds> --mesh_path <obj>/visual.obj \
        --convex_dir <obj>/convex/

    # Batch mode (all objects in a directory):
    ./isaaclab.sh -p dexengine/robots/objects/usd_bake.py --batch --dataset <ds> --mesh_dir /path/to/objects/

    # Dataset mode: mesh dir = assets_objects layer, output = usd layer of the layout, ids optional:
    ./isaaclab.sh -p dexengine/robots/objects/usd_bake.py --batch --dataset hocap --ids G07_4,G05_1 --headless

Output:
    {output_dir}/{obj_id}/object.usd
    {output_dir}/{asset_id}/object_artic.usd   (--articulated · asset_id = artifact_paths.object_asset_id)
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

from dexengine.core.repo import REPO_ROOT  # noqa: E402
from dexengine.rl import isaac_cli as _isaac_cli  # noqa: E402
from dexengine.robots.objects.bake_owners import _data_layout, _decompose, _taco_paths  # noqa: E402

_PROJECT_ROOT = str(REPO_ROOT)


parser = argparse.ArgumentParser(description="Create object USD with CoACD collision.")
parser.add_argument("--mesh_path", type=str, required=False, default=None, help="Path to source .obj mesh file.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help=(
        "Output directory. Default: the `usd` layer of --dataset "
        "(layout.render('usd', ds); taco -> data/usd_cache/objects_taco/)"
    ),
)
parser.add_argument(
    "--obj_id", type=str, default=None, help="Object ID for output subdir. Default: mesh filename stem."
)
parser.add_argument("--density", type=float, default=200.0, help="Object density kg/m³ (ManipTrans: 200)")
parser.add_argument("--mass_cap", type=float, default=0.5, help="Maximum mass in kg (ManipTrans: 0.5)")
parser.add_argument("--friction", type=float, default=2.0, help="Object friction coefficient (ManipTrans: 2.0)")
parser.add_argument(
    "--convex_dir",
    type=str,
    default=None,
    help="Single-object mode: canonical convex parts (0.obj, 1.obj, … + _decomp.json). "
    "Required unless --decompose_missing.",
)
parser.add_argument(
    "--decompose_missing",
    action="store_true",
    help="decompose only objects without convex/ through the decomposition window "
    "(dexengine/data/objects/decompose.py, retarget venv), then bake — an explicit choice. A stale "
    "convex/ (fingerprint mismatch) is refused, never decomposed again.",
)
parser.add_argument("--batch", action="store_true", help="Batch mode: process all objects in --mesh_dir.")
parser.add_argument(
    "--articulated",
    action="store_true",
    help="articulated mode: joins the base and moving parts with one joint and bakes an "
    "articulated USD per asset (artifact_paths.object_asset_id). Targets, parts, axis, position, "
    "child_pose and limits are owned by the axis hand's artic_ranges.json (generator "
    "dexengine/data/artic/ranges.py); --ids are asset ids.",
)
parser.add_argument(
    "--mesh_dir",
    type=str,
    default=None,
    help=(
        "Directory of object subdirs (each with visual.obj + convex/). Default (batch): the `assets_objects` "
        "layer of --dataset."
    ),
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help=(
        "layout dataset id that owns the layer roots (assets_objects -> mesh_dir, usd -> output_dir). Required "
        "— no default."
    ),
)
parser.add_argument(
    "--ids",
    type=str,
    default=None,
    help="Batch mode: comma-separated object ids to bake (subset of --mesh_dir). Unknown ids FAIL (no silent skip).",
)
parser.add_argument(
    "--ids_file", type=str, default=None, help="Batch mode: file with one object id per line (alternative to --ids)."
)
_isaac_cli.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()


def _layer_root(layer: str, dataset: str) -> str:
    """Absolute layer root from the layout (the one owner of `data/<ds>/…` strings). The root is the data root
    (layout.data_root — env DEXENGINE_MAIN_ROOT, else this code tree), not the code tree: a code-only worktree has no
    data/ of its own, so a code-tree root would read no meshes and write USDs into it."""
    _mod = _data_layout()
    return os.path.join(_mod.data_root(_PROJECT_ROOT), _mod.render(layer, dataset))


# Default output = the `usd` layer of --dataset. For taco this is data/usd_cache/objects_taco (STORAGE row).
_DEFAULT_OUTPUT_DIR = _layer_root("usd", args_cli.dataset)


def _frozen_output_reason(output_dir: str, roots: dict | None = None) -> str | None:
    """The refusal reason when the output is a frozen-generation dataset's object USD layer or lies below it, else
    None. `roots` = {ds: layer root} (default = the USD layers of layout.frozen_datasets). A subfolder of the layer
    given as output would receive `<object>/object.usd` below it."""
    if roots is None:
        roots = {ds: os.path.realpath(_layer_root("usd", ds)) for ds in sorted(_data_layout().frozen_datasets())}
    real = os.path.realpath(output_dir)
    for ds, root in roots.items():
        if os.path.commonpath([real, root]) == root:
            return f"output {output_dir} ⊆ object USD layer {root} of frozen-generation dataset {ds} — not baked"
    return None


# A dataset the frozen generation reads, and any output into its USD layer (or below), is not baked — this fails
# before the app starts (rc 2).
if args_cli.dataset in _data_layout().frozen_datasets():
    parser.error(
        f"--dataset {args_cli.dataset}: object USDs of a dataset the frozen generation reads "
        f"(layout.frozen_datasets) are not baked again — the layer is shared with that generation"
    )
if _frozen_output_reason(args_cli.output_dir or _DEFAULT_OUTPUT_DIR):
    parser.error(_frozen_output_reason(args_cli.output_dir or _DEFAULT_OUTPUT_DIR))
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_isaac_cli.exit_nonzero_on_uncaught()  # module-level exceptions below also give rc 1 (Kit would exit with rc 0)

# pxr builders: imported only after the app has started
from dexengine.robots.objects.usd_build import create_articulated_object_usd, create_object_usd  # noqa: E402


#: The owner of the articulated specs (asset · parts · axis · position · child_pose · limits) is the file
#: `dexengine/data/artic/ranges.py` writes over the whole base layer (one canonical joint per asset). No values are
#: stated here; without the file the articulated mode does not start. The specs come from hand base trees, so they
#: live in the per-hand audit folder; object USDs are hand-independent and this tool has no run hand, so it reads the
#: axis hand's (AXIS_HAND = owner of shared outputs) — place: datasets.axis_output (shared output list AXIS_OUTPUTS).
def _artic_spec_path() -> str:
    from dexengine.data import rules as _rr

    return str(_rr.axis_output(_rr.ARTIC_SPEC, must_exist=False))


def _load_artic_specs() -> dict:
    import json as _json

    f = _artic_spec_path()
    if not os.path.isfile(f):
        return {}
    return _json.load(open(f, encoding="utf-8")).get("objects", {})


_ARTIC: dict = {}


def _retarget_py() -> str:
    """The retargeting interpreter (the venv the decomposition window and the render-cover baker run in)."""
    from dexengine.core import paths as _P

    py = _P.retarget_py()
    if not py or not os.path.isfile(py):
        raise RuntimeError(f"retargeting interpreter not found: {py!r}")
    return py


def _ensure_convex(mesh_file: str, convex_dir: str, label: str) -> str:
    """Return ``convex_dir`` when it holds the canonical decomposition of ``mesh_file`` (record fingerprint matches).

    A missing one is made through the decomposition window only with ``--decompose_missing``. A stale one is refused
    (never decomposed again — a changed part count breaks the scene.xml that lists that object's parts). Marker
    ``[BAKE_CONVEX]``.
    """
    D = _decompose()
    reason = D.stale_reason(mesh_file, convex_dir)
    if reason is None:
        print(
            f"[BAKE_CONVEX] {label} parts={len(D.part_files(convex_dir))} "
            f"params_fp={D.params_fingerprint()} fingerprint match",
            flush=True,
        )
        return convex_dir
    if os.path.isdir(convex_dir):
        raise RuntimeError(
            f"{label}: convex parts are not canonical ({reason}) — {convex_dir}. "
            f"Decompose again with dexengine/data/objects/decompose.py and align that object's "
            f"scene.xml part list before baking"
        )
    if not args_cli.decompose_missing:
        raise RuntimeError(
            f"{label}: no convex/ ({convex_dir}) — there is no inline decomposition. "
            f"Pass --decompose_missing or run dexengine/data/objects/decompose.py first"
        )
    import subprocess

    cmd = [
        _retarget_py(),
        os.path.join(_PROJECT_ROOT, "dexengine/data/objects/decompose.py"),
        "--mesh-file",
        mesh_file,
        "--out-dir",
        convex_dir,
    ]
    print(f"[BAKE_CONVEX] {label} no convex → decomposition window: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
    if r.returncode != 0 or "[DECOMP_DONE] requested=1 ok=1 failed=0" not in r.stdout:
        raise RuntimeError(f"{label}: decomposition window failed rc={r.returncode}: {tail}")
    reason = D.stale_reason(mesh_file, convex_dir)
    if reason is not None:
        raise RuntimeError(f"{label}: fingerprint still mismatched after decomposition ({reason})")
    print(
        f"[BAKE_CONVEX] {label} parts={len(D.part_files(convex_dir))} params_fp={D.params_fingerprint()} decomposed",
        flush=True,
    )
    return convex_dir


def _part_convex(mesh_path: str, name: str) -> str:
    """The part's canonical convex folder ``convex_<part stem>/`` (name rule: decompose.part_convex_dir; the same
    decomposition rule as rigid objects — _ensure_convex)."""
    return _ensure_convex(
        mesh_path, _decompose().part_convex_dir(mesh_path), f"{os.path.basename(os.path.dirname(mesh_path))}/{name}"
    )


def _bake_render_covers(output_dir: str, ids: list[str]) -> None:
    """Bake chain: bake the render covers of the training USDs this run wrote right away
    (dexengine/robots/objects/render_overlay.py, retargeting venv) and count that each cover exists. An asset without a
    cover is unfinished → rc 1 (a new asset never appears silently without textures; the render side's gate is
    artifact_paths.render_spawn_usds). An output_dir outside the canonical usd layer is outside the cover generator's
    target, so only the marker is printed — whoever moves it to the canonical place bakes the cover."""
    import subprocess
    import tempfile

    if os.path.realpath(output_dir) != os.path.realpath(_DEFAULT_OUTPUT_DIR):
        print(
            f"[BAKE_RENDER_COVER] skipped — {output_dir} is outside the canonical usd layer ({_DEFAULT_OUTPUT_DIR})",
            flush=True,
        )
        return
    tp = _taco_paths()
    usds = [
        u
        for i in ids
        for n in ("object.usd", tp.TacoPaths.ARTIC_USD_NAME)
        if os.path.isfile(u := os.path.join(output_dir, i, n))
    ]
    out = tempfile.mkdtemp(prefix="render_cover_tab_")
    rc = subprocess.run(
        [
            _retarget_py(),
            os.path.join(_PROJECT_ROOT, "dexengine/robots/objects/render_overlay.py"),
            "--dataset",
            args_cli.dataset,
            "--ids",
            ",".join(sorted(ids)),
            "--out",
            out,
        ],
        cwd=_PROJECT_ROOT,
    ).returncode
    miss = [u for u in usds if not os.path.isfile(tp.render_usd_file(u))]
    print(
        f"[BAKE_RENDER_COVER] rc={rc} · training USDs {len(usds)} · without cover {len(miss)} {miss[:3]} · table {out}",
        flush=True,
    )
    if rc or miss or not usds:
        _exit(1)


def _exit(code: int) -> None:
    """Keep the exit code: ``simulation_app.close()`` ends the process with rc 0 on the spot (a later ``sys.exit(n)`` is
    never reached and the buffered failure list is lost — measured: a batch with 6 failures ended with rc 0). So the
    output is flushed and a failure ends with ``os._exit(n)`` without closing the app. Only success (0) takes the
    normal shutdown path."""
    sys.stdout.flush()
    sys.stderr.flush()
    if code:
        os._exit(code)
    simulation_app.close()
    os._exit(0)


def _batch_ids(mesh_dir):
    """The objects to bake: the articulated spec's assets or the mesh layer's object folders, narrowed to --ids /
    --ids-file (an unknown id fails)."""
    # articulated = the spec's assets (asset id · source mesh folder = spec obj_id) · rigid = object folders of the
    # mesh layer
    specs = (
        {k.split("/", 1)[1]: v for k, v in _ARTIC.items() if v["dataset"] == args_cli.dataset}
        if args_cli.articulated
        else {}
    )
    obj_dirs = (
        sorted(specs)
        if args_cli.articulated
        else sorted(
            [
                d
                for d in os.listdir(mesh_dir)
                if os.path.isdir(os.path.join(mesh_dir, d)) and os.path.isfile(os.path.join(mesh_dir, d, "visual.obj"))
            ]
        )
    )
    want = []
    if args_cli.ids:
        want += [t.strip() for t in args_cli.ids.split(",") if t.strip()]
    if args_cli.ids_file:
        with open(args_cli.ids_file, encoding="utf-8") as _f:
            want += [ln.strip() for ln in _f if ln.strip()]
    if want:
        want = sorted(set(want))
        missing = [w for w in want if w not in obj_dirs]
        if missing:
            # An id we were asked to bake but cannot find is a failure, not a skip.
            where = "assets of the articulated spec" if args_cli.articulated else f"<id>/visual.obj under {mesh_dir}"
            print(f"[ERROR] {len(missing)} requested id(s) are not {where}: {missing[:10]}")
            _exit(2)
        obj_dirs = want
    return specs, obj_dirs, want


def _bake_objects(obj_dirs, mesh_dir, output_dir, specs):
    """Bake each object; a failure is recorded and the batch goes on. Returns (n_ok, failed)."""
    n_ok, failed = 0, []
    for idx, obj_name in enumerate(obj_dirs):
        obj_path = os.path.join(mesh_dir, obj_name, "visual.obj")
        convex_dir = os.path.join(mesh_dir, obj_name, "convex")

        out_path = os.path.join(
            output_dir, obj_name, _taco_paths().TacoPaths.ARTIC_USD_NAME if args_cli.articulated else "object.usd"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        print(f"\n[BATCH {idx + 1}/{len(obj_dirs)}] Processing {obj_name}")
        if args_cli.articulated:
            # parts, axis, position, child_pose and limits are owned by the measured spec file
            spec = specs[obj_name]
            try:
                create_articulated_object_usd(
                    mesh_dir=os.path.join(mesh_dir, spec["obj_id"]),
                    output_path=out_path,
                    spec=spec,
                    density=args_cli.density,
                    mass_cap=args_cli.mass_cap,
                    friction=args_cli.friction,
                    part_convex=_part_convex,
                )
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — one object's failure is recorded; the batch goes on
                print(f"[FAIL] {obj_name}: {type(e).__name__}: {e}")
                failed.append(obj_name)
            continue
        try:
            create_object_usd(
                mesh_path=obj_path,
                output_path=out_path,
                density=args_cli.density,
                mass_cap=args_cli.mass_cap,
                friction=args_cli.friction,
                convex_dir=_ensure_convex(obj_path, convex_dir, f"{args_cli.dataset}/{obj_name}"),
            )
            if os.path.isfile(out_path):
                n_ok += 1
            else:
                failed.append(f"{obj_name}: no artifact at {out_path}")
        except Exception as e:  # noqa: BLE001 — one object's failure is recorded; the batch goes on
            print(f"[BATCH] FAILED {obj_name}: {e}")
            failed.append(f"{obj_name}: {e}")
    return n_ok, failed


def _bake_batch():
    """Batch mode: every (or every --ids) object of the mesh layer; completion = count match."""
    # Batch mode: process all (or --ids) objects in mesh_dir (default: assets_objects layer of --dataset)
    mesh_dir = args_cli.mesh_dir or _layer_root("assets_objects", args_cli.dataset)
    output_dir = args_cli.output_dir or _DEFAULT_OUTPUT_DIR
    print(f"[BAKE] dataset={args_cli.dataset} mesh_dir={mesh_dir} output_dir={output_dir}", flush=True)
    print(
        f"[BAKE_CONVEX] canonical decomposition params_fp={_decompose().params_fingerprint()} (owner decompose.py) "
        f"decompose_missing={int(args_cli.decompose_missing)}",
        flush=True,
    )
    if not os.path.isdir(mesh_dir):
        print(f"[ERROR] --mesh_dir does not exist: {mesh_dir}")
        _exit(2)

    specs, obj_dirs, want = _batch_ids(mesh_dir)

    print(f"[BATCH] Found {len(obj_dirs)} objects in {mesh_dir}" + (f" (--ids subset {len(want)})" if want else ""))

    n_ok, failed = _bake_objects(obj_dirs, mesh_dir, output_dir, specs)

    # Completion = count match, printed as one marker line.
    print(
        f"\n[BATCH] Done. requested={len(obj_dirs)} produced={n_ok} failed={len(failed)} output_dir={output_dir}",
        flush=True,
    )
    for f_ in failed:
        print(f"[BATCH]   ! {f_}")
    if failed or n_ok != len(obj_dirs):
        _exit(1)
    _bake_render_covers(output_dir, obj_dirs)


def _bake_single():
    """Single-object mode (--mesh_path)."""
    # Single object mode
    mesh_path = args_cli.mesh_path
    obj_id = args_cli.obj_id or os.path.splitext(os.path.basename(mesh_path))[0]
    output_dir = args_cli.output_dir or _DEFAULT_OUTPUT_DIR
    convex_dir = _ensure_convex(
        mesh_path, args_cli.convex_dir or os.path.join(os.path.dirname(mesh_path), "convex"), obj_id
    )

    os.makedirs(os.path.join(output_dir, obj_id), exist_ok=True)
    output_path = os.path.join(output_dir, obj_id, "object.usd")

    create_object_usd(
        mesh_path=mesh_path,
        output_path=output_path,
        density=args_cli.density,
        mass_cap=args_cli.mass_cap,
        friction=args_cli.friction,
        convex_dir=convex_dir,
    )
    _bake_render_covers(output_dir, [obj_id])


def main():
    if not args_cli.batch and not args_cli.mesh_path:
        print("[ERROR] --mesh_path is required in single-object mode")
        _exit(1)

    if args_cli.articulated:
        globals()["_ARTIC"] = _load_artic_specs()
        if not _ARTIC:
            print(
                f"[FATAL] articulated spec {_artic_spec_path()} missing — run dexengine/data/artic/ranges.py first",
                file=sys.stderr,
            )
            _exit(2)
    if args_cli.batch:
        _bake_batch()
    else:
        _bake_single()


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 — a failure is rc 1 (_exit, so Kit's shutdown cannot overwrite the rc)
        import traceback

        traceback.print_exc()
        _exit(1)
    _exit(0)
