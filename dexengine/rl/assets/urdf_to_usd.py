#!/usr/bin/env python3
"""Convert a combo's merged full-robot URDF to an instanceable USD (step 1 of dexengine.robots.usd.build).

Uses isaaclab.sim.converters.UrdfConverter programmatically with settings
vendored in configs/robots/urdf_converter.yaml. This is step 1
of the self-contained USD baking pipeline.

Safety: refuses to write anywhere under data/usd_cache/ unless the output
filename contains "_test" (or --allow-commit-write is given).

Usage (inputs are required — no default robot; the driver is scripts/build_robot_usd.py):
  cd <repository root>
  ../IsaacLab/isaaclab.sh -p dexengine/rl/assets/urdf_to_usd.py \\
      --input-urdf <assets/robots/full_robot_gen/<combo>/<combo>.urdf> \\
      --output-usd <data/usd_cache/robots/full_robot_gen/<combo>/<combo>_base_test.usd> --force

the defaults (the first arm's merged URDF and its test
USD) and the md5 guard over that arm's committed USDs left with the arm.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# ---------------------------------------------------------------------------
# Argparse BEFORE AppLauncher.
# ---------------------------------------------------------------------------
from dexengine.core.repo import CONFIGS_DIR, REPO_ROOT  # stdlib only, safe before AppLauncher

_PROJECT_ROOT = str(REPO_ROOT)  # relative URDF / output paths resolve against the checkout root

_DEFAULT_CONFIG = str(CONFIGS_DIR / "robots" / "urdf_converter.yaml")

_parser = argparse.ArgumentParser(description="Convert a combo's merged full-robot URDF to an instanceable USD.")
_parser.add_argument(
    "--input-urdf",
    type=str,
    required=True,
    help="Path to input URDF (absolute or relative to the repository root). Required — no default robot.",
)
_parser.add_argument(
    "--output-usd",
    type=str,
    required=True,
    help="Path to output USD file. Must contain '_test' in filename if under data/usd_cache/. Required.",
)
_parser.add_argument(
    "--config",
    type=str,
    default=_DEFAULT_CONFIG,
    help="Path to urdf_converter.yaml config.",
)
_parser.add_argument(
    "--fix-base",
    choices=("true", "false"),
    default=None,
    help=(
        "Override the config's fix_base. Unset = the yaml value (byte-identical). The floating hand "
        "(DEXENGINE_ARM=floating, experiments/ablations/floating_hand/build_floating_usd.sh) bakes with false — repo "
        "rule "
        "'floating hand = fix_base False'."
    ),
)
_parser.add_argument(
    "--force",
    action="store_true",
    help="Force conversion by setting force_usd_conversion=True.",
)
_parser.add_argument(
    "--allow-commit-write",
    action="store_true",
    default=False,
    help=(
        "Allow writing to a USD path whose filename does not contain '_test'. "
        "Used for fresh-clone regeneration of canonical USDs."
    ),
)

_args, _remaining = _parser.parse_known_args()

from dexengine.rl.isaac_cli import add_app_launcher_args, exit_nonzero_on_uncaught
from dexengine.robots.usd import (
    render_parts as render_visual_parts,  # the owner of the render part link names
)


def _refuse_render_urdf(urdf: str) -> None:
    """A render URDF (``<combo>_render.urdf`` — dexengine.robots.urdf.merge) carries the mount's declared render parts
    as visual-only links; baked, they would land in the training body. The training body gets none (they are
    render only — the render body gets them from render_visual_parts.add), so such a URDF is refused — before Kit
    starts, so the exit code says so."""
    ru_file = urdf if os.path.isabs(urdf) else os.path.join(_PROJECT_ROOT, urdf)
    bad = render_visual_parts.render_links_in_urdf(ru_file) if os.path.exists(ru_file) else []
    if bad:
        print(
            f"[URDF_IMPORT] {os.path.basename(ru_file)}: render-only part links {bad} — a render URDF is not a bake "
            "input (the training body has no parts) · to do: bake <combo>.urdf from the same folder",
            file=sys.stderr,
        )
        sys.exit(1)


_refuse_render_urdf(_args.input_urdf)

from isaaclab.app import AppLauncher

_app_parser = argparse.ArgumentParser()
add_app_launcher_args(_app_parser)
_app_args = _app_parser.parse_args(_remaining)
from dexengine.rl import app as _TK  # noqa: E402 — stdlib only; the owner of the no-render kit rule

_TK.bake_kit(_app_args, "urdf_to_usd")
_app_launcher = AppLauncher(_app_args)
simulation_app = _app_launcher.app
exit_nonzero_on_uncaught()  # module-level exceptions below also exit with rc 1 (Kit would end with rc 0)

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import yaml

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def _fail() -> None:
    """Exit with failure. ``simulation_app.close()`` ends the process on the spot with rc 0, so a later
    ``sys.exit(1)`` never runs (measured on Isaac Lab 0.53.1). Hard-exit with rc 1 without closing
    (as objects.usd_bake._exit and dexengine.robots.objects.cooked_hull do)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def _assert_safe_output(output_usd_abs: str) -> None:
    from dexengine.core.paths import main_root  # owner of the data root (env DEXENGINE_MAIN_ROOT)

    usd_cache = os.path.normpath(
        os.path.join(str(main_root()), "data", "usd_cache")
    )  # the cache lives under the data root
    fname = os.path.basename(output_usd_abs)
    if output_usd_abs.startswith(usd_cache + os.sep) and "_test" not in fname and not _args.allow_commit_write:
        print(
            f"[error] Refusing to write non-test output to {output_usd_abs} — "
            f"use a filename containing '_test' to prevent overwriting committed USDs, "
            f"or pass --allow-commit-write for a fresh-clone regeneration.",
            file=sys.stderr,
        )
        _fail()
    if os.path.exists(output_usd_abs):
        print(
            f"[error] Output already exists: {output_usd_abs}. Remove it first.",
            file=sys.stderr,
        )
        _fail()


def _assert_importer_meshes(urdf_abs: str) -> None:
    """Isaac's URDF importer resolves every mesh itself, relative to the URDF (this code tree) — it does not take the
    one URDF mesh resolver (paths.urdf_mesh: the untracked hand stages come from the data root). Where the two name
    different files, or the file is not there, the import can come out with empty links and exit 0 — so die by name
    first (bake from a checkout whose data root is its own tree, e.g. the canonical one)."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    from dexengine.core.paths import urdf_mesh

    bad = []
    for m in ET.parse(urdf_abs).getroot().iter("mesh"):
        fn = m.get("filename") or ""
        own = urdf_mesh(urdf_abs, fn)
        lex = own if "://" in fn else Path(os.path.dirname(urdf_abs), fn).resolve()  # the importer's own reading
        if own != lex or not own.is_file():
            bad.append(f"{fn} -> importer {lex} · resolver {own}{'' if own.is_file() else ' (missing)'}")
    if bad:
        print(
            f"[URDF_IMPORT] {len(bad)} meshes are read by the Isaac importer from a place other than the resolver's, "
            f"or not at all — e.g. {bad[0]} · to do: bake from a checkout whose data root is the code tree "
            "(without DEXENGINE_MAIN_ROOT)",
            file=sys.stderr,
        )
        _fail()


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path))


def _build_cfg(
    input_urdf_abs: str,
    output_usd_abs: str,
    yaml_cfg: dict,
    force: bool,
) -> UrdfConverterCfg:
    drive_cfg = yaml_cfg["joint_drive"]
    gains = drive_cfg["gains"]
    joint_drive = UrdfConverterCfg.JointDriveCfg(
        drive_type=drive_cfg["drive_type"],
        target_type=drive_cfg["target_type"],
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=float(gains["stiffness"]),
            damping=float(gains["damping"]),
        ),
    )
    return UrdfConverterCfg(
        asset_path=input_urdf_abs,
        usd_dir=os.path.dirname(output_usd_abs),
        usd_file_name=os.path.basename(output_usd_abs),
        force_usd_conversion=bool(yaml_cfg.get("force_usd_conversion", False)) or force,
        make_instanceable=bool(yaml_cfg.get("make_instanceable", True)),
        fix_base=(bool(yaml_cfg["fix_base"]) if _args.fix_base is None else _args.fix_base == "true"),
        root_link_name=yaml_cfg.get("root_link_name"),
        link_density=float(yaml_cfg.get("link_density", 0.0)),
        merge_fixed_joints=bool(yaml_cfg.get("merge_fixed_joints", True)),
        convert_mimic_joints_to_normal_joints=bool(yaml_cfg.get("convert_mimic_joints_to_normal_joints", False)),
        joint_drive=joint_drive,
        collider_type=yaml_cfg.get("collider_type", "convex_hull"),
        self_collision=bool(yaml_cfg.get("self_collision", False)),
        replace_cylinders_with_capsules=bool(yaml_cfg.get("replace_cylinders_with_capsules", False)),
        collision_from_visuals=bool(yaml_cfg.get("collision_from_visuals", False)),
    )


def main() -> None:
    input_urdf_abs = _resolve(_args.input_urdf)
    output_usd_abs = _resolve(_args.output_usd)
    config_abs = _resolve(_args.config)

    if not os.path.exists(input_urdf_abs):
        print(f"[error] input URDF not found: {input_urdf_abs}", file=sys.stderr)
        _fail()
    if not os.path.exists(config_abs):
        print(f"[error] config yaml not found: {config_abs}", file=sys.stderr)
        _fail()

    _assert_safe_output(output_usd_abs)
    _assert_importer_meshes(input_urdf_abs)

    os.makedirs(os.path.dirname(output_usd_abs), exist_ok=True)

    with open(config_abs) as f:
        yaml_cfg = yaml.safe_load(f)

    print(f"[info] input URDF:  {input_urdf_abs}")
    print(f"[info] output USD:  {output_usd_abs}")
    print(f"[info] config yaml: {config_abs}")
    print(f"[info] force:       {_args.force}")

    cfg = _build_cfg(input_urdf_abs, output_usd_abs, yaml_cfg, _args.force)
    converter = UrdfConverter(cfg)

    produced = converter.usd_path
    print(f"[ok] produced USD: {produced}")
    if os.path.exists(produced):
        print(f"[info] size: {os.path.getsize(produced):,} bytes")

    cfg_dir = os.path.join(os.path.dirname(output_usd_abs), "configuration")
    if os.path.isdir(cfg_dir):
        print(f"[info] sublayers in {cfg_dir}:")
        for p in sorted(glob.glob(os.path.join(cfg_dir, "*.usd"))):
            print(f"  - {os.path.basename(p)}: {os.path.getsize(p):,} bytes")


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 — close() in finally ended with rc 0 even on an exception; fail with rc 1 instead
        import traceback

        traceback.print_exc()
        _fail()
    simulation_app.close()
