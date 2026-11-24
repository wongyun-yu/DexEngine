#!/usr/bin/env python3
"""Stage 4 for the 8 combo USDs: contact-reporter bake + canonical naming.

WHY
    The task envs spawn the robot with activate_contact_sensors=False (whole-
    robot contact pairs OOM PhysX), so contact sensors only see bodies whose
    PhysxContactReportAPI is baked into the USD — same mechanism as the legacy
    bake (bake_hand_collision_fullrobot stage 4 / bake_contact_reporter_sharpa).
    This generalizes it: bodies = the hand spec's contact regex on BOTH sides
    (mirror_right_to_left) + both arms' EE links (the palm hulls fixed-merge
    into the EE body on every combo — importer merges fixed joints).

    Output claims the CANONICAL pair name combo_usd_path expects:
        full_robot_gen/<combo>/<combo>.usd   (copy of *_stage3_test.usd + APIs)
    The stage3 file stays untouched for provenance.

    v1 keeps the importer's convex-hull colliders (no CoACD re-bake) —
    recorded as confound C8 vs the legacy CoACD hands.

GATE: per combo, applied-API count > 0 AND physics joint count identical to
stage3 (a silent partial copy would show as a count mismatch).

Run: ../IsaacLab/isaaclab.sh -p dexengine/rl/assets/contact_report_bake.py
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from dexengine.core.paths import main_root  # data root (env DEXENGINE_MAIN_ROOT, else the checkout)

# The USD driver passes no --usd-root, so this default is the real bake target.
_MAIN_ROOT = str(main_root())

parser = argparse.ArgumentParser()
parser.add_argument("--usd-root", default=f"{_MAIN_ROOT}/data/usd_cache/robots/full_robot_gen")
# Authority for the combo list = dexengine.core.mounts.MOUNT_REGISTRY, reached through dexengine/core/paths.py
from dexengine.core.paths import ARM_IDS, HAND_IDS, combo_ids, split_combo
from dexengine.rl.isaac_cli import add_app_launcher_args, exit_nonzero_on_uncaught
from dexengine.robots.vendor import unavailable_hands

_UN = unavailable_hands()  # a hand whose vendor source is not set has no body to bake: left out, said below
_COMBOS = [c for c in combo_ids() if split_combo(c, ARM_IDS, HAND_IDS)[1] not in _UN]
if _UN:
    print(f"[HANDS] combos selected {_COMBOS} · not selected {sorted(_UN)} (vendor source not set)", flush=True)
parser.add_argument("--combos", default=",".join(_COMBOS))  # authority = MOUNT_REGISTRY, minus unavailable hands
parser.add_argument("--threshold", type=float, default=0.0)
parser.add_argument(
    "--combo-dir",
    default=None,
    help="folder that holds every --combos (default none = <usd-root>/<combo>, byte-identical). "
    "A floating hand bakes its two per-side USDs into one folder of the split path rule "
    "(split_artic.split_usd_paths; build_floating_usd.sh).",
)
add_app_launcher_args(parser)
args = parser.parse_args()
from dexengine.rl import app as _TK  # noqa: E402 — stdlib only; the owner of the no-render kit rule

_TK.bake_kit(args, "contact_report_bake")
app = AppLauncher(args).app
exit_nonzero_on_uncaught()  # module-level exceptions below also exit with rc 1 (Kit would end with rc 0)

from pxr import PhysxSchema, Usd, UsdPhysics

# The robot registries are stdlib-only modules: baking a robot USD must not need a dataset.
import dexengine.core.arms as arm_specs
import dexengine.core.hands as hand_specs

_l = hand_specs.mirror_right_to_left

ARMS = {k: v for k, v in arm_specs.ARM_REGISTRY.items() if v is not None}  # authority = the registry
HANDS = {k: v for k, v in hand_specs.HAND_REGISTRY.items() if v is not None}  # authority = the registry


def joint_count(stage):
    return sum(1 for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))


fail = 0
for combo in args.combos.split(","):
    # A variant combo's suffix (e.g. rby1_sharpa_wave_chest00) is an asset variant only; the hand spec is the same.
    # One parser: paths.split_combo.
    try:
        arm_name, hand_name, _var = split_combo(combo, ARMS, HANDS)
    except ValueError as e:
        print(f"[stage4] {combo}: {e}", flush=True)
        fail = 1
        continue
    arm, hand = ARMS[arm_name], HANDS[hand_name]
    d = Path(args.combo_dir) if args.combo_dir else Path(args.usd_root) / combo
    src = d / f"{combo}_stage3_test.usd"
    dst = d / f"{combo}.usd"
    if not src.is_file():
        print(f"[stage4] {combo}: MISSING {src}", flush=True)
        fail = 1
        continue
    shutil.copy2(src, dst)

    pats = [re.compile(hand.contact_body_regex_right), re.compile(_l(hand.contact_body_regex_right))]
    # The floating hand's EE (the hand root) differs per hand: pass the hand read from the combo name (arms ignore it)
    ee_names = {arm.ee_body_name("right", hand=hand_name), arm.ee_body_name("left", hand=hand_name)}

    st_src = Usd.Stage.Open(str(src))
    st = Usd.Stage.Open(str(dst))
    applied, already = [], 0
    for prim in st.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        if not (name in ee_names or any(p.fullmatch(name) for p in pats)):
            continue
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            already += 1
            continue
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr(args.threshold)
        applied.append(name)
    st.GetRootLayer().Save()

    jc_src, jc_dst = joint_count(st_src), joint_count(st)
    ok = (len(applied) + already) > 0 and jc_src == jc_dst
    print(
        f"[stage4] {combo}: applied={len(applied)} already={already} "
        f"joints {jc_src}->{jc_dst} {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    if not ok:
        fail = 1

print(f"[stage4] {'DONE' if not fail else 'FAIL'}", flush=True)
sys.stdout.flush()


os._exit(fail)
