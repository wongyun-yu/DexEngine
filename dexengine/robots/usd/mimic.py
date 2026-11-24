"""Step 3 of the robot USD build (dexengine.robots.usd.build): the single-layer full-robot USD.

Flattens the composed stage-2 USD into one layer and exports it as the stage-3 USD. This stage used to bake
PhysxMimicJointAPI coupling for a coupled hand; every registered hand is fully actuated (no mimic joints), so the
flattened export is the stage-3 product. A USD without any revolute/prismatic joint is an error.

Usage (no SimulationApp needed — pure USD; both paths required, no default robot):
    $ISAAC_PY dexengine/robots/usd/mimic.py \\
        --input-usd data/usd_cache/robots/full_robot_gen/<combo>/<combo>_flat_test.usd \\
        --output-usd data/usd_cache/robots/full_robot_gen/<combo>/<combo>_stage3_test.usd
"""

import argparse
import os

from pxr import Usd

from dexengine.robots.usd.flatten import flatten_canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-usd",
        required=True,
        help="Input (flattened) USD path — required, no default robot.",
    )
    parser.add_argument(
        "--output-usd",
        required=True,
        help="Output USD path — required; must contain '_test' unless --allow-commit-write.",
    )
    parser.add_argument(
        "--allow-commit-write",
        action="store_true",
        default=False,
        help="Allow writing to a path that does not contain '_test' in the filename (for real regeneration runs)",
    )
    parser.add_argument(
        "--sides",
        default="right,left",
        help="Accepted for compatibility with the build scripts; there is nothing per side to bake.",
    )
    args = parser.parse_args()

    # Safety guard: refuse to overwrite committed USDs unless explicitly allowed
    if "_test" not in os.path.basename(args.output_usd) and not args.allow_commit_write:
        raise RuntimeError(
            f"Refusing to overwrite committed USD at {args.output_usd} — use an --output-usd path "
            f"containing '_test' in the filename. Override only by passing --allow-commit-write explicitly."
        )

    print("\nBaking PhysxMimicJointAPI into full robot USD")
    print(f"  Input:  {args.input_usd}")
    print(f"  Output: {args.output_usd}")

    # Flatten the composed USD into a single layer — the one canonical flatten (flatten.flatten_canonical: prototype
    # numbers by content, no absolute path in the layer documentation), the same as step 2
    print("\n  Flattening composed USD to single layer...")
    composed_stage = Usd.Stage.Open(args.input_usd)
    flat_layer = flatten_canonical(composed_stage)
    flat_layer.Export(args.output_usd)
    print(f"  Exported flat USD: {args.output_usd}")
    del composed_stage

    stage = Usd.Stage.Open(args.output_usd)
    n_rev = sum(1 for p in stage.Traverse() if p.GetTypeName() in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"))
    if n_rev == 0:
        raise ValueError("Could not find joint prefix in USD (no joint at all — an empty USD)")
    print(
        f"  [mimic] not a coupled-joint hand ({n_rev} revolute joints) — the flattened export is stage 3 (passthrough)"
    )


if __name__ == "__main__":
    main()
