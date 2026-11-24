"""The chain that registers a new hand: rig, merged URDF, combo USD (each step an existing module; stops at rc != 0)."""

from __future__ import annotations

import os
import subprocess
import sys

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.data.stage1 import table as T

#: (name, interpreter: retarget | system, argv, what it makes). R5's driver starts its Isaac Lab steps itself.
REGISTER_STEPS = (
    (
        "R1 MJCF rig",
        "retarget",
        ("dexengine/robots/hand_rig.py", "{hand}"),
        "the retargeting hand rig (assets/robots/hands/<spec dir>/mjcf)",
    ),
    (
        "R4 merged URDF",
        "system",
        ("dexengine/robots/urdf/merge.py", "{arm}+{hand}"),
        "arm + hand subtree + mount joint",
    ),
    (
        "R5 combo USD",
        "system",
        ("scripts/build_robot_usd.py", "--arm", "{arm}", "--hand", "{hand}"),
        "training USD (convert -> flatten -> export -> contact reporters -> colour check)",
    ),
)


def precheck(hand: str) -> list[str]:
    """What is missing before the chain can start (a HandSpec, its vendor URDFs, a combo)."""
    bad = []
    hs = P.load_registry("hand_specs").HAND_REGISTRY.get(hand)
    if hs is None:
        bad.append(f"{hand!r} is not in HAND_REGISTRY -- add a HandSpec entry to the hand registry")
        return bad
    vu = getattr(hs, "vendor_urdf", "") or ""
    if not vu:
        bad.append(f"{hand}: HandSpec.vendor_urdf is empty -- give the vendor URDF location")
    else:
        for side in ("right", "left"):
            f = P.main_root() / vu.format(side=side)  # vendor trees are untracked data under the data root
            if not f.is_file():
                bad.append(f"{hand}: vendor URDF is missing -- {f}")
    try:
        RR.combo()
    except KeyError as e:
        bad.append(f"{hand}: cannot build the combo -- {e}")
    return bad


def register(hand: str, arm: str, dry_run: bool) -> int:
    """Run the registration chain in order; stop at the first step with rc != 0."""
    print(f"hand registration -- {hand} × {arm}")
    bad = precheck(hand)
    if bad:
        print("missing declarations, cannot start:")
        for b in bad:
            print(f"  ✗ {b}")
        return 2
    env = dict(os.environ, DEXENGINE_HAND=hand)
    pys = T.py("retarget-py")
    for name, interp, cmd, why in REGISTER_STEPS:
        args = [a.format(hand=hand, arm=arm) for a in cmd]
        run, e = [pys if interp == "retarget" else sys.executable, str(T.REPO / args[0]), *args[1:]], env
        print(f"\n[{name}] {why}")
        print(f"  $ {' '.join(run)}")
        if dry_run:
            continue
        r = subprocess.run(run, cwd=T.REPO, env=e)
        print(f"  rc={r.returncode}")
        if r.returncode != 0:
            print(f"✗ stopped at {name}")
            return r.returncode
    print("\nregistration chain passed. Next: the data steps:")
    return 0
