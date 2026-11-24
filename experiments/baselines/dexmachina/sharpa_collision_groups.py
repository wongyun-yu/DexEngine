"""DexMachina hand-processing step 5 for SharpA Wave: collision groups via their own divide_hand_groups()
(dexmachina/hand_proc/inspect_raw_urdf.py). Their CLI path (--gather_geoms) ends in breakpoint(), so this calls the
function directly on the entity loaded with the same morph options BaseRobot uses (envs/robot.py:114).
Their group filter supports ONE mapping for both hands (base_env.py:213 takes the left cfg's), so this checks that
right and left give the same {link local idx: group} and prints it for envs/hand_cfgs/sharpa_wave.py.
run from <workspace>/code/dexmachina/dexmachina with the DexMachina env python."""

import json

import genesis as gs
from dexmachina.hand_proc.inspect_raw_urdf import divide_hand_groups

gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(show_viewer=False)
hands = {}
for side in ("right", "left"):
    hands[side] = scene.add_entity(
        gs.morphs.URDF(
            file=f"assets/sharpa_wave_hand/{side}_sharpa_wave_6dof.urdf",
            fixed=True,
            convexify=True,
            merge_fixed_links=False,
            recompute_inertia=True,
        )
    )
scene.build(n_envs=1)
out = {}
for side, h in hands.items():
    g = divide_hand_groups(h, f"{side}_hand_C_MC")
    coll = [(link.idx_local, link.name) for link in h.links if len(link.geoms) > 0]
    act = [j.name for j in h.joints if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
    out[side] = {
        "groups": {int(k): int(v) for k, v in sorted(g.items())},
        "coll_links": coll,
        "n_links": h.n_links,
        "actuated": act,
        "n_dofs": h.n_dofs,
    }
    print(
        f"[SHARPA_GROUPS] {side} groups={out[side]['groups']} n_coll_links={len(coll)} n_links={h.n_links} "
        f"n_dofs={h.n_dofs} actuated={len(act)}"
    )
assert out["right"]["groups"] == out["left"]["groups"], (
    "left/right collision groups differ (their filter needs one map)"
)
print("[SHARPA_GROUPS] SAME_LR " + json.dumps(out["right"]["groups"]))
print("[SHARPA_GROUPS] coll_links_right " + json.dumps(out["right"]["coll_links"]))
print("[SHARPA_GROUPS] actuated_right " + json.dumps(out["right"]["actuated"]))
