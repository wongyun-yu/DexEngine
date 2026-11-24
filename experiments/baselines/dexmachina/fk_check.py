"""The ``--fk-check`` step of ``taco_to_dexmachina.py``: Genesis forward kinematics of the converted (60 Hz) hand dofs
against the SPIDER fingertip positions (needs a GPU and the DexMachina checkout)."""

import os
import sys

import numpy as np


def fk_check(dm, dm_q, names_dm, bodypos, t50, t60):
    """Genesis FK of the converted (60 Hz) dofs at the 50 Hz frame times vs SPIDER fingertip/palm sites."""
    sys.path.insert(0, dm)  # the DexMachina checkout (third-party, not a package of this repository)
    import torch  # noqa: I001  # keep the original import order (torch before genesis) of the measured runs
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    hands = {
        s: scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(dm, f"assets/sharpa_wave_hand/{s}_sharpa_wave_6dof.urdf"),
                fixed=True,
                merge_fixed_links=False,
                recompute_inertia=True,
            )
        )
        for s in ("right", "left")
    }
    T = len(t60)
    scene.build(n_envs=T)
    res = {}
    for s, h in hands.items():
        act = [j for j in h.joints if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
        assert [j.name for j in act] == names_dm[s], f"{s}: actuated order differs from the converted order"
        q = torch.tensor(dm_q[s], dtype=torch.float32, device="cuda")
        h.set_dofs_position(q, [j.dof_idx_local for j in act], zero_velocity=True)
        lp = h.get_links_pos().cpu().numpy()  # (T, n_links, 3)
        ln = [link.name for link in h.links]
        tips = [f"{s}_{f}_fingertip" for f in ("thumb", "index", "middle", "ring", "pinky")]
        # SPIDER fingertip names: <side>_{thumb..pinky}_tip, <side>_palm (palm site 0.07 m along the flange y)
        k60 = np.clip(np.round(t50 * 60.0).astype(int), 0, T - 1)
        exact = np.abs(k60 / 60.0 - t50) < 1e-9  # 50 Hz frames that land exactly on a 60 Hz frame (every 5th)
        ours = np.stack([lp[k60, ln.index(n)] for n in tips], 1)[exact]
        ref = bodypos[f"{s}_fingertip_pos"][:, :5][exact]
        err = np.linalg.norm(ours - ref, axis=-1)
        res[s] = {
            "frames": int(exact.sum()),
            "tip_err_mm_max": round(float(err.max() * 1000), 3),
            "tip_err_mm_mean": round(float(err.mean() * 1000), 3),
        }
        print(f"[ADAPTER_FK] {s}: {res[s]}")
    return res
