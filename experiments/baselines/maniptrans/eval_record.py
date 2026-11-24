"""ManipTrans evaluation recorder for Tab. 2: runs their train.py (test=true) unchanged, except that
  1. failure termination is off (as in the CHORD evaluation: every world plays to the reference end; our criteria are
     applied after the fact by the scorer), and
  2. the root states of both objects in all worlds are recorded every control step; at ManipTrans' own horizon
     (record_lib.last_progress) they are saved as a CHORD-world npz for experiments/scoring/score.py (motion form).
Runs in the ManipTrans environment (Isaac Gym GPU pipeline, Python 3.8) from the ManipTrans checkout as cwd, with this
repository on PYTHONPATH (maniptrans/gen_cmds.sh writes that call):
    python experiments/baselines/maniptrans/eval_record.py <out.npz> -- <train.py hydra args … test=true num_envs=64
    checkpoint=… dataIndices=[<seq>]>
Markers: [MT_CKPT] (the checkpoint and its sha256) · [MT_F0_SIM] (what moved the objects in the first step) ·
[MT_RECORD] SAVED / FAIL.
"""

import os
import sys
from pathlib import Path

out = sys.argv[1]
assert sys.argv[2] == "--"
sys.argv = [sys.argv[0]] + sys.argv[3:]
# the residual checkpoint is an identity input: an empty `checkpoint=` (e.g. `$(ls -t …/nn/*.pth)` after training never
# ran) would make train.py evaluate a fresh random policy, so it must name one existing file
_ck = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("checkpoint=")]
if len(_ck) != 1 or not _ck[0] or not os.path.isfile(_ck[0]):
    print(f"[MT_RECORD] FAIL checkpoint= must name one existing file, got {_ck}", flush=True)
    sys.exit(2)
import hashlib as _hl

print(f"[MT_CKPT] {_ck[0]} sha256={_hl.sha256(Path(_ck[0]).read_bytes()).hexdigest()[:16]}", flush=True)
from isaacgym import gymapi  # noqa: F401, I001  # Isaac Gym must be imported before torch
import numpy as np
import torch
from experiments.baselines.maniptrans.record_lib import gym_to_world, last_progress, method_reference, write_npz
from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import DexHandManipBiHEnv

_orig_reward = DexHandManipBiHEnv.compute_reward
_orig_post = DexHandManipBiHEnv.post_physics_step
_orig_pre = DexHandManipBiHEnv.pre_physics_step
REC = {"rh_p": [], "rh_q": [], "lh_p": [], "lh_q": [], "step": []}
F0 = {}  # frame-0 diagnosis: spawn vs reference and what the first physics step moved


def _qang_deg(a, b):  # xyzw rows
    d = (a * b).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2 * torch.acos(d))


def pre_physics_step(self, actions):
    if not F0:  # the state the env spawned (reset), before any action or physics step
        for side in ("rh", "lh"):
            dd = self.demo_data_rh if side == "rh" else self.demo_data_lh
            st = getattr(self, f"_manip_obj_{side}_root_state")
            d0 = dd["obj_trajectory"][:, 0]
            from main.dataset.transform import rotmat_to_quat

            q0 = rotmat_to_quat(d0[:, :3, :3])[:, [1, 2, 3, 0]]
            F0[side] = {
                "spawn_p": st[:, :3].clone(),
                "spawn_q": st[:, 3:7].clone(),
                "demo_p": d0[:, :3, 3].clone(),
                "demo_q": q0,
                "wrist_p": getattr(self, f"_{side}_base_state")[:, :3].clone(),
                "act_absmax": float(actions.abs().max()),
            }
    return _orig_pre(self, actions)


def compute_reward(self, actions):
    _orig_reward(self, actions)
    max_len = self.demo_data_rh["seq_len"]  # (N,) — reset exactly at ManipTrans' own success point (last_progress)
    done = self.progress_buf >= last_progress(max_len)
    self.reset_buf = done.long() if self.reset_buf.dtype != torch.bool else done
    self.failure_buf = torch.zeros_like(self.failure_buf)


def post_physics_step(self):
    _orig_post(self)
    if F0 and "done" not in F0:  # after the first step: what moved the objects — the hand (contact forces) or not
        F0["done"] = True
        for side in ("rh", "lh"):
            f = F0[side]
            st = getattr(self, f"_manip_obj_{side}_root_state")
            hd = getattr(self, f"dexhand_{side}_handles")
            hcf = self.net_cf[:, list(hd.values()), :].norm(dim=-1).max(dim=-1).values

            def med(x):
                return float(x.float().median())

            f.update(
                {
                    "spawn_vs_demo_mm": med((f["spawn_p"] - f["demo_p"]).norm(dim=-1) * 1000),
                    "spawn_vs_demo_deg": med(_qang_deg(f["spawn_q"], f["demo_q"])),
                    "step1_obj_mm": med((st[:, :3] - f["spawn_p"]).norm(dim=-1) * 1000),
                    "step1_obj_deg": med(_qang_deg(st[:, 3:7], f["spawn_q"])),
                    "step1_wrist_mm": med(
                        (getattr(self, f"_{side}_base_state")[:, :3] - f["wrist_p"]).norm(dim=-1) * 1000
                    ),
                    "step1_obj_cf_N": med(getattr(self, f"_manip_obj_{side}_cf").norm(dim=-1)),
                    "step1_hand_max_cf_N": med(hcf),
                }
            )
            print(
                f"[MT_F0_SIM] {side} "
                + " · ".join(
                    f"{k}={f[k]:.2f}"
                    for k in (
                        "spawn_vs_demo_mm",
                        "spawn_vs_demo_deg",
                        "step1_obj_mm",
                        "step1_obj_deg",
                        "step1_wrist_mm",
                        "step1_obj_cf_N",
                        "step1_hand_max_cf_N",
                        "act_absmax",
                    )
                ),
                flush=True,
            )
    REC["rh_p"].append(self._manip_obj_rh_root_state[:, :3].cpu().numpy().copy())
    REC["rh_q"].append(self._manip_obj_rh_root_state[:, 3:7].cpu().numpy().copy())
    REC["lh_p"].append(self._manip_obj_lh_root_state[:, :3].cpu().numpy().copy())
    REC["lh_q"].append(self._manip_obj_lh_root_state[:, 3:7].cpu().numpy().copy())
    REC["step"].append(int(self.progress_buf[0].item()))
    # progress was just incremented: the entry recorded now is state p = progress − 1; stop after p = last_progress
    if bool((self.progress_buf - 1 >= last_progress(self.demo_data_rh["seq_len"])).all()):
        m2g = self.mujoco2gym_transf.cpu().numpy()
        bodies = {}
        refs = {}
        for side, dd in (("rh", self.demo_data_rh), ("lh", self.demo_data_lh)):
            P = np.stack(REC[f"{side}_p"], 1)
            Q = np.stack(REC[f"{side}_q"], 1)  # (N,T,3) (N,T,4 xyzw)
            pw, qw = gym_to_world(dd["taco_world_from_data"][0], m2g, P, Q)
            bodies[dd["taco_object_body"][0]] = (pw, qw)
            sl = int(dd["seq_len"][0])
            refs[dd["taco_object_body"][0]] = method_reference(
                dd["obj_trajectory"][0].cpu().numpy(),
                dd["taco_world_from_data"][0],
                m2g,
                float(dd["taco_fps_out"][0]),
                sl,
            )
        assert len(REC["step"]) == last_progress(sl) + 1, (
            len(REC["step"]),
            sl,
        )  # one entry per evaluated state 0..last
        write_npz(out, np.arange(len(REC["step"])) / 60.0, bodies, refs)
        print(
            f"[MT_RECORD] SAVED {out} steps={len(REC['step'])} worlds={P.shape[0]} bodies={sorted(bodies)}", flush=True
        )
        os._exit(0)


DexHandManipBiHEnv.compute_reward = compute_reward
DexHandManipBiHEnv.pre_physics_step = pre_physics_step
DexHandManipBiHEnv.post_physics_step = post_physics_step
import runpy

runpy.run_path(os.path.join(os.getcwd(), "main/rl/train.py"), run_name="__main__")
print("[MT_RECORD] FAIL run ended without reaching reference end", flush=True)
sys.exit(1)
