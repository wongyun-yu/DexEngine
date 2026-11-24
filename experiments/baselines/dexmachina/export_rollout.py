"""DexMachina eval output -> the shared scorer's common rollout npz (experiments/scoring/score.py `npz`), one clip.

Input: their eval file (dexmachina/rl/eval_rl_games.py, multi-object recording) eval_ep<k>.npy:
  obj_state (T_ep, n_envs, sum(7 + n_joints)) = state AFTER eval step i (episode index i+1), objects in env order
  object_names, object_state_dims, dones (T_ep, n_envs)
and the adapter record <key>_adapter.json (60 Hz frame k <-> original-motion time tau = k/60 s).
Conventions:
  * eval_rl_games with n_envs > 1 sets is_eval: the LAST env is a display dummy (base_env._step_env_idxs drops it and
    robot.init_qpos[-1] = 0) -> dropped here.
  * tau of the state after step i = (i+1)/60. tau = 0 = the reset state = demo frame 0 (ArticulatedObject.reset_idx sets
    the demo frame-0 pose exactly), prepended from the demo file.
  * their eval runs every env to the last frame (base_env._get_dones: is_eval -> only timeout / NaN env). A done inside
    the scored steps = the env went non-finite and was reset: from there on the rollout keeps its last finite state
    (the object stops where it was -> the scorer counts the rest as not tracked; score.py takes finite poses only).
  * tau_end = the method's horizon = last 60 Hz demo frame (T60-1)/60 (<= the reference end).
Output keys: t (T,), obj_pos_<side> (N,T,3), obj_quat_<side> (N,T,4 wxyz), step_dt, tau_end, n_rollouts, source.
usage: python export_rollout.py --eval eval_ep0.npy --adapter <key>_adapter.json --demo <processed K_use_01.npy> --out
roll.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

SIDE = {"right_object": "right", "left_object": "left"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev = np.load(a.eval, allow_pickle=True).item()
    rec = json.loads(Path(a.adapter).read_text())
    demo = np.load(a.demo, allow_pickle=True).item()
    names = [str(n) for n in ev["object_names"]]
    dims = [int(d) for d in ev["object_state_dims"]]
    assert sorted(names) == sorted(SIDE), names
    S = ev["obj_state"]  # (T_ep, n_envs, D)
    T_ep, n_envs, D = S.shape
    assert sum(dims) == D, (D, dims)
    assert n_envs >= 2, "need >= 2 eval envs (the last one is the display dummy)"
    S = S[:, : n_envs - 1]
    done = ev["dones"][:, : n_envs - 1].astype(bool) if "dones" in ev else np.zeros(S.shape[:2], bool)
    T60 = int(rec["T60"])
    dt = 1.0 / 60.0
    # steps i = 0 .. T60-2 -> tau = (i+1)/60 .. (T60-1)/60 ; plus tau 0
    n_keep = T60 - 1
    assert T_ep >= n_keep, (T_ep, n_keep)
    t = np.arange(T60) * dt
    N = S.shape[1]
    out = {
        "t": t,
        "step_dt": np.float64(dt),
        "tau_end": np.float64((T60 - 1) * dt),
        "n_rollouts": np.int64(N),
        "source": np.array(
            json.dumps({"eval": a.eval, "clip": rec["clip"], "key": rec["key"], "dropped_dummy_env": n_envs - 1})
        ),
    }
    # a rollout's first done inside the scored steps: the state recorded after that step is already the next
    # episode's reset state -> from there the rollout holds its last valid entry (see the header)
    n_valid = np.full(N, T60, int)  # entries incl. tau 0
    for w in range(N):
        d = np.nonzero(done[:n_keep, w])[0]
        if len(d):
            n_valid[w] = int(d[0]) + 1  # tau0 + records 0..d-1
    off, arr = 0, {}
    for n, d in zip(names, dims, strict=False):
        st = demo["objects"]["states"][n]
        p0, q0 = np.asarray(st["obj_trans"][0], float), np.asarray(st["obj_quat"][0], float)
        pos = np.concatenate(
            [np.broadcast_to(p0, (N, 1, 3)), S[:n_keep, :, off : off + 3].transpose(1, 0, 2)], 1
        ).astype(np.float64)
        quat = np.concatenate(
            [np.broadcast_to(q0, (N, 1, 4)), S[:n_keep, :, off + 3 : off + 7].transpose(1, 0, 2)], 1
        ).astype(np.float64)
        arr[n] = (pos, quat)
        off += d
    fin = np.ones((N, T60), bool)  # an entry is valid only when every object's pose is finite
    for pos, quat in arr.values():
        fin &= np.isfinite(pos).all(2) & np.isfinite(quat).all(2)
    for w in range(N):
        last = int(n_valid[w]) - 1
        while last > 0 and not fin[w, last]:  # a non-finite entry before the done flag: hold from the last finite
            last -= 1
        assert fin[w, last], f"rollout {w}: no finite entry"
        n_valid[w] = last + 1
    for n, (pos, quat) in arr.items():
        for w in range(N):
            pos[w, n_valid[w] :] = pos[w, n_valid[w] - 1]
            quat[w, n_valid[w] :] = quat[w, n_valid[w] - 1]
        assert np.isfinite(pos).all() and np.isfinite(quat).all(), f"{n}: non-finite left after the hold"
        out[f"obj_pos_{SIDE[n]}"], out[f"obj_quat_{SIDE[n]}"] = pos, quat
    out["n_valid"] = n_valid
    np.savez_compressed(a.out, **out)
    print(
        f"[EXPORT] OK {a.out} rollouts={N} steps={len(out['t'])}/{T60} tau_end={float(out['tau_end']):.4f} "
        f"held_rollouts={int((n_valid < T60).sum())} (display dummy env {n_envs - 1} dropped)"
    )


if __name__ == "__main__":
    main()
