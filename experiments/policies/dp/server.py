"""DP policy server: a LeRobot diffusion policy behind Isaac-GR00T's ZMQ PolicyServer protocol, so the evaluation
session talks to DP and GR00T with one client (REQ/REP · msgpack · ndarrays as .npy bytes · endpoints ping, kill,
reset, get_action, get_modality_config). No Isaac-GR00T import: the wire format is reproduced here.

get_action(observation, options), observation in GR00T's layout:
  video    {camera_ego_left, camera_ego_right}: uint8 (B, T, 180, 320, 3); the last n_obs (= 2) frames are used and a
           single frame is repeated (LeRobot's reset behaviour) — send the 2 latest consecutive frames.
  state    {left_arm_joints (7), right_arm_joints (7), head_joints (2), hands_joints (40)}: float (B, T, D), the
           training dataset's 56-column order (meta/modality.json).
  language {human.task_description: [[str]] × B} → observation.environment_state (task-axis one-hot, 4) through the
           training dataset's task_index → axis_index table; options["axis_index"] overrides.
Returns ({group: (B, n_action, D)}, {}) — absolute joint targets (the dataset's action = the next row's PD target),
n_action = the checkpoint's n_action_steps, unnormalised.

`--out <dir>`: after loading, `<dir>/modality.json` (schema e5_policy_modality/1) and `<dir>/READY` are written; the
server ends when `<dir>/STOP` appears (or on the kill endpoint) and rewrites `<dir>/serve_heartbeat` every minute.
`--oracle-map MAP`: the network output is replaced by the training label of the scene's episode in `--root` (MAP: scene
→ {episode, length}; the session sends options e5_scene · e5_frame): action[j … j+n_action−1] (past the end clamps
to the last row), normalised, clamped like the scheduler's clip_sample, unnormalised; `<out>/oracle.jsonl` logs each
query. `--oracle-exact` returns the label itself; `--oracle-hold-end` repeats the last label past the episode end
(episodes shorter than the scene's time axis); `--oracle-root` reads the labels from a labels-only conversion.
`--const-state-fix [--const-state-eps W]`: state columns constant in training get their training value (marker
`[DP_CONST_STATE] dims=…` at start and every reset). `--sampler DDIM|DDPM --infer-steps N`: sampler override (marker
`[E5_DP_SAMPLER] <S> steps=<N> …`, and inference-time medians every 50 calls and at STOP).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time

import msgpack
import numpy as np
import zmq

from experiments.policies.dp.serve_model import SAMPLERS, sampler_override
from experiments.policies.dp.serve_policy import DPServe


def enc(o):
    """msgpack default: ndarray → {__ndarray_class__, as_npy}."""
    if isinstance(o, np.ndarray):
        b = io.BytesIO()
        np.save(b, o, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": b.getvalue()}
    return o


def dec(o):
    """msgpack object hook: {__ndarray_class__, as_npy} → ndarray."""
    if isinstance(o, dict) and "__ndarray_class__" in o:
        return np.load(io.BytesIO(o["as_npy"]), allow_pickle=False)
    return o


def parse(argv=None):
    """Server command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help=".../pretrained_model")
    ap.add_argument("--root", required=True, help="the training dataset (meta/tasks.jsonl · parquet task/axis index)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default loopback)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="dir for modality.json · READY · STOP · serve_heartbeat (unset = none)")
    ap.add_argument("--oracle-map", default="", help="oracle (see the module doc) · needs --out (log)")
    ap.add_argument(
        "--oracle-root",
        default="",
        help="oracle labels · states from this dataset (a labels-only conversion) instead of --root",
    )
    ap.add_argument("--oracle-exact", action="store_true", help="the oracle returns the training label as is")
    ap.add_argument("--oracle-hold-end", action="store_true", help="past the oracle episode end, repeat its last label")
    ap.add_argument("--const-state-fix", action="store_true", help="feed constant training state columns their value")
    ap.add_argument("--const-state-eps", type=float, default=0.0, help="width under which a column counts as constant")
    ap.add_argument("--sampler", default="", choices=("",) + SAMPLERS)
    ap.add_argument("--infer-steps", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    if a.oracle_map and not a.out:
        raise SystemExit("--oracle-map needs --out (the oracle log)")
    smp = sampler_override(a.sampler, a.infer_steps)
    pol = DPServe(
        a.ckpt,
        a.root,
        a.device,
        a.oracle_map,
        f"{a.out}/oracle.jsonl" if a.oracle_map else "",
        a.oracle_root,
        sampler=smp,
        const_state_fix=a.const_state_fix,
        const_state_eps=a.const_state_eps,
        oracle_exact=a.oracle_exact,
        oracle_hold_end=a.oracle_hold_end,
    )
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{a.host}:{a.port}")
    ep = {
        "ping": (lambda: {"status": "ok", "message": "Server is running"}, False),
        "get_action": (pol.get_action, True),
        "reset": (pol.reset, True),
        "get_modality_config": (pol.modality_config, False),
    }
    print(f"[E5_DP_SERVER] listening tcp://{a.host}:{a.port}", flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(f"{a.out}/modality.json", "w") as fh:
            json.dump(pol.modality_json(), fh, indent=1)
        with open(f"{a.out}/READY", "w") as fh:
            fh.write(
                f"READY {time.strftime('%Y-%m-%dT%H:%M:%S')} tcp://{a.host}:{a.port} ckpt={a.ckpt} "
                f"modality={a.out}/modality.json\n"
            )
        print(f"[E5_DP_SERVER] READY modality={a.out}/modality.json stop={a.out}/STOP", flush=True)
    beat = 0.0
    while True:
        if a.out:
            if time.time() - beat > 60:
                with open(f"{a.out}/serve_heartbeat", "w") as fh:
                    fh.write(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
                beat = time.time()
            if os.path.exists(f"{a.out}/STOP"):
                print("[E5_DP_SERVER] STOP seen — ending", flush=True)
                if pol.ms is not None:
                    pol.ms_report()
                break
            if not sock.poll(1000):
                continue
        req = msgpack.unpackb(sock.recv(), object_hook=dec)
        name = req.get("endpoint", "get_action")
        if name == "kill":
            sock.send(msgpack.packb({"status": "ok"}))
            break
        try:
            fn, needs = ep[name]
            out = fn(**req.get("data", {})) if needs else fn()
            sock.send(msgpack.packb(out, default=enc))
        except Exception as e:  # noqa: BLE001  the client raises on {"error": …} — the server keeps serving
            print(f"[E5_DP_SERVER] error in {name}: {e!r}", flush=True)
            sock.send(msgpack.packb({"error": repr(e)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
