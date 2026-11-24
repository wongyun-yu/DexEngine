"""FlashDexRetarget evaluation recorder (Tab. 2 FDR row): the authors' own evaluation, unchanged, on a saved checkpoint,
with
one extra callback that records every eval env's object poses and the reference it was given, per control step.

What runs = the authors' train.py setup with the launcher's overrides (scripts/train_flashdexretarget.sh ·
ROBOT=sharpa)
and the
row's training overrides (replay buffer in host RAM · seed 42), then evaluation.evaluate's own rollout
(_rollout — deterministic actor · every clip from frame 0 · terminations, curriculum, object assist and obs noise
off,
as
config/eval/motion_tracking.yaml) on the training env (2048 envs, the authors' object-aware plan). Eval-only changes:
the
checkpoint is loaded (agent_load_path · strict, with the authors' obs-layout check), eval.num_per_motion = N replicas
per clip
(the authors' plan caps it at the envs that host the clip's object — ≈ 34 at 2048 envs, so N must fit; asserted), no
success
archive, a 200 k-row replay buffer (the 50 M default would pin 88 GB of host RAM; eval never samples it), wandb off.
Nothing is trained (learn() is never called).

Outputs (<out>/): raw/<motion name>.npz per clip — t_frame (K,) motion frame · sim_pos_<side> (N,K,3) ·
sim_quat_<side>
(N,K,4 wxyz) · ref_pos_<side> (K,3) · ref_rotmat_<side> (K,3,3) — the authors' z-up world, env origin subtracted ·
obj_id_<side> · obj_scale_<side> · fps · head (frames the authors' pipeline dropped before motion frame 0, measured
against
the imported human_demo clip) · replicas; authors_eval.json — the authors' own metrics of this rollout (SR@k,
success_ever_*,
per clip pass counts under their obj SR@1.0 and their SPIDER rule); meta.json. Markers [FDR_REC] ….
usage (cwd = <workspace>/code/flashdexretarget, its env sourced, FDR_ROBOT_USD unset):
  python eval_record.py --ckpt <step dir> --motion <motion.pt> --demo-dir <dataset>/human_demo --out <dir> --replicas N
         [--num-envs n (default: the training env; more when N needs more envs per motion)]
         [--variant <fdr/variants/<name>.txt> — the training row's variant: its 'override' lines join the training
          overrides (one owner for the training row and this recorder); without it = the plain authors' env]
         [--probe-steps K — no checkpoint, no policy: zero residual action for K control steps on eval batch 0, then
          probe.json (each active clip's object motion from the reset · its support disks as posed in PhysX vs the
          motion's table · the PhysX object masses) — the variant smoke]"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

F = Path.cwd().resolve()
sys.path.insert(
    0, str(F / "src")
)  # the FlashDexRetarget checkout (third-party; `runner`, `envs`, `common` below are its modules)
ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--ckpt")
ap.add_argument("--motion", required=True)
ap.add_argument("--demo-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--replicas", type=int, required=True)
ap.add_argument(
    "--num-envs",
    type=int,
    default=None,
    help="default = the training env (2048); the authors' planner caps a "
    "motion's rollouts at its object slot's envs, so N = 64 needs 4096 (each clip must get exactly N)",
)
ap.add_argument("--variant", default=None, help="fdr/variants/<name>.txt (the training row's variant)")
ap.add_argument("--probe-steps", type=int, default=0, help="variant smoke: zero action for K steps, no checkpoint")
a = ap.parse_args()
if bool(a.ckpt) == bool(a.probe_steps):
    sys.exit("[FDR_REC] FAIL give exactly one of --ckpt (a record) or --probe-steps (the variant smoke)")
OUT = Path(a.out)
(OUT / "raw").mkdir(parents=True, exist_ok=True)


def say(*x):
    print("[FDR_REC]", *x, flush=True)


def die(msg):
    print(f"[FDR_REC] FAIL {msg}", flush=True)
    sys.stdout.flush()
    os._exit(1)


if a.ckpt:
    ck = Path(a.ckpt).resolve()
    rs = json.loads((ck / "resume_state.json").read_text())
    sha = hashlib.sha256((ck / "flashsac.pt").read_bytes()).hexdigest()
    say(f"CKPT {ck.name} env_step={rs.get('env_step')} flashsac.pt sha256={sha[:16]}")
else:
    ck, rs, sha = Path("/nonexistent-probe"), {}, None
    say(f"PROBE {a.probe_steps} control steps · zero residual action · no checkpoint")
# the variant: 'override <hydra override>' lines join the training overrides — the file is the one owner,
# training reads the same file; its 'proof' lines are checked against the run logs (variant.sh check)
var_ov, var_name, var_sha = [], None, None
if a.variant:
    vp = Path(a.variant).resolve()
    var_name = vp.stem
    var_sha = hashlib.sha256(vp.read_bytes()).hexdigest()
    for ln in vp.read_text().splitlines():
        w = ln.split(None, 1)
        if not w or w[0].startswith("#"):
            continue
        if w[0] == "override":
            var_ov.append(w[1].strip())
        elif w[0] != "proof":
            die(f"variant {vp.name}: unknown line kind {w[0]!r} (override | proof | #)")
    if not var_ov:
        die(f"variant {vp.name} has no override line")
    say(f"VARIANT {var_name} sha256={var_sha[:16]} · overrides {' '.join(var_ov)}")
if "FDR_ROBOT_USD" in os.environ:
    die("FDR_ROBOT_USD is set (env.sh points it at the XHAND USD; the sharpa launcher unsets it)")

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
motion = str(Path(a.motion).resolve())
ds = Path(motion).parent.parent.name
# the launcher's arguments (scripts/train_flashdexretarget.sh, ROBOT=sharpa) + the row's training overrides
train_ov = [
    "envs/robot@robot=sharpa_bimanual",
    f"commands.motion_file={motion}",
    f"logger.wandb.run_name=flashdexretarget-{ds}-sharpa",
    f"logger.wandb.tags=[{ds},sharpa]",
    "runner.agent.buffer_device_type=cpu",
    "logger.wandb.mode=offline",
    "seed=42",
] + var_ov
eval_ov = ([f"agent_load_path={ck}"] if a.ckpt else []) + [
    f"eval.num_per_motion={a.replicas}",
    "eval.save_success_data=false",
    "runner.agent.buffer_max_length=200000",
    "logger.wandb.mode=disabled",
    f"save_path={OUT}/run",
]
ov = train_ov + eval_ov + ([f"num_envs={a.num_envs}"] if a.num_envs else [])
with hydra.initialize_config_dir(version_base=None, config_dir=str(F / "config")):
    cfg = hydra.compose(config_name="flashdexretarget", overrides=ov)
OmegaConf.resolve(cfg)
say("OVERRIDES " + " ".join(o.replace(str(ck), "<ckpt>").replace(motion, "<motion.pt>") for o in ov))
say(
    f"CFG num_envs={cfg.num_envs} eval.num_per_motion={cfg.eval.num_per_motion} "
    f"sampling={cfg.eval.command.sampling_mode} "
    f"start_frame={cfg.eval.command.start_frame} noise={cfg.eval.command.noise_to_initial_level} "
    f"clear_terminations={cfg.eval.env.clear_terminations} disable_object_pin={cfg.eval.env.disable_object_pin} "
    f"control_dt={cfg.sim.dt * cfg.decimation:.6f}"
)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
from isaaclab.app import AppLauncher

AppLauncher({"headless": True})
import numpy as np
import torch
from common.logger import setup_logger
from common.utils import configure_torch_backends, set_seed
from envs import create_envs
from eval_record_probe import probe  # the sibling file (this folder is sys.path[0] when run by path)
from evaluation.evaluate import _rollout, reduce_metrics, tracking_success_metrics
from hydra.utils import instantiate
from runner import create_runner

set_seed(cfg.seed)
configure_torch_backends()
train_env, eval_env = create_envs(cfg, device=cfg.device)
setup_logger(cfg.logger, cfg.runner)
if a.probe_steps:
    probe(a, cfg, eval_env, OUT, ov, motion, var_name, var_sha, say)
runner = create_runner(
    cfg.runner, cfg.eval, train_env, eval_env=eval_env, headless=True
)  # loads the checkpoint (strict)
agent, raw = runner._agent, eval_env.unwrapped
cmd = raw.command_manager.get_term(cfg.eval.command_name)
lib = cmd.motion_lib
names = [f.rsplit("#", 1)[-1] for f in lib.motion_files]
sides = list(cmd._side_list)
packed = torch.load(motion, map_location="cpu", weights_only=False)
say(f"MOTIONS {len(names)} · sides {sides} · eval batches {raw.eval_num_batches} · envs {raw.num_envs}")


class ObjRecord:
    """After the reset and after every eval step: each env's motion frame, sim object pose and the reference object pose
    it was given, env origin subtracted (the authors' RolloutRecord convention) — on the host, float32."""

    rollout_steps = 1

    def on_eval_setup(self, env):
        pass

    def _grab(self, env):
        o = env.scene.env_origins[:, None, :]
        rot = torch.stack([lib.obj_rot[s][cmd._motion_flat_ids] for s in sides], 1)
        return (
            cmd.motion_steps.detach().clone().cpu(),
            (cmd.sim_obj_trans_w - o).float().cpu(),
            cmd.sim_obj_quat_w.float().cpu(),
            (cmd.ref_obj_trans_w - o).float().cpu(),
            rot.float().cpu(),
        )

    def on_start(self, env):
        self.frames = [self._grab(env)]

    def on_step(self, env):
        self.frames.append(self._grab(env))

    def collect_state(self, active_mask=None):
        m = slice(None) if active_mask is None else active_mask.cpu()
        st = [torch.stack([f[i] for f in self.frames], 1)[m] for i in range(5)]  # (E, K+1, ...)
        mid = cmd.motion_ids if active_mask is None else cmd.motion_ids[active_mask]
        return {
            "steps": st[0],
            "sim_pos": st[1],
            "sim_quat": st[2],
            "ref_pos": st[3],
            "ref_rot": st[4],
            "motion_ids": mid.cpu(),
        }

    def reduce_state(self, states):
        return {}


cbs = [instantiate(c) for c in cfg.eval.callbacks if "rollout_record" not in str(c.get("_target_", ""))]
rec = ObjRecord()
cbs.append(rec)
say("CALLBACKS " + " ".join(type(c).__name__ for c in cbs))
t0 = time.time()
states = _rollout(agent, eval_env, cbs, cfg.device)
say(f"ROLLOUT done {time.time() - t0:.1f} s")
metrics = reduce_metrics(cbs, states)
metrics.update(tracking_success_metrics(raw, cfg.eval, cbs, states, {}, None, int(rs.get("env_step") or 0)))
tp = next(c for c in cbs if hasattr(c, "per_env_fail_obj"))
st = states[id(tp)]
fail_obj = tp.per_env_fail_obj(st)[:, tp.k1_idx].cpu()
scored = tp.per_env_scored(st).cpu()
sp_ok = tp.per_env_spider_ok(st).cpu() if hasattr(tp, "per_env_spider_ok") else None
R = states[id(rec)]
mids = torch.cat([s["motion_ids"] for s in R])
if hasattr(tp, "per_env_motion_ids") and not torch.equal(mids.long(), tp.per_env_motion_ids(st).cpu().long()):
    die("the recorder's env order ≠ the authors' metric callback's (motion ids differ)")
nf_all = lib._motion_num_frames.cpu()
demo = Path(a.demo_dir)
per_clip, authors, off = {}, {}, 0
for b, s in enumerate(R):
    for m in sorted(set(s["motion_ids"].tolist())):
        sel = (s["motion_ids"] == m).nonzero().flatten()
        nf = int(nf_all[m])
        name = names[m]
        # the method's own clock, by the authors' convention (TrackingPerformance: "cmd.metrics compare the object with
        # the
        # frame seen at the PREVIOUS callback … computed before motion_steps advances" — the rewards likewise): the
        # state
        # after step e is graded against frame steps[e−1], with the reference read at entry e−1. (env.reset ends with
        # a
        # command update, so the reset entry already reads frame 1: state e ↔ frame e, time e/60 s — the reset state
        # itself,
        # frame 0, is not graded, as in the authors' metrics — simulator/isaacsim/env.py reset →
        # command_manager.compute(dt=0)
        # → _update_command.) Each frame's first graded
        # state is kept (the command clamps at nf − 1 after a clip ends); every replica must show the same sequence
        allst = s["steps"][sel]  # (N, K+1) — entry 0 = after the reset
        st0 = allst[0].numpy()
        if b == 0 and not per_clip:
            say(
                f"CLOCK {name} (nf {nf}): entries {len(st0)} · motion_steps first {st0[:8].tolist()} · "
                f"last {st0[-3:].tolist()}"
            )
        if not bool((allst == allst[:1]).all()):
            die(f"{name}: replicas disagree on the motion frame sequence")
        fprev = st0[:-1]
        keep = np.concatenate([[True], fprev[1:] != fprev[:-1]])
        fr = fprev[keep]
        ridx = np.nonzero(keep)[0]
        sidx = ridx + 1
        gaps = int(np.sum(np.diff(fr) - 1))
        if not np.all(np.diff(fr) >= 1) or fr[0] > 1 or fr[-1] < nf - 2 or gaps > 2:
            die(
                f"{name}: graded frames {fr[:6].tolist()} … {fr[-3:].tolist()} (nf {nf}, {gaps} skipped) — "
                "not the clip's clock "
                f"(motion_steps {st0[:8].tolist()} …)"
            )
        kidx, kref = torch.as_tensor(sidx), torch.as_tensor(ridx)
        # time base = the clip's own frame rate (the reference timeline · exactly 60), after checking that one control
        # step is
        # one frame — not 1/(sim.dt·decimation): sim.dt is the rounded 0.0041666667, and 59.99999976 Hz put the last
        # frame
        # past the reference end (the 340M test #2 scored 0/60 on that: score.py read 4 missing steps)
        d = np.load(demo / f"{name}.npz", allow_pickle=True)
        fps = float(d["fps"])
        ctrl = 1.0 / (float(cfg.sim.dt) * int(cfg.decimation))
        if abs(fps - ctrl) > 1e-3:
            die(f"{name}: clip fps {fps} ≠ control rate {ctrl}")
        out = {
            "t_frame": fr.astype(np.int64),
            "fps": np.float64(fps),
            "replicas": np.int64(len(sel)),
            "batch": np.int64(b),
            "steps_all": st0.astype(np.int16),
            "state_entry": sidx.astype(np.int32),
        }
        heads = []
        for j, side in enumerate(sides):
            rp = s["ref_pos"][sel[0], kref, j].double().numpy()
            # frames the authors' pipeline dropped before motion frame 0 = where the imported clip equals the reference
            # (every object position of the motion = the clip's from that frame on, ≤ 1e-5 m) — measured, never
            # assumed
            hp = d[f"obj_{side}_pose"][:, :3, 3].astype(np.float64)
            hd = [h for h in range(len(hp) - nf + 1) if np.abs(hp[h + fr] - rp).max() <= 1e-5]
            if len(hd) != 1:
                die(f"{name} {side}: the reference matches the imported clip at heads {hd[:5]}")
            heads.append(hd[0])
            slot = int(lib.traj_obj_slot[side][m])
            oid = os.path.basename(str(packed[f"{side}_object_mesh_dirs"][slot]))
            if oid != str(d[f"obj_{side}_id"]):
                die(f"{name} {side}: motion.pt object {oid} ≠ the imported clip's {d[f'obj_{side}_id']}")
            out[f"obj_id_{side}"] = np.array(oid)
            out[f"obj_scale_{side}"] = np.float64(packed[f"{side}_object_mesh_scales"][slot])
            out[f"sim_pos_{side}"] = s["sim_pos"][sel][:, kidx, j].numpy()
            out[f"sim_quat_{side}"] = s["sim_quat"][sel][:, kidx, j].numpy()
            out[f"ref_pos_{side}"] = rp.astype(np.float32)
            out[f"ref_rotmat_{side}"] = s["ref_rot"][sel[0], kref, j].numpy()
            # every replica of a motion was given the same reference — else the env/motion pairing is wrong
            if float((s["ref_pos"][sel][:, kref, j] - s["ref_pos"][sel[:1]][:, kref, j]).abs().max()) > 1e-6:
                die(f"{name} {side}: replicas carry different references")
        if len(set(heads)) != 1:
            die(f"{name}: heads differ by side {heads}")
        out["head"] = np.int64(heads[0])
        # after the reset the sim object stands at the clip's frame 0 (the authors' reset writes it; the reset's own
        # command
        # update has already moved motion_steps to 1 — env.reset calls command_manager.compute — so frame 0 is read
        # from
        # the motion library, env-local as the recorded states)
        r0 = [lib.obj_trans[sd][int(lib.length_starts[m])].float().cpu() for sd in sides]
        g = [float((s["sim_pos"][sel][:, 0, j] - r0[j]).norm(dim=-1).max()) for j in range(len(sides))]
        out["frame0_gap_m"] = np.array(g)
        np.savez_compressed(OUT / "raw" / f"{name}.npz", **out)
        gi = off + sel
        authors[name] = {
            "replicas": int(sel.numel()),
            "obj_sr1_pass": int(((~fail_obj[gi]) & scored[gi]).sum()),
            "spider_pass": int((sp_ok[gi] & scored[gi]).sum()) if sp_ok is not None else None,
        }
        per_clip[name] = {
            "replicas": int(sel.numel()),
            "frames": nf,
            "frames_kept": int(len(fr)),
            "first_frame": int(fr[0]),
            "skipped_frames": gaps,
            "head": heads[0],
            "batch": b,
            "frame0_gap_mm": [round(x * 1e3, 3) for x in g],
        }
    off += s["motion_ids"].shape[0]
reps = sorted({v["replicas"] for v in per_clip.values()})
g0 = max(max(v["frame0_gap_mm"]) for v in per_clip.values())
say(
    f"EXPORTED {len(per_clip)}/{len(names)} clips · replicas {reps} · heads "
    f"{sorted({v['head'] for v in per_clip.values()})} · "
    f"frame-0 sim-vs-reference gap max {g0:.2f} mm"
)
if len(per_clip) != len(names) or reps != [a.replicas]:
    die(f"exported {len(per_clip)}/{len(names)} clips · replicas {reps} ≠ [{a.replicas}]")
if g0 > 5.0:
    die(f"frame-0 sim object ≠ reference by {g0:.2f} mm (> 5 mm) — the recorded object frame is not the reference's")
authors_rec = {
    "env_step": rs.get("env_step"),
    "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
    "per_clip": authors,
    "rule": "obj SR@1.0 = time-mean object error ≤ 3 cm and ≤ 30 deg (config/eval/motion_tracking.yaml)"
    " · spider = the authors' TrackingPerformance SPIDER rule",
}
with open(OUT / "authors_eval.json", "w") as fh:
    json.dump(authors_rec, fh, indent=1)
meta_rec = {
    "ckpt": ck.name,
    "env_step": rs.get("env_step"),
    "flashsac_sha256": sha,
    "replicas": a.replicas,
    "variant": var_name,
    "variant_sha256": var_sha,
    "overrides": [o.replace(str(ck), "<ckpt>").replace(motion, "<motion.pt>") for o in ov],
    "num_envs": int(cfg.num_envs),
    "eval_batches": int(raw.eval_num_batches),
    "sides": sides,
    "clips": per_clip,
}
with open(OUT / "meta.json", "w") as fh:
    json.dump(meta_rec, fh, indent=1)
ao = sum(v["obj_sr1_pass"] >= 1 for v in authors.values())
asp = sum((v["spider_pass"] or 0) >= 1 for v in authors.values())
say(
    f"AUTHORS_EVAL obj SR@1.0 (>=1 of {a.replicas}) {ao}/{len(authors)} · their SPIDER rule {asp}/{len(authors)} · "
    + " ".join(
        f"{k}={v:.4f}"
        for k, v in sorted(metrics.items())
        if isinstance(v, float) and k.startswith(("success_rate_obj_1.0", "success_rate_1.0", "success_ever_spider"))
    )
)
say("DONE")
sys.stdout.flush()
os._exit(0)  # isaacsim 5.1 segfaults in close() (the authors' runner exits the same way)
