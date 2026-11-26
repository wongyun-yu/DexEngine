"""Placement augment Isaac session: the same dump and the same recorder (rollout.recorder.CollectRecorder) as the
record stage-1 session (session.entry --stage record), with augment hooks only. Canonical and record code are not
changed.

Hooks
  traj_record.TrajRecorder -> AugRecorder (a CollectRecorder subclass): variant marks, failure diagnosis (end step,
      object error, wrist error, lift) and a per-clip success table
env: AUG_VARIANTS (variant table json: name -> dx · dy · yaw) · AUG_MODE (A: policy rollout, the only one) · AUG_DIAG
     (diag json output) · AUG_K_KEEP (cap of successful rollouts kept per variant; the canonical record augment gets
     policy.K_KEEP from augment.run run_spec explicitly · empty = the recorder's rollout.recorder.K_TOP)
Markers: [AUG_PLACE] <name> dx dy yaw (per clip; a clip missing from the table dies) · DIAG_MARK …
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.repo import REPO_ROOT

HERE = Path(__file__).resolve().parent
ROOT = REPO_ROOT  # tree root
from dexengine.record.session import entry as CS  # noqa: E402

MARK = "[AUG_SESSION]"
DIAG_MARK, DIAG_WRITTEN = "[AUG_DIAG] ", " rows -> "  # diag line head and its 'written' word (read by augment.run)


def _variants() -> dict:
    p = os.environ.get("AUG_VARIANTS", "")
    if not p:
        raise InputMissingError(f"{MARK} AUG_VARIANTS missing")
    return json.loads(Path(p).read_text())


def _variant_yaws(clip_names) -> list[float]:
    """Variant yaw (deg) per session clip · marker [AUG_PLACE]; a clip missing from the variant table dies."""
    V = _variants()
    miss, yaws = [], []
    for c in clip_names:
        key = c if c in V else (c + "_pl_rt" if c + "_pl_rt" in V else None)
        if key is None:
            miss.append(c)
            yaws.append(0.0)
            continue
        v = V[key]
        yaws.append(float(v["yaw"]))
        print(
            f"[AUG_PLACE] {c} dx={v['dx']:+.3f} dy={v['dy']:+.3f} yaw={v['yaw']:+.1f} "
            f"mode={os.environ.get('AUG_MODE')}",
            flush=True,
        )
    if miss:
        raise SystemExit(f"{MARK} clips missing from the variant table {miss[:3]} … ({len(miss)})")
    return yaws


def _wrist_terms(raw_env) -> list:
    """Wrist terms for the diagnosis (action terms with a virtual wrist)."""
    am = raw_env.action_manager
    terms = [(n, am.get_term(n)) for n in am.active_terms]
    terms = [(n, t) for n, t in terms if hasattr(t, "_virtual_wrist_w") and hasattr(t, "_side_rb")]
    print(f"{MARK} diagnosis wrist terms {[n for n, _ in terms]}", flush=True)
    return terms


def v4_fresh_obs(env, obs, batch: int):
    """First policy obs of a batch where the v4 approach ran: the obs measured at reset (before the approach,
    velocity 0) is read again at the current state (approach end). Oracles: reset obs = obs_buf['policy'] (the wrapper
    transform is the identity) · shape · dtype · sum of term widths. Marker [V4_OBS_REFRESH] = the hand term velocity
    fields (wrist linear and angular velocity + finger joint velocities) before -> after · max robot joint velocity."""
    import torch

    om = env.observation_manager
    if isinstance(obs, dict) or not torch.equal(env.obs_buf["policy"], obs):
        raise SystemExit("[V4_OBS_REFRESH] reset obs != obs_buf['policy']: the wrapper transform is not the identity")
    new = om.compute()["policy"]
    if new.shape != obs.shape or new.dtype != obs.dtype:
        raise SystemExit(f"[V4_OBS_REFRESH] shape {tuple(new.shape)}/{new.dtype} != {tuple(obs.shape)}/{obs.dtype}")
    parts, o = [], 0
    for n, d, cfg in zip(
        om.active_terms["policy"], om.group_obs_term_dim["policy"], om._group_obs_term_cfgs["policy"], strict=True
    ):
        w = 1
        for x in d:
            w *= int(x)
        fid = getattr(cfg.func, "_indep_joint_ids", None)
        # hand term (HandObservationV2): wrist linear and angular velocity + finger joint velocities = [9+F, 15+2F)
        if fid is not None:
            F = len(fid)
            a, b = o + 9 + F, o + 15 + 2 * F
            a0, a1 = obs[:, a:b].abs().max().item(), new[:, a:b].abs().max().item()
            parts.append(f"{n} velocity fields {a0:.4f}->{a1:.4f}")
        o += w
    if o != new.shape[-1]:
        raise SystemExit(f"[V4_OBS_REFRESH] sum of term widths {o} != obs width {new.shape[-1]}")
    jv = " · ".join(
        f"{k} {a.data.joint_vel.abs().max().item():.4f}"
        for k, a in env.scene.articulations.items()
        if k.startswith("robot")
    )
    print(
        f"[V4_OBS_REFRESH] batch {batch} · changed fields {int((new != obs).any(0).sum())}/{new.shape[-1]} · "
        f"velocity term max|·| before->after {' · '.join(parts) or 'none'} · joint velocity max {jv}",
        flush=True,
    )
    return new


class _V4ObsMixin:
    def obs_after_begin(self, obs):  # noqa: V105 — read by name in dexengine/rl/judge/rollouts_batch.py
        """Called by dexengine.rl.judge.rollouts right after begin_batch. When the v4 approach ran in this batch, the
        obs measured at reset (state before the approach, velocity 0) is read again at the approach end state, so the
        first policy step also sees the approach end state (velocities included). Batches without the approach (off,
        warmup real batches) get the received obs back as it is (same bytes). Marker [V4_OBS_REFRESH] = velocity terms
        before -> after."""
        return obs if self._v4_ran is None else v4_fresh_obs(self.env, obs, self._v4_ran)


def _aug_session_hook():
    """First non-None object returned by the plugin factories registered as ("aug_session", name), else None."""
    from dexengine.record import hooks

    hooks.load_plugins()
    for name in hooks.names("aug_session"):
        got = hooks.get("aug_session", name)()
        if got is not None:
            return got
    return None


def _append_diag(self):
    """One diagnostic row per env of the finished batch."""
    L = self.length.cpu().numpy()
    early = self.early.cpu().numpy()
    lift = self._lift.cpu().numpy()
    wm = self._werr_max.cpu().numpy()
    cl = self._clen.cpu().numpy()
    for e in range(self.bs):
        cid = int(self.assign[e])
        self._diag.append(
            {
                "clip": self.clip_names[cid],
                "batch": self.b,
                "env": e,
                "warm": bool(self.warm),
                "succ": not bool(early[e]),
                "length": int(L[e]),
                "clip_len": int(cl[e]),
                "lift_max": float(lift[e]),
                "wrist_err_max": float(wm[e]),
            }
        )


def _wrist_err(self, fids, we):
    """Largest wrist error over the wrist terms; the first failed read switches the terms off."""
    import torch

    try:
        for _n, t in self._terms:
            wp, _wq = t._virtual_wrist_w()
            ref = self.traj_mgr.get_wrist_pos_w(fids, side=t._side_rb) + self.env.scene.env_origins
            we = torch.maximum(we, (wp - ref).norm(dim=-1))
    except Exception as e:  # noqa: BLE001 — any read failure only disables this diagnostic
        print(f"{MARK} wrist error read failed {e}", flush=True)
        self._terms = []
    return we


def _reset_diag(self, n):
    """Per-batch diagnostic tensors (lift, wrist error max/at, object error at, clip lengths)."""
    import torch

    self._lift = torch.zeros(n, device=self.dev)
    self._z0 = None
    self._werr_max = torch.zeros(n, device=self.dev)
    self._werr_at = torch.full((n,), -1.0, device=self.dev)
    self._oerr_at = torch.full((n,), -1.0, device=self.dev)
    self._clen = self.traj_mgr.clip_lengths.clone()


def _obj_cols(cols, r):
    """Object height and object-to-reference error per side column of the last row."""
    zs, oe = [], []
    for c in cols:
        p = r[:, c["obj_lo"] : c["obj_lo"] + 3]
        q = r[:, c["ref_lo"] : c["ref_lo"] + 3]
        zs.append(p[:, 2])
        oe.append((p - q).norm(dim=-1))
    return zs, oe


def _aug_recorder_class(base):
    import torch

    class AugRecorder(_V4ObsMixin, base):
        def __init__(self, raw_env, clip_names, out_path, **kw):
            super().__init__(raw_env, clip_names, out_path, **kw)
            # plugin physics hook (first non-None of the registered aug_session factories; None by default)
            self._v4 = _aug_session_hook()
            self._v4_ran = None  # the batch number when the approach ran in this batch (read by obs_after_begin)
            self._yaw = _variant_yaws(self.clip_names)
            self._terms = _wrist_terms(raw_env)
            # (clip, batch, env, warm, succ, length, clip_len, obj_err, wrist_err_max, lift_max, wrist_err_at)
            self._diag = []
            self._last = None

        def _row(self):
            r = super()._row()
            self._last = r
            return r

        def begin_batch(self, batch_idx, clip_assignment, batch_size):
            self._v4_ran = None
            if self._v4 is not None and self._v4.wants(batch_idx):
                self._v4.run(self, batch_idx, clip_assignment, batch_size)
                self._v4_ran = int(batch_idx)
            super().begin_batch(batch_idx, clip_assignment, batch_size)
            n = self.n_env
            _reset_diag(self, n)
            self._upd(torch.zeros(n, dtype=torch.bool, device=self.dev), init=True)

        def _upd(self, newly, init=False):
            cols = self.sides_cols()
            r = self._last
            zs, oe = _obj_cols(cols, r)
            z = torch.stack(zs, -1).max(-1).values if zs else torch.zeros(self.n_env, device=self.dev)
            if init or self._z0 is None:
                self._z0 = z.clone()
            alive = self.length < 0
            self._lift = torch.where(alive, torch.maximum(self._lift, z - self._z0), self._lift)
            we = torch.zeros(self.n_env, device=self.dev)
            fids = self.env.episode_length_buf
            # the first failed read ends the loop and switches the wrist error off (the try spans the loop for that)
            we = _wrist_err(self, fids, we)
            self._werr_max = torch.where(alive, torch.maximum(self._werr_max, we), self._werr_max)
            if bool(newly.any()):
                self._werr_at[newly] = we[newly]
                o = torch.stack(oe, -1).max(-1).values if oe else torch.zeros(self.n_env, device=self.dev)
                self._oerr_at[newly] = o[newly]

        def after_step(self, step, terminated, truncated, done_mask):
            term, trunc = terminated.reshape(-1).bool(), truncated.reshape(-1).bool()
            newly = (term | trunc) & ~done_mask
            # after reset the state is this row, so a terminated env's value is the previous row's; simplified here: the
            #   max before termination
            super().after_step(step, terminated, truncated, done_mask)
            self._upd(torch.zeros_like(newly))
            if bool(newly.any()):
                self._werr_at[newly] = self._werr_max[newly]

        def end_batch(self, park_invalid=False):
            _append_diag(self)
            super().end_batch(park_invalid)

        def finalize(self, aggregate):
            out_diag = os.environ.get("AUG_DIAG", "")
            if out_diag:
                Path(out_diag).write_text(json.dumps(self._diag))
                print(f"{DIAG_MARK}{len(self._diag)}{DIAG_WRITTEN}{out_diag}", flush=True)
            # [TRAJ_REC] DONE (count equation, atomic write) · [KIT_RC] closing-start marker
            out = super().finalize(aggregate)
            close_skip([self.out_path] + ([out_diag] if out_diag else []))
            return out  # never reached (close_skip ends the process)

    return AugRecorder


CLOSE_SKIP = "[AUG_CLOSE_SKIP]"
#: render branch env: when any is on, no close skip (a record session records physics only, and the trainer never gets
#: DEXENGINE_TRAJ_RECORD together with a render branch; the path that cuts render outputs with os._exit is still
#: closed by construction)
RENDER_ENV = ("DEXENGINE_FROZEN_RENDER",)


def close_skip(paths, exit_=os._exit) -> None:
    """After checking the outputs (trajectory file, diagnosis) and flushing, end with rc 0 without waiting for Kit to
    close; marker [AUG_CLOSE_SKIP]. Why: 21 of 22 augment sessions stopped without a line in the app close
    (wrapped.close · simulation_app.close) after [TRAJ_REC] DONE and the closing-start marker and used the whole
    5,400 s limit. This does not fix the third-party extension exception; it removes the dependency on closing, and
    the outputs check (not rc) decides the result. A missing output dies instead of ending (an exception in our frame
    -> rc 1)."""
    import sys

    render = [k for k in RENDER_ENV if os.environ.get(k, "") not in ("", "0")]
    if render:  # render sessions (video encoder buffers, depth files) could lose outputs to os._exit: close normally
        print(f"{CLOSE_SKIP} not done: render branch env {render} · normal close", flush=True)
        return
    miss = [p for p in paths if not p or not Path(p).is_file() or Path(p).stat().st_size == 0]
    if miss:
        raise InputMissingError(f"{CLOSE_SKIP} outputs missing {miss}: no close skip")
    print(
        f"{CLOSE_SKIP} outputs {len(paths)} checked ({', '.join(Path(p).name for p in paths)}): ending with rc 0 "
        "without waiting for Kit to close (closing dependency removed; the extension exception is not fixed)",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    exit_(0)


def _hook_record(mod):
    from dexengine.record.rollout import recorder as REC

    kk = os.environ.get("AUG_K_KEEP", "")
    # successful rollouts the post collision gate looks at (canonical augment = policy.K_KEEP, apart from the record
    #   K_TOP): marker + meta.k_keep
    if kk:
        REC.K_TOP = int(kk)
        print(f"{CS.HOOK} dexengine.record.rollout.recorder.K_TOP → {REC.K_TOP} (AUG_K_KEEP)", flush=True)
    base = REC.collect_recorder_class(mod.TrajRecorder)
    mod.TrajRecorder = _aug_recorder_class(base)
    print(f"{CS.HOOK} traj_record.TrajRecorder → collect_record.CollectRecorder → aug_session.AugRecorder", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    a = ap.parse_args()
    argv, env = CS.load_dump(a.dump)
    CS.require(argv, "--num_envs")
    if not env.get("DEXENGINE_TRAJ_RECORD") or env.get("DEXENGINE_FROZEN_REPLAY") != "1":
        raise InputMissingError(f"{MARK} the record dump lacks DEXENGINE_TRAJ_RECORD · DEXENGINE_FROZEN_REPLAY=1")
    for k in (
        "AUG_VARIANTS",
        "AUG_MODE",
        "AUG_DIAG",
        "AUG_K_KEEP",
        "AUG_V4_APPROACH",
        "AUG_V4_APPROACH_MODE",
        "AUG_V4_APPROACH_OUT",
    ):
        if os.environ.get(k):
            env[k] = os.environ[k]
    if env.get("AUG_MODE") != "A":
        raise SystemExit(f"{MARK} AUG_MODE = A (got {env.get('AUG_MODE')!r})")
    sd = os.environ.get("AUG_V4_SEED", "")  # v4 retry seed (aug_v4.RETRY; none = the dump as it is, same bytes)
    if sd:
        if not sd.isdigit():
            raise SystemExit(f"{MARK} AUG_V4_SEED={sd!r}: non-negative integers only")
        if "--seed" in argv:
            i = argv.index("--seed")
            argv = argv[:i] + argv[i + 2 :]
        argv = [*argv, "--seed", sd]
        print(f"[AUG_V4_SEED] env seed {sd} (retry · aug_v4.RETRY)", flush=True)
    hooks = {"dexengine.rl.eval.traj_record": _hook_record}
    CS.run_entry(argv, env, hooks)


if __name__ == "__main__":
    main()
