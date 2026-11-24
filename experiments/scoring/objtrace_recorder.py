"""Object poses of every judge episode (default off) — the input of the Ours rows of Tab. 2 (``ours_objtrace.py``).

Why: Tab. 2 scores every method with CHORD's metrics (ADD-AUC · SPIDER SR · ManipTrans SR) on object-pose time series.
The Ours rows need every judge episode's per-step object pose, while the trajectory recorder (traj_record) keeps only
the top-k successes and one failure per clip. This module takes the object and reference columns of every episode from
the same buffer (read only) and writes them to a second file.
Switch: DEXENGINE_E123_OBJTRACE=<.npz> (name owner ``ENV_VAR``; an environment variable because the Isaac Lab judge
  process reads it), only with DEXENGINE_TRAJ_RECORD in the single DEXENGINE_FROZEN_REPLAY=1 judge branch (otherwise
  the launch stops — ``launch_problems``). When on, ``dexengine.rl.train.runner`` wraps the recorder class (including
  the collection hooks' subclasses) with ``recorder_class``: each of the four hooks calls super() and adds this record.
  Off = not wrapped (traj_record unchanged, same files and markers).
Markers: ``[E123_OBJTRACE] on → <path> · sides … · NE …`` · ``DONE episodes … · frames … · sha256=<16>`` ·
``FAIL …``
  (job scripts grep ``[E123_OBJTRACE] DONE .*sha256=``) and one ``[E123_PHYS] batch <b> <side>(<entity>): mass …`` line
  per side and batch (or ``… unavailable — …``).
Count checks (counted apart from the recorder): (1) before the trajectory file is written — every episode row = the
  recorder's rows (values · order) and frames = Σ lengths, else neither file is written; (2) before the second file —
  real-batch episodes = the judge's num_episodes and real-batch successes = the judge's completed count (values from
  the trajectory file's meta.judge, owned by the recorder's finalize).

File (SCHEMA "e123_objtrace/1" · allow_pickle=False):
  meta            JSON — schema · written · recorded · traj_record (path · sha256) · ep_cols · obj_rule ·
  ref_rule ·
                  globals · clip_meta_fields · missing_for_inverse_chain + the fields FROM_TRAJ_META of the trajectory
                  meta
  clip_names      [K] str — clip order (= the trajectory file = _loader.sequence_names)
  ep_table        [E, 7] int64 — EP_COLS (clip, batch, env, warm, valid, success, length) = the trajectory file's
  ep_table ·
                  warm=1 the warm-up the judge discards · valid=0 a PARK-invalid batch · success=1 judge-completed ·
                  valid frames [0, length) · a policy-free PARK probe batch (trajectory meta recorded.probe_batches)
                  has no judge episodes: its episodes still running at the last cut end there (success 0)
  ep_offset       [E] int64 — first row in obj/ref/ref_frame · ep_start_frame [E] int64 — start_frames[env] at batch
  reset side_names · entity_names [S] str — the sides of the second axis · each side's object rigid-body name
  obj             [F, S, 7] float32 — root_link_pose_w (link frame, not the COM) position − env_origins · quat wxyz
  ·
                  ADD vertices are in that spawn USD's prim frame (meta.object_assets — not the original TACO mesh
                  frame)
  ref             [F, S, 7] float32 — get_reference_state(episode_length_buf)[{side}_obj_pos|_obj_quat] — what the
                  tracking termination compares with obj at the same step (env-local)
  ref_frame       [F, S] int32 — the tree frame the step's reference was read from (measured side_abs_frames; the rule
                  check count is in meta.recorded)
  env_origins     [NE, 3] float32 — world position = obj[..., :3] + env_origins[env]
  clip_<name>     [K] — loader values of the run (clip_meta) · globals = meta.globals · values not found =
                  meta.missing_for_inverse_chain
Reading: z = np.load(p, allow_pickle=False); T = z["ep_table"]; i = real-batch ∧ valid rows (T[:, 3] == 0 ∧ T[:, 4] ==
1);
  o, L = z["ep_offset"][i], T[i, 6]; z["obj"][o:o + L, s] · z["ref"][o:o + L, s] — row k = s_k (after step k ·
  k=0 right after the batch reset) · spacing meta.step_dt s.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from experiments.scoring.objtrace_meta import FROM_TRAJ_META, clip_meta, global_meta, object_assets

ENV_VAR = "DEXENGINE_E123_OBJTRACE"  # value = the second output .npz; empty = off. Consumer = recorder_class
SCHEMA = "e123_objtrace/1"
EP_COLS = ("clip", "batch", "env", "warm", "valid", "success", "length")
MARK = "[E123_OBJTRACE]"


def launch_problems(objtrace: str, traj_record: str, replay: bool, worker: bool, render: bool) -> list[str]:
    """Launch check (runner): the reasons this branch has no consumer of the switch ([] = allowed)."""
    if not objtrace:
        return []
    bad = []
    if not traj_record:
        bad.append(f"{ENV_VAR} is consumed only by the trajectory recorder (DEXENGINE_TRAJ_RECORD)")
    if not replay or worker or render:
        bad.append(f"{ENV_VAR} is consumed only by the single DEXENGINE_FROZEN_REPLAY=1 judge branch")
    if not objtrace.endswith(".npz"):
        bad.append(f"{ENV_VAR} output must be a .npz: {objtrace}")
    if traj_record and os.path.abspath(objtrace) == os.path.abspath(traj_record):
        bad.append(f"{ENV_VAR} and DEXENGINE_TRAJ_RECORD are the same path")
    return bad


def recorder_class(base, out_path: str):
    """A subclass of ``base`` (traj_record.TrajRecorder or a subclass — e.g. the collection hooks' recorder) that adds
    the second record. Called by the runner only when the switch is on; every hook calls super() first."""

    class ObjTraceRecorder(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.objtrace = ObjTrace(self, out_path)

        def _capture(self, t: int) -> None:
            super()._capture(t)
            self.objtrace.capture(self, t)

        def begin_batch(self, batch_idx, clip_assignment, batch_size) -> None:
            super().begin_batch(batch_idx, clip_assignment, batch_size)
            self.objtrace.begin_batch(self)

        def after_step(self, step, terminated, truncated, done_mask) -> None:
            super().after_step(step, terminated, truncated, done_mask)
            self.objtrace.after_step(self, terminated, truncated, done_mask)

        def end_batch(self, park_invalid: bool = False) -> None:
            super().end_batch(
                park_invalid
            )  # after _sync — buffer · lengths · early ends hold until the next begin_batch
            self.objtrace.add_batch(
                self, self.buf.numpy(), self.length.cpu().numpy(), self.early.cpu().numpy(), not bool(park_invalid)
            )

        def finalize(self, aggregate):
            self.objtrace.check_rows(self)  # (1) before the trajectory file is written
            out = super().finalize(aggregate)  # [TRAJ_REC] judge check · atomic write · DONE
            self.objtrace.write(self, out)  # (2) judge counts (trajectory meta) → atomic write → DONE
            return out

    ObjTraceRecorder.__name__ = ObjTraceRecorder.__qualname__ = f"ObjTrace{base.__name__}"
    return ObjTraceRecorder


class ObjTrace:
    """The second record attached to the recorder — capture (step) · begin_batch · add_batch (batch end) ·
    check_rows
    (before writing) · write."""

    def __init__(self, rec, out_path: str):
        from dexengine.rl.eval.traj_record import ROOT_W

        self.path = os.path.abspath(out_path)
        if not self.path.endswith(".npz"):
            raise ValueError(f"{MARK} output must be a .npz: {self.path}")
        if self.path == rec.out_path:
            raise ValueError(
                f"{MARK} same path as the trajectory file {self.path} — the second file would overwrite it"
            )
        if rec.buf is not None or rec.rows:
            raise RuntimeError(
                f"{MARK} attached after a batch started (rows {len(rec.rows)}) — earlier episodes would be missing"
            )
        self.root_w = ROOT_W
        self.side_names = [s["side"] for s in rec.sides]
        self.entity_names = [s["entity"] for s in rec.sides]
        self.eps: list[tuple] = []  # per episode (clip, batch, env, warm, valid, success, length) — counted apart
        self.start: list[int] = []  # per episode: start_frames[env] at the batch reset
        self.obj_parts: list[np.ndarray] = []  # per batch [ΣL_b, S, 7] float32
        self.ref_parts: list[np.ndarray] = []
        self.rf_parts: list[np.ndarray] = []  # per batch [ΣL_b, S] int32 — the tree frame the reference was read from
        self.fbuf = None  # [T, NE, S] int32 — side_abs_frames per step (same time axis as the recorder buffer)
        self.rule_mismatch = 0  # frames where ref_frame ≠ the rule (start + floor(k/slow), clamped) — reported
        self._sf: np.ndarray | None = None
        self.head = ""
        self.phys: list[dict] = []  # per batch: measured object physics (phys_probe)
        # per episode, the termination terms that ended it (only with the object-tracking termination off — owner
        # judge.terms): the termination_manager._term_dones bits of its last step (term order = _term_names) →
        # npz ep_term_bits · meta.ep_term_names
        from dexengine.rl.judge import terms as judge_terms

        self.cause_on = judge_terms.objtrack_term()[0] == 0
        self.cause: list[int] = []
        self._cb = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        sd = ",".join(f"{a}:{b}" for a, b in zip(self.side_names, self.entity_names, strict=True))
        print(
            f"{MARK} on → {self.path} · sides {sd} · NE {rec.n_env} · recorder {rec.out_path} · "
            f"every episode (warm-up · PARK-invalid included, flagged) · float32",
            flush=True,
        )

    def begin_batch(self, rec) -> None:
        """Right after the batch reset — check the clip assignment, take the start frames (time origin)."""
        if self.cause_on:
            import torch

            self._cb = torch.full((rec.n_env,), -1, dtype=torch.long, device=rec.dev)
        tm = rec.traj_mgr
        sf, ci = getattr(tm, "start_frames", None), getattr(tm, "clip_ids", None)
        if sf is None or ci is None:
            raise RuntimeError(f"{MARK} trajectory_manager has no start_frames/clip_ids — the time origin is unknown")
        ci = ci[: rec.bs].detach().to("cpu").long().numpy()
        if not np.array_equal(ci, rec.assign[: rec.bs]):
            raise RuntimeError(
                f"{MARK} batch {rec.b}: traj_mgr.clip_ids ≠ the judge assignment — clip labels would be wrong"
            )
        self._sf = sf[: rec.bs].detach().to("cpu").long().numpy().copy()
        self._cl = tm.clip_lengths[: rec.bs].detach().to("cpu").long().numpy().copy()
        self.phys_probe(rec)

    def phys_probe(self, rec) -> None:
        """The scored batch's object physics as the simulator holds it right after the batch reset: per side over the
        batch's envs — mass ÷ default mass · static / dynamic friction and restitution of the object's shapes · COM
        spread across envs. The training's per-reset DR shows ratios ≠ 1 and a COM spread; the frozen eval's
        neutralisation (the runner's frozen-eval snapshot) shows 1.0 · one friction value · zero spread. One line per
        side per batch ([E123_PHYS]) + meta.phys."""
        import torch

        out = {"batch": int(rec.b)}
        for side, ent in zip(self.side_names, self.entity_names, strict=True):
            try:
                obj = rec.env.scene[ent]
                v, n = obj.root_physx_view, int(rec.bs)
                m = v.get_masses().detach().to("cpu").float()[:n]
                d = getattr(obj.data, "default_mass", None)
                r = m / d.detach().to("cpu").float()[:n] if isinstance(d, torch.Tensor) else None
                # [n, max_shapes, 3] μs · μd · e — max_shapes is the scene's maximum, so an object with fewer
                # shapes is zero-padded (dexengine.rl.tasks.mdp.events.randomization friction audit): real shapes =
                # friction ≠ 0
                mat = v.get_material_properties().detach().to("cpu").float()[:n]
                real = mat[..., :2].abs().sum(dim=-1) > 0
                if not bool(real.any()):
                    raise RuntimeError("no shape with a non-zero friction (all padding?)")
                # RigidObject (N, 7) · articulation (N, B, 7) — one shape for both
                # (dexengine.rl.tasks.mdp.events.randomization randomize COM)
                com = v.get_coms().detach().to("cpu").float()
                com = com.reshape(com.shape[0], -1, 7)[:n, :, :3]
                # COM spread WITHIN a clip: envs of one batch carry different clips (env i → clip of the plan) and so
                # different objects with different nominal COMs — only envs of the same clip share one nominal COM,
                # and per-reset COM DR is what would spread them (across clips the spread is the objects', not DR's)
                asg = getattr(rec, "assign", None)
                grp = torch.as_tensor(np.asarray(asg)[:n]) if asg is not None else torch.zeros(n, dtype=torch.long)
                spread = 0.0
                for g in torch.unique(grp):
                    cg = com[grp == g]
                    spread = max(spread, float((cg - cg.median(dim=0).values).norm(dim=-1).max()) * 1000.0)

                def mm(x):
                    return [round(float(x.min()), 5), round(float(x.max()), 5)]

                q = {
                    "mass": mm(m),
                    "mass_ratio": mm(r) if r is not None else None,
                    "mu_s": mm(mat[..., 0][real]),
                    "mu_d": mm(mat[..., 1][real]),
                    "rest": mm(mat[..., 2][real]),
                    "com_spread_mm": round(spread, 3),
                    "envs": n,
                    "clips": int(torch.unique(grp).numel()),
                }
                out[side] = q
                print(
                    f"[E123_PHYS] batch {rec.b} {side}({ent}): mass {q['mass']} kg · ÷ default {q['mass_ratio']} · "
                    f"mu_s {q['mu_s']} · mu_d {q['mu_d']} · rest {q['rest']} · "
                    f"COM spread in-clip {q['com_spread_mm']} mm · envs {n} · clips {q['clips']}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001, PERF203  the physics probe is a report line and never stops the eval; a test double without a PhysX view lands here and is printed, never silent
                why = f"{type(e).__name__}: {e}"[:200]
                out[side] = {"unavailable": why}
                print(f"[E123_PHYS] batch {rec.b} {side}({ent}): unavailable — {why}", flush=True)
        self.phys.append(out)

    def capture(self, rec, t: int) -> None:
        """After the recorder's _capture(t) — the tree frame this step's reference was read from (side_abs_frames per
        side).
        """
        import torch

        fn = getattr(rec.traj_mgr, "side_abs_frames", None)
        if fn is None:
            raise RuntimeError(
                f"{MARK} trajectory_manager.side_abs_frames is missing — the reference frame cannot be measured"
            )
        x = torch.stack([fn(s, rec.env.episode_length_buf) for s in self.side_names], dim=-1).to(torch.int32)
        n = rec.buf.shape[0]
        if self.fbuf is None or self.fbuf.shape[0] < n:
            nb = torch.empty((n, rec.n_env, len(self.side_names)), dtype=torch.int32, pin_memory=rec.pin)
            if self.fbuf is not None:
                rec._sync()
                nb[: self.fbuf.shape[0]].copy_(self.fbuf)
            self.fbuf = nb
        self.fbuf[t].copy_(x, non_blocking=rec.pin)

    def after_step(self, rec, terminated, truncated, done_mask) -> None:
        """After the recorder's after_step — the termination bits of the envs that ended this step (the recorder's
        rule:
        (term | trunc) & ~done_mask)."""
        if not self.cause_on:
            return
        import torch

        newly = (terminated.reshape(-1).bool() | truncated.reshape(-1).bool()) & ~done_mask
        if bool(newly.any()):
            d = rec.env.termination_manager._term_dones.long()
            bits = (d << torch.arange(d.shape[1], device=d.device)[None, :]).sum(dim=1)
            self._cb[newly] = bits[newly]

    def add_batch(self, rec, buf: np.ndarray, L: np.ndarray, early: np.ndarray, valid: bool) -> None:
        """After end_batch — the [0, L) object and reference columns of every env of the batch (warm-up and invalid
        included). The Python loop over envs builds the table rows only."""
        if self._sf is None:
            raise RuntimeError(f"{MARK} batch {rec.b}: end_batch without begin_batch — the start frames are unknown")
        cols = rec.sides_cols()
        if [c["side"] for c in cols] != self.side_names:
            raise RuntimeError(
                f"{MARK} sides {self.side_names} → {[c['side'] for c in cols]} — changed after attaching"
            )
        bs, w = rec.bs, self.root_w
        Lb = L[:bs].astype(np.int64)
        if int(Lb.min()) < 1:
            raise RuntimeError(f"{MARK} batch {rec.b}: an episode did not end (length {int(Lb.min())})")
        T = int(Lb.max())
        keep = np.arange(T)[None, :] < Lb[:, None]  # [bs, T] — env order · time order
        for key, parts in (("obj_lo", self.obj_parts), ("ref_lo", self.ref_parts)):
            x = np.stack([buf[:T, :bs, c[key] : c[key] + w] for c in cols], axis=2)  # [T, bs, S, 7]
            parts.append(np.ascontiguousarray(x.transpose(1, 0, 2, 3)[keep], dtype=np.float32))
        rf = self.fbuf[:T, :bs].numpy().transpose(1, 0, 2)  # [bs, T, S] — measured reference frames
        self.rf_parts.append(np.ascontiguousarray(rf[keep], dtype=np.int32))
        if not bool(
            getattr(rec.traj_mgr, "per_side_clock", False)
        ):  # rule check (reported — the measured frame is kept)
            slow = np.float32(getattr(rec.traj_mgr, "_slowdown_factor", 1.0))
            k = np.arange(T, dtype=np.float32)
            exp = np.floor(k / slow).astype(np.int64)[None, :] + self._sf[:, None]
            exp = np.clip(np.minimum(exp, (self._cl - 1)[:, None]), 0, None)
            self.rule_mismatch += int(((rf != exp[:, :, None]) & keep[:, :, None]).sum())
        for e in range(bs):
            self.eps.append(
                (
                    int(rec.assign[e]),
                    int(rec.b),
                    e,
                    int(bool(rec.warm)),
                    int(bool(valid)),
                    int(not bool(early[e])),
                    int(Lb[e]),
                )
            )
        self.start.extend(int(x) for x in self._sf)
        self._sf = None
        if self.cause_on:
            cb = self._cb[:bs].cpu().numpy()
            if int(cb.min()) < 0:
                raise RuntimeError(f"{MARK} batch {rec.b}: {int((cb < 0).sum())} episodes without termination bits")
            self.cause.extend(int(x) for x in cb)

    def counts(self, rec) -> dict:
        real = [r for r in self.eps if not r[3] and r[1] not in rec.probe_batches]  # the judge counts no probe batch
        return {
            "episodes": len(self.eps),
            "real": len(real),
            "real_success": sum(r[5] for r in real),
            "frames": int(sum(p.shape[0] for p in self.obj_parts)),
            "sum_len": int(sum(r[6] for r in self.eps)),
            "ref_frames": int(sum(p.shape[0] for p in self.ref_parts)),
            "rf_frames": int(sum(p.shape[0] for p in self.rf_parts)),
            "ref_frame_rule_mismatch": self.rule_mismatch,
        }

    def _die(self, why: str) -> None:
        print(f"{MARK} FAIL {self.head}", flush=True)
        raise RuntimeError(f"{MARK} {why} — stopped before writing the second file: {self.head}")

    def check_rows(self, rec) -> None:
        """(1) Before the trajectory file is written — compare with the recorder's rows and frame counts; on a mismatch
        print FAIL and stop (no file is written)."""
        c = self.counts(rec)
        rows = [tuple(int(v) for v in r[:7]) for r in rec.rows]
        ok_r = len(self.eps) == len(rows) and all(a == b for a, b in zip(self.eps, rows, strict=True))
        ok_f = c["frames"] == c["sum_len"] == c["ref_frames"] == c["rf_frames"] and len(self.start) == len(self.eps)

        def m(ok):
            return "✓" if ok else "✗"

        self.head = (
            f"episodes {c['episodes']} = recorder rows {len(rows)} (same values · same order) {m(ok_r)} · "
            f"frames {c['frames']} = Σ lengths {c['sum_len']} = reference {c['ref_frames']} = "
            f"reference frames {c['rf_frames']} {m(ok_f)} · "
            f"reference frame ≠ rule {c['ref_frame_rule_mismatch']} (reported)"
        )
        if not (ok_r and ok_f):
            self._die("episode rows / frame counts differ from the recorder")

    def _stacked_parts(self):
        """The recorded object, reference and reference-frame parts stacked, the episode table, each row offset."""
        S, w = len(self.side_names), self.root_w
        obj = np.concatenate(self.obj_parts, 0) if self.obj_parts else np.zeros((0, S, w), np.float32)
        ref = np.concatenate(self.ref_parts, 0) if self.ref_parts else np.zeros((0, S, w), np.float32)
        rfr = np.concatenate(self.rf_parts, 0) if self.rf_parts else np.zeros((0, S), np.int32)
        table = np.array(self.eps, dtype=np.int64).reshape(-1, len(EP_COLS))
        off = np.concatenate([[0], np.cumsum(table[:, 6])[:-1]]).astype(np.int64)
        return obj, ref, rfr, table, off

    @staticmethod
    def _clip_meta_checked(rec, table):
        """Per-clip values and object assets of the run; fail unless every clip value has one row per clip."""
        cm, cm_doc, missing = clip_meta(rec.env)
        oa, oa_bad = object_assets(rec)
        if oa_bad:
            missing["object_assets"] = f"sides {oa_bad}: the scene asset cfg.spawn has no usd_path (this run)"
        for (
            o
        ) in oa.values():  # one USD per clip (n = K): clip c's object = usds[c] — checked against env % n (reported)
            if len(o["usds"]) == len(rec.clip_names) and not o["random_choice"]:
                o["clip_mismatch"] = int((table[:, 0] != table[:, 2] % len(o["usds"])).sum())
        K = len(rec.clip_names)
        for k, v in cm.items():
            if v.shape[0] != K:
                raise RuntimeError(f"{MARK} clip value {k} has {v.shape[0]} rows ≠ {K} clips")
        return cm, cm_doc, missing, oa

    def _termination_causes(self, rec, table, meta):
        """With the cause recorder on: per-episode termination bits (into meta and the extras) and a count line."""
        extra = {}
        if self.cause_on:
            names = list(rec.env.termination_manager._term_names)
            if len(self.cause) != table.shape[0]:
                raise RuntimeError(f"{MARK} termination bits {len(self.cause)} ≠ episodes {table.shape[0]}")
            meta["ep_term_names"] = names
            meta["ep_term_rule"] = (
                "bit i of ep_term_bits[e] = termination_manager._term_dones[:, i] at the step episode e ended"
            )
            extra["ep_term_bits"] = np.array(self.cause, np.int64)
            real = (table[:, 3] == 0) & (table[:, 4] == 1)
            cnt = {n: int(((extra["ep_term_bits"][real] >> i) & 1).sum()) for i, n in enumerate(names)}
            none_ = int((extra["ep_term_bits"][real] == 0).sum())
            print(
                f"{MARK} termination causes ({int(real.sum())} real-batch episodes · per term · several "
                "per episode possible): " + " ".join(f"{n}={v}" for n, v in cnt.items()) + f" no_bit={none_}",
                flush=True,
            )
        return extra

    def _save(self, rec, meta, table, off, obj, ref, rfr, cm, extra):
        """The object-trace npz, written atomically next to the trajectory file."""
        org = rec.scene.env_origins.detach().to("cpu").float().numpy()
        tmp = self.path[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(
            tmp,
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
            clip_names=np.array(rec.clip_names),
            ep_table=table,
            ep_offset=off,
            ep_start_frame=np.array(self.start, np.int64),
            side_names=np.array(self.side_names),
            entity_names=np.array(self.entity_names),
            obj=obj,
            ref=ref,
            ref_frame=rfr,
            env_origins=org,
            **{f"clip_{k}": v for k, v in cm.items()},
            **extra,
        )
        os.replace(tmp, self.path)

    def write(self, rec, traj_path: str) -> str:
        """(2) After the trajectory file is written — compare with the judge counts (trajectory meta.judge) → atomic
        write (tmp + os.replace) → DONE marker."""
        from dexengine.core.hashing import sha256_file

        tm = json.loads(str(np.load(traj_path, allow_pickle=False)["meta"]))
        c, judge = self.counts(rec), tm["judge"]
        ok_e, ok_s = c["real"] == int(judge["num_episodes"]), c["real_success"] == int(judge["completed"])

        def m(ok):
            return "✓" if ok else "✗"

        self.head = (
            f"real-batch episodes {c['real']} = judge {judge['num_episodes']} {m(ok_e)} · "
            f"successes {c['real_success']} = "
            f"judge completed {judge['completed']} {m(ok_s)} · {self.head}"
        )
        if not (ok_e and ok_s):
            self._die("differs from the judge counts")
        obj, ref, rfr, table, off = self._stacked_parts()
        cm, cm_doc, missing, oa = self._clip_meta_checked(rec, table)
        meta = dict(
            schema=SCHEMA,
            written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **{k: tm[k] for k in FROM_TRAJ_META},
            recorded=c,
            traj_record={"path": traj_path, "sha256": sha256_file(traj_path), "rows": "ep_table equal row by row"},
            ep_cols=list(EP_COLS),
            obj_rule="scene.rigid_objects[entity].data.root_link_pose_w − scene.env_origins (position only) · "
            "quat wxyz",
            ref_rule="trajectory_manager.get_reference_state(episode_length_buf)[{side}_obj_pos|_obj_quat] — "
            "what the tracking termination compares with obj at the same step",
            globals=global_meta(rec.env),
            object_assets=oa,
            clip_meta_fields=cm_doc,
            phys=self.phys,
            missing_for_inverse_chain=missing,
        )
        extra = self._termination_causes(rec, table, meta)
        self._save(rec, meta, table, off, obj, ref, rfr, cm, extra)
        sha = sha256_file(self.path)[:16]
        print(
            f"{MARK} DONE episodes {c['episodes']} (real-batch {c['real']} · successes {c['real_success']}) · "
            f"frames {c['frames']} · sides {','.join(self.side_names)} · {self.head} · file {self.path} "
            f"sha256={sha} · {os.path.getsize(self.path) / 1e6:.1f} MB",
            flush=True,
        )
        return self.path
