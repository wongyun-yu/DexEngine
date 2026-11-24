"""Trajectory recording: the entity rule, the recorder and the replay write-back (replay render; off by default).

The file format (schema version, reader) belongs to ``traj_file``.

A read-only vectorised recorder rides the judge loop (``judge.rollouts`` single-batch loop) and writes, per clip, the
top-k successful episodes to a file. The replay render (``dexengine/rl/render/replay.py``) writes those states back
without physics and renders them.

State rule: every articulation of the scene (root link pose 7 = env-local position + quaternion wxyz, plus all
  joint_pos) and every rigid object (pose 7). Any other dynamic asset type fails. Recording every articulation keeps a
  separate left arm (robot_left) in the file.
Reference object pose (for ranking): sides = env.held_object_assets() ∩ sides with an object-tracking termination term
  (ObjectTrackingExceeded); per side the pose 7 of traj_mgr.get_reference_state(episode_length_buf), the value that
  termination term reads. Zero sides fails.
Time: s_0 = right after the batch reset, s_k = after step k. An env that ends at step L has its post-step state (already
  reset) dropped, so its valid range is [0, L).
Selection (owner ``traj_select``): success ∧ real batch (not warm-up) ∧ PARK valid → score ascending, top k. The
  recorder never writes to the scene.
failure_best: a clip with valid episodes but no success keeps its best failure separately (rank
  traj_select.failure_key) — archived only; the replay render draws ``sel`` only.
Log lines: `[TRAJ_REC] on → …` · `batch …` · `DONE …` (or `FAIL …` before writing when the counts disagree).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from dexengine.core.hashing import sha256_file
from dexengine.core.run_identity import tree_identity
from dexengine.rl.eval import traj_select
from dexengine.rl.eval.traj_file import SCHEMA_VERSION  # the file format belongs to traj_file
from dexengine.rl.eval.traj_meta import _extra_slots_meta

ENV_VAR = "DEXENGINE_TRAJ_RECORD"  # value = output file path (.npz); empty = no recording
K_KEEP = 2  # successful trajectories kept per clip (selected + spare)
ROOT_W = 7  # root link pose columns (pos 3 + quat wxyz 4)
HIDDEN_SLOT_MARGIN_M = 1.0  # empty-slot test margin: up to 1 m above the hidden z of the scene builder (below floor)


# Entity rule (scene -> column layout): recording and replay use the same functions.
def all_obj_body(name: str) -> bool:
    """Whether ``name`` is a static all-object-collision body — not a recorded or replayed entity.

    Those bodies are invisible and never move; their pose is set by the visual prim. With the all-object-collision
    switch off this is always False. The switch and the name prefix belong to ``collision_presets`` (stdlib only,
    imported lazily so the recorder stays importable without Isaac).
    """
    from dexengine.rl.tasks.scene import collision_presets as cp

    return cp.all_obj_collision() and name.startswith(cp.ALL_OBJ_KEY)


def scene_entities(scene) -> list[dict]:
    """Column layout of every dynamic asset of the scene: articulations first (by name), then rigid objects (by
    name). Dies on a dynamic asset type it cannot record."""
    bad = {k: list(getattr(scene, k, {}) or {}) for k in ("rigid_object_collections", "deformable_objects")}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise RuntimeError(
            f"[TRAJ_REC] dynamic assets without a record rule {bad} — only articulations and rigid bodies are handled"
        )
    ents, lo = [], 0
    for kind, table in (("artic", scene.articulations), ("rigid", scene.rigid_objects)):
        for name, alias in _unique_assets(scene, table):
            if kind == "rigid" and all_obj_body(name):
                continue
            jn = list(table[name].joint_names) if kind == "artic" else []
            ents.append(
                {"name": name, "kind": kind, "joint_names": jn, "aliases": alias, "lo": lo, "hi": lo + ROOT_W + len(jn)}
            )
            lo += ROOT_W + len(jn)
    if not ents:
        raise RuntimeError("[TRAJ_REC] the scene has no articulation or rigid body")
    return ents


def _unique_assets(scene, table: dict) -> list[tuple[str, list[str]]]:
    """One entry per asset object even when it is registered under several names (e.g. the alias
    scene._articulations["right_hand"] = robot). The head name is the one that is a scene cfg field (else the first
    by name); the others are kept as aliases."""
    groups: dict[int, list[str]] = {}
    for name in sorted(table):
        groups.setdefault(id(table[name]), []).append(name)
    out = []
    for names in groups.values():
        cfg_names = [n for n in names if hasattr(getattr(scene, "cfg", None), n)]
        head = (cfg_names or names)[0]
        out.append((head, [n for n in names if n != head]))
    return sorted(out)


def state_vector(scene, ents: list[dict]) -> torch.Tensor:
    """[NE, D] float32 in the entity column layout (positions env-local)."""
    org = scene.env_origins
    parts = []
    for e in ents:
        a = scene.articulations[e["name"]] if e["kind"] == "artic" else scene.rigid_objects[e["name"]]
        pose = a.data.root_link_pose_w
        parts.append(pose[:, :3] - org)
        parts.append(pose[:, 3:7])
        if e["kind"] == "artic":
            parts.append(a.data.joint_pos)
    return torch.cat(parts, dim=-1).float()


@dataclass
class Binding:
    """File entity ↔ scene asset (replay). joint_cols = the file columns to read in the scene's joint order (matched
    by name)."""

    name: str
    kind: str
    asset: object
    root_cols: slice
    joint_cols: torch.Tensor | None


def bind(scene, file_ents: list[dict], device, trajs=None, hidden_z: float | None = None) -> list[Binding]:
    """Bind file entities to scene entities; every articulation must have the same joint-name set (else dies — this
    also catches a physics ↔ render USD difference).

    The scene is built from the session's clips, so a slot no session clip uses (a support disc, for example) is not
    in the scene. An entity present only in the file is therefore skipped only when, in every session trajectory
    (trajs, list of [T, D]), its root z stays ≤ hidden_z + HIDDEN_SLOT_MARGIN_M (the scene builder's empty-slot
    position below the floor) — a rule, not a name list. An entity present only in the scene always dies (there is
    no state to place)."""
    have = {e["name"]: e for e in scene_entities(scene)}
    want = {e["name"]: e for e in file_ents}
    only_f, only_s = sorted(set(want) - set(have)), sorted(set(have) - set(want))
    up = [
        n
        for n in only_f
        if not trajs
        or hidden_z is None
        or any(float(t[:, want[n]["lo"] + 2].max()) > hidden_z + HIDDEN_SLOT_MARGIN_M for t in trajs)
    ]
    if only_s or up:
        raise RuntimeError(
            f"[REPLAY_MAP] entity sets differ — file only {only_f} (of them not at the empty-slot place "
            f"{up}) · scene only {only_s}"
        )
    if only_f:
        print(
            f"[REPLAY_MAP] entities only in the file {only_f} — every frame of the {len(trajs)} session trajectories "
            f"has z ≤ {hidden_z}+{HIDDEN_SLOT_MARGIN_M} m (empty-slot place) → not used",
            flush=True,
        )
    out = []
    for name, fe in want.items():
        if name in only_f:
            continue
        if fe["kind"] != have[name]["kind"]:
            raise RuntimeError(f"[REPLAY_MAP] {name} kind {fe['kind']} ≠ scene {have[name]['kind']}")
        asset = scene.articulations[name] if fe["kind"] == "artic" else scene.rigid_objects[name]
        jc = None
        if fe["kind"] == "artic":
            fj, sj = list(fe["joint_names"]), list(asset.joint_names)
            if set(fj) != set(sj) or len(fj) != len(sj):
                raise RuntimeError(
                    f"[REPLAY_MAP] {name} joint name sets differ — "
                    f"file only {sorted(set(fj) - set(sj))} · "
                    f"scene only {sorted(set(sj) - set(fj))}"
                )
            col = {n: fe["lo"] + ROOT_W + i for i, n in enumerate(fj)}
            jc = torch.tensor([col[n] for n in sj], dtype=torch.long, device=device)
        out.append(Binding(name, fe["kind"], asset, slice(fe["lo"], fe["lo"] + ROOT_W), jc))
    return out


def write_states(scene, binds: list[Binding], states: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Replay write: states [n, D] (file column layout) into env_ids. Articulation: root link pose, then joint
    values (zero velocity). Rigid object: root link pose. The recorder never calls this (recording only reads)."""
    org = scene.env_origins[env_ids]
    for b in binds:
        root = states[:, b.root_cols]
        pose = torch.cat([root[:, :3] + org, root[:, 3:7]], dim=-1)
        b.asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        if b.kind == "artic":
            q = states[:, b.joint_cols]
            b.asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


# Recorder
class TrajRecorder:
    """Recorder with the three hooks of the judge's single-batch loop (begin_batch · after_step · end_batch) and
    finalize. Per step: one GPU cat and one non_blocking copy into a pinned CPU buffer — no Python loop over envs."""

    RANK_BY = traj_select.RANK_SCORE  # candidate rank rule (traj_select.rank_key); subclasses change it

    def __init__(
        self,
        raw_env,
        clip_names: list[str],
        out_path: str,
        k_keep: int = K_KEEP,
        meta: dict | None = None,
        term_cls=None,
    ):
        self.env = raw_env
        self.scene = raw_env.scene
        self.traj_mgr = raw_env.trajectory_manager
        self.clip_names = list(clip_names)
        self.out_path = os.path.abspath(out_path)
        if not self.out_path.endswith(".npz"):
            raise ValueError(f"[TRAJ_REC] the output must be .npz: {self.out_path}")
        self.k_keep = int(k_keep)
        self.ents = scene_entities(self.scene)
        self.sides = traj_select.tracked_sides(raw_env, self.ents, term_cls)  # side set (tolerances: first batch)
        self.term_cls, self.tol_live = term_cls, False
        self.d_state = self.ents[-1]["hi"]
        self.ref_lo = self.d_state
        self.D = self.d_state + ROOT_W * len(self.sides)
        self.warmup_batches: int | None = None  # set by the judge entry; begin_batch dies while unknown
        self.n_env = int(raw_env.num_envs)
        self.dev = torch.device(raw_env.device)
        self.pin = self.dev.type == "cuda"  # pinned memory + non_blocking copy only on CUDA
        self.buf: torch.Tensor | None = None
        self.rows: list[tuple] = []  # (clip, batch, env, warmup, valid, success, length, score)
        self.kept: dict[int, list] = {}  # clip → [(rank key, score, row_idx, states[L, D])] ascending, ≤ k_keep
        # clip → (failure_key, score, row_idx, states): the best real-batch, PARK-valid failure (for no-success clips)
        self.fail_best: dict[int, tuple] = {}
        self.meta_in = dict(meta or {})
        self.t0 = time.time()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.tree = tree_identity()
        ent_txt = " ".join(
            f"{e['name']}({'J=' + str(len(e['joint_names'])) if e['kind'] == 'artic' else 'rigid'})" for e in self.ents
        )
        side_txt = ",".join(
            f"{s['side']}:{s['entity']}(tol {s['pos_tol'] * 100:.1f}cm/{np.rad2deg(s['rot_tol']):.0f}°)"
            for s in self.sides
        )
        print(
            f"[TRAJ_REC] on → {self.out_path} · entities {ent_txt} · D={self.D}(state {self.d_state} + reference "
            f"{self.D - self.d_state}) · sides={side_txt} · k_keep={self.k_keep} · NE={self.n_env} · "
            f"module={self.tree['module']} · tree={self.tree['tree']}@{self.tree['commit'][:12]}"
            f"({self.tree['branch']}{', dirty' if self.tree['dirty'] else ''})",
            flush=True,
        )

    # One row [NE, D]: state + reference poses.
    def _row(self) -> torch.Tensor:
        s = state_vector(self.scene, self.ents)
        ref = self.traj_mgr.get_reference_state(self.env.episode_length_buf)
        refs = [
            torch.cat([ref[f"{sd['side']}_obj_pos"], ref[f"{sd['side']}_obj_quat"]], dim=-1).float()
            for sd in self.sides
        ]
        return torch.cat([s, *refs], dim=-1)

    def _capture(self, t: int) -> None:
        if t >= self.buf.shape[0]:  # rare: an env outlived the clip length — grow the buffer
            nb = torch.empty((int(t * 1.5) + 1, self.n_env, self.D), dtype=torch.float32, pin_memory=self.pin)
            self._sync()
            nb[: self.buf.shape[0]].copy_(self.buf)
            self.buf = nb
        self.buf[t].copy_(self._row(), non_blocking=self.pin)

    def _sync(self) -> None:
        if self.pin:
            torch.cuda.synchronize(self.dev)

    def begin_batch(self, batch_idx: int, clip_assignment: torch.Tensor, batch_size: int) -> None:
        """Right after the batch reset (s_0) — called after the judge loop's last reset()."""
        if self.warmup_batches is None:
            raise RuntimeError(
                "[TRAJ_REC] the warm-up batch count is unknown — the batches the judge drops cannot be "
                "left out of the candidates"
            )
        self.b, self.bs = int(batch_idx), int(batch_size)
        if not self.tol_live:  # tolerances as at judge time (after the frozen changes), not the start values
            live = traj_select.tracked_sides(self.env, self.ents, self.term_cls)
            if [(s["side"], s["entity"]) for s in live] != [(s["side"], s["entity"]) for s in self.sides]:
                raise RuntimeError(f"[TRAJ_REC] the side set changed after start-up {self.sides} → {live}")
            self.sides, self.tol_live = live, True
            print(
                "[TRAJ_REC] thresholds (at judge time) "
                + " · ".join(
                    f"{s['side']} {s['pos_tol'] * 100:.1f}cm/{np.rad2deg(s['rot_tol']):.0f}° progress "
                    f"{s['progress']:.2f}"
                    for s in self.sides
                ),
                flush=True,
            )
        self.warm = self.b < int(self.warmup_batches)
        self.assign = clip_assignment.detach().to("cpu").long().numpy().copy()
        t_cap = int(self.traj_mgr.clip_lengths[: self.bs].max().item()) + 2
        if self.buf is None or self.buf.shape[0] < t_cap:
            self.buf = torch.empty((t_cap, self.n_env, self.D), dtype=torch.float32, pin_memory=self.pin)
        self.length = torch.full((self.n_env,), -1, dtype=torch.long, device=self.dev)
        self.early = torch.zeros(self.n_env, dtype=torch.bool, device=self.dev)
        self.t = 0
        self._capture(0)

    def after_step(self, step: int, terminated: torch.Tensor, truncated: torch.Tensor, done_mask: torch.Tensor) -> None:
        """After env.step — done_mask = envs that ended BEFORE this step (the judge loop's mask before its update). An
        env that ends now has a post-step state that is already reset, so it is excluded (length = step)."""
        self.t = int(step)
        self._capture(self.t)
        term, trunc = terminated.reshape(-1).bool(), truncated.reshape(-1).bool()
        newly = (term | trunc) & ~done_mask
        if bool(newly.any()):
            self.length[newly] = self.t
            self.early[newly] = (term & ~trunc)[newly]

    def end_batch(self, park_invalid: bool = False) -> None:
        self._sync()
        L = self.length.cpu().numpy()
        early = self.early.cpu().numpy()
        valid = not bool(park_invalid)
        buf = self.buf.numpy()
        n_succ = n_kept = 0
        sides = self.sides_cols()
        for e in range(self.bs):
            if L[e] < 1:
                raise RuntimeError(
                    f"[TRAJ_REC] batch {self.b} env {e}: unfinished episode (length {L[e]}) — "
                    "out of step with the judge loop"
                )
            cid, succ = int(self.assign[e]), not bool(early[e])
            st = buf[: L[e], e, :]
            sc = traj_select.score(st, sides, tail_frac=traj_select.TAIL_FRAC)
            self.rows.append((cid, self.b, e, self.warm, valid, succ, int(L[e]), sc))
            n_succ += succ
            if succ and valid and not self.warm:
                lst = self.kept.setdefault(cid, [])
                key = traj_select.rank_key(self.RANK_BY, st, sides, sc)
                if len(lst) < self.k_keep or key < lst[-1][0]:
                    lst.append((key, sc, len(self.rows) - 1, st.copy()))
                    lst.sort(key=lambda r: r[0])
                    del lst[self.k_keep :]
                    n_kept += 1
            elif valid and not self.warm:
                fk = traj_select.failure_key(int(L[e]), sc)
                if cid not in self.fail_best or fk < self.fail_best[cid][0]:
                    self.fail_best[cid] = (fk, sc, len(self.rows) - 1, st.copy())
        tag = (" (warm-up — not a candidate)" if self.warm else "") + (
            "" if valid else " (PARK invalid — not a candidate)"
        )
        print(
            f"[TRAJ_REC] batch {self.b}{tag}: "
            f"episodes {self.bs} · successes {n_succ} · candidates updated {n_kept} · steps {self.t} · "
            f"buffer {self.buf.shape[0]}×{self.n_env}×{self.D}",
            flush=True,
        )

    def sides_cols(self) -> list[dict]:
        """Input of traj_select.score: per side the object root column, the reference column and the tolerances."""
        ent = {e["name"]: e for e in self.ents}
        return [
            {
                "side": s["side"],
                "obj_lo": ent[s["entity"]]["lo"],
                "ref_lo": self.ref_lo + ROOT_W * i,
                "pos_tol": s["pos_tol"],
                "rot_tol": s["rot_tol"],
            }
            for i, s in enumerate(self.sides)
        ]

    def finalize(self, aggregate: dict) -> str:
        """Check the counts against the judge's aggregate, write atomically, print DONE. Dies when a count differs."""
        fb, fb_off, head, kept, lens, meta, offs, ok_c, ok_e, ok_f, ok_s, sel, states = self._pack_trajectories(
            aggregate
        )
        if not (ok_e and ok_s and ok_c and ok_f):  # die before writing: no wrong file is left at the output path
            print(f"[TRAJ_REC] FAIL {head}", flush=True)
            raise RuntimeError(
                f"[TRAJ_REC] the count equation broke — the record counted other samples than the judge: {head}"
            )
        rows = np.array([r[:7] for r in self.rows], dtype=np.int64).reshape(-1, 7)
        tmp = self.out_path[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(
            tmp,
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
            entities=np.array(json.dumps(self.ents, ensure_ascii=False)),
            clip_names=np.array(self.clip_names),
            ep_table=rows,
            ep_score=np.array([r[7] for r in self.rows], dtype=np.float64),
            states=(np.concatenate(states, 0) if states else np.zeros((0, self.D), np.float32)),
            kept_offset=np.array(offs, np.int64),
            kept_length=np.array(lens, np.int64),
            kept_clip=np.array([k[0] for k in kept], np.int64),
            kept_rank=np.array([k[1] for k in kept], np.int64),
            kept_score=np.array([k[2] for k in kept], np.float64),
            kept_row=np.array([k[3] for k in kept], np.int64),
            sel=sel,
            fb_clip=np.array([c for c, _ in fb], np.int64),
            fb_offset=np.array(fb_off, np.int64),
            fb_length=np.array([x[3].shape[0] for _c, x in fb], np.int64),
            fb_score=np.array([x[1] for _c, x in fb], np.float64),
            fb_row=np.array([x[2] for _c, x in fb], np.int64),
        )
        os.replace(tmp, self.out_path)
        sha = sha256_file(self.out_path)[:16]
        print(
            f"[TRAJ_REC] DONE {head} · file {self.out_path} sha256={sha} · "
            f"{os.path.getsize(self.out_path) / 1e6:.1f} MB",
            flush=True,
        )
        return self.out_path

    def _pack_trajectories(self, aggregate):
        """Pack the kept / selected / failure-best trajectories and the per-clip table for the trajectory file."""
        real = [r for r in self.rows if not r[3]]
        e_rec, s_rec = len(real), sum(1 for r in real if r[5])
        per = aggregate.get("per_clip") or []
        e_j = int(aggregate.get("num_episodes", -1))
        s_j = int(sum(int(c.get("completed", 0)) for c in per))
        ok_e, ok_s = e_rec == e_j, s_rec == s_j
        K = len(self.clip_names)
        kept = [(c, i, sc, ri, st) for c in sorted(self.kept) for i, (_k, sc, ri, st) in enumerate(self.kept[c])]
        sel = np.full(K, -1, dtype=np.int64)
        offs, lens, states = [], [], []
        o = 0
        for j, (c, i, _sc, _ri, st) in enumerate(kept):
            if i == 0:
                sel[c] = j
            offs.append(o)
            lens.append(st.shape[0])
            states.append(st)
            o += st.shape[0]
        n_sel = int((sel >= 0).sum())
        clip_succ = {r[0] for r in real if r[5] and r[4]}  # clips with a candidate (success ∧ real ∧ PARK valid)
        clip_valid = {r[0] for r in real if r[4]}
        fb = [(c, self.fail_best[c]) for c in sorted(self.fail_best) if sel[c] < 0]  # no-success clips only
        fb_off = []
        for _c, (_fk, _sc, _ri, st) in fb:
            fb_off.append(o)
            states.append(st)
            o += st.shape[0]
        meta = dict(
            schema_version=SCHEMA_VERSION,
            written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            tree=self.tree,
            num_envs=self.n_env,
            step_dt=float(self.env.step_dt),
            physics_dt=float(self.env.physics_dt),
            decimation=int(self.env.cfg.decimation),
            frame="env-local position · quaternion wxyz · root link pose · joint_pos(radian/metre)",
            time="s_0 = after batch reset · s_k = after step k · valid [0, L)",
            k_keep=self.k_keep,
            warmup_batches=int(self.warmup_batches),
            sides=[dict(s, rot_tol_deg=float(np.rad2deg(s["rot_tol"]))) for s in self.sides],
            score={"owner": "dexengine.rl.eval.traj_select", "tail_frac": traj_select.TAIL_FRAC},
            **(
                {}
                if self.RANK_BY == traj_select.RANK_SCORE
                else {"rank": {"by": self.RANK_BY, "rule": traj_select.RANK_RULE[self.RANK_BY]}}
            ),
            judge={"num_episodes": e_j, "completed": s_j, "completion_rate": aggregate.get("completion_rate")},
            recorded={"real_episodes": e_rec, "real_success": s_rec, "all_rows": len(self.rows)},
            clips_with_success=len(clip_succ),
            clips_selected=n_sel,
            env=self.meta_in,
            **_extra_slots_meta(self.env),
            failure_best={
                "clips": len(fb),
                "rule": "traj_select.failure_key(longest first → lowest score)",
                "use": "archive only — not a canonical video",
            },
            elapsed_s=round(time.time() - self.t0, 1),
        )
        mark = lambda ok: "✓" if ok else "✗"  # noqa: E731
        ok_c = n_sel == len(clip_succ)
        ok_f = {c for c, _ in fb} == clip_valid - clip_succ  # one failure_best per no-success clip (set check)
        head = (
            f"real-batch episodes {e_rec} = judge {e_j} {mark(ok_e)} · "
            f"success {s_rec} = judge completed {s_j} {mark(ok_s)} · "
            f"selected clips {n_sel}/{K} = clips with a candidate {len(clip_succ)} {mark(ok_c)} · "
            f"failure_best {len(fb)} = clips with no success {len(clip_valid - clip_succ)} {mark(ok_f)} "
            f"(archive only) · "
            f"kept trajectories {len(kept)} · frames {o}"
        )
        return fb, fb_off, head, kept, lens, meta, offs, ok_c, ok_e, ok_f, ok_s, sel, states
