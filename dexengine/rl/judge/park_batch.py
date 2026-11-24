"""PARK per-batch state: the hooks `_run_single_batch` calls in the judge loop (rules and values: `park`)."""

from __future__ import annotations

import torch

from dexengine.rl.judge.park import (
    AIR_MIN,
    D_AFTER_HOLD,
    DROP_TOL_MM,
    DZ_LIMIT_MM,
    ENV_VAR,
    EXCL_A_MAX_FRAC,
    FF_TOL,
    PARK_Q,
    RESET_WRIST_CUT,
    RESET_WRIST_MM,
    REST_GAP_MM,
    TOUCH_F_MIN,
    air_drop,
    artic_asset,
    d_steps,
    enabled,
    excl_a_mask,
    fall_accum_mask,
    free_fall_mask,
    gravity_reach_mask,
    park_verdict,
    pin_drop_mm,
    pin_fall_mask,
    probe_steps,
)


class Park:
    """PARK state of one judge batch, called at three points of the `_run_single_batch` loop (before_step,
    after_step, at_cut). With PARK off every method returns at once (no tensor op)."""

    def __init__(
        self,
        raw_env,
        batch_idx: int,
        batch_size: int,
        clip_assignment: torch.Tensor | None = None,
        clip_names: list | None = None,
    ):
        self.on = enabled()
        self.probe = self.on and bool(probe_steps())
        self.w0, self.step = None, 0  # probe: reset wrist per action term · last step
        held = (getattr(getattr(raw_env, "cfg", None), "start_held_hands", None) or []) if self.probe else []
        if self.probe and (not held or clip_assignment is None or int(clip_assignment.max()) >= len(held)):
            raise RuntimeError(
                "[PARK] probe on, but cfg.start_held_hands (hands that start holding their own object; "
                "written by dexengine.rl.env_setup.build.build_start_held) is missing or the clip "
                "assignment is outside it"
            )
        self.held = {s: torch.tensor([bool(c[s]) for c in held], dtype=torch.bool) for s in ("right", "left")}
        self.env, self.b, self.n = raw_env, batch_idx, batch_size
        self.assign = clip_assignment
        self.parked_max = 0
        self.invalid = False  # cut verdict (gates C, D) by at_cut; read by traj_record, never acted on
        self.force: dict = {}  # side → max hand-link <-> that side's object force at the cut, per env
        self.terms: list | None = None  # action terms that take the park mask (found at the first park)
        self.sinks: list | None = None  # termination terms that take the park mask (articulation θ skips parked envs)
        self.pushed: torch.Tensor | None = None  # the last park mask sent
        self.e_rows: dict = {}  # (clip, side) → (E samples k, samples n) at this cut (for park_e_oracle)
        self.d_cnt: dict = {}  # side letter → gate D count (G1 → G2 after the hold window)
        self.d_prev: dict = {}  # side → (G1 velocity, G1 hand touch, on a support)
        self.d_at = d_steps() if self.on else (0, 0)
        self.names = clip_names
        self.excl_a: dict = {}  # (clip, side) → median |dz| mm, gate-A-excluded in this batch
        miss = [
            m
            for m in ("pin_objects_to_reference", "object_pin_pose", "hand_object_contact_force")
            if not callable(getattr(raw_env, m, None))
        ]
        if self.on and miss:
            raise RuntimeError(
                f"[EVAL_PARK] {ENV_VAR}=1 but this env ({type(raw_env).__name__}) lacks "
                f"{' · '.join(miss)} — parking cannot be applied or measured (stopped here, not "
                "mid-batch)"
            )
        if self.on and not all(
            callable(getattr(raw_env, m, None))
            for m in ("object_support_gap", "hand_object_contact_force", "reference_frames")
        ):
            raise RuntimeError(
                f"[EVAL_GHOST] gate D is on but this env ({type(raw_env).__name__}) lacks one of "
                "object_support_gap · hand_object_contact_force · reference_frames — "
                "'on a support · no hand touch' cannot be measured"
            )

    def _clip_of(self, ids: torch.Tensor) -> torch.Tensor:
        """env → clip index (idle envs beyond the batch = clip 0, the judge loop's full_assignment convention; −1
        without an assignment)."""
        n_env = int(self.env.num_envs)
        full = torch.full((n_env,), -1, dtype=torch.long)
        if self.assign is not None:
            full[:] = 0
            full[: self.assign.numel()] = self.assign.detach().to("cpu").long()
        return full.to(ids.device)[ids]

    def _push(self, mask: torch.Tensor | None) -> None:
        """Send the park mask to the action terms and the termination terms (`set_parked`: the articulation θ term)
        (None = nothing parked). Dies when envs park and no action term takes the mask. The termination terms are kept
        apart because the [PARK] wrist check does not cover them."""
        if self.sinks is None:
            tm = getattr(self.env, "termination_manager", None)
            self.sinks = [
                c.func
                for c in getattr(tm, "_class_term_cfgs", None) or ()
                if callable(getattr(c.func, "set_parked", None))
            ]
        for s in self.sinks:
            s.set_parked(mask)
        if self.terms is None and mask is not None:
            terms = getattr(getattr(self.env, "action_manager", None), "_terms", None) or {}
            self.terms = [t for t in terms.values() if callable(getattr(t, "set_parked", None))]
            if not self.terms:
                raise RuntimeError(
                    "[PARK] envs are parked but no action term takes the park mask (set_parked) — "
                    "the parked wrist target cannot be set to frame 0"
                )
        self.pushed = mask
        for t in self.terms or ():
            t.set_parked(mask)

    def before_step(self, done_mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.on:
            return actions
        ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        self._push(done_mask.clone() if ids.numel() else None)
        # probe: every env parks from the reset (ids = all envs); the first call is right after the reset
        if self.probe and self.w0 is None:
            self.w0 = [t.parked_wrist_offset(ids).float() for t in self.terms]
        if ids.numel() == 0:
            return actions
        self.parked_max = max(self.parked_max, int(ids.numel()))
        actions = actions.clone()
        actions[ids] = 0.0
        self.env.episode_length_buf[ids] = 0
        self.env.pin_objects_to_reference(ids, joints=False)
        return actions

    def after_step(self, step: int) -> None:
        self.step = step
        if self.on and step in self.d_at:
            self._ghost(step)

    def _touch(self, ids: torch.Tensor) -> dict:
        """E at the cut — side → (ids,) bool: hand links (both hands) of envs parked on the last step touching that
        side's object. Dies when the hand-link measurement reads 0 for a parked env whose fingertip sensor touches the
        object (one measurement checked against the other). One [EVAL_PARK_CUT] line per cut."""
        n_env = int(self.env.num_envs)
        parked = torch.zeros(n_env, dtype=torch.bool) if self.pushed is None else self.pushed
        e, parts, tips, links = {}, [], [], {}
        for side, _obj in self.env.held_object_assets():
            c = self.env.hand_object_contact_force(side)
            own = c.force[side].float()
            oth = c.force.get("left" if side == "right" else "right")
            oth = torch.zeros_like(own) if oth is None else oth.float()
            pk = parked.to(own.device)
            t_own, t_oth = (own > TOUCH_F_MIN) & pk, (oth > TOUCH_F_MIN) & pk
            if c.tip is not None:
                bad = c.tip & pk & ~t_own
                if bool(bad.any()):
                    raise RuntimeError(
                        f"[EVAL_PARK_CUT] batch {self.b} {side}: the hand-link measurement reads 0 for "
                        f"{int(bad.sum())} parked envs whose fingertip sensor touches the object — the filtered "
                        "contact measurement is dead"
                    )
            e[side] = (t_own | t_oth)[ids.to(own.device)]
            self.force[side] = torch.where(pk, torch.maximum(own, oth), torch.zeros_like(own))
            s = side[0].upper()
            parts.append(
                f"{s} {int((t_own | t_oth).sum())}/{n_env}(parked env {int(pk.sum())} · "
                f"own hand {int(t_own.sum())} · other hand {int(t_oth.sum())} · "
                f"max force {float(self.force[side].max()):.2f} N)"
            )
            tips.append(
                f"{s} not checked ({c.tip_note})" if c.tip is None else f"{s} pass {int((c.tip & pk).sum())} env"
            )
            links.update(c.links)
        print(
            f"[EVAL_PARK_CUT] batch {self.b}: parked-hand touch (at the cut) "
            f"{' · '.join(parts) or 'no side'} (> {TOUCH_F_MIN:g} N) · hand links "
            + (" · ".join(f"{h[0].upper()} {k}" for h, k in links.items()) or "-")
            + f" · fingertip sensor check {' · '.join(tips) or '-'}",
            flush=True,
        )
        return e

    def _hand_touch(self, side: str) -> torch.Tensor:
        """(n,) bool — hand links (both hands) <-> `side` object contact force > TOUCH_F_MIN now (first n batch
        envs)."""
        c = self.env.hand_object_contact_force(side)
        own = c.force[side].float()
        oth = c.force.get("left" if side == "right" else "right")
        oth = torch.zeros_like(own) if oth is None else oth.float()
        return ((own > TOUCH_F_MIN) | (oth > TOUCH_F_MIN))[: self.n]

    def _ghost(self, step: int) -> None:
        """Gate D (park module doc): record velocity, hand touch and on-support at G1; count accumulated free falls at
        G2."""
        gdt = abs(float(self.env.sim.cfg.gravity[2])) * float(self.env.step_dt)
        held = self.env.held_object_assets()
        if step == self.d_at[0]:
            ids = torch.arange(self.n, device=held[0][1].data.root_lin_vel_w.device)
            frames = self.env.reference_frames(ids)
            for side, obj in held:
                gap = self.env.object_support_gap(ids, frames, side, obj) * 1000.0
                self.d_prev[side] = (
                    obj.data.root_lin_vel_w[: self.n].clone(),
                    self._hand_touch(side),
                    gap.abs().to(obj.data.root_lin_vel_w.device) <= REST_GAP_MM,
                )
            return
        parts = []
        for side, obj in held:
            v1, t1, rest = self.d_prev[side]
            v2 = obj.data.root_lin_vel_w[: self.n]
            free = ~(t1 | self._hand_touch(side).to(v2.device))
            fall = fall_accum_mask(v1, v2, gdt)
            c = int((fall & rest & free).sum())
            s = side[0].upper()
            self.d_cnt[s] = c
            parts.append((s, c, int(rest.sum()), int((rest & free).sum()), int(fall.sum())))
        cnt = " / ".join(f"{s} {c}" for s, c, *_ in parts)
        det = " · ".join(
            f"{s} on a support {r} · of them no hand touch {f} · accumulated signature (all) {a}"
            for s, _c, r, f, a in parts
        )
        print(
            f"[EVAL_GHOST] batch {self.b}: gate D count {cnt} (gate > {(1.0 - PARK_Q) * self.n:.2f} · step "
            f"{self.d_at[0]}→{self.d_at[1]} = {self.d_at[0] - D_AFTER_HOLD} after the hold window · {det} · "
            f"support |gap| ≤ {REST_GAP_MM:.0f} mm · "
            f"Δv_z = -g·dt {gdt:.4f} m/s ±{FF_TOL * 100:.0f} % · v_z ≤ -2g·dt · env {self.n})",
            flush=True,
        )

    def _print_park(self) -> None:
        """[PARK]: whether every action term used the last step's park mask in process_actions, plus the parked wrist
        <-> reference frame 0 distance (measured by the action terms on the actual virtual wrist)."""
        m = self.pushed
        if m is None:
            print(f"[PARK] batch {self.b}: lookahead=0 parked=0 (no parked env on the last step)", flush=True)
            return
        late = [type(t).__name__ for t in self.terms if t.parked_seen is not m]
        if late:
            raise RuntimeError(
                f"[PARK] batch {self.b}: action terms {late} did not use the last step's park mask — "
                "the parked wrist is not at frame 0"
            )
        ids = m.nonzero(as_tuple=False).squeeze(-1)
        off = [t.parked_wrist_offset(ids).float() for t in self.terms]
        e = torch.cat([o.norm(dim=-1) for o in off]) * 1000.0
        col, rows = self._reset_wrist(ids, off)
        print(
            f"[PARK] batch {self.b}: lookahead=0 parked={ids.numel()} · action terms {len(self.terms)} · "
            f"parked wrist ↔ ref frame 0 p50 {float(e.median()):.1f} · p99 {float(torch.quantile(e, PARK_Q)):.1f} · "
            f"max {float(e.max()):.1f} mm{col}",
            flush=True,
        )
        for r in rows:
            print(f"[EVAL_PARK_CLIPS] batch {self.b}: {r}", flush=True)

    def _reset_wrist(self, ids: torch.Tensor, off: list) -> tuple[str, list]:
        """Probe reset-wrist gate (column, rows over the limit): the number of (clip, hand) whose median wrist motion
        reset → cut (frame 0) exceeds RESET_WRIST_MM; 0 at cut RESET_WRIST_CUT = pass. Excluded: hands that start
        holding their own object (stated in self.held; touches at the cut are not excluded, they are pushed hands)
        and envs beyond the batch."""
        if not self.probe or self.w0 is None:
            return "", []
        cid, n, rows, n_e, mine = self._clip_of(ids), 0, [], 0, ids < self.n
        for t, o, w in zip(self.terms, off, self.w0, strict=True):
            d, held = (o - w[ids]).norm(dim=-1) * 1000.0, self.held[t._side_rb].to(ids.device)[cid] & mine
            n_e += int(held.sum())
            for c in torch.unique(cid[mine]).tolist():
                k = (cid == c) & mine & ~held
                n += int(bool(k.any()))
                if bool(k.any()) and float(d[k].median()) > RESET_WRIST_MM:
                    rows.append(
                        f"clip {c} {t._side_rb} reset wrist motion median {float(d[k].median()):.1f} · max "
                        f"{float(d[k].max()):.1f} mm (count {int(k.sum())}/"
                        f"{int(((cid == c) & mine).sum())})"
                    )
        word = (
            ("stop" if rows else "pass" if n else "unmeasurable (0 hands)")
            if self.step == RESET_WRIST_CUT
            else "reference"
        )
        return (
            f" · reset wrist motion (cut step {self.step}) (clip, hand) median > {RESET_WRIST_MM:.0f} mm "
            f"{len(rows)}/{n} · hands starting with their own object held excluded {n_e} env · "
            f"gate = cut {RESET_WRIST_CUT} — {word}"
        ), rows

    def at_cut(self, done_mask: torch.Tensor) -> None:
        if not self.on:
            return
        ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        touch = self._touch(ids)  # E at the cut (last park mask sent), read before the mask is cleared
        self._print_park()
        self._push(None)  # no parking after the cut: the mask must not leak into what follows
        n_done = int(done_mask[: self.n].sum())
        if ids.numel() == 0:
            print(f"[EVAL_PARK] batch {self.b}: parked 0/{n_done}", flush=True)
            return
        self.env.episode_length_buf[ids] = 0  # compare against the pinned frame 0 (the batch reset follows)
        frames = self.env.reference_frames(ids)
        ref = self.env.trajectory_manager.get_reference_state_for_envs(ids, frames)
        gdt = abs(float(self.env.sim.cfg.gravity[2])) * float(self.env.step_dt)
        drop = pin_drop_mm(float(self.env.sim.cfg.gravity[2]), float(self.env.physics_dt), float(self.env.step_dt))
        a_rows, art_all, cand, ex_all, excl, ffv, g_all, rch, sdz, signed = self._measure_held_objects_at_cut(
            drop, frames, gdt, ids, n_done, ref, touch
        )
        self._print_excl_a(a_rows)
        self._print_air(g_all, cand & torch.cat(ffv).to(cand.device), ex_all, rch, sdz, art_all)
        self._print_excl(ids, signed, excl)

    def _measure_held_objects_at_cut(self, drop, frames, gdt, ids, n_done, ref, touch):
        """Per held object at the cut: drop, lift, free-fall, reach and articulation measurements."""
        dz, lifts, parts, ffs, ffv, signed, excl, grp, reach, art = [], [], {}, [], [], [], [], [], [], []
        cid = self._clip_of(ids)
        for si, (side, obj) in enumerate(self.env.held_object_assets()):
            pos, _q, lift, src = self.env.object_pin_pose(ids, frames, side, obj, ref)  # pose parking used
            z = obj.data.root_pos_w[ids, 2] - self.env.scene.env_origins[ids, 2]
            dz.append((z - pos[:, 2]).abs())
            signed.append((side, z - pos[:, 2], lift))
            lifts.append(lift)
            vel = obj.data.root_lin_vel_w[ids]
            ffv.append(free_fall_mask(vel, gdt))
            ffs.append(pin_fall_mask(vel, (z - pos[:, 2]).float() * 1000.0, gdt, drop))
            reach.append(gravity_reach_mask(vel[:, 2], (z - pos[:, 2]).float() * 1000.0, float(self.env.step_dt)))
            excl.append(touch[side].to(dz[-1].device))
            art.append(torch.full((ids.numel(),), artic_asset(obj), dtype=torch.bool, device=dz[-1].device))
            grp.append(torch.where(ids < self.n, cid * 2 + si, torch.full_like(cid, -1 - si)).to(dz[-1].device))
            for k, v in src.items():
                parts.setdefault(k, []).append(v)
        dzs = torch.cat(dz).float() * 1000.0
        lft = torch.cat(lifts).float() * 1000.0
        ex_all = torch.cat(excl)
        g_all = torch.cat(grp)
        ex_a, a_rows = excl_a_mask(dzs, ex_all, g_all)
        pend = self.probe and self.step < self.d_at[1]  # a probe cut (e.g. cut 3) before gate D's measurement
        sdz, rch, cand = torch.cat([s for _side, s, _l in signed]).float() * 1000.0, torch.cat(reach), lft <= 0
        art_all = torch.cat(art)
        k_air, med_air = air_drop(sdz, lft, torch.cat(ffv), ex_all, rch, art_all)
        v = park_verdict(
            dzs,
            ex_all,
            lft,
            torch.cat(ffs),
            self.d_cnt,
            self.n,
            ex_a,
            pend,
            n_grp=int(torch.unique(g_all).numel()),
            n_grp_a=len(a_rows),
            drop=(drop, k_air, med_air),
        )
        self.invalid = bool(v["why"])
        unchecked = " — not checked" if k_air < AIR_MIN else ""
        air = (
            f"measured in-air median {med_air:+.2f} mm (samples {k_air}{unchecked})"
            if med_air is not None
            else "0 in-air samples — not checked"
        )
        src_txt = " · ".join(
            f"{k} >0 {int((torch.cat(x) > 0).sum())} max {float(torch.cat(x).max()) * 1000.0:.2f}"
            for k, x in parts.items()
        )
        gh = " / ".join(f"{s} {'?' if e is None else e}" for s, e in v["ghost"].items()) or (
            f"not measured (probe cut step {self.step} < {self.d_at[1]})" if pend else "none (not measured)"
        )
        lim = f"{DZ_LIMIT_MM:.0f}"
        print(
            f"[EVAL_PARK] batch {self.b}: parked {ids.numel()}/{n_done} (max {self.parked_max}) · "
            f"parked-env object |dz| at the cut p99={v['p99']:.2f} mm · >{lim} mm {v['over']} · "
            f"gate-A excluded (clip, side) {len(a_rows)}/{v['n_grp']} · samples {v['n_a']}/{v['n']} · "
            f"parked-hand touch excluded {v['n_excl']}/{v['n']} ({100.0 * v['n_excl'] / max(v['n'], 1):.1f} %) · "
            f"of them >{lim} mm {v['excl_over']} · p99 after exclusion={v['p99_eff']:.2f} mm · "
            f"rest >{lim} mm {v['rest_over']} · non-E {v['n_ne']} · floor {v['floor']} · "
            f"parked-reference lift (support penetration) >0 {int((lft > 0).sum())} · "
            f"p99 {float(torch.quantile(lft, PARK_Q)):.2f} · max {float(lft.max()):.2f} mm · "
            f"sources (object_pin.LIFT_SOURCES) {src_txt} mm · lifted-object free fall {v['n_ff']}"
            f"(fixed-pose one-step fall reference −{drop:.2f} ±{DROP_TOL_MM:g} mm · {air}) · "
            f"gate D free fall {gh} (gate > {v['ghost_lim']:.2f})"
            + "".join(f" — {w}" for w in v["why"])
            + "".join(f" · {w}" for w in v["note"]),
            flush=True,
        )
        return a_rows, art_all, cand, ex_all, excl, ffv, g_all, rch, sdz, signed

    def _print_excl_a(self, rows: list) -> None:
        """One line per gate-A-excluded (clip, side) plus a count line (also when 0). Fills self.excl_a."""
        self.excl_a = {}
        sides = [s for s, _o in self.env.held_object_assets()]
        for g, o, k, m, med in rows:
            if g < 0:
                c, side, name = -1, sides[-1 - g], "out of batch"
            else:
                c, side = g // 2, sides[g % 2]
                name = self.names[c] if self.names and c < len(self.names) else f"clip_{c}"
                self.excl_a[(c, side)] = med
            print(
                f"[EVAL_PARK_EXCL_A] batch {self.b}: c{c} {name} {side} {med:.2f} mm · over {o}/{m}(E as 0) · "
                f"non-E {k}",
                flush=True,
            )
        print(
            f"[EVAL_PARK_EXCL_A] batch {self.b}: excluded (clip, side) {len(rows)} · samples "
            f"{sum(r[3] for r in rows)} (rule: most samples have |dz| > {DZ_LIMIT_MM:.0f} mm after exclusion "
            f"— left out of gates A–C only · still scored · cap {EXCL_A_MAX_FRAC:.0%})",
            flush=True,
        )

    def _print_air(self, grp, cand, touch, reach, sdz, artic) -> None:
        """Always print, per (clip, side), the airborne candidates (lift 0, velocity signature) left out of the gate-C
        self-proof: hand touch, outside the gravity reach, articulated part."""
        sides = [s for s, _o in self.env.held_object_assets()]
        out, rows = cand & (touch | ~reach | artic), []
        for g in torch.unique(grp[out]).tolist():
            m = cand & (grp == g)
            c, side = (g // 2, sides[g % 2]) if g >= 0 else (-1, sides[-1 - g])
            name = "out of batch" if g < 0 else self.names[c] if self.names and c < len(self.names) else f"clip_{c}"
            r = m & ~reach
            rows.append(
                f"c{c} {name} {side} touch {int((m & touch).sum())} · "
                f"articulated {int((m & artic).sum())} · "
                f"out of reach {int(r.sum())}/{int(m.sum())}"
                + (f"(dz median {float(sdz[r].median()):+.2f} mm)" if bool(r.any()) else "")
            )
        print(
            f"[EVAL_PARK_AIR] batch {self.b}: airborne candidates left out of the self-proof "
            f"{int(out.sum())}/{int(cand.sum())} (hand touch · outside the gravity reach |v_z|·T · "
            f"articulated part) · (clip, side) {len(rows)}" + "".join(f" · {r}" for r in rows),
            flush=True,
        )

    def _print_excl(self, ids: torch.Tensor, signed: list, excl: list) -> None:
        """Always print the excluded E samples per (clip, side): excluded / that clip's samples, how many of them are
        over the threshold, the max force at the cut. The same counts are kept in `e_rows` (input of the E-clip
        oracle)."""
        cid = self._clip_of(ids)
        rows, self.e_rows = [], {}
        for (side, sdz, _lift), ex in zip(signed, excl, strict=True):
            mm = sdz.float() * 1000.0
            for c in torch.unique(cid[ex]).tolist():
                m = cid == c
                k = m & ex
                self.e_rows[(int(c), side)] = (int(k.sum()), int(m.sum()))
                n_over = int((mm[k].abs() > DZ_LIMIT_MM).sum())
                rows.append(
                    f"c{c} {side} {int(k.sum())}/{int(m.sum())}(>{DZ_LIMIT_MM:.0f} {n_over}"
                    f" · max force {float(self.force[side][ids][k].max()):.2f} N)"
                )
        tail = ": " + " · ".join(rows) if rows else ""
        print(f"[EVAL_PARK_EXCL] batch {self.b}: excluded (clip, side) {len(rows)}{tail}", flush=True)
