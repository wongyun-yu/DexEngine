"""Record stage-1 recorder: a subclass of the canonical trajectory recorder (rl.eval.traj_record.TrajRecorder) that only
appends the record columns (inside Kit).

What the canonical recorder already owns (unchanged): state columns (articulation root pose + joint values, rigid
poses) · reference object pose columns · success (term/trunc of the same env.step as judging, not an early stop) ·
candidates (success ∧ real placement ∧ PARK valid) · score (traj_select.score: object tracking error, smaller is
better) · ranking (traj_select.rank_key, here SELECT_BY) · the judge count check (finalize: a mismatch dies before
writing).
Added here (columns appended to the record rows; places in meta.env.collect.channels):
  action        the policy output fed to env.step (action_manager.action, raw: the residual policy output as is)
  joint_target  per robot articulation the absolute PD target joint values (data.joint_pos_target: what the
                controller actually tracks)
  force_total   per unfiltered hand contact sensor (all hand links) the **total** contact normal force (world, N) on the
                body: the sum over all partners (object, desk, support disc, other hand, own body)
                (net_forces_w_history) · columns = [history newest->oldest][body][xyz]
                Contact sensors of extra slots (dexengine.rl.tasks.scene.extra_slots.SENSOR_PREFIX: one slot body,
                reference lines only) are not hand sensors and are not stored: a count different from the statement
                (dexengine.rl.tasks.scene.extra_slots. slots_of) dies, and the number left out is printed on one
                `[COLLECT_REC] extra slot sensors …` line. Hand <-> extra forces are in the partner decomposition's
                force_partner/<sensor>.extra_obj_<k>, main object <-> extra forces in the extra_obj_<k> class of
                force_object_partner (arm <-> extra, like arm <-> desk, is not recorded).
  force_object  per object asset (env.held_object_assets; a shared object once) per hand link (both hands, hand
                rule) the contact normal force **given to that object** (world, N; sum over every object body,
                articulation links included). Owner = the same read-only PhysX filter view as the judge PARK gate E
                (hand_object_contact.link_forces: object = sensor, hand links = filter) · columns =
                [history][link][xyz]
  Sign: force_total = force received by the link · force_object = force the link gives the object. So the force a
  link receives from non-objects (desk, support disc, other hand, own body) = force_total + Σ_objects force_object
  (same link column). Non-objects cannot be told apart by this record (more views would be needed). PhysX contact
  forces are normal components only (no friction; force_total too).
  Filtered contact sensors (fingertip <-> its object, for rewards and judging) are not stored: their net force copies
  the force_total row and the filter result is a subset of force_object. They serve as an independent oracle
  instead (measured vs measured): when the sensor filter covers every object body, force_object[link] + sensor
  force_matrix_w ≈ 0 (action and reaction, two PhysX views) is measured over the whole rollout, printed in finalize
  as a `[COLLECT_REC] oracle` line, and beyond the threshold it dies before writing.
  History: the scene cfg's contact sensors are not changed (the rollout must equal judging; adding sensors or longer
  history is a scene change). Canonical sensor history 1 -> per policy step (50 Hz) one value of the last physics
  step (physics 100 Hz, decimation 2). The object view is also the last physics step, so both forces share a time
  axis; a sensor history window above 1 (min(decimation, history length) > 1) cannot be followed by the object force
  and dies.
  Partner decomposition (always on; the old switch DEXENGINE_COLLECT_FORCE_PARTNER is retired, its name stays in the
  meta field collect.partner.env; marker `[COLLECT_REC] partner=on …`): read-only PhysX contact views
  (rollout.partner; physics and scene sensor cfg unchanged) give
  force_partner   per hand sensor (same sensors and bodies as force_total) x partner class the force the link
                  **received** from that class: channel `force_partner/<sensor>.<class>` · columns =
                  [history][body][xyz] · Σ classes = force_total (oracle)
  force_object_partner per object (sum over bodies) x partner class the force the object **received** from that
                  class · columns = [history][class][xyz] · Σ classes = the same view's object net (oracle) · the hand
                  class (hand_<side>) = Σ force_object over that side's links
  contact_sep     per object x hand link (same links and order as force_object) the minimum contact point separation
                  [m] (negative = penetration, no contact NaN) · columns = [history][link]
  Owner of the class rule and the oracle thresholds = rollout.partner (module doc). Both oracles are measured over the
  whole rollout and die before writing when exceeded.
Only K_TOP trajectories are kept (choose 1; the top 3 are not needed). The top-TOP_N ranking table is read by
selection_table from the all-episode summary (ep_table, ep_score + the rank key layer rank_table_class).
Switched on by dexengine/record/session/entry.py --stage record (a hook replacing the canonical TrajRecorder by this
class, marker [COLLECT_REC]).
"""

from __future__ import annotations

from pathlib import Path

from dexengine.core.errors import InputMissingError

K_TOP = 1
#: number of top ranks in the selection table meta
TOP_N = 3
#: meta field carrying the per-episode rank key (first component of traj_select.rank_key: end error for RANK_END; the
#: second component = ep_score)
RANK_KEY_META = "ep_rank_key"
# record candidate ranking = smallest end error (traj_select.RANK_END; rule owner traj_select.rank_key; an unknown value
# dies at the first batch). The success judge allows lagging tracking, so a success that ends with the object off the
# target is not chosen (the success count is unchanged).
SELECT_BY = "end"
MARK = "[COLLECT_REC]"
KIND_TOTAL = "force_total"
KIND_OBJECT = "force_object"
KIND_PARTNER = "force_partner"  # partner decomposition: hand sensor x partner class
KIND_OBJ_PARTNER = "force_object_partner"  # partner decomposition: object x partner class
KIND_SEP = "contact_sep"  # partner decomposition: object x hand link minimum separation (not a force)
FORCE_KINDS = (KIND_TOTAL, KIND_OBJECT, KIND_PARTNER, KIND_OBJ_PARTNER)
CONTACT_KINDS = FORCE_KINDS + (KIND_SEP,)
PARTNER_ENV = "DEXENGINE_COLLECT_FORCE_PARTNER"
#: recorded in meta collect.partner.env (the decomposition itself is always on)
# kind of old records (no object force): the values were net_forces_w_history per sensor (filtered sensors too, the
# total force). When rendered or stored again they read as force_total in that sense (so the name does not suggest
# 'object').
LEGACY_KIND = {"force": KIND_TOTAL}
#: force sign notes stored in the record meta and info.json (data; old Korean values -> phase5 legacy_values.tsv)
FORCE_SIGN = {
    KIND_TOTAL: "total contact normal force the link received (any partner)",
    KIND_OBJECT: "contact normal force the link applied to that object (sum over the object bodies) — the force "
    "the link received from the object = −value",
}
# meaning of the partner kinds (stored in the meta and info.json; data, old values -> legacy_values.tsv)
PARTNER_SIGN = {
    KIND_PARTNER: "contact normal force the link received from that partner class (Σ classes = force_total)",
    KIND_OBJ_PARTNER: "contact normal force the object (sum over its bodies) received from that partner class "
    "(Σ classes = object net)",
    KIND_SEP: "minimum separation [m] of the object ↔ hand link contact points — negative = penetration · no contact "
    "point = NaN",
}
# oracle threshold: |force_object[link] + sensor force_matrix_w| <= ORACLE_ABS_N + ORACLE_REL x |sensor value|
# (float32 summation order)
ORACLE_ABS_N = 1e-3
ORACLE_REL = 1e-4


def spawn_hold_steps() -> int:
    """Policy steps after a reset during which the object is written back to its reference pose: the fixed window of
    the force quality marking (store.force). Value owner = the env set by the training canonical table
    (scripts/_rl_recipe.sh: two hands 10, one hand none = 0), read by the same function as train_arm_env.step
    (spawn_hold.spawn_hold_steps; values outside the form die). The env is not parsed again here."""
    from dexengine.rl.tasks.scene.spawn_hold import spawn_hold_steps as owner  # noqa: PLC0415

    return owner()


def log_problems(text: str) -> list[str]:
    """Record session log: the partner decomposition marker must be present."""
    has = any(ln.startswith(f"{MARK} partner=on") for ln in text.splitlines())
    return [] if has else [f"contact-partner marker '{MARK} partner=on' missing"]


def require_partner(col: dict, what: str) -> None:
    """Store entry: a record meta without the partner decomposition (a stage-1 cache made with it off) dies."""
    if not (col.get("partner") or {}).get("on"):
        raise InputMissingError(
            f"{MARK} {what}: partner decomposition is on, but the record has none (a record made with it off)"
        )


def channel_block(c: dict, blk, L: int):
    """Record column block [L, width] -> channel shape: forces [L, history, name, 3] · separation [L, history, name]."""
    tail = () if c["kind"] == KIND_SEP else (3,)
    return blk.reshape(L, c["history"], len(c["names"]), *tail)


def contact_kind(c: dict) -> str | None:
    """Channel -> contact kind (force kind or KIND_SEP), else None."""
    return KIND_SEP if c["kind"] == KIND_SEP else force_kind(c)


def force_kind(c: dict) -> str | None:
    """Channel -> force kind (KIND_TOTAL | KIND_OBJECT), None if not a force. The old records' 'force' reads as
    force_total."""
    k = LEGACY_KIND.get(c["kind"], c["kind"])
    return k if k in FORCE_KINDS else None


def object_force_rows(m_env, n_env: int):
    """(n_env, links, 3) -> record columns [n_env, 1 x links x 3] (history 1, float32)."""
    return m_env.reshape(n_env, -1).float()


def oracle_excess(obj_col, sensor_fm):
    """(n_env, 3) force the object received from that link · (n_env, 3) force the sensor (link) received from the
    object -> (max difference, max excess over the threshold, max sensor |F|); with action and reaction the sum is 0."""
    d = (obj_col + sensor_fm).norm(dim=-1)
    ref = sensor_fm.norm(dim=-1)
    return d.amax(), (d - (ORACLE_ABS_N + ORACLE_REL * ref)).amax(), ref.amax()


def selection_table(tf, clip: str) -> dict:
    """One clip's selection table, read from the all-episode summary (ep_table, ep_score, the record meta rank keys
    RANK_KEY_META; only the one chosen trajectory is kept). Candidates = success ∧ PARK valid ∧ real placement ·
    ranking = (rank key, score, row order) ascending (as the recorder: the first seen wins a tie) · top TOP_N.
    Oracle: the first ranked row = the row of the recorder's kept choice (kept_row); a mismatch dies. A record without
    rank keys (old record) dies."""
    ci = tf.clip_names.index(clip)
    keys = ((tf.meta.get("env") or {}).get("collect") or {}).get(RANK_KEY_META)
    if keys is None or len(keys) != len(tf.ep_table):
        raise SystemExit(
            f"[COLLECT_STORE] {clip}: record meta rank keys ({RANK_KEY_META}) "
            f"{'missing' if keys is None else f'{len(keys)} != rows {len(tf.ep_table)}'}: {tf.path}"
        )
    rows = [
        {
            "row": i,
            "batch": int(r[1]),
            "env": int(r[2]),
            "warmup": bool(r[3]),
            "park_valid": bool(r[4]),
            "success": bool(r[5]),
            "length": int(r[6]),
            "score": float(s),
            "rank_key": float(keys[i]),
        }
        for i, (r, s) in enumerate(zip(tf.ep_table, tf.ep_score, strict=False))
        if int(r[0]) == ci
    ]
    real = [r for r in rows if not r["warmup"]]
    cand = sorted(
        (r for r in real if r["success"] and r["park_valid"]), key=lambda r: (r["rank_key"], r["score"], r["row"])
    )
    top = [
        {"rank": k, "score": r["score"], "rank_key": r["rank_key"], "batch": r["batch"], "env": r["env"]}
        for k, r in enumerate(cand[:TOP_N])
    ]
    j = int(tf.sel[ci])
    kept = int(tf.kept_row[j]) if j >= 0 else None
    if (cand[0]["row"] if cand else None) != kept:
        first = cand[0]["row"] if cand else None
        raise SystemExit(
            f"[COLLECT_STORE] {clip}: first ranked row of the selection table {first} != recorder's chosen row {kept}"
        )
    # the fallback rule text lands in the stored meta (data; the old Korean value is in legacy_values.tsv)
    rank = tf.meta.get("rank") or {
        "by": "score",
        "rule": "lowest score among the successful candidates (recorded before the rank rule was written)",
    }
    return {
        "clip": clip,
        "rollouts": len(real),
        "success": sum(r["success"] for r in real),
        "rank": rank,
        "candidates": len(cand),
        "top": top,
        "selected": top[0] if top else None,
        "rows": sorted(real, key=lambda r: r["score"]),
    }


def rank_table_class(base):
    """Subclass of the canonical recorder (base) adding a per-episode rank key: for every row of each batch (warm-up,
    failure and PARK-invalid included; the candidate predicate belongs to the canonical recorder alone) the first
    component of traj_select.rank_key(RANK_BY) is measured (the same state slice and function: the same value as the
    kept choice), and finalize stores them in meta.env.collect[RANK_KEY_META] in ep_table row order. Rows != keys dies
    before writing."""
    import sys

    class RankTable(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._ep_key: list[float] = []

        def end_batch(self, park_invalid: bool = False) -> None:
            ts = sys.modules[base.__module__].traj_select  # the traj_select the canonical recorder uses (no copy)
            n0 = len(self.rows)
            super().end_batch(park_invalid)
            L, buf, sides = self.length.cpu().numpy(), self.buf.numpy(), self.sides_cols()
            for r in self.rows[n0:]:
                e = int(r[2])
                self._ep_key.append(float(ts.rank_key(self.RANK_BY, buf[: L[e], e, :], sides, r[7])[0]))

        def finalize(self, aggregate):
            if len(self._ep_key) != len(self.rows):
                raise RuntimeError(f"{MARK} rank keys {len(self._ep_key)} != episode rows {len(self.rows)}")
            self.meta_in.setdefault("collect", {})[RANK_KEY_META] = list(self._ep_key)
            return super().finalize(aggregate)

    return RankTable


def _oracle_verdict(self):
    """The object-force oracle line; untouched clips go to oracle_gate; a force mismatch dies (no oracle: one line)."""
    if self._oracle:
        d, ex, ref, nc = (float(x) for x in self._ora.cpu()) if self._ora is not None else (0.0, 0.0, 0.0, 0.0)
        print(
            f"{MARK} oracle sensors {len(self._oracle)} · touched samples {int(nc)} · max |object force + "
            f"sensor force| {d:.4g} N · max sensor |F| {ref:.4g} N · over threshold {max(ex, 0.0):.4g} N",
            flush=True,
        )
        if nc <= 0:  # zero touched samples: oracle_gate (UNTOUCHED | BUMP | dies)
            oracle_gate(self, self._wit)
        if ex > 0:
            raise RuntimeError(
                f"{MARK} object force != fingertip sensor filter force (action and reaction): difference {d:.4g} N"
            )
    else:
        print(
            f"{MARK} no oracle: no filter sensor covers every object body (articulation or shared object side)",
            flush=True,
        )


def _collect_meta(self, raw_env):
    """Record column ranges, meta_in.collect (and its partner block) and the marker lines of the record columns."""
    lo, rows = self.D, []
    for c in self._chs:
        rows.append({k: v for k, v in c.items() if k not in ("read", "obj")} | {"lo": lo, "hi": lo + c["width"]})
        lo += c["width"]
    self.D = lo
    self.meta_in["collect"] = {
        "channels": rows,
        "k_top": K_TOP,
        "top_n": TOP_N,
        "physics_dt": float(raw_env.physics_dt),
        "step_dt": float(raw_env.step_dt),
        "decimation": int(raw_env.cfg.decimation),
        "force_rate_hz": round(1.0 / float(raw_env.physics_dt), 3),
        "force_sign": FORCE_SIGN,
        "force_component": "normal",
        "spawn_hold_steps": spawn_hold_steps(),
        "owner": "scripts/collect/collect_record.py",
    }  # format tag in record meta (kept)
    if self._partner is not None:
        self.meta_in["collect"]["partner"] = {
            "on": True,
            "sign": PARTNER_SIGN,
            "env": PARTNER_ENV,
            "n_filters": len(self._partner.filt),
            "owner": "scripts/collect/collect_partner.py",
        }
        print(f"{MARK} partner=on · {self._partner.describe()}", flush=True)
    n_ref = self.D - self.d_state - sum(c["width"] for c in rows)
    print(
        f"{MARK} on · columns {self.d_state} (state) + {n_ref}"
        f" (reference) + " + " · ".join(f"{c['name']} {c['width']}" for c in rows) + f" = D {self.D} · "
        f"k_top {K_TOP} · physics {1.0 / float(raw_env.physics_dt):.0f} Hz · step "
        f"{1.0 / float(raw_env.step_dt):.0f} Hz · oracle sensors {len(self._oracle)}",
        flush=True,
    )


def collect_recorder_class(base):
    """Record subclass of the canonical TrajRecorder (base); the hook replaces the module attribute with it (on top of
    the rank key layer rank_table_class)."""

    class CollectRecorder(rank_table_class(base)):
        RANK_BY = SELECT_BY

        def __init__(self, raw_env, clip_names, out_path, k_keep=K_TOP, meta=None, term_cls=None):
            super().__init__(raw_env, clip_names, out_path, k_keep=K_TOP, meta=meta, term_cls=term_cls)
            # owner of the column list (reads this module's field names and markers; late import)
            from dexengine.record.rollout import channels as CH

            self._chs, self._oracle, self._partner = CH.channels(raw_env, self.ents)
            self._ora = None
            # paired object channel -> max |object force| over all links (hand_object_contact view, another witness)
            self._wit: dict = {}
            _collect_meta(self, raw_env)

        def _row(self):
            import torch

            if self._partner is not None:
                self._partner.refresh()
            vals = [c["read"]() for c in self._chs]
            self._check_oracle({c["name"]: v for c, v in zip(self._chs, vals, strict=True) if c["kind"] == KIND_OBJECT})
            if self._partner is not None:
                self._partner.check(
                    {
                        c["name"].split("/", 1)[1]: v
                        for c, v in zip(self._chs, vals, strict=True)
                        if c["kind"] == KIND_TOTAL
                    }
                )
            return torch.cat([super()._row(), *vals], dim=-1)

        def _check_oracle(self, obj_vals: dict) -> None:
            import torch

            if not self._oracle:
                return
            n = int(self.env.num_envs)
            if self._ora is None:
                self._ora = torch.zeros(4, device=next(iter(obj_vals.values())).device)
                self._ora[1] = -float("inf")
            for _k, s, ch, j in self._oracle:
                col = obj_vals[ch].reshape(n, -1, 3)[:, j]
                fm = s.data.force_matrix_w.reshape(n, -1, 3).sum(dim=1).float()
                d, ex, ref = oracle_excess(col, fm)
                self._ora[0] = torch.maximum(self._ora[0], d)
                self._ora[1] = torch.maximum(self._ora[1], ex)
                self._ora[2] = torch.maximum(self._ora[2], ref)
                self._ora[3] += (fm.norm(dim=-1) > 0).sum()
            for ch in {ch for _k, _s, ch, _j in self._oracle}:
                w = obj_vals[ch].reshape(n, -1, 3).abs().amax(dim=(0, 2))  # max |object force| per link
                self._wit[ch] = torch.maximum(self._wit[ch], w) if ch in self._wit else w

        def finalize(self, aggregate):
            _oracle_verdict(self)
            if self._partner is not None:
                lines, bad = self._partner.verdict()
                for ln in lines:
                    print(ln, flush=True)
                if bad:
                    raise RuntimeError(f"{MARK} partner oracle failed: " + " · ".join(bad))
            out = super().finalize(aggregate)  # [TRAJ_REC] DONE: written only after the count equation (else dies)
            # the module the canonical entry loaded (hook marker)
            from dexengine.rl.isaac_cli import mark_closing  # noqa: PLC0415

            mark_closing("record stage 1 [TRAJ_REC] DONE · count equation · oracles passed")
            return out

    return CollectRecorder


# ---- oracle gate for zero touched samples ---------------------------------------------------------------------
# Called by CollectRecorder.finalize when the fingertip filter sensors stay 0 over the whole rollout. Witness = the
# partner-object channels (hand_object_contact view) per link max |object force|. Gates:
#   ORACLE_UNTOUCHED  the object channels are also 0 everywhere: no contact to prove (untouched_ok).
#   ORACLE_BUMP       every partner object with force was never touched by the human hand holding it in the human
#                     data (contact sidecar GT 0 frames, phases.human_touch): a non-fingertip link bumped an object
#                     that is not manipulated. Meta collect.oracle_gate = {rule, channels: {ch: {max_n, links,
#                     human_frames}}}.
# Anything else (a manipulated object the human touched, fingertip sensors 0, object force > 0) is a wiring fault and
# dies; unreadable human contact also dies. With several clips per session the human contact is summed.
ORACLE_UNTOUCHED, ORACLE_BUMP = "ORACLE_UNTOUCHED", "ORACLE_UNTOUCHED_BUMP"


def untouched_ok(wit: dict) -> bool:
    """Partner-object channels exist and are 0 over all links and samples (ORACLE_UNTOUCHED); {} is not."""
    return bool(wit) and all(v == 0.0 for v in wit.values())


def oracle_verdict(wit: dict, human) -> tuple[str, dict]:
    """wit = {partner-object channel: [max |object force| per link]}, human(channel) = frames in which the human hand
    holding that object touched it -> (gate name, per-channel record); dies when the gate does not pass."""
    flat = {ch: max(v) if v else 0.0 for ch, v in wit.items()}
    if untouched_ok(flat):
        return ORACLE_UNTOUCHED, {}
    if not wit:
        raise RuntimeError(f"{MARK} oracle has no partner-object channel — cannot judge zero touched samples")
    rec = {}
    for ch, v in sorted(wit.items()):
        if flat[ch] == 0.0:
            continue
        frames = int(human(ch))
        if frames != 0:
            raise RuntimeError(
                f"{MARK} oracle saw no touched sample — the object force is unproven · partner-object "
                f"hand-link force max {flat} · {ch} = {frames} human contact frames (manipulated "
                f"object)"
            )
        rec[ch] = {"max_n": round(flat[ch], 4), "links": sum(x > 0.0 for x in v), "human_frames": 0}
    return ORACLE_BUMP, rec


def human_frames(rec, ch: str) -> int:
    """Recorder rec's partner-object channel ch -> human contact frames of the hands holding it (channel sides), summed
    over clips; read from rec.traj_mgr.cfg.data_dir/<clip> via record.phases.human_touch."""
    from dexengine.record import phases as PH  # noqa: PLC0415

    sides = next((c.get("sides") for c in rec._chs if c["name"] == ch), None)
    root = getattr(getattr(getattr(rec, "traj_mgr", None), "cfg", None), "data_dir", None)
    clips = list(getattr(rec, "clip_names", None) or [])
    if not sides or not root or not clips:
        raise RuntimeError(
            f"{MARK} {ch}: cannot read human contact (sides {sides} · root {root} · clips "
            f"{len(clips)}): cannot judge zero touched samples"
        )
    return sum(int(PH.human_touch(Path(root) / c, s).sum()) for c in clips for s in sides)


def oracle_gate(rec, wit_t: dict) -> str:
    """finalize's zero-touched-samples branch: wit_t = {channel: per-link max (tensor)} -> gate name, meta and marker
    (dies when it does not pass)."""
    wit = {ch: [float(x) for x in v.reshape(-1).cpu()] for ch, v in wit_t.items()}
    rule, chs = oracle_verdict(wit, lambda ch: human_frames(rec, ch))
    rec.meta_in.setdefault("collect", {})["oracle_gate"] = {"rule": rule, "channels": chs}
    if rule == ORACLE_UNTOUCHED:
        print(
            f"{MARK} oracle zero touched samples — partner objects {sorted(wit)} hand-link force 0 (all samples, "
            f"hand_object_contact witness) · nothing to prove ({ORACLE_UNTOUCHED} · gate passed)",
            flush=True,
        )
    else:
        print(
            f"{MARK} oracle zero touched samples — "
            + " · ".join(f"{ch} max {r['max_n']:.4g} N · links {r['links']}" for ch, r in chs.items())
            + " · 0 human contact frames for that hand — a non-fingertip link bumped an object that is not "
            f"manipulated; force stays in the object channel ({ORACLE_BUMP} · gate passed)",
            flush=True,
        )
    return rule
