"""Per-clip render and store (the render and store steps of record.bundle): one Isaac session per clip draws the
round's trajectories (record.episodes.ROUNDS) in turn, with looks as the episode plan says (record.episodes; render
layer = render.clip_phases.render_clip).
Rounds: orig = the original only (session trajectory = the original) · aug = the variants only (the original is loaded
to fix the camera; choice = PER_CLIP of the variants). Original off-screen rule (record.episodes.ORIG_FOV_RULE): when
the orig session finds the original not visible at grasp, the render is skipped (an exclusion with a reason,
round_excluded) and the aug session draws rows 0-3 (no visible variant = row 0 is the original). The aug round starts
a session even for a clip with 0 variant candidates (original visible: nothing to draw -> round_excluded; not visible:
row 0 = the original).
Store place = the round place (record.piece.round_dir: <out>/rounds/<round>/{dataset, preview}).
The fixed session cost (boot, launcher dry run, first settle) is paid once per clip (GPU1 batch_render_probe: grouping
envs does not lower the render cost).

Flow   base record (stage 1, record.bundle) -> augment physics (record.chain.after_store, up to the chosen variant
       trajectories) -> per clip stage2_clip (one dump, one session, plan.json) -> one store item per episode
       -> record.store.store once.
Cache  <out>/stage2c/<key>/<clip>/; key = sha and name of the member trajectory files; reuse = render_clip.json ok,
       same render fingerprint and plan sha.
Markers [COLLECT_CLIP] … · session end line [RENDER_CLIP] DONE episodes=<n>/<m> (render.clip_session) · piece seconds
       [COLLECT_TIME].
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.hashing import sha256_file
from dexengine.record import episodes as EPI
from dexengine.record import tree_ref as TR
from dexengine.record.cache import render_fingerprint
from dexengine.record.looks import rev as RD  # re-render of brightness-gate listed episodes
from dexengine.record.render import episode as CRN  # render output names (media_files); cache reuse only when complete
from dexengine.record.render import fov as FOW  # owner of the original off-screen verdict
from dexengine.record.render import markers as PM  # place markers on (plan field) · drawn-marker check
from dexengine.record.session import launch as CLA

MARK = "[COLLECT_CLIP]"
ROOT_DIR = "stage2c"  # per-clip render root (under out)
EPISODE_GLOB = "*/*/*/*"  # <key>/<clip>/<traj>/<look>: episode render folder shape (record.clean finds them so)
SHADOW_REL = Path("aug") / "shadow"  # shadow root (owner: the augment run, root/"shadow")
#: per-clip render share of the bundle time cap (driver default; spec mode uses the spec's budget_s): a new session
#: per look, so one clip = 1 head + <= 20 episode sessions (aug rows 0-3 x 5 looks) · RTX 5090 shared-load sessions
#: 105-187 s -> 21 x 150 s leaves margin. A hung session is cut first by the session cap (launch.SESSION_TIMEOUT_S).
RENDER_TIMEOUT_PER_CLIP_S = 3600

PER_LOOK_RULE = "per_look_session"  # a new session per look (an RTX cache ghost of the previous look in one session)
PER_LOOK_MARK = "[COLLECT_PER_LOOK]"


def members_key(members: list[dict]) -> str:
    body = [f"{m['clip']}:{sha256_file(Path(m['traj']))[:16]}" for m in members]
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()[:16]


def extra_slots_of(cell: dict) -> str:
    """Extra-slot value of the replay render scene = the one of the record session (the run ledger of the cell's
    checkpoint; owner train_scene.slots_value). The replay loads no checkpoint and so skips the ledger check
    (extra_slots_ledger_check); without passing it, the default-on value slips in and slots absent from the record of
    a ledger-less cell (trained with it off) appear in the scene and the replay binding dies. A value the cell states
    must match."""
    if cell.get("kin"):  # kinematic cell: no checkpoint; the cell env must state the value
        told = [t.split("=", 1)[1] for t in cell["env"] if t.startswith("DEXENGINE_EXTRA_SLOTS=")]
        if not told:
            raise InputMissingError(f"{MARK} kinematic cell env has no DEXENGINE_EXTRA_SLOTS (no default)")
        print(f"{MARK} extra slots {told[-1]}: kinematic cell env", flush=True)
        return told[-1]
    from dexengine.rl.launch import train_scene as TS

    v = TS.slots_value(str(cell["ckpt"]))
    told = [t.split("=", 1)[1] for t in cell["env"] if t.startswith("DEXENGINE_EXTRA_SLOTS=")]
    if told and told[-1] != v:
        raise SystemExit(f"{MARK} cell env DEXENGINE_EXTRA_SLOTS={told[-1]} != checkpoint ledger {v}: {cell['ckpt']}")
    print(f"{MARK} extra slots {v}: checkpoint ledger ({TS.ledger_of(str(cell['ckpt']))})", flush=True)
    return v


def fov_owner_out(a, clip: str) -> bool:
    """Did the owner (--orig-fov-out, render.fov) call this clip's original off screen? No list = False (old
    path)."""
    from dexengine.core import paths as P

    return P.clip_name(clip) in FOW.owner_set(a)


def _member_trees(members, a, key, seq_root):
    """Tree records of the member trajectories (seq1c/<key>) and the members.txt list."""
    trajs = [
        {"name": m["clip"], "traj": str(m["traj"]), "role": "orig" if k == 0 else "aug"} for k, m in enumerate(members)
    ]
    seq1 = a.out / "seq1c" / key
    lsts = [TR.resolve(TR.write(seq1 / m["clip"], seq_root, m["clip"]))[1] for m in members]  # fingerprint per tree
    names = seq1 / "members.txt"
    names.parent.mkdir(parents=True, exist_ok=True)
    names.write_text("".join(Path(f).read_text() for f in lsts))
    return trajs, lsts, names


def _render_sessions(prev, eps_plan, stamp, rfp, gone, clip, done, out, plan, key, cell, seq_root, lsts, a, main_r):
    """Run the per-look render sessions of this clip unless a finished run with the same fingerprint is present."""
    from dexengine.record import bundle as C

    if prev.get("ok") and eps_plan is not None and stamp.is_file() and stamp.read_text().strip() == rfp and not gone:
        print(f"{MARK} present {clip} · episodes {len(eps_plan)} · fingerprint {rfp}", flush=True)
    else:
        if done.is_file():
            done.unlink()
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
        alias = f"collectc_{key}"
        tok = C.env_set(
            [t for t in cell["env"] if not t.startswith(("DEXENGINE_SEQ_LIST=", "SEQ_DIR="))],
            SEQ_DIR=seq_root,
            DEXENGINE_SEQ_LIST=lsts[0],
            GPU=a.gpu,
            DEXENGINE_MAIN_ROOT=main_r,
            DEXENGINE_RL_QUEUE_ROOT=a.queue_root,
            RENDER_ONLY="replay",
            DEXENGINE_EXTRA_SLOTS=extra_slots_of(cell),
        )
        C.dryrun(a.tree, tok, alias, out / "dryrun.out")
        dump = C.ref_dir(a.tree, main_r) / f"{alias}_{clip}_replay_render.log"
        if not dump.is_file():
            raise InputMissingError(f"{MARK} {clip}: no original replay dump {dump}")
        args = ["--stage", "render_clip", "--dump", str(dump), "--seq-dir", str(seq_root)]
        if (
            not plan.get("choose")
            and len(plan["episodes"] or []) <= 1
            and all(e["look"] == "D" for e in plan["episodes"] or [])
        ):  # look D only = one session (no head)
            CLA.session(
                a.tree, a.gpu, args + ["--plan", str(out / "plan.json")], out / "session.log", "[RENDER_CLIP] DONE"
            )
            g = RD.redraw_after(
                lambda t, pp: _part(a, args, out, t, pp), out, plan, "single", json.loads(done.read_text())
            )
            done.write_text(json.dumps(g, indent=1, ensure_ascii=False))
        else:  # several looks · choice (aug) = a new session per look
            done.write_text(json.dumps(per_look(a, args, out, plan), indent=1, ensure_ascii=False))
        stamp.write_text(rfp + "\n")


def _check_rendered(done, plan, out, clip):
    """The finished run: chosen plan, every planned (traj, look) drawn, place markers, no episode left on the redraw
    list."""
    got = json.loads(done.read_text())
    eps_plan, choice = chosen_plan(plan, got)
    want = {(e["traj"], e["look"]) for e in eps_plan}
    have = {
        (e["traj"], e["look"])
        for e in got.get("episodes", [])
        if (out / e["traj"] / e["look"] / "render_meta.json").is_file()
    }
    if not got.get("ok") or want != have:  # as sets, not counts
        raise SystemExit(f"{MARK} {clip}: missing {sorted(want - have)[:4]} · extra {sorted(have - want)[:4]}: {out}")
    PM.check_episodes(out / e["traj"] / e["look"] for e in eps_plan)  # drawn markers == requested (worker env mix-up)
    if RD.redraw_problems(out, got):  # episodes left on the brightness-gate list = a silent pass
        raise SystemExit(f"{MARK} {clip}: {RD.redraw_problems(out, got)[:3]} — {out}")
    return eps_plan, choice


def stage2_clip(a, cell: dict, clip: str, members: list[dict], rnd: str) -> tuple[list[dict], str | None]:
    """One clip, round rnd -> (episode store items (render_dir = <out>/<traj>/<look>), reason when there is nothing to
    draw | None). An episode the session left out dies. For the aug round members = [original (camera), variant
    candidates …] (the original alone with 0 candidates); otherwise [original]."""

    aug = EPI.aug_on(rnd)
    o = Path(members[0]["traj"]).resolve()
    if aug and not (o.is_file() and o.is_relative_to((a.out / "stage1").resolve())):  # original to measure allow on
        raise InputMissingError(
            f"{MARK} {clip}: the aug round's original trajectory is not in the bundle stage1 cache {o}"
        )
    if not aug and len(members) != 1:
        raise SystemExit(f"{MARK} {clip}: round {rnd} with {len(members)} trajectories; orig has the original only")
    main_r = (a.out / SHADOW_REL) if len(members) > 1 and not cell.get("kin") else a.main  # aug with 0 candidates =
    #                                                        original place · kinematic = variant trees under a.main too
    seq_root = main_r / Path(cell["seq_dir"]).resolve().relative_to(a.main)
    key = members_key(members)
    out = a.out / ROOT_DIR / key / clip
    trajs, lsts, names = _member_trees(members, a, key, seq_root)
    from dexengine.record.augment import policy as AP  # owner of the trajectory count PER_CLIP

    own = False if aug and fov_owner_out(a, clip) else None  # owner of the original off-screen verdict (render.fov)
    # kinematic = every received variant (KIN_RULE, kin_n_choose) · otherwise PER_CLIP
    nc = EPI.kin_n_choose(AP.PER_CLIP, len(members) - 1) if aug and cell.get("kin") else AP.PER_CLIP
    if nc != AP.PER_CLIP:
        print(
            f"{EPI.KIN_ALLVAR_MARK} {clip}: kinematic received variants {len(members) - 1} > PER_CLIP "
            f"{AP.PER_CLIP}: all drawn · deck rows {1 + nc}",
            flush=True,
        )
    plan = EPI.plan(
        clip,
        str(out),
        str(seq_root),
        str(names),
        trajs,
        rnd,
        EPI.deck_args(nc),
        n_choose=nc if aug else None,  # augment = the session picks variants by grasp visibility
        orig_fov_owner=own,
        kin=bool(cell.get("kin")),
    )
    PM.stamp_plan(plan)  # place markers on (env -> plan field; off = unchanged)
    stamp, rfp = out / "render_code.txt", f"{render_fingerprint(a.tree)}+{plan['plan_sha']}"
    done = out / "render_clip.json"
    prev = json.loads(done.read_text()) if done.is_file() else {}
    eps_plan = plan["episodes"] if plan["episodes"] is not None else prev.get("episodes_plan")
    if not aug and prev.get("orig_fov") is not None:  # plan fixed by the orig session's off-screen rule (skip = [])
        eps_plan = prev.get("episodes_plan")
    gone = [
        str(d / m)
        for e in eps_plan or []
        for d in [out / e["traj"] / e["look"]]
        for m in CRN.media_files(d)
        if not (d / m).is_file()
    ]  # removed by record.clean -> drawn again
    _render_sessions(prev, eps_plan, stamp, rfp, gone, clip, done, out, plan, key, cell, seq_root, lsts, a, main_r)
    eps_plan, choice = _check_rendered(done, plan, out, clip)
    by = {m["clip"]: m for m in members}
    why = None
    if not eps_plan:
        why = (
            f"{EPI.ORIG_FOV_WHY}(orig round · {EPI.ORIG_FOV_RULE} · off screen at the grasp)"
            if not aug
            else "aug round — original visible · 0 variant candidates (augment physics counts.json)"
        )
        print(f"{EPI.ORIG_FOV_MARK} {clip}: nothing to draw: {why}", flush=True)
    return [
        dict(
            by[e["traj"]],
            render_dir=str(out / e["traj"] / e["look"]),
            look=e["look"],
            **({"choice": choice} if choice else {}),
        )
        for e in eps_plan
    ], why


def per_look(a, args: list, out: Path, plan: dict) -> dict:
    """A new session per look (PER_LOOK_RULE: in one session an earlier look stayed as a 'ghost' in the RTX cache).
    Sessions run in turn with
    a plan copy per episode (episodes = [that episode + index], part_json) and the summaries are merged. Stops when
    the first session finds the original off screen (empty episodes_plan). Head sessions (draw 0): aug =
    per_look=choose (choice + allow) · orig random = measure (off screen + allow) -> _look/allow.json; the drawing
    sessions (fixed) only read that file. Returns the merged render_clip.json (sessions, session_rule). Marker
    [COLLECT_PER_LOOK]."""

    def run(tag: str, pp: dict) -> dict:
        return _part(a, args, out, tag, pp)

    parts, todo = [], plan["episodes"]
    head = "choose" if plan.get("choose") else ("measure" if any(e["look"] != "D" for e in todo or []) else None)
    if head:  # head session: choice and allow measurement (no episode drawn; _look/allow.json)
        g = run("head", dict(plan, per_look=head))
        todo = g.get("episodes_plan") or []
        print(f"{PER_LOOK_MARK} {plan['clip']} head {head} · plan {len(todo)}", flush=True)
        parts.append(("head", g))
    for i, ep in enumerate(todo):
        pp = dict(plan, episodes=[dict(ep, index=i)], per_look="fixed")
        pp.pop("choose", None)
        g = RD.redraw_after(run, out, pp, f"part{i}", run(f"part{i}", pp))
        print(
            f"{PER_LOOK_MARK} {plan['clip']} session {i + 1}/{len(todo)} · {ep['traj']}/{ep['look']} · per_look "
            f"fixed · drawn {g.get('n')}/{g.get('m')}",
            flush=True,
        )
        parts.append((i, g))
        if not g.get("episodes_plan"):
            break
    if not parts:
        raise SystemExit(
            f"{MARK} {plan['clip']}: a new session per look but 0 sessions (plan {len(plan['episodes'] or [])})"
        )
    first = parts[0][1]
    eps = [dict(e, index=i) for i, g in parts for e in g["episodes"]]
    kept = all(g.get("episodes_plan") for _, g in parts)
    m = dict(
        first,
        episodes=eps,
        prim_oracle=[x for _, g in parts for x in g["prim_oracle"]],
        trajs=[t for _, g in parts for t in g["trajs"]],
        looks=next((g["looks"] for _, g in parts if g.get("looks")), None),
        camera_off=parts[-1][1]["camera_off"],
        episodes_plan=todo if kept else [],
        n=sum(g["n"] for _, g in parts),
        total_s=round(sum(g["total_s"] for _, g in parts), 1),
        maxrss_gb=max(g["maxrss_gb"] for _, g in parts),
        sessions=len(parts) + sum(g.get("redraw_sessions", 0) for _, g in parts),
        session_rule=PER_LOOK_RULE,
        redraw=[r for _, g in parts for r in g.get("redraw") or []],
    )
    m.update(m=len(m["episodes_plan"]), ok=all(g["ok"] for _, g in parts) and m["n"] == len(m["episodes_plan"]))
    return m


def _part(a, args: list, out: Path, tag: str, pp: dict) -> dict:
    """One session: plan copy plan.<tag>.json · log session.<tag>.log · summary render_clip.<tag>.json (returned)."""
    pp = dict(pp, part_json=f"render_clip.{tag}.json")
    (out / f"plan.{tag}.json").write_text(json.dumps(pp, indent=1, ensure_ascii=False))
    CLA.session(
        a.tree,
        a.gpu,
        args + ["--plan", str(out / f"plan.{tag}.json")],
        out / f"session.{tag}.log",
        "[RENDER_CLIP] DONE",
    )
    return json.loads((out / pp["part_json"]).read_text())


def chosen_plan(plan: dict, got: dict) -> tuple[list[dict], dict | None]:
    """Session summary (render_clip.json) -> (drawn episode plan, trajectory choice record | None). For a choice plan
    the record is recomputed with the rule (record.episodes.choose_rows) and row 0, the choice and the episode plan
    must match (an independent oracle; a mismatch dies). The orig round needs the original off-screen record
    (orig_fov; proof that the rule ran) and must equal the plan recomputed from its visibility value."""
    if not plan.get("choose"):
        of = got.get("orig_fov")
        if plan["round"] != "orig":
            return plan["episodes"], None
        if not of or of.get("rule") != EPI.ORIG_FOV_RULE:
            raise InputMissingError(
                f"{MARK} {plan['clip']}: the orig session summary has no original off-screen "
                f"record (orig_fov · {EPI.ORIG_FOV_RULE}): the rule did not run"
            )
        kept = EPI.kin_visible(plan, EPI.orig_kept((of.get("ego_view") or {}).get("grasp_visible")))
        want = plan["episodes"] if kept else []
        if kept != of.get("orig_kept") or got.get("episodes_plan") != want:
            raise SystemExit(
                f"{MARK} {plan['clip']}: original off-screen record != rule: visible {kept} · record "
                f"{of.get('orig_kept')} · plan {len(got.get('episodes_plan') or [])} != {len(want)}"
            )
        return want, None
    ch, eps = got.get("choice"), got.get("episodes_plan")
    if not ch or eps is None:
        raise InputMissingError(
            f"{MARK} {plan['clip']}: a choice plan but the session summary has no choice or episodes_plan"
        )
    r0, re, _m = EPI.choose_rows(ch["cands"], int(plan["choose"]["n"]))
    names = [t["name"] for t in plan["trajs"]]
    if (
        re != ch["chosen"]
        or r0 != ch.get("row0")
        or EPI.aug_episodes(names[0], r0, re, int(plan["deck"]["n_rows"])) != eps
        or [c["name"] for c in ch["cands"]] + list(ch.get("unprobed", [])) != names
    ):
        raise SystemExit(
            f"{MARK} {plan['clip']}: choice record != rule: record {ch['chosen']} · row 0 {ch.get('row0')}"
            f" · recomputed {re} · row 0 {r0} · candidates {[c['name'] for c in ch['cands']]} + "
            f"unprobed {ch.get('unprobed')} != plan {names}"
        )
    keep = ("rule", "n", "chosen", "orig_visible", "visible_chosen", "cands", "unprobed", "row0", "orig_fov")
    return eps, {k: ch.get(k) for k in keep}
