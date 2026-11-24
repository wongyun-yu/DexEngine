"""Owner of the record episode plan: rounds (ROUNDS; one explicit input, no default) x trajectories x looks.
Order = original (orig, per hand) -> placement augment (aug, same hand order). The text track is not drawn separately:
frames are taken from the look-D episodes of the orig pieces (text.sources dataset_index; the same scene is not drawn
twice).
  orig  original trajectory x 5 looks (D + 4 random for that trajectory)   - deck row 0
  aug   placement trajectories <= PER_CLIP x 5 looks                      - deck rows 1 … PER_CLIP (chosen order) ·
        placement candidate judging · augment physics
A clip's look cards come from **one deck independent of the round** (deck_args: rows = 1 + PER_CLIP · columns = rows x
RANDOM_PER_TRAJ; the shape does not change with trajectory count or round): the random looks of row k = columns
4k … 4k+3 (dealt cards, no overlap) · name r<row>_s<col> (look_name). So the (trajectory, card) pairs of orig ∪ aug
equal the 20 of one session drawing the original + PER_CLIP variants (card names, specs, place seeds). The plan JSON's
deck field carries the deck shape to the render session (this module uses the standard library only; PER_CLIP's owner
augment.policy is read by render.clip, which builds the plan).

The render session (render.clip_session) draws the plan JSON's episodes in this order: look D first for every
trajectory (light settles once), then each trajectory's random looks. The allowed list of random look specs depends
on the scene, so they are drawn inside the session with looks.augvis.sample.sample_set(allow, key, **plan.deck).
Trajectory choice (choose, aug round only): from the original camera angle the render session measures, per variant
candidate, the grasp-moment visibility (render.framing ego_view vis_strict; single owner of the criterion
render.framing.VIS_RULE, strict) and chooses PER_CLIP: visible variants (candidate order) -> invisible variants. The
original's exclusion trigger is separate: grasp_visible (render.framing.ORIG_OUT_RULE).
Original off-screen rule (ORIG_FOV_RULE: of the clip's 4 trajectories = original + 3 variants, an off-screen original
is dropped and all 4 become in-view placement variants · trigger = ORIG_OUT_RULE · variant visibility = VIS_RULE):
  orig round  after calibrate the session measures the original's grasp visibility (grasp_visible): visible -> the
              original x 5 looks as before (plan unchanged) · invisible -> the render is skipped and only the record
              (render_clip.json orig_fov, marker [ORIG_FOV], an exclusion with a reason round_excluded) is kept.
  aug round   the session measures the original again by the same criterion (same original, same camera rule -> the
              same answer; no state between rounds). A visible original -> rows 1-3 as before (choose). Invisible ->
              rows 0-3 (choose_rows) = the first 1 + n of the FILL_RULE order: row 0 = the first slot · trajectories
              = as many as passing variants (judge cap; no original filler) · with 0 variants row 0 = the original.
  Off-screen original episodes an old orig round already drew are dropped as 'replaced' when that clip's aug result
  (row 0) is merged (the merge's job).
"""

from __future__ import annotations

import hashlib
import json

# look D name; value owner looks.look_d.LOOK (standard library imports only)
from dexengine.record.looks.look_d import LOOK as LOOK_D

# 2 = round and deck shape fields (1 = original + variants in one session; the deck followed the trajectory count)
PLAN_VERSION = 2
RANDOM_PER_TRAJ = 4  # random looks per trajectory (5 looks = D 1 + 4)
#: episodes one render session draws: a new session per look (render.clip PER_LOOK_RULE; RTX cache ghosts within a
#: session). Readers: the lane time model (sessions = episodes / this value) · session entry.
EPISODES_PER_SESSION = 1


def episodes_per_session(rule: str | None) -> int:
    """The plan's session_rule -> episodes one render session draws (only None: a new session per look)."""
    if rule is None:
        return EPISODES_PER_SESSION
    raise ValueError(f"session_rule={rule!r} — only None")


CHOOSE_RULE = "grasp_visible_first_aug_only"
#: E5 kinematic baseline (experiments.policies.kin_data.collect): no choice by visibility (off-screen original skip and
#: variant visibility choice both off; measured values are still recorded). Switched on by plan(kin=True) only (plan
#: body field kin; the render cache fingerprint differs) · default off = same bytes.
KIN_RULE = "e5_kin_no_visibility_select"


def kin_visible(plan: dict, measured):
    """Visibility decision value: true for a kinematic plan (plan.kin; no choice), else the measured value as is
    (canonical)."""
    return True if plan.get("kin") else measured


KIN_ALLVAR_MARK = "[KIN_ALLVAR]"


def kin_n_choose(per_clip: int, n_given: int) -> int:
    """Trajectory count of a kinematic aug plan (= deck variant rows): no choice (KIN_RULE), so every variant given.
    Up to PER_CLIP it stays PER_CLIP (same plan bytes, finished pieces, render cache valid). More = clips whose final
    data holds PER_CLIP + 1 variants through the off-screen original row-0 fill (ORIG_FOV_RULE): cutting at PER_CLIP
    silently dropped the last-ranked variant as unprobed (maug_hrdexdb_05: 45 != 50)."""
    return max(int(per_clip), int(n_given))


# off-screen original rule (module doc): the name the orig skip and the aug row-0 choice store
ORIG_FOV_RULE = "orig_fov_rows0_3"
ORIG_FOV_MARK = "[ORIG_FOV]"
# head of the orig skip reason (data); the case gate reads it as handed to augment
ORIG_FOV_WHY = "original off-screen → aug rows 0–3"
#: row 0-3 fill of originally off-screen clips (in-view variants 4 instead of the original + the visibility ladder):
#: of the levels (render.fov.VIS_LADDER, strictest first L0 = VIS_RULE … L3), the passing variants of the strictest
#: level with at least min(rows 1 + n, variant count) passing variants (stricter passes first -> higher visibility
#: score -> candidate order) -> when no level has enough, the bottom (render.fov.VIS_LADDER_BOTTOM) = top by score.
#: With too few variants the rows shrink to match (no original filler) · 0 variants = row 0 original (level = bottom).
#: Level and score are kept in orig_fov.ladder and per candidate in ego_view.vis_ladder.
#: A threshold of the row count alone would write the bottom even when variants 1-3 are all visible, mixing 'too
#: little visibility' with 'too few variants' -> min.
#: FOV_LADDER = the name of this rule (orig_fov.fill · _minvar = that threshold · _noorig = no original filler); no code
#: reads the value as text (render.clip_session prints it on the session line).
FOV_LADDER = "vis_ladder_L0_L3_score_top_minvar_noorig"
FILL_RULE = FOV_LADDER
# looks to draw: unset = all · D = look D only (random after the new-session path) · else dies
LOOKS_ENV = "DEXENGINE_COLLECT_LOOKS"
LOOKS_MARK = "[COLLECT_LOOKS]"


def looks_mode() -> str:
    """Look range to draw: '' = all · 'D' = look D only (orig round only, marker [COLLECT_LOOKS]). Unknown values
    die."""
    import os

    v = os.environ.get("DEXENGINE_COLLECT_LOOKS", "")  # = LOOKS_ENV (the place the registry check reads)
    if v not in ("", LOOK_D):  # no random-only branch: D and the 4 random are one job with the same seed
        raise ValueError(
            f"{LOOKS_ENV}={v!r}: unset (all) or {LOOK_D!r} (look D only); no random-only branch (seed rule)"
        )
    return v


def look_kept(look: str, only: str) -> bool:
    """Does the look range only (looks_mode) keep the look? '' = all · D = look D only. Home of the rule: the plan's
    episode filter, the spec expect (expect_of) and the job env read it together."""
    if only not in ("", LOOK_D):
        raise ValueError(f"look range {only!r}: '' or {LOOK_D!r} only (seed rule)")
    return not only or look == LOOK_D


def looks_of(looks: list[str]) -> str:  # noqa: V103 — used by a deployment's render lane
    """Spec looks (list of look names to draw) -> the looks_mode value of that range: look D only = D · both = ''.
    Random without look D dies (seed rule: the 4 random share the job and seed with D)."""
    has_d = LOOK_D in looks
    if looks and not has_d:
        raise ValueError(f"spec looks {looks} lack look {LOOK_D}: no random-only branch (seed rule)")
    return "" if not looks or len(looks) > 1 else LOOK_D


ROUNDS = ("orig", "aug")  # rounds: one explicit input (driver --round; no default)
ORIG_ROW = 0  # deck row of the original trajectory (variants = 1 … PER_CLIP, chosen order)


def check_round(rnd: str) -> str:
    if rnd not in ROUNDS:
        raise ValueError(f"round {rnd!r}: one of {ROUNDS} (no default)")
    return rnd


def aug_on(rnd: str) -> bool:
    """Does the round run the placement augment physics (candidate generation, rollout, post gate, choice)? aug
    only."""
    return check_round(rnd) == "aug"


def deck_args(per_clip: int) -> dict:
    """Clip deck shape: rows = original 1 + variants per_clip · columns = rows x RANDOM_PER_TRAJ. Independent of round
    and trajectory count (module doc)."""
    if int(per_clip) < 0:
        raise ValueError(f"per_clip={per_clip} < 0")
    n_rows = 1 + int(per_clip)
    return {"n_looks": n_rows * RANDOM_PER_TRAJ, "n_rows": n_rows}


def look_key(clip: str) -> str:
    """Key for dealing random look cards: the clip base name alone (the same clip gets the same looks on any machine
    and round)."""
    return f"collect_look:{clip}"


def look_name(row: int, col: int, n_rows: int) -> str:
    """Random look slot name: row (trajectory) row · column col (from 0) · row count n_rows. One row: s<col> · several:
    r<row>_s<col>. Single owner of the name: looks.augvis.sample.sample_set names specs with this function and the
    render session checks both. The record deck always has several rows (deck_args: 1 + PER_CLIP), so r<row>_s<col>;
    the one-row form is the visual augment pilot."""
    return f"s{col + 1:02d}" if n_rows == 1 else f"r{row}_s{col + 1:02d}"


def _rows(rnd: str, orig: str, augs: list[str], row0: str | None = None) -> list[tuple[str, int]]:
    """augs = trajectories of rows 1… (aug round). The original may stand in row 0 or rows 1… only under the
    off-screen original rule (with row0)."""
    if not orig:
        raise ValueError("no original trajectory")
    slots = ([row0] if row0 is not None else []) + list(augs)
    if len(set(slots)) != len(slots):
        raise ValueError(f"trajectory names overlap {slots}")
    if rnd == "orig":
        if augs or row0 is not None:
            raise ValueError(f"the {rnd} round is original only: variants {augs} · row 0 {row0}")
        return [(orig, ORIG_ROW)]
    if orig in augs:  # the original stands only in row 0 (off-screen, 0 variants) (judge cap)
        raise ValueError(f"original {orig} in aug rows 1…")
    if not slots:
        raise ValueError("the aug round has no chosen variant")
    # off-screen original (ORIG_FOV_RULE): the aug round draws row 0 too
    head = [] if row0 is None else [(row0, ORIG_ROW)]
    return head + [(t, ORIG_ROW + 1 + k) for k, t in enumerate(augs)]


def episodes(rnd: str, orig: str, augs: list[str], n_rows: int, row0: str | None = None) -> list[dict]:
    """Round rnd · original orig · chosen variants augs (aug round only, chosen order) · deck row count n_rows ->
    [{traj, look, row, col}] in render order (look D first). Rows are fixed by role (original row 0, variants 1 …);
    the round does not shift them. row0 (aug round only, choose_rows) = the row-0 trajectory of a clip whose original
    is off-screen (a visible variant, else the original); None = row 0 belongs to the orig round (as now)."""
    rows = _rows(check_round(rnd), orig, list(augs), row0)
    if max(r for _, r in rows) >= n_rows:
        raise ValueError(f"more trajectories than deck rows {n_rows}: {rows}")
    return [{"traj": t, "look": LOOK_D, "row": r, "col": None} for t, r in rows] + [
        {"traj": t, "look": look_name(r, RANDOM_PER_TRAJ * r + j, n_rows), "row": r, "col": RANDOM_PER_TRAJ * r + j}
        for t, r in rows
        for j in range(RANDOM_PER_TRAJ)
    ]


def aug_episodes(orig: str, row0: str | None, chosen: list[str], n_rows: int) -> list[dict]:
    """Episodes the aug round draws (choose_rows' answer); empty when neither row 0 nor variants exist (visible
    original, 0 variant candidates)."""
    return episodes("aug", orig, chosen, n_rows, row0) if (row0 is not None or chosen) else []


def orig_kept(orig_visible) -> bool:
    """Is the original row (row 0) drawn as the original = the original's grasp visibility (value of
    render.framing.VIS_RULE)? A non-bool dies. The orig round's skip and the aug round's row-0 choice read this one
    function."""
    if not isinstance(orig_visible, bool):
        raise ValueError(f"original grasp-moment visibility is not a bool {orig_visible!r}")
    return orig_visible


def choose(cands: list[dict], n: int) -> tuple[list[str], dict]:
    """Trajectory choice (module doc, aug round): cands = [{name, role(orig|aug), rank (candidate order, original 0),
    grasp_visible (bool, original exclusion trigger), vis_strict (bool, variant visibility VIS_RULE)}] -> (chosen
    variant names <= n, record). The original is recorded only (not chosen). Not exactly one original, overlapping
    names or non-bool visibility values die."""
    if n < 1:
        raise ValueError(f"n={n} < 1")
    names = [c["name"] for c in cands]
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names overlap {names}")
    if sum(c.get("role") == "orig" for c in cands) != 1:
        raise ValueError(f"not exactly one original {[(c['name'], c.get('role')) for c in cands]}")
    bad = [
        c["name"]
        for c in cands
        if not (isinstance(c.get("grasp_visible"), bool) and isinstance(c.get("vis_strict"), bool))
    ]
    if bad:
        raise ValueError(f"grasp-moment visibility values missing {bad}")
    order = sorted((c for c in cands if c["role"] != "orig"), key=lambda c: (not c["vis_strict"], int(c["rank"])))
    chosen = [c["name"] for c in order[:n]]
    orig = next(c for c in cands if c["role"] == "orig")
    return chosen, {
        "rule": CHOOSE_RULE,
        "n": n,
        "chosen": chosen,
        "orig_visible": orig["grasp_visible"],
        "visible_chosen": sum(c["vis_strict"] for c in cands if c["name"] in chosen),
        "cands": [{k: c.get(k) for k in ("name", "role", "rank", "grasp_visible", "vis_strict")} for c in cands],
    }


def ladder_order(orig: dict, var: list[dict], need: int) -> tuple[list[str], str, dict]:
    """Row fill order of an originally off-screen clip (FOV_LADDER, single owner): var = variant candidates (candidate
    order) -> (order names (variants only; [original] with 0 variants), chosen level, scores {name: score}). Every
    candidate needs vis_ladder {pass: {level: bool}, score: 0..1}, and L0 passing must equal vis_strict (the same
    computation; a difference means the measurements split, dies).
    need = row count (1 + n) · level threshold = min(need, variant count) (module comment FOV_LADDER)."""
    from dexengine.record.render import fov as CL

    names = [lv["name"] for lv in CL.VIS_LADDER]
    bad = []
    for c in var:
        L = c.get("vis_ladder")
        sc = (L or {}).get("score")
        if (
            not isinstance(L, dict)
            or set(L.get("pass") or {}) != set(names)
            or isinstance(sc, bool)
            or not isinstance(sc, (int, float))
            or not 0.0 <= float(sc) <= 1.0
        ):
            bad.append(f"{c.get('name')}: ladder value missing or out of form")
        elif bool(L["pass"][names[0]]) != bool(c["vis_strict"]):
            bad.append(f"{c.get('name')}: {names[0]} {L['pass'][names[0]]} ≠ vis_strict {c['vis_strict']}")
    if bad:
        raise ValueError(f"the visibility ladder cannot be used {bad[:3]}")

    def strict_of(c):  # strictest level passed (bottom when none); strictest first within rows too
        return next((i for i, lv in enumerate(names) if c["vis_ladder"]["pass"][lv]), len(names))

    by = sorted(var, key=lambda c: (strict_of(c), -float(c["vis_ladder"]["score"]), int(c["rank"])))
    scores = {c["name"]: float(c["vis_ladder"]["score"]) for c in var}
    k = min(need, len(var))  # 0 = no variants -> bottom
    for lv in names:
        ok = [c["name"] for c in by if c["vis_ladder"]["pass"][lv]]
        if k and len(ok) >= k:
            return ok + [c["name"] for c in by if c["name"] not in ok], lv, scores
    return ([c["name"] for c in by] or [orig["name"]]), CL.VIS_LADDER_BOTTOM, scores


def choose_rows(cands: list[dict], n: int) -> tuple[str | None, list[str], dict]:
    """aug-round choice + off-screen original rule (module doc, ORIG_FOV_RULE) -> (row-0 trajectory | None, rows 1…
    trajectories, record). A visible original -> (None, choose(cands, n)) as before. Invisible -> the first 1 + n of
    the ladder_order (FILL_RULE): row 0 = the first slot · rows 1… = the other variants (no original filler; 0 variants
    = row 0 original). Record = the choose record (chosen = rows 1…) + row0 +
    orig_fov {rule, fill, orig_kept, row0, row0_source, row0_fill, ladder{level, rows}, orig_vis, visible_*, rows}."""
    chosen, meta = choose(cands, n)  # value checks (name overlap, one original, bool) are choose's
    # every measured candidate (rechecked by render.clip and the pilot check; ladder values too)
    full = [
        dict(c0, vis_in_frame=c.get("vis_in_frame"), vis_ladder=c.get("vis_ladder"))
        for c0, c in zip(meta["cands"], cands, strict=True)
    ]
    orig = next(c for c in cands if c["role"] == "orig")
    var = sorted((c for c in cands if c["role"] != "orig"), key=lambda c: int(c["rank"]))
    vis = [c for c in var if c["vis_strict"]]
    row0, fill, ladder, sc = None, None, None, {}
    if not orig_kept(orig["grasp_visible"]):
        order, level, sc = ladder_order(orig, var, 1 + n)
        row0, chosen = order[0], order[1 : 1 + n]
        fill = "orig" if row0 == orig["name"] else level
        ladder = {"level": level, "rows": {x: sc.get(x) for x in [row0, *chosen]}}
        meta = dict(meta, chosen=chosen, visible_chosen=sum(c["vis_strict"] for c in cands if c["name"] in chosen))
    # the clip's four trajectories (rows 0, 1 …): row 0 = original (orig round's) | row0
    slots = [row0 or orig["name"]] + chosen
    meta = dict(
        meta,
        cands=full,
        row0=row0,
        orig_fov={
            "rule": ORIG_FOV_RULE,
            "fill": FILL_RULE,
            "orig_kept": row0 is None,
            "row0": row0,
            "row0_source": None if row0 is None else ("orig" if row0 == orig["name"] else "variant"),
            "row0_fill": fill,
            "ladder": ladder,
            "orig_vis": orig.get("vis_in_frame") if row0 is not None else None,
            "visible_variants": len(vis),
            "visible_rows": sum(bool(next(c for c in cands if c["name"] == x)["vis_strict"]) for x in slots),
            "rows": len(slots),
        },
    )
    return row0, chosen, meta


def plan(
    clip: str,
    out: str,
    seq_dir: str,
    seq_list: str,
    trajs: list[dict],
    rnd: str,
    deck: dict,
    n_choose: int | None = None,
    orig_fov_owner: bool | None = None,
    kin: bool = False,
) -> dict:
    """Plan JSON (v2): trajs = [{name, traj, role}] (original first, role orig|aug) · deck = deck_args(PER_CLIP).
    The aug round requires n_choose (= PER_CLIP); episodes is None (the render session fixes episodes after choose).
    orig is the original only (one traj); episodes are fixed here."""
    check_round(rnd)
    if not trajs or trajs[0].get("role") != "orig":
        raise ValueError(f"{clip}: the original trajectory is not first")
    if aug_on(rnd) != bool(n_choose):
        raise ValueError(
            f"{clip}: round {rnd} with n_choose={n_choose}: choosing is aug-round only, and aug requires it"
        )
    # the aug round builds a plan even with 0 variant candidates: when the session measures the original off-screen,
    # row 0 = the original is drawn (ORIG_FOV_RULE)
    if not aug_on(rnd) and len(trajs) != 1:
        raise ValueError(f"{clip}: the {rnd} round takes one original; got {[t['name'] for t in trajs]}")
    if orig_fov_owner is not None and (not n_choose or orig_fov_owner is not False):  # collect_fov_owner
        raise ValueError(f"{clip}: orig_fov_owner={orig_fov_owner!r} takes only False (off-screen) in the aug round")
    eps = None if n_choose else episodes(rnd, trajs[0]["name"], [], int(deck["n_rows"]))
    only = looks_mode()
    if only and n_choose:
        raise ValueError(f"{clip}: {LOOKS_ENV}={only} is for the orig round only (aug after the new-session path)")
    if only:
        eps = [e for e in eps if look_kept(e["look"], only)]
        print(f"{LOOKS_MARK} {clip}: {LOOKS_ENV}={only}: episodes {len(eps)}", flush=True)
    body = dict(
        version=PLAN_VERSION,
        clip=clip,
        out=out,
        seq_dir=seq_dir,
        seq_list=seq_list,
        trajs=trajs,
        round=rnd,
        deck=dict(deck),
        look_key=look_key(clip),
        random_per_traj=0 if only == LOOK_D else RANDOM_PER_TRAJ,
        episodes=eps,
        **({"looks": only} if only else {}),
        **(
            {
                "choose": dict(
                    n=int(n_choose),
                    rule=CHOOSE_RULE,
                    fill=FILL_RULE,
                    **({} if orig_fov_owner is None else {"orig_fov_owner": orig_fov_owner}),
                )
            }
            if n_choose
            else {}
        ),
    )
    from dexengine.record.render import cameras as CC

    if kin:  # kinematic: visibility choice off (KIN_RULE)
        body["kin"] = {"rule": KIN_RULE}
    # exo off: the drawn videos differ (cache fingerprint; default on = no field)
    if not CC.exo_on():
        body["videos"] = list(CC.videos())
    body["plan_sha"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "out"}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return body


def allow_traj(plan: dict) -> str:
    """Trajectory whose random look allow list is measured: the original regardless of round (deck row 0, first of
    plan.trajs). In a one-session 20 the first random episode was the original, so orig and aug get the same allow ->
    the same cards (the render session's looks.look_run follows)."""
    t = plan["trajs"][0]
    if t.get("role") != "orig":
        raise ValueError(f"the original measured for allow is not first in plan.trajs {t.get('name')}")
    return t["name"]


def check_plan(d: dict) -> None:
    """Version, round and deck shape of the plan JSON (called by the render session entry render.clip_session); a
    mismatch raises ValueError."""
    if d.get("version") != PLAN_VERSION:
        raise ValueError(f"version {d.get('version')!r} != {PLAN_VERSION}")
    check_round(d.get("round"))
    if not isinstance(d.get("deck"), dict) or sorted(d["deck"]) != ["n_looks", "n_rows"]:
        raise ValueError(f"deck shape {d.get('deck')!r}: deck_args form (n_looks, n_rows)")


def expected(rnd: str, n_trajs: int) -> int:  # noqa: V103 — public API
    check_round(rnd)
    only = looks_mode()
    return n_trajs * {"": 1 + RANDOM_PER_TRAJ, LOOK_D: 1}[only]
