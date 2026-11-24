"""Owner of the canonical record chain's switches and of the stage that follows the base store; called by
record.bundle.
External API: canon_class has no caller in this tree; a collect dashboard plugin calls it by name on the collect
tree it reads.

Switches (meta/collect.json switches): force_partner (rollout.recorder), head_view (render.visibility) and
  extras_overhang (scene.extras_rules) are fixed on; aug_placement (the placement augment stage, augment.stage on,
  announce, run) is set by the round.
Stage-2 fingerprint tail (render_tag): '+head_view' (the head view changes the render outputs).
Time (extra_s): with augment on, augment.stage.TIMEOUT_PER_CLIP_S per clip is added to the bundle cap.
Round (apply_round, record.episodes.ROUNDS): the round sets the placement augment switch (aug = on · orig = off; no
  two inputs with the same meaning). An explicit env DEXENGINE_COLLECT_AUG that disagrees with the round dies;
  otherwise the round's value is set and inherited by the render sessions and augment children (marker
  [COLLECT_ROUND] round=<round> · aug=<0|1>).
Following stage (after_store, augment on): per cell the base-stored clips (items whose checkpoint is that cell's) go
  to augment.stage_run.run. Target clips (cell table target rows) without a base store (0 successes, generation drop,
  below the threshold; owner of the reason rule base_exclusions, which the driver check driver_check.verify also
  reads) leave the augment and get one `[AUG_COLLECT] skipped <clip> <reason>` line each + a count line (a clip without
  a reason fails). A failure (SystemExit or exception of the augment stage) prints the AUG_FAIL line and returns
  false; record.bundle ends with RC_AUG_FAILED (the old form let the SystemExit leave as rc 1, colliding with 'some
  clips excluded', and skipped the augment without a marker when any clip had 0 successes).
"""

from __future__ import annotations

import os
import re
import traceback

from dexengine.record import episodes as EPI  # owner of the rounds (ROUNDS, aug_on)
from dexengine.record.augment import stage as AUGC
from dexengine.record.augment import stage_run as ASR
from dexengine.record.augment import xr_deps as XD

AUG_FAIL = f"{AUGC.MARK} FAIL"  # augment failure line (the driver looks for it)
#: clip hold: one clip's defect (hold_why -> (kind augment.policy.HOLD_*, reason)) must not kill the whole bundle, so
#: the augment is called again without that clip (finished clips are cached and do not run again) · re-handing = that
#: clip only. One line form `<CANON_HOLD> <clip> <kind> <reason>` (read by held_of) · counts.json held{clip: {kind,
#: why}} · skipped{clip: '<kind> <reason>'}. When every clip is held it is a job defect, not a clip defect: the
#: HOLD_ALL line is printed and the bundle dies as failed (no silent success).
#: The hold count line has another head (HOLD_SUM): two forms under one head would make held_of read the count line as
#: a clip.
CANON_HOLD, HOLD_ALL, HOLD_SUM = "[AUG_CANON_HOLD]", "[AUG_CANON_HOLD_ALL]", "[AUG_CANON_HOLD_SUM]"
ROUND_MARK = "[COLLECT_ROUND]"


def hold_line(clip: str, kind: str, why: str) -> str:
    """One hold line (writer side, after_store). The form is owned by this function and parse_hold as a pair
    (readers: held_of and dashboard plugins)."""
    from dexengine.record.augment import policy as AP

    if kind not in AP.HOLD_KINDS:
        raise SystemExit(f"{CANON_HOLD} {clip}: unknown hold kind {kind!r}: {AP.HOLD_KINDS}")
    return f"{CANON_HOLD} {clip} {kind} {why[:200]}"


def kind_why(body: str) -> tuple[str, str] | None:
    """'<kind> <reason>' -> (kind, reason): the hold line body and the counts.json skipped values share this form.
    The old form without a kind field (older trees; then the only hold was the canonical cell technical
    failure, and the reason head was an augment.stage.CANON_TECH word) -> (HOLD_CANON_TECH, body). Neither -> None
    (not a hold; the caller decides whether to die)."""
    from dexengine.record.augment import policy as AP

    kinds = sorted(AP.HOLD_KINDS, key=len, reverse=True)
    kind = next((k for k in kinds if body == k or body.startswith(k + " ")), None)
    if kind is not None:
        return kind, body[len(kind) :].strip()
    if body.split(" ", 1)[0] in AUGC.CANON_TECH:
        return AP.HOLD_CANON_TECH, body.strip()
    return None


#: hold count line form of older trees (head CANON_HOLD, body 'hold <n> clips' after the legacy-value
#: migration); that line now uses the HOLD_SUM head. The loop rereading old job records read
#: this line as a hold line and died. Only this form is 'not a hold line'; any other unknown form still dies (a rule,
#: not a name table).
OLD_HOLD_SUM_RE = re.compile(re.escape(CANON_HOLD) + r" hold \d+ clips")


def parse_hold(line: str) -> tuple[str, str, str] | None:
    """Hold line -> (clip, kind, reason) · None if not a hold line. A hold-line head whose body is not of the kind_why
    form, or has no clip, dies (an unknown form is never read silently as 'not a hold'). The older death text
    '✗ <clip>: <canonical cell (pv00)> …' is not a hold line (None)."""
    from dexengine.record.augment import policy as AP

    if not line.startswith(CANON_HOLD + " "):
        return None
    # hold count line of older job records (CANON_HOLD head then, HOLD_SUM now)
    if OLD_HOLD_SUM_RE.fullmatch(line):
        return None
    clip, _, body = line[len(CANON_HOLD) + 1 :].partition(" ")
    kw = kind_why(body)
    if not clip or kw is None:
        raise SystemExit(f"{CANON_HOLD} unknown hold line form {line[:160]!r}: kinds are {AP.HOLD_KINDS}")
    return clip, *kw


def holds_of(run_log: str) -> list[tuple[str, str, str]]:
    """Every hold line of a bundle run record: parse_hold as it is (line order)."""
    return [h for ln in run_log.splitlines() if (h := parse_hold(ln))]


#: head of the 'canonical cell bake seed missing' (input missing) reason; written by augment.bake_run.bake_all as why
#: 'bake' + detail. That file is covered by the aug code fingerprint, so no function lives there
#: (editing it would change the cache key). Data string: matched against stored reasons; it stays byte-equal to the
#: bake_run.py writer until both move together.
SEED_MISSING = "bake canonical cell bake seed missing"
#: read-only forms of older trees (when the bake reproduction check cut pv00: three reason heads; a dashboard
#: plugin still reads hold lines and counts.json of older jobs). New trees do not write these forms (the
#: reproduction check only records). The 'canon_oracle' in augment.stage.CANON_TECH exists for the same reason (old
#: lines without a kind field, old generation records read by canon_tech_fail). Once no older job or piece is
#: read, the three, the canon_class branch and the 'canon_oracle' in augment.stage.CANON_TECH can be removed
#: together.
ORACLE_DIFF = "canon_oracle max diff "
ORACLE_LEN = "canon_oracle T "
ORACLE_SEED = "canon_oracle bake seed missing"
#: canon_class branches: 5090 again (failed the reproduction check) · input missing (not an oracle verdict)
CANON_5090, CANON_INPUT = "5090", "input_missing"


def canon_class(kind: str, why: str) -> tuple[str, str] | None:  # noqa: V103 — public API for dashboard plugins
    """Hold (kind, reason) -> canonical oracle class (shared by a dashboard's log path and the piece path).
    Rule: only clips held by the oracle's reproduction check on a GPU other than the canonical one (RTX 5090) go to a
    machine with the canonical GPU:
      max difference > tolerance -> (CANON_5090, difference text) · length differs (reproduced trajectory length !=
      canonical) -> (CANON_5090, '-') · bake seed missing = input missing -> (CANON_INPUT, reason) (new form
      SEED_MISSING, old form ORACLE_SEED; not re-sent; the caller adds the reasoned hold and the alarm) · not a
      canonical oracle (another kind, another bake, length) -> None.
    The max-difference and length branches read older records only (see the comment above ORACLE_DIFF).
    Unknown kinds, canon_oracle reasons of unknown form, or a non-number after 'max difference' die."""
    from dexengine.record.augment import policy as AP

    if kind not in AP.HOLD_KINDS:
        raise SystemExit(f"{CANON_HOLD} unknown hold kind {kind!r}: {AP.HOLD_KINDS}")
    if kind == AP.HOLD_CANON_TECH and why.startswith(SEED_MISSING):
        return CANON_INPUT, why
    if kind != AP.HOLD_CANON_TECH or why.split(" ", 1)[0] != "canon_oracle":
        return None
    if why.startswith(ORACLE_DIFF):
        v = why[len(ORACLE_DIFF) :].split(" ", 1)[0]
        try:
            float(v)
        except ValueError:
            raise SystemExit(f"{CANON_HOLD} canonical oracle max difference is not a number {why[:120]!r}") from None
        return CANON_5090, v
    if why.startswith(ORACLE_LEN):
        return CANON_5090, "-"
    if why.startswith(ORACLE_SEED):
        return CANON_INPUT, why
    raise SystemExit(f"{CANON_HOLD} canonical oracle reason of unknown form {why[:120]!r}")


def held_of(run_log: str) -> set[str]:
    """Held clips of a bundle run record (the clips of holds_of)."""
    return {c for c, _k, _w in holds_of(run_log)}


def round_aug(rnd: str) -> bool:
    """Placement augment switch of round rnd (on for aug only). An explicit env value that disagrees dies; this does
    not set it (driver gate)."""
    try:
        want = EPI.aug_on(rnd)
    except ValueError as e:
        raise SystemExit(f"{ROUND_MARK} {e}") from None
    told = AUGC.explicit()
    if told is not None and told != ("1" if want else "0"):
        raise SystemExit(
            f"{ROUND_MARK} {AUGC.ENV}={told} explicit != round {rnd} (placement augment "
            f"{'on' if want else 'off'}): the round sets the switch; drop the env and rerun"
        )
    return want


def apply_round(rnd: str) -> dict:
    """Set the round's placement augment value in env (switches, the augment stage and child sessions read the same
    value) · one marker line."""
    told, want = AUGC.explicit(), round_aug(rnd)
    os.environ[AUGC.ENV] = "1" if want else "0"
    note = f" · env explicit {told}" if told is not None else ""
    print(f"{ROUND_MARK} round={rnd} · aug={int(want)}{note}", flush=True)
    return {"name": rnd, "aug_placement": want, "aug_env_told": told}


def switches() -> dict:
    """meta/collect.json "switches". force_partner, head_view and extras_overhang are fixed on; only the aug placement
    switch (set by round_switch) varies and appears in explicit."""
    AUGC.announce()
    told = AUGC.explicit()
    return {
        "force_partner": True,
        "head_view": True,
        "aug_placement": AUGC.on(),
        "extras_overhang": True,
        "explicit": {AUGC.ENV: told} if told is not None else {},
    }


def render_tag() -> str:
    """Stage-2 cache key tail (head view on)."""
    return "+head_view"


def extra_s(rnd: str) -> int:
    """Seconds per clip the placement augment stage adds to the bundle time cap (part of the driver's
    --bundle-timeout-per-clip default); only when the round turns the augment on (round_aug)."""
    return AUGC.TIMEOUT_PER_CLIP_S if round_aug(rnd) else 0


def base_exclusions(src: dict, want) -> dict[str, str]:
    """Clips left out of the base store -> reason (from the meta/collect.json source src): 0 successes
    (not_collected) · generation drop (cell generation_dropped) · a target (want) no cell chose = its completion rate
    in that cell's table is below the threshold. Single owner of the reason rule: the augment skip (after_store) and
    the bundle check (driver_check.verify) read it. The reason texts land in counts.json skipped (data)."""
    excl: dict[str, str] = {}
    for c in src.get("not_collected", []):
        excl[c] = "success 0 (rollout " + "·".join(map(str, src.get("rollout_levels", []))) + ")"
    cells = src.get("cells", [])
    # clips the cells chose (owner entry: each cell its own)
    picked = {c for cell in cells for c in cell.get("clips", [])}
    for cell in cells:
        for c, why in (cell.get("generation_dropped") or {}).items():
            excl[c] = f"generation dropped: {why[:100]}"
    # clips the round has nothing to draw for (aug: 0 variants)
    for c, why in (src.get("round_excluded") or {}).items():
        excl.setdefault(c, why)
    # a target no cell chose = completion rate of its clip record below the threshold (one table row is the reason)
    for cell in cells:
        for row in cell.get("clip_rates", []):
            c = row["clip"]
            if c in want and c not in picked and c not in excl and row["rate"] < src.get("min_rate", 0):
                excl[c] = (
                    f"completion rate {row['rate']:.3f} < threshold {src.get('min_rate')} "
                    f"({cell.get('bundle')} {cell.get('iter')})"
                )
    return excl


def hold_why(gen, name: str) -> tuple[str, str] | None:
    """The clip-only hold (kind, reason) left by generation (aug/clips/<key>/<clip>/gen): the judged cell bundle refusal
    record (augment.judged.refused_path; kind = the refusal's kind) · canonical cell technical failure (generation
    record, AUGC.canon_tech_fail) · None. Unknown kinds die."""
    import json

    from dexengine.record.augment import judged as AJ
    from dexengine.record.augment import policy as AP

    rj, vj = AJ.refused_path(gen, name), gen / f"variants_{name}.json"
    if rj.is_file():
        r = json.loads(rj.read_text())
        if r.get("kind") not in AP.HOLD_KINDS:
            raise SystemExit(f"{CANON_HOLD} {rj}: unknown hold kind {r.get('kind')!r}: {AP.HOLD_KINDS}")
        return r["kind"], r["why"]
    t = AUGC.canon_tech_fail(json.loads(vj.read_text())) if vj.is_file() else ""
    return (AP.HOLD_CANON_TECH, t) if t else None


def canon_held(a, per: list, held: dict) -> dict[str, tuple[str, str]]:
    """After the augment died: new clips with a hold (kind, reason) left by generation (hold_why) -> those. None -> {}
    (another death)."""
    from dexengine.core import paths as P

    out, fp = {}, AUGC.code_fp()
    for cell, clips in per:
        try:
            key = AUGC.judge_key(fp, AUGC.judged_bundle(a, cell))
        # an unreadable bundle has no generation record either (the original death is raised); the types are what
        # judged_bundle / judge_key raise (input missing or a refusal = SystemExit, file and record errors)
        # The line carries no exception text: driver_check.reason_of takes the last line with Error/✗/SystemExit as the
        # bundle's failure reason, and this scan runs after the real death.
        except (SystemExit, OSError, ValueError, KeyError):
            print(f"[AUG_HOLD_SCAN] skip cell {cell.get('bundle')}: no judged bundle", flush=True)
            continue
        for c in clips:
            if c not in held and (t := hold_why(a.out / "aug" / "clips" / key / c / "gen", c.removesuffix(P.TREE_SFX))):
                out[c] = t
    return out


def after_store(a, plan: list, items: list, source: dict) -> bool:
    """plan = [(cell, chosen names)] · items = chosen trajectories of the base record -> placement augment physics (when
    on; through the chosen variant trajectory files; render and store are the per-clip render.clip) -> true (done,
    off) · false (failed; the AUG_FAIL line was printed)."""
    if not AUGC.on():
        return True
    try:
        print(XD.xr_preflight(a.gpu), flush=True)  # the bake's machine gate (cross-repair deps, a real kernel compile)
        saved = {it["clip"] for it in items}
        target = {r["clip"] for c in source.get("cells", []) for r in c.get("clip_rates", []) if r.get("target")}
        excl = base_exclusions(source, target)
        skip = {c: excl.get(c, "") for c in sorted(target - saved)}
        nowhy = [c for c, why in skip.items() if not why]
        if nowhy:
            raise SystemExit(f"{AUGC.MARK} {len(nowhy)} target clips with neither a store nor a reason: {nowhy[:3]}")
        for c, why in skip.items():
            print(f"{AUGC.MARK} skipped {c} {why}", flush=True)
        print(f"{AUGC.MARK} skipped {len(skip)} · targets {len(target)} · clips to augment {len(saved)}", flush=True)
        per = [(c, [it["clip"] for it in items if it["cell"]["ckpt"] == str(c["ckpt"])]) for c, _ in plan]
        held: dict[str, tuple[str, str]] = {}
        while True:
            try:
                ASR.run(
                    a,
                    [(c, [x for x in cl if x not in held]) for c, cl in per],
                    skipped={**skip, **{c: f"{k} {w}" for c, (k, w) in held.items()}},
                    held={c: {"kind": k, "why": w} for c, (k, w) in held.items()},
                )
                break
            except SystemExit:
                new = canon_held(a, per, held)
                # a death other than a canonical cell technical failure = bundle failure as before
                if not new:
                    raise
                held.update(new)
                for c, (kind, why) in new.items():
                    print(hold_line(c, kind, why), flush=True)
                if not saved - set(held):
                    msg = f"{HOLD_ALL} held {len(held)}/{len(saved)}: every clip = a job defect (not one clip's defect)"
                    print(msg, flush=True)
                    raise SystemExit(msg) from None
        print(f"{HOLD_SUM} held {len(held)} clips", flush=True)
    # any augment failure becomes the bundle failure marker (never swallowed)
    except (SystemExit, Exception) as e:  # noqa: BLE001
        if not isinstance(e, SystemExit):
            traceback.print_exc()
        print(f"{AUG_FAIL} {type(e).__name__}: {str(e)[:400]}", flush=True)
        return False
    return True
