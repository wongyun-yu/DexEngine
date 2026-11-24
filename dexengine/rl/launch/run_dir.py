"""Per-hand RL recipe ledger: the condition a checkpoint was trained under, kept next to the checkpoint and checked
by every path that loads it again (resume ``--resume_from``/``--resume``, the fresh-process judge FROZEN_REPLAY, the
standalone eval ``--checkpoint``). The recipe switch defaults to on, so without this check a re-judge of an old
checkpoint would silently pick up today's recipe.

Ledger = ``<run folder>/hand_recipe.txt`` (run folder = the parent of ``checkpoints/``, next to coll_preset.txt).
First line = ``hand=<hand> switch=<0|1> source=<default|explicit> override=<field,...|-> <field>=<value|-> ...
finger_penalty_span=<span> finger_penalty_w_scale=<weight scale> finger_penalty_dz_scale=<dead-zone scale>``. Recipe
fields hold the values the training launch declared (argv and env after the recipe was applied, i.e. the real
condition, ``hand_recipe.declared_values``); ``finger_penalty_span`` is the finger-penalty span the training used
(resolved by ``dexengine.core.hands.resolve_finger_penalty_span``); the two scales are the rule that span name meant
at training time (``dexengine.core.hands.finger_penalty_plan``, n = HandSpec.n_fingers), so a later change of the
rule behind the same name is caught on resume. The ledger fixes the rule (scales), not the base dead-zone and weight
flags. The condition = hand, recipe fields, span and both scales (``CONDITION_KEYS``); switch, source and override
are informational.

Check rules (``check_checkpoint``):
* No ledger (a checkpoint from before the recipe): nothing to compare with, so the switch must be given
  explicitly (``DEXENGINE_HAND_RECIPE=0|1``); the default dies. On resume the finger-penalty span must be given too
  (``--finger_penalty_span``): such checkpoints trained with the legacy span, and resuming without it would
  silently change the reward.
* A ledger: the hand must match and every field's declared value must match (switch on/off may differ when the
  values agree). A recipe table changed after training dies. An intended cross-judge names the field in
  ``DEXENGINE_HAND_RECIPE_OVERRIDE=<field>`` and gives the value; that field is skipped and listed in the marker.
  Span and scales are checked on resume only (both penalties are training rewards; judge and eval do not use them).
  A ledger without one of the three (written before those fields) requires an explicit span on resume. A retired
  span dies on resume (that reward can no longer be built; start a new run).
* A recipe field missing from the ledger's first line (a channel added after the ledger) reads as not declared
  (``-``, the writer's notation) and is named in the marker, so adding a channel does not break reloading old
  checkpoints.

Record rule (``record``, training start): an existing ledger in the run folder is kept only when the condition is
the same; otherwise it dies (old checkpoints in the folder must not be tied to a new condition; an override or a
different span needs a new EXP_NAME). Hand and recipe fields must be equal; the three finger-penalty fields go
through the same rule as the resume check (``_fpen_rule``): a field the ledger lacks (``?``) passes with an explicit
span and the ledger is not rewritten. A folder without a ledger that already holds ``checkpoints/agent_*.pt``
(trained before ledgers) dies instead of getting a first ledger. Imports neither Isaac Lab nor torch.
"""

from __future__ import annotations

import glob
import os

from dexengine.core import hands as _hs
from dexengine.rl.launch import hand_recipe as _hr

LEDGER_NAME = "hand_recipe.txt"
_HEAD = (
    "# per-hand RL recipe ledger: the checkpoints in this folder are loaded again only under the same condition "
    "(dexengine.rl.launch.run_dir.check_checkpoint)"
)
#: Finger-penalty span field = the trainer flag name (``--finger_penalty_span``).
SPAN_FIELD = "finger_penalty_span"
#: The span's two scales at training time (``FingerPenaltyPlan.w_scale`` and ``dz_scale``).
W_SCALE_FIELD = "finger_penalty_w_scale"
DZ_SCALE_FIELD = "finger_penalty_dz_scale"
#: The three finger-penalty fields (span, weight scale, dead-zone scale), checked on resume only.
FPEN_FIELDS = (SPAN_FIELD, W_SCALE_FIELD, DZ_SCALE_FIELD)
#: Value of those fields in a ledger written before them: unknown (resume requires an explicit span).
SPAN_UNKNOWN = "?"
#: What a checkpoint trained under: the keys of the record check. The reload check compares the finger-penalty
#: fields on resume only.
CONDITION_KEYS = ("hand", *_hr.FIELDS, *FPEN_FIELDS)
#: Source value ``dexengine.core.hands.resolve_finger_penalty_span`` returns for an explicitly given span.
_SPAN_GIVEN = _hs.SPAN_GIVEN


def _val(v) -> str:
    return "-" if v is None else repr(float(v))


def _span(args_cli) -> tuple[str, str]:
    try:
        return _hs.resolve_finger_penalty_span(getattr(args_cli, SPAN_FIELD, None))
    except ValueError as e:
        raise _hr.RecipeError(str(e)) from e


def _fpen_values(hand: str, span: str) -> dict:
    """Span -> the three finger-penalty ledger fields (scales to 6 significant digits, as the marker prints them).
    n = HandSpec.n_fingers (the training path checks that the action term agrees)."""
    spec = _hs.HAND_REGISTRY.get(hand)
    if spec is None:
        raise _hr.RecipeError(f"hand {hand!r} is not in HAND_REGISTRY — cannot compute the finger-penalty scales")
    try:
        plan = _hs.finger_penalty_plan(hand, span, spec.n_fingers)
    except ValueError as e:
        raise _hr.RecipeError(str(e)) from e
    return {SPAN_FIELD: span, W_SCALE_FIELD: f"{plan.w_scale:.6g}", DZ_SCALE_FIELD: f"{plan.dz_scale:.6g}"}


def current(args_cli, environ) -> dict:
    """This launch's condition (one ledger line as a dict): hand, switch (and its source), overrides, the declared
    value of every field, the finger-penalty span and scales."""
    rec = {
        "hand": str(args_cli.hand),
        "switch": "1" if _hr.switch_on(environ) else "0",
        "source": "explicit" if _hr.SWITCH_ENV in environ else "default",
        "override": ",".join(sorted(_hr.override_keys(environ))) or "-",
    }
    rec.update({f: _val(v) for f, v in _hr.declared_values(args_cli, environ).items()})
    rec.update(_fpen_values(rec["hand"], _span(args_cli)[0]))
    return rec


def ledger_line(args_cli, environ) -> str:
    return " ".join(f"{k}={v}" for k, v in current(args_cli, environ).items())


def read(path: str) -> dict | None:
    """Ledger first line -> dict. Without hand or switch the ledger is broken (dies). Missing recipe fields read as
    ``-`` and missing finger-penalty fields as ``SPAN_UNKNOWN``; the filled names go to ``_absent`` (for the marker)."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
    rec = dict(tok.split("=", 1) for tok in first.split() if "=" in tok)
    missing = [k for k in ("hand", "switch") if k not in rec]
    if missing:
        raise _hr.RecipeError(f"ledger {path}: first line lacks {missing}: {first!r}")
    absent = [f for f in (*_hr.FIELDS, *FPEN_FIELDS) if f not in rec]
    for f in absent:
        rec[f] = SPAN_UNKNOWN if f in FPEN_FIELDS else "-"
    rec["_absent"] = ",".join(absent) or "-"
    return rec


def _absent_note(rec: dict) -> str:
    """Marker part for fields the ledger lacked: recipe fields = not declared (``-``), finger-penalty fields =
    unknown (``SPAN_UNKNOWN``)."""
    absent = [] if rec["_absent"] == "-" else rec["_absent"].split(",")
    recipe, fpen = [f for f in absent if f not in FPEN_FIELDS], [f for f in absent if f in FPEN_FIELDS]
    parts = ([f"recipe {recipe} = not declared (-)"] if recipe else []) + (
        [f"finger penalty {fpen} = unknown ({SPAN_UNKNOWN})"] if fpen else []
    )
    return f" · fields the ledger lacked: {' · '.join(parts)}" if parts else ""


def ledger_for(ckpt_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path))), LEDGER_NAME)


def _fpen_rule(rec: dict | None, args_cli, cur: dict, *, ctx: str, path: str) -> str:
    """The one finger-penalty compatibility rule: may this launch (``cur`` = ``current``) continue the reward the
    checkpoint of ledger ``rec`` (None = no ledger) learned? Used by the resume check (``_span_rule``) and the
    same-folder record check (``record``). Order: retired span, recorded fields compared, unrecorded (``?``) fields
    need an explicit span. Returns a marker part or raises RecipeError."""
    span, src = cur[SPAN_FIELD], _span(args_cli)[1]
    now = " ".join(f"{f}={cur[f]}" for f in FPEN_FIELDS)
    trained = dict.fromkeys(FPEN_FIELDS, SPAN_UNKNOWN) if rec is None else {f: rec[f] for f in FPEN_FIELDS}
    if trained[SPAN_FIELD] in _hs.RETIRED_FINGER_PENALTY_SPANS:
        raise _hr.RecipeError(
            f"{ctx}: the checkpoint learned with the retired finger-penalty span {trained[SPAN_FIELD]} "
            f"({_hs.RETIRED_FINGER_PENALTY_SPANS[trained[SPAN_FIELD]]}) — that reward can no longer be built, so it "
            f"cannot be resumed with the same reward. Start a new run (cold start) ({path})"
        )
    diffs = [
        f"{f}: trained {trained[f]} != now {cur[f]}"
        for f in FPEN_FIELDS
        if trained[f] != SPAN_UNKNOWN and trained[f] != cur[f]
    ]
    if diffs:
        raise _hr.RecipeError(
            f"{ctx}: finger penalty {diffs} ({src}) — a resume keeps the same reward. For another span resume with "
            f"--{SPAN_FIELD} {trained[SPAN_FIELD]}; the same span name with other scales means the rule of that "
            f"name changed after training — a different reward needs a new run (cold start) ({path})"
        )
    unknown = [f for f in FPEN_FIELDS if trained[f] == SPAN_UNKNOWN]
    if unknown:
        if src != _SPAN_GIVEN:
            why = "no ledger" if rec is None else f"ledger lacks {unknown}"
            raise _hr.RecipeError(
                f"{ctx}: no finger-penalty span/scale record for the checkpoint to resume ({why}) — checkpoints "
                f"from before the ledger learned with the legacy span. Resuming with the default ({span}) would "
                f"silently change the reward. Give --{SPAN_FIELD} legacy (same reward) or "
                f"--{SPAN_FIELD} {_hs.CANON_FINGER_PENALTY_SPAN} (intended reward change) ({path})"
            )
        return f"finger penalty not recorded {unknown} = unknown ({SPAN_UNKNOWN}) · resuming with given {span} ({now})"
    return f"finger-penalty span and scales match {now} ({src})"


def _span_rule(rec: dict | None, args_cli, cur: dict, *, resume: bool, ctx: str, path: str) -> str:
    """Finger-penalty check: ``_fpen_rule`` on resume; a reload (judge, eval) does not compare."""
    if not resume:
        trained = SPAN_UNKNOWN if rec is None else rec[SPAN_FIELD]
        return (
            f"finger-penalty span not compared (reload — a training-only reward; trained {trained}, "
            f"now {cur[SPAN_FIELD]})"
        )
    return _fpen_rule(rec, args_cli, cur, ctx=ctx, path=path)


def check_checkpoint(ckpt_path: str, args_cli, environ=None, *, ctx: str, resume: bool) -> str:
    """Ledger of checkpoint ``ckpt_path`` against this launch. ``resume`` = continue training on it (the
    finger-penalty span is compared too). Returns one marker line or raises RecipeError."""
    environ = os.environ if environ is None else environ
    path = ledger_for(ckpt_path)
    cur, rec = current(args_cli, environ), read(path)
    head = f"[HAND_RECIPE] {ctx}: checkpoint ledger"
    sw = _hr.SWITCH_ENV
    if rec is None:
        if cur["source"] != "explicit":
            raise _hr.RecipeError(
                f"{ctx}: no recipe ledger ({LEDGER_NAME}) next to checkpoint {ckpt_path} — it was trained before the "
                f"recipe. Reloading with the default switch (on) would silently add today's recipe. State the "
                f"training condition: {sw}=0 (hand values as the old command's flags and env) or {sw}=1 (today's "
                f"recipe). The finger-penalty span is outside this switch — on resume also give --{SPAN_FIELD}"
            )
        span = _span_rule(None, args_cli, cur, resume=resume, ctx=ctx, path=path)
        decl = [f"{f}={cur[f]}" for f in _hr.FIELDS if cur[f] != "-"]
        return (
            f"{head} missing (checkpoint from before the recipe) · going on with given {sw}={cur['switch']} · "
            f"declared {decl} · {span} · {path}"
        )
    if rec["hand"] != cur["hand"]:
        raise _hr.RecipeError(f"{ctx}: ledger hand {rec['hand']} != hand now {cur['hand']} ({path})")
    ovr = set(cur["override"].split(",")) - {"-"}
    diffs = [f"{f}: trained {rec[f]} != now {cur[f]}" for f in _hr.FIELDS if f not in ovr and rec[f] != cur[f]]
    if diffs:
        raise _hr.RecipeError(
            f"{ctx}: hand settings the checkpoint trained with != this launch {diffs} — the recipe table changed "
            f"after training or the launch flags differ. To measure under the same condition use the ledger values; "
            f"for an intended cross-judge name the field in {_hr.OVERRIDE_ENV}=<field> and give the value ({path})"
        )
    span = _span_rule(rec, args_cli, cur, resume=resume, ctx=ctx, path=path)
    skipped = sorted(f for f in ovr if rec[f] != cur[f])
    absent = _absent_note(rec)
    return (
        f"{head} matches · trained switch={rec['switch']}({rec.get('source', '?')}) <-> "
        f"now switch={cur['switch']}({cur['source']}) · skipped by override {skipped if skipped else 'none'} · "
        f"{span}{absent} · {path}"
    )


def enforce(ckpt_path: str | None, args_cli, *, ctx: str, resume: bool) -> None:
    """The one call of every path that reloads a checkpoint: prints the check marker, SystemExit on a mismatch.
    ``resume`` has no default (resume and reload are never mixed)."""
    if not ckpt_path:
        return
    try:
        print(check_checkpoint(ckpt_path, args_cli, ctx=ctx, resume=resume), flush=True)
    except _hr.RecipeError as e:
        raise SystemExit(f"[HAND_RECIPE] {e}") from e


def record(run_dir: str, args_cli, environ=None) -> str:
    """Training start: write the ledger into the run folder (atomic replace) and print the marker. The judge
    (FROZEN_REPLAY) does not call this (it is not the training condition). An existing ledger is kept only when the
    condition is the same, otherwise SystemExit; a folder without a ledger that already has checkpoints also dies."""
    environ = os.environ if environ is None else environ
    try:
        cur = current(args_cli, environ)
        path = os.path.join(run_dir, LEDGER_NAME)
        old = read(path)
    except _hr.RecipeError as e:
        raise SystemExit(f"[HAND_RECIPE] ledger record: {e}") from e
    line = " ".join(f"{k}={v}" for k, v in cur.items())
    if old is not None:
        diffs = [
            f"{k}: ledger {old[k]} != now {cur[k]}" for k in CONDITION_KEYS if k not in FPEN_FIELDS and old[k] != cur[k]
        ]
        if diffs:
            raise SystemExit(
                f"[HAND_RECIPE] ledger record: the condition of the ledger in run folder {run_dir} != this training "
                f"{diffs} — the folder's checkpoints learned under the ledger's condition and overwriting would tie "
                f"them to a new one. Use a new EXP_NAME for another condition ({path})"
            )
        # The finger-penalty fields follow the resume rule (_fpen_rule): what did this folder's checkpoints learn
        # with, whether or not this launch resumes. Passing on an unknown (?) field does not rewrite the ledger.
        try:
            fpen = _fpen_rule(old, args_cli, cur, ctx="ledger record", path=path)
        except _hr.RecipeError as e:
            raise SystemExit(
                f"[HAND_RECIPE] {e} — use a new EXP_NAME so the folder's old checkpoints are not tied "
                f"to a new condition"
            ) from e
        print(
            f"[HAND_RECIPE] ledger kept {path}: same condition (hand and recipe fields match · {fpen}) — "
            f"record line unchanged{_absent_note(old)} · now {line}",
            flush=True,
        )
        return path
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "agent_*.pt")))
    if ckpts:
        raise SystemExit(
            f"[HAND_RECIPE] ledger record: run folder {run_dir} has no ledger but {len(ckpts)} checkpoints — it was "
            f"trained before ledgers existed. A first ledger here would record those checkpoints under this "
            f"launch's condition. Train under a new EXP_NAME"
        )
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(line + "\n" + _HEAD + "\n")
    os.replace(tmp, path)
    print(f"[HAND_RECIPE] ledger written {path}: {line}", flush=True)
    return path
