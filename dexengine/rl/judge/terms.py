"""Termination rules of the judge: the values the judge standard owns (a dependency-free module; the shell reads it
with python3).

Contact termination (``--enable_contact_termination``: end the episode when the reference says contact and hand
contact is lost for more than the grace steps) is a training-shaping device. The judge's success definition is object
tracking (position and rotation thresholds); it never carries contact termination, for bimanual and one-hand runs
alike. Bimanual training has it off too (COLL_PRESET: training = judge = 0); one-hand training has it on, so the
one-hand judge path overrides the training flag with this value (``dexengine.rl.launch.plan.judge_contact_term``).
This is the one axis where the judge differs from training.

The articulation theta-deviation termination (``ARTIC_THETA_TERM``) is carried: it is the articulated object's success
definition ("opened / closed?"), training = judge. The term name lives here, thresholds and predicate in
``dexengine.rl.judge.artic``, registration in ``dexengine.rl.env_setup.build`` (per articulated side, training and eval
alike). Registration prints ``[JUDGE_TERM] artic_theta=...`` (``artic_marker``, the statement); the judge line
``[FROZEN EVAL iter ...] artic_theta=<n>`` is the number of theta terms the termination manager measured; the sidecar
compares the two. PARK-ed envs are not counted by the theta term (the PARK mask is passed on).

Explicit exception (a measurement arm, not a judge): ``DEXENGINE_JUDGE_CONTACT_TERM=0|1``, consumed by the one-hand
judge path only; the sidecar refuses it (the judge table never records a non-standard value); other values die.

Proof of application: the start-up check of the fresh-process judge (``dexengine.rl.judge.launch.check_judge_terms``)
compares the trainer argument with this table, prints ``[JUDGE_TERM] contact_term=... (judge standard | given ...)``
and dies on a mismatch. The sidecar compares the judge log's ``[FROZEN EVAL iter ...] contact_term=`` (measured by the
termination manager) with this value (the statement): an independent oracle.

Object-tracking termination (``object_tracking``, bimanual left ``object_tracking_left``: 5 cm / 90 degrees with a lag
window and grace) is on in the judge standard (it defines a completed clip). Explicit exception (a measurement arm,
not a judge): ``DEXENGINE_JUDGE_OBJTRACK_TERM=0`` turns off only those terms and the episode runs to the clip end
(trajectory_done); all other terminations stay standard. It is used to roll the method's policy out under the same
protocol as the baselines that always run to the end; completion and pass10 of such a judge carry no meaning. Only
the fresh-process judge consumes it (``check_launch`` refuses it elsewhere); values other than 0|1 die. Applied as the
termination term parameter ``judge_off`` (``dexengine.rl.env_setup.objtrack.apply_judge_objtrack_term``: errors and
diagnostics stay, only the termination is not raised). Statement ``[JUDGE_TERM] objtrack=on(...)`` |
``[JUDGE_TERM] objtrack=off(...)``; measurement = the judge log's ``[TERMDIAG] step ...`` lines (with the term off
every object_tracking* count is 0).

CLI: ``python3 -m dexengine.rl.judge.terms contact-term`` (value to apply, explicit exception included),
``... contact-term --standard`` (the standard value, the sidecar's reference), ``... objtrack-term [--standard]``
(object-tracking termination, 1 = on, 0 = off).
"""

from __future__ import annotations

import os
import sys

CONTACT_TERM = 0  # contact termination of the judge standard (off)
OVERRIDE_VAR = "DEXENGINE_JUDGE_CONTACT_TERM"  # explicit exception (measurement arm); one-hand judge path only
ARTIC_THETA_TERM = "obj_arti_theta"  # articulation theta term of the judge termination set (left: _left)
OBJTRACK_TERM = 1  # object-tracking termination of the judge standard (on)
OBJTRACK_VAR = "DEXENGINE_JUDGE_OBJTRACK_TERM"  # explicit exception (measurement arm, run to the end); fresh judge
OBJTRACK_TERMS = ("object_tracking", "object_tracking_left")  # the terms it turns off (per side)
#: Source of a value: the judge standard, or an explicit environment input.
STANDARD, GIVEN = "standard", "given"


def contact_term(env=None) -> tuple[int, str]:
    """(contact termination value to apply, source STANDARD | GIVEN). A given value other than 0|1 dies."""
    env = os.environ if env is None else env
    raw = (env.get(OVERRIDE_VAR, "") or "").strip()
    if OVERRIDE_VAR not in env:
        return CONTACT_TERM, STANDARD
    if raw not in ("0", "1"):
        raise ValueError(f"[JUDGE_TERM] {OVERRIDE_VAR}={raw!r} — only 0 or 1")
    return int(raw), GIVEN


def marker(value: int, src: str) -> str:
    """The judge log's application marker; the sidecar looks for the standard form ``contact_term=<std>(judge
    standard``."""
    if src == STANDARD:
        return (
            f"[JUDGE_TERM] contact_term={value}(judge standard — contact termination shapes training and is not "
            "part of the judge · owner dexengine.rl.judge.terms)"
        )
    return (
        f"[JUDGE_TERM] ⚠ contact_term={value}(given {OVERRIDE_VAR} — away from the judge standard {CONTACT_TERM} "
        "this is a measurement arm, not a judge)"
    )


def objtrack_term(env=None) -> tuple[int, str]:
    """(object-tracking termination value 1|0 to apply, source STANDARD | GIVEN). A given value other than 0|1 dies."""
    env = os.environ if env is None else env
    if OBJTRACK_VAR not in env:
        return OBJTRACK_TERM, STANDARD
    raw = (env.get(OBJTRACK_VAR, "") or "").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"[JUDGE_TERM] {OBJTRACK_VAR}={raw!r} — only 0 or 1")
    return int(raw), GIVEN


def objtrack_marker(value: int, src: str, terms: list[str]) -> str:
    """Object-tracking termination marker (the statement); a different line from the ``contact_term=`` marker."""
    t = ",".join(terms) or "none"
    if value == 1:
        how = "judge standard" if src == STANDARD else f"given {OBJTRACK_VAR}=1 — the standard value"
        return f"[JUDGE_TERM] objtrack=on({how} · terms {t} · owner dexengine.rl.judge.terms)"
    return (
        f"[JUDGE_TERM] objtrack=off(given {OBJTRACK_VAR}=0 — object-tracking termination off · episodes run to the "
        f"clip end · other terminations standard · terms {t} · a measurement arm, not a judge · completion carries "
        f"no meaning)"
    )


def artic_marker(terms: list[str], desc: str) -> str:
    """Articulation theta term registration marker; a different line from the ``contact_term=`` marker."""
    return (
        f"[JUDGE_TERM] artic_theta={','.join(terms)}(judge standard — articulation success = theta tracking · "
        f"training = judge · {desc} · term name owner dexengine.rl.judge.terms)"
    )


def _main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("contact-term", "objtrack-term") or argv[1:] not in ([], ["--standard"]):
        print("usage: python3 -m dexengine.rl.judge.terms contact-term|objtrack-term [--standard]", file=sys.stderr)
        return 2
    std, fn = (CONTACT_TERM, contact_term) if argv[0] == "contact-term" else (OBJTRACK_TERM, objtrack_term)
    if argv[1:] == ["--standard"]:
        print(std)
        return 0
    try:
        v, _src = fn()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(v)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
