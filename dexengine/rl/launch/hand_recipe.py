"""Per-hand RL recipe: applied at launch, verified on the built env cfg. The values live in one table,
``dexengine.core.hands.HAND_RL_RECIPES``.

* ``apply_launch`` (the launcher, ``dexengine.rl.launch.plan``). With the switch ``DEXENGINE_HAND_RECIPE`` on (the
  default; unset = 1) the recipe values are put on the trainer flags and env. A value the caller already gave
  equal to the recipe is left alone; a different value dies. ``DEXENGINE_HAND_RECIPE_OVERRIDE=<field,...>`` names
  fields where the caller's input (including none) wins. With the switch explicitly 0 nothing is changed and the
  marker only notes whether the launch matches the recipe (that comparison never dies).
* ``verify_env_cfg`` (``create_env`` calls it right before gym.make). Compares the declared values (argv and env)
  with the recipe, and the values the three consumers (finger actuators, force-curriculum action terms, imitation
  release) actually put on the cfg with the declaration. A mismatch, or a consumer that silently skipped, dies.
* The checkpoint ledger (``dexengine.rl.launch.run_dir``) records the training condition next to the checkpoints
  and checks every path that loads a checkpoint again.

Imports neither Isaac Lab nor torch. CLI: ``python3 -m dexengine.rl.launch.hand_recipe show|env-channels``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from dexengine.core import hands as _hs

HAND_REGISTRY = _hs.HAND_REGISTRY
HAND_RL_RECIPES = _hs.HAND_RL_RECIPES
RECIPE_CHANNELS = _hs.RECIPE_CHANNELS
finger_effort_applied = _hs.finger_effort_applied
finger_effort_scale_from_env = _hs.finger_effort_scale_from_env
rl_recipe_for = _hs.rl_recipe_for

SWITCH_ENV = "DEXENGINE_HAND_RECIPE"
OVERRIDE_ENV = "DEXENGINE_HAND_RECIPE_OVERRIDE"
#: Env channel of the hand name: the fallback when the flags carry no --hand.
HAND_ENV = "DEXENGINE_HAND"
SOURCE = "dexengine.core.hands.HAND_RL_RECIPES"
FIELDS = tuple(f for f, _k, _n, _d in RECIPE_CHANNELS)
#: Paths a cfg check runs on; the caller states it (no default).
RECIPE_PATHS = ("train", "eval")
#: Switch value when unset: on. The checkpoint ledger tells an explicit switch from the default.
SWITCH_DEFAULT = "1"


class RecipeError(ValueError):
    """An input or applied value that differs from the recipe (or a malformed recipe input)."""


def switch_on(environ=None) -> bool:
    """``DEXENGINE_HAND_RECIPE``: unset = ``SWITCH_DEFAULT`` (on), "0" = off, "1" = on; anything else dies."""
    raw = (os.environ if environ is None else environ).get(SWITCH_ENV, SWITCH_DEFAULT)
    if raw not in ("0", "1"):
        raise RecipeError(f"{SWITCH_ENV}={raw!r} — only 0 or 1")
    return raw == "1"


def override_keys(environ=None) -> frozenset[str]:
    """Field names in ``DEXENGINE_HAND_RECIPE_OVERRIDE``; an unknown name dies."""
    raw = (os.environ if environ is None else environ).get(OVERRIDE_ENV, "")
    keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
    unknown = sorted(keys - set(FIELDS))
    if unknown:
        raise RecipeError(f"{OVERRIDE_ENV}: unknown fields {unknown} — fields are {list(FIELDS)}")
    return keys


def _num(v) -> float | None:
    return None if v is None else float(v)


def _same(a, b) -> bool:
    a, b = _num(a), _num(b)
    return a == b


def _fmt(v) -> str:
    """Text put on a channel: repr(float), so 0.0 does not become 0."""
    return repr(float(v))


def _recipe_or_none(hand: str):
    try:
        return rl_recipe_for(hand)
    except KeyError as e:
        raise RecipeError(str(e)) from e


def _as_dict(v) -> dict:
    return dict(v) if isinstance(v, dict) else {"*": v}


def _check_switch_inputs(environ, hand):
    """Resolve switch, overrides and recipe at once: (on, recipe | None, override fields)."""
    on = switch_on(environ)
    ovr = override_keys(environ)
    if not on:
        if ovr:
            raise RecipeError(f"{OVERRIDE_ENV}={sorted(ovr)} while {SWITCH_ENV} is off — an input nothing consumes")
        return False, None, ovr
    recipe = _recipe_or_none(hand)
    if recipe is None and ovr:
        raise RecipeError(f"hand {hand!r} has no recipe — {OVERRIDE_ENV} is an input nothing consumes")
    return True, recipe, ovr


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch_tokens(extra: str) -> list[str]:
    """Split the extra flags the way the trainer argv is built: on spaces, tabs and newlines, no quote handling."""
    return [t for t in re.split(r"[ \t\n]+", extra) if t]


def _flag_values(tokens: list[str], flag: str) -> list[str]:
    """Values of ``flag`` in the tokens (``--f v`` and ``--f=v``); a trailing flag without a value dies."""
    out = []
    for i, t in enumerate(tokens):
        if t == flag:
            if i + 1 >= len(tokens):
                raise RecipeError(f"{flag} has no value in the extra flags")
            out.append(tokens[i + 1])
        elif t.startswith(flag + "="):
            out.append(t[len(flag) + 1 :])
    return out


def _resolve_hand(hand, tokens: list[str], environ) -> str | None:
    """Hand to apply: the argument, else the last ``--hand`` of the flags (argparse last-wins), else env
    ``DEXENGINE_HAND``; None when none is given."""
    if hand:
        return hand
    vals = _flag_values(tokens, "--hand")
    if vals:
        return vals[-1]
    return environ.get(HAND_ENV) or None


def _declared_on_launch(tokens: list[str], environ, kind: str, name: str) -> list[str]:
    """Values the launch gave channel ``name`` (env: zero or one; flag: in order of appearance)."""
    if kind == "env":
        return [environ[name]] if name in environ else []
    return _flag_values(tokens, name)


def _off_report(hand, extra: str, environ) -> str:
    """With the switch off: a note on whether the launch matches the recipe. Never raises."""
    try:
        tokens = launch_tokens(extra)
        h = _resolve_hand(hand, tokens, environ)
        if h is None:
            return "no comparison with the recipe (hand unknown)"
        recipe = rl_recipe_for(h)
        if recipe is None:
            return f"hand={h} has no recipe"
        diffs = []
        for field, kind, name, default in RECIPE_CHANNELS:
            want = getattr(recipe, field)
            got_list = _declared_on_launch(tokens, environ, kind, name)
            got = got_list[-1] if got_list else None  # argparse and env: the last value wins
            if not _same(default if got is None else got, default if want is None else want):
                diffs.append(f"{field}: given {'none' if got is None else got} != recipe {want}")
        return f"hand={h} vs recipe: " + (f"{len(diffs)} differ {diffs}" if diffs else "match")
    except Exception as e:  # noqa: BLE001 — an off launch never dies of this comparison
        return f"no comparison with the recipe ({type(e).__name__}: {e})"


def apply_launch(hand, extra: str, environ=None, *, ctx: str = "launch"):
    """(new extra flags, env to export, marker line). With the switch off: the input unchanged plus an off marker.

    ``hand`` None = the last ``--hand`` of the flags, else env ``DEXENGINE_HAND``."""
    environ = os.environ if environ is None else environ
    if not switch_on(environ):
        _check_switch_inputs(environ, None)  # off + overrides dies; off never looks up a recipe
        return (
            extra,
            {},
            (
                f"[HAND_RECIPE] {ctx}: off ({SWITCH_ENV}=0 given — hand settings as the caller gave "
                f"them) · " + _off_report(hand, extra, environ)
            ),
        )
    tokens = launch_tokens(extra)
    h = _resolve_hand(hand, tokens, environ)
    if h is None:
        raise RecipeError(f"{SWITCH_ENV}=1 but the hand is unknown — no --hand in the flags and no env {HAND_ENV}")
    extra_hands = _flag_values(tokens, "--hand")
    if extra_hands and extra_hands[-1] != h:
        raise RecipeError(f"hand {h!r} != --hand {extra_hands[-1]!r} of the flags — not loading another hand's recipe")
    hand = h
    _on, recipe, ovr = _check_switch_inputs(environ, hand)
    if recipe is None:
        return (
            extra,
            {},
            (f"[HAND_RECIPE] {ctx}: hand={hand} has no recipe — flags and env as given · source {SOURCE}[{hand!r}]"),
        )
    flags = [n for _f, k, n, _d in RECIPE_CHANNELS if k == "flag"]
    for t in tokens:  # argparse accepts prefixes; a value given by prefix would silently lose to the recipe value
        nm = t.split("=", 1)[0]
        if len(nm) > 2 and nm.startswith("--") and nm not in flags and any(f.startswith(nm) for f in flags):
            raise RecipeError(f"{nm} in the flags reads as a prefix of a recipe flag — give the full name ({flags})")
    appended, exports, given, overridden = [], {}, [], []
    for field, kind, name, default in RECIPE_CHANNELS:
        want = getattr(recipe, field)
        got_list = _declared_on_launch(tokens, environ, kind, name)
        if len(got_list) > 1:
            raise RecipeError(f"{name} appears {len(got_list)} times in the flags ({got_list}) — give it once")
        got = got_list[0] if got_list else None
        if field in ovr:
            overridden.append(f"{field}={got if got is not None else 'none'}(recipe {want})")
            continue
        if got is None:
            if _same(want, default):
                continue
            if kind == "env":
                exports[name] = _fmt(want)
            else:
                appended += [name, _fmt(want)]
            continue
        try:
            same = _same(got, want if want is not None else default)
        except ValueError as e:
            raise RecipeError(f"{name}={got!r} is not a number") from e
        if not same:
            raise RecipeError(
                f"hand={hand}: {name}={got} != recipe {field}={want} ({SOURCE}[{hand!r}]) — the recipe wins; "
                f"to change it name the field in {OVERRIDE_ENV}={field}"
            )
        given.append(f"{name}={got}")
    # Appended at the end; no extra space when the flags already end in whitespace (so the manifest's EXTRA text
    # is the same as a launch that gave these values itself).
    sep = "" if (not extra or extra[-1].isspace()) else " "
    new_extra = extra + (sep + " ".join(appended) if appended else "")
    applied = [f"{appended[i]} {appended[i + 1]}" for i in range(0, len(appended), 2)]
    applied += [f"{k}={v}" for k, v in exports.items()]
    marker = (
        f"[HAND_RECIPE] {ctx}: hand={hand} (switch {'given' if SWITCH_ENV in environ else 'default'} on) "
        f"applied {len(applied)}{applied if applied else ''} · "
        f"already given {len(given)}{given if given else ''} · "
        f"overridden {overridden if overridden else 'none'} · source {SOURCE}[{hand!r}]"
    )
    return new_extra, exports, marker


# ---------------------------------------------------------------------------
# Env cfg check (create_env)
# ---------------------------------------------------------------------------


def declared_values(args_cli, environ=None) -> dict:
    """Values the launch declared (argv and env): field -> float | None."""
    out = {}
    for field, kind, _name, _default in RECIPE_CHANNELS:
        if kind == "env":
            out[field] = finger_effort_scale_from_env(environ)
        else:
            out[field] = _num(getattr(args_cli, field, None))
    return out


def _value_set(v) -> tuple[float, ...]:
    """The set of cap values (keys ignored: split articulations filter keys per side); None is the empty set."""
    vals = v.values() if isinstance(v, dict) else ([] if v is None else [v])
    return tuple(sorted({round(float(x), 9) for x in vals}))


def _objs_with(container, pred):
    """Public attributes of ``container`` that satisfy ``pred`` (the same dir() walk the cfg overrides use)."""
    out = []
    for nm in dir(container) if container is not None else ():
        if nm.startswith("_"):
            continue
        obj = getattr(container, nm, None)
        if obj is not None and not callable(obj) and pred(obj):
            out.append((nm, obj))
    return out


def expected_shape(args_cli, *, split_bi: bool) -> tuple[int, int]:
    """(articulations carrying the finger actuator, force-curriculum action terms). Bimanual = ``--bimanual 1``.
    Articulations: split bimanual = 2 (robot, robot_left); shared bimanual articulation and one-hand = 1. Action
    terms: bimanual = 2, one-hand = 1."""
    bi = int(getattr(args_cli, "bimanual", 0) or 0) == 1
    return (2 if (bi and split_bi) else 1), (2 if bi else 1)


def applied_problems(
    cfg, hand: str, declared: dict, finger_actuator_key: str, *, build_cap: bool, n_artic: int, n_sides: int
) -> tuple[list[str], list[str]]:
    """Values the consumers put on ``cfg`` against the declaration: (problems, notes).

    ``build_cap``: this path's cfg overrides write ``--finger_effort`` to the actuators (training, and the eval
    parity path). There is one expected cap: the build cap on such a path, else the applied cap times the scale."""
    problems, notes = [], []
    p0, rel, v = _check_finger_terms(
        cfg, hand, declared, finger_actuator_key, build_cap, n_artic, n_sides, notes, problems
    )
    if v is None and p0 is None:
        if rel is not None:
            problems.append(f"cfg.finger_imit_release={rel} but the declaration is off")
        notes.append("imitation release off")
    elif not isinstance(rel, dict):
        problems.append("imitation release declared but cfg.finger_imit_release is missing (consumer skipped)")
    elif not (_same(rel.get("min"), v) and _same(rel.get("start"), p0)):
        problems.append(f"cfg.finger_imit_release min/start={rel.get('min')}/{rel.get('start')} != declared {v}/{p0}")
    elif not rel.get("terms"):
        problems.append("cfg.finger_imit_release has no target terms (the consumer took none)")
    elif n_sides == 2 and not (
        any(str(t).endswith("_left") for t in rel["terms"]) and any(not str(t).endswith("_left") for t in rel["terms"])
    ):
        problems.append(
            f"bimanual but the imitation release targets one side only {rel['terms']} (left terms end in '_left')"
        )
    else:
        notes.append(f"{len(rel.get('terms', ()))} imitation release terms = {v}@{p0}")
    return problems, notes


def _check_finger_terms(cfg, hand, declared, finger_actuator_key, build_cap, n_artic, n_sides, notes, problems):
    """Check the finger actuators and force-curriculum terms against the recipe; return the imitation-release values."""
    # 1) Finger actuators: the number of articulations must match (catches a consumer that skipped a side).
    acts = [
        (nm, o.actuators[finger_actuator_key])
        for nm, o in _objs_with(
            getattr(cfg, "scene", None),
            lambda o: isinstance(getattr(o, "actuators", None), dict) and finger_actuator_key in o.actuators,
        )
    ]
    if len(acts) != n_artic:
        problems.append(
            f"articulations with finger actuator {finger_actuator_key!r} {[nm for nm, _a in acts]}: "
            f"{len(acts)} != expected {n_artic}"
        )
    scale = declared["finger_effort_scale"]
    hw_caps = _as_dict(finger_effort_applied(HAND_REGISTRY[hand]))
    if build_cap and declared["finger_effort"] is not None:
        want_set, want_txt = _value_set(declared["finger_effort"]), f"build cap {declared['finger_effort']:g}"
    else:
        want_set = _value_set({k: float(v) * scale for k, v in hw_caps.items()})
        want_txt = f"applied cap x{scale:g}"
    for nm, a in acts:
        got = _value_set(a.effort_limit_sim)
        if got != want_set:
            problems.append(f"{nm}.{finger_actuator_key}.effort_limit_sim {got} != expected {want_txt} {want_set}")
        else:
            notes.append(f"{nm} cap {want_txt}")
    # 2) Force curriculum: each side's action term carries finger_effort_anneal_start and the final cap
    #    finger_effort_hw.
    fas = declared["finger_effort_anneal_start"]
    terms = _objs_with(getattr(cfg, "actions", None), lambda o: hasattr(o, "finger_effort_anneal_start"))
    hw_set = _value_set(hw_caps)
    if fas is not None and len(terms) != n_sides:
        problems.append(
            f"action terms taking --finger_effort_anneal_start {[nm for nm, _t in terms]}: {len(terms)} "
            f"!= expected {n_sides} (no consumer, or a side missing)"
        )
    for nm, t in terms:
        if not _same(t.finger_effort_anneal_start, fas):
            problems.append(f"actions.{nm}.finger_effort_anneal_start={t.finger_effort_anneal_start} != declared {fas}")
        elif fas is not None and _value_set(getattr(t, "finger_effort_hw", None)) != hw_set:
            problems.append(
                f"actions.{nm}.finger_effort_hw={_value_set(getattr(t, 'finger_effort_hw', None))} "
                f"!= hardware cap {hw_set} (finger_effort_applied)"
            )
        elif fas is not None and not (
            _same(getattr(t, "finger_effort_build", None), declared["finger_effort"])
            and getattr(t, "finger_effort_build_applied", None) is bool(build_cap)
        ):
            problems.append(
                f"actions.{nm}: start cap finger_effort_build={getattr(t, 'finger_effort_build', None)}, "
                f"applied {getattr(t, 'finger_effort_build_applied', None)} != declared "
                f"{declared['finger_effort']}, this path {bool(build_cap)}"
            )
    notes.append(f"{len(terms)} force-curriculum terms = {fas}")
    # 3) Imitation release: cfg.finger_imit_release.
    v, p0 = declared["imit_release_min"], declared["imit_release_start"]
    rel = getattr(cfg, "finger_imit_release", None)
    return p0, rel, v


def verify_env_cfg(
    cfg, args_cli, *, finger_actuator_key: str, ctx: str, path: str, build_cap: bool, split_bi: bool, environ=None
) -> str:
    """The recipe check of create_env. Prints and returns one PASS line; SystemExit on a mismatch.

    ``path`` = "train" | "eval"; ``build_cap`` = this path writes ``--finger_effort`` to the actuators;
    ``split_bi`` = split bimanual articulations. All three are stated by the caller."""
    environ = os.environ if environ is None else environ
    hand = getattr(args_cli, "hand", None)
    if path not in RECIPE_PATHS:
        raise SystemExit(f"[HAND_RECIPE] {ctx}: path={path!r} — one of {RECIPE_PATHS}")
    if path == "train" and not build_cap:
        raise SystemExit(
            f"[HAND_RECIPE] {ctx}: the training path writes --finger_effort to the actuators "
            "— build_cap=False contradicts it"
        )
    try:
        on, recipe, ovr = _check_switch_inputs(environ, hand)
    except RecipeError as e:
        raise SystemExit(f"[HAND_RECIPE] {ctx}: {e}") from e
    if not on:
        line = f"[HAND_RECIPE] {ctx}: off ({SWITCH_ENV}=0 given — hand settings as the command gave them, no check)"
    elif recipe is None:
        line = (
            f"[HAND_RECIPE] {ctx}: hand={hand} has no recipe — command as given (no check) · source {SOURCE}[{hand!r}]"
        )
    else:
        declared = declared_values(args_cli, environ)
        problems = []
        for field in FIELDS:
            want = getattr(recipe, field)
            if field not in ovr and not _same(declared[field], want):
                problems.append(f"declared {field}={declared[field]} != recipe {want}")
        n_artic, n_sides = expected_shape(args_cli, split_bi=split_bi)
        ap, notes = applied_problems(
            cfg, hand, declared, finger_actuator_key, build_cap=build_cap, n_artic=n_artic, n_sides=n_sides
        )
        problems += ap
        if problems:
            raise SystemExit(
                f"[HAND_RECIPE] {ctx}: hand={hand} path={path} check failed ({len(problems)}) — "
                + " | ".join(problems)
                + f" (recipe = {SOURCE}[{hand!r}]; to change: {OVERRIDE_ENV}=<field>)"
            )
        vals = " · ".join(f"{f}={declared[f]}" for f in FIELDS)
        line = (
            f"[HAND_RECIPE] {ctx}: hand={hand} path={path} check PASS · {vals} · "
            f"shape: articulations {n_artic} · sides {n_sides} · build cap {'used' if build_cap else 'unused'} · "
            f"applied: {' · '.join(notes)} · "
            f"overridden {sorted(ovr) if ovr else 'none'} · source {SOURCE}[{hand!r}]"
        )
    print(line, flush=True)
    return line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="the recipe table")
    s.add_argument("--hand", default=None)
    sub.add_parser(
        "env-channels",
        help="env channel names of the recipe, one per line (the judge sidecar compares "
        "them between the training and judge manifests)",
    )
    ns = ap.parse_args(argv)
    if ns.cmd == "env-channels":
        print("\n".join(n for _f, k, n, _d in RECIPE_CHANNELS if k == "env"))
        return 0
    try:
        for h in [ns.hand] if ns.hand else sorted(HAND_RL_RECIPES):
            print(f"{h}\t{_recipe_or_none(h)}")
        return 0
    except RecipeError as e:
        print(f"[HAND_RECIPE] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_main())
