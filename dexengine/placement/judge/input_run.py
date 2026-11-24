"""Run-level fields of the judge input record: rs · rules · budget · collider (measured once per run)."""

from __future__ import annotations

from dexengine.core.errors import InputMissingError
from dexengine.placement.judge.input_fields import (
    _CANON,
    _PAREN_TAIL,
    ARM_FLIP_FLAG,
    JUDGE_STAGE,
    STAGE_POSITION,
    STAGES,
    _csp,
    _h,
    _rr,
)


def canon_flags() -> dict:
    """Judge flags whose effective values enter the record → their default text (judge.config.canon_flags, read from
    configs/placement/judge.yaml). Once per process."""
    if "flags" not in _CANON:
        from dexengine.placement.judge import config as _JC  # noqa: PLC0415

        _CANON["flags"] = dict(_JC.canon_flags())
    return _CANON["flags"]


def _norm(v) -> str:
    """One flag value as comparable text, so an argparse value (judge) and a default text become the same text.
    A flag without value / True → "1", numbers → %.10g, everything else as it is."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return ""
    s = str(v)
    if s == "":
        return "1"
    try:
        return format(float(s), ".10g")
    except ValueError:
        return s


def _dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def stage_of(token: str) -> tuple:
    """One registry stage token → the stage hashes its rule enters (STAGE_POSITION — whole token, else the head of a
    parenthesised tail). A token without a position dies."""
    t = token.strip()
    if t in STAGE_POSITION:
        return STAGE_POSITION[t]
    m = _PAREN_TAIL.fullmatch(t)
    if m and m.group(1).strip() in STAGE_POSITION:
        return STAGE_POSITION[m.group(1).strip()]
    raise ValueError(
        f"judge input record: registry stage token {t!r} has no pipeline position — add it to "
        "judge.input_fields.STAGE_POSITION (j if it is a stage before the judge that builds judge input)"
    )


def _flag_of(ref: str) -> str:
    """The judge flag a registry reference names: `<file>:--flag` → `--flag`; a key of the judge configuration file
    (`configs/placement/judge.yaml:vol_frac`) → `--vol-frac`; anything else → its symbol part."""
    path, _, name = ref.rpartition(":")
    if path.endswith("placement/judge.yaml"):
        return "--" + name.replace("_", "-")
    return name


def stage_thresholds(ds: str, scene: str) -> dict:
    """{stage key: {owner symbol: value}} — threshold **values** of the registry rules at stages that change judge
    results (stage_of) and apply to this (dataset, scene). Keys are the value owners' symbols (`path:name`). Rule names,
    display names and card text are left out, so tidying the registry keeps the record (only a value change makes it
    stale). Stage positions are resolved for every rule before filtering (an unknown token dies on any axis)."""
    if "rules" not in _CANON:
        _CANON["rules"] = _rr().resolve()
    out: dict = {k: {} for k in STAGES}
    for r in _CANON["rules"]:
        hit = sorted({k for t in str(r.stage).split("·") for k in stage_of(t)}, key=STAGES.index)
        if not hit or not _rr().covers(r, ds) or scene not in (r.scenes or ()):
            continue
        for label, ref in r.thresholds:
            # thresholds that point at a judge flag of the record are already carried by the flags / budget fields —
            # adding them again would put the budget into the rules hash and defeat the monotone budget comparison
            if _flag_of(str(ref)) in canon_flags():
                continue
            for k in hit:
                out[k][str(ref)] = r.resolved.get(label)
    return out


def _rules_body(ds: str, scene: str, flags: dict) -> dict:
    """The body the rules field hashes, {stage key: {...}}. Only the judge stage adds the judge flags, the dz layers
    and the search origin. The search origin (`--center-scene`, canonical text) adds a key **only when on**; the
    arm-flip rule (`--arm-flip`) is not in flags and adds the key `arm_flip` only when it differs from the default —
    with the default the body is the same text as without the rule."""
    keys = sorted(k for k in canon_flags() if k not in ("--exhaust-max", ARM_FLIP_FLAG))
    th = stage_thresholds(ds, scene)
    body = {k: {"thresholds": th[k]} for k in STAGES}
    body[JUDGE_STAGE].update(flags={k: _norm(flags.get(k)) for k in keys}, dz=_norm(flags.get("--dz-cells")))
    cs = _csp().norm(flags.get("--center-scene"))
    if cs:
        body[JUDGE_STAGE]["center_scene"] = cs
    af = _norm(flags.get(ARM_FLIP_FLAG))
    if af != _norm(canon_flags()[ARM_FLIP_FLAG]):  # only a run that differs from the default gets the key
        body[JUDGE_STAGE]["arm_flip"] = af
    return body


def _rules_field(ds: str, scene: str, flags: dict) -> str:
    """The rules field `j:<h>,b:<h>,g:<h>` — one hash of the body (_rules_body) per stage."""
    body = _rules_body(ds, scene, flags)
    return ",".join(f"{k}:{_h(body[k])}" for k in STAGES)


def collider(arm: str) -> str:
    """The collider field — the arm collider shape digest (sim_collider.measured_digest), plus, when the search
    filter's proxy geometry is the capsule, its identity `+capsule-<radius rule>-<table hash>+guard-mesh`
    (arm_proxy.judge_identity) and the desk ruler. Once per run."""
    key = ("collider", arm)
    if key not in _CANON:
        from dexengine.placement.collision import arm_proxy as _AP  # noqa: PLC0415 — lazy, as below
        from dexengine.placement.collision import (
            sim_collider as _SC,  # noqa: PLC0415 — lazy: imported before Kit starts
        )

        d = _SC.measured_digest(arm)
        # '+desk-<GUARD_KIND>': the last desk verdict of dp_accept is on the deciding ruler (judge DESK_CTX) —
        # chunks whose verdict was the capsule filter carry no such suffix and read stale.
        _CANON[key] = d if _AP.COLL_PROXY == "mesh" else f"{d}+{_AP.judge_identity()}+desk-{_AP.GUARD_KIND}"
    return _CANON[key]


def _run_field(k: str, fn, where: tuple):
    """Measure one run field. A failure always dies: at judge time the judge could not prove its input; as the
    comparison's current value, a run field that cannot be measured is the same for every row, so counting it as
    'unevaluable' would silently empty the re-judge list."""
    try:
        return fn()
    except (Exception, SystemExit) as e:  # noqa: BLE001 — dies with the reason and a reproduce command
        ds, scene, arm = where
        raise InputMissingError(
            f"[JUDGE_FP] ✗ run field {k} cannot be measured now — {type(e).__name__}: {str(e).strip()[-800:]}\n"
            "  No judge input record row can be compared in this environment (no same / stale verdicts — this does "
            "not end as an empty re-judge list).\n"
            f"  Reproduce: python3 -m dexengine.placement.judge.input_record --canon-conditions {ds} {scene} {arm}"
        ) from e


def conditions(ds: str, scene: str, arm: str, flags: dict, exhaust_max) -> dict:
    """Run fields {rs, rules, budget, collider}. `flags` = {judge flag: value} (judge = effective argparse values ·
    comparison = defaults) + `--dz-cells`. exhaust-max is left out of rules and carried by budget (monotone). A field
    that cannot be measured dies (_run_field)."""
    out = {"rs": str(_rr().RULESET_VERSION)}
    for k, fn in (
        ("rules", lambda: _rules_field(ds, scene, flags)),
        ("budget", lambda: str(int(exhaust_max))),
        ("collider", lambda: collider(arm)),
    ):
        out[k] = _run_field(k, fn, (ds, scene, arm))
    return out


def canon_run_flags(ds: str, scene: str) -> dict:
    """The flags a default judge run uses for (dataset, scene): the config defaults + dz layers + search origin
    (search_policy)."""
    f = dict(canon_flags())
    f["--dz-cells"] = _csp().dz_cells_for(ds, scene)
    f["--center-scene"] = _csp().target_for(ds, scene)
    return f


def canon_conditions(ds: str, scene: str, arm: str) -> dict:
    """The run fields a default judge run would write **now** — the comparison's (dexengine.placement.train_ready)
    current value. Any field that cannot be measured now dies (_run_field)."""
    key = ("cond", ds, scene, arm)
    if key not in _CANON:
        f = _run_field("rules", lambda: canon_run_flags(ds, scene), (ds, scene, arm))
        _CANON[key] = conditions(ds, scene, arm, f, f.get("--exhaust-max"))
    return _CANON[key]


def judge_conditions(a, ds: str, scene: str, exhaust_max) -> dict:
    """The run fields of one judge run — the **effective** argparse value (`a`) of every record flag (an override
    shows as it is). Dies when one cannot be read."""
    f = {k: getattr(a, _dest(k)) for k in canon_flags()}
    f["--dz-cells"] = getattr(a, "dz_cells", None)
    f["--center-scene"] = getattr(a, "center_scene", "")  # effective value (after the judge resolved the policy)
    return conditions(ds, scene, a.arm, f, exhaust_max)
