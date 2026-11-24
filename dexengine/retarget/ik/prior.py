"""Posture prior and hyperextension cut — the retargeting IK's rules, importable by any interpreter.

`ik.py` builds its PosturePrior and cuts its joint ranges from these functions; the windows that apply the same cut
outside the IK (`hyperext_limits`: the truncated training asset and the placement repairs' reference finger range)
import this module, never `ik.py`. The bake/judge interpreter (the Isaac venv) has no spider runtime
(loguru · tyro · viewer), so the rules live apart and import the standard library and omegaconf only
(and `ik.py` defines none of these names again) — tests/retarget/test_prior_and_window.py.

Numbers vs. topology (no hand is the base of another): the numbers (λ · synergy springs · damping …) are owned
by PRIOR_TEMPLATE alone — hand-independent constants promoted from what was measured on wuji, judged by
absolute gates. A hand's yaml states only its chain topology (joint names · flexion sign · neutral angle); a hand that
needs another number names the key under `overrides:` (no silent re-definition; no numbers back in a yaml).
v2: a new hand's default has no synergy / mcp-share springs and no abduction penalty — they tie a
high-DOF hand to a human hand's shape. What stays is the joint-limit cut: the hyperextension branch (where a human
hand does not go) is removed, from pip/dip up to mcp (`clip_roles`), on the side opposite to flexion only (the chain's
`sign` says which), so a one-sided joint is unchanged. wuji keeps the three terms through its `overrides:` (its
trained runs used them); a new hand does not copy that block.
"""

from __future__ import annotations

import os

from omegaconf import OmegaConf

from dexengine.core.hands.spec import HYPEREXT_ROLES

PRIOR_SCHEMA_VERSION = 1
PRIOR_TEMPLATE_VERSION = "v2-20260922"

#: **Canonical order** of the flexion roles that can be clipped. The list order is the clip order, and it is also the
#: order of `hyperext_limited_joints` (the ledger written into the npz) — changing it changes the output.
_PRIOR_CLIP_ROLE_CHOICES = HYPEREXT_ROLES  # one tuple with the hand spec's sim allowance roles
#: The retargeting IK never lets a flexion joint past neutral (the cut and the reference finger range use 0); only the
#: sim body may open further, by the hand spec's sim_finger_hyperext_allow_deg (the window's `--sim`).
IK_HYPEREXT_ALLOW_RAD = 0.0

PRIOR_TEMPLATE: dict[str, float | bool | tuple[str, ...]] = {
    # --- structural switches ---
    "limit_hyperextension": True,  # cut the hyperextension branch in the IK model only
    #: Roles to clip. v2 default = the whole flexion chain (mcp included) — only the side opposite to `sign`'s flexion.
    "clip_roles": ("mcp", "pip", "dip"),
    "synergy_one_sided": True,  # synergy · mcp-share springs act only "when the distal joint leads"
    # --- soft spring strengths ---
    # v2: the three below default to 0 (module doc). The v1 values 0.6 / 0.2 / 0.3 apply to wuji only, via its
    # `overrides:`.
    "lambda_synergy": 0.0,
    "lambda_mcp_share": 0.0,
    "lambda_rest": 0.0,
    "lambda_abduction": 0.0,
    # --- limit avoidance (barrier) ---------------------------------
    # A different term from `lambda_abduction` (the spring that pulls abduction to neutral — v2 default 0): a one-sided
    # barrier that grows only when a joint comes within `limit_margin_deg` of its own limit (0 in the middle of the
    # range — it does not shrink the reachable range). Default 0.0 = off = byte-identical. A hand that turns it on
    # states it under `overrides:`.
    "lambda_limit_avoid": 0.0,
    "limit_margin_deg": 5.0,
    #: Finger roles that get the barrier. Default = abduction of fingers 2-5 (thumb opposition is a swing) — the
    #: thumb only through `overrides:`.
    "limit_avoid_roles": ("index", "middle", "ring", "pinky"),
    "lambda_vel": 0.2,
    "lambda_hyperextension": 0.0,  # soft hyperextension penalty (0: the structural cut is used)
    # --- numerical stability ---
    "damping": 0.02,
    "contact_weight_scale": 0.3,
    "slew_rate_limit": 0.35,
    #: {joint: [lo_deg, hi_deg]}: that joint's finger stays in the window (IK: ik/branch.py · repairs: finger_ranges)
    "branch_window_deg": {},
    # --- finger coupling ratios ---
    "dip_pip_ratio": 0.7,
    "pip_mcp_ratio": 1.30,
    # --- rest posture (flexion coordinates; inactive when lambda_rest=0) ---
    "rest_mcp": 0.35,
    "rest_pip": 0.30,
    "rest_dip": 0.25,
    # --- default neutral angle of a joint entry ---
    "neutral": 0.0,
}

# Canonical order of the finger roles — the position in the list is the MANO contact column (contact_col); the yaml
# does not state column numbers.
PRIOR_FINGER_ROLES = ("thumb", "index", "middle", "ring", "pinky")
PRIOR_SIDES = ("right", "left")


def emb_sides(embodiment_type):
    """embodiment → **the sides actually present in that scene**. This function is the one owner — looking up
    every side of the hand yaml in the rig would kill every clip of a one-hand scene (the other side has no joints). The
    side set is a rule — an unknown emb dies."""
    try:
        return {"right": ("right",), "left": ("left",), "bimanual": ("right", "left")}[str(embodiment_type)]
    except KeyError:
        raise ValueError(
            f"[PRIOR] unknown embodiment_type {embodiment_type!r} — no side set is made up "
            f"(allowed {['right', 'left', 'bimanual']})"
        ) from None


_PRIOR_YAML_KEYS = {"schema", "template", "enabled", "chains", "overrides"}
_PRIOR_CHAIN_PARTS = ("mcp", "pip", "dip", "abd")
_PRIOR_CHAIN_REQUIRED = ("mcp", "pip", "abd")  # dip is optional (two-joint fingers)
_PRIOR_FLEX_KEYS = {"joint", "sign", "neutral"}
_PRIOR_ABD_KEYS = {"joint", "neutral"}


class PriorSchemaError(ValueError):
    """A hand's prior yaml broke the schema — never passed over silently."""


def _prior_die(path, what):
    raise PriorSchemaError(f"[PRIOR] {path}: {what}")


def _prior_flex_entry(path, where, val, tmpl):
    """Flexion joint entry (mcp/pip/dip) → {joint, sign, neutral}. `sign` is **required** — the flexion direction is
    an identity set by the hand's geometry, and a wrong one silently becomes a spring that pulls the other way (no
    silent default)."""
    if not isinstance(val, dict):
        _prior_die(path, f"{where}: must be a mapping (joint · sign required), got {val!r}")
    unknown = set(val) - _PRIOR_FLEX_KEYS
    if unknown:
        _prior_die(path, f"{where}: unknown keys {sorted(unknown)} (allowed {sorted(_PRIOR_FLEX_KEYS)})")
    for k in ("joint", "sign"):
        if k not in val:
            _prior_die(path, f"{where}: `{k}` is missing")
    sign = float(val["sign"])
    if sign not in (1.0, -1.0):
        _prior_die(path, f"{where}: sign must be +1 or -1, got {sign}")
    return {
        "joint": str(val["joint"]),
        "sign": sign,
        "neutral": float(val.get("neutral", tmpl["neutral"])),
    }


def _prior_abd_entry(path, where, val, tmpl):
    """Abduction joint entry → {joint, neutral}. A plain string takes the template's neutral angle.

    No sign is taken — the abduction term only pulls q to its neutral angle and does not use flexion coordinates."""
    if isinstance(val, str):
        val = {"joint": val}
    if not isinstance(val, dict):
        _prior_die(path, f"{where}: must be a joint-name string or a mapping, got {val!r}")
    unknown = set(val) - _PRIOR_ABD_KEYS
    if unknown:
        _prior_die(path, f"{where}: unknown keys {sorted(unknown)} (allowed {sorted(_PRIOR_ABD_KEYS)})")
    if "joint" not in val:
        _prior_die(path, f"{where}: `joint` is missing")
    return {"joint": str(val["joint"]), "neutral": float(val.get("neutral", tmpl["neutral"]))}


def _prior_clip_roles(path, val):
    """`clip_roles` value → tuple in canonical order. Unknown values, duplicates, an empty list or a string **die**
    (`tmpl.update` does not check types — iterating `"pip"` letter by letter would silently mean something else).
    The order is `_PRIOR_CLIP_ROLE_CHOICES`, not the input — the clip order is the order of the npz ledger
    (`hyperext_limited_joints`), so the code owns it."""
    if isinstance(val, str) or not isinstance(val, (list, tuple)):
        _prior_die(path, f"clip_roles: must be a list of roles (e.g. [pip, dip]), got {val!r}")
    items = [str(x) for x in val]
    unknown = [x for x in items if x not in _PRIOR_CLIP_ROLE_CHOICES]
    if unknown:
        _prior_die(path, f"clip_roles: unknown roles {unknown} (allowed {list(_PRIOR_CLIP_ROLE_CHOICES)})")
    if len(set(items)) != len(items):
        _prior_die(path, f"clip_roles: duplicate roles {items}")
    if not items:
        _prior_die(
            path,
            "clip_roles: empty — to clip nothing, state that decision as "
            "`limit_hyperextension: false` (not through an empty list)",
        )
    return tuple(r for r in _PRIOR_CLIP_ROLE_CHOICES if r in set(items))


def _prior_limit_avoid_roles(path, val):
    """`limit_avoid_roles` value → tuple in `PRIOR_FINGER_ROLES` order (judged here for the same reason as
    `clip_roles`). An empty list dies — the decision not to use the barrier is stated as `lambda_limit_avoid: 0.0` (so
    a switched-on λ with 0 targets does not pass silently)."""
    if isinstance(val, str) or not isinstance(val, (list, tuple)):
        _prior_die(path, f"limit_avoid_roles: must be a list of roles (e.g. [index, middle]), got {val!r}")
    items = [str(x) for x in val]
    unknown = [x for x in items if x not in PRIOR_FINGER_ROLES]
    if unknown:
        _prior_die(path, f"limit_avoid_roles: unknown roles {unknown} (allowed {list(PRIOR_FINGER_ROLES)})")
    if len(set(items)) != len(items):
        _prior_die(path, f"limit_avoid_roles: duplicate roles {items}")
    if not items:
        _prior_die(
            path,
            "limit_avoid_roles: empty — to use no barrier, state that decision as "
            "`lambda_limit_avoid: 0.0` (not through an empty list)",
        )
    return tuple(r for r in PRIOR_FINGER_ROLES if r in set(items))


def _prior_fingers(path, chains, want, tmpl):
    """Build the finger list from the declared `chains:`, **requested sides only** (a piece of `build_prior_conf`).
    The side rule is the caller's `want` (= `emb_sides()`) alone — the emb is not read again here."""
    fingers = []
    for side in PRIOR_SIDES:  # canonical order — not the order written in the yaml
        if side not in want:
            continue  # a side absent from this scene — not looked up in the rig
        if side not in chains:
            continue
        sconf = chains[side]
        if not isinstance(sconf, dict):
            _prior_die(path, f"chains.{side}: must be a mapping")
        unknown_roles = set(sconf) - set(PRIOR_FINGER_ROLES)
        if unknown_roles:
            _prior_die(
                path, f"chains.{side}: unknown roles {sorted(unknown_roles)} (allowed {list(PRIOR_FINGER_ROLES)})"
            )
        missing_roles = [r for r in PRIOR_FINGER_ROLES if r not in sconf]
        if missing_roles:
            _prior_die(
                path,
                f"chains.{side}: missing roles {missing_roles} — a declared side lists all five "
                "roles (a missing finger shifts the contact columns)",
            )
        for col, role in enumerate(PRIOR_FINGER_ROLES):
            fconf = sconf[role]
            where = f"chains.{side}.{role}"
            if not isinstance(fconf, dict):
                _prior_die(path, f"{where}: must be a mapping")
            unknown_parts = set(fconf) - set(_PRIOR_CHAIN_PARTS)
            if unknown_parts:
                _prior_die(
                    path, f"{where}: unknown entries {sorted(unknown_parts)} (allowed {list(_PRIOR_CHAIN_PARTS)})"
                )
            for part in _PRIOR_CHAIN_REQUIRED:
                if part not in fconf:
                    _prior_die(path, f"{where}: `{part}` is missing")
            chain = {}
            for part in ("mcp", "pip", "dip"):
                if part not in fconf:
                    continue  # only dip may be omitted (checked above)
                e = _prior_flex_entry(path, f"{where}.{part}", fconf[part], tmpl)
                e["rest"] = float(tmpl[f"rest_{part}"])
                chain[part] = e
            abd_raw = fconf["abd"]
            if not isinstance(abd_raw, list):
                _prior_die(path, f"{where}.abd: must be a list (an empty list `[]` when there is none)")
            abd = [_prior_abd_entry(path, f"{where}.abd[{i}]", a, tmpl) for i, a in enumerate(abd_raw)]
            fingers.append(
                {
                    "name": f"{side}_{role}",
                    "side": side,
                    "role": role,
                    "contact_col": col,
                    "chain": chain,
                    "abd": abd,
                }
            )

    return fingers


def build_prior_conf(path, hand, sides):
    """Hand prior yaml (chain topology) + PRIOR_TEMPLATE (numbers) → PosturePrior config.

    **This function is the only producer of the PosturePrior internal config** — nothing else makes up the
    `fingers:` structure.

    Hand yaml schema — **this docstring owns the schema** (the docs point here):

        schema: 1                     # required (= PRIOR_SCHEMA_VERSION)
        template: v2-20260922         # required (= PRIOR_TEMPLATE_VERSION); a different value dies
        enabled: true                 # required; when false, this is the only key allowed
        chains:                       # required; a declared side lists all five roles
          right:
            thumb:
              mcp: {joint: …, sign: +1|-1, neutral: <optional>}
              pip: {joint: …, sign: …}
              dip: {joint: …, sign: …}      # optional (omitted for two-joint fingers)
              abd: [joint_name, …]          # required (state an empty list `[]` when there is none)
            index: … / middle: … / ring: … / pinky: …
          left: …
        overrides:                    # optional; keys must be PRIOR_TEMPLATE keys
          lambda_synergy: 0.4
          clip_roles: [pip, dip]      # roles to clip (a subset); v2 default = [mcp, pip, dip]
          lambda_limit_avoid: 3.0     # limit-avoidance barrier strength; default 0.0 = off (byte-identical)
          limit_margin_deg: 5.0       # limit margin (°) where that barrier acts; 0 with λ>0 dies
          limit_avoid_roles: [index, middle, ring, pinky]   # fingers that get the barrier (default = these four)

    `sides` = the sides present in this scene (given by `emb_sides` — the emb is not interpreted again). The chain of a
    requested side must exist (else it dies) and an unrequested side is not built — not an allow-list: a wrong joint
    name on a requested side dies.

    Returns `(conf, meta)`. With `enabled: false` the conf is None and that hand uses neither the prior nor the
    hyperextension cut — **declared, not skipped because a file was missing**.
    """
    want = tuple(sides)
    raw = _prior_raw(path, hand, want)
    if not bool(raw["enabled"]):
        extra = set(raw) - {"enabled"}
        if extra:
            _prior_die(
                path,
                f"`enabled: false` but {sorted(extra)} present — a hand that does not use the prior "
                "keeps the one declaration line (values that look applied are an accident)",
            )
        return None, {
            "hand": hand,
            "enabled": False,
            "template": "-",
            "overrides": 0,
            "chains": 0,
            "override_keys": [],
            "clip_roles": [],
            "sides": list(want),
        }
    tmpl, overrides = _prior_template(path, raw)
    fingers = _prior_fingers(path, _prior_chains(path, raw["chains"], want), want, tmpl)
    meta = {
        "hand": hand,
        "enabled": True,
        "template": PRIOR_TEMPLATE_VERSION,
        "sides": list(want),
        "overrides": len(overrides),
        "chains": len(fingers),
        "override_keys": sorted(overrides),
        "clip_roles": list(tmpl["clip_roles"]) if tmpl["limit_hyperextension"] else [],
    }
    conf = _prior_conf(tmpl, fingers)
    from dexengine.retarget.ik.branch import parse_windows  # noqa: PLC0415 (numpy only at import)

    if windows := parse_windows(tmpl["branch_window_deg"], fingers, want, lambda w: _prior_die(path, w)):
        conf["branch_windows"] = windows  # only when declared: a hand without windows keeps its conf
    return conf, meta


def _prior_raw(path, hand, want):
    """The hand yaml as a mapping, after the side set, the file, the top-level keys and `enabled` are checked."""
    if not want or any(x not in PRIOR_SIDES for x in want):
        _prior_die(
            path,
            f"sides={list(want)} is wrong (allowed {list(PRIOR_SIDES)}, cannot be empty) — "
            "the owner of the side set is `emb_sides()`",
        )
    if not os.path.exists(path):
        # A missing prior is a failure, not a silent pass (retargeting without the prior grows the
        # fingertip error).
        raise FileNotFoundError(
            f"[PRIOR] posture prior missing: {path}\n"
            f"        make the retargeting posture prior of hand {hand!r}, or, if it is not to be used,\n"
            f"        declare that decision in a file — a one-line `enabled: false` stub is enough\n"
            f"        (example: assets/robots/sharpa_wave/retarget_prior.yaml).\n"
            f"        A run without a prior is never silent."
        )
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        _prior_die(path, "the top level is not a mapping")
    unknown = set(raw) - _PRIOR_YAML_KEYS
    if unknown:
        _prior_die(
            path,
            f"unknown top-level keys {sorted(unknown)} (allowed {sorted(_PRIOR_YAML_KEYS)}). "
            "The numbers are owned by prior.PRIOR_TEMPLATE, not by the hand yaml — "
            "if this hand alone must differ, name the key under `overrides:`",
        )
    if "enabled" not in raw:
        _prior_die(path, "state `enabled` — whether the prior is used has no silent default")
    return raw


def _prior_template(path, raw):
    """PRIOR_TEMPLATE with the hand's `overrides:` applied, after the schema and template versions are checked."""
    for k in ("schema", "template", "chains"):
        if k not in raw:
            _prior_die(path, f"`{k}` is missing (required for a hand with enabled: true)")
    if int(raw["schema"]) != PRIOR_SCHEMA_VERSION:
        _prior_die(path, f"schema {raw['schema']} != this code's {PRIOR_SCHEMA_VERSION}")
    if str(raw["template"]) != PRIOR_TEMPLATE_VERSION:
        _prior_die(
            path,
            f"template {raw['template']!r} != this code's {PRIOR_TEMPLATE_VERSION!r} — "
            "the template numbers changed. Gate the hand yaml again and bump the version",
        )

    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        _prior_die(path, "`overrides:` must be a mapping")
    unknown_ov = set(overrides) - set(PRIOR_TEMPLATE)
    if unknown_ov:
        _prior_die(path, f"unknown override keys {sorted(unknown_ov)} (allowed {sorted(PRIOR_TEMPLATE)})")
    tmpl = dict(PRIOR_TEMPLATE)
    tmpl.update(overrides)
    tmpl["clip_roles"] = _prior_clip_roles(path, tmpl["clip_roles"])
    tmpl["limit_avoid_roles"] = _prior_limit_avoid_roles(path, tmpl["limit_avoid_roles"])
    if float(tmpl["limit_margin_deg"]) <= 0 and float(tmpl["lambda_limit_avoid"]) > 0:
        _prior_die(
            path,
            f"lambda_limit_avoid={tmpl['lambda_limit_avoid']} but "
            f"limit_margin_deg={tmpl['limit_margin_deg']} — with a zero margin "
            "the barrier acts on no frame (a switched-on flag with no consumer)",
        )
    return tmpl, overrides


def _prior_chains(path, chains, want):
    """`chains:` after its shape and sides are checked: every side the scene requires is declared."""
    if not isinstance(chains, dict) or not chains:
        _prior_die(path, "`chains:` must be a non-empty mapping")
    unknown_sides = set(chains) - set(PRIOR_SIDES)
    if unknown_sides:
        _prior_die(path, f"unknown hand sides {sorted(unknown_sides)} (allowed {list(PRIOR_SIDES)})")

    missing_side = [x for x in want if x not in chains]
    if missing_side:
        # A side the scene requires but the yaml does not declare is a failure, not a silent pass.
        _prior_die(
            path,
            f"this scene (emb) requires {list(want)} but `chains:` lacks "
            f"{missing_side} (declared sides {sorted(chains)})",
        )
    return chains


def _prior_conf(tmpl, fingers):
    """The PosturePrior internal config from the resolved template numbers and the built fingers."""
    return {
        "enabled": True,
        "limit_hyperextension": bool(tmpl["limit_hyperextension"]),
        "clip_roles": tuple(tmpl["clip_roles"]),
        "synergy_one_sided": bool(tmpl["synergy_one_sided"]),
        "lambda_synergy": float(tmpl["lambda_synergy"]),
        "lambda_mcp_share": float(tmpl["lambda_mcp_share"]),
        "lambda_rest": float(tmpl["lambda_rest"]),
        "lambda_abduction": float(tmpl["lambda_abduction"]),
        "lambda_limit_avoid": float(tmpl["lambda_limit_avoid"]),
        "limit_margin_deg": float(tmpl["limit_margin_deg"]),
        "limit_avoid_roles": tuple(tmpl["limit_avoid_roles"]),
        "lambda_vel": float(tmpl["lambda_vel"]),
        "lambda_hyperextension": float(tmpl["lambda_hyperextension"]),
        "damping": float(tmpl["damping"]),
        "contact_weight_scale": float(tmpl["contact_weight_scale"]),
        "slew_rate_limit": float(tmpl["slew_rate_limit"]),
        "defaults": {
            "dip_pip_ratio": float(tmpl["dip_pip_ratio"]),
            "pip_mcp_ratio": float(tmpl["pip_mcp_ratio"]),
        },
        "fingers": fingers,
    }


def prior_marker(meta):
    """Applied marker: a run whose log lacks this line is invalid. `chains=<n>` = number of fingers with a declared
    chain · `clip_roles=…` = the declaration (the joints actually clipped are reported by `[PRIOR] clipped=…` after the
    model is built — declaration and measurement are not written on one line)."""
    if not meta["enabled"]:
        return (
            f"[PRIOR] sides={meta.get('sides', [])} "
            f"hand={meta['hand']} template=- overrides=0 chains=0 "
            "(enabled:false — this hand **declared** that it does not use the prior. "
            "It was not skipped because a file was missing.)"
        )
    line = (
        f"[PRIOR] sides={meta['sides']} "
        f"hand={meta['hand']} template={meta['template']} "
        f"overrides={meta['overrides']} chains={meta['chains']} "
        f"clip_roles={meta['clip_roles']}"
    )
    if meta["override_keys"]:
        line += f" override_keys={meta['override_keys']}"
    return line


def hyperext_clip_targets(prior_conf):
    """Joints that get the hyperextension cut, as `(name, sign, neutral, role)` — the one owner of **which joints are
    clipped in which order**.

    Targets = the chain roles chosen by `clip_roles` (v2 default mcp · pip · dip — wuji uses [pip, dip] via
    overrides). Iteration order = the finger list (canonical side · role order of `build_prior_conf`) ×
    `_PRIOR_CLIP_ROLE_CHOICES` (following the yaml's written order would make the order of the npz ledger
    `hyperext_limited_joints` differ between files). dip is optional and skipped when the chain lacks it.
    Callers: `ik.main()` (IK model) · `hyperext_limits.py` (the window for the training asset and the reference finger
    range)."""
    roles = tuple(prior_conf["clip_roles"])
    for finger in prior_conf["fingers"]:
        for part in _PRIOR_CLIP_ROLE_CHOICES:
            if part in roles and part in finger["chain"]:
                p = finger["chain"][part]
                yield p["joint"], float(p["sign"]), float(p["neutral"]), part


def hyperext_clip_range(lo, hi, sign, neutral, allow):
    """Range `(lo, hi)` of one joint → the range with the hyperextension branch cut off — **the one owner of the cut
    rule**.

    Only the side where the flexion coordinate f = sign·(q − neutral) is negative (hyperextension) is cut, at f = −allow
    (rad; IK_HYPEREXT_ALLOW_RAD for the IK): sign ≥ 0 raises the lower limit to max(lo, neutral − allow), sign < 0
    lowers the upper limit to min(hi, neutral + allow) (e.g. the DG-5F-S right thumb MCP [−153°, 0] is cut at the top).
    If the cut leaves an empty range or a single point (the cut beyond the opposite limit), the original range is
    returned. The IK model, the training asset and the
    reference finger range all call this one function — the three ranges come from one rule."""
    lo, hi, neutral, allow = float(lo), float(hi), float(neutral), float(allow)
    if float(sign) >= 0:
        new_lo = max(lo, neutral - allow)
        return (new_lo, hi) if new_lo < hi else (lo, hi)
    new_hi = min(hi, neutral + allow)
    return (lo, new_hi) if new_hi > lo else (lo, hi)
