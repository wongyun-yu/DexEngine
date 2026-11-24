"""Hand and arm ids, hand spec dirs, combo names (split_combo) and ids, hand rig hashes and per-hand clip/mode
records."""

from __future__ import annotations

import re
from pathlib import Path

from dexengine.core.locations.modules import data_layout, load_registry


def split_combo(combo: str, arm_names, hand_names, *, unsplit_ok: bool = False) -> tuple[str, str, str]:
    """Combo string → (arm, hand, variant suffix). **The one combo parser** —
    shells use the window `dexengine.core.paths --what split-combo --combo C` (no `${COMBO#*_}`, no private prefix
    loop).

    Unsplittable **raises ValueError**: a ("", "", "") marker used as a pair matches every other unsplittable name
    (a cleanup ledger can open a delete root that way). `unsplit_ok=True` is for a
    **classifier** — a caller whose input need not be a combo at all (a folder scan, a `--usd` value that may be
    a bare variant, a chunk identity naming a retired hand) — and it gets ("", "", "") and owns that answer.

    Variant suffix = `DEXENGINE_COMBO_VARIANT` (e.g. `rby1_sharpa_wave_d20x30` → `_d20x30`,
    `rby1_sharpa_wave.v2keep` → `.v2keep`). The registry is canonical for the names, so the caller
    passes in the list of registered names — no literal is nailed down here.
    Longest names are matched first so that `sharpa_wave` cannot intercept `sharpa_wave_pre_final`.
    Boundary rule: a variant is its own token, so the character right after the hand name must not be a letter or
    digit — `rby1_wujiv2` is not wuji + `v2` (a bare `startswith(hand)` would swallow it), and `.v2keep` still
    splits (a `hand + "_"` rule could not read it).
    """
    for arm in sorted((a for a in arm_names if a), key=len, reverse=True):
        if not combo.startswith(arm + "_"):
            continue
        rest = combo[len(arm) + 1 :]
        for hand in sorted((h for h in hand_names if h), key=len, reverse=True):
            if rest == hand or (rest.startswith(hand) and not rest[len(hand)].isalnum()):
                return arm, hand, rest[len(hand) :]
    if unsplit_ok:
        return "", "", ""
    raise ValueError(
        f"combo {combo!r} does not split into a registered (arm, hand) "
        f"(arms {sorted(a for a in arm_names if a)} · hands {sorted(h for h in hand_names if h)})"
    )


def _hand_spec_dirs() -> dict:
    out = {}
    for hid, hs in load_registry("hand_specs").HAND_REGISTRY.items():
        if hs is None:
            continue
        d = getattr(hs, "retarget_dirname", "") or ""
        if not d:
            raise ValueError(
                f"hand {hid!r} has an empty retarget_dirname — fill it in the registry."
                " Inventing a default here would make a second table."
            )
        out[hid] = d
    return out


HAND_SPEC_DIR = _hand_spec_dirs()  # hand id → spec directory (authority = the registry)
HAND_IDS = tuple(sorted(HAND_SPEC_DIR))
#: Hand ids dropped from the registry (Wuji 1) that stored records still name — e.g. the hand scope of an
#: exclusion-ledger line. A reader drops them from a scope (that never widens it); any other unknown id still raises.
RETIRED_HAND_IDS = ("wuji",)
ARM_IDS = tuple(sorted(load_registry("arm_specs").ARM_REGISTRY))  # authority = the registry


def combo_for(hand: str, arm: str = "") -> str:
    """The registered combo **of this hand**, narrowed by the arm when one is given.

    A rule, not a lookup table: a hand never seen before still gets an answer if it is registered in
    `combo_ids()` (authority = MOUNT_REGISTRY); zero or two-plus candidates **die instead of inventing one**.
    """
    h = hand_id(hand)
    pairs = {x: split_combo(x, ARM_IDS, HAND_IDS)[:2] for x in combo_ids()}  # the one combo parser — no endswith(_h)
    c = [x for x, (a, hx) in pairs.items() if hx == h and (not arm or a == arm)]
    if len(c) == 1:
        return c[0]
    raise ValueError(
        f"cannot pick one combo for hand {hand!r} (id {h}) — candidates {c or 'none'}"
        f" (registered combos {list(combo_ids())}). State it with --combo or --arm"
    )


def combo_dirname(combo: str) -> str:
    """Combo id -> the DIRECTORY name its generated assets live under.

    The directory name is the combo id plus the arm's canonical generation suffix (owner
    `dexengine.core.arms.canon_variant` — this is only the window onto it). The rby1 suffix is empty today, so the
    two coincide; a consumer still asks here instead of using the id AS the directory, so a non-empty suffix
    (another arm, a later generation) cannot silently point it at a folder that does not exist.
    """
    arm = split_combo(combo, ARM_IDS, HAND_IDS)[0]  # split_combo raises ValueError when it cannot split
    return combo + load_registry("arm_specs").canon_variant(arm)


def combo_usd_relpath(combo: str) -> str:
    """Combo id -> its canonical full-robot USD, relative to ``data/usd_cache/robots`` (a pair artefact: the combo
    directory, ``combo_dirname``). The training spawns it; the USD bake's plan checks its output against it."""
    d = combo_dirname(combo)
    return f"full_robot_gen/{d}/{d}.usd"


def combo_ids() -> tuple[str, ...]:
    """Every registered combo (`<arm>_<hand>`). The authority is `dexengine.core.mounts.MOUNT_REGISTRY`.

    Write the list down in the shell and registering a new hand or arm leaves that one script
    behind on the old list.

    A floating-hand pair (dexengine.core.arms.FLOATING_ARM) is **not a merged combo**: the path that mounts a hand
    on an arm body and bakes one URDF·USD (scripts/build_robot_usd.py · the split generator) does not exist for it;
    each side's free-root hand is baked separately (experiments/ablations/floating_hand/make_floating_urdf.py ·
    build_floating_usd.sh). So it is left out here — including it would make the default merged-bake list die on a
    URDF that does not exist.
    """
    reg = load_registry("mount_specs").MOUNT_REGISTRY
    arms = load_registry("arm_specs").ARM_REGISTRY
    return tuple(sorted(f"{arm}_{hand}" for arm, hand in reg if not getattr(arms.get(arm), "is_floating", False)))


#: **Runtime output roots** without a dataset axis — they live under the data root (main_root). `logs/` is written
#: by training (its checkpoints).
#: A worktree points at the data root through DEXENGINE_MAIN_ROOT alone.
RUNTIME_DIRS = ("logs",)


def hand_spec_dir(hand: str) -> str:
    """Hand id → spec directory name. An unknown hand is **never let through silently**."""
    if hand not in HAND_SPEC_DIR:
        raise KeyError(
            f"unregistered hand: {hand!r} (registered: {sorted(HAND_SPEC_DIR)}). "
            "Register a new hand by following the naming-contract checklist."
        )
    return HAND_SPEC_DIR[hand]


def combo_id(arm: str, hand: str, variant: str = "") -> str:
    """`<arm>_<hand><variant>`. Do not join a combo string by hand."""
    if arm not in ARM_IDS:
        raise KeyError(f"unregistered arm: {arm!r} (registered: {list(ARM_IDS)})")
    return f"{arm}_{hand_id(hand)}{variant}"


def hand_id(hand: str) -> str:
    """Even given a spec directory name, returns the hand id (idempotent both ways)."""
    if hand in HAND_SPEC_DIR:
        return hand
    for k, v in HAND_SPEC_DIR.items():
        if v == hand:
            return k
    raise KeyError(f"unregistered hand: {hand!r} (registered: {sorted(HAND_SPEC_DIR)})")


def hand_rig_sha(hand: str) -> str:
    """Hand (hand id or spec directory) → the hand's **registered rig identity**
    (dexengine.core.layout.RETARGET_HAND_ASSETS[<spec dir>].sha256_xml3).

    The one window — every place that asks for the registered rig by hand name (one-hand scene piece freshness,
    the training-tree rig check, the stage-1 rig identity) calls this.
    Why: the asset table is keyed by spec directory, yet it holds a **rejected asset key** with the same name as a
    hand id (`"wuji"`). A caller that uses the hand id as the key does not die with KeyError — it is silently
    compared against another rig (and every bake of that hand dies on a rig mismatch). So the name kind is normalised
    here (dexengine.core.paths.hand_id — idempotent both ways) and the chosen asset row is checked to be **that
    hand's canon asset** (isaac_hand_id = that hand id · status canon). Unknown name · no registered row · another
    hand's row · not canon → KeyError."""
    hid = hand_id(hand)
    spec = hand_spec_dir(hid)
    a = data_layout.RETARGET_HAND_ASSETS.get(spec)
    if a is None or a.isaac_hand_id != hid or a.status != "canon":
        got = "none" if a is None else f"isaac_hand_id={a.isaac_hand_id!r} status={a.status!r}"
        raise KeyError(
            f"hand {hand!r} (id {hid} · spec {spec}) has no canon retargeting asset ({got}) — "
            "register it in dexengine.core.layout.RETARGET_HAND_ASSETS with its source and status"
        )
    return a.sha256_xml3


#: Mode-name shape of a run that writes nothing (dry · count · measure) — a lowercase word of letters and digits
#: starting with a letter (a rule, not a table).
MODE_RE = re.compile(r"[a-z][a-z0-9]*")


def clips_record(path, scoped: bool) -> Path:
    """Record location of a scoped run (one that wrote only part of a dataset via `--clips`) — **the one rule
    owner**. The record `<stem><suffix>` of a write run over the whole dataset belongs to that run alone
    (scoped=False → that name). A scoped run writes only `<stem>.clips<suffix>` — overwriting the same name would
    silently turn the whole-dataset record (counts · completion · every row) into a record of a few trees. A run that
    writes nothing applies mode_record on top of this location.

    The returned place is a **write** place, so its folder is created here (every caller writes the record next): run
    output folders (`run_path` layers such as `exp_shared` = records) do not exist in a fresh checkout, and a
    writer that assumed them died there (the articulation surface ledger, stage 1 step 2b). A dry run of a caller may
    leave that folder empty."""
    p = Path(path)
    p = p.with_name(f"{p.stem}.clips{p.suffix}") if scoped else p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def mode_record(path, mode: str | None) -> Path:
    """Split the location of a run's records (ledger · report · list) by mode — **the one rule owner**.
    The record `<stem><suffix>` of a write run belongs to the write run alone (mode=None → that name unchanged). A run
    of the same tool that writes nothing (dry · count · measure) writes only `<stem>.<mode><suffix>` (e.g.
    ledger_taco.tsv → ledger_taco.need.tsv · LEDGER.tsv → LEDGER.dryrun.tsv) — overwriting the same name would
    silently turn the last write run's record into that of a run that wrote nothing.
    One call shape: ``mode_record(p, None if write else "<mode>")``. A mode string outside the shape (MODE_RE) dies —
    an empty string would silently pass through to the write run's name. Records inside the data tree are not split by
    this rule (a run that writes nothing never writes into the canonical tree — the caller demands an explicit place,
    e.g. the --report of cross_repair --dry-run)."""
    p = Path(path)
    if mode is None:
        return p
    if not isinstance(mode, str) or not MODE_RE.fullmatch(mode):
        raise ValueError(
            f"mode {mode!r} of a run that writes nothing — outside the shape {MODE_RE.pattern} (an empty "
            "mode = the write run's name, forbidden · a write run passes None)"
        )
    return p.with_name(f"{p.stem}.{mode}{p.suffix}")
