"""Shared names of the long-clip split (rule D36) and the aerial tail cut (D41): rule versions, the active hand, the
base-layer and registry locations."""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.core import (
    paths as _P,  # noqa: E402 — owner of layer paths (layer_path); data layer paths are not joined by hand here
)
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)
from dexengine.data import rules as RR  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
#: Version name of the registry rule, written to the registry file's "rule" (stored text). v6: switches to a non-main
#: object only when that object moved (cut_rule, case.move_runs).
RULE_NAME = (
    "D36 v6 — MANO timeline + sustained grasp-switch cuts (no nested assist, no grasp under MIN_DURATION_S), "
    "switch to a non-main object only if it moved (case.move_runs), "
    "all parents, contained units removed, hand-independent registry (segment_split.py)"
)
#: Version name of the D41 overlay, written to the registry's "tail_rule" (separate from the D36 version; the overlay
#: never changes the D36 statement).
TAIL_RULE_NAME = "D41 v2 — aerial scene parents (scene_of): last window ends where a main object stays unsupported"
#: Rule table name of D41; the scope (datasets) belongs to data.rules.rule_datasets (d41_in_scope).
D41_RULE = "D41"
HAND = RR.hand_dir()
COMBO = RR.combo()


def _tc():
    """First-round candidate module (owner of the aerial-path decision), imported late (other paths in this file do not
    need it)."""
    from dexengine.data import candidate_rules as train_candidates  # noqa: PLC0415

    return train_candidates


def _base_emb(ds: str, emb: str, hand_dir: str = "") -> Path:
    """emb folder of the hand's (default: this hand) base layer; the path comes from the owner paths.layer_path."""
    return Path(_P.layer_path("base", ds, hand=hand_dir or HAND, emb=emb))


def _base_dir(ds: str, emb: str, clip: str) -> Path:
    return _base_emb(ds, emb) / clip


def _stale_dir(ds: str, emb: str) -> Path:
    """Quarantine for re-sliced and unregistered pieces data/<ds>/base_stale/<hand>/<emb>; not a layer (no layer glob
    matches it).
    Derived from the base layer location (base → base_stale in `…/base/<hand>/<emb>`); no other root is joined."""
    b = _base_emb(ds, emb)
    return b.parent.parent.with_name("base_stale") / HAND / emb


def split_root(ds: str, emb: str) -> Path:
    """A run touches one data root: every layer pieces read and write (base · registry · hand-independent sidecars ·
    wrists · quarantine) must lie under the data root (paths.main_root, env DEXENGINE_MAIN_ROOT). If any layer sits
    under
    another root (quarantine root, mixed generation roots), pieces of one generation would be built from another
    generation's parents and sidecars, so it fails. Returns that root (marker [SPLIT_ROOT])."""
    root = _data_root()
    where = {
        "base": _base_emb(ds, emb),
        "base_stale": _stale_dir(ds, emb),
        "registry": _registry_path(ds, emb),
        "contact_all": Path(_P.layer_path("contact_all", ds)),
        "contact_base": Path(_P.layer_path("contact_base", ds)),
        "wrist_base": Path(_P.layer_path("wrist_base", ds, hand=HAND, emb=emb)),
    }
    off = {k: str(v) for k, v in where.items() if not Path(os.path.abspath(v)).is_relative_to(root)}
    if off:
        raise SystemExit(
            f"[SPLIT_ROOT] ✗ {ds}/{emb}: layers outside the data root {root}: {off} — mixed roots (one run = one root)"
        )
    return root


def _registry_path(ds: str, emb: str) -> Path:
    return _P.units_registry(ds, emb, _data_root())


def _own_registry_path(ds: str, emb: str) -> Path:
    """This hand's per-hand window registry <name>.own_<hand>.json next to the shared registry (derived from its
    location; no separate root). Owns the windows of parents whose static-articulation rule (STATIC_ARTIC) applied only
    in this hand's trees, so their object shapes differ from the reference hand's (static_artic_own). Written only by
    define_step."""
    reg = _registry_path(ds, emb)
    return reg.with_name(f"{reg.stem}.own_{HAND}{reg.suffix}")
