"""Location of one contact sidecar file: the shared layer, or the per-hand layer for clips a hand registers."""

from __future__ import annotations

import os

from dexengine.core.layout.layers import render, variant_source

# ── Location of one contact sidecar file: the shared layer or the per-hand layer ─────────────────────
# Contact sidecars (contact_base · contact_all) are a hand-agnostic shared layer — one file per (dataset, clip, side).
# Some oakink2 takes must be built by a new hand from **different input** than the first-generation hands used, so
# that hand cannot write them to the shared place (it would overwrite the first-generation input). Such clips are
# listed in the hand's registry (contact_base_own/{hand}/CLIPS.txt — shared by both layers); the registry is the
# explicit choice. Resolution reads **only the registry**: judge pods, folding and bundling stages run without the
# campaign env (a pod's INNER_BODY passes a fixed env), so an env switch would kill each of them. The registry ships
# with the judge inputs (coords.setup_inputs), so every machine resolves the same way. env DEXENGINE_SIDECAR_OWN=1 is
# stage 1 declaring that it uses this path, plus an end check on the consumption marker. Rules:
#   clip in that hand's registry   → per-hand place (marker [SCOPE_ADD] sidecar_own … once per layer·dataset·hand)
#   hand unknown (empty) + clip in any hand's registry → dies (unknown identity — never read the shared file silently)
#   anything else (clip outside the registry · hand without a registry) → the shared place, byte-identical paths
# Sidecar file paths are built only by this function — readers never build `layer_path(layer) / f"{clip}_{side}.npz"`.
SIDECAR_OWN_LAYER = {"contact_base": "contact_base_own", "contact_all": "contact_all_own"}
SIDECAR_OWN_REGISTRY = "CLIPS.txt"
_OWN_REG_CACHE: dict = {}
_OWN_MARKED: set = set()


class SidecarOwnError(RuntimeError):
    """Per-hand sidecar rule violation — never falls back to the shared place silently."""


def sidecar_own_registry(ds: str, hand: str) -> str:
    """The per-hand sidecar registry of hand `hand` (spec directory), repo-relative — one clip (tree) name per
    line · '#' header lines."""
    return f"{render('contact_base_own', ds, hand=hand)}/{SIDECAR_OWN_REGISTRY}"


def sidecar_own_clips(root, ds: str, hand: str) -> frozenset:
    p = os.path.join(str(root), sidecar_own_registry(ds, hand))
    try:
        st = os.stat(p)
    except FileNotFoundError:
        return frozenset()
    k = (p, st.st_mtime_ns, st.st_size)
    if k not in _OWN_REG_CACHE:
        with open(p) as f:
            _OWN_REG_CACHE[k] = frozenset(x.strip() for x in f if x.strip() and not x.startswith("#"))
    return _OWN_REG_CACHE[k]


def _own_hands_of(root, ds: str, clip: str) -> list:
    """The hands that register this clip (for a placement variant, its source clip — variant_source)."""
    d = os.path.join(str(root), os.path.dirname(render("contact_base_own", ds, hand="X")))
    try:
        hands = sorted(os.listdir(d))
    except FileNotFoundError:
        return []
    return [h for h in hands if variant_source(clip) in sidecar_own_clips(root, ds, h)]


def sidecar_shared_rel(layer: str, ds: str, clip: str, side: str) -> str:
    """The sidecar file in the shared place (repo-relative) — the place of every clip that has no per-hand place."""
    return f"{render(layer, ds)}/{clip}_{side}.npz"


def sidecar_rel(root, layer: str, ds: str, clip: str, side: str, hand: str) -> str:
    """The repo-relative path of one sidecar file (layer contact_base|contact_all · clip = tree name · side) — the
    single owner.

    root = data root (where the registry is read) · hand = spec directory of the hand reading or writing ('' when
    unknown — dies for a registered clip). Registration is judged on the clip's source (variant_source — a placement
    variant `<registered clip>_pv<NN>` also goes to the per-hand place, under its own variant file name)."""
    if layer not in SIDECAR_OWN_LAYER:
        raise KeyError(f"sidecar layer {layer!r} — must be one of {sorted(SIDECAR_OWN_LAYER)}")
    if side not in ("right", "left"):
        raise ValueError(f"side {side!r}")
    shared = sidecar_shared_rel(layer, ds, clip, side)
    if not hand:
        who = _own_hands_of(root, ds, clip)
        if who:
            raise SidecarOwnError(
                f"{ds}/{clip}: registered per-hand sidecar clip (hands {who}) but the reading hand "
                f"is unknown — not reading the shared place"
            )
        return shared
    if variant_source(clip) not in sidecar_own_clips(root, ds, hand):
        return shared
    own = render(SIDECAR_OWN_LAYER[layer], ds, hand=hand)
    if (layer, ds, hand) not in _OWN_MARKED:
        _OWN_MARKED.add((layer, ds, hand))
        import sys

        print(
            f"[SCOPE_ADD] sidecar_own layer={layer} ds={ds} hand={hand} "
            f"registered_clips={len(sidecar_own_clips(root, ds, hand))} place={own}",
            file=sys.stderr,
            flush=True,
        )
    return f"{own}/{clip}_{side}.npz"
