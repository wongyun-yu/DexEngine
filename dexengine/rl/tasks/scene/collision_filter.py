"""Permanent OFF rows of the split articulation (DEXENGINE_BI_SPLIT_OFF).

The rest of collision filter authoring lives in topic modules, which callers import directly: body and prim lookups
(collision_bodies), robot/desk/disc/arm-object pairs (collision_pairs), the cross-pair curriculum and its runtime
switches (collision_curriculum) and the bimanual self-collision scene (collision_selfcoll).
"""

from __future__ import annotations

import time

from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.scene.collision_bodies import (
    _split_robot_bodies,
    _split_robot_left_bodies,
    classify_body,
)
from dexengine.rl.tasks.scene.collision_curriculum import (
    _collcurr_edit,
)

# ---------------------------------------------------------------------------
# Split articulation: author the rows **between** the two articulations permanently OFF.
# Why: ABL (the physics really trained = one articulation, no filters) and CANON (split articulation) differ in
# three things only: R2 (right <-> left hand·arm), R3 left (left arm·hand <-> torso) and the articulation structure.
# Telling which one decides a performance gap needs CANON runs with one row off at a time. The ramp turns (a)(b)(c)
# off as one bundle and back on over time, which does not separate rows; this does only **permanent OFF, per row**.
# Authoring: env_0 pairs -> replicated to every env through the `_collcurr_edit` substitution (one ChangeBlock).
# A count mismatch fails. Cost: every target costs host memory and VRAM; r2 is ~700 targets per env, so the
# launcher picks NE.
# ---------------------------------------------------------------------------
_SPLIT_OFF_ENV = "DEXENGINE_BI_SPLIT_OFF"
_SPLIT_OFF_ROWS = ("r2", "r3l")


def split_off_rows() -> set[str]:
    """``DEXENGINE_BI_SPLIT_OFF`` = a comma list ⊆ {r2, r3l}. Unset = the empty set (= CANON as is)."""
    raw = env_flags.get(_SPLIT_OFF_ENV).strip().lower()
    if not raw:
        return set()
    rows = {r.strip() for r in raw.split(",") if r.strip()}
    bad = rows - set(_SPLIT_OFF_ROWS)
    if bad:
        raise ValueError(
            f"{_SPLIT_OFF_ENV}={raw!r}: allowed values are comma lists of {_SPLIT_OFF_ROWS} "
            f"(unknown values {sorted(bad)})"
        )
    return rows


def split_author_off_rows(scene, rows: set[str]) -> dict[str, list[str]]:
    """r2 = right arm·hand links (RA/RH of /Robot) <-> every /RobotL link · r3l = torso·head·base (OTHER of /Robot) <->
    every /RobotL link."""
    roots = list(getattr(scene, "env_prim_paths", []) or [])
    if not roots:
        raise RuntimeError("[SPLIT_OFF] scene.env_prim_paths is empty.")
    stage = scene.stage
    right = _split_robot_bodies(stage, roots[0])
    left = _split_robot_left_bodies(stage, roots[0])

    def _cls(p):
        return classify_body(p.rsplit("/", 1)[-1])

    src: list[str] = []
    if "r2" in rows:
        src += [p for p in right if _cls(p) in ("RA", "RH")]
    if "r3l" in rows:
        src += [p for p in right if _cls(p) not in ("RA", "RH", "LA", "LH")]
    if not src or not left:
        raise RuntimeError(f"[SPLIT_OFF] no links to pair: rows={sorted(rows)} src={len(src)} left={len(left)}")
    pairs = {p: list(left) for p in src}
    t0 = time.perf_counter()
    n_rel, n_env = _collcurr_edit(scene, pairs, range(len(roots)), add=True)
    want = len(src) * len(left) * len(roots)
    if n_env != len(roots) or n_rel != want:
        raise RuntimeError(
            f"[SPLIT_OFF] authoring incomplete: env {n_env}/{len(roots)}, targets {n_rel}/{want} — "
            f"env_k prim paths differ from env_0's substitution. Training like this means different physics per env."
        )
    scene._split_off_pairs = pairs
    print(
        f"[SPLIT_OFF] {sorted(rows)}: per env {len(src)}×{len(left)}={len(src) * len(left)} pairs × {len(roots)} envs "
        f"= {n_rel} targets permanently OFF ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    return pairs
