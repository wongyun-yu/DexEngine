"""Train-set scene keys: the L0/L1/L2 list check of the E5 evaluation.

A training set's key set is read from its LeRobot `meta/episodes.jsonl`: one key `(e5_source.clip, e5_source.look)` per
episode. A scene name is `<clip>.<look>` (a job path resolves to its folder name).

Rules (a violation is an error, never a silent filter):
- L0 scenes are a subset of the keys (seen in training).
- L1 scenes do not intersect the keys (held-out placements of seen clips).
- L2 scenes share no recording with the keys: the recording of `<clip>[_pv<k>]_pl_rt[.<look>]` is `<clip>`.

Lines starting with `#` in a list file are header text, not scenes.
CLI: `python -m experiments.sim_eval.train_keys check <task|set> <L0|L1|L2> <list> --root <lerobot_train root>`
Last line: `[TRAIN_KEYS] OK …` or `[TRAIN_KEYS] FAIL <why>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

MARK = "[TRAIN_KEYS]"
#: The two training sets of each task (ours and kinematic replay) share one scene set; a task name checks both.
TASK_SETS = {
    "lift": ("dense_pnp_ours_v1", "dense_pnp_kin_v1"),
    "pour": ("dense_pour_ours_v1", "dense_pour_kin_v1"),
    "sweep": ("dense_sweep_ours_v1", "dense_sweep_kin_v1"),
}
LEVELS = ("L0", "L1", "L2")


def _read(ds: str, root: str | Path) -> bytes:
    f = Path(root) / ds / "meta" / "episodes.jsonl"
    if not f.is_file():
        raise SystemExit(f"{MARK} FAIL {ds}: {f} not found")
    return f.read_bytes()


def set_keys(ds: str, root: str | Path) -> tuple[set, str, int]:
    """Keys of one training set → (keys, sha256 of episodes.jsonl, episode count)."""
    b = _read(ds, root)
    ks, n = set(), 0
    for ln in b.decode().splitlines():
        if not ln.strip():
            continue
        s = json.loads(ln).get("e5_source") or {}
        if not s.get("clip") or not s.get("look"):
            raise SystemExit(f"{MARK} FAIL {ds}: episode without e5_source.clip/look {ln[:120]}")
        ks.add((s["clip"], s["look"]))
        n += 1
    if len(ks) != n:
        raise SystemExit(f"{MARK} FAIL {ds}: duplicate key ({n} episodes, {len(ks)} keys)")
    return ks, hashlib.sha256(b).hexdigest(), n


def keys(name: str, root: str | Path) -> tuple[set, str]:
    """Keys of a task (both training sets must agree) or of one set → (keys, source description)."""
    sets = TASK_SETS.get(name, (name,))
    got = [(ds, *set_keys(ds, root)) for ds in sets]
    if any(g[1] != got[0][1] for g in got):
        raise SystemExit(
            f"{MARK} FAIL {name}: the training sets disagree " + " · ".join(f"{g[0]} {len(g[1])}" for g in got)
        )
    return got[0][1], " · ".join(f"{g[0]} {g[3]} sha {g[2][:12]}" for g in got)


def scene_name(item: str) -> str:
    """A list entry (job path or scene name) → scene name."""
    return os.path.basename(os.path.dirname(item)) if "/" in item else item


def key_of(item: str) -> tuple[str, str]:
    """Scene `<clip>.<look>` → (clip, look)."""
    clip, dot, look = scene_name(item).rpartition(".")
    if not dot or not clip or not look:
        raise SystemExit(f"{MARK} FAIL scene name '{item}' is not <clip>.<look>")
    return clip, look


def rec_of(item: str) -> str:
    """Scene (or job path) → recording (source clip) of `<clip>[_pv<k>]_pl_rt[.<look>]`."""
    m = re.fullmatch(r"(.+?)(?:_pv\d+)?_pl_rt(?:\.[A-Za-z0-9]+)?", scene_name(item))
    if not m:
        raise SystemExit(f"{MARK} FAIL scene name '{item}' is not <clip>[_pv<k>]_pl_rt[.<look>]")
    return m.group(1)


def violations(items: list, level: str, ks: set) -> list:
    """List entries that break the rule of `level` against key set `ks`."""
    if level == "L2":
        recs = {rec_of(k[0]) for k in ks}
        return [x for x in items if rec_of(x) in recs]
    if level == "L0":
        return [x for x in items if key_of(x) not in ks]
    if level == "L1":
        return [x for x in items if key_of(x) in ks]
    raise SystemExit(f"{MARK} FAIL level '{level}' — L0|L1|L2")


def read_list(path: str | Path) -> list[str]:
    """Scene list entries (blank and `#` header lines skipped)."""
    return [x.strip() for x in open(path) if x.strip() and not x.lstrip().startswith("#")]


def check(name: str, level: str, list_path: str | Path, root: str | Path) -> str:
    """Full check of one list → OK line; raises SystemExit with the FAIL line."""
    items = read_list(list_path)
    ks, src = keys(name, root)
    bad = violations(items, level, ks)
    if bad:
        what = {"L0": "outside the training set", "L2": "shares a recording with the training set"}.get(
            level, "overlaps the training set"
        )
        ex = ", ".join(scene_name(b) for b in bad[:3])
        raise SystemExit(f"{MARK} FAIL {name} {level} {what} {len(bad)}/{len(items)} · e.g. {ex} · {src}")
    return f"{MARK} OK {name} {level} {len(items)} scenes · keys {len(ks)} · {src}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("name")
    c.add_argument("level", choices=LEVELS)
    c.add_argument("list")
    c.add_argument("--root", required=True, help="folder holding the LeRobot training sets")
    a = ap.parse_args(argv)
    print(check(a.name, a.level, a.list, a.root))


if __name__ == "__main__":
    main(sys.argv[1:])
