"""The RL launch recipe (`configs/rl/recipe.yaml`): one value per knob, resolved for one launch.

The file has two tables with a `common` column and one column per path (`uni` = one hand, `bi` = bimanual): the
canonical recipe table (top level, also read by the shell pipeline through `scripts/_rl_recipe.sh`) and the launcher's
own defaults (`launcher`). A value the caller gives (a `--set KEY=VALUE` override or the same name in a recorded launch
environment) wins when it is non-empty. Three values are derived rather than stated: the simulation USD variant of the
(arm, hand) pair, whether the arm has joint PD gains, and `DEXENGINE_MOUNT_TOL`. Pure Python (no Isaac Lab, no torch).

    python3 -m dexengine.rl.launch.recipe --shell uni|bi   # `export K=V` lines of the table (scripts/_rl_recipe.sh)
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import yaml

from dexengine.core import arms
from dexengine.core.repo import CONFIGS_DIR

RECIPE_FILE = CONFIGS_DIR / "rl" / "recipe.yaml"
PATHS = ("uni", "bi")
#: Trainer-environment prefix: table keys with it are exported to the Isaac Lab process.
ENV_PREFIX = "DEXENGINE_"


class RecipeError(ValueError):
    """A launch input the recipe cannot accept."""


@functools.lru_cache(maxsize=1)
def _tables() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """(canonical table, launcher defaults), each {column: {key: value}} with every value a string."""
    raw = yaml.safe_load(RECIPE_FILE.read_text())
    out = []
    for where, src in (("table", raw), ("launcher", raw.get("launcher") or {})):
        cols = {}
        for col in ("common", *PATHS):
            vals = src.get(col) or {}
            bad = [k for k, v in vals.items() if not isinstance(v, str)]
            if bad:
                raise RecipeError(f"{RECIPE_FILE}: {where} column {col!r} has non-string values for {bad} (quote them)")
            cols[col] = dict(vals)
        out.append(cols)
    table, launcher = out
    for col in table:
        if both := set(table[col]) & set(launcher[col]):
            raise RecipeError(f"{RECIPE_FILE}: {sorted(both)} are in both the table and launcher column {col!r}")
    return table, launcher


def table_columns() -> dict[str, dict[str, str]]:
    """The canonical recipe table as {column: {key: value}} (what the shell pipeline reads)."""
    return _tables()[0]


def columns() -> dict[str, dict[str, str]]:
    """Every recipe knob as {column: {key: value}}: the canonical table plus the launcher defaults."""
    table, launcher = _tables()
    return {col: {**table[col], **launcher[col]} for col in table}


def path_of(side: str) -> str:
    """Recipe path of a side: right|left -> uni, both -> bi."""
    if side in ("right", "left"):
        return "uni"
    if side == "both":
        return "bi"
    raise RecipeError(f"side must be right|left|both (got {side!r})")


def defaults(path: str) -> dict[str, str]:
    """Table values of one path (common column, then the path column on top)."""
    if path not in PATHS:
        raise RecipeError(f"recipe path must be one of {PATHS} (got {path!r})")
    cols = columns()
    return {**cols["common"], **cols[path]}


def plate_variant(arm: str, hand: str) -> str:
    """Robot plate (USD variant) every process of (arm, hand) spawns — owner dexengine.core.arms.plate_variant.

    An arm with a generation axis always has a non-empty plate (the render material tail), so an empty one is a broken
    table and is refused."""
    if arm not in arms.ARM_REGISTRY:
        raise RecipeError(f"unregistered arm {arm!r} (dexengine.core.arms.ARM_REGISTRY)")
    v = arms.plate_variant(arm, hand)
    if not v and arm in arms.ARM_GENERATIONS:
        raise RecipeError(
            f"arm {arm!r} hand {hand!r}: the plate is empty (a plate carries the render material tail — broken table)"
        )
    return v


def arm_joint_pd(arm: str) -> bool:
    """True when the arm has joint PD gains (--arm_kp/--arm_kd); the floating hand has none."""
    if arm not in arms.ARM_REGISTRY:
        raise RecipeError(f"unregistered arm {arm!r} (dexengine.core.arms.ARM_REGISTRY)")
    spec = arms.ARM_REGISTRY[arm]
    return not (spec is not None and spec.is_floating)


def resolve(path: str, arm: str, hand: str, given: Mapping[str, str]) -> dict[str, str]:
    """All recipe values of one launch: table defaults with non-empty `given` values winning, plus the derived
    DEXENGINE_COMBO_USD_VARIANT (given, else USD_VARIANT, else the (arm, hand) lookup), ARM_JOINT_PD ("1"/"0") and
    DEXENGINE_MOUNT_TOL."""
    out = {k: (given.get(k) or v) for k, v in defaults(path).items()}
    # A knob of the other path's column given explicitly (e.g. MB or POLICY_SIZE on a one-hand run) applies too.
    out.update({k: given[k] for k in set().union(*columns().values()) - set(out) if given.get(k)})
    out["DEXENGINE_COMBO_USD_VARIANT"] = (
        given.get("DEXENGINE_COMBO_USD_VARIANT") or given.get("USD_VARIANT") or plate_variant(arm, hand)
    )
    if why := arms.retired_plate_why(arm, hand, out["DEXENGINE_COMBO_USD_VARIANT"]):
        raise RecipeError(why)
    out["ARM_JOINT_PD"] = "1" if arm_joint_pd(arm) else "0"
    out["DEXENGINE_MOUNT_TOL"] = given.get("DEXENGINE_MOUNT_TOL") or out["MOUNT_TOL"]
    return out


def trainer_env(values: Mapping[str, str]) -> dict[str, str]:
    """The recipe values that go to the trainer process environment (DEXENGINE_* keys)."""
    return {k: v for k, v in values.items() if k.startswith(ENV_PREFIX)}


def table_values(names, env: Mapping[str, str]) -> dict[str, str]:
    """Recipe values of `names` under a launch environment `env` (the reader run_inputs and the campaign use).

    `env["CTE_PATH"]` picks the column (uni|bi, default uni); arm and hand come from DEXENGINE_ARM/DEXENGINE_HAND
    (or ARM/HAND); DEXENGINE_DATASET and DEXENGINE_EMBODIMENT must be set. A name the table does not hold is read
    from `env` ("" when unset)."""
    path = env.get("CTE_PATH") or "uni"
    for k in ("DEXENGINE_DATASET", "DEXENGINE_EMBODIMENT"):
        if not env.get(k):
            raise RecipeError(f"{k} is required (no default)")
    arm = env.get("ARM") or env.get("DEXENGINE_ARM") or ""
    hand = env.get("HAND") or env.get("DEXENGINE_HAND") or ""
    vals = resolve(path, arm, hand, env)
    return {n: vals[n] if n in vals else env.get(n, "") for n in names}


def shell_view(path: str, env: Mapping[str, str]) -> dict[str, str]:
    """What `scripts/_rl_recipe.sh` exports under `env`: the canonical table of `path` (non-empty values in `env`
    win), the three derived values, and DEXENGINE_DATASET / DEXENGINE_EMBODIMENT (required)."""
    for k in ("DEXENGINE_DATASET", "DEXENGINE_EMBODIMENT"):
        if not env.get(k):
            raise RecipeError(f"{k} is required (no default)")
    arm = env.get("ARM") or env.get("DEXENGINE_ARM") or ""
    hand = env.get("HAND") or env.get("DEXENGINE_HAND") or ""
    vals = resolve(path, arm, hand, env)
    table = table_columns()
    keys = [*table["common"], *table[path], "DEXENGINE_COMBO_USD_VARIANT", "ARM_JOINT_PD", "DEXENGINE_MOUNT_TOL"]
    return {
        **{k: vals[k] for k in keys},
        "DEXENGINE_DATASET": env["DEXENGINE_DATASET"],
        "DEXENGINE_EMBODIMENT": env["DEXENGINE_EMBODIMENT"],
    }


def main(argv=None) -> int:
    import argparse
    import os
    import shlex
    import sys

    ap = argparse.ArgumentParser(description="Shell view of the canonical RL recipe table.")
    ap.add_argument("--shell", required=True, metavar="uni|bi", help="print `export K=V` lines for this path")
    a = ap.parse_args(argv)
    if a.shell not in PATHS:
        print(f"[recipe] --shell must be one of {PATHS} (got {a.shell!r})", file=sys.stderr)
        return 4
    try:
        vals = shell_view(a.shell, os.environ)
    except RecipeError as e:
        print(f"[recipe] {e}", file=sys.stderr)
        return 4
    print("".join(f"export {k}={shlex.quote(v)}\n" for k, v in vals.items()), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
